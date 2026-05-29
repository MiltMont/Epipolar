"""ICP pipeline: pairwise depth registration → global trajectory.

Given a TUM Sequence and an ICPConfig the pipeline:
  1. Loads consecutive depth frames and back-projects to point clouds.
  2. Optionally subsamples each cloud (voxel_downsample > 1).
  3. Runs ICP between each consecutive pair, optionally seeded with an
     epipolar-derived initial transform.
  4. Chains relative SE(3) transforms into a global trajectory.

trajectory[i] is T_world_cam_i (4×4 SE(3)), trajectory[0] = identity.
ICP uses metric depth, so Umeyama alignment for ATE must use scale=False.

ICP convention per pair:
  source = cloud_i  (current frame)
  target = cloud_(i+1)  (next frame)
  result.R, result.t maps source → target, i.e. T_{cam_(i+1)←cam_i}.
  trajectory[i+1] = trajectory[i] @ inv(T_rel).
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from tray.config import ICPConfig
from tray.data.tum import Sequence
from tray.icp.iterate import icp
from tray.icp.pointcloud import depth_to_pointcloud, downsample

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SE(3) helpers (local copies; keep icp/ self-contained)
# ─────────────────────────────────────────────────────────────────────────────

def se3_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 4×4 SE(3) matrix from (3,3) R and (3,) t."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def se3_inv(T: np.ndarray) -> np.ndarray:
    """Invert a 4×4 SE(3) matrix without np.linalg.inv."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


# ─────────────────────────────────────────────────────────────────────────────
# Depth loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_depth(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Cannot read depth image: {path}")
    return depth


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_icp_pipeline(
    seq: Sequence,
    cfg: ICPConfig,
    *,
    epipolar_trajectory: np.ndarray | None = None,
    max_frames: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Run the full ICP pipeline and return a global trajectory.

    Args:
        seq:                   Loaded TUM Sequence (from tray.data.tum.load_sequence).
        cfg:                   ICPConfig controlling ICP hyperparameters.
        epipolar_trajectory:   Optional (N, 4, 4) trajectory from the epipolar
                               pipeline.  Required when cfg.init_from_epipolar
                               is True; ignored otherwise.
        max_frames:            If set, truncate the frame list to this length.
        rng:                   RNG for reproducible voxel downsampling.

    Returns:
        trajectory: (N, 4, 4) float64 array.  trajectory[i] = T_world_cam_i,
                    a 4×4 SE(3) rigid transform.  trajectory[0] = identity.

    Notes:
        * Depth images are metric (TUM scale = 5000); ATE evaluation must use
          Umeyama with scale=False for this pipeline.
        * If a frame pair fails (empty cloud, ICP exception), that step keeps
          the previous pose so the trajectory stays contiguous.
        * Epipolar init provides rotation direction only; the translation scale
          from a monocular epipolar pipeline is not metric, but the direction
          helps ICP converge faster when motion between frames is large.
    """
    frames = seq.frames[:max_frames] if max_frames is not None else seq.frames
    N = len(frames)
    if N == 0:
        return np.zeros((0, 4, 4), dtype=np.float64)

    if cfg.init_from_epipolar and epipolar_trajectory is None:
        raise ValueError(
            "cfg.init_from_epipolar=True requires a non-None epipolar_trajectory"
        )

    K = seq.intrinsics.K
    depth_scale = seq.intrinsics.depth_scale

    trajectory = np.zeros((N, 4, 4), dtype=np.float64)
    trajectory[0] = np.eye(4)

    cloud_cache: dict[int, np.ndarray] = {}

    def _get_cloud(idx: int) -> np.ndarray:
        if idx not in cloud_cache:
            depth = _load_depth(frames[idx].depth_path)
            cloud = depth_to_pointcloud(depth, K, depth_scale=depth_scale)
            cloud = downsample(cloud, cfg.voxel_downsample, rng=rng)
            cloud_cache[idx] = cloud
            # Keep at most the two most-recently accessed frames in memory.
            if len(cloud_cache) > 2:
                oldest = min(k for k in cloud_cache if k != idx)
                del cloud_cache[oldest]
        return cloud_cache[idx]

    for i in tqdm(range(N - 1), desc=f"icp/{cfg.name}", unit="frame", leave=False):
        cloud_i = _get_cloud(i)
        cloud_next = _get_cloud(i + 1)

        if len(cloud_i) < 10 or len(cloud_next) < 10:
            logger.warning(
                "pair %d→%d: too few points (%d, %d); keeping previous pose",
                i, i + 1, len(cloud_i), len(cloud_next),
            )
            trajectory[i + 1] = trajectory[i]
            continue

        # Optional epipolar seed: extract the relative transform from the
        # epipolar trajectory that maps cloud_i → cloud_(i+1).
        # T_{cam_(i+1)←cam_i} = inv(T_world_cam_(i+1)) @ T_world_cam_i
        init_R: np.ndarray | None = None
        init_t: np.ndarray | None = None
        if cfg.init_from_epipolar and epipolar_trajectory is not None:
            T_rel_epi = se3_inv(epipolar_trajectory[i + 1]) @ epipolar_trajectory[i]
            init_R = T_rel_epi[:3, :3]
            init_t = T_rel_epi[:3, 3]

        try:
            result = icp(
                cloud_i,
                cloud_next,
                max_iter=cfg.max_iter,
                tolerance=cfg.tolerance,
                reject_dist=cfg.reject_dist_m,
                init_R=init_R,
                init_t=init_t,
            )
        except Exception as exc:
            logger.warning(
                "pair %d→%d: ICP failed (%s); keeping previous pose",
                i, i + 1, exc,
            )
            trajectory[i + 1] = trajectory[i]
            continue

        # result.R, result.t: maps cloud_i → cloud_(i+1), i.e. T_{cam_(i+1)←cam_i}.
        T_rel = se3_from_Rt(result.R, result.t)
        trajectory[i + 1] = trajectory[i] @ se3_inv(T_rel)

    return trajectory
