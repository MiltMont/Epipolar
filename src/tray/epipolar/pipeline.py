"""Epipolar geometry pipeline: pairwise F → E → (R, t) → global trajectory.

Given a TUM Sequence and an EpipolarConfig the pipeline:
  1. Loads consecutive RGB frames as grayscale.
  2. Detects + matches features (ORB/SIFT; top-N or Lowe ratio).
  3. Estimates F (8pt / 7pt / RANSAC).
  4. Computes E = K^T F K → recover_pose → (R, t).
  5. Chains relative SE(3) transforms into a global trajectory.

The returned trajectory[i] is T_world_cam_i (4x4 SE(3)), with trajectory[0]
set to the identity.  Monocular translation is up-to-scale by construction.

If a pair has too few matches or pose estimation fails, that step falls back to
the previous pose (identity relative motion) so the trajectory remains
contiguous.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from tray.config import EpipolarConfig
from tray.data.tum import Sequence
from tray.epipolar.fundamental import eight_point, ransac, sampson_distance, seven_point
from tray.epipolar.pose import essential_from_F, recover_pose
from tray.features.detectors import Features, detect_and_compute, make_detector
from tray.features.matching import match_lowe, match_top_n

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SE(3) helpers
# ─────────────────────────────────────────────────────────────────────────────

def se3_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 4x4 SE(3) matrix from a (3,3) rotation and (3,) translation."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def se3_inv(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 SE(3) matrix without calling np.linalg.inv."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


# ─────────────────────────────────────────────────────────────────────────────
# F estimation dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_F(
    cfg: EpipolarConfig,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate F according to cfg.f_method; return (F, inlier_mask).

    inlier_mask is a boolean array of length len(pts1).  The Sampson threshold
    used for inlier classification is cfg.ransac_threshold_px.

    7pt:   takes the top-7 matches (already distance-sorted by match_top_n or
           equivalent), runs seven_point, then picks the candidate with the
           most inliers across *all* matches.
    8pt:   runs eight_point on all matches; inliers are those with Sampson
           distance below the threshold.
    ransac: delegates to ransac() which returns its own mask.
    """
    thresh_sq = cfg.ransac_threshold_px ** 2

    if cfg.f_method == "8pt":
        F = eight_point(pts1, pts2)
        mask = sampson_distance(F, pts1, pts2) < thresh_sq
        return F, mask

    if cfg.f_method == "7pt":
        if len(pts1) < 7:
            raise ValueError(f"7-point needs ≥ 7 matches, got {len(pts1)}")
        candidates = seven_point(pts1[:7], pts2[:7])
        if not candidates:
            raise RuntimeError("seven_point returned no candidates")
        # Select the candidate with the most inliers over all matches.
        best_F = max(
            candidates,
            key=lambda Fc: int((sampson_distance(Fc, pts1, pts2) < thresh_sq).sum()),
        )
        mask = sampson_distance(best_F, pts1, pts2) < thresh_sq
        return best_F, mask

    if cfg.f_method == "ransac":
        F, mask = ransac(pts1, pts2, threshold=cfg.ransac_threshold_px)
        return F, mask

    raise ValueError(f"Unknown f_method {cfg.f_method!r}; expected '8pt', '7pt', or 'ransac'")


# ─────────────────────────────────────────────────────────────────────────────
# Image loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_epipolar_pipeline(
    seq: Sequence,
    cfg: EpipolarConfig,
    *,
    max_frames: int | None = None,
) -> np.ndarray:
    """Run the full epipolar pipeline and return a global trajectory.

    Args:
        seq:        Loaded TUM Sequence (from tray.data.tum.load_sequence).
        cfg:        EpipolarConfig controlling detector, matcher, and F method.
        max_frames: If set, truncate the frame list to this length (useful for
                    quick smoke-tests and partial runs during development).

    Returns:
        trajectory: (N, 4, 4) float64 array.  trajectory[i] = T_world_cam_i,
                    a 4x4 SE(3) rigid transform.  trajectory[0] = identity.

    Notes:
        * Monocular translation has no metric scale (||t|| = 1 per frame pair).
        * ATE evaluation must apply Umeyama alignment with scale=True before
          comparing against ground truth.
        * If a frame pair fails (too few matches, numerical error), that step
          uses identity relative motion so the trajectory stays contiguous.
    """
    frames = seq.frames[:max_frames] if max_frames is not None else seq.frames
    N = len(frames)
    if N == 0:
        return np.zeros((0, 4, 4), dtype=np.float64)

    K = seq.intrinsics.K
    detector = make_detector(cfg.detector, n_features=cfg.n_features)

    trajectory = np.zeros((N, 4, 4), dtype=np.float64)
    trajectory[0] = np.eye(4)

    # Rolling feature cache: keep the last two frames' features to avoid
    # re-running detection on every second frame of each pair.
    feat_cache: dict[int, Features] = {}

    def _get_features(idx: int) -> Features:
        if idx not in feat_cache:
            img = _load_gray(frames[idx].rgb_path)
            feat_cache[idx] = detect_and_compute(detector, img)
            # Evict all entries except the two most recent indices.
            if len(feat_cache) > 2:
                oldest = min(k for k in feat_cache if k != idx)
                del feat_cache[oldest]
        return feat_cache[idx]

    min_matches = 7 if cfg.f_method == "7pt" else 8

    for i in tqdm(range(N - 1), desc=f"epipolar/{cfg.name}", unit="frame", leave=False):
        feat1 = _get_features(i)
        feat2 = _get_features(i + 1)

        # --- Match ---
        if cfg.matcher == "topn":
            m = match_top_n(feat1, feat2, cfg.top_n)
        elif cfg.matcher == "lowe":
            m = match_lowe(feat1, feat2, cfg.lowe_ratio)
        else:
            raise ValueError(f"Unknown matcher {cfg.matcher!r}; expected 'topn' or 'lowe'")

        if len(m) < min_matches:
            logger.warning(
                "pair %d→%d: %d matches < %d required; keeping previous pose",
                i, i + 1, len(m), min_matches,
            )
            trajectory[i + 1] = trajectory[i]
            continue

        # --- Estimate F → E → (R, t) ---
        try:
            F, inlier_mask = _estimate_F(cfg, m.pts1, m.pts2)

            pts1_in = m.pts1[inlier_mask]
            pts2_in = m.pts2[inlier_mask]

            if len(pts1_in) < 5:
                logger.warning(
                    "pair %d→%d: only %d inliers; keeping previous pose",
                    i, i + 1, len(pts1_in),
                )
                trajectory[i + 1] = trajectory[i]
                continue

            E = essential_from_F(F, K)
            R, t, _ = recover_pose(E, K, pts1_in, pts2_in)

        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            logger.warning(
                "pair %d→%d: pose estimation failed (%s); keeping previous pose",
                i, i + 1, exc,
            )
            trajectory[i + 1] = trajectory[i]
            continue

        # --- Chain SE(3) ---
        # T_rel encodes X_{i+1} = R X_i + t  (camera-i coords → camera-(i+1) coords).
        # T_world_cam(i+1) = T_world_cami @ T_cami_cam(i+1) = T_world_cami @ inv(T_rel).
        T_rel = se3_from_Rt(R, t)
        trajectory[i + 1] = trajectory[i] @ se3_inv(T_rel)

    return trajectory
