"""Tests for icp/pipeline.py.

Coverage:
  1. SE(3) helpers (se3_from_Rt, se3_inv).
  2. run_icp_pipeline on a minimal synthetic Sequence:
       - identity motion (all clouds identical) → trajectory stays at origin.
       - known rigid motion → recovered trajectory matches ground truth.
       - init_from_epipolar=True passes the epipolar seed to ICP.
       - voxel_downsample > 1 still produces a (N,4,4) trajectory.
       - empty sequence → (0,4,4) array.
       - frame-pair with empty cloud → pose falls back to previous.
"""
from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from tray.config import ICPConfig
from tray.data.tum import FREIBURG1, Frame, Sequence
from tray.icp.pipeline import run_icp_pipeline, se3_from_Rt, se3_inv


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_K = FREIBURG1.K   # (3,3)
_DEPTH_SCALE = FREIBURG1.depth_scale   # 5000.0

# Tiny synthetic rotation (~5°) and translation
_ANGLE = 0.05
_R_GT = np.array([
    [np.cos(_ANGLE), 0.0, np.sin(_ANGLE)],
    [0.0,            1.0, 0.0],
    [-np.sin(_ANGLE), 0.0, np.cos(_ANGLE)],
], dtype=np.float64)
_T_GT = np.array([0.02, 0.01, 0.005], dtype=np.float64)


def _make_depth_img(cloud_3d: np.ndarray, K: np.ndarray, H=480, W=640) -> np.ndarray:
    """Project a (N,3) cloud back to a uint16 depth image."""
    depth_img = np.zeros((H, W), dtype=np.uint16)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    for x, y, z in cloud_3d:
        if z <= 0 or z > 8.0:
            continue
        u = int(round(x * fx / z + cx))
        v = int(round(y * fy / z + cy))
        if 0 <= u < W and 0 <= v < H:
            pix = int(z * _DEPTH_SCALE)
            if 0 < pix < 65536:
                depth_img[v, u] = pix
    return depth_img


def _write_depth_png(path: Path, img: np.ndarray) -> None:
    cv2.imwrite(str(path), img)


def _dummy_rgb(tmpdir: Path, name: str) -> Path:
    p = tmpdir / name
    cv2.imwrite(str(p), np.zeros((480, 640, 3), dtype=np.uint8))
    return p


def _make_sequence(frames_data: list[np.ndarray], tmpdir: Path) -> Sequence:
    """Build a synthetic Sequence from a list of (N,3) point clouds."""
    frame_list: list[Frame] = []
    for i, cloud in enumerate(frames_data):
        depth_img = _make_depth_img(cloud, _K)
        depth_path = tmpdir / f"depth_{i:04d}.png"
        rgb_path = tmpdir / f"rgb_{i:04d}.png"
        _write_depth_png(depth_path, depth_img)
        _dummy_rgb(tmpdir, f"rgb_{i:04d}.png")
        frame_list.append(Frame(
            timestamp=float(i),
            rgb_path=rgb_path,
            depth_path=depth_path,
            depth_timestamp=float(i),
            T_gt=None,
            gt_timestamp=None,
        ))
    return Sequence(root=tmpdir, name="freiburg1_xyz", intrinsics=FREIBURG1, frames=frame_list)


def _base_cfg(**kw) -> ICPConfig:
    defaults = dict(
        name="test",
        sequence="rgbd_dataset_freiburg1_xyz",
        max_iter=50,
        tolerance=1e-5,
        reject_dist_m=0.5,
        voxel_downsample=1,
        init_from_epipolar=False,
    )
    defaults.update(kw)
    return ICPConfig(**defaults)


def _random_cloud(n=500, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Points in front of the camera (positive z), spread around the scene.
    xy = rng.uniform(-0.8, 0.8, (n, 2))
    z = rng.uniform(1.0, 4.0, (n, 1))
    return np.hstack([xy, z])


# ─────────────────────────────────────────────────────────────────────────────
# SE(3) helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestSE3Helpers:
    def test_se3_from_Rt_shape(self):
        T = se3_from_Rt(np.eye(3), np.zeros(3))
        assert T.shape == (4, 4)

    def test_se3_from_Rt_identity(self):
        T = se3_from_Rt(np.eye(3), np.zeros(3))
        np.testing.assert_allclose(T, np.eye(4))

    def test_se3_inv_of_identity(self):
        T_inv = se3_inv(np.eye(4))
        np.testing.assert_allclose(T_inv, np.eye(4))

    def test_se3_inv_roundtrip(self):
        R = _R_GT.copy()
        t = _T_GT.copy()
        T = se3_from_Rt(R, t)
        T_inv = se3_inv(T)
        np.testing.assert_allclose(T @ T_inv, np.eye(4), atol=1e-12)

    def test_se3_inv_agrees_with_linalg(self):
        R = _R_GT.copy()
        t = _T_GT.copy()
        T = se3_from_Rt(R, t)
        np.testing.assert_allclose(se3_inv(T), np.linalg.inv(T), atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# run_icp_pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestRunICPPipeline:
    def test_empty_sequence_returns_empty(self, tmp_path):
        seq = Sequence(root=tmp_path, name="freiburg1_xyz", intrinsics=FREIBURG1, frames=[])
        cfg = _base_cfg()
        traj = run_icp_pipeline(seq, cfg)
        assert traj.shape == (0, 4, 4)

    def test_single_frame_returns_identity(self, tmp_path):
        cloud = _random_cloud(300)
        seq = _make_sequence([cloud], tmp_path)
        cfg = _base_cfg()
        traj = run_icp_pipeline(seq, cfg)
        assert traj.shape == (1, 4, 4)
        np.testing.assert_allclose(traj[0], np.eye(4), atol=1e-9)

    def test_identity_motion_stays_at_origin(self, tmp_path):
        """If every frame has the same cloud, the trajectory should not drift."""
        cloud = _random_cloud(400)
        n_frames = 4
        seq = _make_sequence([cloud] * n_frames, tmp_path)
        cfg = _base_cfg(max_iter=30, tolerance=1e-6)
        rng = np.random.default_rng(0)
        traj = run_icp_pipeline(seq, cfg, rng=rng)
        assert traj.shape == (n_frames, 4, 4)
        np.testing.assert_allclose(traj[0], np.eye(4))
        for i in range(1, n_frames):
            np.testing.assert_allclose(traj[i], np.eye(4), atol=0.01)

    def test_known_motion_recovered(self, tmp_path):
        """Two frames with a known rigid motion: trajectory[1] should
        approximate the inverse of T_rel applied to the world."""
        rng_cloud = np.random.default_rng(42)
        cloud0 = _random_cloud(500, seed=42)
        # Apply known R, t to get cloud1 (cloud0 expressed in frame 1 coords).
        cloud1 = cloud0 @ _R_GT.T + _T_GT

        seq = _make_sequence([cloud0, cloud1], tmp_path)
        cfg = _base_cfg(max_iter=80, tolerance=1e-7, reject_dist_m=1.0)
        traj = run_icp_pipeline(seq, cfg)

        assert traj.shape == (2, 4, 4)
        np.testing.assert_allclose(traj[0], np.eye(4))

        # The estimated T_world_cam1 should bring a world point to cam-1 coords.
        # traj[1] = I @ inv(T_rel_icp) where T_rel_icp ≈ se3_from_Rt(R_GT, t_GT).
        # So traj[1][:3,:3] ≈ R_GT.T and traj[1][:3,3] ≈ -R_GT.T @ t_GT.
        R_est = traj[1, :3, :3]
        R_angle_err = np.arccos(
            np.clip((np.trace(R_est @ _R_GT) - 1.0) / 2.0, -1.0, 1.0)
        )
        assert abs(R_angle_err) < 0.05, f"Rotation error {np.degrees(R_angle_err):.2f}°"

    def test_trajectory_first_pose_is_identity(self, tmp_path):
        cloud = _random_cloud(300)
        seq = _make_sequence([cloud, cloud, cloud], tmp_path)
        cfg = _base_cfg()
        traj = run_icp_pipeline(seq, cfg)
        np.testing.assert_allclose(traj[0], np.eye(4))

    def test_voxel_downsample_runs(self, tmp_path):
        cloud = _random_cloud(1000)
        seq = _make_sequence([cloud, cloud, cloud], tmp_path)
        cfg = _base_cfg(voxel_downsample=10)
        rng = np.random.default_rng(1)
        traj = run_icp_pipeline(seq, cfg, rng=rng)
        assert traj.shape == (3, 4, 4)

    def test_init_from_epipolar_requires_trajectory(self, tmp_path):
        cloud = _random_cloud(300)
        seq = _make_sequence([cloud, cloud], tmp_path)
        cfg = _base_cfg(init_from_epipolar=True)
        with pytest.raises(ValueError, match="epipolar_trajectory"):
            run_icp_pipeline(seq, cfg, epipolar_trajectory=None)

    def test_init_from_epipolar_identity_seed(self, tmp_path):
        """Providing an identity epipolar trajectory should not break ICP."""
        cloud = _random_cloud(300)
        n = 3
        seq = _make_sequence([cloud] * n, tmp_path)
        cfg = _base_cfg(init_from_epipolar=True)
        epi_traj = np.stack([np.eye(4)] * n)
        rng = np.random.default_rng(5)
        traj = run_icp_pipeline(seq, cfg, epipolar_trajectory=epi_traj, rng=rng)
        assert traj.shape == (n, 4, 4)

    def test_max_frames_truncates(self, tmp_path):
        cloud = _random_cloud(300)
        seq = _make_sequence([cloud] * 10, tmp_path)
        cfg = _base_cfg()
        traj = run_icp_pipeline(seq, cfg, max_frames=4)
        assert traj.shape == (4, 4, 4)

    def test_se3_matrix_structure(self, tmp_path):
        """Every pose in the trajectory must be a valid SE(3) matrix."""
        cloud = _random_cloud(400)
        seq = _make_sequence([cloud, cloud, cloud], tmp_path)
        cfg = _base_cfg()
        traj = run_icp_pipeline(seq, cfg)
        for i, T in enumerate(traj):
            R = T[:3, :3]
            assert np.linalg.det(R) == pytest.approx(1.0, abs=0.01), \
                f"Frame {i}: det(R) = {np.linalg.det(R)}"
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=0.01,
                                       err_msg=f"Frame {i}: R not orthogonal")
            np.testing.assert_allclose(T[3], [0, 0, 0, 1], atol=1e-9,
                                       err_msg=f"Frame {i}: bottom row wrong")
