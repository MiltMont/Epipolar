"""Tests for epipolar/pipeline.py.

Three layers of coverage:
  1. SE(3) math helpers (`se3_from_Rt`, `se3_inv`, chain correctness).
  2. `_estimate_F` dispatcher — each f_method on synthetic correspondences.
  3. `run_epipolar_pipeline` smoke test with a minimal synthetic Sequence.
"""
from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from tray.config import EpipolarConfig
from tray.data.tum import FREIBURG1, Frame, Sequence
from tray.epipolar.pipeline import (
    _estimate_F,
    run_epipolar_pipeline,
    se3_from_Rt,
    se3_inv,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared synthetic geometry
# ─────────────────────────────────────────────────────────────────────────────

_K = np.array([[800.0, 0.0, 320.0],
               [0.0,  800.0, 240.0],
               [0.0,   0.0,   1.0]], dtype=np.float64)

_angle = 0.15
_R_gt = np.array([
    [np.cos(_angle), 0.0, np.sin(_angle)],
    [0.0,            1.0, 0.0],
    [-np.sin(_angle), 0.0, np.cos(_angle)],
], dtype=np.float64)
_t_gt = np.array([0.4, 0.03, 0.08], dtype=np.float64)


def _project(R, t, X):
    Xc = (R @ X.T).T + t
    xh = (_K @ Xc.T).T
    return xh[:, :2] / xh[:, 2:3]


def _make_corr(n=80, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform([-1.5, -1.0, 3.5], [1.5, 1.0, 6.5], (n, 3))
    return _project(np.eye(3), np.zeros(3), X), _project(_R_gt, _t_gt, X)


# ─────────────────────────────────────────────────────────────────────────────
# SE(3) helpers
# ─────────────────────────────────────────────────────────────────────────────

def test_se3_from_Rt_structure() -> None:
    T = se3_from_Rt(_R_gt, _t_gt)
    assert T.shape == (4, 4)
    np.testing.assert_array_equal(T[:3, :3], _R_gt)
    np.testing.assert_array_equal(T[:3, 3], _t_gt)
    np.testing.assert_array_equal(T[3], [0, 0, 0, 1])


def test_se3_inv_roundtrip() -> None:
    T = se3_from_Rt(_R_gt, _t_gt)
    err = np.linalg.norm(T @ se3_inv(T) - np.eye(4))
    assert err < 1e-12, f"T @ inv(T) deviation: {err:.2e}"


def test_se3_inv_matches_numpy_inv() -> None:
    T = se3_from_Rt(_R_gt, _t_gt)
    np.testing.assert_allclose(se3_inv(T), np.linalg.inv(T), atol=1e-12)


def test_se3_chain_three_poses() -> None:
    """Chain three known relative motions; the composition should match the product."""
    angles = [0.1, -0.05, 0.08]
    ts = [[0.1, 0.0, 0.0], [0.0, 0.05, 0.0], [-0.02, 0.0, 0.1]]
    Ts = []
    for a, t in zip(angles, ts):
        R = np.array([[np.cos(a), 0, np.sin(a)],
                      [0, 1, 0],
                      [-np.sin(a), 0, np.cos(a)]])
        Ts.append(se3_from_Rt(R, np.array(t)))

    T_chain = np.eye(4)
    for T in Ts:
        T_chain = T_chain @ se3_inv(T)

    # Recompute via direct composition.
    T_direct = se3_inv(Ts[0]) @ se3_inv(Ts[1]) @ se3_inv(Ts[2])
    np.testing.assert_allclose(T_chain, T_direct, atol=1e-12)


def test_se3_inv_identity() -> None:
    np.testing.assert_allclose(se3_inv(np.eye(4)), np.eye(4), atol=1e-15)


# ─────────────────────────────────────────────────────────────────────────────
# _estimate_F dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def _base_cfg(**kwargs) -> EpipolarConfig:
    defaults = dict(
        name="test",
        sequence="rgbd_dataset_freiburg1_xyz",
        detector="orb",
        matcher="topn",
        f_method="8pt",
        n_features=500,
        top_n=100,
        lowe_ratio=0.75,
        ransac_threshold_px=1.0,
    )
    defaults.update(kwargs)
    return EpipolarConfig(**defaults)


def test_estimate_F_8pt_returns_rank2() -> None:
    p1, p2 = _make_corr(50, noise=0.0)
    cfg = _base_cfg(f_method="8pt")
    F, mask = _estimate_F(cfg, p1, p2)
    assert F.shape == (3, 3)
    assert np.linalg.matrix_rank(F, tol=1e-8) == 2
    assert mask.dtype == bool
    assert mask.shape == (50,)


def test_estimate_F_8pt_inliers_noise_free() -> None:
    """Noise-free data → nearly all points should be inliers."""
    p1, p2 = _make_corr(60, noise=0.0)
    cfg = _base_cfg(f_method="8pt", ransac_threshold_px=1.0)
    _, mask = _estimate_F(cfg, p1, p2)
    assert mask.sum() >= 55, f"only {mask.sum()}/60 inliers on noise-free data"


def test_estimate_F_7pt_returns_rank2() -> None:
    p1, p2 = _make_corr(50, noise=0.0)
    cfg = _base_cfg(f_method="7pt")
    F, mask = _estimate_F(cfg, p1, p2)
    assert F.shape == (3, 3)
    assert np.linalg.matrix_rank(F, tol=1e-8) == 2
    assert mask.shape == (50,)


def test_estimate_F_7pt_too_few_raises() -> None:
    p1, p2 = _make_corr(6, noise=0.0)
    cfg = _base_cfg(f_method="7pt")
    with pytest.raises(ValueError, match="7-point needs"):
        _estimate_F(cfg, p1, p2)


def test_estimate_F_ransac_returns_rank2() -> None:
    p1, p2 = _make_corr(60, noise=0.5)
    cfg = _base_cfg(f_method="ransac", ransac_threshold_px=1.5)
    F, mask = _estimate_F(cfg, p1, p2)
    assert F.shape == (3, 3)
    assert np.linalg.matrix_rank(F, tol=1e-8) == 2
    assert mask.dtype == bool


def test_estimate_F_unknown_method_raises() -> None:
    p1, p2 = _make_corr(20)
    cfg = _base_cfg(f_method="bad_method")
    with pytest.raises(ValueError, match="Unknown f_method"):
        _estimate_F(cfg, p1, p2)


def test_estimate_F_all_methods_epipolar_constraint() -> None:
    """Each method should satisfy x2^T F x1 ≈ 0 on noise-free inliers."""
    p1, p2 = _make_corr(80, noise=0.0)
    for method in ("8pt", "7pt", "ransac"):
        cfg = _base_cfg(f_method=method)
        F, mask = _estimate_F(cfg, p1, p2)
        p1h = np.hstack([p1[mask], np.ones((mask.sum(), 1))])
        p2h = np.hstack([p2[mask], np.ones((mask.sum(), 1))])
        resid = np.abs((p2h * (F @ p1h.T).T).sum(axis=1))
        assert resid.max() < 0.1, f"method={method}: max epipolar resid {resid.max():.3e}"


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Sequence helper
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic_sequence(tmp_path: Path, n_frames: int = 4) -> Sequence:
    """Create a tiny TUM-style Sequence from numpy-generated images.

    Images are 480x640 uint8 with random dots so ORB finds features.
    No real depth or GT is needed for the pipeline smoke test.
    """
    rgb_dir = tmp_path / "rgb"
    rgb_dir.mkdir()

    rng = np.random.default_rng(0)
    frames = []
    for i in range(n_frames):
        # Build an image with random blob-like features that ORB can detect.
        img = np.zeros((480, 640), dtype=np.uint8)
        for _ in range(200):
            cx = int(rng.integers(10, 630))
            cy = int(rng.integers(10, 470))
            r = int(rng.integers(2, 6))
            cv2.circle(img, (cx, cy), r, int(rng.integers(100, 255)), -1)
        path = rgb_dir / f"frame_{i:04d}.png"
        cv2.imwrite(str(path), img)
        frames.append(Frame(
            timestamp=float(i),
            rgb_path=path,
            depth_path=path,        # dummy — pipeline doesn't read depth
            depth_timestamp=float(i),
            T_gt=None,
            gt_timestamp=None,
        ))

    return Sequence(
        root=tmp_path,
        name="rgbd_dataset_freiburg1_xyz",
        intrinsics=FREIBURG1,
        frames=frames,
    )


# ─────────────────────────────────────────────────────────────────────────────
# run_epipolar_pipeline smoke tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_seq(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("synth_seq")
    return _make_synthetic_sequence(tmp_path, n_frames=5)


def test_pipeline_trajectory_shape(synthetic_seq) -> None:
    cfg = _base_cfg(f_method="8pt")
    traj = run_epipolar_pipeline(synthetic_seq, cfg)
    assert traj.shape == (5, 4, 4)
    assert traj.dtype == np.float64


def test_pipeline_first_pose_is_identity(synthetic_seq) -> None:
    cfg = _base_cfg(f_method="8pt")
    traj = run_epipolar_pipeline(synthetic_seq, cfg)
    np.testing.assert_allclose(traj[0], np.eye(4), atol=1e-15)


def test_pipeline_all_poses_valid_se3(synthetic_seq) -> None:
    """Every pose should be a proper SE(3): det(R)=+1, R R^T = I, bottom row [0,0,0,1]."""
    cfg = _base_cfg(f_method="ransac", ransac_threshold_px=2.0)
    traj = run_epipolar_pipeline(synthetic_seq, cfg)
    for i, T in enumerate(traj):
        R = T[:3, :3]
        assert abs(np.linalg.det(R) - 1.0) < 1e-9, f"frame {i}: det(R) = {np.linalg.det(R):.6f}"
        err = np.linalg.norm(R @ R.T - np.eye(3))
        assert err < 1e-9, f"frame {i}: R R^T deviation {err:.2e}"
        np.testing.assert_array_equal(T[3], [0, 0, 0, 1])


def test_pipeline_max_frames_respected(synthetic_seq) -> None:
    cfg = _base_cfg(f_method="8pt")
    traj = run_epipolar_pipeline(synthetic_seq, cfg, max_frames=3)
    assert traj.shape == (3, 4, 4)


def test_pipeline_empty_sequence() -> None:
    empty_seq = Sequence(
        root=Path("/tmp"),
        name="rgbd_dataset_freiburg1_xyz",
        intrinsics=FREIBURG1,
        frames=[],
    )
    cfg = _base_cfg()
    traj = run_epipolar_pipeline(empty_seq, cfg)
    assert traj.shape == (0, 4, 4)


def test_pipeline_single_frame(tmp_path) -> None:
    seq = _make_synthetic_sequence(tmp_path, n_frames=1)
    cfg = _base_cfg()
    traj = run_epipolar_pipeline(seq, cfg)
    assert traj.shape == (1, 4, 4)
    np.testing.assert_allclose(traj[0], np.eye(4))
