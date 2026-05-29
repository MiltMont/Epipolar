"""Tests for epipolar/pose.py.

Verification strategy:
  1. Synthetic noise-free scene — mathematical properties (constraint, det, depth).
  2. Numerical recovery — (R, t) recovered by recover_pose should match the
     ground-truth rotation up to sign of t (monocular up-to-scale).
  3. Cross-check triangulate_dlt reprojection error against OpenCV.
  4. Cross-check recover_pose against cv2.recoverPose inlier count.
"""
from __future__ import annotations

import cv2
import numpy as np

from tray.epipolar.fundamental import eight_point, ransac
from tray.epipolar.pose import (
    decompose_E,
    essential_from_F,
    recover_pose,
    triangulate_dlt,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared synthetic scene
# ─────────────────────────────────────────────────────────────────────────────

_K = np.array([[800.0, 0.0, 320.0],
               [0.0, 800.0, 240.0],
               [0.0,   0.0,   1.0]], dtype=np.float64)

_angle = 0.2
_R_gt = np.array([
    [np.cos(_angle), 0.0, np.sin(_angle)],
    [0.0,            1.0, 0.0],
    [-np.sin(_angle), 0.0, np.cos(_angle)],
], dtype=np.float64)
_t_gt = np.array([0.5, 0.05, 0.1], dtype=np.float64)
_t_gt_unit = _t_gt / np.linalg.norm(_t_gt)


def _project(R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xc = (R @ X.T).T + t
    xh = (_K @ Xc.T).T
    return xh[:, :2] / xh[:, 2:3]


def _make_corr(n: int = 80, noise: float = 0.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pts1, pts2, X_3d)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(low=[-1.5, -1.0, 3.5], high=[1.5, 1.0, 6.5], size=(n, 3))
    p1 = _project(np.eye(3), np.zeros(3), X)
    p2 = _project(_R_gt, _t_gt, X)
    if noise > 0.0:
        p1 += rng.normal(0.0, noise, p1.shape)
        p2 += rng.normal(0.0, noise, p2.shape)
    return p1, p2, X


# ─────────────────────────────────────────────────────────────────────────────
# essential_from_F
# ─────────────────────────────────────────────────────────────────────────────

def test_essential_epipolar_constraint() -> None:
    """Normalised coords x_n = K^{-1} x must satisfy x2_n^T E x1_n ≈ 0."""
    p1, p2, _ = _make_corr(50, noise=0.0)
    F = eight_point(p1, p2)
    E = essential_from_F(F, _K)

    Ki = np.linalg.inv(_K)
    p1h = np.hstack([p1, np.ones((len(p1), 1))])
    p2h = np.hstack([p2, np.ones((len(p2), 1))])
    p1_n = (Ki @ p1h.T).T
    p2_n = (Ki @ p2h.T).T

    residuals = np.abs((p2_n * (E @ p1_n.T).T).sum(axis=1))
    assert residuals.max() < 1e-6, f"max epipolar residual: {residuals.max():.2e}"


def test_essential_shape() -> None:
    p1, p2, _ = _make_corr(30)
    F = eight_point(p1, p2)
    E = essential_from_F(F, _K)
    assert E.shape == (3, 3)


def test_essential_singular_values() -> None:
    """True E should have two equal non-zero singular values and one zero."""
    # Construct E directly from ground truth as a reference.
    t_cross = np.array([
        [0., -_t_gt[2], _t_gt[1]],
        [_t_gt[2], 0., -_t_gt[0]],
        [-_t_gt[1], _t_gt[0], 0.],
    ])
    E_gt = t_cross @ _R_gt

    sv = np.linalg.svd(E_gt, compute_uv=False)
    assert abs(sv[0] - sv[1]) / (sv[0] + 1e-10) < 0.01, f"sv: {sv}"
    assert sv[2] < 1e-10, f"smallest sv: {sv[2]:.2e}"


# ─────────────────────────────────────────────────────────────────────────────
# decompose_E
# ─────────────────────────────────────────────────────────────────────────────

def test_decompose_returns_four_candidates() -> None:
    p1, p2, _ = _make_corr(50, noise=0.0)
    F = eight_point(p1, p2)
    E = essential_from_F(F, _K)
    candidates = decompose_E(E)
    assert len(candidates) == 4


def test_decompose_rotation_det_one() -> None:
    p1, p2, _ = _make_corr(50, noise=0.0)
    F = eight_point(p1, p2)
    E = essential_from_F(F, _K)
    for R, _ in decompose_E(E):
        assert R.shape == (3, 3)
        assert abs(np.linalg.det(R) - 1.0) < 1e-9, f"det(R) = {np.linalg.det(R):.6f}"


def test_decompose_unit_translation() -> None:
    p1, p2, _ = _make_corr(50, noise=0.0)
    F = eight_point(p1, p2)
    E = essential_from_F(F, _K)
    # Translations come in ±t pairs; check each has unit norm.
    for _, t in decompose_E(E):
        assert abs(np.linalg.norm(t) - 1.0) < 1e-9, f"|t| = {np.linalg.norm(t):.6f}"


def test_decompose_rotation_orthogonality() -> None:
    p1, p2, _ = _make_corr(50, noise=0.0)
    F = eight_point(p1, p2)
    E = essential_from_F(F, _K)
    for R, _ in decompose_E(E):
        err = np.linalg.norm(R @ R.T - np.eye(3))
        assert err < 1e-9, f"R R^T deviation: {err:.2e}"


# ─────────────────────────────────────────────────────────────────────────────
# triangulate_dlt
# ─────────────────────────────────────────────────────────────────────────────

def test_triangulate_reprojection_error_noise_free() -> None:
    """With ground-truth P matrices, reprojection error should be near zero."""
    p1, p2, X_gt = _make_corr(50, noise=0.0)

    P1 = _K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = _K @ np.hstack([_R_gt, _t_gt.reshape(3, 1)])

    X = triangulate_dlt(P1, P2, p1, p2)
    assert X.shape == (50, 3)

    # Reproject and measure error in pixels.
    Xh = np.hstack([X, np.ones((len(X), 1))])
    p1_repr = (P1 @ Xh.T).T
    p1_repr = p1_repr[:, :2] / p1_repr[:, 2:3]
    err = np.linalg.norm(p1_repr - p1, axis=1)
    assert err.max() < 1e-5, f"max reprojection error: {err.max():.2e} px"


def test_triangulate_3d_accuracy_noise_free() -> None:
    """Triangulated points should match ground-truth 3-D coordinates."""
    p1, p2, X_gt = _make_corr(30, noise=0.0)

    P1 = _K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = _K @ np.hstack([_R_gt, _t_gt.reshape(3, 1)])

    X = triangulate_dlt(P1, P2, p1, p2)
    err = np.linalg.norm(X - X_gt, axis=1)
    assert err.max() < 1e-6, f"max 3-D error: {err.max():.2e} m"


def test_triangulate_vs_opencv_reprojection() -> None:
    """Our reprojection error should be close to cv2.triangulatePoints."""
    p1, p2, _ = _make_corr(40, noise=0.5)

    P1 = _K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = _K @ np.hstack([_R_gt, _t_gt.reshape(3, 1)])

    X_ours = triangulate_dlt(P1, P2, p1, p2)

    X_cv4 = cv2.triangulatePoints(
        P1.astype(np.float32), P2.astype(np.float32),
        p1.T.astype(np.float32), p2.T.astype(np.float32),
    )  # (4, N)
    X_cv = (X_cv4[:3] / X_cv4[3]).T  # (N, 3)

    # Both should produce similar 3-D points (noisy input → noisy output).
    diff = np.linalg.norm(X_ours - X_cv, axis=1)
    assert diff.mean() < 0.05, f"mean 3-D diff vs cv2: {diff.mean():.4f} m"


def test_triangulate_output_shape() -> None:
    p1, p2, _ = _make_corr(20)
    P1 = _K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = _K @ np.hstack([_R_gt, _t_gt.reshape(3, 1)])
    X = triangulate_dlt(P1, P2, p1, p2)
    assert X.shape == (20, 3)


# ─────────────────────────────────────────────────────────────────────────────
# recover_pose
# ─────────────────────────────────────────────────────────────────────────────

def _gt_E() -> np.ndarray:
    """Ground-truth essential matrix from the known geometry."""
    t_cross = np.array([
        [0., -_t_gt[2], _t_gt[1]],
        [_t_gt[2], 0., -_t_gt[0]],
        [-_t_gt[1], _t_gt[0], 0.],
    ])
    return t_cross @ _R_gt


def test_recover_pose_returns_correct_rotation() -> None:
    """With the ground-truth E, the recovered R should match _R_gt."""
    p1, p2, _ = _make_corr(80, noise=0.0)
    E = _gt_E()
    R, t, mask = recover_pose(E, _K, p1, p2)

    # Rotation error angle.
    angle_err = np.arccos(
        np.clip((np.trace(R @ _R_gt.T) - 1.0) / 2.0, -1.0, 1.0)
    )
    assert angle_err < 0.01, f"rotation error: {np.degrees(angle_err):.3f} deg"


def test_recover_pose_translation_direction() -> None:
    """Recovered t (unit) should align with the ground-truth unit translation."""
    p1, p2, _ = _make_corr(80, noise=0.0)
    E = _gt_E()
    _, t, _ = recover_pose(E, _K, p1, p2)

    # monocular t is only up to sign.
    alignment = abs(float(np.dot(t, _t_gt_unit)))
    assert alignment > 0.999, f"|cos θ| = {alignment:.6f}"


def test_recover_pose_cheirality_mask() -> None:
    """All points flagged as valid by the mask should be in front of both cameras."""
    p1, p2, X_gt = _make_corr(60, noise=0.0)
    E = _gt_E()
    R, t, mask = recover_pose(E, _K, p1, p2)

    P1 = _K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = _K @ np.hstack([R, t.reshape(3, 1)])
    X = triangulate_dlt(P1, P2, p1, p2)

    z1 = X[mask, 2]
    z2 = ((R @ X[mask].T).T + t)[:, 2]
    assert (z1 > 0).all(), "some masked points have negative depth in cam-1"
    assert (z2 > 0).all(), "some masked points have negative depth in cam-2"


def test_recover_pose_most_points_valid() -> None:
    """With noise-free GT inputs, almost all points should pass cheirality."""
    p1, p2, _ = _make_corr(100, noise=0.0)
    E = _gt_E()
    _, _, mask = recover_pose(E, _K, p1, p2)
    assert mask.sum() >= 95, f"only {mask.sum()}/100 points passed cheirality"


def test_recover_pose_vs_opencv() -> None:
    """Our inlier count from cheirality should be close to cv2.recoverPose."""
    p1, p2, _ = _make_corr(100, noise=0.5, seed=7)
    F, inlier_mask = ransac(p1, p2, threshold=1.5, max_iter=1000, seed=7)
    E = essential_from_F(F, _K)

    p1_in = p1[inlier_mask]
    p2_in = p2[inlier_mask]

    _, _, mask_ours = recover_pose(E, _K, p1_in, p2_in)
    n_ours = int(mask_ours.sum())

    n_cv, _, _, _ = cv2.recoverPose(
        E.astype(np.float64),
        p1_in.astype(np.float64),
        p2_in.astype(np.float64),
        _K.astype(np.float64),
    )

    # Accept within 10 points of cv2 (differences arise from float32 vs float64
    # internal normalisation in OpenCV).
    assert abs(n_ours - n_cv) <= 10, f"ours: {n_ours}, cv2: {n_cv}"


def test_recover_pose_output_shapes() -> None:
    p1, p2, _ = _make_corr(50)
    E = _gt_E()
    R, t, mask = recover_pose(E, _K, p1, p2)
    assert R.shape == (3, 3)
    assert t.shape == (3,)
    assert mask.shape == (50,)
    assert mask.dtype == bool
