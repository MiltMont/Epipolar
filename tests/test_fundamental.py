"""Tests for epipolar/fundamental.py.

Each from-scratch estimator (eight_point, seven_point, ransac) is verified two ways:
  1. Epipolar constraint on synthetic noise-free correspondences.
  2. Cross-check against cv2.findFundamentalMat on the same data.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from tray.epipolar.fundamental import (
    eight_point,
    ransac,
    sampson_distance,
    seven_point,
)

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic scene — shared across all tests
# ─────────────────────────────────────────────────────────────────────────────

# Camera intrinsics (common to both cameras in the synthetic pair).
_K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])

# Camera-2 pose relative to camera-1: small Y-rotation + lateral baseline.
_angle = 0.2
_R2 = np.array([
    [np.cos(_angle), 0, np.sin(_angle)],
    [0, 1, 0],
    [-np.sin(_angle), 0, np.cos(_angle)],
])
_t2 = np.array([0.5, 0.05, 0.1])


def _project(R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Project 3-D points X (N, 3) → pixel coords (N, 2) with intrinsics _K."""
    Xc = (R @ X.T).T + t
    xh = (_K @ Xc.T).T
    return xh[:, :2] / xh[:, 2:3]


def _make_corr(
    n: int = 50,
    noise: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate n point correspondences from the shared synthetic camera pair."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(low=[-1.5, -1.0, 3.5], high=[1.5, 1.0, 6.5], size=(n, 3))
    p1 = _project(np.eye(3), np.zeros(3), X)
    p2 = _project(_R2, _t2, X)
    if noise > 0.0:
        p1 += rng.normal(0.0, noise, p1.shape)
        p2 += rng.normal(0.0, noise, p2.shape)
    return p1, p2


def _make_corr_with_outliers(
    n_inliers: int = 80,
    n_outliers: int = 20,
    noise: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inliers from the synthetic geometry + random pixel-space outliers."""
    rng = np.random.default_rng(seed)
    p1_in, p2_in = _make_corr(n_inliers, noise=noise, seed=seed)
    p1_out = rng.uniform(0.0, 640.0, (n_outliers, 2))
    p2_out = rng.uniform(0.0, 480.0, (n_outliers, 2))
    p1 = np.vstack([p1_in, p1_out])
    p2 = np.vstack([p2_in, p2_out])
    gt_mask = np.zeros(n_inliers + n_outliers, dtype=bool)
    gt_mask[:n_inliers] = True
    return p1, p2, gt_mask


def _algebraic_residuals(F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Unnormalized algebraic epipolar error |x2^T F x1| for each pair."""
    p1h = np.column_stack([pts1, np.ones(len(pts1))])
    p2h = np.column_stack([pts2, np.ones(len(pts2))])
    return np.abs((p2h * (F @ p1h.T).T).sum(axis=1))


# ─────────────────────────────────────────────────────────────────────────────
# eight_point
# ─────────────────────────────────────────────────────────────────────────────

def test_eight_point_satisfies_epipolar_constraint() -> None:
    """With exact correspondences the Sampson distance should be near machine zero."""
    p1, p2 = _make_corr(50, noise=0.0)
    F = eight_point(p1, p2)

    assert F.shape == (3, 3)
    dist = sampson_distance(F, p1, p2)
    assert dist.max() < 1e-8, f"max Sampson: {dist.max():.2e}"


def test_eight_point_rank_is_two() -> None:
    """The returned F must be rank-2."""
    p1, p2 = _make_corr(50, noise=0.0)
    F = eight_point(p1, p2)
    rank = np.linalg.matrix_rank(F, tol=1e-8)
    assert rank == 2, f"expected rank 2, got {rank}"


def test_eight_point_frobenius_normalized() -> None:
    p1, p2 = _make_corr(30, noise=0.5)
    F = eight_point(p1, p2)
    assert abs(np.linalg.norm(F) - 1.0) < 1e-10


def test_eight_point_vs_opencv_matrix_exact() -> None:
    """On noise-free data our F should be proportional (±sign) to cv2's FM_8POINT."""
    p1, p2 = _make_corr(50, noise=0.0)
    F = eight_point(p1, p2)
    F_cv, _ = cv2.findFundamentalMat(
        p1.astype(np.float32), p2.astype(np.float32), cv2.FM_8POINT
    )
    assert F_cv is not None

    F_cv_n = F_cv / np.linalg.norm(F_cv)
    diff = min(np.linalg.norm(F - F_cv_n), np.linalg.norm(F + F_cv_n))
    assert diff < 0.05, f"matrix diff after sign alignment: {diff:.4f}"


def test_eight_point_vs_opencv_sampson_noisy() -> None:
    """On noisy data, median Sampson distances should be within 30% of each other."""
    p1, p2 = _make_corr(100, noise=0.5, seed=7)
    F = eight_point(p1, p2)
    F_cv, _ = cv2.findFundamentalMat(
        p1.astype(np.float32), p2.astype(np.float32), cv2.FM_8POINT
    )
    assert F_cv is not None

    med_ours = float(np.median(sampson_distance(F, p1, p2)))
    med_cv = float(np.median(sampson_distance(F_cv, p1, p2)))
    if med_cv > 0:
        assert abs(med_ours - med_cv) / med_cv < 0.30, (
            f"median Sampson: ours={med_ours:.4f}, cv2={med_cv:.4f}"
        )


def test_eight_point_raises_on_too_few() -> None:
    p1, p2 = _make_corr(7)
    with pytest.raises(ValueError, match="≥ 8"):
        eight_point(p1, p2)


# ─────────────────────────────────────────────────────────────────────────────
# seven_point
# ─────────────────────────────────────────────────────────────────────────────

def test_seven_point_at_least_one_candidate() -> None:
    """A real degree-3 polynomial always has at least one real root."""
    p1, p2 = _make_corr(7, noise=0.0)
    candidates = seven_point(p1, p2)
    assert len(candidates) >= 1


def test_seven_point_all_candidates_rank_two() -> None:
    p1, p2 = _make_corr(7, noise=0.0)
    for F in seven_point(p1, p2):
        assert F.shape == (3, 3)
        assert np.linalg.matrix_rank(F, tol=1e-8) == 2


def test_seven_point_satisfies_constraint_on_inputs() -> None:
    """All 7 input correspondences lie in the null space of every candidate F.

    Since F = αF₁ + F₂ and both F₁, F₂ satisfy A·vec(F) = 0, any linear
    combination also satisfies the constraint for the 7 training points.
    """
    p1, p2 = _make_corr(7, noise=0.0)
    for F in seven_point(p1, p2):
        resid = _algebraic_residuals(F, p1, p2)
        assert resid.max() < 1e-5, f"max algebraic residual: {resid.max():.2e}"


def test_seven_point_best_candidate_generalizes() -> None:
    """Pick the candidate with lowest Sampson on a holdout; it should still be small."""
    p1_train, p2_train = _make_corr(7, noise=0.0, seed=0)
    p1_test, p2_test = _make_corr(50, noise=0.0, seed=99)

    candidates = seven_point(p1_train, p2_train)
    best = min(candidates, key=lambda F: sampson_distance(F, p1_test, p2_test).mean())

    dist_test = sampson_distance(best, p1_test, p2_test)
    assert dist_test.max() < 1e-6, f"holdout max Sampson: {dist_test.max():.2e}"


def test_seven_point_vs_opencv_holdout() -> None:
    """Our best candidate and cv2's FM_7POINT should give comparable holdout Sampson.

    cv2.FM_7POINT may return 1–3 stacked (3×3) matrices in a (3k×3) array;
    we pick the best block from cv2 just as we pick the best from our candidates.
    """
    p1_7, p2_7 = _make_corr(7, noise=0.0, seed=3)
    p1_h, p2_h = _make_corr(50, noise=0.0, seed=77)

    candidates = seven_point(p1_7, p2_7)
    best_ours = min(candidates, key=lambda F: sampson_distance(F, p1_h, p2_h).mean())

    F_cv, _ = cv2.findFundamentalMat(
        p1_7.astype(np.float32), p2_7.astype(np.float32), cv2.FM_7POINT
    )
    assert F_cv is not None
    # Split (3k, 3) → list of k (3, 3) blocks and pick the best one.
    n_cv = F_cv.shape[0] // 3
    cv_blocks = [F_cv[3 * i : 3 * i + 3] for i in range(n_cv)]
    best_cv = min(cv_blocks, key=lambda F: sampson_distance(F, p1_h, p2_h).mean())

    d_ours = float(sampson_distance(best_ours, p1_h, p2_h).max())
    d_cv = float(sampson_distance(best_cv, p1_h, p2_h).max())

    # Our best candidate should satisfy the holdout at near-machine precision.
    assert d_ours < 1e-5, f"ours holdout max Sampson: {d_ours:.2e}"
    # cv2 operates in float32, so allow a looser bound.
    assert d_cv < 1.0, f"cv2 best holdout max Sampson: {d_cv:.2e}"


def test_seven_point_raises_on_wrong_count() -> None:
    p1, p2 = _make_corr(8)
    with pytest.raises(ValueError, match="exactly 7"):
        seven_point(p1, p2)


# ─────────────────────────────────────────────────────────────────────────────
# ransac
# ─────────────────────────────────────────────────────────────────────────────

def test_ransac_recovers_inlier_set() -> None:
    """RANSAC should identify most genuine inliers and reject most outliers."""
    p1, p2, gt_mask = _make_corr_with_outliers(80, 20, noise=0.5, seed=42)
    F, inlier_mask = ransac(p1, p2, threshold=1.5, max_iter=1000, seed=42)

    assert F.shape == (3, 3)
    assert inlier_mask.shape == (100,)

    # Recall: at least 87.5% of the 80 true inliers should be found.
    tp = inlier_mask[:80].sum()
    assert tp >= 70, f"true inlier recall: {tp}/80"

    # Precision: at most 5 outliers misclassified.
    fp = inlier_mask[80:].sum()
    assert fp <= 5, f"false positives: {fp}/20"


def test_ransac_outperforms_naive_eight_point() -> None:
    """With 40% outliers the naive 8-point should clearly lose to RANSAC."""
    p1, p2, _ = _make_corr_with_outliers(60, 40, noise=0.5, seed=1)
    thresh_sq = 1.5 ** 2

    F_naive = eight_point(p1, p2)
    naive_inliers = int((sampson_distance(F_naive, p1, p2) < thresh_sq).sum())

    F_rans, rans_mask = ransac(p1, p2, threshold=1.5, max_iter=1000, seed=0)
    rans_inliers = int(rans_mask.sum())

    assert rans_inliers > naive_inliers, (
        f"RANSAC ({rans_inliers}) should beat naive 8-pt ({naive_inliers})"
    )


def test_ransac_vs_opencv_inlier_count() -> None:
    """Our inlier count should reach at least 90% of OpenCV's FM_RANSAC count."""
    p1, p2, _ = _make_corr_with_outliers(80, 20, noise=0.5, seed=0)
    _, inlier_mask = ransac(p1, p2, threshold=1.5, max_iter=1000, seed=7)

    _, mask_cv = cv2.findFundamentalMat(
        p1.astype(np.float32), p2.astype(np.float32),
        cv2.FM_RANSAC,
        ransacReprojThreshold=1.5,
        confidence=0.99,
        maxIters=1000,
    )
    n_cv = int(mask_cv.ravel().sum()) if mask_cv is not None else 0
    n_ours = int(inlier_mask.sum())

    assert n_ours >= 0.9 * n_cv, f"ours: {n_ours}, cv2: {n_cv}"


def test_ransac_deterministic_with_seed() -> None:
    """Same seed should produce identical results across calls."""
    p1, p2, _ = _make_corr_with_outliers(80, 20)
    F1, m1 = ransac(p1, p2, threshold=1.5, max_iter=200, seed=99)
    F2, m2 = ransac(p1, p2, threshold=1.5, max_iter=200, seed=99)
    np.testing.assert_array_equal(m1, m2)
    np.testing.assert_array_almost_equal(F1, F2)


def test_ransac_raises_on_too_few() -> None:
    p1 = np.random.randn(7, 2)
    p2 = np.random.randn(7, 2)
    with pytest.raises(ValueError):
        ransac(p1, p2)
