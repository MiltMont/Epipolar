"""From-scratch fundamental matrix estimators.

Three algorithms:
  eight_point — Hartley-normalized 8-point algorithm.
  seven_point — 7-point algorithm; up to 3 candidate matrices via a cubic root.
  ransac      — RANSAC with 8-point inner solver and Sampson-distance inlier test.

Cross-check against cv2.findFundamentalMat in tests/ — never use OpenCV here.
"""
from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_pts(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalization: centroid → origin, mean distance → √2.

    Returns (pts_normalized (N, 2), T (3, 3)).
    """
    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    mean_dist = np.sqrt((shifted ** 2).sum(axis=1)).mean()
    if mean_dist < 1e-8:
        return pts.copy(), np.eye(3)
    s = np.sqrt(2.0) / mean_dist
    T = np.array([
        [s, 0.0, -s * centroid[0]],
        [0.0, s, -s * centroid[1]],
        [0.0, 0.0, 1.0],
    ])
    pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=pts.dtype)])
    pts_n = (T @ pts_h.T).T[:, :2]
    return pts_n, T


def _build_A(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Constraint matrix A — one row per correspondence.

    Row i: [x2·x1, x2·y1, x2, y2·x1, y2·y1, y2, x1, y1, 1]
    """
    x1, y1 = pts1[:, 0], pts1[:, 1]
    x2, y2 = pts2[:, 0], pts2[:, 1]
    return np.column_stack([
        x2 * x1, x2 * y1, x2,
        y2 * x1, y2 * y1, y2,
        x1, y1, np.ones(len(pts1), dtype=pts1.dtype),
    ])


def _enforce_rank2(F: np.ndarray) -> np.ndarray:
    """Project F to the nearest rank-2 matrix: zero the smallest singular value."""
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0.0
    return U @ np.diag(S) @ Vt


def _denorm(F_n: np.ndarray, T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """Undo Hartley normalization, return Frobenius-normalized result.

    Derivation: x2_n = T2 x2, x1_n = T1 x1, x2_n^T F_n x1_n = 0
      ⟹  x2^T (T2^T F_n T1) x1 = 0  ⟹  F = T2^T F_n T1.
    """
    F = T2.T @ F_n @ T1
    nrm = np.linalg.norm(F)
    return F / nrm if nrm > 1e-10 else F


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def sampson_distance(F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Sampson approximation to the symmetric epipolar distance.

    d_S(x, x') = (x'^T F x)² / ((Fx)[0]² + (Fx)[1]² + (F^T x')[0]² + (F^T x')[1]²)

    Args:
        F:    (3, 3) fundamental matrix.
        pts1: (N, 2) float64 pixel coordinates in image 1.
        pts2: (N, 2) float64 pixel coordinates in image 2.

    Returns:
        (N,) Sampson distances in units of pixels² (approx. square reprojection error).
    """
    ones = np.ones((len(pts1), 1), dtype=np.float64)
    p1h = np.hstack([pts1.astype(np.float64), ones])    # (N, 3)
    p2h = np.hstack([pts2.astype(np.float64), ones])    # (N, 3)
    Fp1 = (F @ p1h.T).T                                  # (N, 3)
    FtP2 = (F.T @ p2h.T).T                               # (N, 3)
    num = (p2h * Fp1).sum(axis=1) ** 2
    denom = Fp1[:, 0] ** 2 + Fp1[:, 1] ** 2 + FtP2[:, 0] ** 2 + FtP2[:, 1] ** 2
    return num / (denom + 1e-10)


def eight_point(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Hartley-normalized 8-point fundamental matrix.

    1. Normalize each point set to zero-centroid, mean-dist = √2 (Hartley).
    2. Build the (N×9) constraint matrix A.
    3. F_n = right singular vector of A for the smallest singular value.
    4. Enforce rank-2 (zero the smallest singular value of F_n).
    5. Denormalize: F = T2^T F_n T1.

    Args:
        pts1: (N, 2) float64 pixel coordinates in image 1 (N ≥ 8).
        pts2: (N, 2) float64 pixel coordinates in image 2 (N ≥ 8).

    Returns:
        F: (3, 3) rank-2 fundamental matrix, Frobenius-normalized.

    Raises:
        ValueError: if fewer than 8 correspondences are provided.
    """
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    if len(pts1) < 8:
        raise ValueError(f"eight_point requires ≥ 8 correspondences, got {len(pts1)}")
    if len(pts1) != len(pts2):
        raise ValueError("pts1 and pts2 must have the same length")

    pts1_n, T1 = _normalize_pts(pts1)
    pts2_n, T2 = _normalize_pts(pts2)
    A = _build_A(pts1_n, pts2_n)
    _, _, Vt = np.linalg.svd(A, full_matrices=True)
    F_n = _enforce_rank2(Vt[-1].reshape(3, 3))
    return _denorm(F_n, T1, T2)


def seven_point(pts1: np.ndarray, pts2: np.ndarray) -> list[np.ndarray]:
    """7-point algorithm producing up to 3 candidate fundamental matrices.

    With 7 correspondences the constraint matrix A is (7×9); its null space is
    2-D (spanned by F1, F2 — the last two right singular vectors).  Any valid F
    lies in span{F1, F2}: F = αF1 + F2.  det(αF1 + F2) = 0 is a degree-3
    polynomial in α; each real root yields one candidate.

    Args:
        pts1: (7, 2) float64 correspondences in image 1.
        pts2: (7, 2) float64 correspondences in image 2.

    Returns:
        List of 1–3 rank-2 fundamental matrices (Frobenius-normalized).

    Raises:
        ValueError: if the inputs do not have exactly 7 rows.
    """
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    if len(pts1) != 7 or len(pts2) != 7:
        raise ValueError("seven_point requires exactly 7 correspondences")

    pts1_n, T1 = _normalize_pts(pts1)
    pts2_n, T2 = _normalize_pts(pts2)
    A = _build_A(pts1_n, pts2_n)        # (7, 9)
    _, _, Vt = np.linalg.svd(A, full_matrices=True)
    F1 = Vt[-1].reshape(3, 3)           # right null vector 1
    F2 = Vt[-2].reshape(3, 3)           # right null vector 2

    # Build det(α·F1 + F2) as a degree-3 polynomial in α.
    # Each element e(i,j) = α·F1[i,j] + F2[i,j] is degree-1 in α.
    def e(i: int, j: int) -> np.poly1d:
        return np.poly1d([F1[i, j], F2[i, j]])

    det_poly: np.poly1d = (
        e(0, 0) * (e(1, 1) * e(2, 2) - e(1, 2) * e(2, 1))
        - e(0, 1) * (e(1, 0) * e(2, 2) - e(1, 2) * e(2, 0))
        + e(0, 2) * (e(1, 0) * e(2, 1) - e(1, 1) * e(2, 0))
    )

    roots = np.roots(det_poly.coeffs)
    candidates: list[np.ndarray] = []
    for root in roots:
        # Reject roots with a significant imaginary part (relative to real part).
        if abs(root.imag) / (abs(root.real) + 1.0) > 1e-4:
            continue
        alpha = float(root.real)
        F_cand = _enforce_rank2(alpha * F1 + F2)
        candidates.append(_denorm(F_cand, T1, T2))

    return candidates


def ransac(
    pts1: np.ndarray,
    pts2: np.ndarray,
    threshold: float = 1.0,
    max_iter: int = 1000,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """RANSAC fundamental matrix estimation.

    Minimal solver: 8-point algorithm on random 8-point subsets.
    Scoring:        Sampson distance; inlier iff d_S < threshold².
    Final estimate: refit eight_point on all consensus-set inliers.

    Args:
        pts1:      (N, 2) float64 pixel coordinates in image 1 (N ≥ 8).
        pts2:      (N, 2) float64 pixel coordinates in image 2 (N ≥ 8).
        threshold: Pixel threshold; point is inlier iff Sampson dist < threshold².
        max_iter:  Number of RANSAC iterations.
        seed:      Optional RNG seed for reproducibility.

    Returns:
        F:       (3, 3) rank-2 fundamental matrix.
        inliers: (N,) bool array — True for each inlier correspondence.

    Raises:
        ValueError:  if fewer than 8 correspondences are provided.
        RuntimeError: if no valid model is found in any iteration.
    """
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    n = len(pts1)
    if n < 8:
        raise ValueError(f"ransac requires ≥ 8 correspondences, got {n}")
    if len(pts1) != len(pts2):
        raise ValueError("pts1 and pts2 must have the same length")

    rng = np.random.default_rng(seed)
    thresh_sq = threshold ** 2

    best_F: np.ndarray | None = None
    best_mask = np.zeros(n, dtype=bool)
    best_count = 0

    for _ in range(max_iter):
        idx = rng.choice(n, 8, replace=False)
        try:
            F_cand = eight_point(pts1[idx], pts2[idx])
        except (np.linalg.LinAlgError, ValueError):
            continue

        dist = sampson_distance(F_cand, pts1, pts2)
        mask = dist < thresh_sq
        count = int(mask.sum())

        if count > best_count:
            best_count = count
            best_F = F_cand
            best_mask = mask.copy()

    if best_F is None:
        raise RuntimeError("RANSAC found no valid model in any iteration")

    # Refit on all inliers for a more stable final estimate.
    if best_count >= 8:
        try:
            F_refit = eight_point(pts1[best_mask], pts2[best_mask])
            dist_refit = sampson_distance(F_refit, pts1, pts2)
            return F_refit, dist_refit < thresh_sq
        except (np.linalg.LinAlgError, ValueError):
            pass

    return best_F, best_mask
