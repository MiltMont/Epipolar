"""Single-step SVD rigid alignment for ICP.

Given N corresponding point pairs (source[i] ↔ target[i]) the function
computes the (R, t) that minimises sum ||R @ source[i] + t - target[i]||^2.

Algorithm (Horn 1987 / Arun et al. 1987):
  1. Compute centroids mu_s, mu_t.
  2. Centre the clouds: S = source - mu_s, T = target - mu_t.
  3. Cross-covariance: H = S.T @ T.
  4. SVD: U, Σ, V^T = svd(H).
  5. R = V U^T.
  6. If det(R) < 0 (reflection), negate the last column of V and recompute.
  7. t = mu_t - R @ mu_s.
"""
from __future__ import annotations

import numpy as np


def align_svd(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the rigid transform (R, t) aligning source to target.

    Args:
        source: (N, 3) source point cloud.
        target: (N, 3) target point cloud (same ordering as source).

    Returns:
        R: (3, 3) rotation matrix, det(R) = +1.
        t: (3,) translation vector.

    The aligned source is  source @ R.T + t  (or equivalently R @ source.T + t
    applied column-wise).
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(
            f"source and target must both be (N, 3); got {source.shape} and {target.shape}"
        )

    mu_s = source.mean(0)
    mu_t = target.mean(0)

    S = source - mu_s
    T = target - mu_t

    H = S.T @ T  # (3, 3) cross-covariance

    U, _, Vt = np.linalg.svd(H)

    # V columns, not V^T rows
    V = Vt.T

    R = V @ U.T

    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = V @ U.T

    t = mu_t - R @ mu_s

    return R, t
