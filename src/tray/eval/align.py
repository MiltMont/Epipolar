"""Umeyama similarity-transform alignment of an estimated trajectory to ground truth.

Reference: S. Umeyama, "Least-squares estimation of transformation parameters
between two point patterns", IEEE PAMI 13(4), 1991.
"""
from __future__ import annotations

import numpy as np


def umeyama(
    P: np.ndarray,
    Q: np.ndarray,
    *,
    with_scale: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Find the similarity transform that minimises MSE between P and Q.

    Solves:  Q ≈ c * R @ P.T + t  (applied row-by-row).

    Args:
        P: (N, 3) estimated positions (source).
        Q: (N, 3) ground-truth positions (target).
        with_scale: True → solve for scale c (use for monocular/epipolar,
                    which is up-to-scale). False → fix c = 1.0 (use for
                    metric ICP).

    Returns:
        R: (3, 3) rotation matrix.
        t: (3,) translation vector.
        c: float scale factor.

    Usage:
        R, t, c = umeyama(P, Q, with_scale=True)
        P_aligned = apply_alignment(P, R, t, c)
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"P and Q must both be (N, 3); got {P.shape} and {Q.shape}")

    N = len(P)
    mu_P = P.mean(0)
    mu_Q = Q.mean(0)
    P_c = P - mu_P
    Q_c = Q - mu_Q

    var_P = float((P_c ** 2).sum() / N)

    Sigma = (Q_c.T @ P_c) / N  # (3, 3) cross-covariance

    U, S, Vt = np.linalg.svd(Sigma)

    # Correct for reflections so det(R) = +1.
    d = np.linalg.det(U) * np.linalg.det(Vt)
    diag = np.ones(3, dtype=np.float64)
    if d < 0:
        diag[-1] = -1.0

    R = U @ np.diag(diag) @ Vt

    if with_scale and var_P > 0.0:
        c = float((S * diag).sum() / var_P)
    else:
        c = 1.0

    t = mu_Q - c * (R @ mu_P)

    return R, t, c


def apply_alignment(
    P: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    c: float,
) -> np.ndarray:
    """Apply similarity transform: returns c * (P @ R.T) + t row-by-row."""
    P = np.asarray(P, dtype=np.float64)
    return c * (P @ R.T) + t
