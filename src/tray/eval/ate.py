"""Absolute Trajectory Error (ATE) as defined in PATHS.md.

Reports: mean, median, std, max, RMSE of per-frame positional errors after
Umeyama alignment of the estimated trajectory to ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tray.eval.align import apply_alignment, umeyama


@dataclass
class ATEResult:
    mean: float
    median: float
    std: float
    max: float
    rmse: float
    errors: np.ndarray  # (N,) per-frame errors in metres

    def as_dict(self) -> dict[str, float]:
        return {
            "ate_mean": self.mean,
            "ate_median": self.median,
            "ate_std": self.std,
            "ate_max": self.max,
            "ate_rmse": self.rmse,
        }


def umeyama_align(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
    *,
    with_scale: bool = True,
) -> np.ndarray:
    """Return estimated trajectory spatially aligned to GT via Umeyama.

    Args:
        estimated:    (N, 4, 4) estimated SE(3) trajectory.
        ground_truth: (N, 4, 4) ground-truth SE(3) trajectory (same N).
        with_scale:   True for monocular/epipolar; False for ICP.

    Returns:
        (N, 4, 4) array with translation columns replaced by aligned positions.
        Rotations are left unchanged (Umeyama aligns positions only).
    """
    estimated = np.asarray(estimated, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)

    P = estimated[:, :3, 3]
    Q = ground_truth[:, :3, 3]

    R, t, c = umeyama(P, Q, with_scale=with_scale)
    P_aligned = apply_alignment(P, R, t, c)

    out = estimated.copy()
    out[:, :3, 3] = P_aligned
    return out


def compute_ate(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
    *,
    with_scale: bool = True,
) -> ATEResult:
    """Compute ATE after Umeyama alignment.

    Args:
        estimated:    (N, 4, 4) estimated trajectory (SE(3) matrices).
        ground_truth: (N, 4, 4) ground-truth trajectory (same N, same order).
        with_scale:   True for monocular/epipolar (up-to-scale); False for ICP.

    Returns:
        ATEResult with mean/median/std/max/rmse and per-frame error vector.
    """
    estimated = np.asarray(estimated, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)

    P = estimated[:, :3, 3]      # (N, 3) estimated positions
    Q = ground_truth[:, :3, 3]   # (N, 3) GT positions

    R, t, c = umeyama(P, Q, with_scale=with_scale)
    P_aligned = apply_alignment(P, R, t, c)

    errors = np.linalg.norm(P_aligned - Q, axis=1)

    return ATEResult(
        mean=float(errors.mean()),
        median=float(np.median(errors)),
        std=float(errors.std()),
        max=float(errors.max()),
        rmse=float(np.sqrt((errors ** 2).mean())),
        errors=errors,
    )
