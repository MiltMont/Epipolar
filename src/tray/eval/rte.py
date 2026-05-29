"""Relative Trajectory Error (RTE) as defined in PATHS.md.

Measures how accurately the pipeline estimates inter-frame displacements,
independent of global drift. For each pair (i, i+delta) the translational
part of the relative-motion error is accumulated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RTEResult:
    mean: float
    median: float
    std: float
    max: float
    rmse: float
    errors: np.ndarray  # (M,) per-pair translational errors in metres

    def as_dict(self) -> dict[str, float]:
        return {
            "rte_mean": self.mean,
            "rte_median": self.median,
            "rte_std": self.std,
            "rte_max": self.max,
            "rte_rmse": self.rmse,
        }


def _translational_error(
    T_est_i: np.ndarray,
    T_est_j: np.ndarray,
    T_gt_i: np.ndarray,
    T_gt_j: np.ndarray,
) -> float:
    """L2 norm of the translation part of the relative-motion error."""
    # Relative motions
    Q_est = np.linalg.inv(T_est_i) @ T_est_j
    Q_gt = np.linalg.inv(T_gt_i) @ T_gt_j
    # Error: how far the estimated relative motion deviates from GT
    E = np.linalg.inv(Q_gt) @ Q_est
    return float(np.linalg.norm(E[:3, 3]))


def compute_rte(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
    *,
    delta: int = 1,
) -> RTEResult:
    """Compute RTE over consecutive pairs separated by `delta` frames.

    Args:
        estimated:    (N, 4, 4) estimated trajectory (SE(3) matrices).
        ground_truth: (N, 4, 4) ground-truth trajectory (same N, same order).
        delta:        Frame-step for relative-motion pairs (default 1).

    Returns:
        RTEResult with mean/median/std/max/rmse and per-pair error vector.
    """
    estimated = np.asarray(estimated, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)

    N = len(estimated)
    errors: list[float] = []

    for i in range(N - delta):
        j = i + delta
        err = _translational_error(
            estimated[i], estimated[j],
            ground_truth[i], ground_truth[j],
        )
        errors.append(err)

    e = np.array(errors, dtype=np.float64)

    return RTEResult(
        mean=float(e.mean()),
        median=float(np.median(e)),
        std=float(e.std()),
        max=float(e.max()),
        rmse=float(np.sqrt((e ** 2).mean())),
        errors=e,
    )
