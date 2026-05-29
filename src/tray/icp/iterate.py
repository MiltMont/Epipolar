"""Iterative Closest Point loop.

Algorithm per iteration:
  1. Build KD-tree on the (fixed) target cloud.
  2. Query nearest neighbour for every source point.
  3. Reject correspondences whose NN distance exceeds reject_dist.
  4. Call align_svd on the accepted pairs → (R_step, t_step).
  5. Apply (R_step, t_step) to the working source cloud.
  6. Accumulate the global transform: R_acc, t_acc.
  7. Stop when |prev_mean_dist - mean_dist| < tolerance or max_iter reached.

The function returns the accumulated (R, t) that maps the original source into
the target frame, plus convergence metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from tray.icp.align import align_svd


@dataclass
class ICPResult:
    R: np.ndarray           # (3, 3) accumulated rotation
    t: np.ndarray           # (3,) accumulated translation
    n_iterations: int
    converged: bool
    mean_dist_history: list[float] = field(default_factory=list)


def icp(
    source: np.ndarray,
    target: np.ndarray,
    *,
    max_iter: int = 50,
    tolerance: float = 1e-4,
    reject_dist: float = 0.5,
    init_R: np.ndarray | None = None,
    init_t: np.ndarray | None = None,
) -> ICPResult:
    """Run ICP to align source cloud to target cloud.

    Args:
        source:      (N, 3) source point cloud (will not be modified).
        target:      (M, 3) target point cloud (fixed reference).
        max_iter:    Maximum number of ICP iterations.
        tolerance:   Stop when the change in mean NN-distance drops below this.
        reject_dist: Reject correspondences with NN distance above this (metres).
        init_R:      Optional (3, 3) initial rotation (e.g. from epipolar).
        init_t:      Optional (3,) initial translation.

    Returns:
        ICPResult with accumulated R, t and convergence info.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    # Apply optional initial transform.
    if init_R is not None and init_t is not None:
        src = source @ init_R.T + init_t
        R_acc = np.asarray(init_R, dtype=np.float64).copy()
        t_acc = np.asarray(init_t, dtype=np.float64).copy()
    else:
        src = source.copy()
        R_acc = np.eye(3, dtype=np.float64)
        t_acc = np.zeros(3, dtype=np.float64)

    tree = cKDTree(target)  # built once; target is fixed

    prev_mean = np.inf
    mean_dist_history: list[float] = []
    converged = False

    for iteration in range(max_iter):
        dists, idx = tree.query(src, k=1, workers=1)

        valid = dists < reject_dist
        n_valid = int(valid.sum())

        if n_valid < 3:
            break

        src_valid = src[valid]
        tgt_valid = target[idx[valid]]

        R_step, t_step = align_svd(src_valid, tgt_valid)

        src = src @ R_step.T + t_step

        # Accumulate: new_R = R_step @ R_acc, new_t = R_step @ t_acc + t_step
        R_acc = R_step @ R_acc
        t_acc = R_step @ t_acc + t_step

        mean_dist = float(dists[valid].mean())
        mean_dist_history.append(mean_dist)

        if abs(prev_mean - mean_dist) < tolerance:
            converged = True
            break

        prev_mean = mean_dist

    return ICPResult(
        R=R_acc,
        t=t_acc,
        n_iterations=len(mean_dist_history),
        converged=converged,
        mean_dist_history=mean_dist_history,
    )
