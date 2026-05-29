"""3D trajectory visualisation using Matplotlib.

Each plot satisfies all VISUALISATION.md requirements:
  - Title, axis labels, legend, interpretable scale, dataset + config name.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(
    estimated: np.ndarray,
    ground_truth: np.ndarray | None = None,
    *,
    title: str = "Trajectory",
    dataset: str = "",
    config_name: str = "",
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot a single estimated trajectory with optional ground-truth overlay.

    Args:
        estimated:    (N, 4, 4) or (N, 3) estimated trajectory.
        ground_truth: (N, 4, 4) or (N, 3) GT trajectory (optional).
        title:        Base title string.
        dataset:      Dataset name appended to title.
        config_name:  Config name appended to title.
        save_path:    If provided, the figure is saved there (PNG).

    Returns:
        The matplotlib Figure (caller may call plt.show() or close it).
    """
    est_pos = _extract_positions(estimated)

    full_title = _build_title(title, dataset, config_name)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(est_pos[:, 0], est_pos[:, 1], est_pos[:, 2],
            color="steelblue", linewidth=1.2, label="Estimated")
    ax.scatter(*est_pos[0], color="steelblue", marker="o", s=40, zorder=5, label="Start (est)")

    if ground_truth is not None:
        gt_pos = _extract_positions(ground_truth)
        ax.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2],
                color="tomato", linewidth=1.2, linestyle="--", label="Ground truth")
        ax.scatter(*gt_pos[0], color="tomato", marker="o", s=40, zorder=5, label="Start (GT)")

    ax.set_title(full_title, fontsize=12)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(loc="upper left", fontsize=9)
    _equal_aspect_3d(ax)

    plt.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_trajectory_comparison(
    trajectories: dict[str, np.ndarray],
    ground_truth: np.ndarray | None = None,
    *,
    title: str = "Trajectory Comparison",
    dataset: str = "",
    save_path: Path | None = None,
) -> plt.Figure:
    """Overlay multiple estimated trajectories (one per config) on one 3D plot.

    Args:
        trajectories: Mapping config_name → (N, 4, 4) or (N, 3) trajectory.
        ground_truth: Optional GT trajectory.
        title:        Base title.
        dataset:      Dataset name appended to title.
        save_path:    If provided, the figure is saved there.

    Returns:
        The matplotlib Figure.
    """
    full_title = _build_title(title, dataset, "")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    for idx, (cfg_name, traj) in enumerate(trajectories.items()):
        pos = _extract_positions(traj)
        color = colors[idx % len(colors)]
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                color=color, linewidth=1.0, label=cfg_name)

    if ground_truth is not None:
        gt_pos = _extract_positions(ground_truth)
        ax.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2],
                color="black", linewidth=1.8, linestyle="--", label="Ground truth")

    ax.set_title(full_title, fontsize=12)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(loc="upper left", fontsize=8)
    _equal_aspect_3d(ax)

    plt.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_positions(traj: np.ndarray) -> np.ndarray:
    traj = np.asarray(traj)
    if traj.ndim == 3 and traj.shape[1:] == (4, 4):
        return traj[:, :3, 3]
    if traj.ndim == 2 and traj.shape[1] == 3:
        return traj
    raise ValueError(
        f"Trajectory must be (N, 4, 4) SE(3) or (N, 3) positions; got {traj.shape}"
    )


def _build_title(base: str, dataset: str, config_name: str) -> str:
    parts = [base]
    if dataset:
        parts.append(dataset)
    if config_name:
        parts.append(config_name)
    return " | ".join(parts)


def _equal_aspect_3d(ax: plt.Axes) -> None:
    """Force equal axis ranges for interpretable scale on a 3D axis."""
    data = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    midpoints = data.mean(axis=1)
    max_range = (data[:, 1] - data[:, 0]).max() / 2.0
    ax.set_xlim3d(midpoints[0] - max_range, midpoints[0] + max_range)
    ax.set_ylim3d(midpoints[1] - max_range, midpoints[1] + max_range)
    ax.set_zlim3d(midpoints[2] - max_range, midpoints[2] + max_range)


def _maybe_save(fig: plt.Figure, save_path: Path | None) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
