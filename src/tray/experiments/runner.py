"""Experiment runner: YAML config → pipeline → results/<config_name>/."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np

from tray.config import EpipolarConfig, ICPConfig, load_yaml
from tray.data.tum import Sequence, load_sequence
from tray.epipolar.pipeline import run_epipolar_pipeline
from tray.eval.ate import compute_ate
from tray.eval.rte import compute_rte
from tray.icp.pipeline import run_icp_pipeline
from tray.viz.trajectory import plot_trajectory

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config factories
# ─────────────────────────────────────────────────────────────────────────────

def _epipolar_cfg(raw: dict) -> EpipolarConfig:
    return EpipolarConfig(
        name=raw["name"],
        sequence=raw["sequence"],
        detector=raw["detector"],
        matcher=raw["matcher"],
        f_method=raw["f_method"],
        n_features=int(raw.get("n_features", 1000)),
        top_n=int(raw.get("top_n", 200)),
        lowe_ratio=float(raw.get("lowe_ratio", 0.75)),
        ransac_threshold_px=float(raw.get("ransac_threshold_px", 1.0)),
    )


def _icp_cfg(raw: dict) -> ICPConfig:
    return ICPConfig(
        name=raw["name"],
        sequence=raw["sequence"],
        max_iter=int(raw.get("max_iter", 50)),
        tolerance=float(raw.get("tolerance", 1e-4)),
        reject_dist_m=float(raw.get("reject_dist_m", 0.5)),
        voxel_downsample=int(raw.get("voxel_downsample", 1)),
        init_from_epipolar=bool(raw.get("init_from_epipolar", False)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# GT extraction
# ─────────────────────────────────────────────────────────────────────────────

def _paired_trajectories(
    seq: Sequence,
    estimated: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (est_filtered, gt_filtered) keeping only frames with a GT match."""
    est_list = []
    gt_list = []
    for i, frame in enumerate(seq.frames):
        if i >= len(estimated):
            break
        if frame.T_gt is not None:
            est_list.append(estimated[i])
            gt_list.append(frame.T_gt)
    return np.array(est_list, dtype=np.float64), np.array(gt_list, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runners
# ─────────────────────────────────────────────────────────────────────────────

def _run_epipolar(
    raw: dict,
    data_dir: Path,
    max_frames: int | None,
) -> tuple[EpipolarConfig, Sequence, np.ndarray]:
    cfg = _epipolar_cfg(raw)
    seq = load_sequence(data_dir / cfg.sequence)
    logger.info("Running epipolar pipeline: %s (%d frames)", cfg.name, len(seq.frames))
    traj = run_epipolar_pipeline(seq, cfg, max_frames=max_frames)
    return cfg, seq, traj


def _run_icp(
    raw: dict,
    data_dir: Path,
    config_path: Path,
    max_frames: int | None,
) -> tuple[ICPConfig, Sequence, np.ndarray]:
    cfg = _icp_cfg(raw)
    seq = load_sequence(data_dir / cfg.sequence)

    epi_traj: np.ndarray | None = None
    if cfg.init_from_epipolar:
        epi_config_key = raw.get("epipolar_config")
        if epi_config_key is None:
            raise ValueError(
                "ICP config with init_from_epipolar=True must include an "
                "'epipolar_config' key with a path (relative to this config) "
                "to an epipolar YAML config."
            )
        epi_path = config_path.parent / epi_config_key
        epi_raw = load_yaml(epi_path)
        epi_cfg = _epipolar_cfg(epi_raw)
        epi_seq = load_sequence(data_dir / epi_cfg.sequence)
        logger.info("Running epipolar seed: %s", epi_cfg.name)
        epi_traj = run_epipolar_pipeline(epi_seq, epi_cfg, max_frames=max_frames)

    logger.info("Running ICP pipeline: %s (%d frames)", cfg.name, len(seq.frames))
    traj = run_icp_pipeline(seq, cfg, epipolar_trajectory=epi_traj, max_frames=max_frames)
    return cfg, seq, traj


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    config_path: Path,
    *,
    results_dir: Path = Path("results"),
    data_dir: Path = Path("data"),
    max_frames: int | None = None,
) -> dict:
    """Load a YAML config, run the pipeline, persist results, return metrics.

    Output layout (all under `results_dir/<config_name>/`):
        metrics.json      — ATE + RTE stats (JSON)
        trajectory.npy    — (N, 4, 4) estimated SE(3) trajectory
        trajectory.png    — 3D plot of estimated vs ground truth
    """
    config_path = Path(config_path)
    raw = load_yaml(config_path)
    pipeline_type = raw.get("pipeline")

    if pipeline_type == "epipolar":
        cfg, seq, traj = _run_epipolar(raw, data_dir, max_frames)
        with_scale = True
    elif pipeline_type == "icp":
        cfg, seq, traj = _run_icp(raw, data_dir, config_path, max_frames)
        with_scale = False
    else:
        raise ValueError(
            f"Config must have pipeline: 'epipolar' or 'icp'; got {pipeline_type!r}"
        )

    # ── Evaluation ────────────────────────────────────────────────────────────
    est_paired, gt_paired = _paired_trajectories(seq, traj)

    metrics: dict = {
        "config": cfg.name,
        "pipeline": pipeline_type,
        "n_frames": int(len(traj)),
        "n_gt_frames": int(len(gt_paired)),
    }

    if len(gt_paired) >= 2:
        ate = compute_ate(est_paired, gt_paired, with_scale=with_scale)
        rte = compute_rte(est_paired, gt_paired)
        metrics.update(ate.as_dict())
        metrics.update(rte.as_dict())
        logger.info(
            "%s — ATE RMSE=%.4f m | RTE RMSE=%.4f",
            cfg.name, ate.rmse, rte.rmse,
        )
    else:
        logger.warning("%s — not enough GT frames to compute ATE/RTE", cfg.name)

    # ── Persist ───────────────────────────────────────────────────────────────
    out_dir = Path(results_dir) / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "trajectory.npy", traj)

    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    est_for_plot = est_paired if len(est_paired) >= 2 else traj
    gt_for_plot = gt_paired if len(gt_paired) >= 2 else None

    fig = plot_trajectory(
        est_for_plot,
        gt_for_plot,
        title="Trajectory",
        dataset=cfg.sequence,
        config_name=cfg.name,
        save_path=out_dir / "trajectory.png",
    )
    plt.close(fig)

    logger.info("Results written to %s", out_dir)
    return metrics
