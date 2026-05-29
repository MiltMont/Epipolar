"""Run all 10 experiment configs sequentially and print a summary table."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from tray.experiments.runner import run_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CONFIGS = [
    # Epipolar (fast)
    "configs/epipolar_orb_8pt.yaml",
    "configs/epipolar_orb_7pt.yaml",
    "configs/epipolar_orb_ransac.yaml",
    "configs/epipolar_sift_lowe75.yaml",
    "configs/epipolar_sift_lowe60.yaml",
    # ICP (slow — ~10-15 min each)
    "configs/icp_baseline.yaml",
    "configs/icp_tight.yaml",
    "configs/icp_loose_reject.yaml",
    "configs/icp_voxel.yaml",
    "configs/icp_epipolar_init.yaml",
]


def _fmt(v: float | None, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}" if v is not None else "N/A"


def main() -> None:
    results = []
    failed = []

    for cfg_path in CONFIGS:
        logger.info("=" * 60)
        logger.info("Running: %s", cfg_path)
        logger.info("=" * 60)
        try:
            m = run_experiment(Path(cfg_path))
            results.append(m)
            ate_rmse = m.get("ate_rmse")
            rte_rmse = m.get("rte_rmse")
            logger.info(
                "DONE %s | ATE_RMSE=%.4f | RTE_RMSE=%.4f",
                m["config"],
                ate_rmse if ate_rmse else -1,
                rte_rmse if rte_rmse else -1,
            )
        except Exception as exc:
            logger.error("FAILED %s: %s", cfg_path, exc, exc_info=True)
            failed.append((cfg_path, str(exc)))

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'Config':<35} {'Pipeline':<10} {'ATE_RMSE':>10} {'ATE_mean':>10} {'RTE_RMSE':>10}")
    print("-" * 90)
    for m in results:
        print(
            f"{m['config']:<35} {m['pipeline']:<10} "
            f"{_fmt(m.get('ate_rmse')):>10} "
            f"{_fmt(m.get('ate_mean')):>10} "
            f"{_fmt(m.get('rte_rmse')):>10}"
        )
    print("=" * 90)

    # Save combined results
    out = Path("results") / "all_metrics.json"
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved combined metrics → %s", out)

    if failed:
        logger.error("Failed configs: %s", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
