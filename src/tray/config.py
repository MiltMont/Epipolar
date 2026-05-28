"""Experiment config dataclasses + YAML loader.

Filled out alongside `experiments/runner.py` in plan Phase 2 step 11.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class EpipolarConfig:
    name: str
    sequence: str  # e.g. "rgbd_dataset_freiburg1_xyz"
    detector: str  # "orb" | "sift"
    matcher: str   # "topn" | "lowe"
    f_method: str  # "8pt" | "7pt" | "ransac"
    n_features: int = 1000
    top_n: int = 200
    lowe_ratio: float = 0.75
    ransac_threshold_px: float = 1.0


@dataclass(frozen=True)
class ICPConfig:
    name: str
    sequence: str  # e.g. "rgbd_dataset_freiburg2_pioneer_slam"
    max_iter: int = 50
    tolerance: float = 1e-4
    reject_dist_m: float = 0.5
    voxel_downsample: int = 1   # 1 = no downsampling
    init_from_epipolar: bool = False


def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)
