"""TUM RGB-D loader.

Handles three tasks:

1. Per-sequence pinhole intrinsics (different K/distortion for fr1/fr2/fr3).
2. Parsing TUM list files (`rgb.txt`, `depth.txt`) and ground-truth files.
3. Nearest-timestamp association between rgb/depth/groundtruth streams,
   using the greedy algorithm from the official TUM `associate.py`.

Reference: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics + radial-tangential distortion + TUM depth scale."""

    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, float, float, float, float]  # k1, k2, p1, p2, k3
    depth_scale: float = 5000.0  # TUM convention: depth_meters = png_uint16 / 5000

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def dist_coeffs(self) -> np.ndarray:
        return np.asarray(self.distortion, dtype=np.float64)


# Values from https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
FREIBURG1 = Intrinsics(
    fx=517.3, fy=516.5, cx=318.6, cy=255.3,
    distortion=(0.2624, -0.9531, -0.0054, 0.0026, 1.1633),
)
FREIBURG2 = Intrinsics(
    fx=520.9, fy=521.0, cx=325.1, cy=249.7,
    distortion=(0.2312, -0.7849, -0.0033, -0.0001, 0.9172),
)
FREIBURG3 = Intrinsics(
    fx=535.4, fy=539.2, cx=320.1, cy=247.6,
    distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
)


def intrinsics_for(sequence_name: str) -> Intrinsics:
    """Pick intrinsics by sequence directory name.

    Accepts either the full directory ("rgbd_dataset_freiburg1_xyz") or the
    bare family token ("freiburg1").
    """
    name = sequence_name.lower()
    if "freiburg1" in name:
        return FREIBURG1
    if "freiburg2" in name:
        return FREIBURG2
    if "freiburg3" in name:
        return FREIBURG3
    raise ValueError(f"Cannot infer intrinsics from sequence name: {sequence_name!r}")


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------

def read_file_list(path: Path) -> list[tuple[float, str]]:
    """Parse a TUM list file (`rgb.txt`, `depth.txt`).

    Each non-comment line is `<timestamp> <relative_path>`. Timestamps are
    seconds. Returns the entries in file order.
    """
    entries: list[tuple[float, str]] = []
    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            ts_str, rel = line.split(maxsplit=1)
            entries.append((float(ts_str), rel))
    return entries


def read_groundtruth(path: Path) -> list[tuple[float, np.ndarray]]:
    """Parse `groundtruth.txt`. Each row is `t tx ty tz qx qy qz qw`.

    Returns a list of `(timestamp, T_world_camera)` with T as a 4x4 SE(3) matrix.
    """
    entries: list[tuple[float, np.ndarray]] = []
    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                raise ValueError(f"Malformed groundtruth row in {path}: {raw!r}")
            t = float(parts[0])
            tx, ty, tz = (float(x) for x in parts[1:4])
            qx, qy, qz, qw = (float(x) for x in parts[4:8])
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = (tx, ty, tz)
            entries.append((t, T))
    return entries


# ---------------------------------------------------------------------------
# Timestamp association (TUM's associate.py algorithm)
# ---------------------------------------------------------------------------

def associate(
    a: Iterable[tuple[float, object]],
    b: Iterable[tuple[float, object]],
    *,
    max_delta: float = 0.02,
) -> list[tuple[float, float, object, object]]:
    """Greedy nearest-timestamp pairing, same algorithm as TUM's associate.py.

    Yields `(t_a, t_b, payload_a, payload_b)` for each pair, in order of
    increasing `t_a`. A pair is only emitted if `|t_a - t_b| <= max_delta`,
    and each entry is used at most once (greedy by smallest delta first).

    O(N*M) in time but bounded by `max_delta` in practice; fine for TUM
    sequences (a few thousand frames).
    """
    a_list = list(a)
    b_list = list(b)

    candidates: list[tuple[float, int, int]] = []
    for i, (ta, _) in enumerate(a_list):
        for j, (tb, _) in enumerate(b_list):
            d = abs(ta - tb)
            if d <= max_delta:
                candidates.append((d, i, j))

    candidates.sort(key=lambda x: x[0])
    used_a: set[int] = set()
    used_b: set[int] = set()
    chosen: list[tuple[int, int]] = []
    for _, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        chosen.append((i, j))

    chosen.sort(key=lambda ij: a_list[ij[0]][0])
    return [
        (a_list[i][0], b_list[j][0], a_list[i][1], b_list[j][1])
        for i, j in chosen
    ]


# ---------------------------------------------------------------------------
# Sequence loading
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Frame:
    timestamp: float       # rgb timestamp
    rgb_path: Path
    depth_path: Path
    depth_timestamp: float
    T_gt: np.ndarray | None        # 4x4 world←camera, or None if no GT match
    gt_timestamp: float | None


@dataclass(frozen=True)
class Sequence:
    root: Path
    name: str
    intrinsics: Intrinsics
    frames: list[Frame]


def load_sequence(root: Path, *, max_delta: float = 0.02) -> Sequence:
    """Load a TUM sequence directory and associate rgb ↔ depth ↔ groundtruth.

    Expected layout:
        root/rgb.txt
        root/depth.txt
        root/groundtruth.txt
        root/rgb/<files>
        root/depth/<files>

    Frames whose rgb has no depth pair within `max_delta` are dropped.
    Frames whose rgb has no GT pair within `max_delta` are kept but get
    `T_gt = None` — useful so estimation can still run on the full rgb stream.
    """
    root = Path(root)
    rgb_entries = read_file_list(root / "rgb.txt")
    depth_entries = read_file_list(root / "depth.txt")
    gt_entries = read_groundtruth(root / "groundtruth.txt")

    rgb_depth = associate(rgb_entries, depth_entries, max_delta=max_delta)

    # Build a quick lookup: rgb_ts -> (depth_ts, depth_rel)
    rgb_to_depth = {ta: (tb, dep) for ta, tb, _, dep in rgb_depth}

    # Associate rgb (only the ones we kept above) with GT.
    rgb_kept = [(ta, None) for ta in rgb_to_depth]
    gt_pairs = associate(rgb_kept, gt_entries, max_delta=max_delta)
    rgb_to_gt: dict[float, tuple[float, np.ndarray]] = {
        ta: (tg, T) for ta, tg, _, T in gt_pairs
    }

    frames: list[Frame] = []
    for ta, rgb_rel in rgb_entries:
        if ta not in rgb_to_depth:
            continue
        depth_ts, depth_rel = rgb_to_depth[ta]
        if ta in rgb_to_gt:
            gt_ts, T_gt = rgb_to_gt[ta]
        else:
            gt_ts, T_gt = None, None
        frames.append(Frame(
            timestamp=ta,
            rgb_path=root / rgb_rel,
            depth_path=root / depth_rel,
            depth_timestamp=depth_ts,
            T_gt=T_gt,
            gt_timestamp=gt_ts,
        ))

    return Sequence(
        root=root,
        name=root.name,
        intrinsics=intrinsics_for(root.name),
        frames=frames,
    )
