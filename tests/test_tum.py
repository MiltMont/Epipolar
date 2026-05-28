from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tray.data import tum


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------

def test_intrinsics_K_shape_and_values() -> None:
    K = tum.FREIBURG1.K
    assert K.shape == (3, 3)
    assert K[0, 0] == pytest.approx(517.3)
    assert K[1, 1] == pytest.approx(516.5)
    assert K[0, 2] == pytest.approx(318.6)
    assert K[1, 2] == pytest.approx(255.3)
    assert K[2, 2] == 1.0
    # Off-diagonal upper-left zeros
    assert K[0, 1] == 0.0
    assert K[1, 0] == 0.0


def test_intrinsics_for_matches_family() -> None:
    assert tum.intrinsics_for("rgbd_dataset_freiburg1_xyz") is tum.FREIBURG1
    assert tum.intrinsics_for("rgbd_dataset_freiburg2_pioneer_slam") is tum.FREIBURG2
    assert tum.intrinsics_for("freiburg3_long_office_household") is tum.FREIBURG3


def test_intrinsics_for_unknown_raises() -> None:
    with pytest.raises(ValueError):
        tum.intrinsics_for("kitti_00")


def test_depth_scale_is_5000() -> None:
    # TUM-specific: depth_meters = png_uint16 / 5000
    assert tum.FREIBURG1.depth_scale == 5000.0


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------

def test_read_file_list_skips_comments_and_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "rgb.txt"
    p.write_text(
        "# color images\n"
        "# timestamp filename\n"
        "\n"
        "1305031102.175304 rgb/1305031102.175304.png\n"
        "1305031102.211214 rgb/1305031102.211214.png\n"
    )
    entries = tum.read_file_list(p)
    assert entries == [
        (1305031102.175304, "rgb/1305031102.175304.png"),
        (1305031102.211214, "rgb/1305031102.211214.png"),
    ]


def test_read_groundtruth_identity_quaternion(tmp_path: Path) -> None:
    p = tmp_path / "groundtruth.txt"
    # Identity quaternion (qx=qy=qz=0, qw=1), translation (1, 2, 3)
    p.write_text(
        "# t tx ty tz qx qy qz qw\n"
        "0.0 1.0 2.0 3.0 0.0 0.0 0.0 1.0\n"
    )
    entries = tum.read_groundtruth(p)
    assert len(entries) == 1
    t, T = entries[0]
    assert t == 0.0
    np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], [1.0, 2.0, 3.0])
    assert T[3, 3] == 1.0


def test_read_groundtruth_quaternion_90deg_z(tmp_path: Path) -> None:
    p = tmp_path / "groundtruth.txt"
    # 90° rotation about z: q = (0, 0, sin(45°), cos(45°))
    s = np.sqrt(0.5)
    p.write_text(f"0.0 0 0 0 0 0 {s} {s}\n")
    [(_, T)] = tum.read_groundtruth(p)
    expected = np.array([[0.0, -1.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(T[:3, :3], expected, atol=1e-12)


def test_read_groundtruth_rejects_malformed_row(tmp_path: Path) -> None:
    p = tmp_path / "gt.txt"
    p.write_text("0.0 1 2 3 0 0 0\n")  # missing qw
    with pytest.raises(ValueError):
        tum.read_groundtruth(p)


# ---------------------------------------------------------------------------
# Association
# ---------------------------------------------------------------------------

def test_associate_basic_pairing() -> None:
    a = [(0.00, "a0"), (0.10, "a1"), (0.20, "a2")]
    b = [(0.01, "b0"), (0.11, "b1"), (0.19, "b2")]
    pairs = tum.associate(a, b, max_delta=0.02)
    assert [(ta, tb) for ta, tb, _, _ in pairs] == [(0.00, 0.01), (0.10, 0.11), (0.20, 0.19)]


def test_associate_drops_pairs_beyond_max_delta() -> None:
    a = [(0.0, "a"), (1.0, "a")]
    b = [(0.5, "b"), (1.01, "b")]
    pairs = tum.associate(a, b, max_delta=0.05)
    # Only (1.0, 1.01) is within 0.05; (0.0, 0.5) is too far.
    assert [(ta, tb) for ta, tb, _, _ in pairs] == [(1.0, 1.01)]


def test_associate_is_greedy_smallest_delta_first() -> None:
    # If both a0 and a1 could pair with b0, the one with smaller delta wins.
    a = [(0.00, "a0"), (0.02, "a1")]
    b = [(0.03, "b0")]
    pairs = tum.associate(a, b, max_delta=0.05)
    # a1 (delta=0.01) beats a0 (delta=0.03), so only a1 is paired.
    assert [(ta, tb) for ta, tb, _, _ in pairs] == [(0.02, 0.03)]


def test_associate_each_entry_used_at_most_once() -> None:
    a = [(0.00, "a0"), (0.01, "a1")]
    b = [(0.005, "b0")]
    pairs = tum.associate(a, b, max_delta=0.05)
    # b0 can only pair once; a0 has the smaller delta.
    assert len(pairs) == 1


def test_associate_empty_inputs() -> None:
    assert tum.associate([], [(0.0, "b")]) == []
    assert tum.associate([(0.0, "a")], []) == []


# ---------------------------------------------------------------------------
# load_sequence on the real dataset (skipped if not downloaded)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
XYZ_DIR = REPO_ROOT / "data" / "rgbd_dataset_freiburg1_xyz"


@pytest.mark.skipif(not XYZ_DIR.exists(), reason="TUM xyz dataset not downloaded")
def test_load_sequence_xyz_real_data() -> None:
    seq = tum.load_sequence(XYZ_DIR)
    assert seq.name == "rgbd_dataset_freiburg1_xyz"
    assert seq.intrinsics is tum.FREIBURG1
    assert len(seq.frames) > 100  # xyz is ~30 s @ 30 Hz, plenty of frames

    f0 = seq.frames[0]
    assert f0.rgb_path.exists()
    assert f0.depth_path.exists()
    # rgb/depth should be paired within 0.02 s by default
    assert abs(f0.timestamp - f0.depth_timestamp) <= 0.02

    # Pick a frame that does have GT (mocap dropouts can leave a few without).
    f_gt = next(f for f in seq.frames if f.T_gt is not None)

    # GT covers nearly the whole xyz sequence; the mocap stream has a few
    # ~100 ms dropouts so a handful of rgb frames legitimately lack a GT
    # match within 0.02 s. We still expect the overwhelming majority covered.
    n_with_gt = sum(1 for f in seq.frames if f.T_gt is not None)
    assert n_with_gt >= int(0.99 * len(seq.frames))

    # T_gt should be a valid SE(3): rotation orthonormal, last row = [0,0,0,1].
    T = f_gt.T_gt
    np.testing.assert_allclose(T[3], [0, 0, 0, 1])
    R = T[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)
