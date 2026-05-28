from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from tray.features.detectors import detect_and_compute, make_detector
from tray.features.matching import match_lowe, match_top_n


# ---------------------------------------------------------------------------
# Synthetic image pair: a scatter of dark blobs on a light background, with
# the second image shifted by (dx, dy). Both ORB and SIFT can detect blobs
# like these, so we can exercise the full pipeline without real photos.
# ---------------------------------------------------------------------------

DX, DY = 25, 10


def _make_image_pair(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    canvas = np.full((400, 600), 230, dtype=np.uint8)
    centres = rng.integers(low=40, high=(360, 560), size=(60, 2))
    for cy, cx in centres:
        cv2.circle(canvas, (int(cx), int(cy)), radius=6, color=20, thickness=-1)
    # Translate by (DX, DY) using a constant-border shift.
    M = np.float32([[1, 0, DX], [0, 1, DY]])
    shifted = cv2.warpAffine(canvas, M, (canvas.shape[1], canvas.shape[0]),
                             borderValue=230)
    return canvas, shifted


# ---------------------------------------------------------------------------
# Detector factory
# ---------------------------------------------------------------------------

def test_make_detector_orb_returns_uint8_descriptors() -> None:
    img, _ = _make_image_pair()
    det = make_detector("orb", n_features=500)
    feat = detect_and_compute(det, img)
    assert len(feat) > 0
    assert feat.descriptors.dtype == np.uint8
    assert feat.descriptors.shape[1] == 32  # 256-bit packed
    assert feat.points.shape == (len(feat), 2)


def test_make_detector_sift_returns_float32_descriptors() -> None:
    img, _ = _make_image_pair()
    det = make_detector("sift", n_features=500)
    feat = detect_and_compute(det, img)
    assert len(feat) > 0
    assert feat.descriptors.dtype == np.float32
    assert feat.descriptors.shape[1] == 128


def test_make_detector_is_case_insensitive() -> None:
    assert make_detector("ORB") is not None
    assert make_detector("Sift") is not None


def test_make_detector_unknown_raises() -> None:
    with pytest.raises(ValueError):
        make_detector("akaze")


def test_detect_and_compute_rejects_color_image() -> None:
    det = make_detector("orb")
    color = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        detect_and_compute(det, color)


# ---------------------------------------------------------------------------
# Matching — translation recovery sanity check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("detector_kind", ["orb", "sift"])
def test_match_top_n_recovers_known_translation(detector_kind: str) -> None:
    img1, img2 = _make_image_pair()
    det = make_detector(detector_kind, n_features=500)
    f1 = detect_and_compute(det, img1)
    f2 = detect_and_compute(det, img2)

    m = match_top_n(f1, f2, n=50)
    assert len(m) > 0
    assert m.pts1.shape == m.pts2.shape == (len(m), 2)

    # Median displacement of the top matches should match the synthetic shift.
    # 5 px tolerance: SIFT keypoints near the warp border pick up the
    # constant-border edge and pull the median around; the downstream RANSAC
    # in the epipolar pipeline is what filters those — this is just a smoke
    # test that the matcher returns something coherent.
    delta = m.pts2 - m.pts1
    np.testing.assert_allclose(np.median(delta, axis=0), [DX, DY], atol=5.0)


@pytest.mark.parametrize("detector_kind", ["orb", "sift"])
def test_match_lowe_recovers_known_translation(detector_kind: str) -> None:
    img1, img2 = _make_image_pair()
    det = make_detector(detector_kind, n_features=500)
    f1 = detect_and_compute(det, img1)
    f2 = detect_and_compute(det, img2)

    m = match_lowe(f1, f2, ratio=0.75)
    assert len(m) > 0
    delta = m.pts2 - m.pts1
    np.testing.assert_allclose(np.median(delta, axis=0), [DX, DY], atol=5.0)


def test_match_top_n_returns_at_most_n() -> None:
    img1, img2 = _make_image_pair()
    det = make_detector("orb", n_features=500)
    f1 = detect_and_compute(det, img1)
    f2 = detect_and_compute(det, img2)
    m = match_top_n(f1, f2, n=10)
    assert len(m) <= 10
    # Distances should be sorted ascending.
    assert np.all(np.diff(m.distances) >= 0)


def test_match_handles_empty_features() -> None:
    det = make_detector("orb")
    empty_feat = detect_and_compute(det, np.full((50, 50), 128, dtype=np.uint8))
    # Random low-contrast noise may produce 0 features; force the empty path:
    from tray.features.detectors import Features
    empty = Features(keypoints=(), descriptors=np.zeros((0, 32), dtype=np.uint8))
    m = match_top_n(empty, empty_feat, n=10)
    assert len(m) == 0
    m2 = match_lowe(empty, empty_feat)
    assert len(m2) == 0


# ---------------------------------------------------------------------------
# Real-data smoke test on consecutive TUM xyz frames
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
XYZ_DIR = REPO_ROOT / "data" / "rgbd_dataset_freiburg1_xyz"


@pytest.mark.skipif(not XYZ_DIR.exists(), reason="TUM xyz dataset not downloaded")
def test_match_on_real_consecutive_frames() -> None:
    from tray.data import tum
    seq = tum.load_sequence(XYZ_DIR)
    img1 = cv2.imread(str(seq.frames[0].rgb_path), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(seq.frames[1].rgb_path), cv2.IMREAD_GRAYSCALE)
    assert img1 is not None and img2 is not None

    det = make_detector("orb", n_features=1000)
    f1 = detect_and_compute(det, img1)
    f2 = detect_and_compute(det, img2)
    assert len(f1) > 100 and len(f2) > 100

    m = match_top_n(f1, f2, n=200)
    # Consecutive xyz frames are ~33 ms apart with mostly translation; the
    # median displacement should be a small handful of pixels.
    assert len(m) > 50
    median_disp = np.median(np.linalg.norm(m.pts2 - m.pts1, axis=1))
    assert median_disp < 20.0
