"""Descriptor matching between two `Features` sets.

Provides two strategies the spec calls out:

- `match_top_n` — BFMatcher with cross-check, sort by distance, keep top N.
- `match_lowe`  — knn-2 BFMatcher with Lowe's ratio test.

Both return a `Matches` object with parallel `(N, 2)` point arrays so the
epipolar code can drop straight into fundamental-matrix estimation.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tray.features.detectors import Features


@dataclass(frozen=True)
class Matches:
    """N point correspondences `x ↔ x'` between image 1 and image 2."""

    pts1: np.ndarray   # (N, 2) float64, in image-1 pixel coords
    pts2: np.ndarray   # (N, 2) float64, in image-2 pixel coords
    distances: np.ndarray  # (N,) descriptor distance for each match

    def __len__(self) -> int:
        return len(self.distances)


def _norm_for(descriptors: np.ndarray) -> int:
    """Pick the BFMatcher norm from descriptor dtype.

    Binary descriptors (ORB/BRIEF/BRISK) are uint8 packed bits → Hamming;
    real-valued descriptors (SIFT/SURF) are float32 → L2.
    """
    if descriptors.dtype == np.uint8:
        return cv2.NORM_HAMMING
    return cv2.NORM_L2


def _pack(feat1: Features, feat2: Features, dmatches: list[cv2.DMatch]) -> Matches:
    if not dmatches:
        return Matches(
            pts1=np.zeros((0, 2), dtype=np.float64),
            pts2=np.zeros((0, 2), dtype=np.float64),
            distances=np.zeros((0,), dtype=np.float64),
        )
    p1 = np.asarray([feat1.keypoints[m.queryIdx].pt for m in dmatches], dtype=np.float64)
    p2 = np.asarray([feat2.keypoints[m.trainIdx].pt for m in dmatches], dtype=np.float64)
    dist = np.asarray([m.distance for m in dmatches], dtype=np.float64)
    return Matches(pts1=p1, pts2=p2, distances=dist)


def match_top_n(feat1: Features, feat2: Features, n: int) -> Matches:
    """BFMatcher with cross-check; return the `n` matches with smallest distance.

    Cross-check forces mutual nearest-neighbour pairs, which already discards
    most outliers before downstream filtering.
    """
    if len(feat1) == 0 or len(feat2) == 0:
        return _pack(feat1, feat2, [])
    bf = cv2.BFMatcher(_norm_for(feat1.descriptors), crossCheck=True)
    matches = sorted(bf.match(feat1.descriptors, feat2.descriptors), key=lambda m: m.distance)
    return _pack(feat1, feat2, matches[:n])


def match_lowe(feat1: Features, feat2: Features, ratio: float = 0.75) -> Matches:
    """BFMatcher knn-2 with Lowe's ratio test (`d1 < ratio * d2`).

    Cross-check is incompatible with knnMatch, so it's off here — the ratio
    test plays the role of outlier rejection.
    """
    if len(feat1) < 2 or len(feat2) < 2:
        return _pack(feat1, feat2, [])
    bf = cv2.BFMatcher(_norm_for(feat1.descriptors), crossCheck=False)
    knn = bf.knnMatch(feat1.descriptors, feat2.descriptors, k=2)
    good: list[cv2.DMatch] = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return _pack(feat1, feat2, good)
