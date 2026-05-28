"""Feature detector factories and a thin Features container.

The factory takes a short string ("orb" | "sift") plus knobs so configs can
stay declarative; detection itself just delegates to OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Features:
    """Output of `detect_and_compute` for a single image.

    `keypoints` is OpenCV's `cv2.KeyPoint` list; `descriptors` is the matching
    `(N, D)` array (uint8 for ORB, float32 for SIFT). `points` returns the
    keypoint locations as an `(N, 2)` float array, which is what downstream
    epipolar code consumes.
    """

    keypoints: tuple[cv2.KeyPoint, ...]
    descriptors: np.ndarray

    @property
    def points(self) -> np.ndarray:
        if not self.keypoints:
            return np.zeros((0, 2), dtype=np.float64)
        return np.asarray([kp.pt for kp in self.keypoints], dtype=np.float64)

    def __len__(self) -> int:
        return len(self.keypoints)


def make_detector(kind: str, *, n_features: int = 1000) -> cv2.Feature2D:
    """Build an OpenCV detector. `kind` is "orb" or "sift" (case-insensitive)."""
    k = kind.lower()
    if k == "orb":
        return cv2.ORB_create(nfeatures=n_features)
    if k == "sift":
        # SIFT uses nfeatures=0 to mean "no cap"; pass `n_features` directly.
        return cv2.SIFT_create(nfeatures=n_features)
    raise ValueError(f"Unknown detector kind: {kind!r} (expected 'orb' or 'sift')")


def detect_and_compute(detector: cv2.Feature2D, image: np.ndarray) -> Features:
    """Run detect+compute on a grayscale `image` (HxW uint8).

    Returns empty descriptors `(0, D)` if the detector finds nothing, so
    callers don't have to special-case `None`.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected grayscale image, got shape {image.shape}")
    kps, desc = detector.detectAndCompute(image, None)
    if desc is None:
        # Pick a sensible empty shape per detector type.
        d = 32 if isinstance(detector, type(cv2.ORB_create())) else 128
        desc = np.zeros((0, d), dtype=np.uint8 if d == 32 else np.float32)
    return Features(keypoints=tuple(kps), descriptors=desc)
