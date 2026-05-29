"""cv2.drawMatches wrapper for feature correspondence visualisation."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def draw_matches(
    img1: np.ndarray,
    kp1: list,
    img2: np.ndarray,
    kp2: list,
    matches: list,
    *,
    max_matches: int = 50,
    title: str = "",
    save_path: Path | None = None,
) -> np.ndarray:
    """Visualise feature correspondences between two consecutive images.

    Args:
        img1:        Grayscale or BGR image 1.
        kp1:         List of cv2.KeyPoint for image 1.
        img2:        Grayscale or BGR image 2.
        kp2:         List of cv2.KeyPoint for image 2.
        matches:     List of cv2.DMatch sorted by distance (best first).
        max_matches: Cap on the number of correspondences drawn.
        title:       Text label overlaid on the output image.
        save_path:   If provided, save the output image as a PNG.

    Returns:
        BGR numpy array with drawn correspondences.
    """
    draw_list = matches[:max_matches]

    out = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        draw_list,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    if title:
        cv2.putText(
            out, title,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), out)

    return out
