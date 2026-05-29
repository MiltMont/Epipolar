"""Depth image → 3-D point cloud for the ICP pipeline.

TUM RGB-D specifics:
  - 16-bit PNG depth images; divide by 5000 to get metres.
  - Zero pixels indicate invalid / missing depth — must be masked out.
  - Back-projection: (u, v, z) → ((u-cx)*z/fx, (v-cy)*z/fy, z).
"""
from __future__ import annotations

import numpy as np


def depth_to_pointcloud(
    depth: np.ndarray,
    K: np.ndarray,
    *,
    depth_scale: float = 5000.0,
    max_depth: float = 10.0,
) -> np.ndarray:
    """Back-project a TUM depth image into a (N, 3) point cloud.

    Args:
        depth:       (H, W) uint16 (or float) depth image.
        K:           (3, 3) camera intrinsics [fx 0 cx; 0 fy cy; 0 0 1].
        depth_scale: Divisor to convert raw pixel values to metres (TUM = 5000).
        max_depth:   Points farther than this (metres) are discarded.

    Returns:
        (N, 3) float64 array of 3-D points in the camera frame.
        N ≤ H*W; zero-depth and far-depth pixels are excluded.
    """
    depth = np.asarray(depth, dtype=np.float64)
    H, W = depth.shape

    z = depth / depth_scale          # metres

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = np.arange(W, dtype=np.float64)
    v = np.arange(H, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)        # (H, W) each

    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy

    valid = (depth > 0) & (z <= max_depth)

    return np.stack([x[valid], y[valid], z[valid]], axis=-1)


def downsample(
    cloud: np.ndarray,
    factor: int,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Randomly subsample a point cloud by an integer factor.

    Args:
        cloud:  (N, 3) point cloud.
        factor: Keep roughly 1/factor of the points.  1 = no-op.
        rng:    Optional RNG for reproducibility.

    Returns:
        (M, 3) subsampled cloud, M ≈ N // factor.
    """
    if factor <= 1:
        return cloud
    if rng is None:
        rng = np.random.default_rng()
    N = len(cloud)
    keep = rng.choice(N, size=max(1, N // factor), replace=False)
    return cloud[keep]


def transform_cloud(cloud: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply rigid transform (R, t) to every point in a cloud.

    Returns cloud @ R.T + t  (equivalent to (R @ cloud.T).T + t).
    """
    return cloud @ R.T + t
