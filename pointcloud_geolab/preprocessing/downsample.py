"""Voxel grid downsampling."""

from __future__ import annotations

import numpy as np


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Downsample points by replacing each occupied voxel with its centroid."""

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if voxel_size <= 0:
        return pts.copy()
    if len(pts) == 0:
        return pts.copy()

    voxel_indices = np.floor(pts / voxel_size).astype(np.int64)
    _, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
    sums = np.zeros((inverse.max() + 1, 3), dtype=float)
    counts = np.bincount(inverse)
    np.add.at(sums, inverse, pts)
    return sums / counts[:, None]
