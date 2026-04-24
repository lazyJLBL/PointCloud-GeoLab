"""AABB and PCA-based OBB computation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pca import pca_analysis


@dataclass(frozen=True, slots=True)
class AABB:
    min_bound: np.ndarray
    max_bound: np.ndarray
    center: np.ndarray
    extent: np.ndarray
    corners: np.ndarray


@dataclass(frozen=True, slots=True)
class OBB:
    center: np.ndarray
    rotation: np.ndarray
    extent: np.ndarray
    corners: np.ndarray


def compute_aabb(points: np.ndarray) -> AABB:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        raise ValueError("points must have non-empty shape (N, 3)")
    min_bound = pts.min(axis=0)
    max_bound = pts.max(axis=0)
    center = (min_bound + max_bound) / 2.0
    extent = max_bound - min_bound
    corners = _corners_from_bounds(min_bound, max_bound)
    return AABB(min_bound=min_bound, max_bound=max_bound, center=center, extent=extent, corners=corners)


def compute_obb(points: np.ndarray) -> OBB:
    """Compute an oriented bounding box from PCA axes."""

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        raise ValueError("points must have non-empty shape (N, 3)")
    pca = pca_analysis(pts)
    axes = pca.eigenvectors
    local = (pts - pca.center) @ axes
    local_min = local.min(axis=0)
    local_max = local.max(axis=0)
    extent = local_max - local_min
    local_center = (local_min + local_max) / 2.0
    center = pca.center + local_center @ axes.T
    local_corners = _corners_from_bounds(local_min, local_max)
    corners = pca.center + local_corners @ axes.T
    return OBB(center=center, rotation=axes, extent=extent, corners=corners)


def _corners_from_bounds(min_bound: np.ndarray, max_bound: np.ndarray) -> np.ndarray:
    x0, y0, z0 = min_bound
    x1, y1, z1 = max_bound
    return np.asarray(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x0, y1, z0],
            [x1, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x0, y1, z1],
            [x1, y1, z1],
        ],
        dtype=float,
    )

