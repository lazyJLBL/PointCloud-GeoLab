"""Distance queries for geometric primitives."""

from __future__ import annotations

import numpy as np


def point_to_plane_distances(points: np.ndarray, plane_model: np.ndarray) -> np.ndarray:
    """Compute distances from points to ``ax + by + cz + d = 0``."""

    pts = np.asarray(points, dtype=float)
    plane = np.asarray(plane_model, dtype=float).reshape(4)
    normal = plane[:3]
    denom = np.linalg.norm(normal)
    if denom == 0:
        raise ValueError("plane normal must be non-zero")
    return np.abs(pts @ normal + plane[3]) / denom


def point_to_line_distances(
    points: np.ndarray,
    line_point: np.ndarray,
    line_direction: np.ndarray,
) -> np.ndarray:
    """Compute distances from points to a 3D line."""

    pts = np.asarray(points, dtype=float)
    anchor = np.asarray(line_point, dtype=float).reshape(3)
    direction = np.asarray(line_direction, dtype=float).reshape(3)
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("line direction must be non-zero")
    unit = direction / norm
    cross = np.cross(pts - anchor, unit)
    return np.linalg.norm(cross, axis=1)
