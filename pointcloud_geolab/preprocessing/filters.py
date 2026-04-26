"""Additional point cloud preprocessing filters."""

from __future__ import annotations

import numpy as np


def normalize_point_cloud(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Center a point cloud and scale it to unit radius.

    Returns ``(normalized_points, center, scale)`` so callers can invert the
    normalization if needed.
    """

    pts = _ensure_points(points)
    if len(pts) == 0:
        return pts.copy(), np.zeros(3), 1.0
    center = pts.mean(axis=0)
    centered = pts - center
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if scale <= 0:
        scale = 1.0
    return centered / scale, center, scale


def crop_by_aabb(
    points: np.ndarray,
    min_bound: np.ndarray | list[float],
    max_bound: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Crop points inside an axis-aligned bounding box."""

    pts = _ensure_points(points)
    min_b = np.asarray(min_bound, dtype=float).reshape(3)
    max_b = np.asarray(max_bound, dtype=float).reshape(3)
    mask = np.all((pts >= min_b) & (pts <= max_b), axis=1)
    return pts[mask], np.flatnonzero(mask)


def random_sample(
    points: np.ndarray,
    count: int,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample up to ``count`` points without replacement."""

    pts = _ensure_points(points)
    if count <= 0:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=int)
    if count >= len(pts):
        indices = np.arange(len(pts), dtype=int)
        return pts.copy(), indices
    rng = np.random.default_rng(random_state)
    indices = np.sort(rng.choice(len(pts), size=count, replace=False))
    return pts[indices], indices


def farthest_point_sample(
    points: np.ndarray,
    count: int,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample points by iterative farthest point sampling."""

    pts = _ensure_points(points)
    if count <= 0:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=int)
    if count >= len(pts):
        indices = np.arange(len(pts), dtype=int)
        return pts.copy(), indices

    rng = np.random.default_rng(random_state)
    selected = np.empty(count, dtype=int)
    selected[0] = int(rng.integers(len(pts)))
    min_dist_sq = np.sum((pts - pts[selected[0]]) ** 2, axis=1)
    for i in range(1, count):
        selected[i] = int(np.argmax(min_dist_sq))
        dist_sq = np.sum((pts - pts[selected[i]]) ** 2, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq)
    return pts[selected], selected


def _ensure_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("points must have shape (N, 3) or wider")
    return pts[:, :3].copy()
