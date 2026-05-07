"""Outlier removal based on the custom KD-Tree."""

from __future__ import annotations

import numpy as np

from pointcloud_geolab.kdtree.kdtree import KDTree


def remove_statistical_outliers(
    points: np.ndarray,
    nb_neighbors: int = 16,
    std_ratio: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove points whose mean KNN distance is above a global threshold."""

    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return pts.copy(), np.asarray([], dtype=int)
    if nb_neighbors <= 0 or len(pts) <= 2:
        return pts.copy(), np.arange(len(pts), dtype=int)

    tree = KDTree(pts)
    k = min(nb_neighbors + 1, len(pts))
    mean_distances = np.empty(len(pts), dtype=float)
    for i, point in enumerate(pts):
        neighbors = tree.knn_search(point, k=k)
        distances = np.asarray([distance for idx, distance in neighbors if idx != i], dtype=float)
        if len(distances) == 0:
            mean_distances[i] = 0.0
        else:
            mean_distances[i] = float(distances.mean())
    threshold = float(mean_distances.mean() + std_ratio * mean_distances.std())
    inlier_mask = mean_distances <= threshold
    inlier_indices = np.flatnonzero(inlier_mask)
    return pts[inlier_indices], inlier_indices


def remove_radius_outliers(
    points: np.ndarray,
    radius: float = 0.1,
    min_neighbors: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep points with at least ``min_neighbors`` neighbors inside ``radius``."""

    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return pts.copy(), np.asarray([], dtype=int)
    if radius <= 0:
        return pts.copy(), np.arange(len(pts), dtype=int)

    tree = KDTree(pts)
    inliers = []
    for i, point in enumerate(pts):
        neighbors = tree.radius_search(point, radius=radius)
        count = sum(1 for idx, _ in neighbors if idx != i)
        if count >= min_neighbors:
            inliers.append(i)
    inlier_indices = np.asarray(inliers, dtype=int)
    return pts[inlier_indices], inlier_indices
