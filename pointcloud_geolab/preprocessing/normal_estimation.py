"""Normal estimation with local PCA."""

from __future__ import annotations

import numpy as np

from pointcloud_geolab.kdtree.kdtree import KDTree


def estimate_normals(points: np.ndarray, k: int = 16) -> np.ndarray:
    """Estimate unoriented normals from KNN neighborhoods."""

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if len(pts) == 0:
        return np.empty((0, 3), dtype=float)
    if len(pts) < 3:
        return np.tile(np.asarray([0.0, 0.0, 1.0]), (len(pts), 1))

    tree = KDTree(pts)
    neighbor_count = min(max(k, 3), len(pts))
    normals = np.empty_like(pts)
    for i, point in enumerate(pts):
        neighbor_ids = [idx for idx, _ in tree.knn_search(point, neighbor_count)]
        neighborhood = pts[neighbor_ids]
        centered = neighborhood - neighborhood.mean(axis=0)
        covariance = centered.T @ centered / max(len(neighborhood), 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, np.argmin(eigenvalues)]
        norm = np.linalg.norm(normal)
        normals[i] = normal / norm if norm > 0 else np.asarray([0.0, 0.0, 1.0])
    return normals

