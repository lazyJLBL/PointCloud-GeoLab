"""Region growing segmentation using spatial distance and normal similarity."""

from __future__ import annotations

import numpy as np

from pointcloud_geolab.kdtree import KDTree
from pointcloud_geolab.preprocessing import estimate_normals
from pointcloud_geolab.segmentation.clustering import ClusteringResult, _build_result


def region_growing_segmentation(
    points: np.ndarray,
    normals: np.ndarray | None = None,
    radius: float = 0.1,
    angle_threshold_degrees: float = 25.0,
    min_cluster_size: int = 5,
) -> ClusteringResult:
    """Segment smooth regions by BFS over points with similar normals."""

    pts = _ensure_points(points)
    if len(pts) == 0:
        return ClusteringResult(np.empty(0, dtype=int), [], np.empty(0, dtype=int))
    if radius <= 0:
        raise ValueError("radius must be positive")
    if normals is None:
        nrm = estimate_normals(pts, k=min(24, max(3, len(pts))))
    else:
        nrm = np.asarray(normals, dtype=float)
    if nrm.shape != pts.shape:
        raise ValueError("normals must have shape (N, 3)")

    tree = KDTree(pts)
    cos_threshold = float(np.cos(np.radians(angle_threshold_degrees)))
    visited = np.zeros(len(pts), dtype=bool)
    labels = np.full(len(pts), -1, dtype=int)
    cluster_id = 0

    for seed in range(len(pts)):
        if visited[seed]:
            continue
        queue = [seed]
        component = []
        visited[seed] = True
        while queue:
            current = queue.pop()
            component.append(current)
            current_normal = _unit(nrm[current])
            for neighbor in [idx for idx, _ in tree.radius_search(pts[current], radius)]:
                if visited[neighbor]:
                    continue
                similarity = abs(float(current_normal @ _unit(nrm[neighbor])))
                if similarity >= cos_threshold:
                    visited[neighbor] = True
                    queue.append(neighbor)
        if len(component) >= min_cluster_size:
            labels[component] = cluster_id
            cluster_id += 1

    return _build_result(pts, labels)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return np.asarray([0.0, 0.0, 1.0])
    return vector / norm


def _ensure_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    return pts
