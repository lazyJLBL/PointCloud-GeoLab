"""Point cloud clustering algorithms based on spatial neighborhoods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pointcloud_geolab.geometry import AABB, compute_aabb
from pointcloud_geolab.kdtree import KDTree


@dataclass(slots=True)
class ClusterInfo:
    """Summary for one cluster."""

    label: int
    indices: np.ndarray
    point_count: int
    aabb: AABB


@dataclass(slots=True)
class ClusteringResult:
    """Labels and cluster summaries."""

    labels: np.ndarray
    clusters: list[ClusterInfo]
    noise_indices: np.ndarray

    @property
    def cluster_count(self) -> int:
        return len(self.clusters)


def dbscan_clustering(points: np.ndarray, eps: float, min_points: int = 10) -> ClusteringResult:
    """Cluster points with a from-scratch DBSCAN implementation.

    Labels use the standard convention: ``-1`` means noise.
    """

    pts = _ensure_points(points)
    if eps <= 0:
        raise ValueError("eps must be positive")
    if min_points <= 0:
        raise ValueError("min_points must be positive")
    if len(pts) == 0:
        return ClusteringResult(np.empty(0, dtype=int), [], np.empty(0, dtype=int))

    tree = KDTree(pts)
    labels = np.full(len(pts), -99, dtype=int)
    cluster_id = 0

    for point_id in range(len(pts)):
        if labels[point_id] != -99:
            continue
        neighbors = _radius_indices(tree, pts[point_id], eps)
        if len(neighbors) < min_points:
            labels[point_id] = -1
            continue

        labels[point_id] = cluster_id
        seeds = list(neighbors)
        cursor = 0
        while cursor < len(seeds):
            neighbor_id = seeds[cursor]
            cursor += 1
            if labels[neighbor_id] == -1:
                labels[neighbor_id] = cluster_id
            if labels[neighbor_id] != -99:
                continue
            labels[neighbor_id] = cluster_id
            neighbor_neighbors = _radius_indices(tree, pts[neighbor_id], eps)
            if len(neighbor_neighbors) >= min_points:
                for candidate in neighbor_neighbors:
                    if labels[candidate] in {-99, -1} and candidate not in seeds:
                        seeds.append(candidate)
        cluster_id += 1

    labels[labels == -99] = -1
    return _build_result(pts, labels)


def euclidean_clustering(
    points: np.ndarray,
    tolerance: float,
    min_points: int = 1,
    max_points: int | None = None,
) -> ClusteringResult:
    """Cluster connected components under a fixed Euclidean distance."""

    pts = _ensure_points(points)
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    tree = KDTree(pts)
    labels = np.full(len(pts), -1, dtype=int)
    visited = np.zeros(len(pts), dtype=bool)
    cluster_id = 0

    for start in range(len(pts)):
        if visited[start]:
            continue
        queue = [start]
        component = []
        visited[start] = True
        while queue:
            current = queue.pop()
            component.append(current)
            for neighbor in _radius_indices(tree, pts[current], tolerance):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        if len(component) >= min_points and (max_points is None or len(component) <= max_points):
            labels[component] = cluster_id
            cluster_id += 1

    return _build_result(pts, labels)


def cluster_statistics(points: np.ndarray, labels: np.ndarray) -> list[dict[str, object]]:
    """Return JSON-friendly cluster statistics."""

    pts = _ensure_points(points)
    result = _build_result(pts, np.asarray(labels, dtype=int))
    stats = []
    for cluster in result.clusters:
        stats.append(
            {
                "label": cluster.label,
                "point_count": cluster.point_count,
                "min_bound": cluster.aabb.min_bound.tolist(),
                "max_bound": cluster.aabb.max_bound.tolist(),
                "extent": cluster.aabb.extent.tolist(),
            }
        )
    return stats


def _build_result(points: np.ndarray, labels: np.ndarray) -> ClusteringResult:
    clusters = []
    for label in sorted(int(value) for value in np.unique(labels) if value >= 0):
        indices = np.flatnonzero(labels == label)
        clusters.append(
            ClusterInfo(
                label=label,
                indices=indices,
                point_count=len(indices),
                aabb=compute_aabb(points[indices]),
            )
        )
    return ClusteringResult(
        labels=labels,
        clusters=clusters,
        noise_indices=np.flatnonzero(labels == -1),
    )


def _radius_indices(tree: KDTree, point: np.ndarray, radius: float) -> list[int]:
    return [idx for idx, _ in tree.radius_search(point, radius)]


def _ensure_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    return pts
