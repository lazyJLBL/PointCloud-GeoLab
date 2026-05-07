"""A small custom KD-Tree for nearest-neighbor search in arbitrary dimension."""

from __future__ import annotations

import heapq
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class KDNode:
    point: np.ndarray
    index: int
    axis: int
    left: "KDNode | None" = None
    right: "KDNode | None" = None


class KDTree:
    """Balanced KD-Tree supporting NN, KNN, radius, and batch queries.

    The implementation is intentionally explicit rather than a wrapper around
    SciPy or sklearn: it exposes median splitting, branch pruning, and stable
    tie-breaking while still handling practical edge cases such as empty point
    sets, repeated points, and non-3D feature spaces.
    """

    def __init__(self, points: np.ndarray):
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2:
            raise ValueError("points must have shape (N, D)")
        if pts.shape[1] == 0:
            raise ValueError("points must have at least one dimension")
        self.points = pts.copy()
        self.dim = int(pts.shape[1])
        indices = np.arange(len(self.points), dtype=int)
        self.root = self._build(indices, depth=0)

    def _build(self, indices: np.ndarray, depth: int) -> KDNode | None:
        if len(indices) == 0:
            return None
        axis = depth % self.dim
        order = np.argsort(self.points[indices, axis], kind="mergesort")
        sorted_indices = indices[order]
        median = len(sorted_indices) // 2
        node_index = int(sorted_indices[median])
        return KDNode(
            point=self.points[node_index],
            index=node_index,
            axis=axis,
            left=self._build(sorted_indices[:median], depth + 1),
            right=self._build(sorted_indices[median + 1 :], depth + 1),
        )

    def nearest_neighbor(self, query_point: np.ndarray) -> tuple[int, float]:
        """Return ``(index, distance)`` of the nearest point."""

        if self.root is None:
            raise ValueError("cannot query an empty KDTree")
        query = self._validate_query(query_point)
        best_index, best_dist_sq = self._nearest(self.root, query, -1, float("inf"))
        return best_index, float(np.sqrt(best_dist_sq))

    def _nearest(
        self,
        node: KDNode | None,
        query: np.ndarray,
        best_index: int,
        best_dist_sq: float,
    ) -> tuple[int, float]:
        if node is None:
            return best_index, best_dist_sq

        dist_sq = float(np.sum((query - node.point) ** 2))
        if dist_sq < best_dist_sq or (
            np.isclose(dist_sq, best_dist_sq) and (best_index < 0 or node.index < best_index)
        ):
            best_index, best_dist_sq = node.index, dist_sq

        axis = node.axis
        diff = float(query[axis] - node.point[axis])
        near = node.left if diff <= 0 else node.right
        far = node.right if diff <= 0 else node.left

        best_index, best_dist_sq = self._nearest(near, query, best_index, best_dist_sq)
        if diff * diff <= best_dist_sq:
            best_index, best_dist_sq = self._nearest(far, query, best_index, best_dist_sq)
        return best_index, best_dist_sq

    def knn_search(self, query_point: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return the ``k`` nearest neighbors sorted by distance."""

        if k <= 0:
            return []
        if self.root is None:
            return []
        query = self._validate_query(query_point)
        heap: list[tuple[float, int]] = []
        self._knn(self.root, query, min(k, len(self.points)), heap)
        result = [(-neg_idx, float(np.sqrt(-neg_dist_sq))) for neg_dist_sq, neg_idx in heap]
        result.sort(key=lambda item: (item[1], item[0]))
        return result

    def _knn(
        self,
        node: KDNode | None,
        query: np.ndarray,
        k: int,
        heap: list[tuple[float, int]],
    ) -> None:
        if node is None:
            return
        dist_sq = float(np.sum((query - node.point) ** 2))
        entry = (-dist_sq, -node.index)
        if len(heap) < k:
            heapq.heappush(heap, entry)
        else:
            worst_dist_sq = -heap[0][0]
            worst_index = -heap[0][1]
            if dist_sq < worst_dist_sq or (
                np.isclose(dist_sq, worst_dist_sq) and node.index < worst_index
            ):
                heapq.heapreplace(heap, entry)

        axis = node.axis
        diff = float(query[axis] - node.point[axis])
        near = node.left if diff <= 0 else node.right
        far = node.right if diff <= 0 else node.left
        self._knn(near, query, k, heap)

        worst_dist_sq = -heap[0][0] if len(heap) == k else float("inf")
        if diff * diff <= worst_dist_sq:
            self._knn(far, query, k, heap)

    def radius_search(self, query_point: np.ndarray, radius: float) -> list[tuple[int, float]]:
        """Return all points inside ``radius`` sorted by distance."""

        if radius < 0:
            raise ValueError("radius must be non-negative")
        if self.root is None:
            return []
        query = self._validate_query(query_point)
        radius_sq = radius * radius
        result: list[tuple[int, float]] = []
        self._radius(self.root, query, radius_sq, result)
        result.sort(key=lambda item: (item[1], item[0]))
        return result

    def _radius(
        self,
        node: KDNode | None,
        query: np.ndarray,
        radius_sq: float,
        result: list[tuple[int, float]],
    ) -> None:
        if node is None:
            return
        dist_sq = float(np.sum((query - node.point) ** 2))
        if dist_sq <= radius_sq:
            result.append((node.index, float(np.sqrt(dist_sq))))

        axis = node.axis
        diff = float(query[axis] - node.point[axis])
        if diff <= 0:
            self._radius(node.left, query, radius_sq, result)
            if diff * diff <= radius_sq:
                self._radius(node.right, query, radius_sq, result)
        else:
            self._radius(node.right, query, radius_sq, result)
            if diff * diff <= radius_sq:
                self._radius(node.left, query, radius_sq, result)

    def batch_nearest(
        self,
        query_points: np.ndarray,
        parallel: bool = False,
        workers: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return nearest-neighbor indices and distances for many queries."""

        queries = self._validate_queries(query_points)
        if len(queries) == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=float)
        if self.root is None:
            return (
                np.full(len(queries), -1, dtype=int),
                np.full(len(queries), np.inf, dtype=float),
            )

        results = self._map_queries(queries, self.nearest_neighbor, parallel, workers)
        indices = np.asarray([idx for idx, _ in results], dtype=int)
        distances = np.asarray([distance for _, distance in results], dtype=float)
        return indices, distances

    def batch_knn_search(
        self,
        query_points: np.ndarray,
        k: int,
        parallel: bool = False,
        workers: int | None = None,
    ) -> tuple[list[list[int]], list[list[float]]]:
        """Return KNN indices and distances for many queries.

        Lists are returned instead of padded arrays because ``k`` can exceed the
        number of stored points, and empty trees naturally produce empty rows.
        """

        queries = self._validate_queries(query_points)
        if len(queries) == 0:
            return [], []
        results = self._map_queries(
            queries,
            lambda query: self.knn_search(query, k),
            parallel,
            workers,
        )
        indices = [[idx for idx, _ in row] for row in results]
        distances = [[distance for _, distance in row] for row in results]
        return indices, distances

    def batch_radius_search(
        self,
        query_points: np.ndarray,
        radius: float,
        parallel: bool = False,
        workers: int | None = None,
    ) -> tuple[list[list[int]], list[list[float]]]:
        """Return radius-neighborhood indices and distances for many queries."""

        queries = self._validate_queries(query_points)
        if len(queries) == 0:
            return [], []
        results = self._map_queries(
            queries,
            lambda query: self.radius_search(query, radius),
            parallel,
            workers,
        )
        indices = [[idx for idx, _ in row] for row in results]
        distances = [[distance for _, distance in row] for row in results]
        return indices, distances

    def _validate_query(self, query_point: np.ndarray) -> np.ndarray:
        query = np.asarray(query_point, dtype=float).reshape(-1)
        if query.shape[0] != self.dim:
            raise ValueError(f"query point must have shape ({self.dim},)")
        return query

    def _validate_queries(self, query_points: np.ndarray) -> np.ndarray:
        queries = np.asarray(query_points, dtype=float)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError(f"query_points must have shape (M, {self.dim})")
        return queries

    def _map_queries(
        self,
        queries: np.ndarray,
        fn,
        parallel: bool,
        workers: int | None,
    ):
        if not parallel or len(queries) <= 1:
            return [fn(query) for query in queries]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(fn, queries))
