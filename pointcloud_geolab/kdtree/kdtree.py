"""A small custom KD-Tree for 3D nearest-neighbor search."""

from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np


@dataclass(slots=True)
class KDNode:
    point: np.ndarray
    index: int
    axis: int
    left: "KDNode | None" = None
    right: "KDNode | None" = None


class KDTree:
    """Balanced KD-Tree supporting NN, KNN, and radius queries."""

    def __init__(self, points: np.ndarray):
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        self.points = pts.copy()
        self.dim = 3
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
        if dist_sq < best_dist_sq or (np.isclose(dist_sq, best_dist_sq) and node.index < best_index):
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

    def _knn(self, node: KDNode | None, query: np.ndarray, k: int, heap: list[tuple[float, int]]) -> None:
        if node is None:
            return
        dist_sq = float(np.sum((query - node.point) ** 2))
        entry = (-dist_sq, -node.index)
        if len(heap) < k:
            heapq.heappush(heap, entry)
        else:
            worst_dist_sq = -heap[0][0]
            worst_index = -heap[0][1]
            if dist_sq < worst_dist_sq or (np.isclose(dist_sq, worst_dist_sq) and node.index < worst_index):
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

    def _validate_query(self, query_point: np.ndarray) -> np.ndarray:
        query = np.asarray(query_point, dtype=float).reshape(-1)
        if query.shape[0] != self.dim:
            raise ValueError("query point must have shape (3,)")
        return query
