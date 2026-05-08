"""Voxel hash grid for local point cloud neighborhood queries."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np


@dataclass(slots=True)
class VoxelHashGrid:
    """Hash grid spatial index for fixed-radius and axis-aligned box queries."""

    voxel_size: float
    points: np.ndarray = field(init=False, repr=False)
    dim: int = field(init=False)
    buckets: dict[tuple[int, ...], list[int]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.voxel_size <= 0:
            raise ValueError("voxel_size must be positive")

    @classmethod
    def build(cls, points: np.ndarray, voxel_size: float) -> "VoxelHashGrid":
        """Build a grid from ``points``."""

        grid = cls(voxel_size=voxel_size)
        grid._set_points(points)
        return grid

    def _set_points(self, points: np.ndarray) -> None:
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2:
            raise ValueError("points must have shape (N, D)")
        if pts.shape[1] == 0:
            raise ValueError("points must have at least one dimension")
        self.points = pts.copy()
        self.dim = int(pts.shape[1])
        self.buckets = {}
        for index, point in enumerate(self.points):
            key = self.voxel_key(point)
            self.buckets.setdefault(key, []).append(index)

    def voxel_key(self, point: np.ndarray) -> tuple[int, ...]:
        """Return the integer voxel key for a point."""

        query = self._validate_point(point)
        return tuple(np.floor(query / self.voxel_size).astype(int).tolist())

    def radius_search(self, query_point: np.ndarray, radius: float) -> list[tuple[int, float]]:
        """Return all indexed points inside ``radius`` sorted by distance."""

        if radius < 0:
            raise ValueError("radius must be non-negative")
        if len(self.points) == 0:
            return []
        query = self._validate_point(query_point)
        center_key = np.asarray(self.voxel_key(query), dtype=int)
        span = int(np.ceil(radius / self.voxel_size))
        candidates: list[int] = []
        for offset in np.ndindex(*([2 * span + 1] * self.dim)):
            delta = np.asarray(offset, dtype=int) - span
            candidates.extend(self.buckets.get(tuple((center_key + delta).tolist()), []))

        result = []
        radius_sq = radius * radius
        for index in candidates:
            distance_sq = float(np.sum((self.points[index] - query) ** 2))
            if distance_sq <= radius_sq:
                result.append((index, float(np.sqrt(distance_sq))))
        result.sort(key=lambda item: (item[1], item[0]))
        return result

    def nearest_neighbor(
        self,
        query_point: np.ndarray,
        max_radius: float | None = None,
    ) -> tuple[int, float]:
        """Return the nearest indexed point to ``query_point``.

        The search expands voxel shells around the query cell until the current
        best distance is smaller than the next shell boundary. ``max_radius`` is
        useful when the grid is used as a bounded local correspondence index.
        """

        if len(self.points) == 0:
            raise ValueError("cannot query an empty VoxelHashGrid")
        query = self._validate_point(query_point)
        if max_radius is not None and max_radius < 0:
            raise ValueError("max_radius must be non-negative")

        best_index = -1
        best_distance_sq = float("inf")
        radius_sq = None if max_radius is None else max_radius * max_radius
        ordered_buckets = sorted(
            self.buckets,
            key=lambda key: self._bucket_min_distance_sq(key, query),
        )
        for key in ordered_buckets:
            bucket_min = self._bucket_min_distance_sq(key, query)
            if bucket_min > best_distance_sq:
                break
            if radius_sq is not None and bucket_min > radius_sq:
                break
            for index in self.buckets[key]:
                distance_sq = float(np.sum((self.points[index] - query) ** 2))
                if radius_sq is not None and distance_sq > radius_sq:
                    continue
                if distance_sq < best_distance_sq or (
                    np.isclose(distance_sq, best_distance_sq)
                    and (best_index < 0 or index < best_index)
                ):
                    best_index = index
                    best_distance_sq = distance_sq

        if best_index < 0:
            raise ValueError("no point found within max_radius")
        return best_index, float(np.sqrt(best_distance_sq))

    def knn_search(self, query_point: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return the ``k`` nearest indexed points sorted by distance."""

        if k <= 0:
            raise ValueError("k must be positive")
        if len(self.points) == 0:
            return []
        query = self._validate_point(query_point)
        target_count = min(k, len(self.points))
        candidates: dict[int, float] = {}
        ordered_buckets = sorted(
            self.buckets,
            key=lambda key: self._bucket_min_distance_sq(key, query),
        )
        kth_distance = float("inf")
        for key in ordered_buckets:
            bucket_min = float(np.sqrt(self._bucket_min_distance_sq(key, query)))
            if len(candidates) >= target_count and bucket_min > kth_distance:
                break
            for index in self.buckets[key]:
                candidates[index] = float(np.linalg.norm(self.points[index] - query))
            if len(candidates) >= target_count:
                ordered = sorted(candidates.items(), key=lambda item: (item[1], item[0]))
                kth_distance = ordered[target_count - 1][1]
        return sorted(candidates.items(), key=lambda item: (item[1], item[0]))[:target_count]

    def box_query(self, min_bound: np.ndarray, max_bound: np.ndarray) -> list[int]:
        """Return point indices inside an axis-aligned box."""

        if len(self.points) == 0:
            return []
        lo = self._validate_point(min_bound)
        hi = self._validate_point(max_bound)
        if np.any(hi < lo):
            raise ValueError("max_bound must be greater than or equal to min_bound")
        min_key = np.floor(lo / self.voxel_size).astype(int)
        max_key = np.floor(hi / self.voxel_size).astype(int)
        ranges = [range(int(a), int(b) + 1) for a, b in zip(min_key, max_key, strict=True)]
        result = []
        for key in product(*ranges):
            for index in self.buckets.get(tuple(key), []):
                point = self.points[index]
                if np.all(point >= lo) and np.all(point <= hi):
                    result.append(index)
        return sorted(result)

    def voxel_downsample(self) -> tuple[np.ndarray, np.ndarray]:
        """Return voxel centroids and representative source indices."""

        if len(self.points) == 0:
            return np.empty((0, self.dim), dtype=float), np.empty(0, dtype=int)
        centroids = []
        representatives = []
        for indices in self.buckets.values():
            bucket_points = self.points[indices]
            centroids.append(bucket_points.mean(axis=0))
            representatives.append(int(indices[0]))
        return np.asarray(centroids, dtype=float), np.asarray(representatives, dtype=int)

    def _validate_point(self, point: np.ndarray) -> np.ndarray:
        query = np.asarray(point, dtype=float).reshape(-1)
        if query.shape[0] != self.dim:
            raise ValueError(f"point must have shape ({self.dim},)")
        return query

    def _max_search_span(self, max_radius: float | None) -> int:
        if max_radius is not None:
            return int(np.ceil(max_radius / self.voxel_size))
        if len(self.buckets) == 0:
            return 0
        keys = np.asarray(list(self.buckets), dtype=int)
        extent = keys.max(axis=0) - keys.min(axis=0)
        return int(np.max(extent)) + 1

    def _shell_keys(self, center_key: np.ndarray, span: int) -> list[tuple[int, ...]]:
        if span == 0:
            return [tuple(center_key.tolist())]
        keys = []
        for offset in np.ndindex(*([2 * span + 1] * self.dim)):
            delta = np.asarray(offset, dtype=int) - span
            if np.max(np.abs(delta)) == span:
                keys.append(tuple((center_key + delta).tolist()))
        return keys

    def _bucket_min_distance_sq(self, key: tuple[int, ...], query: np.ndarray) -> float:
        lower = np.asarray(key, dtype=float) * self.voxel_size
        upper = lower + self.voxel_size
        delta = np.maximum(0.0, np.maximum(lower - query, query - upper))
        return float(delta @ delta)
