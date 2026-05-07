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
