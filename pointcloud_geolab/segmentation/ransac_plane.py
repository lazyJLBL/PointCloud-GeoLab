"""RANSAC plane fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pointcloud_geolab.geometry.distance import point_to_plane_distances


@dataclass(slots=True)
class PlaneResult:
    plane_model: np.ndarray
    inliers: np.ndarray
    outliers: np.ndarray
    inlier_ratio: float

    def __iter__(self):
        yield self.plane_model
        yield self.inliers


def ransac_plane_fitting(
    points: np.ndarray,
    threshold: float = 0.02,
    max_iterations: int = 1000,
    seed: int | None = None,
) -> PlaneResult:
    """Fit the dominant plane using RANSAC."""

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if len(pts) < 3:
        raise ValueError("at least 3 points are required")
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    rng = np.random.default_rng(seed)
    best_model: np.ndarray | None = None
    best_inliers = np.asarray([], dtype=int)

    for _ in range(max_iterations):
        sample_ids = rng.choice(len(pts), size=3, replace=False)
        model = _plane_from_three_points(pts[sample_ids])
        if model is None:
            continue
        distances = point_to_plane_distances(pts, model)
        inliers = np.flatnonzero(distances <= threshold)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = model

    if best_model is None:
        raise RuntimeError("failed to find a non-degenerate plane")

    outlier_mask = np.ones(len(pts), dtype=bool)
    outlier_mask[best_inliers] = False
    outliers = np.flatnonzero(outlier_mask)
    return PlaneResult(
        plane_model=best_model,
        inliers=best_inliers,
        outliers=outliers,
        inlier_ratio=float(len(best_inliers) / len(pts)),
    )


def _plane_from_three_points(points: np.ndarray) -> np.ndarray | None:
    p0, p1, p2 = points
    normal = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return None
    normal = normal / norm
    d = -float(normal @ p0)
    model = np.asarray([normal[0], normal[1], normal[2], d], dtype=float)
    first_nonzero = np.flatnonzero(np.abs(model[:3]) > 1e-12)
    if len(first_nonzero) > 0 and model[first_nonzero[0]] < 0:
        model *= -1.0
    return model

