from __future__ import annotations

import numpy as np
import pytest

from pointcloud_geolab.geometry.distance import point_to_plane_distances
from pointcloud_geolab.segmentation.ransac_plane import ransac_plane_fitting


def test_ransac_plane_recovers_noisy_dominant_plane() -> None:
    rng = np.random.default_rng(4)
    xy = rng.uniform(-1.0, 1.0, size=(300, 2))
    z = 0.2 * xy[:, 0] - 0.1 * xy[:, 1] + 0.35 + rng.normal(0, 0.002, size=300)
    plane_points = np.column_stack([xy, z])
    outliers = rng.uniform(-1.0, 1.0, size=(60, 3))
    points = np.vstack([plane_points, outliers])

    result = ransac_plane_fitting(points, threshold=0.015, max_iterations=500, seed=5)

    assert len(result.inliers) >= 290
    assert result.inlier_ratio > 0.75
    assert point_to_plane_distances(plane_points, result.plane_model).mean() < 0.01


def test_ransac_raises_for_all_collinear_points() -> None:
    t = np.linspace(-1.0, 1.0, 20)
    points = np.column_stack([t, 2 * t, -t])

    with pytest.raises(RuntimeError):
        ransac_plane_fitting(points, threshold=0.01, max_iterations=50, seed=1)


def test_ransac_all_coplanar_points_are_inliers() -> None:
    rng = np.random.default_rng(15)
    xy = rng.uniform(-1.0, 1.0, size=(150, 2))
    points = np.column_stack([xy, np.full(150, 0.25)])

    result = ransac_plane_fitting(points, threshold=1e-8, max_iterations=100, seed=2)

    assert len(result.inliers) == len(points)
    assert result.inlier_ratio == 1.0


def test_ransac_handles_high_outlier_ratio() -> None:
    rng = np.random.default_rng(16)
    xy = rng.uniform(-1.0, 1.0, size=(120, 2))
    plane_points = np.column_stack([xy, 0.1 * xy[:, 0] + 0.2])
    outliers = rng.uniform(-3.0, 3.0, size=(300, 3))
    points = np.vstack([plane_points, outliers])

    result = ransac_plane_fitting(points, threshold=0.02, max_iterations=2000, seed=3)

    assert len(result.inliers) >= 110
    assert point_to_plane_distances(plane_points, result.plane_model).mean() < 0.02


def test_ransac_threshold_controls_inlier_count() -> None:
    rng = np.random.default_rng(17)
    xy = rng.uniform(-1.0, 1.0, size=(100, 2))
    z = rng.normal(0.0, 0.01, size=100)
    points = np.column_stack([xy, z])

    tight = ransac_plane_fitting(points, threshold=0.001, max_iterations=300, seed=4)
    loose = ransac_plane_fitting(points, threshold=0.05, max_iterations=300, seed=4)

    assert len(loose.inliers) > len(tight.inliers)
    assert len(loose.inliers) == len(points)
