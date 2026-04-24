from __future__ import annotations

import numpy as np

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

