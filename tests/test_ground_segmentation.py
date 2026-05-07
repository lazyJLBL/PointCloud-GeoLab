from __future__ import annotations

import numpy as np

from pointcloud_geolab.segmentation import ground_object_segmentation, remove_ground_plane


def _scene() -> np.ndarray:
    rng = np.random.default_rng(160)
    ground_xy = rng.uniform([-1.5, -1.5], [1.5, 1.5], size=(240, 2))
    ground = np.column_stack([ground_xy, rng.normal(0.0, 0.003, size=len(ground_xy))])
    obj_a = rng.normal(loc=[0.5, 0.4, 0.35], scale=0.04, size=(50, 3))
    obj_b = rng.normal(loc=[-0.6, -0.4, 0.45], scale=0.04, size=(55, 3))
    return np.vstack([ground, obj_a, obj_b])


def test_ground_removal_identifies_main_plane() -> None:
    points = _scene()

    result = remove_ground_plane(points, threshold=0.02, ground_axis="z")

    assert len(result.ground_indices) > 200
    assert result.normal_angle_degrees < 5.0


def test_ground_object_segmentation_reports_cluster_stats() -> None:
    points = _scene()

    result = ground_object_segmentation(points, eps=0.14, min_points=10)
    stats = [cluster.to_dict() for cluster in result.clusters]

    assert len(result.clusters) >= 2
    assert {"label", "point_count", "centroid", "aabb", "obb", "volume"} <= set(stats[0])
