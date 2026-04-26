from __future__ import annotations

import numpy as np

from pointcloud_geolab.segmentation import (
    cluster_statistics,
    dbscan_clustering,
    euclidean_clustering,
    region_growing_segmentation,
)


def test_dbscan_splits_two_far_clusters_and_noise() -> None:
    rng = np.random.default_rng(40)
    cluster_a = rng.normal(scale=0.02, size=(40, 3))
    cluster_b = rng.normal(scale=0.02, size=(40, 3)) + np.asarray([1.0, 0.0, 0.0])
    noise = np.asarray([[3.0, 3.0, 3.0]])
    points = np.vstack([cluster_a, cluster_b, noise])

    result = dbscan_clustering(points, eps=0.08, min_points=5)

    assert result.cluster_count == 2
    assert result.labels[-1] == -1
    assert len(result.noise_indices) == 1


def test_euclidean_clustering_output_statistics_are_stable() -> None:
    rng = np.random.default_rng(41)
    points = np.vstack(
        [
            rng.normal(scale=0.01, size=(20, 3)),
            rng.normal(scale=0.01, size=(25, 3)) + np.asarray([0.5, 0.5, 0.0]),
        ]
    )

    result = euclidean_clustering(points, tolerance=0.06, min_points=3)
    stats = cluster_statistics(points, result.labels)

    assert result.cluster_count == 2
    assert [item["point_count"] for item in stats] == [20, 25]
    assert {"label", "point_count", "min_bound", "max_bound", "extent"} <= set(stats[0])


def test_region_growing_terminates() -> None:
    rng = np.random.default_rng(42)
    points = rng.normal(scale=0.03, size=(30, 3))
    normals = np.tile(np.asarray([0.0, 0.0, 1.0]), (len(points), 1))

    result = region_growing_segmentation(
        points,
        normals=normals,
        radius=0.2,
        angle_threshold_degrees=10.0,
        min_cluster_size=3,
    )

    assert result.cluster_count == 1
    assert len(result.labels) == len(points)
