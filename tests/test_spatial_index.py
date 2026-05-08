from __future__ import annotations

import numpy as np

from pointcloud_geolab.spatial import VoxelHashGrid


def test_voxel_hash_radius_search_matches_brute_force() -> None:
    rng = np.random.default_rng(110)
    points = rng.random((100, 3))
    query = rng.random(3)
    radius = 0.22
    grid = VoxelHashGrid.build(points, voxel_size=0.15)

    result = grid.radius_search(query, radius)
    distances = np.linalg.norm(points - query, axis=1)
    expected = sorted(
        [(idx, float(distance)) for idx, distance in enumerate(distances) if distance <= radius],
        key=lambda item: (item[1], item[0]),
    )

    assert [idx for idx, _ in result] == [idx for idx, _ in expected]
    assert np.allclose([dist for _, dist in result], [dist for _, dist in expected])


def test_voxel_hash_box_query_and_downsample() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.05, 0.05, 0.05],
            [1.0, 1.0, 1.0],
        ]
    )
    grid = VoxelHashGrid.build(points, voxel_size=0.2)

    assert grid.box_query([-0.1, -0.1, -0.1], [0.2, 0.2, 0.2]) == [0, 1]
    centroids, representatives = grid.voxel_downsample()
    assert len(centroids) == 2
    assert representatives.shape == (2,)


def test_voxel_hash_nearest_and_knn_match_brute_force() -> None:
    rng = np.random.default_rng(111)
    points = rng.normal(size=(120, 3))
    query = rng.normal(size=3)
    grid = VoxelHashGrid.build(points, voxel_size=0.25)

    nearest_index, nearest_distance = grid.nearest_neighbor(query)
    distances = np.linalg.norm(points - query, axis=1)
    expected_order = np.argsort(distances, kind="mergesort")

    assert nearest_index == int(expected_order[0])
    assert np.isclose(nearest_distance, distances[expected_order[0]])

    knn = grid.knn_search(query, k=7)
    assert [index for index, _ in knn] == [int(index) for index in expected_order[:7]]
    assert np.allclose([distance for _, distance in knn], distances[expected_order[:7]])


def test_voxel_hash_nearest_respects_max_radius() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    grid = VoxelHashGrid.build(points, voxel_size=0.5)

    assert grid.nearest_neighbor([0.1, 0.0, 0.0], max_radius=0.2)[0] == 0

    try:
        grid.nearest_neighbor([1.0, 0.0, 0.0], max_radius=0.1)
    except ValueError as exc:
        assert "max_radius" in str(exc)
    else:  # pragma: no cover - failure branch
        raise AssertionError("expected bounded nearest-neighbor query to fail")
