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
