from __future__ import annotations

import numpy as np

from pointcloud_geolab.kdtree import KDTree


def test_kdtree_high_dimensional_nearest_matches_brute_force() -> None:
    rng = np.random.default_rng(100)
    points = rng.normal(size=(120, 6))
    queries = rng.normal(size=(8, 6))
    tree = KDTree(points)

    indices, distances = tree.batch_nearest(queries)
    brute_distances = np.linalg.norm(points[None, :, :] - queries[:, None, :], axis=2)
    expected = np.argmin(brute_distances, axis=1)

    assert indices.tolist() == expected.tolist()
    assert np.allclose(distances, brute_distances[np.arange(len(queries)), expected])


def test_batch_knn_and_radius_match_single_point_queries() -> None:
    rng = np.random.default_rng(101)
    points = rng.random((80, 3))
    queries = rng.random((5, 3))
    tree = KDTree(points)

    knn_indices, knn_distances = tree.batch_knn_search(queries, k=4, parallel=True, workers=2)
    radius_indices, radius_distances = tree.batch_radius_search(queries, radius=0.25)

    for i, query in enumerate(queries):
        single_knn = tree.knn_search(query, 4)
        single_radius = tree.radius_search(query, 0.25)
        assert knn_indices[i] == [idx for idx, _ in single_knn]
        assert np.allclose(knn_distances[i], [dist for _, dist in single_knn])
        assert radius_indices[i] == [idx for idx, _ in single_radius]
        assert np.allclose(radius_distances[i], [dist for _, dist in single_radius])


def test_repeated_points_are_stable() -> None:
    points = np.asarray([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    tree = KDTree(points)

    assert tree.nearest_neighbor([0.0, 0.0]) == (0, 0.0)
    assert [idx for idx, _ in tree.radius_search([0.0, 0.0], 0.0)] == [0, 1]
