from __future__ import annotations

import numpy as np
import pytest

from pointcloud_geolab.kdtree.kdtree import KDTree


def test_empty_tree_query_behavior() -> None:
    tree = KDTree(np.empty((0, 3)))

    with pytest.raises(ValueError):
        tree.nearest_neighbor(np.zeros(3))
    assert tree.knn_search(np.zeros(3), k=3) == []
    assert tree.radius_search(np.zeros(3), radius=1.0) == []


def test_single_point_and_exact_query() -> None:
    points = np.asarray([[1.0, 2.0, 3.0]])
    tree = KDTree(points)

    assert tree.nearest_neighbor(points[0]) == (0, 0.0)
    assert tree.knn_search(points[0], k=5) == [(0, 0.0)]
    assert tree.radius_search(points[0], radius=0.0) == [(0, 0.0)]


def test_nearest_neighbor_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    points = rng.random((200, 3))
    query = np.asarray([0.33, 0.41, 0.77])
    tree = KDTree(points)

    idx, distance = tree.nearest_neighbor(query)
    distances = np.linalg.norm(points - query, axis=1)
    brute_idx = int(np.argmin(distances))

    assert idx == brute_idx
    assert np.isclose(distance, distances[brute_idx])


def test_knn_matches_brute_force() -> None:
    rng = np.random.default_rng(1)
    points = rng.random((150, 3))
    query = np.asarray([0.21, 0.76, 0.18])
    tree = KDTree(points)

    knn = tree.knn_search(query, k=8)
    distances = np.linalg.norm(points - query, axis=1)
    expected = np.argsort(distances)[:8].tolist()

    assert [idx for idx, _ in knn] == expected
    assert np.allclose([dist for _, dist in knn], distances[expected])


def test_knn_with_k_larger_than_point_count() -> None:
    rng = np.random.default_rng(10)
    points = rng.random((5, 3))
    query = rng.random(3)
    tree = KDTree(points)

    result = tree.knn_search(query, k=20)

    assert len(result) == len(points)
    assert [idx for idx, _ in result] == np.argsort(np.linalg.norm(points - query, axis=1)).tolist()


def test_radius_search_matches_brute_force() -> None:
    rng = np.random.default_rng(2)
    points = rng.random((100, 3))
    query = np.asarray([0.5, 0.5, 0.5])
    radius = 0.35
    tree = KDTree(points)

    result = tree.radius_search(query, radius=radius)
    distances = np.linalg.norm(points - query, axis=1)
    expected = sorted(
        [(idx, float(dist)) for idx, dist in enumerate(distances) if dist <= radius],
        key=lambda item: (item[1], item[0]),
    )

    assert [idx for idx, _ in result] == [idx for idx, _ in expected]
    assert np.allclose([dist for _, dist in result], [dist for _, dist in expected])


def test_radius_zero_and_large_radius() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    tree = KDTree(points)

    assert tree.radius_search(points[1], radius=0.0) == [(1, 0.0)]
    assert [idx for idx, _ in tree.radius_search(points[1], radius=10.0)] == [1, 0, 2]
