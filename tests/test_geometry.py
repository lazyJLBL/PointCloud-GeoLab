from __future__ import annotations

import numpy as np

from pointcloud_geolab.geometry.bounding_box import compute_aabb, compute_obb
from pointcloud_geolab.geometry.distance import point_to_line_distances, point_to_plane_distances
from pointcloud_geolab.geometry.pca import pca_analysis
from pointcloud_geolab.utils.transform import apply_transform, rotation_matrix_from_euler


def test_aabb_extent_and_center() -> None:
    points = np.asarray([[-1.0, 0.0, 2.0], [3.0, 4.0, -2.0], [1.0, 2.0, 0.0]])

    aabb = compute_aabb(points)

    assert np.allclose(aabb.min_bound, [-1.0, 0.0, -2.0])
    assert np.allclose(aabb.max_bound, [3.0, 4.0, 2.0])
    assert np.allclose(aabb.center, [1.0, 2.0, 0.0])
    assert np.allclose(aabb.extent, [4.0, 4.0, 4.0])
    assert aabb.corners.shape == (8, 3)


def test_pca_and_obb_shapes() -> None:
    rng = np.random.default_rng(6)
    local = rng.uniform([-1.0, -0.2, -0.1], [1.0, 0.2, 0.1], size=(400, 3))
    rotation = rotation_matrix_from_euler(0.3, 0.1, -0.4)
    points = apply_transform(local, rotation, np.asarray([0.2, 0.3, -0.1]))

    pca = pca_analysis(points)
    obb = compute_obb(points)

    assert pca.eigenvalues[0] > pca.eigenvalues[1] > pca.eigenvalues[2]
    assert np.isclose(abs(np.linalg.det(pca.eigenvectors)), 1.0, atol=1e-8)
    assert obb.corners.shape == (8, 3)
    assert np.all(obb.extent > 0)


def test_distance_functions() -> None:
    points = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -2.0], [2.0, 0.0, 0.0]])
    plane = np.asarray([0.0, 0.0, 1.0, 0.0])

    assert np.allclose(point_to_plane_distances(points, plane), [1.0, 2.0, 0.0])
    assert np.allclose(
        point_to_line_distances(points, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        [0.0, 0.0, 2.0],
    )

