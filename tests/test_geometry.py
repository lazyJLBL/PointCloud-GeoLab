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


def test_single_point_aabb_and_obb() -> None:
    points = np.asarray([[1.0, -2.0, 3.0]])

    aabb = compute_aabb(points)
    obb = compute_obb(points)

    assert np.allclose(aabb.center, points[0])
    assert np.allclose(aabb.extent, np.zeros(3))
    assert np.allclose(obb.center, points[0])
    assert np.allclose(obb.extent, np.zeros(3))
    assert np.allclose(obb.rotation, np.eye(3))


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


def test_collinear_and_coplanar_geometry() -> None:
    t = np.linspace(-1.0, 1.0, 20)
    line_points = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    plane_xy = (
        np.asarray(np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))).reshape(2, -1).T
    )
    plane_points = np.column_stack([plane_xy, np.zeros(len(plane_xy))])

    line_pca = pca_analysis(line_points)
    plane_pca = pca_analysis(plane_points)
    line_obb = compute_obb(line_points)
    plane_obb = compute_obb(plane_points)

    assert line_pca.eigenvalues[0] > 0
    assert np.allclose(line_pca.eigenvalues[1:], 0.0)
    assert np.count_nonzero(line_obb.extent > 1e-10) == 1
    assert np.isclose(plane_pca.eigenvalues[2], 0.0)
    assert np.count_nonzero(plane_obb.extent > 1e-10) == 2


def test_cube_aabb_extent() -> None:
    points = np.asarray([[x, y, z] for x in (-1.0, 1.0) for y in (-2.0, 2.0) for z in (-3.0, 3.0)])

    aabb = compute_aabb(points)

    assert np.allclose(aabb.extent, [2.0, 4.0, 6.0])


def test_distance_functions() -> None:
    points = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -2.0], [2.0, 0.0, 0.0]])
    plane = np.asarray([0.0, 0.0, 1.0, 0.0])

    assert np.allclose(point_to_plane_distances(points, plane), [1.0, 2.0, 0.0])
    assert np.allclose(
        point_to_line_distances(points, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        [0.0, 0.0, 2.0],
    )
