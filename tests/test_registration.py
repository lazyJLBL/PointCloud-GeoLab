from __future__ import annotations

import numpy as np

from pointcloud_geolab.registration.icp import point_to_point_icp
from pointcloud_geolab.registration.svd_solver import estimate_rigid_transform
from pointcloud_geolab.utils.transform import apply_transform, rotation_matrix_from_euler


def test_svd_solver_recovers_known_transform() -> None:
    rng = np.random.default_rng(3)
    source = rng.normal(size=(80, 3))
    rotation = rotation_matrix_from_euler(0.2, -0.1, 0.15)
    translation = np.asarray([0.4, -0.2, 0.3])
    target = apply_transform(source, rotation, translation)

    result = estimate_rigid_transform(source, target)

    assert np.allclose(result.rotation, rotation, atol=1e-10)
    assert np.allclose(result.translation, translation, atol=1e-10)
    assert np.allclose(apply_transform(source, result.rotation, result.translation), target, atol=1e-10)


def test_icp_converges_on_translated_grid() -> None:
    axis = np.linspace(-1.0, 1.0, 5)
    target = np.asarray(np.meshgrid(axis, axis, axis), dtype=float).reshape(3, -1).T
    translation = np.asarray([0.03, -0.02, 0.01])
    source = target + translation

    result = point_to_point_icp(source, target, max_iterations=30, tolerance=1e-10)

    assert result.final_rmse < 1e-8
    assert np.allclose(result.translation, -translation, atol=1e-8)
    assert result.iterations > 0


def test_icp_identical_clouds_converge_to_identity() -> None:
    rng = np.random.default_rng(11)
    points = rng.normal(size=(100, 3))

    result = point_to_point_icp(points, points, max_iterations=10, tolerance=1e-10)

    assert result.final_rmse < 1e-10
    assert np.allclose(result.rotation, np.eye(3), atol=1e-10)
    assert np.allclose(result.translation, np.zeros(3), atol=1e-10)


def test_icp_converges_on_small_rotation() -> None:
    rng = np.random.default_rng(12)
    target = rng.normal(size=(200, 3))
    rotation = rotation_matrix_from_euler(0.02, -0.01, 0.015)
    source = apply_transform(target, rotation, np.zeros(3))

    result = point_to_point_icp(source, target, max_iterations=40, tolerance=1e-10)

    assert result.final_rmse < 1e-6
    assert np.allclose(result.rotation, rotation.T, atol=1e-5)


def test_icp_converges_on_small_rotation_and_translation() -> None:
    rng = np.random.default_rng(13)
    target = rng.normal(size=(200, 3))
    rotation = rotation_matrix_from_euler(0.015, 0.01, -0.02)
    translation = np.asarray([0.015, -0.01, 0.02])
    source = apply_transform(target, rotation, translation)

    result = point_to_point_icp(source, target, max_iterations=50, tolerance=1e-10)

    assert result.final_rmse < 1e-6
    assert np.allclose(result.rotation, rotation.T, atol=1e-5)
    assert np.allclose(result.translation, -rotation.T @ translation, atol=1e-5)


def test_icp_stops_when_correspondence_threshold_is_too_small() -> None:
    rng = np.random.default_rng(14)
    target = rng.normal(size=(80, 3))
    source = target + np.asarray([10.0, 0.0, 0.0])

    result = point_to_point_icp(
        source,
        target,
        max_iterations=20,
        tolerance=1e-10,
        max_correspondence_distance=0.001,
    )

    assert result.iterations == 0
    assert not result.converged

