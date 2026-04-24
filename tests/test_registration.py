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

