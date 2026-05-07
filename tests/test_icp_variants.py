from __future__ import annotations

import numpy as np

from pointcloud_geolab.registration import (
    multiscale_icp,
    point_to_plane_icp,
    point_to_point_icp,
    registration_success,
    robust_icp,
    rotation_error_deg,
    translation_error,
)
from pointcloud_geolab.utils.transform import apply_transform, rotation_matrix_from_euler


def test_multiscale_icp_recovers_small_transform() -> None:
    rng = np.random.default_rng(120)
    target = rng.normal(scale=0.4, size=(180, 3))
    rotation = rotation_matrix_from_euler(0.02, -0.01, 0.015)
    translation = np.asarray([0.02, -0.01, 0.015])
    source = apply_transform(target, rotation, translation)

    result = multiscale_icp(source, target, voxel_sizes=[0.3, 0.15], max_iterations_per_level=25)

    assert len(result.diagnostics) == 2
    assert result.final_rmse < 1e-3


def test_robust_icp_is_less_sensitive_to_source_outliers() -> None:
    rng = np.random.default_rng(121)
    target = rng.normal(scale=0.35, size=(160, 3))
    source = target + np.asarray([0.03, -0.02, 0.01])
    source_with_outliers = np.vstack([source, rng.uniform(2.0, 3.0, size=(40, 3))])

    plain = point_to_point_icp(source_with_outliers, target, max_iterations=40)
    robust = robust_icp(
        source_with_outliers,
        target,
        robust_kernel="huber",
        trim_ratio=0.75,
        max_iterations=40,
        max_correspondence_distance=0.8,
    )

    assert robust.final_rmse <= plain.final_rmse
    assert robust.diagnostics["robust_kernel"] == "huber"


def test_point_to_plane_icp_runs_with_normals_and_reports_condition() -> None:
    rng = np.random.default_rng(122)
    xy = rng.uniform(-1.0, 1.0, size=(120, 2))
    target = np.column_stack([xy, 0.1 * xy[:, 0] + 0.2])
    source = target + np.asarray([0.0, 0.0, 0.03])
    normals = np.tile(np.asarray([-0.1, 0.0, 1.0]), (len(target), 1))
    normals /= np.linalg.norm(normals, axis=1)[:, None]

    result = point_to_plane_icp(
        source,
        target,
        target_normals=normals,
        max_iterations=20,
        max_correspondence_distance=0.2,
    )

    assert result.final_rmse < 0.04
    assert "condition_number" in result.diagnostics


def test_registration_metrics() -> None:
    rotation = rotation_matrix_from_euler(0.01, 0.0, 0.0)
    assert rotation_error_deg(rotation, rotation) == 0.0
    assert translation_error([0, 0, 0], [0, 0, 0.01]) == 0.01
    assert registration_success(1.0, 0.02, 0.01)
