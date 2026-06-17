from __future__ import annotations

import numpy as np

from pointcloud_geolab.datasets import make_cylinder, make_plane, make_sphere
from pointcloud_geolab.geometry import compute_obb, pca_analysis
from pointcloud_geolab.geometry.primitive_fitting import ransac_fit_primitive
from pointcloud_geolab.registration import (
    generalized_icp,
    point_to_plane_icp,
    point_to_point_icp,
    robust_icp,
)
from pointcloud_geolab.utils.transform import apply_transform, rotation_matrix_from_euler


def test_icp_recovers_small_transform_with_noise() -> None:
    rng = np.random.default_rng(220)
    target = rng.normal(scale=0.45, size=(240, 3))
    rotation = rotation_matrix_from_euler(0.025, -0.02, 0.015)
    translation = np.asarray([0.03, -0.025, 0.02])
    source = apply_transform(target, rotation, translation)
    source += rng.normal(0.0, 0.004, size=source.shape)

    result = point_to_point_icp(
        source,
        target,
        max_iterations=60,
        tolerance=1e-8,
        max_correspondence_distance=0.2,
    )

    assert result.final_rmse < 0.01
    assert result.final_rmse < result.initial_rmse * 0.2
    assert result.fitness == 1.0


def test_trimmed_icp_handles_low_overlap_with_source_outliers() -> None:
    rng = np.random.default_rng(221)
    target = rng.normal(scale=0.4, size=(220, 3))
    translation = np.asarray([0.025, -0.02, 0.015])
    overlapping_source = target[:120] + translation
    source = np.vstack([overlapping_source, rng.uniform(1.8, 2.5, size=(80, 3))])

    result = robust_icp(
        source,
        target,
        robust_kernel="huber",
        trim_ratio=0.6,
        max_iterations=50,
        max_correspondence_distance=0.4,
    )

    assert result.final_rmse < 0.01
    assert result.fitness >= 0.55
    assert result.diagnostics["raw_final_rmse"] > 1.0


def test_icp_bad_initialization_fails_cleanly_with_tight_correspondence_gate() -> None:
    rng = np.random.default_rng(222)
    target = rng.normal(scale=0.4, size=(180, 3))
    source = target + np.asarray([2.0, 0.0, 0.0])

    result = point_to_point_icp(
        source,
        target,
        max_iterations=30,
        max_correspondence_distance=0.05,
    )

    assert result.iterations == 0
    assert not result.converged
    assert result.fitness == 0.0


def test_point_to_plane_icp_reports_degenerate_planar_system() -> None:
    rng = np.random.default_rng(223)
    xy = rng.uniform(-1.0, 1.0, size=(160, 2))
    target = np.column_stack([xy, np.zeros(len(xy))])
    source = target + np.asarray([0.0, 0.0, 0.04])
    normals = np.tile(np.asarray([0.0, 0.0, 1.0]), (len(target), 1))

    result = point_to_plane_icp(
        source,
        target,
        target_normals=normals,
        max_iterations=20,
        max_correspondence_distance=0.2,
    )

    assert result.iterations == 0
    assert np.isinf(result.diagnostics["condition_number"])
    assert np.isclose(result.final_rmse, result.initial_rmse)


def test_gicp_recovers_planar_translation_with_regularized_covariances() -> None:
    rng = np.random.default_rng(224)
    xy = rng.uniform(-1.0, 1.0, size=(160, 2))
    target = np.column_stack([xy, np.zeros(len(xy))])
    source = target + np.asarray([0.0, 0.0, 0.04])

    result = generalized_icp(
        source,
        target,
        max_iterations=20,
        max_correspondence_distance=0.2,
        k_neighbors=12,
        regularization=1e-3,
    )

    assert result.final_rmse < 1e-6
    assert result.final_rmse < result.initial_rmse
    assert result.diagnostics["used_correspondences"] == len(source)


def test_ransac_primitives_succeed_under_seeded_outlier_ratios() -> None:
    cases = [
        ("plane", make_plane, {"d": 0.2, "noise": 0.002}, 0.5, 0.45, 230, 240, 250),
        (
            "sphere",
            make_sphere,
            {"center": [0.1, -0.2, 0.3], "radius": 0.7, "noise": 0.002},
            0.5,
            0.45,
            231,
            241,
            251,
        ),
        (
            "cylinder",
            make_cylinder,
            {"radius": 0.45, "noise": 0.003},
            0.35,
            0.55,
            300,
            335,
            336,
        ),
    ]
    for (
        model,
        factory,
        kwargs,
        outlier_ratio,
        min_score,
        data_seed,
        outlier_seed,
        fit_seed,
    ) in cases:
        points = factory(220, random_state=data_seed, **kwargs)
        rng = np.random.default_rng(outlier_seed)
        outlier_count = int(len(points) * outlier_ratio / (1.0 - outlier_ratio))
        points = np.vstack([points, rng.uniform(-2.0, 2.0, size=(outlier_count, 3))])

        result = ransac_fit_primitive(
            points,
            model,
            threshold=0.06,
            max_iterations=1500,
            random_state=fit_seed,
        )

        assert result.score >= min_score
        assert result.residual_mean < 0.03


def test_pca_obb_extents_and_volume_are_stable_under_rigid_rotation() -> None:
    rng = np.random.default_rng(260)
    local = rng.uniform([-1.0, -0.4, -0.2], [1.0, 0.4, 0.2], size=(1000, 3))
    rotation = rotation_matrix_from_euler(0.4, -0.25, 0.2)
    translation = np.asarray([0.5, -0.2, 0.1])
    rotated = apply_transform(local, rotation, translation)

    original_obb = compute_obb(local)
    rotated_obb = compute_obb(rotated)
    original_pca = pca_analysis(local)
    rotated_pca = pca_analysis(rotated)
    expected_main_axis = rotation @ original_pca.eigenvectors[:, 0]

    assert np.allclose(np.sort(rotated_obb.extent), np.sort(original_obb.extent), atol=1e-10)
    assert np.isclose(np.prod(rotated_obb.extent), np.prod(original_obb.extent), atol=1e-10)
    assert abs(float(rotated_pca.eigenvectors[:, 0] @ expected_main_axis)) > 0.999
