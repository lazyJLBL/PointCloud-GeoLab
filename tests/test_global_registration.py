from __future__ import annotations

import numpy as np
import pytest

from pointcloud_geolab.datasets import make_sphere
from pointcloud_geolab.registration.global_registration import (
    evaluate_registration,
    register_fpfh_ransac_icp,
)
from pointcloud_geolab.utils.transform import apply_transform, rotation_matrix_from_euler


def test_global_registration_recovers_known_transform() -> None:
    pytest.importorskip("open3d")
    rng = np.random.default_rng(60)
    target = np.vstack(
        [
            make_sphere(260, radius=0.7, random_state=60),
            rng.normal(scale=0.06, size=(50, 3)) + np.asarray([0.5, 0.2, 0.15]),
        ]
    )
    rotation = rotation_matrix_from_euler(0.1, -0.05, 0.2)
    translation = np.asarray([0.2, -0.1, 0.05])
    source = apply_transform(target, rotation, translation)

    result = register_fpfh_ransac_icp(
        source,
        target,
        voxel_size=0.12,
        threshold=0.2,
        seed=61,
    )
    metrics = evaluate_registration(source, target, result.refined_transform, threshold=0.2)

    assert result.initial_transform.shape == (4, 4)
    assert result.refined_transform.shape == (4, 4)
    assert "iterations" in result.refined.metadata
    assert result.coarse.fitness >= 0.0
    assert result.coarse.inlier_rmse >= 0.0
    assert metrics["rmse"] < 1e-6


def test_global_registration_noise_does_not_crash() -> None:
    pytest.importorskip("open3d")
    rng = np.random.default_rng(62)
    target = make_sphere(220, radius=0.7, noise=0.002, random_state=62)
    target = np.vstack([target, rng.normal(scale=0.05, size=(40, 3)) + [0.4, 0.2, 0.1]])
    source = apply_transform(
        target + rng.normal(scale=0.001, size=target.shape),
        rotation_matrix_from_euler(0.0, 0.0, 0.15),
        np.asarray([0.1, -0.05, 0.02]),
    )

    result = register_fpfh_ransac_icp(source, target, voxel_size=0.12, threshold=0.25, seed=63)

    assert result.refined_transform.shape == (4, 4)
    assert result.refined.fitness >= 0.0
