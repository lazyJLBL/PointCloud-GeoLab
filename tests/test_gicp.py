from __future__ import annotations

import numpy as np

from pointcloud_geolab.registration import estimate_local_covariances, generalized_icp
from pointcloud_geolab.utils.transform import apply_transform, rotation_matrix_from_euler


def test_estimate_local_covariances_returns_regularized_tensors() -> None:
    rng = np.random.default_rng(130)
    points = rng.normal(size=(40, 3))

    result = estimate_local_covariances(points, k_neighbors=8, regularization=1e-4)

    assert result.covariances.shape == (40, 3, 3)
    assert result.k_neighbors == 8
    assert np.all(np.linalg.eigvalsh(result.covariances) > 0)


def test_generalized_icp_recovers_small_transform() -> None:
    rng = np.random.default_rng(131)
    xy = rng.uniform(-0.8, 0.8, size=(180, 2))
    z = 0.12 * np.sin(3.0 * xy[:, 0]) + 0.08 * xy[:, 1]
    target = np.column_stack([xy, z])
    rotation = rotation_matrix_from_euler(0.025, -0.018, 0.02)
    translation = np.asarray([0.035, -0.02, 0.018])
    source = apply_transform(target, rotation, translation)

    result = generalized_icp(
        source,
        target,
        max_iterations=35,
        max_correspondence_distance=0.25,
        k_neighbors=12,
        regularization=1e-3,
    )

    assert result.final_rmse < result.initial_rmse
    assert result.final_rmse < 0.025
    assert result.diagnostics["method"] == "generalized_icp"
    assert len(result.diagnostics["mahalanobis_rmse_history"]) >= 1
