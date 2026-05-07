from __future__ import annotations

import numpy as np

from pointcloud_geolab.registration.feature_registration import estimate_rigid_transform_ransac
from pointcloud_geolab.utils.transform import (
    apply_homogeneous_transform,
    make_transform,
    rotation_matrix_from_euler,
)


def test_ransac_transform_recovers_known_correspondences() -> None:
    rng = np.random.default_rng(140)
    source = rng.normal(size=(40, 3))
    rotation = rotation_matrix_from_euler(0.08, -0.02, 0.05)
    translation = np.asarray([0.1, -0.03, 0.04])
    target = source @ rotation.T + translation
    correspondences = np.column_stack([np.arange(len(source)), np.arange(len(source))])

    result = estimate_rigid_transform_ransac(
        source,
        target,
        correspondences,
        threshold=1e-6,
        max_iterations=80,
        seed=141,
    )

    aligned = apply_homogeneous_transform(source, result.transformation)
    assert result.fitness == 1.0
    assert np.allclose(aligned, target, atol=1e-8)
    assert np.allclose(result.transformation, make_transform(rotation, translation), atol=1e-8)
