from __future__ import annotations

import numpy as np
import pytest

from pointcloud_geolab.registration.feature_registration import (
    estimate_rigid_transform_ransac,
    register_iss_descriptor_ransac_icp,
)
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


def test_iss_descriptor_registration_does_not_silently_fallback(monkeypatch) -> None:
    rng = np.random.default_rng(142)
    target = rng.normal(scale=0.2, size=(80, 3))
    source = target + np.asarray([0.02, -0.01, 0.015])
    monkeypatch.setattr(
        "pointcloud_geolab.registration.feature_registration.match_descriptors",
        lambda *args, **kwargs: np.empty((0, 2), dtype=int),
    )

    with pytest.raises(ValueError, match="fallback is not descriptor registration success"):
        register_iss_descriptor_ransac_icp(
            source,
            target,
            salient_radius=0.12,
            non_max_radius=0.08,
            descriptor_radius=0.18,
            threshold=0.08,
            seed=143,
        )


def test_iss_descriptor_geometry_fallback_is_diagnostic_only(monkeypatch) -> None:
    rng = np.random.default_rng(144)
    target = rng.normal(scale=0.2, size=(80, 3))
    source = target + np.asarray([0.02, -0.01, 0.015])
    monkeypatch.setattr(
        "pointcloud_geolab.registration.feature_registration.match_descriptors",
        lambda *args, **kwargs: np.empty((0, 2), dtype=int),
    )

    result = register_iss_descriptor_ransac_icp(
        source,
        target,
        salient_radius=0.12,
        non_max_radius=0.08,
        descriptor_radius=0.18,
        threshold=0.08,
        seed=145,
        allow_geometry_fallback=True,
    )

    assert result.coarse.metadata["geometry_fallback_used"] is True
    assert result.coarse.metadata["fallback_not_descriptor_success"] is True
    assert result.coarse.metadata["descriptor_matches"] == 0
    assert result.refined.metadata["num_correspondences"] > 0
