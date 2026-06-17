from __future__ import annotations

import numpy as np
import pytest

from pointcloud_geolab.datasets import make_cylinder, make_plane, make_sphere
from pointcloud_geolab.geometry.primitive_fitting import ransac_fit_primitive


@pytest.mark.parametrize(
    ("model", "inliers", "threshold", "max_iterations"),
    [
        ("plane", make_plane(180, d=0.2, noise=0.002, random_state=201), 0.02, 400),
        (
            "sphere",
            make_sphere(220, center=[0.2, -0.1, 0.15], radius=0.7, noise=0.002, random_state=202),
            0.04,
            700,
        ),
        ("cylinder", make_cylinder(280, radius=0.45, noise=0.003, random_state=203), 0.05, 900),
    ],
)
@pytest.mark.parametrize("outlier_ratio", [0.0, 0.35])
def test_ransac_primitives_recover_inliers_under_outliers(
    model: str,
    inliers: np.ndarray,
    threshold: float,
    max_iterations: int,
    outlier_ratio: float,
) -> None:
    rng = np.random.default_rng(210 + int(outlier_ratio * 100))
    outlier_count = int(len(inliers) * outlier_ratio / max(1.0 - outlier_ratio, 1e-12))
    outliers = rng.uniform(-2.0, 2.0, size=(outlier_count, 3))
    points = np.vstack([inliers, outliers])
    truth = np.zeros(len(points), dtype=bool)
    truth[: len(inliers)] = True

    result = ransac_fit_primitive(
        points,
        model,
        threshold=threshold,
        max_iterations=max_iterations,
        random_state=220,
    )
    predicted = np.zeros(len(points), dtype=bool)
    predicted[result.inlier_indices] = True

    true_positive = int(np.sum(predicted & truth))
    precision = true_positive / max(int(np.sum(predicted)), 1)
    recall = true_positive / len(inliers)

    assert precision >= 0.85
    assert recall >= 0.75
