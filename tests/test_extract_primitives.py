from __future__ import annotations

import json

import numpy as np

from pointcloud_geolab.datasets import make_plane, make_sphere
from pointcloud_geolab.geometry import extract_primitives


def test_sequential_ransac_extracts_plane_and_sphere() -> None:
    rng = np.random.default_rng(150)
    plane = make_plane(140, d=0.0, noise=0.002, random_state=150)
    sphere = make_sphere(120, center=[1.2, 0.0, 0.35], radius=0.3, noise=0.002, random_state=151)
    outliers = rng.uniform(-1.5, 1.5, size=(30, 3))
    points = np.vstack([plane, sphere, outliers])

    result = extract_primitives(
        points,
        model_types=["plane", "sphere"],
        threshold=0.03,
        max_models=3,
        min_inliers=50,
        max_iterations=500,
        random_state=152,
    )

    model_types = {primitive.model_type for primitive in result.primitives}
    assert {"plane", "sphere"} <= model_types
    assert len(result.primitives) <= 3
    json.dumps([primitive.get_params() for primitive in result.primitives])


def test_sequential_ransac_handles_high_outliers() -> None:
    rng = np.random.default_rng(153)
    plane = make_plane(100, noise=0.003, random_state=154)
    outliers = rng.uniform(-2.0, 2.0, size=(120, 3))

    result = extract_primitives(
        np.vstack([plane, outliers]),
        model_types=["plane", "sphere"],
        threshold=0.04,
        max_models=2,
        min_inliers=40,
        max_iterations=400,
        random_state=155,
    )

    assert len(result.primitives) >= 1
