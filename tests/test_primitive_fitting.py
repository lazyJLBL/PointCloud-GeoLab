from __future__ import annotations

import numpy as np

from pointcloud_geolab.datasets import make_cylinder, make_plane, make_sphere
from pointcloud_geolab.geometry.primitive_fitting import ransac_fit_primitive


def test_ransac_sphere_recovers_known_radius() -> None:
    rng = np.random.default_rng(30)
    points = make_sphere(220, center=[0.2, -0.1, 0.3], radius=0.8, noise=0.002, random_state=30)
    points = np.vstack([points, rng.uniform(-2, 2, size=(60, 3))])

    result = ransac_fit_primitive(
        points,
        "sphere",
        threshold=0.04,
        max_iterations=600,
        random_state=31,
    )
    params = result.model.get_params()

    assert result.score > 0.7
    assert np.linalg.norm(np.asarray(params["center"]) - [0.2, -0.1, 0.3]) < 0.03
    assert abs(float(params["radius"]) - 0.8) < 0.03


def test_ransac_plane_is_reproducible() -> None:
    points = make_plane(180, d=0.25, noise=0.002, random_state=32)

    first = ransac_fit_primitive(points, "plane", threshold=0.02, random_state=33)
    second = ransac_fit_primitive(points, "plane", threshold=0.02, random_state=33)

    assert np.array_equal(first.inlier_indices, second.inlier_indices)
    assert first.model.get_params() == second.model.get_params()


def test_ransac_cylinder_runs_with_outliers() -> None:
    rng = np.random.default_rng(34)
    points = make_cylinder(300, radius=0.5, noise=0.003, random_state=34)
    points = np.vstack([points, rng.uniform(-2, 2, size=(50, 3))])

    result = ransac_fit_primitive(
        points,
        "cylinder",
        threshold=0.05,
        max_iterations=700,
        random_state=35,
    )
    params = result.model.get_params()

    assert result.score > 0.5
    assert abs(float(params["radius"]) - 0.5) < 0.08
    assert np.isclose(np.linalg.norm(params["axis_direction"]), 1.0)
