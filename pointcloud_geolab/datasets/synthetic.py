"""Synthetic point cloud generators used by demos, tests, and benchmarks."""

from __future__ import annotations

import numpy as np


def make_plane(
    n: int = 300,
    normal: np.ndarray | list[float] | None = None,
    d: float = 0.0,
    noise: float = 0.0,
    random_state: int | None = None,
) -> np.ndarray:
    """Generate points on a plane ``normal^T x + d = 0``."""

    rng = np.random.default_rng(random_state)
    nrm = np.asarray([0.0, 0.0, 1.0] if normal is None else normal, dtype=float)
    nrm = nrm / np.linalg.norm(nrm)
    helper = np.asarray([1.0, 0.0, 0.0]) if abs(nrm[0]) < 0.9 else np.asarray([0.0, 1.0, 0.0])
    u = np.cross(nrm, helper)
    u /= np.linalg.norm(u)
    v = np.cross(nrm, u)
    coeffs = rng.uniform(-1.0, 1.0, size=(n, 2))
    origin = -d * nrm
    points = origin + coeffs[:, :1] * u + coeffs[:, 1:] * v
    if noise > 0:
        points += rng.normal(scale=noise, size=points.shape)
    return points


def make_sphere(
    n: int = 400,
    center: np.ndarray | list[float] | None = None,
    radius: float = 1.0,
    noise: float = 0.0,
    random_state: int | None = None,
) -> np.ndarray:
    """Generate points on a sphere surface."""

    rng = np.random.default_rng(random_state)
    c = np.asarray([0.0, 0.0, 0.0] if center is None else center, dtype=float)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    cos_theta = rng.uniform(-1.0, 1.0, size=n)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    points = np.column_stack([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])
    points = c + radius * points
    if noise > 0:
        points += rng.normal(scale=noise, size=points.shape)
    return points


def make_cylinder(
    n: int = 500,
    radius: float = 0.5,
    height: float = 2.0,
    center: np.ndarray | list[float] | None = None,
    noise: float = 0.0,
    random_state: int | None = None,
) -> np.ndarray:
    """Generate points on a vertical cylinder surface."""

    rng = np.random.default_rng(random_state)
    c = np.asarray([0.0, 0.0, 0.0] if center is None else center, dtype=float)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    z = rng.uniform(-height / 2.0, height / 2.0, size=n)
    points = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), z]) + c
    if noise > 0:
        points += rng.normal(scale=noise, size=points.shape)
    return points


def make_box(
    n: int = 400,
    size: float = 1.0,
    center: np.ndarray | list[float] | None = None,
    random_state: int | None = None,
) -> np.ndarray:
    """Generate points on a cube-like box surface."""

    rng = np.random.default_rng(random_state)
    c = np.asarray([0.0, 0.0, 0.0] if center is None else center, dtype=float)
    points = rng.uniform(-size / 2.0, size / 2.0, size=(n, 3))
    faces = rng.integers(0, 3, size=n)
    signs = rng.choice([-1.0, 1.0], size=n)
    points[np.arange(n), faces] = signs * size / 2.0
    return points + c


def make_mixed_scene(
    random_state: int | None = None,
    noise: float = 0.01,
    outliers: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate plane, sphere, cylinder, and noise with ground-truth labels."""

    rng = np.random.default_rng(random_state)
    plane = make_plane(250, d=0.4, noise=noise, random_state=random_state)
    sphere = make_sphere(220, center=[1.8, 0.0, 0.4], radius=0.35, noise=noise, random_state=2)
    cylinder = make_cylinder(240, center=[-1.8, 0.0, 0.2], noise=noise, random_state=3)
    noise_points = rng.uniform([-3.0, -2.0, -1.0], [3.0, 2.0, 1.5], size=(outliers, 3))
    points = np.vstack([plane, sphere, cylinder, noise_points])
    labels = np.concatenate(
        [
            np.zeros(len(plane), dtype=int),
            np.ones(len(sphere), dtype=int),
            np.full(len(cylinder), 2, dtype=int),
            np.full(len(noise_points), -1, dtype=int),
        ]
    )
    return points, labels
