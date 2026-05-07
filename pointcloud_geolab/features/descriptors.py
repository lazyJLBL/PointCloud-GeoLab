"""Lightweight local geometric descriptors from covariance spectra."""

from __future__ import annotations

import numpy as np

from pointcloud_geolab.kdtree import KDTree


def compute_local_geometric_descriptors(
    points: np.ndarray,
    keypoint_indices: np.ndarray | list[int],
    radius: float,
    min_neighbors: int = 6,
    normalize: bool = True,
) -> np.ndarray:
    """Compute fixed-length descriptors for each keypoint.

    Descriptor layout:
    ``[linearity, planarity, scattering, curvature, anisotropy,
    omnivariance, eigenentropy, local_density]``.
    """

    pts = _ensure_points(points)
    indices = np.asarray(keypoint_indices, dtype=int).reshape(-1)
    if radius <= 0:
        raise ValueError("radius must be positive")
    if np.any(indices < 0) or np.any(indices >= len(pts)):
        raise ValueError("keypoint_indices are out of range")
    if len(indices) == 0:
        return np.empty((0, 8), dtype=float)

    tree = KDTree(pts)
    descriptors = np.zeros((len(indices), 8), dtype=float)
    volume = 4.0 / 3.0 * np.pi * radius**3
    for row, index in enumerate(indices):
        neighbor_ids = [idx for idx, _ in tree.radius_search(pts[index], radius)]
        if len(neighbor_ids) < min_neighbors:
            continue
        neighborhood = pts[neighbor_ids]
        centered = neighborhood - neighborhood.mean(axis=0)
        covariance = centered.T @ centered / max(len(neighborhood), 1)
        values = np.maximum(np.linalg.eigvalsh(covariance)[::-1], 0.0)
        total = float(np.sum(values))
        if total <= 1e-12 or values[0] <= 1e-12:
            continue
        normalized_values = values / total
        linearity = (values[0] - values[1]) / values[0]
        planarity = (values[1] - values[2]) / values[0]
        scattering = values[2] / values[0]
        curvature = values[2] / total
        anisotropy = (values[0] - values[2]) / values[0]
        omnivariance = float(np.cbrt(np.prod(np.maximum(normalized_values, 1e-12))))
        eigenentropy = -float(np.sum(normalized_values * np.log(normalized_values + 1e-12)))
        density = len(neighbor_ids) / volume
        descriptors[row] = [
            linearity,
            planarity,
            scattering,
            curvature,
            anisotropy,
            omnivariance,
            eigenentropy,
            density,
        ]

    if normalize and len(descriptors):
        scale = np.linalg.norm(descriptors, axis=1)
        mask = scale > 1e-12
        descriptors[mask] /= scale[mask, None]
    return descriptors


def _ensure_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    return pts
