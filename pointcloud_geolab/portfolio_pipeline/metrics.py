"""Metric helpers for the portfolio pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pointcloud_geolab.kdtree import KDTree


def _point_cloud_metrics(points: np.ndarray, path: Path) -> dict[str, Any]:
    return {
        "num_points": int(len(points)),
        "dimension": int(points.shape[1]),
        "bounds": _bounds_metrics(points),
        "has_color": _file_has_properties(path, {"red", "green", "blue", "rgb"}),
        "has_normals": _file_has_properties(path, {"nx", "ny", "nz", "normal_x"}),
    }


def _bounds_metrics(points: np.ndarray) -> dict[str, Any]:
    pts = np.asarray(points, dtype=float)
    return {
        "min": pts.min(axis=0).tolist(),
        "max": pts.max(axis=0).tolist(),
        "extent": (pts.max(axis=0) - pts.min(axis=0)).tolist(),
    }


def _file_has_properties(path: Path, property_names: set[str]) -> bool:
    suffix = path.suffix.lower()
    try:
        if suffix == ".ply":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip().lower()
                    if stripped == "end_header":
                        break
                    parts = stripped.split()
                    if len(parts) >= 3 and parts[0] == "property" and parts[-1] in property_names:
                        return True
        if suffix == ".pcd":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip().lower()
                    if stripped.startswith("fields"):
                        fields = set(stripped.split()[1:])
                        return bool(fields & property_names)
                    if stripped.startswith("data"):
                        break
    except UnicodeDecodeError:
        return False
    return False


def _auto_voxel_size(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    extent = pts.max(axis=0) - pts.min(axis=0)
    diagonal = float(np.linalg.norm(extent))
    return max(diagonal / 70.0, 1e-4)


def _auto_dbscan_eps(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    extent = pts.max(axis=0) - pts.min(axis=0)
    diagonal = float(np.linalg.norm(extent))
    return max(diagonal / 35.0, 1e-3)


def _feature_metrics(
    points: np.ndarray, normals: np.ndarray, rng: np.random.Generator
) -> dict[str, Any]:
    pts = np.asarray(points, dtype=float)
    sample_indices = _sample_indices(len(pts), min(len(pts), 300), rng)
    tree = KDTree(pts)
    density_radius = _auto_dbscan_eps(pts) * 0.8
    neighbor_counts = []
    curvatures = []
    k = min(16, len(pts))
    for index in sample_indices:
        point = pts[index]
        neighbor_counts.append(max(len(tree.radius_search(point, density_radius)) - 1, 0))
        if k >= 3:
            neighbors = tree.knn_search(point, k)
            neighborhood = pts[[idx for idx, _ in neighbors]]
            centered = neighborhood - neighborhood.mean(axis=0)
            covariance = centered.T @ centered / max(len(neighborhood), 1)
            eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
            total = float(eigenvalues.sum())
            curvatures.append(float(eigenvalues[0] / total) if total > 0 else 0.0)

    counts = np.asarray(neighbor_counts, dtype=float)
    curvature_values = np.asarray(curvatures, dtype=float)
    return {
        "estimated_normals": True,
        "normal_mean": normals.mean(axis=0).tolist(),
        "normal_abs_z_mean": float(np.mean(np.abs(normals[:, 2]))),
        "density_radius": float(density_radius),
        "local_density_mean": float(counts.mean()) if len(counts) else 0.0,
        "local_density_min": int(counts.min()) if len(counts) else 0,
        "local_density_max": int(counts.max()) if len(counts) else 0,
        "curvature_mean": float(curvature_values.mean()) if len(curvature_values) else 0.0,
    }


def _sample_indices(count: int, sample_count: int, rng: np.random.Generator) -> np.ndarray:
    if count <= sample_count:
        return np.arange(count, dtype=int)
    return np.sort(rng.choice(count, size=sample_count, replace=False))
