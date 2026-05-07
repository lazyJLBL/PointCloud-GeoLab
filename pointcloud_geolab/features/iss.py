"""Intrinsic Shape Signatures (ISS) keypoint detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pointcloud_geolab.kdtree import KDTree


@dataclass(frozen=True, slots=True)
class ISSKeypointResult:
    """Detected ISS keypoint indices and saliency values."""

    indices: np.ndarray
    saliency: np.ndarray
    eigenvalues: np.ndarray


def detect_iss_keypoints(
    points: np.ndarray,
    salient_radius: float,
    non_max_radius: float,
    gamma21: float = 0.975,
    gamma32: float = 0.975,
    min_neighbors: int = 8,
) -> ISSKeypointResult:
    """Detect ISS keypoints from local covariance eigenvalue ratios."""

    pts = _ensure_points(points)
    if salient_radius <= 0 or non_max_radius <= 0:
        raise ValueError("salient_radius and non_max_radius must be positive")
    if min_neighbors < 3:
        raise ValueError("min_neighbors must be at least 3")
    if len(pts) == 0:
        empty = np.empty(0, dtype=int)
        return ISSKeypointResult(empty, np.empty(0), np.empty((0, 3)))

    tree = KDTree(pts)
    eigenvalues = np.zeros((len(pts), 3), dtype=float)
    saliency = np.zeros(len(pts), dtype=float)
    candidates = []

    for index, point in enumerate(pts):
        neighbor_ids = [idx for idx, _ in tree.radius_search(point, salient_radius)]
        if len(neighbor_ids) < min_neighbors:
            continue
        neighborhood = pts[neighbor_ids]
        centered = neighborhood - neighborhood.mean(axis=0)
        covariance = centered.T @ centered / max(len(neighborhood), 1)
        values = np.linalg.eigvalsh(covariance)[::-1]
        eigenvalues[index] = values
        if values[0] <= 1e-12 or values[1] <= 1e-12:
            continue
        if values[1] / values[0] < gamma21 and values[2] / values[1] < gamma32:
            saliency[index] = values[2]
            candidates.append(index)

    selected = _non_maximum_suppression(candidates, pts, saliency, tree, non_max_radius)
    return ISSKeypointResult(
        indices=np.asarray(selected, dtype=int),
        saliency=saliency[np.asarray(selected, dtype=int)] if selected else np.empty(0),
        eigenvalues=eigenvalues[np.asarray(selected, dtype=int)] if selected else np.empty((0, 3)),
    )


def _non_maximum_suppression(
    candidates: list[int],
    points: np.ndarray,
    saliency: np.ndarray,
    tree: KDTree,
    radius: float,
) -> list[int]:
    if not candidates:
        return []
    candidate_set = set(candidates)
    selected = []
    for index in sorted(candidates, key=lambda idx: (-saliency[idx], idx)):
        if index not in candidate_set:
            continue
        selected.append(index)
        for neighbor, _ in tree.radius_search(points[index], radius):
            if neighbor != index:
                candidate_set.discard(neighbor)
    selected.sort()
    return selected


def _ensure_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    return pts
