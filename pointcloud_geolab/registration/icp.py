"""Point-to-point ICP using the custom KD-Tree."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pointcloud_geolab.kdtree.kdtree import KDTree
from pointcloud_geolab.registration.metrics import fitness, rmse
from pointcloud_geolab.registration.svd_solver import estimate_rigid_transform
from pointcloud_geolab.utils.transform import apply_transform, make_transform


@dataclass(slots=True)
class ICPResult:
    rotation: np.ndarray
    translation: np.ndarray
    transformation: np.ndarray
    aligned_points: np.ndarray
    rmse_history: list[float]
    initial_rmse: float
    final_rmse: float
    fitness: float
    iterations: int
    converged: bool


def point_to_point_icp(
    source_points: np.ndarray,
    target_points: np.ndarray,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    max_correspondence_distance: float | None = None,
) -> ICPResult:
    """Register ``source_points`` to ``target_points`` with point-to-point ICP."""

    source = np.asarray(source_points, dtype=float)
    target = np.asarray(target_points, dtype=float)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != 3 or target.shape[1] != 3:
        raise ValueError("source_points and target_points must have shape (N, 3)")
    if len(source) < 3 or len(target) < 3:
        raise ValueError("ICP requires at least 3 source and target points")

    tree = KDTree(target)
    aligned = source.copy()
    total_rotation = np.eye(3)
    total_translation = np.zeros(3)

    initial_distances = _nearest_distances(aligned, tree)
    initial_rmse = rmse(initial_distances)
    rmse_history = [initial_rmse]
    previous_error = initial_rmse
    converged = False
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        src_corr, tgt_corr = _find_correspondences(aligned, target, tree, max_correspondence_distance)
        if len(src_corr) < 3:
            break

        step = estimate_rigid_transform(src_corr, tgt_corr)
        aligned = apply_transform(aligned, step.rotation, step.translation)
        total_rotation = step.rotation @ total_rotation
        total_translation = step.rotation @ total_translation + step.translation

        transformed_corr = apply_transform(src_corr, step.rotation, step.translation)
        distances = np.linalg.norm(transformed_corr - tgt_corr, axis=1)
        current_error = rmse(distances)
        rmse_history.append(current_error)
        iterations = iteration

        if abs(previous_error - current_error) < tolerance:
            converged = True
            break
        previous_error = current_error

    final_distances = _nearest_distances(aligned, tree)
    final_rmse = rmse(final_distances)
    inlier_threshold = max_correspondence_distance
    return ICPResult(
        rotation=total_rotation,
        translation=total_translation,
        transformation=make_transform(total_rotation, total_translation),
        aligned_points=aligned,
        rmse_history=rmse_history,
        initial_rmse=initial_rmse,
        final_rmse=final_rmse,
        fitness=fitness(final_distances, inlier_threshold),
        iterations=iterations,
        converged=converged,
    )


def _find_correspondences(
    source: np.ndarray,
    target: np.ndarray,
    tree: KDTree,
    max_distance: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    src_rows = []
    tgt_rows = []
    for point in source:
        idx, distance = tree.nearest_neighbor(point)
        if max_distance is None or distance <= max_distance:
            src_rows.append(point)
            tgt_rows.append(target[idx])
    return np.asarray(src_rows, dtype=float), np.asarray(tgt_rows, dtype=float)


def _nearest_distances(points: np.ndarray, tree: KDTree) -> np.ndarray:
    return np.asarray([tree.nearest_neighbor(point)[1] for point in points], dtype=float)

