"""Generalized ICP with local covariance weighting.

This module keeps the correspondence search and optimization in NumPy. It uses
the GICP objective idea, where each correspondence residual is measured through
the combined local covariance of the source and target neighborhoods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pointcloud_geolab.kdtree import KDTree
from pointcloud_geolab.registration.icp import ICPResult
from pointcloud_geolab.registration.metrics import fitness, rmse
from pointcloud_geolab.registration.svd_solver import RigidTransformResult
from pointcloud_geolab.utils.transform import apply_transform, make_transform


@dataclass(frozen=True, slots=True)
class CovarianceEstimationResult:
    """Local covariance tensors and the neighborhood size used for them."""

    covariances: np.ndarray
    k_neighbors: int
    regularization: float


def estimate_local_covariances(
    points: np.ndarray,
    k_neighbors: int = 20,
    regularization: float = 1e-3,
) -> CovarianceEstimationResult:
    """Estimate one regularized 3x3 covariance matrix per point."""

    pts = _ensure_points(points, "points")
    if k_neighbors < 3:
        raise ValueError("k_neighbors must be at least 3")
    if regularization <= 0:
        raise ValueError("regularization must be positive")

    k = min(k_neighbors, len(pts))
    tree = KDTree(pts)
    covariances = np.zeros((len(pts), 3, 3), dtype=float)
    identity = np.eye(3)
    for index, point in enumerate(pts):
        neighbors = tree.knn_search(point, k)
        neighbor_indices = [idx for idx, _ in neighbors]
        if len(neighbor_indices) < 3:
            covariances[index] = regularization * identity
            continue
        neighborhood = pts[neighbor_indices]
        centered = neighborhood - neighborhood.mean(axis=0)
        covariance = centered.T @ centered / max(len(neighborhood), 1)
        values, vectors = np.linalg.eigh(covariance)
        values = np.maximum(values, regularization)
        covariances[index] = vectors @ np.diag(values) @ vectors.T
    return CovarianceEstimationResult(
        covariances=covariances,
        k_neighbors=k,
        regularization=regularization,
    )


def generalized_icp(
    source_points: np.ndarray,
    target_points: np.ndarray,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    max_correspondence_distance: float | None = None,
    k_neighbors: int = 20,
    regularization: float = 1e-3,
    min_correspondences: int = 6,
) -> ICPResult:
    """Register source to target with a covariance-weighted GICP loop.

    The full GICP objective is nonlinear in rotation. This implementation keeps
    the project compact by using the covariance objective to derive scalar
    Mahalanobis weights, then solves each rigid update with weighted SVD.
    """

    source = _ensure_points(source_points, "source_points")
    target = _ensure_points(target_points, "target_points")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if min_correspondences < 3:
        raise ValueError("min_correspondences must be at least 3")

    source_cov = estimate_local_covariances(source, k_neighbors, regularization).covariances
    target_cov_result = estimate_local_covariances(target, k_neighbors, regularization)
    target_cov = target_cov_result.covariances
    tree = KDTree(target)

    aligned = source.copy()
    total_rotation = np.eye(3)
    total_translation = np.zeros(3)
    initial_distances = _nearest_distances(aligned, tree)
    initial_rmse = rmse(initial_distances)
    rmse_history = [initial_rmse]
    mahalanobis_history: list[float] = []
    previous_error = initial_rmse
    iterations = 0
    converged = False
    used_correspondences = 0

    for iteration in range(1, max_iterations + 1):
        src_corr, tgt_corr, src_ids, tgt_ids, distances = _find_correspondences(
            aligned,
            target,
            tree,
            max_correspondence_distance,
        )
        if len(src_corr) < min_correspondences:
            break

        weights, mahalanobis_rmse = _mahalanobis_weights(
            src_corr,
            tgt_corr,
            source_cov[src_ids],
            target_cov[tgt_ids],
            total_rotation,
            regularization,
        )
        step = _estimate_weighted_transform(src_corr, tgt_corr, weights)
        aligned = apply_transform(aligned, step.rotation, step.translation)
        total_rotation = step.rotation @ total_rotation
        total_translation = step.rotation @ total_translation + step.translation

        transformed_corr = apply_transform(src_corr, step.rotation, step.translation)
        current_error = rmse(np.linalg.norm(transformed_corr - tgt_corr, axis=1))
        rmse_history.append(current_error)
        mahalanobis_history.append(mahalanobis_rmse)
        iterations = iteration
        used_correspondences = len(src_corr)
        if abs(previous_error - current_error) < tolerance:
            converged = True
            break
        previous_error = current_error

    final_distances = _nearest_distances(aligned, tree)
    final_rmse = rmse(final_distances)
    return ICPResult(
        rotation=total_rotation,
        translation=total_translation,
        transformation=make_transform(total_rotation, total_translation),
        aligned_points=aligned,
        rmse_history=rmse_history,
        initial_rmse=initial_rmse,
        final_rmse=final_rmse,
        fitness=fitness(final_distances, max_correspondence_distance),
        iterations=iterations,
        converged=converged,
        diagnostics={
            "method": "generalized_icp",
            "k_neighbors": target_cov_result.k_neighbors,
            "regularization": regularization,
            "used_correspondences": used_correspondences,
            "mahalanobis_rmse_history": mahalanobis_history,
        },
    )


def _find_correspondences(
    source: np.ndarray,
    target: np.ndarray,
    tree: KDTree,
    max_distance: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    src_rows = []
    tgt_rows = []
    src_ids = []
    tgt_ids = []
    distances = []
    for source_index, point in enumerate(source):
        target_index, distance = tree.nearest_neighbor(point)
        if max_distance is None or distance <= max_distance:
            src_rows.append(point)
            tgt_rows.append(target[target_index])
            src_ids.append(source_index)
            tgt_ids.append(target_index)
            distances.append(distance)
    return (
        np.asarray(src_rows, dtype=float),
        np.asarray(tgt_rows, dtype=float),
        np.asarray(src_ids, dtype=int),
        np.asarray(tgt_ids, dtype=int),
        np.asarray(distances, dtype=float),
    )


def _mahalanobis_weights(
    source: np.ndarray,
    target: np.ndarray,
    source_covariances: np.ndarray,
    target_covariances: np.ndarray,
    rotation: np.ndarray,
    regularization: float,
) -> tuple[np.ndarray, float]:
    costs = np.zeros(len(source), dtype=float)
    weights = np.ones(len(source), dtype=float)
    identity = np.eye(3)
    for index, (src, tgt) in enumerate(zip(source, target, strict=True)):
        covariance = (
            target_covariances[index]
            + rotation @ source_covariances[index] @ rotation.T
            + regularization * identity
        )
        residual = src - tgt
        try:
            solved = np.linalg.solve(covariance, residual)
        except np.linalg.LinAlgError:
            solved = np.linalg.pinv(covariance) @ residual
        cost = max(float(residual @ solved), 0.0)
        costs[index] = cost
        weights[index] = 1.0 / (1.0 + cost)
    return weights, float(np.sqrt(np.mean(costs))) if len(costs) else 0.0


def _nearest_distances(points: np.ndarray, tree: KDTree) -> np.ndarray:
    return np.asarray([tree.nearest_neighbor(point)[1] for point in points], dtype=float)


def _estimate_weighted_transform(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> RigidTransformResult:
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-12:
        weights = np.ones(len(source), dtype=float)
        weight_sum = float(len(source))
    normalized = weights / weight_sum
    source_centroid = np.sum(source * normalized[:, None], axis=0)
    target_centroid = np.sum(target * normalized[:, None], axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    covariance = source_centered.T @ (target_centered * normalized[:, None])
    u, _, vh = np.linalg.svd(covariance)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0:
        vh[-1, :] *= -1
        rotation = vh.T @ u.T
    translation = target_centroid - rotation @ source_centroid
    return RigidTransformResult(
        rotation=rotation,
        translation=translation,
        transformation=make_transform(rotation, translation),
    )


def _ensure_points(points: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if len(values) < 3:
        raise ValueError(f"{name} must contain at least 3 points")
    return values
