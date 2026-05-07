"""ICP variants using the custom KD-Tree."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pointcloud_geolab.kdtree.kdtree import KDTree
from pointcloud_geolab.preprocessing import estimate_normals, voxel_downsample
from pointcloud_geolab.registration.metrics import fitness, rmse
from pointcloud_geolab.registration.svd_solver import RigidTransformResult, estimate_rigid_transform
from pointcloud_geolab.utils.transform import (
    apply_homogeneous_transform,
    apply_transform,
    make_transform,
)


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
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class MultiScaleICPResult:
    """Coarse-to-fine ICP result with per-level diagnostics."""

    transformation: np.ndarray
    aligned_points: np.ndarray
    diagnostics: list[dict[str, object]]
    final_rmse: float
    fitness: float
    converged: bool


def point_to_point_icp(
    source_points: np.ndarray,
    target_points: np.ndarray,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    max_correspondence_distance: float | None = None,
    robust_kernel: str = "none",
    trim_ratio: float = 1.0,
    robust_delta: float | None = None,
) -> ICPResult:
    """Register ``source_points`` to ``target_points`` with point-to-point ICP."""

    source = np.asarray(source_points, dtype=float)
    target = np.asarray(target_points, dtype=float)
    _validate_icp_inputs(source, target)
    _validate_robust_options(robust_kernel, trim_ratio)
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
    used_correspondences = 0

    for iteration in range(1, max_iterations + 1):
        src_corr, tgt_corr, corr_distances = _find_correspondences(
            aligned,
            target,
            tree,
            max_correspondence_distance,
        )
        src_corr, tgt_corr, corr_distances, weights = _robust_correspondence_filter(
            src_corr,
            tgt_corr,
            corr_distances,
            trim_ratio=trim_ratio,
            robust_kernel=robust_kernel,
            robust_delta=robust_delta,
        )
        if len(src_corr) < 3:
            break

        step = _estimate_weighted_transform(src_corr, tgt_corr, weights)
        aligned = apply_transform(aligned, step.rotation, step.translation)
        total_rotation = step.rotation @ total_rotation
        total_translation = step.rotation @ total_translation + step.translation

        transformed_corr = apply_transform(src_corr, step.rotation, step.translation)
        distances = np.linalg.norm(transformed_corr - tgt_corr, axis=1)
        current_error = rmse(distances)
        rmse_history.append(current_error)
        iterations = iteration
        used_correspondences = len(src_corr)

        if abs(previous_error - current_error) < tolerance:
            converged = True
            break
        previous_error = current_error

    final_distances = _nearest_distances(aligned, tree)
    final_rmse = _reported_rmse(final_distances, trim_ratio)
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
        diagnostics={
            "method": "point_to_point",
            "robust_kernel": robust_kernel,
            "trim_ratio": trim_ratio,
            "used_correspondences": used_correspondences,
            "raw_final_rmse": rmse(final_distances),
        },
    )


def point_to_plane_icp(
    source_points: np.ndarray,
    target_points: np.ndarray,
    target_normals: np.ndarray | None = None,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    max_correspondence_distance: float | None = None,
    robust_kernel: str = "none",
    trim_ratio: float = 1.0,
    robust_delta: float | None = None,
    min_correspondences: int = 6,
) -> ICPResult:
    """Register with linearized point-to-plane ICP.

    For correspondence ``(p, q, n)`` the small-angle residual is
    ``n^T((w x p) + t + p - q)``. Each iteration solves for ``[w, t]`` by
    weighted least squares.
    """

    source = np.asarray(source_points, dtype=float)
    target = np.asarray(target_points, dtype=float)
    _validate_icp_inputs(source, target)
    _validate_robust_options(robust_kernel, trim_ratio)
    if min_correspondences < 6:
        raise ValueError("min_correspondences must be at least 6")
    normals = _prepare_normals(target, target_normals)

    tree = KDTree(target)
    transform = np.eye(4)
    aligned = source.copy()
    initial_distances = _nearest_distances(aligned, tree)
    initial_rmse = rmse(initial_distances)
    rmse_history = [initial_rmse]
    previous_error = initial_rmse
    converged = False
    iterations = 0
    last_condition = float("nan")
    used_correspondences = 0

    for iteration in range(1, max_iterations + 1):
        aligned = apply_homogeneous_transform(source, transform)
        src_corr, tgt_corr, corr_distances, normal_corr = _find_point_to_plane_correspondences(
            aligned,
            target,
            normals,
            tree,
            max_correspondence_distance,
        )
        src_corr, tgt_corr, corr_distances, weights, keep = _robust_correspondence_filter(
            src_corr,
            tgt_corr,
            corr_distances,
            trim_ratio=trim_ratio,
            robust_kernel=robust_kernel,
            robust_delta=robust_delta,
            return_keep=True,
        )
        normal_corr = normal_corr[keep]
        if len(src_corr) < min_correspondences:
            break

        rows = np.column_stack((np.cross(src_corr, normal_corr), normal_corr))
        rhs = -np.einsum("ij,ij->i", normal_corr, src_corr - tgt_corr)
        sqrt_weights = np.sqrt(np.clip(weights, 0.0, None))
        weighted_rows = rows * sqrt_weights[:, None]
        weighted_rhs = rhs * sqrt_weights
        try:
            last_condition = float(np.linalg.cond(weighted_rows))
            if not np.isfinite(last_condition) or last_condition > 1e12:
                break
            solution, *_ = np.linalg.lstsq(weighted_rows, weighted_rhs, rcond=None)
        except np.linalg.LinAlgError:
            break

        delta = make_transform(_rodrigues(solution[:3]), solution[3:])
        transform = delta @ transform
        current_error = rmse(corr_distances)
        rmse_history.append(current_error)
        iterations = iteration
        used_correspondences = len(src_corr)
        if abs(previous_error - current_error) < tolerance:
            converged = True
            break
        previous_error = current_error

    aligned = apply_homogeneous_transform(source, transform)
    final_distances = _nearest_distances(aligned, tree)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return ICPResult(
        rotation=rotation,
        translation=translation,
        transformation=transform,
        aligned_points=aligned,
        rmse_history=rmse_history,
        initial_rmse=initial_rmse,
        final_rmse=_reported_rmse(final_distances, trim_ratio),
        fitness=fitness(final_distances, max_correspondence_distance),
        iterations=iterations,
        converged=converged,
        diagnostics={
            "method": "point_to_plane",
            "robust_kernel": robust_kernel,
            "trim_ratio": trim_ratio,
            "condition_number": last_condition,
            "used_correspondences": used_correspondences,
            "raw_final_rmse": rmse(final_distances),
        },
    )


def robust_icp(
    source_points: np.ndarray,
    target_points: np.ndarray,
    method: str = "point_to_point",
    robust_kernel: str = "huber",
    trim_ratio: float = 0.8,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    max_correspondence_distance: float | None = None,
) -> ICPResult:
    """Convenience wrapper for trimmed or robust-kernel ICP."""

    return _run_icp_variant(
        source_points,
        target_points,
        method=method,
        max_iterations=max_iterations,
        tolerance=tolerance,
        max_correspondence_distance=max_correspondence_distance,
        robust_kernel=robust_kernel,
        trim_ratio=trim_ratio,
    )


def multiscale_icp(
    source_points: np.ndarray,
    target_points: np.ndarray,
    voxel_sizes: list[float] | tuple[float, ...],
    max_iterations_per_level: int | list[int] | tuple[int, ...] = 30,
    method: str = "point_to_point",
    tolerance: float = 1e-6,
    max_correspondence_distance: float | None = None,
    robust_kernel: str = "none",
    trim_ratio: float = 1.0,
) -> MultiScaleICPResult:
    """Run coarse-to-fine ICP from larger voxels to smaller voxels."""

    source = np.asarray(source_points, dtype=float)
    target = np.asarray(target_points, dtype=float)
    _validate_icp_inputs(source, target)
    sizes = [float(value) for value in voxel_sizes]
    if not sizes or any(value <= 0 for value in sizes):
        raise ValueError("voxel_sizes must contain positive values")
    if isinstance(max_iterations_per_level, int):
        iterations_by_level = [max_iterations_per_level] * len(sizes)
    else:
        iterations_by_level = [int(value) for value in max_iterations_per_level]
        if len(iterations_by_level) != len(sizes):
            raise ValueError("max_iterations_per_level must match voxel_sizes")

    transform = np.eye(4)
    diagnostics = []
    converged = False
    for level, (voxel_size, max_iterations) in enumerate(
        zip(sizes, iterations_by_level, strict=True)
    ):
        src_down = voxel_downsample(source, voxel_size)
        tgt_down = voxel_downsample(target, voxel_size)
        initialized = apply_homogeneous_transform(src_down, transform)
        threshold = max_correspondence_distance or voxel_size * 2.5
        result = _run_icp_variant(
            initialized,
            tgt_down,
            method=method,
            max_iterations=max_iterations,
            tolerance=tolerance,
            max_correspondence_distance=threshold,
            robust_kernel=robust_kernel,
            trim_ratio=trim_ratio,
        )
        transform = result.transformation @ transform
        converged = result.converged
        diagnostics.append(
            {
                "level": level,
                "voxel_size": voxel_size,
                "source_points": len(src_down),
                "target_points": len(tgt_down),
                "iterations": result.iterations,
                "rmse": result.final_rmse,
                "fitness": result.fitness,
                "converged": result.converged,
            }
        )

    aligned = apply_homogeneous_transform(source, transform)
    tree = KDTree(target)
    final_distances = _nearest_distances(aligned, tree)
    return MultiScaleICPResult(
        transformation=transform,
        aligned_points=aligned,
        diagnostics=diagnostics,
        final_rmse=rmse(final_distances),
        fitness=fitness(final_distances, max_correspondence_distance),
        converged=converged,
    )


def _find_correspondences(
    source: np.ndarray,
    target: np.ndarray,
    tree: KDTree,
    max_distance: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src_rows = []
    tgt_rows = []
    distances = []
    for point in source:
        idx, distance = tree.nearest_neighbor(point)
        if max_distance is None or distance <= max_distance:
            src_rows.append(point)
            tgt_rows.append(target[idx])
            distances.append(distance)
    return (
        np.asarray(src_rows, dtype=float),
        np.asarray(tgt_rows, dtype=float),
        np.asarray(distances, dtype=float),
    )


def _find_point_to_plane_correspondences(
    source: np.ndarray,
    target: np.ndarray,
    normals: np.ndarray,
    tree: KDTree,
    max_distance: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    src_rows = []
    tgt_rows = []
    distances = []
    normal_rows = []
    for point in source:
        idx, distance = tree.nearest_neighbor(point)
        if max_distance is None or distance <= max_distance:
            src_rows.append(point)
            tgt_rows.append(target[idx])
            distances.append(distance)
            normal_rows.append(normals[idx])
    return (
        np.asarray(src_rows, dtype=float),
        np.asarray(tgt_rows, dtype=float),
        np.asarray(distances, dtype=float),
        np.asarray(normal_rows, dtype=float),
    )


def _nearest_distances(points: np.ndarray, tree: KDTree) -> np.ndarray:
    return np.asarray([tree.nearest_neighbor(point)[1] for point in points], dtype=float)


def _run_icp_variant(
    source_points: np.ndarray,
    target_points: np.ndarray,
    method: str,
    max_iterations: int,
    tolerance: float,
    max_correspondence_distance: float | None,
    robust_kernel: str,
    trim_ratio: float,
) -> ICPResult:
    if method == "point_to_point":
        return point_to_point_icp(
            source_points,
            target_points,
            max_iterations=max_iterations,
            tolerance=tolerance,
            max_correspondence_distance=max_correspondence_distance,
            robust_kernel=robust_kernel,
            trim_ratio=trim_ratio,
        )
    if method == "point_to_plane":
        return point_to_plane_icp(
            source_points,
            target_points,
            max_iterations=max_iterations,
            tolerance=tolerance,
            max_correspondence_distance=max_correspondence_distance,
            robust_kernel=robust_kernel,
            trim_ratio=trim_ratio,
        )
    raise ValueError("method must be 'point_to_point' or 'point_to_plane'")


def _robust_correspondence_filter(
    source: np.ndarray,
    target: np.ndarray,
    distances: np.ndarray,
    trim_ratio: float,
    robust_kernel: str,
    robust_delta: float | None,
    return_keep: bool = False,
):
    if len(source) == 0:
        weights = np.empty(0, dtype=float)
        keep = np.empty(0, dtype=int)
        if return_keep:
            return source, target, distances, weights, keep
        return source, target, distances, weights

    order = np.argsort(distances, kind="mergesort")
    keep_count = max(1, int(np.ceil(len(order) * trim_ratio)))
    keep = order[:keep_count]
    filtered_distances = distances[keep]
    weights = _robust_weights(filtered_distances, robust_kernel, robust_delta)
    if return_keep:
        return source[keep], target[keep], filtered_distances, weights, keep
    return source[keep], target[keep], filtered_distances, weights


def _robust_weights(
    distances: np.ndarray,
    robust_kernel: str,
    robust_delta: float | None,
) -> np.ndarray:
    kernel = robust_kernel.lower()
    if kernel == "none":
        return np.ones(len(distances), dtype=float)
    delta = robust_delta
    if delta is None:
        finite = distances[np.isfinite(distances)]
        delta = float(np.median(finite) * 1.4826) if len(finite) else 1.0
        delta = max(delta, 1e-6)
    scaled = distances / delta
    if kernel == "huber":
        return np.where(scaled <= 1.0, 1.0, 1.0 / np.maximum(scaled, 1e-12))
    if kernel == "tukey":
        weights = np.zeros(len(distances), dtype=float)
        mask = scaled < 1.0
        weights[mask] = (1.0 - scaled[mask] ** 2) ** 2
        return weights
    raise ValueError("robust_kernel must be one of: none, huber, tukey")


def _reported_rmse(distances: np.ndarray, trim_ratio: float) -> float:
    if len(distances) == 0:
        return rmse(distances)
    if trim_ratio >= 1.0:
        return rmse(distances)
    count = max(1, int(np.ceil(len(distances) * trim_ratio)))
    return rmse(np.sort(distances)[:count])


def _estimate_weighted_transform(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
):
    if np.allclose(weights, weights[0]):
        return estimate_rigid_transform(source, target)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-12:
        return estimate_rigid_transform(source, target)
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


def _prepare_normals(target: np.ndarray, normals: np.ndarray | None) -> np.ndarray:
    if normals is None:
        return estimate_normals(target, k=min(24, max(3, len(target))))
    values = np.asarray(normals, dtype=float)
    if values.shape != target.shape:
        raise ValueError("target_normals must have the same shape as target_points")
    lengths = np.linalg.norm(values, axis=1)
    safe = lengths > 1e-12
    prepared = np.zeros_like(values)
    prepared[safe] = values[safe] / lengths[safe, None]
    prepared[~safe] = np.asarray([0.0, 0.0, 1.0])
    return prepared


def _validate_icp_inputs(source: np.ndarray, target: np.ndarray) -> None:
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != 3 or target.shape[1] != 3:
        raise ValueError("source_points and target_points must have shape (N, 3)")
    if len(source) < 3 or len(target) < 3:
        raise ValueError("ICP requires at least 3 source and target points")


def _validate_robust_options(robust_kernel: str, trim_ratio: float) -> None:
    if robust_kernel not in {"none", "huber", "tukey"}:
        raise ValueError("robust_kernel must be one of: none, huber, tukey")
    if not (0.0 < trim_ratio <= 1.0):
        raise ValueError("trim_ratio must be in (0, 1]")


def _rodrigues(vector: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(vector))
    if theta < 1e-12:
        return np.eye(3)
    axis = vector / theta
    kx = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(theta) * kx + (1.0 - np.cos(theta)) * (kx @ kx)
