"""From-scratch feature registration with ISS, descriptors, matching, and RANSAC."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pointcloud_geolab.features import (
    compute_local_geometric_descriptors,
    detect_iss_keypoints,
    match_descriptors,
)
from pointcloud_geolab.registration.global_registration import (
    GlobalRegistrationResult,
    RegistrationStage,
)
from pointcloud_geolab.registration.icp import point_to_plane_icp, point_to_point_icp
from pointcloud_geolab.registration.metrics import rmse
from pointcloud_geolab.registration.svd_solver import estimate_rigid_transform
from pointcloud_geolab.utils.transform import apply_homogeneous_transform


@dataclass(slots=True)
class RansacTransformResult:
    """Rigid transform estimated from feature correspondences."""

    transformation: np.ndarray
    inlier_indices: np.ndarray
    fitness: float
    inlier_rmse: float
    iterations: int
    metadata: dict[str, Any] = field(default_factory=dict)


def estimate_rigid_transform_ransac(
    source_points: np.ndarray,
    target_points: np.ndarray,
    correspondences: np.ndarray,
    threshold: float,
    max_iterations: int = 1000,
    seed: int | None = 7,
) -> RansacTransformResult:
    """Estimate a rigid transform by sampling 3 correspondence pairs."""

    src = _ensure_points(source_points, "source_points")
    tgt = _ensure_points(target_points, "target_points")
    corr = np.asarray(correspondences, dtype=int)
    if corr.ndim != 2 or corr.shape[1] != 2:
        raise ValueError("correspondences must have shape (M, 2)")
    if len(corr) < 3:
        raise ValueError("at least 3 correspondences are required")
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    rng = np.random.default_rng(seed)
    best_transform = np.eye(4)
    best_inliers = np.empty(0, dtype=int)
    best_rmse = float("inf")
    iterations = 0
    src_corr = src[corr[:, 0]]
    tgt_corr = tgt[corr[:, 1]]

    for iteration in range(1, max_iterations + 1):
        sample = rng.choice(len(corr), size=3, replace=False)
        try:
            candidate = estimate_rigid_transform(src_corr[sample], tgt_corr[sample])
        except (ValueError, np.linalg.LinAlgError):
            continue
        transformed = apply_homogeneous_transform(src_corr, candidate.transformation)
        distances = np.linalg.norm(transformed - tgt_corr, axis=1)
        inliers = np.flatnonzero(distances <= threshold)
        if len(inliers) == 0:
            continue
        inlier_rmse = rmse(distances[inliers])
        is_better = len(inliers) > len(best_inliers) or (
            len(inliers) == len(best_inliers) and inlier_rmse < best_rmse
        )
        if is_better:
            best_transform = candidate.transformation
            best_inliers = inliers
            best_rmse = inlier_rmse
            iterations = iteration

    if len(best_inliers) >= 3:
        refined = estimate_rigid_transform(src_corr[best_inliers], tgt_corr[best_inliers])
        best_transform = refined.transformation
        transformed = apply_homogeneous_transform(src_corr, best_transform)
        distances = np.linalg.norm(transformed - tgt_corr, axis=1)
        best_inliers = np.flatnonzero(distances <= threshold)
        best_rmse = rmse(distances[best_inliers]) if len(best_inliers) else float("inf")

    return RansacTransformResult(
        transformation=best_transform,
        inlier_indices=best_inliers,
        fitness=float(len(best_inliers) / len(corr)),
        inlier_rmse=best_rmse,
        iterations=iterations,
        metadata={"correspondences": len(corr), "threshold": threshold},
    )


def register_iss_descriptor_ransac_icp(
    source: np.ndarray,
    target: np.ndarray,
    salient_radius: float = 0.18,
    non_max_radius: float = 0.12,
    descriptor_radius: float = 0.25,
    threshold: float = 0.08,
    gamma21: float = 0.98,
    gamma32: float = 0.98,
    min_neighbors: int = 8,
    ratio: float = 0.9,
    seed: int | None = 7,
    icp_method: str = "point_to_point",
    max_ransac_iterations: int = 1000,
) -> GlobalRegistrationResult:
    """Run ISS + local descriptor + RANSAC coarse registration and ICP refinement."""

    src = _ensure_points(source, "source")
    tgt = _ensure_points(target, "target")
    source_keys = detect_iss_keypoints(
        src,
        salient_radius=salient_radius,
        non_max_radius=non_max_radius,
        gamma21=gamma21,
        gamma32=gamma32,
        min_neighbors=min_neighbors,
    ).indices
    target_keys = detect_iss_keypoints(
        tgt,
        salient_radius=salient_radius,
        non_max_radius=non_max_radius,
        gamma21=gamma21,
        gamma32=gamma32,
        min_neighbors=min_neighbors,
    ).indices
    source_keys = _ensure_enough_keypoints(source_keys, len(src), seed)
    target_keys = _ensure_enough_keypoints(
        target_keys, len(tgt), None if seed is None else seed + 1
    )

    source_descriptors = compute_local_geometric_descriptors(src, source_keys, descriptor_radius)
    target_descriptors = compute_local_geometric_descriptors(tgt, target_keys, descriptor_radius)
    descriptor_matches = match_descriptors(source_descriptors, target_descriptors, ratio=ratio)
    if len(descriptor_matches) < 3:
        descriptor_matches = _nearest_geometry_fallback(src, tgt, source_keys, target_keys)
    correspondences = np.column_stack(
        [source_keys[descriptor_matches[:, 0]], target_keys[descriptor_matches[:, 1]]]
    )
    coarse_ransac = estimate_rigid_transform_ransac(
        src,
        tgt,
        correspondences,
        threshold=threshold,
        max_iterations=max_ransac_iterations,
        seed=seed,
    )

    initialized = apply_homogeneous_transform(src, coarse_ransac.transformation)
    if icp_method == "point_to_point":
        refined_icp = point_to_point_icp(
            initialized,
            tgt,
            max_iterations=60,
            tolerance=1e-7,
            max_correspondence_distance=threshold,
        )
    elif icp_method == "point_to_plane":
        refined_icp = point_to_plane_icp(
            initialized,
            tgt,
            max_iterations=60,
            tolerance=1e-7,
            max_correspondence_distance=threshold,
        )
    else:
        raise ValueError("icp_method must be 'point_to_point' or 'point_to_plane'")

    refined_transform = refined_icp.transformation @ coarse_ransac.transformation
    coarse_stage = RegistrationStage(
        transformation=coarse_ransac.transformation,
        fitness=coarse_ransac.fitness,
        inlier_rmse=coarse_ransac.inlier_rmse,
        metadata={
            **coarse_ransac.metadata,
            "source_keypoints": len(source_keys),
            "target_keypoints": len(target_keys),
            "matches": len(descriptor_matches),
            "inlier_correspondences": len(coarse_ransac.inlier_indices),
        },
    )
    refined_stage = RegistrationStage(
        transformation=refined_transform,
        fitness=refined_icp.fitness,
        inlier_rmse=refined_icp.final_rmse,
        metadata=refined_icp.diagnostics | {"iterations": refined_icp.iterations},
    )
    return GlobalRegistrationResult(
        initial_transform=coarse_ransac.transformation,
        refined_transform=refined_transform,
        coarse=coarse_stage,
        refined=refined_stage,
        method=f"iss_descriptor_ransac_{icp_method}",
        source_downsampled=len(source_keys),
        target_downsampled=len(target_keys),
    )


def _ensure_enough_keypoints(indices: np.ndarray, point_count: int, seed: int | None) -> np.ndarray:
    if len(indices) >= 3:
        return indices
    rng = np.random.default_rng(seed)
    count = min(max(3, len(indices)), point_count)
    if point_count <= count:
        return np.arange(point_count, dtype=int)
    return np.sort(rng.choice(point_count, size=min(128, point_count), replace=False))


def _nearest_geometry_fallback(
    source: np.ndarray,
    target: np.ndarray,
    source_keys: np.ndarray,
    target_keys: np.ndarray,
) -> np.ndarray:
    source_centered = source[source_keys] - source[source_keys].mean(axis=0)
    target_centered = target[target_keys] - target[target_keys].mean(axis=0)
    distances = np.linalg.norm(source_centered[:, None, :] - target_centered[None, :, :], axis=2)
    order = np.argsort(distances, axis=1, kind="mergesort")
    matches = [(row, int(cols[0])) for row, cols in enumerate(order)]
    return np.asarray(matches, dtype=int)


def _ensure_points(points: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if len(values) < 3:
        raise ValueError(f"{name} must contain at least 3 points")
    return values
