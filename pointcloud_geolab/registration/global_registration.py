"""Feature-based global registration followed by ICP refinement.

The pipeline is:

``Downsample -> Normal Estimation -> FPFH -> RANSAC -> ICP``

FPFH extraction and correspondence RANSAC use Open3D because they are standard
feature primitives, while refinement can use the project's own point-to-point
ICP or the lightweight point-to-plane solver implemented in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pointcloud_geolab.kdtree import KDTree
from pointcloud_geolab.preprocessing import estimate_normals as estimate_normals_numpy
from pointcloud_geolab.preprocessing import voxel_downsample as voxel_downsample_numpy
from pointcloud_geolab.registration.icp import point_to_point_icp
from pointcloud_geolab.registration.metrics import fitness, rmse
from pointcloud_geolab.utils.transform import apply_homogeneous_transform, make_transform


@dataclass(slots=True)
class RegistrationStage:
    """Metrics and transform for one registration stage."""

    transformation: np.ndarray
    fitness: float
    inlier_rmse: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GlobalRegistrationResult:
    """Result for FPFH + RANSAC + ICP registration."""

    initial_transform: np.ndarray
    refined_transform: np.ndarray
    coarse: RegistrationStage
    refined: RegistrationStage
    method: str
    source_downsampled: int
    target_downsampled: int


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Downsample an ``(N, 3)`` point cloud using the existing voxel grid."""

    return voxel_downsample_numpy(points, voxel_size)


def estimate_normals(point_cloud, radius: float, max_nn: int = 30):
    """Estimate normals for an Open3D point cloud or NumPy point array.

    Open3D point clouds are modified in place and returned. NumPy arrays return
    an ``(N, 3)`` array of normals from the project's local PCA estimator.
    """

    if isinstance(point_cloud, np.ndarray):
        del radius
        return estimate_normals_numpy(point_cloud, k=max_nn)

    o3d = _require_open3d()
    point_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    return point_cloud


def compute_fpfh_features(point_cloud, radius: float, max_nn: int = 100):
    """Compute Open3D FPFH features for a point cloud with normals."""

    o3d = _require_open3d()
    estimate_normals(point_cloud, radius=radius * 0.5, max_nn=30)
    return o3d.pipelines.registration.compute_fpfh_feature(
        point_cloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn),
    )


def execute_global_registration(
    source: np.ndarray,
    target: np.ndarray,
    voxel_size: float,
    seed: int | None = 7,
) -> tuple[RegistrationStage, np.ndarray, np.ndarray]:
    """Run FPFH feature matching and RANSAC coarse alignment.

    Returns the coarse stage plus the downsampled source and target arrays used
    for feature matching.
    """

    o3d = _require_open3d()
    if seed is not None:
        try:
            o3d.utility.random.seed(int(seed))
        except AttributeError:
            pass

    source_down = _to_o3d(source).voxel_down_sample(voxel_size)
    target_down = _to_o3d(target).voxel_down_sample(voxel_size)
    normal_radius = voxel_size * 2.0
    feature_radius = voxel_size * 5.0
    estimate_normals(source_down, normal_radius, max_nn=30)
    estimate_normals(target_down, normal_radius, max_nn=30)
    source_fpfh = compute_fpfh_features(source_down, feature_radius, max_nn=100)
    target_fpfh = compute_fpfh_features(target_down, feature_radius, max_nn=100)

    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(40000, 0.999),
    )
    stage = RegistrationStage(
        transformation=np.asarray(result.transformation, dtype=float),
        fitness=float(result.fitness),
        inlier_rmse=float(result.inlier_rmse),
        metadata={"distance_threshold": distance_threshold},
    )
    return stage, np.asarray(source_down.points), np.asarray(target_down.points)


def refine_registration_icp(
    source: np.ndarray,
    target: np.ndarray,
    init_transform: np.ndarray,
    threshold: float,
    method: str = "point_to_point",
    max_iterations: int = 60,
    tolerance: float = 1e-7,
) -> RegistrationStage:
    """Refine an initial transform with point-to-point or point-to-plane ICP."""

    matrix = np.asarray(init_transform, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("init_transform must have shape (4, 4)")
    if method == "point_to_point":
        initialized = apply_homogeneous_transform(source, matrix)
        result = point_to_point_icp(
            initialized,
            target,
            max_iterations=max_iterations,
            tolerance=tolerance,
            max_correspondence_distance=threshold,
        )
        refined = result.transformation @ matrix
        return RegistrationStage(
            transformation=refined,
            fitness=result.fitness,
            inlier_rmse=result.final_rmse,
            metadata={"iterations": result.iterations, "converged": result.converged},
        )
    if method == "point_to_plane":
        return _point_to_plane_icp(source, target, matrix, threshold, max_iterations, tolerance)
    raise ValueError("method must be 'point_to_point' or 'point_to_plane'")


def register_fpfh_ransac_icp(
    source: np.ndarray,
    target: np.ndarray,
    voxel_size: float,
    icp_method: str = "point_to_point",
    threshold: float | None = None,
    seed: int | None = 7,
) -> GlobalRegistrationResult:
    """Run coarse FPFH/RANSAC registration and ICP refinement."""

    src = _ensure_points(source, "source")
    tgt = _ensure_points(target, "target")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    coarse, source_down, target_down = execute_global_registration(src, tgt, voxel_size, seed=seed)
    icp_threshold = threshold if threshold is not None else voxel_size * 1.5
    refined = refine_registration_icp(
        src,
        tgt,
        coarse.transformation,
        threshold=icp_threshold,
        method=icp_method,
    )
    return GlobalRegistrationResult(
        initial_transform=coarse.transformation,
        refined_transform=refined.transformation,
        coarse=coarse,
        refined=refined,
        method=f"fpfh_ransac_{icp_method}",
        source_downsampled=len(source_down),
        target_downsampled=len(target_down),
    )


def evaluate_registration(
    source: np.ndarray,
    target: np.ndarray,
    transform: np.ndarray,
    threshold: float | None = None,
) -> dict[str, float]:
    """Compute nearest-neighbor RMSE and fitness after applying a transform."""

    aligned = apply_homogeneous_transform(source, transform)
    tree = KDTree(target)
    distances = np.asarray([tree.nearest_neighbor(point)[1] for point in aligned], dtype=float)
    return {
        "rmse": rmse(distances),
        "fitness": fitness(distances, threshold),
    }


def _point_to_plane_icp(
    source: np.ndarray,
    target: np.ndarray,
    init_transform: np.ndarray,
    threshold: float,
    max_iterations: int,
    tolerance: float,
) -> RegistrationStage:
    """Linearized point-to-plane ICP.

    For each correspondence ``(p, q, n)``, solve the small-angle update:
    ``n^T((w x p) + t + p - q) = 0``. The least-squares variables are the
    rotation vector ``w`` and translation ``t``.
    """

    transform = init_transform.copy()
    target_normals = estimate_normals_numpy(target, k=min(24, max(3, len(target))))
    tree = KDTree(target)
    previous_error = float("inf")
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        transformed = apply_homogeneous_transform(source, transform)
        rows = []
        rhs = []
        distances = []
        for point in transformed:
            idx, distance = tree.nearest_neighbor(point)
            if distance > threshold:
                continue
            normal = target_normals[idx]
            target_point = target[idx]
            rows.append(np.hstack((np.cross(point, normal), normal)))
            rhs.append(-float(normal @ (point - target_point)))
            distances.append(distance)
        if len(rows) < 6:
            break
        solution, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)
        delta = make_transform(_rodrigues(solution[:3]), solution[3:])
        transform = delta @ transform
        current_error = rmse(np.asarray(distances))
        iterations = iteration
        if abs(previous_error - current_error) < tolerance:
            break
        previous_error = current_error

    metrics = evaluate_registration(source, target, transform, threshold=threshold)
    return RegistrationStage(
        transformation=transform,
        fitness=metrics["fitness"],
        inlier_rmse=metrics["rmse"],
        metadata={"iterations": iterations},
    )


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


def _to_o3d(points: np.ndarray):
    o3d = _require_open3d()
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(_ensure_points(points, "points"))
    return point_cloud


def _ensure_points(points: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if len(arr) < 4:
        raise ValueError(f"{name} must contain at least 4 points")
    return arr


def _require_open3d():
    try:
        import open3d as o3d  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Open3D is required for FPFH global registration. Install with "
            "`python -m pip install open3d`."
        ) from exc
    return o3d
