"""Programmatic task API for PointCloud-GeoLab workflows."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pointcloud_geolab.geometry import compute_aabb, compute_obb, pca_analysis
from pointcloud_geolab.io.pointcloud_io import load_point_cloud, save_point_cloud
from pointcloud_geolab.io.visualization import (
    save_error_curve,
    save_point_cloud_projection,
    visualize_point_clouds,
)
from pointcloud_geolab.kdtree import KDTree
from pointcloud_geolab.preprocessing import (
    estimate_normals,
    remove_radius_outliers,
    remove_statistical_outliers,
    voxel_downsample,
)
from pointcloud_geolab.registration import point_to_point_icp
from pointcloud_geolab.segmentation.ransac_plane import ransac_plane_fitting
from pointcloud_geolab.utils.transform import apply_transform, rotation_matrix_from_euler


@dataclass(slots=True)
class TaskResult:
    """JSON-friendly result envelope shared by the API and CLI."""

    task: str
    success: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(
            {
                "task": self.task,
                "success": self.success,
                "metrics": self.metrics,
                "artifacts": self.artifacts,
                "parameters": self.parameters,
                "data": self.data,
                "error": self.error,
            }
        )


def run_icp(
    source: str | Path,
    target: str | Path,
    output_dir: str | Path = "results",
    voxel_size: float = 0.0,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    max_correspondence_distance: float | None = None,
    save_results: bool = False,
    visualize: bool = False,
) -> TaskResult:
    """Run point-to-point ICP and return a structured result."""

    parameters = _parameters(locals())
    try:
        source_points = load_point_cloud(source)
        target_points = load_point_cloud(target)
        if voxel_size > 0:
            source_points = voxel_downsample(source_points, voxel_size)
            target_points = voxel_downsample(target_points, voxel_size)

        result = point_to_point_icp(
            source_points,
            target_points,
            max_iterations=max_iterations,
            tolerance=tolerance,
            max_correspondence_distance=max_correspondence_distance,
        )

        out_dir = Path(output_dir)
        artifacts: dict[str, str] = {}
        if save_results:
            before_path = out_dir / "icp_before.png"
            after_path = out_dir / "icp_after.png"
            curve_path = out_dir / "icp_error_curve.png"
            aligned_path = out_dir / "aligned_source.ply"
            save_point_cloud_projection(
                before_path,
                [source_points, target_points],
                labels=["source", "target"],
                title="ICP Before",
            )
            save_point_cloud_projection(
                after_path,
                [result.aligned_points, target_points],
                labels=["aligned source", "target"],
                title="ICP After",
            )
            save_error_curve(curve_path, result.rmse_history)
            save_point_cloud(aligned_path, result.aligned_points)
            artifacts.update(
                {
                    "before_projection": str(before_path),
                    "after_projection": str(after_path),
                    "rmse_curve": str(curve_path),
                    "aligned_source": str(aligned_path),
                }
            )

        if visualize:
            visualize_point_clouds([result.aligned_points, target_points], window_name="ICP Result")

        task_result = TaskResult(
            task="icp",
            success=True,
            metrics={
                "source_points": len(source_points),
                "target_points": len(target_points),
                "iterations": result.iterations,
                "initial_rmse": result.initial_rmse,
                "final_rmse": result.final_rmse,
                "fitness": result.fitness,
                "converged": result.converged,
            },
            artifacts=artifacts,
            parameters=parameters,
            data={
                "rotation": result.rotation,
                "translation": result.translation,
                "transformation": result.transformation,
                "rmse_history": result.rmse_history,
            },
        )
    except Exception as exc:  # pragma: no cover - exercised through CLI failures
        task_result = TaskResult("icp", False, parameters=parameters, error=str(exc))
    return _finalize_result(task_result, output_dir)


def run_plane_segmentation(
    input_path: str | Path,
    output_dir: str | Path = "results",
    voxel_size: float = 0.0,
    threshold: float = 0.02,
    max_iterations: int = 1000,
    seed: int | None = 7,
    save_results: bool = False,
    visualize: bool = False,
) -> TaskResult:
    """Run RANSAC dominant-plane segmentation."""

    parameters = _parameters(locals())
    try:
        points = load_point_cloud(input_path)
        if voxel_size > 0:
            points = voxel_downsample(points, voxel_size)
        result = ransac_plane_fitting(
            points,
            threshold=threshold,
            max_iterations=max_iterations,
            seed=seed,
        )

        inlier_points = points[result.inliers]
        outlier_points = points[result.outliers]
        out_dir = Path(output_dir)
        artifacts: dict[str, str] = {}
        if save_results:
            projection_path = out_dir / "ransac_plane.png"
            inliers_path = out_dir / "plane_inliers.ply"
            outliers_path = out_dir / "plane_outliers.ply"
            save_point_cloud_projection(
                projection_path,
                [inlier_points, outlier_points],
                labels=["plane inliers", "outliers"],
                title="RANSAC Plane",
            )
            save_point_cloud(inliers_path, inlier_points)
            save_point_cloud(outliers_path, outlier_points)
            artifacts.update(
                {
                    "projection": str(projection_path),
                    "inliers": str(inliers_path),
                    "outliers": str(outliers_path),
                }
            )

        if visualize:
            visualize_point_clouds([inlier_points, outlier_points], window_name="RANSAC Plane")

        task_result = TaskResult(
            task="plane",
            success=True,
            metrics={
                "point_count": len(points),
                "inliers": len(result.inliers),
                "outliers": len(result.outliers),
                "inlier_ratio": result.inlier_ratio,
            },
            artifacts=artifacts,
            parameters=parameters,
            data={"plane_model": result.plane_model},
        )
    except Exception as exc:  # pragma: no cover - exercised through CLI failures
        task_result = TaskResult("plane", False, parameters=parameters, error=str(exc))
    return _finalize_result(task_result, output_dir)


def run_geometry_analysis(
    input_path: str | Path,
    output_dir: str | Path = "results",
    save_results: bool = False,
    visualize: bool = False,
) -> TaskResult:
    """Compute AABB, OBB, and PCA metrics for a point cloud."""

    parameters = _parameters(locals())
    try:
        points = load_point_cloud(input_path)
        aabb = compute_aabb(points)
        obb = compute_obb(points)
        pca = pca_analysis(points)

        artifacts: dict[str, str] = {}
        if save_results:
            projection_path = Path(output_dir) / "obb_visualization.png"
            save_point_cloud_projection(
                projection_path,
                [points, obb.corners],
                labels=["points", "OBB corners"],
                title="PCA-based OBB",
            )
            artifacts["projection"] = str(projection_path)

        if visualize:
            visualize_point_clouds([points], window_name="Geometry Analysis")

        task_result = TaskResult(
            task="geometry",
            success=True,
            metrics={
                "point_count": len(points),
                "center": points.mean(axis=0),
                "aabb_extent": aabb.extent,
                "obb_extent": obb.extent,
                "pca_eigenvalues": pca.eigenvalues,
                "main_direction": pca.eigenvectors[:, 0],
            },
            artifacts=artifacts,
            parameters=parameters,
            data={
                "aabb": {
                    "min_bound": aabb.min_bound,
                    "max_bound": aabb.max_bound,
                    "center": aabb.center,
                },
                "obb": {
                    "center": obb.center,
                    "rotation": obb.rotation,
                    "corners": obb.corners,
                },
            },
        )
    except Exception as exc:  # pragma: no cover - exercised through CLI failures
        task_result = TaskResult("geometry", False, parameters=parameters, error=str(exc))
    return _finalize_result(task_result, output_dir)


def run_preprocessing(
    input_path: str | Path,
    output: str | Path | None = None,
    output_dir: str | Path = "results",
    voxel_size: float = 0.0,
    statistical_nb_neighbors: int = 16,
    statistical_std_ratio: float = 2.0,
    radius: float = 0.0,
    min_neighbors: int = 4,
    estimate_normals_flag: bool = False,
    save_results: bool = False,
    visualize: bool = False,
) -> TaskResult:
    """Run the preprocessing pipeline."""

    parameters = _parameters(locals())
    parameters["estimate_normals"] = parameters.pop("estimate_normals_flag")
    try:
        points = load_point_cloud(input_path)
        original_count = len(points)

        if voxel_size > 0:
            points = voxel_downsample(points, voxel_size)
        after_downsample = len(points)

        points, statistical_inliers = remove_statistical_outliers(
            points,
            nb_neighbors=statistical_nb_neighbors,
            std_ratio=statistical_std_ratio,
        )
        after_statistical = len(points)

        if radius > 0:
            points, _ = remove_radius_outliers(
                points,
                radius=radius,
                min_neighbors=min_neighbors,
            )
        after_radius = len(points)

        normals = estimate_normals(points) if estimate_normals_flag else None

        artifacts: dict[str, str] = {}
        if output:
            save_point_cloud(output, points)
            artifacts["output"] = str(output)

        if save_results:
            projection_path = Path(output_dir) / "preprocessing.png"
            save_point_cloud_projection(
                projection_path,
                [points],
                labels=["preprocessed"],
                title="Preprocessed Point Cloud",
            )
            artifacts["projection"] = str(projection_path)

        if visualize:
            visualize_point_clouds([points], window_name="Preprocessed Point Cloud")

        task_result = TaskResult(
            task="preprocess",
            success=True,
            metrics={
                "original_points": original_count,
                "after_voxel_downsample": after_downsample,
                "after_statistical_filter": after_statistical,
                "after_radius_filter": after_radius,
                "kept_statistical_inliers": len(statistical_inliers),
                "estimated_normals": normals is not None,
                "final_points": len(points),
            },
            artifacts=artifacts,
            parameters=parameters,
        )
    except Exception as exc:  # pragma: no cover - exercised through CLI failures
        task_result = TaskResult("preprocess", False, parameters=parameters, error=str(exc))
    return _finalize_result(task_result, output_dir)


def run_benchmark(
    benchmark: str,
    output_dir: str | Path = "results",
    quick: bool = True,
    full: bool = False,
    save_json: str | Path | None = None,
    save_md: str | Path | None = None,
    seed: int = 42,
    queries: int = 100,
    points: list[int] | None = None,
) -> TaskResult:
    """Run a built-in benchmark suite."""

    parameters = _parameters(locals())
    try:
        if benchmark == "kdtree":
            rows = _benchmark_kdtree(
                seed=seed,
                queries=queries,
                point_counts=points
                or ([1000, 5000, 10000, 50000] if full else [1000, 5000, 10000]),
            )
            markdown = _format_kdtree_table(rows)
        elif benchmark == "icp":
            rows = _benchmark_icp(seed=seed, full=full, quick=quick)
            markdown = _format_icp_table(rows)
        else:
            raise ValueError("benchmark must be one of: kdtree, icp")

        artifacts: dict[str, str] = {}
        if save_json:
            json_path = Path(save_json)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(_json_ready(rows), indent=2) + "\n", encoding="utf-8")
            artifacts["benchmark_json"] = str(json_path)
        if save_md:
            md_path = Path(save_md)
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(markdown + "\n", encoding="utf-8")
            artifacts["benchmark_markdown"] = str(md_path)

        task_result = TaskResult(
            task=f"benchmark:{benchmark}",
            success=True,
            metrics={
                "benchmark": benchmark,
                "cases": len(rows),
                "quick": quick,
                "full": full,
                "all_correct": all(bool(row.get("correct", True)) for row in rows),
            },
            artifacts=artifacts,
            parameters=parameters,
            data={"rows": rows, "markdown": markdown},
        )
    except Exception as exc:  # pragma: no cover - exercised through CLI failures
        task_result = TaskResult(
            f"benchmark:{benchmark}",
            False,
            parameters=parameters,
            error=str(exc),
        )
    return _finalize_result(task_result, output_dir)


def _benchmark_kdtree(seed: int, queries: int, point_counts: list[int]) -> list[dict[str, Any]]:
    rows = []
    for i, points_count in enumerate(point_counts):
        rng = np.random.default_rng(seed + i)
        points = rng.random((points_count, 3))
        query_points = rng.random((queries, 3))

        start = time.perf_counter()
        tree = KDTree(points)
        build_time = time.perf_counter() - start

        start = time.perf_counter()
        brute_indices = []
        for query in query_points:
            distances = np.linalg.norm(points - query, axis=1)
            brute_indices.append(int(np.argmin(distances)))
        brute_time = time.perf_counter() - start

        start = time.perf_counter()
        kd_indices = [tree.nearest_neighbor(query)[0] for query in query_points]
        kd_time = time.perf_counter() - start

        row: dict[str, Any] = {
            "points": points_count,
            "queries": queries,
            "build_time": build_time,
            "brute_time": brute_time,
            "kd_time": kd_time,
            "speedup": brute_time / kd_time if kd_time > 0 else float("inf"),
            "correct": brute_indices == kd_indices,
        }
        open3d_time = _optional_open3d_kdtree_time(points, query_points)
        if open3d_time is not None:
            row["open3d_time"] = open3d_time
        rows.append(row)
    return rows


def _optional_open3d_kdtree_time(points: np.ndarray, queries: np.ndarray) -> float | None:
    try:
        import open3d as o3d  # type: ignore
    except ImportError:
        return None

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    start = time.perf_counter()
    tree = o3d.geometry.KDTreeFlann(point_cloud)
    for query in queries:
        tree.search_knn_vector_3d(query, 1)
    return time.perf_counter() - start


def _benchmark_icp(seed: int, full: bool, quick: bool) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    target_count = 500 if full else 250
    target = rng.normal(size=(target_count, 3))
    if full:
        rotations = [5.0, 15.0, 30.0, 60.0]
        translations = [0.1, 0.3, 0.5, 1.0]
        noises = [0.0, 0.01, 0.03]
    elif quick:
        rotations = [5.0, 15.0]
        translations = [0.1, 0.3]
        noises = [0.0]
    else:
        rotations = [5.0, 15.0, 30.0]
        translations = [0.1, 0.3]
        noises = [0.0, 0.01]

    rows = []
    case_seed = seed
    for rotation_degrees in rotations:
        for translation_magnitude in translations:
            for noise in noises:
                rows.append(
                    _benchmark_icp_case(
                        target,
                        rotation_degrees,
                        translation_magnitude,
                        noise,
                        case_seed,
                    )
                )
                case_seed += 1
    return rows


def _benchmark_icp_case(
    target: np.ndarray,
    rotation_degrees: float,
    translation_magnitude: float,
    noise: float,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    rotation = rotation_matrix_from_euler(
        np.radians(rotation_degrees) * 0.6,
        -np.radians(rotation_degrees) * 0.35,
        np.radians(rotation_degrees),
    )
    direction = np.asarray([1.0, -0.45, 0.32], dtype=float)
    direction /= np.linalg.norm(direction)
    translation = direction * translation_magnitude
    source = apply_transform(target, rotation, translation)
    if noise > 0:
        source = source + rng.normal(scale=noise, size=source.shape)

    result = point_to_point_icp(source, target, max_iterations=80, tolerance=1e-7)
    expected_rotation = rotation.T
    expected_translation = -rotation.T @ translation
    return {
        "rotation_degrees": rotation_degrees,
        "translation": translation_magnitude,
        "noise": noise,
        "converged": result.converged,
        "iterations": result.iterations,
        "final_rmse": result.final_rmse,
        "rotation_error_degrees": _rotation_error_degrees(result.rotation, expected_rotation),
        "translation_error": float(np.linalg.norm(result.translation - expected_translation)),
    }


def _rotation_error_degrees(estimated: np.ndarray, expected: np.ndarray) -> float:
    delta = estimated @ expected.T
    cos_angle = (np.trace(delta) - 1.0) / 2.0
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def _format_kdtree_table(rows: list[dict[str, Any]]) -> str:
    has_open3d = any("open3d_time" in row for row in rows)
    if has_open3d:
        lines = [
            "| Points | Queries | Build Time (s) | Brute Force (s) | "
            "KD-Tree (s) | Open3D (s) | Speedup | Correct |",
            "|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    else:
        lines = [
            "| Points | Queries | Build Time (s) | Brute Force (s) | "
            "KD-Tree (s) | Speedup | Correct |",
            "|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    for row in rows:
        if has_open3d:
            lines.append(
                "| {points:,} | {queries:,} | {build_time:.4f} | {brute_time:.4f} | "
                "{kd_time:.4f} | {open3d_time:.4f} | {speedup:.2f}x | {correct} |".format(
                    points=row["points"],
                    queries=row["queries"],
                    build_time=row["build_time"],
                    brute_time=row["brute_time"],
                    kd_time=row["kd_time"],
                    open3d_time=row.get("open3d_time", float("nan")),
                    speedup=row["speedup"],
                    correct="yes" if row["correct"] else "no",
                )
            )
        else:
            lines.append(
                "| {points:,} | {queries:,} | {build_time:.4f} | {brute_time:.4f} | "
                "{kd_time:.4f} | {speedup:.2f}x | {correct} |".format(
                    points=row["points"],
                    queries=row["queries"],
                    build_time=row["build_time"],
                    brute_time=row["brute_time"],
                    kd_time=row["kd_time"],
                    speedup=row["speedup"],
                    correct="yes" if row["correct"] else "no",
                )
            )
    return "\n".join(lines)


def _format_icp_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Rotation (deg) | Translation | Noise | Converged | Iterations | "
        "Final RMSE | Rotation Error (deg) | Translation Error |",
        "|---:|---:|---:|:---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {rotation_degrees:.0f} | {translation:.2f} | {noise:.2f} | {converged} | "
            "{iterations} | {final_rmse:.6f} | {rotation_error_degrees:.4f} | "
            "{translation_error:.6f} |".format(
                rotation_degrees=row["rotation_degrees"],
                translation=row["translation"],
                noise=row["noise"],
                converged="yes" if row["converged"] else "no",
                iterations=row["iterations"],
                final_rmse=row["final_rmse"],
                rotation_error_degrees=row["rotation_error_degrees"],
                translation_error=row["translation_error"],
            )
        )
    return "\n".join(lines)


def _finalize_result(result: TaskResult, output_dir: str | Path) -> TaskResult:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    result.artifacts["metrics_json"] = str(metrics_path)
    metrics_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return result


def _parameters(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if key not in {"self"}}


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value
