"""Programmatic task API for PointCloud-GeoLab workflows."""

from __future__ import annotations

import json
import time
from csv import DictWriter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pointcloud_geolab.datasets import make_sphere
from pointcloud_geolab.geometry import compute_aabb, compute_obb, pca_analysis
from pointcloud_geolab.geometry.primitive_fitting import ransac_fit_primitive
from pointcloud_geolab.io.pointcloud_io import load_point_cloud, save_point_cloud
from pointcloud_geolab.io.visualization import (
    save_error_curve,
    save_point_cloud_projection,
    visualize_point_clouds,
)
from pointcloud_geolab.kdtree import KDTree
from pointcloud_geolab.preprocessing import (
    crop_by_aabb,
    estimate_normals,
    farthest_point_sample,
    normalize_point_cloud,
    random_sample,
    remove_radius_outliers,
    remove_statistical_outliers,
    voxel_downsample,
)
from pointcloud_geolab.registration import evaluate_registration, point_to_point_icp
from pointcloud_geolab.registration.global_registration import register_fpfh_ransac_icp
from pointcloud_geolab.segmentation import (
    cluster_statistics,
    dbscan_clustering,
    euclidean_clustering,
    region_growing_segmentation,
)
from pointcloud_geolab.segmentation.ransac_plane import ransac_plane_fitting
from pointcloud_geolab.utils.transform import (
    apply_homogeneous_transform,
    apply_transform,
    rotation_matrix_from_euler,
)
from pointcloud_geolab.visualization import (
    export_point_cloud_html,
    export_registration_html,
    label_colors,
    save_colored_point_cloud,
    visualize_inliers_outliers,
)


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
    normalize: bool = False,
    crop_min: list[float] | np.ndarray | None = None,
    crop_max: list[float] | np.ndarray | None = None,
    sample_count: int | None = None,
    sample_method: str = "random",
    seed: int | None = None,
    save_results: bool = False,
    visualize: bool = False,
) -> TaskResult:
    """Run the preprocessing pipeline."""

    parameters = _parameters(locals())
    parameters["estimate_normals"] = parameters.pop("estimate_normals_flag")
    try:
        points = load_point_cloud(input_path)
        original_count = len(points)

        if crop_min is not None or crop_max is not None:
            if crop_min is None or crop_max is None:
                raise ValueError("crop_min and crop_max must be provided together")
            points, _ = crop_by_aabb(points, crop_min, crop_max)
        after_crop = len(points)

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

        if sample_count is not None:
            if sample_method == "random":
                points, _ = random_sample(points, sample_count, random_state=seed)
            elif sample_method == "farthest":
                points, _ = farthest_point_sample(points, sample_count, random_state=seed)
            else:
                raise ValueError("sample_method must be 'random' or 'farthest'")
        after_sample = len(points)

        normalization_center = None
        normalization_scale = None
        if normalize:
            points, normalization_center, normalization_scale = normalize_point_cloud(points)

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
                "after_crop": after_crop,
                "after_voxel_downsample": after_downsample,
                "after_statistical_filter": after_statistical,
                "after_radius_filter": after_radius,
                "after_sampling": after_sample,
                "kept_statistical_inliers": len(statistical_inliers),
                "estimated_normals": normals is not None,
                "final_points": len(points),
            },
            artifacts=artifacts,
            parameters=parameters,
            data={
                "normalization_center": normalization_center,
                "normalization_scale": normalization_scale,
            },
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
        elif benchmark == "ransac":
            rows = _benchmark_ransac(seed=seed, quick=quick)
            markdown = _format_generic_table(rows)
        elif benchmark == "registration":
            rows = _benchmark_registration(seed=seed, quick=quick)
            markdown = _format_generic_table(rows)
        elif benchmark == "all":
            rows = []
            for suite in ["kdtree", "icp", "ransac", "registration"]:
                suite_result = run_benchmark(
                    suite,
                    output_dir=Path(output_dir) / suite,
                    quick=quick,
                    full=full,
                    seed=seed,
                    queries=queries,
                    points=points,
                )
                rows.extend(
                    {"suite": suite, **row}
                    for row in suite_result.to_dict().get("data", {}).get("rows", [])
                )
            markdown = _format_generic_table(rows)
        else:
            raise ValueError("benchmark must be one of: kdtree, icp, ransac, registration, all")

        artifacts: dict[str, str] = {}
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{benchmark}_benchmark.csv"
        md_default_path = out_dir / f"{benchmark}_benchmark.md"
        json_default_path = out_dir / f"{benchmark}_benchmark.json"
        png_path = out_dir / f"{benchmark}_benchmark.png"
        _write_csv(csv_path, rows)
        md_default_path.write_text(markdown + "\n", encoding="utf-8")
        json_default_path.write_text(
            json.dumps({"benchmark": benchmark, "rows": _json_ready(rows)}, indent=2) + "\n",
            encoding="utf-8",
        )
        _save_benchmark_plot(png_path, benchmark, rows)
        artifacts.update(
            {
                "benchmark_csv": str(csv_path),
                "benchmark_markdown": str(md_default_path),
                "benchmark_json": str(json_default_path),
                "benchmark_plot": str(png_path),
            }
        )
        if save_json:
            json_path = Path(save_json)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(_json_ready(rows), indent=2) + "\n", encoding="utf-8")
            artifacts["benchmark_json_custom"] = str(json_path)
        if save_md:
            md_path = Path(save_md)
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(markdown + "\n", encoding="utf-8")
            artifacts["benchmark_markdown_custom"] = str(md_path)

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


def run_global_registration(
    source: str | Path,
    target: str | Path,
    output: str | Path | None = None,
    save_transform: str | Path | None = None,
    output_dir: str | Path = "outputs/registration",
    voxel_size: float = 0.05,
    method: str = "fpfh_ransac_icp",
    icp_method: str = "point_to_point",
    threshold: float | None = None,
    seed: int | None = 7,
    save_results: bool = False,
    export_html: str | Path | None = None,
) -> TaskResult:
    """Run feature-based global registration and ICP refinement."""

    parameters = _parameters(locals())
    try:
        if method != "fpfh_ransac_icp":
            raise ValueError("only method='fpfh_ransac_icp' is currently supported")
        source_points = load_point_cloud(source)
        target_points = load_point_cloud(target)
        result = register_fpfh_ransac_icp(
            source_points,
            target_points,
            voxel_size=voxel_size,
            icp_method=icp_method,
            threshold=threshold,
            seed=seed,
        )
        eval_threshold = threshold if threshold is not None else voxel_size * 1.5
        metrics = evaluate_registration(
            source_points,
            target_points,
            result.refined_transform,
            threshold=eval_threshold,
        )
        artifacts: dict[str, str] = {}
        aligned = apply_homogeneous_transform(source_points, result.refined_transform)
        if output:
            save_point_cloud(output, aligned)
            artifacts["aligned_source"] = str(output)
        if save_transform:
            transform_path = Path(save_transform)
            transform_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(transform_path, result.refined_transform, fmt="%.10f")
            artifacts["transform"] = str(transform_path)
        if save_results:
            out_dir = Path(output_dir)
            before_path = out_dir / "registration_before.png"
            coarse_path = out_dir / "registration_coarse.png"
            refined_path = out_dir / "registration_refined.png"
            save_point_cloud_projection(
                before_path,
                [source_points, target_points],
                labels=["source", "target"],
                title="Registration Before",
            )
            save_point_cloud_projection(
                coarse_path,
                [
                    apply_homogeneous_transform(source_points, result.initial_transform),
                    target_points,
                ],
                labels=["coarse source", "target"],
                title="FPFH RANSAC Coarse Registration",
            )
            save_point_cloud_projection(
                refined_path,
                [aligned, target_points],
                labels=["refined source", "target"],
                title="ICP Refined Registration",
            )
            artifacts.update(
                {
                    "before_projection": str(before_path),
                    "coarse_projection": str(coarse_path),
                    "refined_projection": str(refined_path),
                }
            )
        if export_html:
            export_registration_html(
                source_points,
                target_points,
                result.refined_transform,
                export_html,
            )
            artifacts["html"] = str(export_html)
        task_result = TaskResult(
            task="register",
            success=True,
            metrics={
                "coarse_fitness": result.coarse.fitness,
                "coarse_inlier_rmse": result.coarse.inlier_rmse,
                "refined_fitness": result.refined.fitness,
                "refined_inlier_rmse": result.refined.inlier_rmse,
                "final_rmse": metrics["rmse"],
                "final_fitness": metrics["fitness"],
                "source_downsampled": result.source_downsampled,
                "target_downsampled": result.target_downsampled,
            },
            artifacts=artifacts,
            parameters=parameters,
            data={
                "initial_transform": result.initial_transform,
                "refined_transform": result.refined_transform,
                "method": result.method,
            },
        )
    except Exception as exc:  # pragma: no cover - exercised through CLI failures
        task_result = TaskResult("register", False, parameters=parameters, error=str(exc))
    return _finalize_result(task_result, output_dir)


def run_primitive_fitting(
    input_path: str | Path,
    model: str,
    output: str | Path | None = None,
    output_dir: str | Path = "outputs/primitives",
    threshold: float = 0.02,
    max_iterations: int = 1000,
    min_inliers: int = 0,
    seed: int | None = 7,
    save_results: bool = False,
    export_html: str | Path | None = None,
) -> TaskResult:
    """Fit a plane, sphere, or cylinder primitive with RANSAC."""

    parameters = _parameters(locals())
    try:
        points = load_point_cloud(input_path)
        result = ransac_fit_primitive(
            points,
            model_type=model,
            threshold=threshold,
            max_iterations=max_iterations,
            min_inliers=min_inliers,
            random_state=seed,
        )
        inlier_mask = np.zeros(len(points), dtype=bool)
        inlier_mask[result.inlier_indices] = True
        artifacts: dict[str, str] = {}
        if output:
            save_point_cloud(output, points[result.inlier_indices])
            artifacts["inliers"] = str(output)
        if save_results:
            projection_path = Path(output_dir) / f"{model}_ransac.png"
            save_point_cloud_projection(
                projection_path,
                [points[result.inlier_indices], points[result.outlier_indices]],
                labels=["inliers", "outliers"],
                title=f"RANSAC {model.title()} Fitting",
            )
            artifacts["projection"] = str(projection_path)
        if export_html:
            visualize_inliers_outliers(points, inlier_mask, output_path=export_html)
            artifacts["html"] = str(export_html)
        task_result = TaskResult(
            task="fit-primitive",
            success=True,
            metrics={
                "model": model,
                "point_count": len(points),
                "inliers": len(result.inlier_indices),
                "outliers": len(result.outlier_indices),
                "inlier_ratio": result.score,
                "residual_mean": result.residual_mean,
                "iterations": result.iterations,
            },
            artifacts=artifacts,
            parameters=parameters,
            data={"model_params": result.model.get_params()},
        )
    except Exception as exc:  # pragma: no cover - exercised through CLI failures
        task_result = TaskResult("fit-primitive", False, parameters=parameters, error=str(exc))
    return _finalize_result(task_result, output_dir)


def run_segmentation(
    input_path: str | Path,
    output: str | Path | None = None,
    output_dir: str | Path = "outputs/segmentation",
    method: str = "dbscan",
    eps: float = 0.05,
    min_points: int = 20,
    tolerance: float | None = None,
    radius: float = 0.1,
    angle_threshold: float = 25.0,
    export_html: str | Path | None = None,
) -> TaskResult:
    """Segment a point cloud with DBSCAN, Euclidean clustering, or region growing."""

    parameters = _parameters(locals())
    try:
        points = load_point_cloud(input_path)
        if method == "dbscan":
            result = dbscan_clustering(points, eps=eps, min_points=min_points)
        elif method == "euclidean":
            result = euclidean_clustering(
                points,
                tolerance=tolerance if tolerance is not None else eps,
                min_points=min_points,
            )
        elif method == "region_growing":
            result = region_growing_segmentation(
                points,
                radius=radius,
                angle_threshold_degrees=angle_threshold,
                min_cluster_size=min_points,
            )
        else:
            raise ValueError("method must be one of: dbscan, euclidean, region_growing")

        artifacts: dict[str, str] = {}
        if output:
            save_colored_point_cloud(output, points, result.labels)
            artifacts["colored_point_cloud"] = str(output)
        if export_html:
            export_point_cloud_html(
                points,
                label_colors(result.labels),
                export_html,
                title="Segmentation",
            )
            artifacts["html"] = str(export_html)

        task_result = TaskResult(
            task="segment",
            success=True,
            metrics={
                "point_count": len(points),
                "cluster_count": result.cluster_count,
                "noise_points": len(result.noise_indices),
            },
            artifacts=artifacts,
            parameters=parameters,
            data={
                "labels": result.labels,
                "clusters": cluster_statistics(points, result.labels),
            },
        )
    except Exception as exc:  # pragma: no cover - exercised through CLI failures
        task_result = TaskResult("segment", False, parameters=parameters, error=str(exc))
    return _finalize_result(task_result, output_dir)


def run_visualization(
    input_path: str | Path,
    output: str | Path,
    mode: str = "pointcloud",
    labels_path: str | Path | None = None,
    source: str | Path | None = None,
    target: str | Path | None = None,
    transform_path: str | Path | None = None,
    output_dir: str | Path = "outputs/visualization",
) -> TaskResult:
    """Export a point cloud or registration visualization to HTML."""

    parameters = _parameters(locals())
    try:
        artifacts: dict[str, str] = {}
        if mode == "registration":
            if source is None or target is None:
                raise ValueError("source and target are required for registration visualization")
            src = load_point_cloud(source)
            tgt = load_point_cloud(target)
            transform = np.loadtxt(transform_path) if transform_path else None
            export_registration_html(src, tgt, transform, output)
        else:
            points = load_point_cloud(input_path)
            colors = None
            if mode == "clusters":
                if labels_path is None:
                    raise ValueError("labels_path is required for cluster visualization")
                labels = np.loadtxt(labels_path, dtype=int)
                colors = label_colors(labels)
            export_point_cloud_html(points, colors, output, title=mode.title())
        artifacts["html"] = str(output)
        task_result = TaskResult(
            task="visualize",
            success=True,
            metrics={"mode": mode},
            artifacts=artifacts,
            parameters=parameters,
        )
    except Exception as exc:  # pragma: no cover - exercised through CLI failures
        task_result = TaskResult("visualize", False, parameters=parameters, error=str(exc))
    return _finalize_result(task_result, output_dir)


def run_train_pointnet(
    output: str | Path,
    output_dir: str | Path = "outputs/ml",
    epochs: int = 5,
    batch_size: int = 16,
    samples_per_class: int = 24,
    points_per_sample: int = 128,
    seed: int = 7,
) -> TaskResult:
    """Train the optional synthetic PointNet classifier."""

    parameters = _parameters(locals())
    try:
        from pointcloud_geolab.ml.train_pointnet import train_pointnet

        metrics = train_pointnet(
            output,
            epochs=epochs,
            batch_size=batch_size,
            samples_per_class=samples_per_class,
            points_per_sample=points_per_sample,
            seed=seed,
        )
        task_result = TaskResult(
            task="train-pointnet",
            success=True,
            metrics=metrics,
            artifacts={"model": str(output)},
            parameters=parameters,
        )
    except Exception as exc:  # pragma: no cover - optional dependency path
        task_result = TaskResult("train-pointnet", False, parameters=parameters, error=str(exc))
    return _finalize_result(task_result, output_dir)


def run_infer_pointnet(
    model: str | Path,
    input_path: str | Path,
    output_dir: str | Path = "outputs/ml",
    points_per_sample: int | None = None,
) -> TaskResult:
    """Run optional PointNet inference for one point cloud."""

    parameters = _parameters(locals())
    try:
        from pointcloud_geolab.ml.infer_pointnet import infer_pointnet

        prediction = infer_pointnet(model, input_path, points_per_sample=points_per_sample)
        task_result = TaskResult(
            task="infer-pointnet",
            success=True,
            metrics=prediction,
            parameters=parameters,
        )
    except Exception as exc:  # pragma: no cover - optional dependency path
        task_result = TaskResult("infer-pointnet", False, parameters=parameters, error=str(exc))
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
        scipy_time = _optional_scipy_ckdtree_time(points, query_points)
        if scipy_time is not None:
            row["scipy_ckdtree_time"] = scipy_time
        sklearn_time = _optional_sklearn_kdtree_time(points, query_points)
        if sklearn_time is not None:
            row["sklearn_kdtree_time"] = sklearn_time
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


def _optional_scipy_ckdtree_time(points: np.ndarray, queries: np.ndarray) -> float | None:
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except ImportError:
        return None
    start = time.perf_counter()
    tree = cKDTree(points)
    tree.query(queries, k=1)
    return time.perf_counter() - start


def _optional_sklearn_kdtree_time(points: np.ndarray, queries: np.ndarray) -> float | None:
    try:
        from sklearn.neighbors import KDTree as SklearnKDTree  # type: ignore
    except ImportError:
        return None
    start = time.perf_counter()
    tree = SklearnKDTree(points)
    tree.query(queries, k=1)
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


def _benchmark_ransac(seed: int, quick: bool) -> list[dict[str, Any]]:
    ratios = [0.0, 0.3, 0.6] if quick else [0.0, 0.1, 0.3, 0.5, 0.7]
    rows = []
    for i, ratio in enumerate(ratios):
        rng = np.random.default_rng(seed + i)
        inlier_count = 160
        xy = rng.uniform(-1.0, 1.0, size=(inlier_count, 2))
        z = 0.2 * xy[:, 0] - 0.1 * xy[:, 1] + 0.3 + rng.normal(0, 0.003, inlier_count)
        inliers = np.column_stack([xy, z])
        outlier_count = int(inlier_count * ratio / max(1.0 - ratio, 1e-6))
        outliers = rng.uniform(-1.5, 1.5, size=(outlier_count, 3))
        points = np.vstack([inliers, outliers])
        truth_mask = np.zeros(len(points), dtype=bool)
        truth_mask[:inlier_count] = True
        result = ransac_fit_primitive(
            points,
            "plane",
            threshold=0.02,
            max_iterations=800,
            random_state=seed + i,
        )
        predicted = np.zeros(len(points), dtype=bool)
        predicted[result.inlier_indices] = True
        true_positive = int(np.sum(predicted & truth_mask))
        precision = true_positive / max(int(np.sum(predicted)), 1)
        recall = true_positive / max(int(np.sum(truth_mask)), 1)
        rows.append(
            {
                "outlier_ratio": ratio,
                "inliers": len(result.inlier_indices),
                "precision": precision,
                "recall": recall,
                "residual_mean": result.residual_mean,
            }
        )
    return rows


def _benchmark_registration(seed: int, quick: bool) -> list[dict[str, Any]]:
    angles = [5.0, 35.0] if quick else [5.0, 20.0, 45.0, 70.0]
    rng = np.random.default_rng(seed)
    target = np.vstack(
        [
            make_sphere(220, radius=0.7, random_state=seed),
            rng.normal(scale=0.08, size=(40, 3)) + np.asarray([0.45, 0.2, 0.1]),
        ]
    )
    rows = []
    for angle in angles:
        rotation = rotation_matrix_from_euler(0.0, 0.0, np.radians(angle))
        translation = np.asarray([0.25, -0.1, 0.08])
        source = apply_transform(target, rotation, translation)
        icp = point_to_point_icp(source, target, max_iterations=60, tolerance=1e-7)
        rows.append(
            {
                "method": "icp",
                "initial_angle_degrees": angle,
                "success": icp.final_rmse < 0.05,
                "rmse": icp.final_rmse,
                "fitness": icp.fitness,
            }
        )
        try:
            global_result = register_fpfh_ransac_icp(
                source,
                target,
                voxel_size=0.2,
                icp_method="point_to_point",
                seed=seed,
            )
            metrics = evaluate_registration(source, target, global_result.refined_transform, 0.3)
            rows.append(
                {
                    "method": "fpfh_ransac_icp",
                    "initial_angle_degrees": angle,
                    "success": metrics["rmse"] < 0.08,
                    "rmse": metrics["rmse"],
                    "fitness": metrics["fitness"],
                }
            )
        except Exception:
            rows.append(
                {
                    "method": "fpfh_ransac_icp",
                    "initial_angle_degrees": angle,
                    "success": False,
                    "rmse": float("nan"),
                    "fitness": 0.0,
                }
            )
    return rows


def _rotation_error_degrees(estimated: np.ndarray, expected: np.ndarray) -> float:
    delta = estimated @ expected.T
    cos_angle = (np.trace(delta) - 1.0) / 2.0
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def _format_kdtree_table(rows: list[dict[str, Any]]) -> str:
    has_open3d = any("open3d_time" in row for row in rows)
    has_scipy = any("scipy_ckdtree_time" in row for row in rows)
    has_sklearn = any("sklearn_kdtree_time" in row for row in rows)
    headers = ["Points", "Queries", "Build Time (s)", "Brute Force (s)", "KD-Tree (s)"]
    if has_open3d:
        headers.append("Open3D (s)")
    if has_scipy:
        headers.append("SciPy cKDTree (s)")
    if has_sklearn:
        headers.append("sklearn KDTree (s)")
    headers.extend(["Speedup", "Correct"])
    aligns = ["---:" for _ in headers[:-1]] + [":---:"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for row in rows:
        values = [
            f"{row['points']:,}",
            f"{row['queries']:,}",
            f"{row['build_time']:.4f}",
            f"{row['brute_time']:.4f}",
            f"{row['kd_time']:.4f}",
        ]
        if has_open3d:
            values.append(_format_optional_time(row, "open3d_time"))
        if has_scipy:
            values.append(_format_optional_time(row, "scipy_ckdtree_time"))
        if has_sklearn:
            values.append(_format_optional_time(row, "sklearn_kdtree_time"))
        values.extend([f"{row['speedup']:.2f}x", "yes" if row["correct"] else "no"])
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_optional_time(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    return "" if value is None else f"{value:.4f}"


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


def _format_generic_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    headers = sorted({key for row in rows for key in row.keys()})
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_ready(row))


def _save_benchmark_plot(path: Path, benchmark: str, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    if benchmark == "kdtree":
        xs = [row["points"] for row in rows]
        ax.plot(xs, [row["kd_time"] for row in rows], marker="o", label="custom KDTree")
        ax.plot(xs, [row["brute_time"] for row in rows], marker="o", label="brute force")
        if any("open3d_time" in row for row in rows):
            ax.plot(
                xs,
                [row.get("open3d_time", np.nan) for row in rows],
                marker="o",
                label="Open3D",
            )
        if any("scipy_ckdtree_time" in row for row in rows):
            ax.plot(
                xs,
                [row.get("scipy_ckdtree_time", np.nan) for row in rows],
                marker="o",
                label="SciPy cKDTree",
            )
        if any("sklearn_kdtree_time" in row for row in rows):
            ax.plot(
                xs,
                [row.get("sklearn_kdtree_time", np.nan) for row in rows],
                marker="o",
                label="sklearn KDTree",
            )
        ax.set_xlabel("Points")
        ax.set_ylabel("Query Time (s)")
    elif benchmark == "icp":
        xs = list(range(len(rows)))
        ax.plot(xs, [row["final_rmse"] for row in rows], marker="o")
        ax.set_xlabel("Case")
        ax.set_ylabel("Final RMSE")
    elif benchmark == "ransac":
        xs = [row["outlier_ratio"] for row in rows]
        ax.plot(xs, [row["precision"] for row in rows], marker="o", label="precision")
        ax.plot(xs, [row["recall"] for row in rows], marker="o", label="recall")
        ax.set_xlabel("Outlier Ratio")
        ax.set_ylabel("Score")
    elif benchmark == "registration":
        for method in sorted({row["method"] for row in rows}):
            subset = [row for row in rows if row["method"] == method]
            ax.plot(
                [row["initial_angle_degrees"] for row in subset],
                [row["rmse"] for row in subset],
                marker="o",
                label=method,
            )
        ax.set_xlabel("Initial Rotation (deg)")
        ax.set_ylabel("RMSE")
    else:
        xs = list(range(len(rows)))
        ys = [float(row.get("rmse", row.get("residual_mean", 0.0)) or 0.0) for row in rows]
        ax.plot(xs, ys, marker="o")
        ax.set_xlabel("Case")
        ax.set_ylabel("Metric")
    ax.set_title(f"{benchmark} benchmark")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
