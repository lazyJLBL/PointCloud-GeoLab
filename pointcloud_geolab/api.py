"""Programmatic task API for PointCloud-GeoLab workflows."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from csv import DictWriter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pointcloud_geolab.datasets import make_sphere
from pointcloud_geolab.features import compute_local_geometric_descriptors, detect_iss_keypoints
from pointcloud_geolab.geometry import compute_aabb, compute_obb, pca_analysis
from pointcloud_geolab.geometry.primitive_fitting import (
    PlaneModel,
    extract_primitives,
    ransac_fit_primitive,
)
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
from pointcloud_geolab.reconstruction import reconstruct_surface
from pointcloud_geolab.registration import (
    evaluate_registration,
    generalized_icp,
    multiscale_icp,
    point_to_point_icp,
    robust_icp,
)
from pointcloud_geolab.registration.feature_registration import register_iss_descriptor_ransac_icp
from pointcloud_geolab.registration.global_registration import register_fpfh_ransac_icp
from pointcloud_geolab.segmentation import (
    cluster_statistics,
    dbscan_clustering,
    euclidean_clustering,
    ground_object_segmentation,
    region_growing_segmentation,
    write_cluster_report,
)
from pointcloud_geolab.segmentation.ransac_plane import ransac_plane_fitting
from pointcloud_geolab.spatial import VoxelHashGrid
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


def run_robust_icp(
    source: str | Path,
    target: str | Path,
    output_dir: str | Path = "outputs/registration",
    method: str = "point_to_point",
    robust_kernel: str = "huber",
    trim_ratio: float = 0.8,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    max_correspondence_distance: float | None = None,
) -> TaskResult:
    """Run robust ICP and return JSON-friendly diagnostics."""

    parameters = _parameters(locals())
    try:
        source_points = load_point_cloud(source)
        target_points = load_point_cloud(target)
        result = robust_icp(
            source_points,
            target_points,
            method=method,
            robust_kernel=robust_kernel,
            trim_ratio=trim_ratio,
            max_iterations=max_iterations,
            tolerance=tolerance,
            max_correspondence_distance=max_correspondence_distance,
        )
        task_result = TaskResult(
            task="robust-icp",
            success=True,
            metrics={
                "iterations": result.iterations,
                "final_rmse": result.final_rmse,
                "fitness": result.fitness,
                "converged": result.converged,
            },
            parameters=parameters,
            data={
                "transformation": result.transformation,
                "rmse_history": result.rmse_history,
                "diagnostics": result.diagnostics,
            },
        )
    except Exception as exc:  # pragma: no cover - CLI failure path
        task_result = TaskResult("robust-icp", False, parameters=parameters, error=str(exc))
    return _finalize_result(task_result, output_dir)


def run_multiscale_icp(
    source: str | Path,
    target: str | Path,
    output_dir: str | Path = "outputs/registration",
    voxel_sizes: list[float] | None = None,
    max_iterations_per_level: int = 30,
    method: str = "point_to_point",
    robust_kernel: str = "none",
    trim_ratio: float = 1.0,
    output: str | Path | None = None,
    save_diagnostics: str | Path | None = None,
) -> TaskResult:
    """Run coarse-to-fine ICP."""

    parameters = _parameters(locals())
    try:
        source_points = load_point_cloud(source)
        target_points = load_point_cloud(target)
        result = multiscale_icp(
            source_points,
            target_points,
            voxel_sizes=voxel_sizes or [0.2, 0.1, 0.05],
            max_iterations_per_level=max_iterations_per_level,
            method=method,
            robust_kernel=robust_kernel,
            trim_ratio=trim_ratio,
        )
        artifacts: dict[str, str] = {}
        if output:
            save_point_cloud(output, result.aligned_points)
            artifacts["aligned_source"] = str(output)
        if save_diagnostics:
            path = Path(save_diagnostics)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(_json_ready(result.diagnostics), indent=2) + "\n",
                encoding="utf-8",
            )
            artifacts["diagnostics"] = str(path)
        task_result = TaskResult(
            task="multiscale-icp",
            success=True,
            metrics={
                "levels": len(result.diagnostics),
                "final_rmse": result.final_rmse,
                "fitness": result.fitness,
                "converged": result.converged,
            },
            artifacts=artifacts,
            parameters=parameters,
            data={"transformation": result.transformation, "diagnostics": result.diagnostics},
        )
    except Exception as exc:  # pragma: no cover - CLI failure path
        task_result = TaskResult("multiscale-icp", False, parameters=parameters, error=str(exc))
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
        conclusion = _benchmark_conclusion(benchmark)
        if benchmark == "kdtree":
            rows = _benchmark_kdtree(
                seed=seed,
                queries=queries,
                point_counts=points or ([1000, 10000, 100000] if full or quick else [1000, 10000]),
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
        elif benchmark == "gicp":
            rows = _benchmark_gicp(seed=seed, quick=quick)
            markdown = _format_generic_table(rows)
        elif benchmark == "segmentation":
            rows = _benchmark_segmentation(seed=seed, quick=quick)
            markdown = _format_generic_table(rows)
        elif benchmark == "all":
            rows = []
            suite_summaries = []
            for suite in ["kdtree", "icp", "ransac", "registration", "gicp", "segmentation"]:
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
                suite_summaries.append(
                    {
                        "suite": suite,
                        "cases": suite_result.metrics.get("cases", 0),
                        "conclusion": _benchmark_conclusion(suite),
                    }
                )
            markdown = _format_generic_table(rows)
            conclusion = "Quick portfolio benchmark completed across geometry core suites."
        else:
            raise ValueError(
                "benchmark must be one of: kdtree, icp, ransac, registration, gicp, "
                "segmentation, all"
            )

        artifacts: dict[str, str] = {}
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{benchmark}_benchmark.csv"
        md_default_path = out_dir / f"{benchmark}_benchmark.md"
        json_default_path = out_dir / f"{benchmark}_benchmark.json"
        png_path = out_dir / f"{benchmark}_benchmark.png"
        _write_csv(csv_path, rows)
        md_default_path.write_text(markdown + f"\n\nConclusion: {conclusion}\n", encoding="utf-8")
        json_default_path.write_text(
            json.dumps(
                {"benchmark": benchmark, "conclusion": conclusion, "rows": _json_ready(rows)},
                indent=2,
            )
            + "\n",
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
            md_path.write_text(markdown + f"\n\nConclusion: {conclusion}\n", encoding="utf-8")
            artifacts["benchmark_markdown_custom"] = str(md_path)
        if benchmark == "all":
            summary_rows = suite_summaries if "suite_summaries" in locals() else []
            summary_md = _format_benchmark_summary(summary_rows)
            summary_json = out_dir / "benchmark_summary.json"
            summary_md_path = out_dir / "benchmark_summary.md"
            summary_md_path.write_text(summary_md + "\n", encoding="utf-8")
            summary_json.write_text(
                json.dumps({"suites": summary_rows}, indent=2) + "\n",
                encoding="utf-8",
            )
            artifacts["benchmark_summary_markdown"] = str(summary_md_path)
            artifacts["benchmark_summary_json"] = str(summary_json)

        task_result = TaskResult(
            task=f"benchmark:{benchmark}",
            success=True,
            metrics={
                "benchmark": benchmark,
                "cases": len(rows),
                "quick": quick,
                "full": full,
                "all_correct": all(bool(row.get("correct", True)) for row in rows),
                "conclusion": conclusion,
            },
            artifacts=artifacts,
            parameters=parameters,
            data={"rows": rows, "markdown": markdown, "conclusion": conclusion},
        )
    except Exception as exc:  # pragma: no cover - exercised through CLI failures
        task_result = TaskResult(
            f"benchmark:{benchmark}",
            False,
            parameters=parameters,
            error=str(exc),
        )
    return _finalize_result(task_result, output_dir)


def run_iss_keypoints(
    input_path: str | Path,
    output_dir: str | Path = "outputs/features",
    salient_radius: float = 0.18,
    non_max_radius: float = 0.12,
    descriptor_radius: float = 0.25,
    export_html: str | Path | None = None,
) -> TaskResult:
    """Detect ISS keypoints and compute local geometric descriptors."""

    parameters = _parameters(locals())
    try:
        points = load_point_cloud(input_path)
        result = detect_iss_keypoints(points, salient_radius, non_max_radius)
        descriptors = compute_local_geometric_descriptors(
            points,
            result.indices,
            radius=descriptor_radius,
        )
        artifacts: dict[str, str] = {}
        if export_html:
            colors = np.full((len(points), 3), 0.65, dtype=float)
            colors[result.indices] = np.asarray([0.9, 0.1, 0.1])
            export_point_cloud_html(points, colors, export_html, title="ISS Keypoints")
            artifacts["html"] = str(export_html)
        task_result = TaskResult(
            task="iss-keypoints",
            success=True,
            metrics={
                "point_count": len(points),
                "keypoints": len(result.indices),
                "descriptor_dimension": descriptors.shape[1] if descriptors.ndim == 2 else 0,
            },
            artifacts=artifacts,
            parameters=parameters,
            data={
                "indices": result.indices,
                "saliency": result.saliency,
                "descriptors": descriptors,
            },
        )
    except Exception as exc:  # pragma: no cover - CLI failure path
        task_result = TaskResult("iss-keypoints", False, parameters=parameters, error=str(exc))
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
    multiscale: bool = False,
    voxel_sizes: list[float] | None = None,
    robust_kernel: str = "none",
    trim_ratio: float = 1.0,
    save_diagnostics: str | Path | None = None,
) -> TaskResult:
    """Run feature-based global registration and ICP refinement."""

    parameters = _parameters(locals())
    try:
        source_points = load_point_cloud(source)
        target_points = load_point_cloud(target)
        if method == "fpfh_ransac_icp":
            result = register_fpfh_ransac_icp(
                source_points,
                target_points,
                voxel_size=voxel_size,
                icp_method=icp_method,
                threshold=threshold,
                seed=seed,
            )
        elif method == "iss_descriptor_ransac_icp":
            result = register_iss_descriptor_ransac_icp(
                source_points,
                target_points,
                salient_radius=max(voxel_size * 3.0, 0.05),
                non_max_radius=max(voxel_size * 2.0, 0.04),
                descriptor_radius=max(voxel_size * 4.0, 0.08),
                threshold=threshold or max(voxel_size * 2.0, 0.05),
                seed=seed,
                icp_method=icp_method,
            )
        else:
            raise ValueError("method must be one of: fpfh_ransac_icp, iss_descriptor_ransac_icp")
        eval_threshold = threshold if threshold is not None else voxel_size * 1.5
        refined_transform = result.refined_transform
        refined_fitness = result.refined.fitness
        refined_rmse = result.refined.inlier_rmse
        diagnostics = {"coarse": result.coarse.metadata, "refined": result.refined.metadata}
        if multiscale:
            initialized = apply_homogeneous_transform(source_points, result.initial_transform)
            ms_result = multiscale_icp(
                initialized,
                target_points,
                voxel_sizes=voxel_sizes or [voxel_size * 4.0, voxel_size * 2.0, voxel_size],
                method=icp_method,
                robust_kernel=robust_kernel,
                trim_ratio=trim_ratio,
                max_correspondence_distance=eval_threshold,
            )
            refined_transform = ms_result.transformation @ result.initial_transform
            refined_fitness = ms_result.fitness
            refined_rmse = ms_result.final_rmse
            diagnostics["multiscale"] = ms_result.diagnostics
        elif robust_kernel != "none" or trim_ratio < 1.0:
            initialized = apply_homogeneous_transform(source_points, result.initial_transform)
            robust_result = robust_icp(
                initialized,
                target_points,
                method=icp_method,
                robust_kernel=robust_kernel,
                trim_ratio=trim_ratio,
                max_correspondence_distance=eval_threshold,
            )
            refined_transform = robust_result.transformation @ result.initial_transform
            refined_fitness = robust_result.fitness
            refined_rmse = robust_result.final_rmse
            diagnostics["robust"] = robust_result.diagnostics
        metrics = evaluate_registration(
            source_points,
            target_points,
            refined_transform,
            threshold=eval_threshold,
        )
        artifacts: dict[str, str] = {}
        aligned = apply_homogeneous_transform(source_points, refined_transform)
        if output:
            save_point_cloud(output, aligned)
            artifacts["aligned_source"] = str(output)
        if save_transform:
            transform_path = Path(save_transform)
            transform_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(transform_path, refined_transform, fmt="%.10f")
            artifacts["transform"] = str(transform_path)
        if save_diagnostics:
            diagnostics_path = Path(save_diagnostics)
            diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            diagnostics_path.write_text(
                json.dumps(_json_ready(diagnostics), indent=2) + "\n",
                encoding="utf-8",
            )
            artifacts["diagnostics"] = str(diagnostics_path)
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
                refined_transform,
                export_html,
            )
            artifacts["html"] = str(export_html)
        task_result = TaskResult(
            task="register",
            success=True,
            metrics={
                "coarse_fitness": result.coarse.fitness,
                "coarse_inlier_rmse": result.coarse.inlier_rmse,
                "refined_fitness": refined_fitness,
                "refined_inlier_rmse": refined_rmse,
                "final_rmse": metrics["rmse"],
                "final_fitness": metrics["fitness"],
                "source_downsampled": result.source_downsampled,
                "target_downsampled": result.target_downsampled,
            },
            artifacts=artifacts,
            parameters=parameters,
            data={
                "initial_transform": result.initial_transform,
                "refined_transform": refined_transform,
                "method": result.method,
                "diagnostics": diagnostics,
            },
        )
    except Exception as exc:  # pragma: no cover - exercised through CLI failures
        task_result = TaskResult("register", False, parameters=parameters, error=str(exc))
    return _finalize_result(task_result, output_dir)


def run_feature_registration(
    source: str | Path,
    target: str | Path,
    output_dir: str | Path = "outputs/registration",
    output: str | Path | None = None,
    threshold: float = 0.08,
    seed: int | None = 7,
) -> TaskResult:
    """Run the self-implemented ISS descriptor RANSAC registration pipeline."""

    return run_global_registration(
        source=source,
        target=target,
        output=output,
        output_dir=output_dir,
        voxel_size=max(threshold / 2.0, 0.02),
        method="iss_descriptor_ransac_icp",
        threshold=threshold,
        seed=seed,
        save_results=True,
    )


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


def run_extract_primitives(
    input_path: str | Path,
    models: list[str] | None = None,
    output: str | Path | None = None,
    output_dir: str | Path = "outputs/primitives",
    threshold: float = 0.03,
    max_models: int = 5,
    min_inliers: int = 30,
    max_iterations: int = 800,
    seed: int | None = 7,
    export_html: str | Path | None = None,
) -> TaskResult:
    """Extract multiple geometric primitives with sequential RANSAC."""

    parameters = _parameters(locals())
    try:
        points = load_point_cloud(input_path)
        result = extract_primitives(
            points,
            model_types=models or ["plane", "sphere", "cylinder"],
            threshold=threshold,
            max_models=max_models,
            min_inliers=min_inliers,
            max_iterations=max_iterations,
            random_state=seed,
        )
        labels = np.full(len(points), -1, dtype=int)
        for label, primitive in enumerate(result.primitives):
            labels[primitive.inlier_indices] = label

        artifacts: dict[str, str] = {}
        if output:
            save_colored_point_cloud(output, points, labels)
            artifacts["colored_primitives"] = str(output)
        if export_html:
            export_point_cloud_html(
                points,
                label_colors(labels),
                export_html,
                title="Sequential Primitive Extraction",
            )
            artifacts["html"] = str(export_html)
        json_path = Path(output_dir) / "primitive_models.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps([primitive.get_params() for primitive in result.primitives], indent=2)
            + "\n",
            encoding="utf-8",
        )
        artifacts["models_json"] = str(json_path)
        task_result = TaskResult(
            task="extract-primitives",
            success=True,
            metrics={
                "point_count": len(points),
                "model_count": len(result.primitives),
                "remaining_points": len(result.remaining_indices),
            },
            artifacts=artifacts,
            parameters=parameters,
            data={
                "labels": labels,
                "primitives": [primitive.get_params() for primitive in result.primitives],
                "remaining_indices": result.remaining_indices,
            },
        )
    except Exception as exc:  # pragma: no cover - CLI failure path
        task_result = TaskResult(
            "extract-primitives",
            False,
            parameters=parameters,
            error=str(exc),
        )
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
    remove_ground: bool = False,
    ground_axis: str = "z",
    ground_angle_threshold: float = 20.0,
    export_report: str | Path | None = None,
) -> TaskResult:
    """Segment a point cloud with DBSCAN, Euclidean clustering, or region growing."""

    parameters = _parameters(locals())
    try:
        points = load_point_cloud(input_path)
        if remove_ground:
            ground_result = ground_object_segmentation(
                points,
                ground_threshold=radius if radius > 0 else eps,
                ground_axis=ground_axis,
                ground_angle_threshold=ground_angle_threshold,
                cluster_method="dbscan" if method == "dbscan" else "euclidean",
                eps=tolerance if tolerance is not None else eps,
                min_points=min_points,
            )
            labels = ground_result.labels
            artifacts: dict[str, str] = {}
            if output:
                save_colored_point_cloud(output, points, labels)
                artifacts["colored_point_cloud"] = str(output)
            if export_html:
                export_point_cloud_html(
                    points,
                    label_colors(labels),
                    export_html,
                    title="Ground Removal and Object Clustering",
                )
                artifacts["html"] = str(export_html)
            if export_report:
                write_cluster_report(ground_result, export_report)
                artifacts["cluster_report"] = str(export_report)
            task_result = TaskResult(
                task="segment",
                success=True,
                metrics={
                    "point_count": len(points),
                    "ground_points": len(ground_result.ground.ground_indices),
                    "cluster_count": len(ground_result.clusters),
                    "noise_points": len(ground_result.noise_indices),
                },
                artifacts=artifacts,
                parameters=parameters,
                data={
                    "labels": labels,
                    "ground": {
                        "plane_model": ground_result.ground.plane_model,
                        "normal_angle_degrees": ground_result.ground.normal_angle_degrees,
                    },
                    "clusters": [cluster.to_dict() for cluster in ground_result.clusters],
                },
            )
            return _finalize_result(task_result, output_dir)

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


def run_ground_object_segmentation(
    input_path: str | Path,
    output: str | Path | None = None,
    output_dir: str | Path = "outputs/segmentation",
    eps: float = 0.15,
    min_points: int = 10,
    ground_axis: str = "z",
    ground_angle_threshold: float = 20.0,
    export_html: str | Path | None = None,
    export_report: str | Path | None = None,
) -> TaskResult:
    """Run the ground-removal and object-clustering pipeline."""

    return run_segmentation(
        input_path=input_path,
        output=output,
        output_dir=output_dir,
        method="euclidean",
        eps=eps,
        min_points=min_points,
        radius=0.03,
        export_html=export_html,
        remove_ground=True,
        ground_axis=ground_axis,
        ground_angle_threshold=ground_angle_threshold,
        export_report=export_report,
    )


def run_reconstruction(
    input_path: str | Path,
    output: str | Path,
    output_dir: str | Path = "outputs/reconstruction",
    method: str = "poisson",
    normal_radius: float = 0.15,
    poisson_depth: int = 6,
    alpha: float = 0.2,
) -> TaskResult:
    """Run Open3D-backed surface reconstruction."""

    parameters = _parameters(locals())
    try:
        points = load_point_cloud(input_path)
        result = reconstruct_surface(
            points,
            method=method,
            output=output,
            normal_radius=normal_radius,
            poisson_depth=poisson_depth,
            alpha=alpha,
        )
        task_result = TaskResult(
            task="reconstruct",
            success=True,
            metrics={
                "method": result.method,
                "vertices": len(result.vertices),
                "triangles": len(result.triangles),
            },
            artifacts={"mesh": str(output)},
            parameters=parameters,
        )
    except Exception as exc:  # pragma: no cover - optional dependency path
        task_result = TaskResult("reconstruct", False, parameters=parameters, error=str(exc))
    return _finalize_result(task_result, output_dir)


def run_portfolio_verification(
    output_dir: str | Path = "outputs",
    quick: bool = True,
) -> TaskResult:
    """Run a reproducible portfolio smoke-check and write a Markdown report."""

    parameters = _parameters(locals())
    root = Path(__file__).resolve().parents[1]
    out_dir = Path(output_dir).resolve()
    report_path = out_dir / "portfolio_check_report.md"
    commands = [
        [sys.executable, "examples/generate_demo_data.py"],
        [sys.executable, "examples/global_registration_demo.py"],
        [sys.executable, "examples/primitive_fitting_demo.py"],
        [sys.executable, "examples/segmentation_demo.py"],
        [sys.executable, "examples/benchmark_demo.py"],
        [sys.executable, "examples/visualization_demo.py"],
        [sys.executable, "examples/gallery_demo.py"],
        [
            sys.executable,
            "-m",
            "pointcloud_geolab",
            "benchmark",
            "--suite",
            "all",
            "--quick" if quick else "--no-quick",
            "--output",
            str(out_dir / "benchmarks"),
        ],
    ]
    cli_smoke = [
        [
            sys.executable,
            "-m",
            "pointcloud_geolab",
            "benchmark",
            "--suite",
            "kdtree",
            "--quick",
            "--output",
            str(out_dir / "benchmarks" / "kdtree_smoke"),
        ],
        [
            sys.executable,
            "-m",
            "pointcloud_geolab",
            "extract-primitives",
            "--input",
            "data/synthetic_scene.ply",
            "--models",
            "plane",
            "sphere",
            "cylinder",
            "--threshold",
            "0.04",
            "--max-models",
            "3",
            "--min-inliers",
            "20",
            "--output",
            str(out_dir / "primitives" / "verify_primitives.ply"),
        ],
    ]
    commands.extend(cli_smoke)

    passed = []
    failed = []
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=240,
            check=False,
        )
        record = {
            "command": " ".join(command),
            "returncode": completed.returncode,
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-2000:],
        }
        if completed.returncode == 0:
            passed.append(record)
        else:
            failed.append(record)

    artifacts = sorted(str(path.relative_to(root)) for path in out_dir.rglob("*") if path.is_file())
    missing = _missing_readme_artifacts(root)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _format_portfolio_report(passed, failed, artifacts, missing),
        encoding="utf-8",
    )
    task_result = TaskResult(
        task="verify-portfolio",
        success=not failed and not missing,
        metrics={
            "passed_commands": len(passed),
            "failed_commands": len(failed),
            "generated_artifacts": len(artifacts),
            "missing_readme_artifacts": len(missing),
        },
        artifacts={"report": str(report_path)},
        parameters=parameters,
        data={
            "passed": passed,
            "failed": failed,
            "artifacts": artifacts,
            "missing_readme_artifacts": missing,
        },
        error=None if not failed else "one or more portfolio commands failed",
    )
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
            if mode in {"clusters", "primitives", "inliers_outliers"}:
                if labels_path is None:
                    raise ValueError(f"labels_path is required for {mode} visualization")
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

        radius_queries = query_points[: min(len(query_points), 25)]
        radius = 0.08
        start = time.perf_counter()
        kd_radius_counts = [len(tree.radius_search(query, radius)) for query in radius_queries]
        kd_radius_time = time.perf_counter() - start

        start = time.perf_counter()
        voxel = VoxelHashGrid.build(points, voxel_size=radius)
        voxel_build_time = time.perf_counter() - start
        start = time.perf_counter()
        voxel_radius_counts = [len(voxel.radius_search(query, radius)) for query in radius_queries]
        voxel_radius_time = time.perf_counter() - start

        row: dict[str, Any] = {
            "points": points_count,
            "queries": queries,
            "build_time": build_time,
            "brute_time": brute_time,
            "kd_time": kd_time,
            "kd_radius_time": kd_radius_time,
            "voxel_build_time": voxel_build_time,
            "voxel_radius_time": voxel_radius_time,
            "voxel_radius_correct": kd_radius_counts == voxel_radius_counts,
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
    except Exception:
        return None

    try:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        start = time.perf_counter()
        tree = o3d.geometry.KDTreeFlann(point_cloud)
        for query in queries:
            tree.search_knn_vector_3d(query, 1)
        return time.perf_counter() - start
    except Exception:
        return None


def _optional_scipy_ckdtree_time(points: np.ndarray, queries: np.ndarray) -> float | None:
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception:
        return None
    try:
        start = time.perf_counter()
        tree = cKDTree(points)
        tree.query(queries, k=1)
        return time.perf_counter() - start
    except Exception:
        return None


def _optional_sklearn_kdtree_time(points: np.ndarray, queries: np.ndarray) -> float | None:
    try:
        from sklearn.neighbors import KDTree as SklearnKDTree  # type: ignore
    except Exception:
        return None
    try:
        start = time.perf_counter()
        tree = SklearnKDTree(points)
        tree.query(queries, k=1)
        return time.perf_counter() - start
    except Exception:
        return None


def _optional_open3d_icp_metrics(
    source: np.ndarray,
    target: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    try:
        import open3d as o3d  # type: ignore
    except Exception as exc:
        return {
            "open3d_icp_status": f"unavailable: {exc}",
            "open3d_icp_rmse": None,
            "open3d_icp_fitness": None,
            "open3d_icp_time": None,
        }

    try:
        source_cloud = o3d.geometry.PointCloud()
        target_cloud = o3d.geometry.PointCloud()
        source_cloud.points = o3d.utility.Vector3dVector(source)
        target_cloud.points = o3d.utility.Vector3dVector(target)
        start = time.perf_counter()
        result = o3d.pipelines.registration.registration_icp(
            source_cloud,
            target_cloud,
            threshold,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        runtime = time.perf_counter() - start
        return {
            "open3d_icp_status": "ok",
            "open3d_icp_rmse": float(result.inlier_rmse),
            "open3d_icp_fitness": float(result.fitness),
            "open3d_icp_time": runtime,
        }
    except Exception as exc:
        return {
            "open3d_icp_status": f"error: {exc}",
            "open3d_icp_rmse": None,
            "open3d_icp_fitness": None,
            "open3d_icp_time": None,
        }


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

    start = time.perf_counter()
    result = point_to_point_icp(source, target, max_iterations=80, tolerance=1e-7)
    custom_time = time.perf_counter() - start

    outlier_count = max(1, len(source) // 8)
    outliers = rng.uniform(-3.0, 3.0, size=(outlier_count, 3))
    source_with_outliers = np.vstack([source, outliers])
    outlier_threshold = max(0.6, translation_magnitude * 2.0)
    plain_outlier = point_to_point_icp(
        source_with_outliers,
        target,
        max_iterations=60,
        tolerance=1e-7,
        max_correspondence_distance=outlier_threshold,
    )
    huber_outlier = robust_icp(
        source_with_outliers,
        target,
        robust_kernel="huber",
        trim_ratio=0.9,
        max_iterations=60,
        tolerance=1e-7,
        max_correspondence_distance=outlier_threshold,
    )
    trimmed_outlier = robust_icp(
        source_with_outliers,
        target,
        robust_kernel="none",
        trim_ratio=0.85,
        max_iterations=60,
        tolerance=1e-7,
        max_correspondence_distance=outlier_threshold,
    )
    open3d_metrics = _optional_open3d_icp_metrics(source, target, threshold=outlier_threshold)
    expected_rotation = rotation.T
    expected_translation = -rotation.T @ translation
    row = {
        "rotation_degrees": rotation_degrees,
        "translation": translation_magnitude,
        "noise": noise,
        "converged": result.converged,
        "iterations": result.iterations,
        "final_rmse": result.final_rmse,
        "custom_icp_time": custom_time,
        "plain_outlier_rmse": plain_outlier.final_rmse,
        "huber_outlier_rmse": huber_outlier.final_rmse,
        "trimmed_outlier_rmse": trimmed_outlier.final_rmse,
        "outlier_ratio": outlier_count / len(source_with_outliers),
        "rotation_error_degrees": _rotation_error_degrees(result.rotation, expected_rotation),
        "translation_error": float(np.linalg.norm(result.translation - expected_translation)),
    }
    row.update(open3d_metrics)
    return row


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
        numpy_baseline = _numpy_plane_baseline(points, truth_mask, threshold=0.02)
        open3d_baseline = _optional_open3d_plane_baseline(
            points,
            truth_mask,
            threshold=0.02,
            seed=seed + i,
        )
        rows.append(
            {
                "outlier_ratio": ratio,
                "custom_ransac_inliers": len(result.inlier_indices),
                "custom_precision": precision,
                "custom_recall": recall,
                "custom_residual_mean": result.residual_mean,
                **numpy_baseline,
                **open3d_baseline,
            }
        )
    return rows


def _numpy_plane_baseline(
    points: np.ndarray,
    truth_mask: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    model = PlaneModel.fit(points)
    predicted = model.residuals(points) <= threshold
    true_positive = int(np.sum(predicted & truth_mask))
    precision = true_positive / max(int(np.sum(predicted)), 1)
    recall = true_positive / max(int(np.sum(truth_mask)), 1)
    residuals = model.residuals(points)
    return {
        "numpy_pca_inliers": int(np.sum(predicted)),
        "numpy_pca_precision": precision,
        "numpy_pca_recall": recall,
        "numpy_pca_residual_mean": (
            float(residuals[predicted].mean()) if np.any(predicted) else None
        ),
    }


def _optional_open3d_plane_baseline(
    points: np.ndarray,
    truth_mask: np.ndarray,
    threshold: float,
    seed: int,
) -> dict[str, Any]:
    try:
        import open3d as o3d  # type: ignore
    except Exception as exc:
        return {
            "open3d_plane_status": f"unavailable: {exc}",
            "open3d_plane_inliers": None,
            "open3d_plane_precision": None,
            "open3d_plane_recall": None,
        }

    try:
        try:
            o3d.utility.random.seed(int(seed))
        except AttributeError:
            pass
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        _, inliers = point_cloud.segment_plane(
            distance_threshold=threshold,
            ransac_n=3,
            num_iterations=800,
        )
        predicted = np.zeros(len(points), dtype=bool)
        predicted[np.asarray(inliers, dtype=int)] = True
        true_positive = int(np.sum(predicted & truth_mask))
        precision = true_positive / max(int(np.sum(predicted)), 1)
        recall = true_positive / max(int(np.sum(truth_mask)), 1)
        return {
            "open3d_plane_status": "ok",
            "open3d_plane_inliers": int(np.sum(predicted)),
            "open3d_plane_precision": precision,
            "open3d_plane_recall": recall,
        }
    except Exception as exc:
        return {
            "open3d_plane_status": f"error: {exc}",
            "open3d_plane_inliers": None,
            "open3d_plane_precision": None,
            "open3d_plane_recall": None,
        }


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


def _benchmark_gicp(seed: int, quick: bool) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    point_count = 240 if quick else 500
    xy = rng.uniform(-1.0, 1.0, size=(point_count, 2))
    z = 0.12 * np.sin(2.5 * xy[:, 0]) + 0.08 * np.cos(3.0 * xy[:, 1])
    target = np.column_stack([xy, z])
    cases = [4.0, 10.0] if quick else [4.0, 10.0, 18.0]
    rows = []
    for angle in cases:
        rotation = rotation_matrix_from_euler(
            np.radians(angle) * 0.35,
            -np.radians(angle) * 0.25,
            np.radians(angle) * 0.45,
        )
        translation = np.asarray([0.05, -0.03, 0.02]) * (1.0 + angle / 20.0)
        source = apply_transform(target, rotation, translation)

        start = time.perf_counter()
        icp = point_to_point_icp(
            source,
            target,
            max_iterations=60,
            tolerance=1e-7,
            max_correspondence_distance=0.35,
        )
        icp_time = time.perf_counter() - start
        rows.append(
            {
                "method": "point_to_point_icp",
                "initial_angle_degrees": angle,
                "points": point_count,
                "runtime": icp_time,
                "initial_rmse": icp.initial_rmse,
                "final_rmse": icp.final_rmse,
                "iterations": icp.iterations,
            }
        )

        start = time.perf_counter()
        gicp = generalized_icp(
            source,
            target,
            max_iterations=60,
            tolerance=1e-7,
            max_correspondence_distance=0.35,
            k_neighbors=16,
        )
        gicp_time = time.perf_counter() - start
        rows.append(
            {
                "method": "generalized_icp",
                "initial_angle_degrees": angle,
                "points": point_count,
                "runtime": gicp_time,
                "initial_rmse": gicp.initial_rmse,
                "final_rmse": gicp.final_rmse,
                "iterations": gicp.iterations,
            }
        )
    return rows


def _benchmark_segmentation(seed: int, quick: bool) -> list[dict[str, Any]]:
    counts = [300, 900] if quick else [300, 900, 1800]
    rows = []
    for count in counts:
        rng = np.random.default_rng(seed + count)
        clusters = [
            rng.normal(scale=0.035, size=(count // 3, 3)) + np.asarray([0.0, 0.0, 0.2]),
            rng.normal(scale=0.035, size=(count // 3, 3)) + np.asarray([0.8, 0.0, 0.2]),
            rng.normal(scale=0.035, size=(count - 2 * (count // 3), 3))
            + np.asarray([0.0, 0.8, 0.2]),
        ]
        points = np.vstack(clusters)
        start = time.perf_counter()
        result = euclidean_clustering(points, tolerance=0.12, min_points=8)
        runtime = time.perf_counter() - start
        rows.append(
            {
                "points": len(points),
                "method": "euclidean",
                "runtime": runtime,
                "clusters": result.cluster_count,
                "noise_points": len(result.noise_indices),
            }
        )
    return rows


def _benchmark_conclusion(benchmark: str) -> str:
    conclusions = {
        "kdtree": (
            "Custom KDTree demonstrates pruning logic; SciPy/sklearn are expected "
            "to win raw throughput at large N, while voxel hash is competitive for "
            "fixed-radius locality."
        ),
        "icp": (
            "Point-to-point ICP is accurate near the solution but remains sensitive "
            "to initialization, noise, and outliers."
        ),
        "ransac": (
            "RANSAC remains stable with moderate outliers until clean minimal "
            "samples become unlikely."
        ),
        "registration": (
            "Feature-based global registration expands ICP's basin of convergence "
            "by estimating a coarse pose first."
        ),
        "gicp": (
            "GICP uses local covariance structure to weight correspondences; it is "
            "more expensive per iteration than point-to-point ICP but exposes "
            "surface-aware residuals."
        ),
        "segmentation": (
            "Euclidean clustering runtime scales with radius-neighborhood queries "
            "and benefits directly from spatial indexing."
        ),
    }
    return conclusions.get(benchmark, "Benchmark completed.")


def _format_benchmark_summary(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Benchmark Summary",
        "",
        "| Suite | Cases | Conclusion |",
        "|---|---:|---|",
    ]
    for row in rows:
        lines.append(f"| {row['suite']} | {row['cases']} | {row['conclusion']} |")
    return "\n".join(lines)


def _rotation_error_degrees(estimated: np.ndarray, expected: np.ndarray) -> float:
    delta = estimated @ expected.T
    cos_angle = (np.trace(delta) - 1.0) / 2.0
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def _format_kdtree_table(rows: list[dict[str, Any]]) -> str:
    has_open3d = any("open3d_time" in row for row in rows)
    has_scipy = any("scipy_ckdtree_time" in row for row in rows)
    has_sklearn = any("sklearn_kdtree_time" in row for row in rows)
    headers = [
        "Points",
        "Queries",
        "Build Time (s)",
        "Brute Force (s)",
        "KD-Tree (s)",
        "KD Radius (s)",
        "Voxel Radius (s)",
    ]
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
            f"{row['kd_radius_time']:.4f}",
            f"{row['voxel_radius_time']:.4f}",
        ]
        if has_open3d:
            values.append(_format_optional_time(row, "open3d_time"))
        if has_scipy:
            values.append(_format_optional_time(row, "scipy_ckdtree_time"))
        if has_sklearn:
            values.append(_format_optional_time(row, "sklearn_kdtree_time"))
        correct = bool(row["correct"]) and bool(row.get("voxel_radius_correct", True))
        values.extend([f"{row['speedup']:.2f}x", "yes" if correct else "no"])
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_optional_time(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    return "" if value is None else f"{value:.4f}"


def _format_optional_float(value: Any) -> str:
    return "" if value is None else f"{float(value):.6f}"


def _format_icp_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Rotation (deg) | Translation | Noise | Converged | Iterations | "
        "Final RMSE | Huber Outlier RMSE | Trimmed Outlier RMSE | Open3D RMSE | "
        "Rotation Error (deg) | Translation Error |",
        "|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {rotation_degrees:.0f} | {translation:.2f} | {noise:.2f} | {converged} | "
            "{iterations} | {final_rmse:.6f} | {huber_outlier_rmse:.6f} | "
            "{trimmed_outlier_rmse:.6f} | {open3d_icp_rmse} | "
            "{rotation_error_degrees:.4f} | {translation_error:.6f} |".format(
                rotation_degrees=row["rotation_degrees"],
                translation=row["translation"],
                noise=row["noise"],
                converged="yes" if row["converged"] else "no",
                iterations=row["iterations"],
                final_rmse=row["final_rmse"],
                huber_outlier_rmse=row.get("huber_outlier_rmse", float("nan")),
                trimmed_outlier_rmse=row.get("trimmed_outlier_rmse", float("nan")),
                open3d_icp_rmse=_format_optional_float(row.get("open3d_icp_rmse")),
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
        ax.plot(xs, [row["custom_precision"] for row in rows], marker="o", label="custom precision")
        ax.plot(xs, [row["custom_recall"] for row in rows], marker="o", label="custom recall")
        ax.plot(
            xs,
            [row["numpy_pca_precision"] for row in rows],
            marker="o",
            label="NumPy PCA precision",
        )
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
    elif benchmark == "gicp":
        for method in sorted({row["method"] for row in rows}):
            subset = [row for row in rows if row["method"] == method]
            ax.plot(
                [row["initial_angle_degrees"] for row in subset],
                [row["final_rmse"] for row in subset],
                marker="o",
                label=method,
            )
        ax.set_xlabel("Initial Rotation (deg)")
        ax.set_ylabel("Final RMSE")
    elif benchmark == "segmentation":
        xs = [row["points"] for row in rows]
        ax.plot(xs, [row["runtime"] for row in rows], marker="o", label="euclidean")
        ax.set_xlabel("Points")
        ax.set_ylabel("Runtime (s)")
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


def _missing_readme_artifacts(root: Path) -> list[str]:
    readme = root / "README.md"
    if not readme.exists():
        return ["README.md"]
    text = readme.read_text(encoding="utf-8")
    candidates = []
    for marker in ["docs/assets/", "outputs/gallery/"]:
        start = 0
        while True:
            index = text.find(marker, start)
            if index == -1:
                break
            end = index
            while end < len(text) and text[end] not in {")", '"', "'", "`", " ", "\n"}:
                end += 1
            candidates.append(text[index:end])
            start = end
    missing = []
    for candidate in sorted(set(candidates)):
        if not (root / candidate).exists():
            missing.append(candidate)
    return missing


def _format_portfolio_report(
    passed: list[dict[str, Any]],
    failed: list[dict[str, Any]],
    artifacts: list[str],
    missing: list[str],
) -> str:
    lines = [
        "# Portfolio Check Report",
        "",
        "## Passed Commands",
        "",
    ]
    lines.extend(f"- `{item['command']}`" for item in passed)
    lines.extend(["", "## Failed Commands", ""])
    if failed:
        for item in failed:
            lines.append(f"- `{item['command']}` returned {item['returncode']}")
            if item["stderr"]:
                lines.append(f"  - stderr tail: `{item['stderr'].splitlines()[-1]}`")
    else:
        lines.append("- None")
    lines.extend(["", "## Generated Artifacts", ""])
    lines.extend(f"- `{artifact}`" for artifact in artifacts[:80])
    if len(artifacts) > 80:
        lines.append(f"- ... {len(artifacts) - 80} more")
    lines.extend(["", "## Missing README Artifacts", ""])
    lines.extend(f"- `{item}`" for item in missing) if missing else lines.append("- None")
    lines.extend(
        [
            "",
            "## Next Actions",
            "",
            (
                "- Re-run failed commands after dependency installation "
                "if optional extras are missing."
            ),
            "- Regenerate gallery assets before updating README image references.",
        ]
    )
    return "\n".join(lines) + "\n"


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
