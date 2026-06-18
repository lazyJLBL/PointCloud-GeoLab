"""Runner for the portfolio pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from pointcloud_geolab.api import TaskResult
from pointcloud_geolab.geometry import compute_aabb
from pointcloud_geolab.io.pointcloud_io import load_point_cloud, save_point_cloud
from pointcloud_geolab.io.visualization import save_point_cloud_projection
from pointcloud_geolab.portfolio_pipeline.figures import (
    _save_bounding_box_or_normals_figure,
    _save_registration_figure,
    _save_segmentation_figure,
)
from pointcloud_geolab.portfolio_pipeline.inputs import (
    _load_registration_pair,
    _resolve_pipeline_inputs,
)
from pointcloud_geolab.portfolio_pipeline.metrics import (
    _auto_dbscan_eps,
    _auto_voxel_size,
    _bounds_metrics,
    _feature_metrics,
    _point_cloud_metrics,
)
from pointcloud_geolab.portfolio_pipeline.reports import (
    _format_pipeline_html_report,
    _format_pipeline_report,
    _json_ready,
)
from pointcloud_geolab.preprocessing import (
    estimate_normals,
    remove_statistical_outliers,
    voxel_downsample,
)
from pointcloud_geolab.registration import point_to_point_icp
from pointcloud_geolab.segmentation import cluster_statistics, dbscan_clustering


def run_portfolio_pipeline(
    input_path: str | Path = "examples/demo_data",
    output_dir: str | Path = "outputs/portfolio_demo",
    voxel_size: float | None = None,
    eps: float | None = None,
    min_points: int = 10,
    seed: int = 42,
    html_report: bool = True,
) -> TaskResult:
    """Run the single-command portfolio demo and write metrics, figures, and report."""

    parameters = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "voxel_size": voxel_size,
        "eps": eps,
        "min_points": min_points,
        "seed": seed,
        "html_report": html_report,
    }
    try:
        start_time = time.perf_counter()
        if min_points <= 0:
            raise ValueError("min_points must be positive")

        inputs = _resolve_pipeline_inputs(input_path)
        out_dir = Path(output_dir)
        figures_dir = out_dir / "figures"
        artifacts_dir = out_dir / "artifacts"
        figures_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(seed)
        raw_points = load_point_cloud(inputs.main_cloud)
        if len(raw_points) < 3:
            raise ValueError("pipeline input cloud must contain at least 3 points")

        input_metrics = _point_cloud_metrics(raw_points, inputs.main_cloud)

        selected_voxel_size = (
            float(voxel_size) if voxel_size is not None else _auto_voxel_size(raw_points)
        )
        downsampled = voxel_downsample(raw_points, selected_voxel_size)
        filtered, statistical_inliers = remove_statistical_outliers(
            downsampled,
            nb_neighbors=min(16, max(len(downsampled) - 1, 1)),
            std_ratio=2.0,
        )
        preprocessing_note = "statistical outlier removal applied"
        if len(filtered) < 3:
            filtered = downsampled
            statistical_inliers = np.arange(len(downsampled), dtype=int)
            preprocessing_note = "statistical outlier removal skipped because it removed too much"
        processed = filtered
        processed_aabb = compute_aabb(processed)
        preprocessing_metrics = {
            "num_points_before": int(len(raw_points)),
            "num_points_after_voxel": int(len(downsampled)),
            "num_points_after": int(len(processed)),
            "downsample_ratio": float(len(processed) / len(raw_points)),
            "voxel_size": selected_voxel_size,
            "statistical_inliers": int(len(statistical_inliers)),
            "bounds_after": _bounds_metrics(processed),
            "note": preprocessing_note,
        }

        normals = estimate_normals(processed, k=min(16, max(len(processed), 3)))
        feature_metrics = _feature_metrics(processed, normals, rng)

        source_points, target_points, registration_note = _load_registration_pair(
            inputs, raw_points
        )
        registration_voxel = _auto_voxel_size(np.vstack([source_points, target_points]))
        source_reg = voxel_downsample(source_points, registration_voxel)
        target_reg = voxel_downsample(target_points, registration_voxel)
        if len(source_reg) < 3 or len(target_reg) < 3:
            source_reg = source_points
            target_reg = target_points
        icp_result = point_to_point_icp(
            source_reg,
            target_reg,
            max_iterations=50,
            tolerance=1e-7,
            max_correspondence_distance=None,
        )
        registration_metrics = {
            "rmse_before": float(icp_result.initial_rmse),
            "rmse_after": float(icp_result.final_rmse),
            "fitness": float(icp_result.fitness),
            "iterations": int(icp_result.iterations),
            "converged": bool(icp_result.converged),
            "transformation": icp_result.transformation.tolist(),
            "note": registration_note,
        }

        segmentation_points = load_point_cloud(inputs.segmentation_cloud)
        selected_eps = float(eps) if eps is not None else _auto_dbscan_eps(segmentation_points)
        clustering = dbscan_clustering(
            segmentation_points,
            eps=selected_eps,
            min_points=min(min_points, max(1, len(segmentation_points))),
        )
        cluster_sizes = [int(cluster.point_count) for cluster in clustering.clusters]
        segmentation_metrics = {
            "num_clusters": int(clustering.cluster_count),
            "cluster_sizes": cluster_sizes,
            "noise_points": int(len(clustering.noise_indices)),
            "noise_ratio": float(len(clustering.noise_indices) / max(len(segmentation_points), 1)),
            "eps": selected_eps,
            "min_points": int(min_points),
            "clusters": cluster_statistics(segmentation_points, clustering.labels),
        }

        figure_paths = {
            "raw_pointcloud": figures_dir / "raw_pointcloud.png",
            "downsampled": figures_dir / "downsampled.png",
            "registration_before_after": figures_dir / "registration_before_after.png",
            "segmentation_result": figures_dir / "segmentation_result.png",
            "bounding_box_or_normals": figures_dir / "bounding_box_or_normals.png",
        }
        save_point_cloud_projection(
            figure_paths["raw_pointcloud"],
            [raw_points],
            labels=["raw"],
            title="Raw point cloud",
        )
        save_point_cloud_projection(
            figure_paths["downsampled"],
            [processed],
            labels=["processed"],
            title="Downsampled and filtered cloud",
        )
        _save_registration_figure(
            figure_paths["registration_before_after"],
            source_reg,
            target_reg,
            icp_result.aligned_points,
        )
        _save_segmentation_figure(
            figure_paths["segmentation_result"],
            segmentation_points,
            clustering.labels,
            title="DBSCAN segmentation",
        )
        _save_bounding_box_or_normals_figure(
            figure_paths["bounding_box_or_normals"],
            processed,
            processed_aabb.corners,
            normals,
            rng,
        )

        processed_cloud_path = artifacts_dir / "processed_cloud.ply"
        transformation_path = artifacts_dir / "transformation.json"
        metrics_path = out_dir / "metrics.json"
        report_path = out_dir / "report.md"
        html_report_path = out_dir / "report.html"
        save_point_cloud(processed_cloud_path, processed)
        transformation_payload = {
            "rmse_before": registration_metrics["rmse_before"],
            "rmse_after": registration_metrics["rmse_after"],
            "transformation": registration_metrics["transformation"],
        }
        transformation_path.write_text(
            json.dumps(transformation_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        runtime_seconds = time.perf_counter() - start_time

        metrics = {
            "input": input_metrics
            | {
                "requested_input": str(inputs.requested_root),
                "resolved_input": str(inputs.resolved_root),
                "main_cloud": str(inputs.main_cloud),
                "segmentation_cloud": str(inputs.segmentation_cloud),
            },
            "preprocessing": preprocessing_metrics,
            "features": feature_metrics,
            "registration": registration_metrics,
            "segmentation": segmentation_metrics,
            "runtime": {"total_seconds": runtime_seconds},
        }
        metrics_path.write_text(json.dumps(_json_ready(metrics), indent=2) + "\n", encoding="utf-8")
        markdown_report = _format_pipeline_report(
            metrics, figure_paths, processed_cloud_path, transformation_path
        )
        report_path.write_text(markdown_report, encoding="utf-8")
        if html_report:
            html_report_path.write_text(
                _format_pipeline_html_report(
                    metrics,
                    figure_paths,
                    processed_cloud_path,
                    transformation_path,
                    markdown_report,
                ),
                encoding="utf-8",
            )

        artifacts = {
            "report": str(report_path),
            "metrics_json": str(metrics_path),
            "processed_cloud": str(processed_cloud_path),
            "transformation_json": str(transformation_path),
            **{name: str(path) for name, path in figure_paths.items()},
        }
        if html_report:
            artifacts["html_report"] = str(html_report_path)
        return TaskResult(
            task="pipeline",
            success=True,
            metrics=metrics,
            artifacts=artifacts,
            parameters=parameters,
            data={
                "output_dir": str(out_dir),
                "figures_dir": str(figures_dir),
                "artifacts_dir": str(artifacts_dir),
            },
        )
    except Exception as exc:  # pragma: no cover - exercised by CLI failures
        return TaskResult(
            task="pipeline",
            success=False,
            parameters=parameters,
            error=f"portfolio pipeline failed: {exc}",
        )
