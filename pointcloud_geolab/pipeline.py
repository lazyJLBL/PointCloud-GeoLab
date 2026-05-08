"""Portfolio-ready end-to-end point cloud pipeline."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pointcloud_geolab.api import TaskResult
from pointcloud_geolab.geometry import compute_aabb
from pointcloud_geolab.io.pointcloud_io import load_point_cloud, save_point_cloud
from pointcloud_geolab.io.visualization import save_point_cloud_projection
from pointcloud_geolab.kdtree import KDTree
from pointcloud_geolab.preprocessing import (
    estimate_normals,
    remove_statistical_outliers,
    voxel_downsample,
)
from pointcloud_geolab.registration import point_to_point_icp
from pointcloud_geolab.segmentation import cluster_statistics, dbscan_clustering
from pointcloud_geolab.utils.transform import apply_transform, rotation_matrix_from_euler
from pointcloud_geolab.visualization import label_colors

POINT_EXTENSIONS = {".ply", ".pcd", ".xyz", ".txt", ".bin", ".las", ".laz"}


@dataclass(frozen=True, slots=True)
class PipelineInputs:
    """Resolved input files used by the portfolio pipeline."""

    requested_root: Path
    resolved_root: Path
    main_cloud: Path
    segmentation_cloud: Path
    registration_source: Path | None
    registration_target: Path | None


def run_portfolio_pipeline(
    input_path: str | Path = "examples/demo_data",
    output_dir: str | Path = "outputs/portfolio_demo",
    voxel_size: float | None = None,
    eps: float | None = None,
    min_points: int = 10,
    seed: int = 42,
) -> TaskResult:
    """Run the single-command portfolio demo and write metrics, figures, and report."""

    parameters = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "voxel_size": voxel_size,
        "eps": eps,
        "min_points": min_points,
        "seed": seed,
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
        report_path.write_text(
            _format_pipeline_report(
                metrics, figure_paths, processed_cloud_path, transformation_path
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


def _resolve_pipeline_inputs(input_path: str | Path) -> PipelineInputs:
    requested = Path(input_path)
    root = requested
    if not root.exists():
        repo_root = Path(__file__).resolve().parents[1]
        normalized = requested.as_posix().strip("/")
        if normalized == "examples/demo_data" and (repo_root / "data").exists():
            root = repo_root / "data"
        else:
            raise FileNotFoundError(f"input path does not exist: {requested}")

    if root.is_file():
        main_cloud = root
        return PipelineInputs(
            requested_root=requested,
            resolved_root=root.parent,
            main_cloud=main_cloud,
            segmentation_cloud=main_cloud,
            registration_source=None,
            registration_target=None,
        )

    if not root.is_dir():
        raise ValueError(f"input path must be a point cloud file or directory: {requested}")

    main_cloud = _first_existing(
        root,
        ["object.ply", "synthetic_scene.ply", "lidar_scene.ply", "room.pcd", "room.xyz"],
    )
    segmentation_cloud = _first_existing(
        root,
        ["lidar_scene.ply", "synthetic_scene.ply", "room.pcd", "object.ply", main_cloud.name],
    )
    registration_source = _optional_existing(root, ["bunny_source.ply", "source.ply"])
    registration_target = _optional_existing(root, ["bunny_target.ply", "target.ply"])
    return PipelineInputs(
        requested_root=requested,
        resolved_root=root,
        main_cloud=main_cloud,
        segmentation_cloud=segmentation_cloud,
        registration_source=registration_source,
        registration_target=registration_target,
    )


def _first_existing(root: Path, names: list[str]) -> Path:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    discovered = sorted(path for path in root.iterdir() if path.suffix.lower() in POINT_EXTENSIONS)
    if discovered:
        return discovered[0]
    raise FileNotFoundError(f"no supported point cloud files found in {root}")


def _optional_existing(root: Path, names: list[str]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def _load_registration_pair(
    inputs: PipelineInputs,
    fallback_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    if inputs.registration_source and inputs.registration_target:
        return (
            load_point_cloud(inputs.registration_source),
            load_point_cloud(inputs.registration_target),
            "using source/target demo clouds from the input directory",
        )

    target = np.asarray(fallback_points, dtype=float)
    rotation = rotation_matrix_from_euler(0.08, -0.05, 0.10)
    translation = np.asarray([0.16, -0.08, 0.10], dtype=float)
    source = apply_transform(target, rotation, translation)
    return source, target, "using a deterministic transformed copy of the input cloud"


def _point_cloud_metrics(points: np.ndarray, path: Path) -> dict[str, Any]:
    return {
        "num_points": int(len(points)),
        "dimension": int(points.shape[1]),
        "bounds": _bounds_metrics(points),
        "has_color": _file_has_properties(path, {"red", "green", "blue", "rgb"}),
        "has_normals": _file_has_properties(path, {"nx", "ny", "nz", "normal_x"}),
    }


def _bounds_metrics(points: np.ndarray) -> dict[str, Any]:
    pts = np.asarray(points, dtype=float)
    return {
        "min": pts.min(axis=0).tolist(),
        "max": pts.max(axis=0).tolist(),
        "extent": (pts.max(axis=0) - pts.min(axis=0)).tolist(),
    }


def _file_has_properties(path: Path, property_names: set[str]) -> bool:
    suffix = path.suffix.lower()
    try:
        if suffix == ".ply":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip().lower()
                    if stripped == "end_header":
                        break
                    parts = stripped.split()
                    if len(parts) >= 3 and parts[0] == "property" and parts[-1] in property_names:
                        return True
        if suffix == ".pcd":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip().lower()
                    if stripped.startswith("fields"):
                        fields = set(stripped.split()[1:])
                        return bool(fields & property_names)
                    if stripped.startswith("data"):
                        break
    except UnicodeDecodeError:
        return False
    return False


def _auto_voxel_size(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    extent = pts.max(axis=0) - pts.min(axis=0)
    diagonal = float(np.linalg.norm(extent))
    return max(diagonal / 70.0, 1e-4)


def _auto_dbscan_eps(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    extent = pts.max(axis=0) - pts.min(axis=0)
    diagonal = float(np.linalg.norm(extent))
    return max(diagonal / 35.0, 1e-3)


def _feature_metrics(
    points: np.ndarray, normals: np.ndarray, rng: np.random.Generator
) -> dict[str, Any]:
    pts = np.asarray(points, dtype=float)
    sample_indices = _sample_indices(len(pts), min(len(pts), 300), rng)
    tree = KDTree(pts)
    density_radius = _auto_dbscan_eps(pts) * 0.8
    neighbor_counts = []
    curvatures = []
    k = min(16, len(pts))
    for index in sample_indices:
        point = pts[index]
        neighbor_counts.append(max(len(tree.radius_search(point, density_radius)) - 1, 0))
        if k >= 3:
            neighbors = tree.knn_search(point, k)
            neighborhood = pts[[idx for idx, _ in neighbors]]
            centered = neighborhood - neighborhood.mean(axis=0)
            covariance = centered.T @ centered / max(len(neighborhood), 1)
            eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
            total = float(eigenvalues.sum())
            curvatures.append(float(eigenvalues[0] / total) if total > 0 else 0.0)

    counts = np.asarray(neighbor_counts, dtype=float)
    curvature_values = np.asarray(curvatures, dtype=float)
    return {
        "estimated_normals": True,
        "normal_mean": normals.mean(axis=0).tolist(),
        "normal_abs_z_mean": float(np.mean(np.abs(normals[:, 2]))),
        "density_radius": float(density_radius),
        "local_density_mean": float(counts.mean()) if len(counts) else 0.0,
        "local_density_min": int(counts.min()) if len(counts) else 0,
        "local_density_max": int(counts.max()) if len(counts) else 0,
        "curvature_mean": float(curvature_values.mean()) if len(curvature_values) else 0.0,
    }


def _sample_indices(count: int, sample_count: int, rng: np.random.Generator) -> np.ndarray:
    if count <= sample_count:
        return np.arange(count, dtype=int)
    return np.sort(rng.choice(count, size=sample_count, replace=False))


def _save_registration_figure(
    path: Path,
    source: np.ndarray,
    target: np.ndarray,
    aligned: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11, 5))
    before = fig.add_subplot(121, projection="3d")
    after = fig.add_subplot(122, projection="3d")
    _scatter_cloud(before, target, "#1f77b4", "target")
    _scatter_cloud(before, source, "#d62728", "source before")
    before.set_title("Before ICP")
    before.legend(loc="best")
    _scatter_cloud(after, target, "#1f77b4", "target")
    _scatter_cloud(after, aligned, "#2ca02c", "source after")
    after.set_title("After ICP")
    after.legend(loc="best")
    all_points = np.vstack([source, target, aligned])
    _set_axes_equal(before, all_points)
    _set_axes_equal(after, all_points)
    for axis in [before, after]:
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_segmentation_figure(
    path: Path,
    points: np.ndarray,
    labels: np.ndarray,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = np.asarray(points, dtype=float)
    colors = label_colors(labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for label in sorted(int(value) for value in np.unique(labels)):
        mask = labels == label
        name = "noise" if label < 0 else f"cluster {label}"
        color = colors[np.flatnonzero(mask)[0]]
        ax.scatter(
            pts[mask, 0],
            pts[mask, 1],
            pts[mask, 2],
            s=4,
            alpha=0.8,
            color=color,
            label=name,
        )
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="best")
    _set_axes_equal(ax, pts)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_bounding_box_or_normals_figure(
    path: Path,
    points: np.ndarray,
    corners: np.ndarray,
    normals: np.ndarray,
    rng: np.random.Generator,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = np.asarray(points, dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    _scatter_cloud(ax, pts, "#1f77b4", "processed")
    for start, end in _aabb_edges():
        segment = corners[[start, end]]
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="#d62728", linewidth=1.5)
    normal_indices = _sample_indices(len(pts), min(40, len(pts)), rng)
    normal_scale = max(float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) / 25.0, 0.02)
    ax.quiver(
        pts[normal_indices, 0],
        pts[normal_indices, 1],
        pts[normal_indices, 2],
        normals[normal_indices, 0],
        normals[normal_indices, 1],
        normals[normal_indices, 2],
        length=normal_scale,
        color="#2ca02c",
        linewidth=0.8,
        normalize=True,
    )
    ax.set_title("AABB and estimated normals")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    _set_axes_equal(ax, np.vstack([pts, corners]))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _scatter_cloud(ax: Any, points: np.ndarray, color: str, label: str) -> None:
    pts = np.asarray(points, dtype=float)
    if len(pts) > 3000:
        step = int(np.ceil(len(pts) / 3000))
        pts = pts[::step]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=3, alpha=0.72, color=color, label=label)


def _aabb_edges() -> list[tuple[int, int]]:
    return [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]


def _set_axes_equal(ax: Any, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    if radius <= 0:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def _format_pipeline_report(
    metrics: dict[str, Any],
    figure_paths: dict[str, Path],
    processed_cloud_path: Path,
    transformation_path: Path,
) -> str:
    input_metrics = metrics["input"]
    preprocessing = metrics["preprocessing"]
    registration = metrics["registration"]
    segmentation = metrics["segmentation"]
    features = metrics["features"]
    runtime = metrics["runtime"]
    lines = [
        "# PointCloud-GeoLab Portfolio Demo",
        "",
        (
            "PointCloud-GeoLab is a lightweight point-cloud processing and visualization "
            "lab for registration, preprocessing, segmentation, and portfolio-ready "
            "geometry experiments."
        ),
        "",
        "## Pipeline Steps",
        "",
        "1. Load a demo point cloud and record basic geometry metadata.",
        "2. Apply voxel downsampling and statistical outlier filtering.",
        "3. Estimate local normals, density, and curvature statistics.",
        "4. Run point-to-point ICP and export the transformation matrix.",
        "5. Run DBSCAN clustering and summarize cluster sizes and noise.",
        "6. Save static figures and machine-readable metrics.",
        "",
        "## Runtime",
        "",
        f"- Total runtime: {runtime['total_seconds']:.3f} seconds",
        "",
        "## Input Point Cloud",
        "",
        f"- Points: {input_metrics['num_points']}",
        f"- Dimension: {input_metrics['dimension']}",
        f"- Bounds min: `{input_metrics['bounds']['min']}`",
        f"- Bounds max: `{input_metrics['bounds']['max']}`",
        f"- Has color: {input_metrics['has_color']}",
        f"- Has normals: {input_metrics['has_normals']}",
        "",
        "## Preprocessing Results",
        "",
        f"- Before: {preprocessing['num_points_before']} points",
        f"- After voxel/outlier filtering: {preprocessing['num_points_after']} points",
        f"- Downsample ratio: {preprocessing['downsample_ratio']:.3f}",
        f"- Voxel size: {preprocessing['voxel_size']:.6f}",
        f"- Note: {preprocessing['note']}",
        "",
        "## Feature and Geometry Results",
        "",
        f"- Mean local density: {features['local_density_mean']:.2f}",
        f"- Mean curvature proxy: {features['curvature_mean']:.6f}",
        f"- Mean absolute Z normal component: {features['normal_abs_z_mean']:.3f}",
        "",
        "## Registration Results",
        "",
        f"- RMSE before ICP: {registration['rmse_before']:.6f}",
        f"- RMSE after ICP: {registration['rmse_after']:.6f}",
        f"- ICP iterations: {registration['iterations']}",
        f"- Transformation JSON: `{_relative_artifact(transformation_path)}`",
        f"- Note: {registration['note']}",
        "",
        "Transformation matrix:",
        "",
        "```text",
        _format_matrix(registration["transformation"]),
        "```",
        "",
        "## Segmentation Results",
        "",
        f"- Clusters: {segmentation['num_clusters']}",
        f"- Cluster sizes: `{segmentation['cluster_sizes']}`",
        f"- Noise ratio: {segmentation['noise_ratio']:.3f}",
        "",
        "## Output Figures",
        "",
    ]
    for label, path in figure_paths.items():
        lines.append(f"- {label}: [{path.name}]({_relative_artifact(path)})")
    lines.extend(
        [
            "",
            "## Generated File Index",
            "",
            "| File | Purpose |",
            "|---|---|",
            "| `report.md` | Human-readable portfolio report. |",
            (
                "| `metrics.json` | Machine-readable input, preprocessing, registration, "
                "segmentation, and runtime metrics. |"
            ),
            "| `figures/raw_pointcloud.png` | Raw input cloud projection. |",
            "| `figures/downsampled.png` | Downsampled and filtered cloud projection. |",
            "| `figures/registration_before_after.png` | ICP before/after comparison. |",
            "| `figures/segmentation_result.png` | DBSCAN cluster labels. |",
            "| `figures/bounding_box_or_normals.png` | AABB and local normal visualization. |",
            "| `artifacts/processed_cloud.ply` | Processed output cloud. |",
            "| `artifacts/transformation.json` | ICP transformation and RMSE summary. |",
            "",
            "## Artifacts",
            "",
            f"- Processed cloud: `{_relative_artifact(processed_cloud_path)}`",
            f"- Transformation: `{_relative_artifact(transformation_path)}`",
            "",
            "## Current Limitations",
            "",
            "- This pipeline is a compact portfolio demo, not a replacement for Open3D or PCL.",
            "- DBSCAN uses a simple global radius and may need tuning for very uneven densities.",
            "- ICP is a local optimizer; it is stable for the demo pair but not a global matcher.",
            "- Normals and curvature are estimated with local PCA and are not globally oriented.",
            "",
            "## Interviewer Focus Points",
            "",
            "- From-scratch KDTree-backed neighborhoods used by ICP, normal estimation, "
            "and DBSCAN.",
            "- Deterministic metrics and artifacts suitable for CI smoke testing.",
            "- Clear separation between IO, preprocessing, registration, segmentation, "
            "and CLI layers.",
            "- Honest boundaries around simplified algorithms and optional heavy dependencies.",
            "",
        ]
    )
    return "\n".join(lines)


def _format_matrix(matrix: list[list[float]]) -> str:
    return "\n".join(
        "[" + " ".join(f"{float(value): .6f}" for value in row) + "]" for row in matrix
    )


def _relative_artifact(path: Path) -> str:
    parts = path.parts
    if "figures" in parts:
        index = parts.index("figures")
        return "/".join(parts[index:])
    if "artifacts" in parts:
        index = parts.index("artifacts")
        return "/".join(parts[index:])
    return path.name


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
