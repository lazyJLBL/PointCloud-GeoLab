"""Segment a user-provided KITTI-like Velodyne frame after ground removal."""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from html import escape
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pointcloud_geolab.datasets import load_velodyne_frame
from pointcloud_geolab.segmentation import ground_object_segmentation, write_cluster_report
from pointcloud_geolab.visualization import label_colors, save_colored_point_cloud


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frame",
        type=Path,
        default=ROOT / "data" / "external" / "kitti" / "velodyne" / "000000.bin",
        help="User-provided KITTI-like float32 xyzi .bin frame.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "kitti_segmentation")
    parser.add_argument("--eps", type=float, default=0.65)
    parser.add_argument("--min-points", type=int, default=12)
    parser.add_argument("--max-points", type=int, default=12000)
    parser.add_argument("--ground-threshold", type=float, default=0.18)
    parser.add_argument("--ground-angle-threshold", type=float, default=35.0)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.frame.exists():
        print(
            "KITTI frame not found. Expected a user-provided Velodyne `.bin` file at "
            f"{args.frame}. See docs/datasets.md for the download layout.",
            file=sys.stderr,
        )
        return 2

    try:
        result = run_kitti_segmentation(args)
    except Exception as exc:
        print(f"KITTI segmentation failed for {args.frame}: {exc}", file=sys.stderr)
        return 1

    metrics = result["metrics"]
    print(f"Points: {metrics['input']['points']}")
    print(f"Ground points: {metrics['segmentation']['ground_points']}")
    print(f"Object clusters: {metrics['segmentation']['clusters']}")
    print(f"Artifacts: {args.output_dir}")
    return 0


def run_kitti_segmentation(args: argparse.Namespace) -> dict[str, Any]:
    """Run the KITTI-like segmentation workflow and write artifacts."""

    validate_args(args)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    started_tracing = not tracemalloc.is_tracing()
    if started_tracing:
        tracemalloc.start()
    started = time.perf_counter()
    timings: dict[str, float] = {}
    try:
        load_started = time.perf_counter()
        xyzi = load_velodyne_frame(args.frame, include_intensity=True)
        timings["load_seconds"] = time.perf_counter() - load_started
        if len(xyzi) > args.max_points:
            xyzi = xyzi[: args.max_points]
        validate_frame(args.frame, xyzi)
        points = xyzi[:, :3]
        intensities = xyzi[:, 3]

        segment_started = time.perf_counter()
        segmentation = ground_object_segmentation(
            points,
            ground_threshold=args.ground_threshold,
            ground_axis="z",
            ground_angle_threshold=args.ground_angle_threshold,
            cluster_method="euclidean",
            eps=args.eps,
            min_points=args.min_points,
            seed=args.seed,
        )
        timings["segmentation_seconds"] = time.perf_counter() - segment_started

        artifact_started = time.perf_counter()
        artifacts = write_outputs(output_dir, args.frame, points, intensities, segmentation)
        timings["artifact_seconds"] = time.perf_counter() - artifact_started
        timings["total_seconds"] = time.perf_counter() - started
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        metrics = build_metrics(
            args=args,
            points=points,
            intensities=intensities,
            segmentation=segmentation,
            timings=timings,
            memory={
                "available": True,
                "method": "tracemalloc",
                "current_bytes": int(current_bytes),
                "peak_bytes": int(peak_bytes),
                "note": "Local Python allocation peak for this one-frame workflow.",
            },
            artifacts=artifacts,
        )
        metrics_path = output_dir / "metrics.json"
        report_path = output_dir / "report.md"
        html_path = output_dir / "report.html"
        artifacts.update(
            {
                "metrics_json": str(metrics_path),
                "report": str(report_path),
                "html_report": str(html_path),
            }
        )
        metrics["artifacts"] = artifacts
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        report_path.write_text(format_markdown_report(metrics), encoding="utf-8")
        html_path.write_text(format_html_report(metrics), encoding="utf-8")
        return {"metrics": metrics, "artifacts": artifacts}
    finally:
        if started_tracing and tracemalloc.is_tracing():
            tracemalloc.stop()


def write_outputs(
    output_dir: Path,
    frame: Path,
    points: np.ndarray,
    intensities: np.ndarray,
    result,
) -> dict[str, str]:
    """Write point cloud, figures, and cluster report artifacts."""

    cluster_cloud = output_dir / "kitti_clusters.ply"
    bev_path = output_dir / "kitti_bev.png"
    clusters_path = output_dir / "kitti_clusters.png"
    histogram_path = output_dir / "kitti_height_histogram.png"
    cluster_report = output_dir / "cluster_report.md"
    save_colored_point_cloud(cluster_cloud, points, result.labels)
    save_bev_plot(bev_path, points, intensities, None, "KITTI-like LiDAR BEV")
    save_bev_plot(
        clusters_path,
        points,
        intensities,
        result.labels,
        "Ground Removal and Object Clustering",
    )
    save_height_histogram(histogram_path, points)
    write_cluster_report(result, cluster_report)
    return {
        "frame": str(frame),
        "cluster_cloud": str(cluster_cloud),
        "bev_figure": str(bev_path),
        "cluster_figure": str(clusters_path),
        "height_histogram": str(histogram_path),
        "cluster_report": str(cluster_report),
    }


def validate_args(args: argparse.Namespace) -> None:
    """Validate KITTI workflow parameters before creating artifacts."""

    if args.eps <= 0:
        raise ValueError("--eps must be positive")
    if args.min_points < 1:
        raise ValueError("--min-points must be at least 1")
    if args.max_points < 3:
        raise ValueError("--max-points must be at least 3")
    if args.ground_threshold <= 0:
        raise ValueError("--ground-threshold must be positive")
    if not 0.0 <= args.ground_angle_threshold <= 90.0:
        raise ValueError("--ground-angle-threshold must be between 0 and 90 degrees")


def validate_frame(path: Path, xyzi: np.ndarray) -> None:
    """Validate a loaded KITTI-like frame with path-aware messages."""

    if xyzi.ndim != 2 or xyzi.shape[1] != 4:
        raise ValueError(f"{path}: KITTI-like frame must have shape (N, 4)")
    if len(xyzi) < 3:
        raise ValueError(f"{path}: KITTI-like frame has {len(xyzi)} points; at least 3 required")
    if not np.all(np.isfinite(xyzi)):
        raise ValueError(f"{path}: KITTI-like frame contains NaN or infinite values")


def build_metrics(
    args: argparse.Namespace,
    points: np.ndarray,
    intensities: np.ndarray,
    segmentation,
    timings: dict[str, float],
    memory: dict[str, Any],
    artifacts: dict[str, str],
) -> dict[str, Any]:
    """Build the machine-readable metrics payload."""

    return {
        "workflow": "kitti_lidar_segmentation",
        "scope": "user-provided KITTI-like single-frame workflow, not an official benchmark",
        "input": {
            "frame": str(args.frame),
            "points": int(len(points)),
            "max_points": int(args.max_points),
            "bounds": bounds(points),
            "intensity": {
                "min": float(np.min(intensities)) if len(intensities) else 0.0,
                "max": float(np.max(intensities)) if len(intensities) else 0.0,
                "mean": float(np.mean(intensities)) if len(intensities) else 0.0,
            },
        },
        "parameters": {
            "eps": float(args.eps),
            "min_points": int(args.min_points),
            "ground_threshold": float(args.ground_threshold),
            "ground_angle_threshold": float(args.ground_angle_threshold),
            "seed": int(args.seed),
        },
        "segmentation": {
            "ground_points": int(len(segmentation.ground.ground_indices)),
            "non_ground_points": int(len(segmentation.ground.non_ground_indices)),
            "noise_points": int(len(segmentation.noise_indices)),
            "clusters": int(len(segmentation.clusters)),
            "cluster_sizes": [int(cluster.point_count) for cluster in segmentation.clusters],
            "ground_normal_angle_degrees": float(segmentation.ground.normal_angle_degrees),
            "plane_model": segmentation.ground.plane_model,
            "cluster_summaries": [cluster.to_dict() for cluster in segmentation.clusters],
        },
        "timing": {key: float(value) for key, value in timings.items()},
        "memory": memory,
        "artifacts": artifacts,
        "limitations": [
            "This is a user-provided single-frame workflow, not an official KITTI benchmark.",
            "Fixed-radius clustering is sensitive to LiDAR range and scene density.",
            "The repository does not commit real KITTI data.",
        ],
    }


def save_bev_plot(
    output_path: Path,
    points: np.ndarray,
    intensities: np.ndarray,
    labels: np.ndarray | None,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    if labels is None:
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            s=1.0,
            c=intensities,
            cmap="viridis",
        )
        fig.colorbar(scatter, ax=ax, label="intensity")
    else:
        ax.scatter(points[:, 0], points[:, 1], s=1.0, c=label_colors(labels))
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_height_histogram(output_path: Path, points: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(points[:, 2], bins=40, color="#4c78a8")
    ax.set_title("KITTI-like frame height distribution")
    ax.set_xlabel("z")
    ax.set_ylabel("points")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def format_markdown_report(metrics: dict[str, Any]) -> str:
    rows = [
        "| Metric | Value |",
        "|---|---:|",
        f"| Points | {metrics['input']['points']} |",
        f"| Ground points | {metrics['segmentation']['ground_points']} |",
        f"| Non-ground points | {metrics['segmentation']['non_ground_points']} |",
        f"| Clusters | {metrics['segmentation']['clusters']} |",
        f"| Noise points | {metrics['segmentation']['noise_points']} |",
        f"| Total seconds | {metrics['timing']['total_seconds']:.4f} |",
        f"| Peak memory bytes | {metrics['memory']['peak_bytes']} |",
    ]
    artifacts = "\n".join(f"- `{name}`: `{path}`" for name, path in metrics["artifacts"].items())
    limitations = "\n".join(f"- {item}" for item in metrics["limitations"])
    return "\n".join(
        [
            "# KITTI-like LiDAR Segmentation Result",
            "",
            "This report is generated from a user-provided KITTI-like `.bin` frame.",
            "It is not an official KITTI benchmark and no real KITTI data is committed.",
            "",
            "## Summary",
            "",
            *rows,
            "",
            "## Cluster Sizes",
            "",
            f"`{metrics['segmentation']['cluster_sizes']}`",
            "",
            "## Artifacts",
            "",
            artifacts,
            "",
            "## Limitations",
            "",
            limitations,
            "",
        ]
    )


def format_html_report(metrics: dict[str, Any]) -> str:
    artifact_links = "\n".join(
        _format_html_artifact_link(name, path) for name, path in metrics["artifacts"].items()
    )
    limitations = "\n".join(f"<li>{escape(item)}</li>" for item in metrics["limitations"])
    metrics_json = escape(json.dumps(metrics, indent=2))
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="utf-8">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1">',
            "  <title>KITTI-like LiDAR Segmentation Result</title>",
            "  <style>",
            "    body { font-family: system-ui, sans-serif; margin: 2rem; color: #1f2933; }",
            "    main { max-width: 1000px; margin: 0 auto; }",
            "    img { max-width: 100%; height: auto; }",
            "    pre { overflow-x: auto; background: #111827; color: #f9fafb; padding: 1rem; }",
            "  </style>",
            "</head>",
            "<body><main>",
            "  <h1>KITTI-like LiDAR Segmentation Result</h1>",
            "  <p>User-provided single-frame workflow; not an official KITTI benchmark.</p>",
            "  <h2>Summary</h2>",
            "  <ul>",
            f"    <li>Points: {metrics['input']['points']}</li>",
            f"    <li>Ground points: {metrics['segmentation']['ground_points']}</li>",
            f"    <li>Clusters: {metrics['segmentation']['clusters']}</li>",
            f"    <li>Peak memory bytes: {metrics['memory']['peak_bytes']}</li>",
            "  </ul>",
            "  <h2>Figures</h2>",
            '  <p><img src="kitti_bev.png" alt="KITTI-like BEV"></p>',
            '  <p><img src="kitti_clusters.png" alt="KITTI-like clusters"></p>',
            '  <p><img src="kitti_height_histogram.png" alt="Height histogram"></p>',
            "  <h2>Artifacts</h2>",
            f"  <ul>{artifact_links}</ul>",
            "  <h2>Limitations</h2>",
            f"  <ul>{limitations}</ul>",
            "  <h2>Metrics JSON</h2>",
            f"  <pre>{metrics_json}</pre>",
            "</main></body></html>",
            "",
        ]
    )


def _format_html_artifact_link(name: str, path: str) -> str:
    if name == "frame":
        return f"<li><code>{escape(name)}</code>: <code>{escape(path)}</code></li>"
    href = escape(Path(path).name)
    return f'<li><a href="{href}">{escape(name)}</a>: <code>{escape(path)}</code></li>'


def bounds(points: np.ndarray) -> dict[str, list[float]]:
    return {
        "min": points.min(axis=0).tolist(),
        "max": points.max(axis=0).tolist(),
        "extent": (points.max(axis=0) - points.min(axis=0)).tolist(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
