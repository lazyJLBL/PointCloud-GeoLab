"""Markdown and HTML report formatting for the portfolio pipeline."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import numpy as np


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
            "| `report.html` | Static browser-readable portfolio report. |",
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
            "- ICP is a local optimizer; it works for the demo pair but is not a global matcher.",
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


def _format_pipeline_html_report(
    metrics: dict[str, Any],
    figure_paths: dict[str, Path],
    processed_cloud_path: Path,
    transformation_path: Path,
    markdown_report: str,
) -> str:
    registration = metrics["registration"]
    segmentation = metrics["segmentation"]
    preprocessing = metrics["preprocessing"]
    summary_cards = [
        ("Input points", metrics["input"]["num_points"]),
        ("Processed points", preprocessing["num_points_after"]),
        ("RMSE before ICP", f"{registration['rmse_before']:.6f}"),
        ("RMSE after ICP", f"{registration['rmse_after']:.6f}"),
        ("Clusters", segmentation["num_clusters"]),
        ("Noise ratio", f"{segmentation['noise_ratio']:.3f}"),
    ]
    figure_html = "\n".join(
        (
            "<figure>"
            f'<img src="{escape(_relative_artifact(path))}" alt="{escape(label)}">'
            f"<figcaption>{escape(label.replace('_', ' ').title())}</figcaption>"
            "</figure>"
        )
        for label, path in figure_paths.items()
    )
    card_html = "\n".join(
        (
            '<div class="metric">'
            f"<span>{escape(label)}</span>"
            f"<strong>{escape(str(value))}</strong>"
            "</div>"
        )
        for label, value in summary_cards
    )
    metrics_json = escape(json.dumps(_json_ready(metrics), indent=2))
    markdown_excerpt = escape(_markdown_excerpt(markdown_report))
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="utf-8">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1">',
            "  <title>PointCloud-GeoLab Portfolio Report</title>",
            "  <style>",
            "    body { font-family: system-ui, sans-serif; margin: 2rem; color: #1f2933; }",
            "    main { max-width: 1100px; margin: 0 auto; }",
            "    h1, h2 { line-height: 1.2; }",
            "    .summary { display: grid; }",
            "    .summary { grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); }",
            "    .summary { gap: 0.75rem; }",
            "    .metric { border: 1px solid #d7dee8; padding: 0.75rem; }",
            "    .metric { border-radius: 6px; background: #f8fafc; }",
            "    .metric span { display: block; font-size: 0.85rem; color: #52606d; }",
            "    .metric strong { display: block; margin-top: 0.25rem; font-size: 1.15rem; }",
            "    .figures { display: grid; }",
            "    .figures { grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }",
            "    .figures { gap: 1rem; }",
            "    figure { margin: 0; border: 1px solid #d7dee8; padding: 0.75rem; }",
            "    figure { border-radius: 6px; }",
            "    img { max-width: 100%; height: auto; display: block; }",
            "    figcaption { margin-top: 0.5rem; color: #52606d; font-size: 0.9rem; }",
            "    pre { overflow-x: auto; background: #111827; color: #f9fafb; }",
            "    pre { padding: 1rem; border-radius: 6px; }",
            "    a { color: #0969da; }",
            "  </style>",
            "</head>",
            "<body>",
            "<main>",
            "  <h1>PointCloud-GeoLab Portfolio Report</h1>",
            "  <p>Static report generated from deterministic synthetic demo artifacts.</p>",
            "  <section>",
            "    <h2>Summary</h2>",
            f'    <div class="summary">{card_html}</div>',
            "  </section>",
            "  <section>",
            "    <h2>Figures</h2>",
            f'    <div class="figures">{figure_html}</div>',
            "  </section>",
            "  <section>",
            "    <h2>Artifacts</h2>",
            "    <ul>",
            '      <li><a href="report.md">Markdown report</a></li>',
            '      <li><a href="metrics.json">Metrics JSON</a></li>',
            (
                '      <li><a href="'
                f"{escape(_relative_artifact(processed_cloud_path))}"
                '">Processed cloud</a></li>'
            ),
            (
                '      <li><a href="'
                f"{escape(_relative_artifact(transformation_path))}"
                '">Transformation JSON</a></li>'
            ),
            "    </ul>",
            "  </section>",
            "  <section>",
            "    <h2>Metrics JSON</h2>",
            f"    <pre>{metrics_json}</pre>",
            "  </section>",
            "  <section>",
            "    <h2>Limitations</h2>",
            "    <ul>",
            "      <li>This synthetic demo is a smoke test, not real-data validation.</li>",
            "      <li>The pipeline is a compact demo, not a PCL/Open3D replacement.</li>",
            "      <li>ICP is local and does not replace global registration.</li>",
            "      <li>Benchmark and memory numbers are local machine references only.</li>",
            "    </ul>",
            "  </section>",
            "  <section>",
            "    <h2>Markdown Report Excerpt</h2>",
            f"    <pre>{markdown_excerpt}</pre>",
            "  </section>",
            "</main>",
            "</body>",
            "</html>",
            "",
        ]
    )


def _markdown_excerpt(markdown_report: str, max_lines: int = 80) -> str:
    lines = markdown_report.splitlines()
    if len(lines) <= max_lines:
        return markdown_report
    return "\n".join([*lines[:max_lines], "..."])


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
