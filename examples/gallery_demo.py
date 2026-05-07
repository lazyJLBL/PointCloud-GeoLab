"""Build reproducible gallery assets for the portfolio README."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pointcloud_geolab.api import (
    run_benchmark,
    run_extract_primitives,
    run_ground_object_segmentation,
)
from pointcloud_geolab.datasets import make_mixed_scene
from pointcloud_geolab.io import save_point_cloud
from pointcloud_geolab.io.visualization import save_point_cloud_projection
from pointcloud_geolab.registration import multiscale_icp, point_to_point_icp, robust_icp
from pointcloud_geolab.utils.transform import apply_transform, rotation_matrix_from_euler
from pointcloud_geolab.visualization import export_point_cloud_html, label_colors


def main() -> int:
    rng = np.random.default_rng(500)
    gallery = ROOT / "outputs" / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)

    _registration_gallery(gallery, rng)
    _primitive_gallery(gallery)
    _segmentation_gallery(gallery, rng)
    _benchmark_gallery(gallery)
    _write_readme_gallery(gallery)

    print(f"Gallery assets written to {gallery}")
    return 0


def _registration_gallery(gallery: Path, rng: np.random.Generator) -> None:
    target = rng.normal(scale=0.35, size=(260, 3))
    target[:80] += np.asarray([0.6, 0.0, 0.2])
    rotation = rotation_matrix_from_euler(0.05, -0.04, 0.08)
    translation = np.asarray([0.08, -0.04, 0.03])
    source_clean = apply_transform(target, rotation, translation)
    outliers = rng.uniform(-1.4, 1.4, size=(60, 3))
    source_with_outliers = np.vstack([source_clean, outliers])

    plain = point_to_point_icp(source_clean, target, max_iterations=50, tolerance=1e-8)
    robust = robust_icp(
        source_with_outliers,
        target,
        robust_kernel="huber",
        trim_ratio=0.8,
        max_iterations=50,
        max_correspondence_distance=0.5,
    )
    multi = multiscale_icp(
        source_clean,
        target,
        voxel_sizes=[0.25, 0.12, 0.06],
        max_iterations_per_level=20,
    )

    save_point_cloud_projection(
        gallery / "registration_before_after.png",
        [source_clean, plain.aligned_points, target],
        labels=["source", "aligned", "target"],
        title="Registration Before/After",
    )
    _plot_curve(
        gallery / "multiscale_icp_curve.png",
        [item["rmse"] for item in multi.diagnostics],
        "Multi-scale ICP RMSE by Level",
        "Level",
        "RMSE",
    )
    _plot_bars(
        gallery / "robust_icp_outlier_comparison.png",
        ["plain", "robust"],
        [
            point_to_point_icp(source_with_outliers, target, max_iterations=50).final_rmse,
            robust.final_rmse,
        ],
        "ICP Under Outliers",
        "Final RMSE",
    )


def _primitive_gallery(gallery: Path) -> None:
    scene, _ = make_mixed_scene(random_state=510, noise=0.006, outliers=45)
    path = gallery / "primitive_scene.ply"
    save_point_cloud(path, scene)
    result = run_extract_primitives(
        path,
        output=gallery / "primitive_extraction_scene.ply",
        output_dir=gallery / "primitive_extraction",
        threshold=0.04,
        max_models=3,
        min_inliers=40,
    )
    labels = np.asarray(result.to_dict()["data"].get("labels", np.full(len(scene), -1)), dtype=int)
    _write_html_or_fallback(
        scene,
        labels,
        gallery / "primitive_extraction_scene.html",
        "Sequential Primitive Extraction",
    )


def _segmentation_gallery(gallery: Path, rng: np.random.Generator) -> None:
    ground_xy = rng.uniform([-2.0, -1.5], [2.0, 1.5], size=(500, 2))
    ground = np.column_stack([ground_xy, rng.normal(0.0, 0.004, size=len(ground_xy))])
    objects = [
        rng.normal(loc=[0.6, 0.4, 0.35], scale=[0.08, 0.08, 0.12], size=(90, 3)),
        rng.normal(loc=[-0.8, -0.3, 0.45], scale=[0.10, 0.07, 0.15], size=(110, 3)),
        rng.normal(loc=[1.2, -0.7, 0.25], scale=[0.06, 0.06, 0.08], size=(80, 3)),
    ]
    scene = np.vstack([ground, *objects])
    path = gallery / "segmentation_ground_objects.ply"
    save_point_cloud(path, scene)
    result = run_ground_object_segmentation(
        path,
        output=gallery / "segmentation_ground_objects_colored.ply",
        output_dir=gallery / "ground_segmentation",
        eps=0.18,
        min_points=20,
        export_report=gallery / "cluster_report.md",
    )
    labels = np.asarray(result.to_dict()["data"]["labels"], dtype=int)
    _write_html_or_fallback(
        scene,
        labels,
        gallery / "segmentation_ground_objects.html",
        "Ground Removal and Object Clustering",
    )


def _benchmark_gallery(gallery: Path) -> None:
    kdtree = run_benchmark(
        "kdtree", output_dir=gallery / "kdtree_benchmark", quick=True, queries=25
    )
    ransac = run_benchmark("ransac", output_dir=gallery / "ransac_benchmark", quick=True)
    _copy_artifact(kdtree.artifacts["benchmark_plot"], gallery / "kdtree_benchmark.png")
    _copy_artifact(ransac.artifacts["benchmark_plot"], gallery / "ransac_outlier_benchmark.png")


def _write_readme_gallery(gallery: Path) -> None:
    content = """# README Gallery

| Asset | Purpose |
|---|---|
| registration_before_after.png | ICP alignment before/after evidence |
| multiscale_icp_curve.png | Coarse-to-fine ICP diagnostics |
| robust_icp_outlier_comparison.png | Robust ICP under outliers |
| primitive_extraction_scene.html | Sequential primitive extraction |
| segmentation_ground_objects.html | Ground removal and object clustering |
| kdtree_benchmark.png | Spatial index benchmark |
| ransac_outlier_benchmark.png | RANSAC robustness benchmark |
"""
    (gallery / "README_gallery.md").write_text(content, encoding="utf-8")


def _plot_curve(path: Path, values: list[float], title: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(values)), values, marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_bars(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=["#4c78a8", "#f58518"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_html_or_fallback(points: np.ndarray, labels: np.ndarray, path: Path, title: str) -> None:
    try:
        export_point_cloud_html(points, label_colors(labels), path, title=title)
    except ImportError:
        path.write_text(f"<html><body><h1>{title}</h1><p>Plotly not installed.</p></body></html>")


def _copy_artifact(source: str, target: Path) -> None:
    target.write_bytes(Path(source).read_bytes())


if __name__ == "__main__":
    raise SystemExit(main())
