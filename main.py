"""Command-line interface for PointCloud-GeoLab."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from pointcloud_geolab.geometry import compute_aabb, compute_obb, pca_analysis
from pointcloud_geolab.io.pointcloud_io import load_point_cloud, save_point_cloud
from pointcloud_geolab.io.visualization import (
    save_error_curve,
    save_point_cloud_projection,
    visualize_point_clouds,
)
from pointcloud_geolab.preprocessing import (
    estimate_normals,
    remove_radius_outliers,
    remove_statistical_outliers,
    voxel_downsample,
)
from pointcloud_geolab.registration.icp import point_to_point_icp
from pointcloud_geolab.segmentation.ransac_plane import ransac_plane_fitting


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PointCloud-GeoLab: point cloud registration and 3D geometry processing"
    )
    parser.add_argument("--mode", required=True, choices=["icp", "plane", "geometry", "preprocess"])
    parser.add_argument("--source", help="source point cloud for ICP")
    parser.add_argument("--target", help="target point cloud for ICP")
    parser.add_argument("--input", help="input point cloud")
    parser.add_argument("--output", help="output point cloud")
    parser.add_argument("--voxel-size", type=float, default=0.0)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--visualize", action="store_true", help="open an interactive Open3D window")
    parser.add_argument("--save-results", action="store_true", help="save PNG outputs under results/")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--statistical-nb-neighbors", type=int, default=16)
    parser.add_argument("--statistical-std-ratio", type=float, default=2.0)
    parser.add_argument("--radius", type=float, default=0.0)
    parser.add_argument("--min-neighbors", type=int, default=4)
    parser.add_argument("--estimate-normals", action="store_true")
    parser.add_argument("--max-correspondence-distance", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.mode == "icp":
            _run_icp(args)
        elif args.mode == "plane":
            _run_plane(args)
        elif args.mode == "geometry":
            _run_geometry(args)
        elif args.mode == "preprocess":
            _run_preprocess(args)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _require_path(value: str | None, name: str) -> str:
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _run_icp(args: argparse.Namespace) -> None:
    source_path = _require_path(args.source, "--source")
    target_path = _require_path(args.target, "--target")
    source = load_point_cloud(source_path)
    target = load_point_cloud(target_path)
    if args.voxel_size > 0:
        source = voxel_downsample(source, args.voxel_size)
        target = voxel_downsample(target, args.voxel_size)

    result = point_to_point_icp(
        source,
        target,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        max_correspondence_distance=args.max_correspondence_distance,
    )

    print("ICP Registration Result")
    print("-----------------------")
    print(f"Iterations: {result.iterations}")
    print(f"Initial RMSE: {result.initial_rmse:.6f}")
    print(f"Final RMSE: {result.final_rmse:.6f}")
    print(f"Fitness: {result.fitness:.4f}")
    print(f"Converged: {result.converged}")
    print("\nRotation Matrix:")
    print(np.array2string(result.rotation, precision=6, suppress_small=True))
    print("\nTranslation Vector:")
    print(np.array2string(result.translation, precision=6, suppress_small=True))
    print("\nTransformation Matrix:")
    print(np.array2string(result.transformation, precision=6, suppress_small=True))

    results_dir = Path(args.results_dir)
    if args.save_results:
        save_point_cloud_projection(
            results_dir / "icp_before.png",
            [source, target],
            labels=["source", "target"],
            title="ICP Before",
        )
        save_point_cloud_projection(
            results_dir / "icp_after.png",
            [result.aligned_points, target],
            labels=["aligned source", "target"],
            title="ICP After",
        )
        save_error_curve(results_dir / "icp_error_curve.png", result.rmse_history)
        save_point_cloud(results_dir / "aligned_source.ply", result.aligned_points)

    if args.visualize:
        visualize_point_clouds([result.aligned_points, target], window_name="ICP Result")


def _run_plane(args: argparse.Namespace) -> None:
    input_path = _require_path(args.input, "--input")
    points = load_point_cloud(input_path)
    if args.voxel_size > 0:
        points = voxel_downsample(points, args.voxel_size)
    result = ransac_plane_fitting(
        points,
        threshold=args.threshold,
        max_iterations=args.max_iterations,
        seed=args.seed,
    )

    a, b, c, d = result.plane_model
    print("RANSAC Plane Fitting Result")
    print("---------------------------")
    print(f"Best plane: {a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")
    print(f"Inliers: {len(result.inliers)}")
    print(f"Outliers: {len(result.outliers)}")
    print(f"Inlier Ratio: {result.inlier_ratio:.2%}")

    inlier_points = points[result.inliers]
    outlier_points = points[result.outliers]
    if args.save_results:
        save_point_cloud_projection(
            Path(args.results_dir) / "ransac_plane.png",
            [inlier_points, outlier_points],
            labels=["plane inliers", "outliers"],
            title="RANSAC Plane",
        )
        save_point_cloud(Path(args.results_dir) / "plane_inliers.ply", inlier_points)
        save_point_cloud(Path(args.results_dir) / "plane_outliers.ply", outlier_points)
    if args.visualize:
        visualize_point_clouds([inlier_points, outlier_points], window_name="RANSAC Plane")


def _run_geometry(args: argparse.Namespace) -> None:
    input_path = _require_path(args.input, "--input")
    points = load_point_cloud(input_path)
    aabb = compute_aabb(points)
    obb = compute_obb(points)
    pca = pca_analysis(points)

    print("Point Cloud Geometry")
    print("--------------------")
    print(f"Center: {np.array2string(points.mean(axis=0), precision=6, suppress_small=True)}")
    print(f"AABB Extent: {np.array2string(aabb.extent, precision=6, suppress_small=True)}")
    print(f"OBB Extent: {np.array2string(obb.extent, precision=6, suppress_small=True)}")
    print("\nPCA Eigenvalues:")
    print(np.array2string(pca.eigenvalues, precision=6, suppress_small=True))
    print("\nMain Direction:")
    print(np.array2string(pca.eigenvectors[:, 0], precision=6, suppress_small=True))

    if args.save_results:
        save_point_cloud_projection(
            Path(args.results_dir) / "obb_visualization.png",
            [points, obb.corners],
            labels=["points", "OBB corners"],
            title="PCA-based OBB",
        )
    if args.visualize:
        visualize_point_clouds([points], window_name="Geometry Analysis")


def _run_preprocess(args: argparse.Namespace) -> None:
    input_path = _require_path(args.input, "--input")
    points = load_point_cloud(input_path)
    original_count = len(points)

    if args.voxel_size > 0:
        points = voxel_downsample(points, args.voxel_size)
    after_downsample = len(points)

    points, statistical_inliers = remove_statistical_outliers(
        points,
        nb_neighbors=args.statistical_nb_neighbors,
        std_ratio=args.statistical_std_ratio,
    )
    after_statistical = len(points)

    if args.radius > 0:
        points, _ = remove_radius_outliers(points, radius=args.radius, min_neighbors=args.min_neighbors)
    after_radius = len(points)

    normals = None
    if args.estimate_normals:
        normals = estimate_normals(points)

    if args.output:
        save_point_cloud(args.output, points)

    print("Preprocessing Result")
    print("--------------------")
    print(f"Original points: {original_count}")
    print(f"After voxel downsample: {after_downsample}")
    print(f"After statistical filter: {after_statistical}")
    print(f"After radius filter: {after_radius}")
    print(f"Kept statistical inliers: {len(statistical_inliers)}")
    if normals is not None:
        print(f"Estimated normals: {normals.shape}")

    if args.save_results:
        save_point_cloud_projection(
            Path(args.results_dir) / "preprocessing.png",
            [points],
            labels=["preprocessed"],
            title="Preprocessed Point Cloud",
        )
    if args.visualize:
        visualize_point_clouds([points], window_name="Preprocessed Point Cloud")


if __name__ == "__main__":
    raise SystemExit(main())

