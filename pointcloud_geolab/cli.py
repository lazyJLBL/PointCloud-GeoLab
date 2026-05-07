"""Command-line interface for PointCloud-GeoLab."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from pointcloud_geolab.api import (
    TaskResult,
    run_benchmark,
    run_extract_primitives,
    run_geometry_analysis,
    run_global_registration,
    run_icp,
    run_infer_pointnet,
    run_plane_segmentation,
    run_portfolio_verification,
    run_preprocessing,
    run_primitive_fitting,
    run_reconstruction,
    run_segmentation,
    run_train_pointnet,
    run_visualization,
)
from pointcloud_geolab.pipeline import run_portfolio_pipeline

DEFAULTS: dict[str, dict[str, Any]] = {
    "icp": {
        "output_dir": "results",
        "voxel_size": 0.0,
        "max_iterations": 50,
        "tolerance": 1e-6,
        "max_correspondence_distance": None,
        "save_results": False,
        "visualize": False,
    },
    "plane": {
        "output_dir": "results",
        "voxel_size": 0.0,
        "threshold": 0.02,
        "max_iterations": 1000,
        "seed": 7,
        "save_results": False,
        "visualize": False,
    },
    "geometry": {
        "output_dir": "results",
        "save_results": False,
        "visualize": False,
    },
    "preprocess": {
        "output_dir": "results",
        "voxel_size": 0.0,
        "statistical_nb_neighbors": 16,
        "statistical_std_ratio": 2.0,
        "radius": 0.0,
        "min_neighbors": 4,
        "estimate_normals": False,
        "normalize": False,
        "crop_min": None,
        "crop_max": None,
        "sample_count": None,
        "sample_method": "random",
        "seed": None,
        "save_results": False,
        "visualize": False,
    },
    "register": {
        "output_dir": "outputs/registration",
        "method": "fpfh_ransac_icp",
        "icp_method": "point_to_point",
        "voxel_size": 0.05,
        "threshold": None,
        "seed": 7,
        "output": None,
        "save_transform": None,
        "save_results": False,
        "export_html": None,
        "multiscale": False,
        "voxel_sizes": None,
        "robust_kernel": "none",
        "trim_ratio": 1.0,
        "save_diagnostics": None,
    },
    "fit-primitive": {
        "output_dir": "outputs/primitives",
        "model": "plane",
        "threshold": 0.02,
        "max_iterations": 1000,
        "min_inliers": 0,
        "seed": 7,
        "output": None,
        "save_results": False,
        "export_html": None,
    },
    "extract-primitives": {
        "output_dir": "outputs/primitives",
        "models": ["plane", "sphere", "cylinder"],
        "threshold": 0.03,
        "max_models": 5,
        "min_inliers": 30,
        "max_iterations": 800,
        "seed": 7,
        "output": None,
        "export_html": None,
    },
    "segment": {
        "output_dir": "outputs/segmentation",
        "method": "dbscan",
        "eps": 0.05,
        "min_points": 20,
        "tolerance": None,
        "radius": 0.1,
        "angle_threshold": 25.0,
        "output": None,
        "export_html": None,
        "remove_ground": False,
        "ground_axis": "z",
        "ground_angle_threshold": 20.0,
        "export_report": None,
    },
    "visualize": {
        "output_dir": "outputs/visualization",
        "mode": "pointcloud",
        "labels_path": None,
        "source": None,
        "target": None,
        "transform_path": None,
    },
    "reconstruct": {
        "output_dir": "outputs/reconstruction",
        "method": "poisson",
        "output": "outputs/reconstruction/object_mesh.ply",
        "normal_radius": 0.15,
        "poisson_depth": 6,
        "alpha": 0.2,
    },
    "verify-portfolio": {
        "output_dir": "outputs",
        "quick": True,
    },
    "train-pointnet": {
        "output_dir": "outputs/ml",
        "output": "outputs/pointnet_model.pt",
        "epochs": 2,
        "batch_size": 16,
        "samples_per_class": 16,
        "points_per_sample": 128,
        "seed": 7,
    },
    "infer-pointnet": {
        "output_dir": "outputs/ml",
        "model": None,
        "points_per_sample": 128,
    },
    "benchmark": {
        "output_dir": "results",
        "benchmark_name": "kdtree",
        "suite": None,
        "quick": True,
        "full": False,
        "save_json": None,
        "save_md": None,
        "seed": 42,
        "queries": 100,
        "points": None,
    },
    "pipeline": {
        "input": "examples/demo_data",
        "output_dir": "outputs/portfolio_demo",
        "voxel_size": None,
        "eps": None,
        "min_points": 10,
        "seed": 42,
    },
}


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if _contains_legacy_mode(args_list):
        return _main_legacy(args_list)

    parser = build_parser()
    args = parser.parse_args(args_list)
    _configure_logging(args.log_level)

    if args.batch:
        return _run_batch(args)

    if not args.command:
        parser.error("a subcommand or --batch is required")

    try:
        config = _load_yaml(args.config) if args.config else {}
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    params = _merge_parameters(args.command, config, _namespace_values(args))
    result = _execute_task(args.command, params)
    _emit_result(result, args.format or "text")
    return 0 if result.success else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PointCloud-GeoLab: point cloud registration and 3D geometry processing"
    )
    _add_batch_options(parser)

    subparsers = parser.add_subparsers(dest="command")

    icp = subparsers.add_parser("icp", help="run point-to-point ICP registration")
    _add_common_options(icp)
    icp.add_argument("--source", help="source point cloud")
    icp.add_argument("--target", help="target point cloud")
    icp.add_argument("--voxel-size", type=float)
    icp.add_argument("--max-iterations", type=int)
    icp.add_argument("--tolerance", type=float)
    icp.add_argument("--max-correspondence-distance", type=float)

    plane = subparsers.add_parser("plane", help="fit a dominant plane with RANSAC")
    _add_common_options(plane)
    plane.add_argument("--input", help="input point cloud")
    plane.add_argument("--voxel-size", type=float)
    plane.add_argument("--threshold", type=float)
    plane.add_argument("--max-iterations", type=int)

    geometry = subparsers.add_parser("geometry", help="compute AABB, OBB, and PCA metrics")
    _add_common_options(geometry)
    geometry.add_argument("--input", help="input point cloud")

    preprocess = subparsers.add_parser("preprocess", help="run preprocessing filters")
    _add_common_options(preprocess)
    preprocess.add_argument("--input", help="input point cloud")
    preprocess.add_argument("--output", help="output point cloud")
    preprocess.add_argument("--voxel-size", type=float)
    preprocess.add_argument("--statistical-nb-neighbors", type=int)
    preprocess.add_argument("--statistical-std-ratio", type=float)
    preprocess.add_argument("--radius", type=float)
    preprocess.add_argument("--min-neighbors", type=int)
    preprocess.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=None)
    preprocess.add_argument("--crop-min", nargs=3, type=float)
    preprocess.add_argument("--crop-max", nargs=3, type=float)
    preprocess.add_argument("--sample-count", type=int)
    preprocess.add_argument("--sample-method", choices=["random", "farthest"])
    preprocess.add_argument(
        "--estimate-normals",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="estimate normals after filtering",
    )

    register = subparsers.add_parser("register", help="run feature-based global registration")
    _add_common_options(register)
    register.add_argument("--source", required=False, help="source point cloud")
    register.add_argument("--target", required=False, help="target point cloud")
    register.add_argument("--method", choices=["fpfh_ransac_icp", "iss_descriptor_ransac_icp"])
    register.add_argument("--icp-method", choices=["point_to_point", "point_to_plane"])
    register.add_argument("--voxel-size", type=float)
    register.add_argument("--voxel-sizes", nargs="*", type=float)
    register.add_argument("--threshold", type=float)
    register.add_argument("--multiscale", action=argparse.BooleanOptionalAction, default=None)
    register.add_argument("--robust-kernel", choices=["none", "huber", "tukey"])
    register.add_argument("--trim-ratio", type=float)
    register.add_argument("--output", type=Path)
    register.add_argument("--save-transform", type=Path)
    register.add_argument("--save-diagnostics", type=Path)
    register.add_argument("--export-html", type=Path)

    primitive = subparsers.add_parser("fit-primitive", help="fit plane/sphere/cylinder with RANSAC")
    _add_common_options(primitive)
    primitive.add_argument("--input", help="input point cloud")
    primitive.add_argument("--model", choices=["plane", "sphere", "cylinder"])
    primitive.add_argument("--threshold", type=float)
    primitive.add_argument("--max-iterations", type=int)
    primitive.add_argument("--min-inliers", type=int)
    primitive.add_argument("--output", type=Path)
    primitive.add_argument("--export-html", type=Path)

    extract = subparsers.add_parser(
        "extract-primitives",
        help="sequentially extract plane/sphere/cylinder primitives",
    )
    _add_common_options(extract, include_save_flags=False)
    extract.add_argument("--input", help="input point cloud")
    extract.add_argument("--models", nargs="+", choices=["plane", "sphere", "cylinder"])
    extract.add_argument("--threshold", type=float)
    extract.add_argument("--max-models", type=int)
    extract.add_argument("--min-inliers", type=int)
    extract.add_argument("--max-iterations", type=int)
    extract.add_argument("--output", type=Path)
    extract.add_argument("--export-html", type=Path)

    segment = subparsers.add_parser("segment", help="segment points into clusters")
    _add_common_options(segment, include_save_flags=False)
    segment.add_argument("--input", help="input point cloud")
    segment.add_argument("--method", choices=["dbscan", "euclidean", "region_growing"])
    segment.add_argument("--eps", type=float)
    segment.add_argument("--min-points", type=int)
    segment.add_argument("--tolerance", type=float)
    segment.add_argument("--radius", type=float)
    segment.add_argument("--angle-threshold", type=float)
    segment.add_argument("--remove-ground", action=argparse.BooleanOptionalAction, default=None)
    segment.add_argument("--ground-axis", choices=["x", "y", "z"])
    segment.add_argument("--ground-angle-threshold", type=float)
    segment.add_argument("--output", type=Path)
    segment.add_argument("--export-html", type=Path)
    segment.add_argument("--export-report", type=Path)

    visualize = subparsers.add_parser("visualize", help="export HTML visualizations")
    _add_common_options(visualize, include_save_flags=False)
    visualize.add_argument("--input", help="input point cloud")
    visualize.add_argument(
        "--mode",
        choices=["pointcloud", "clusters", "registration", "primitives", "inliers_outliers"],
    )
    visualize.add_argument("--labels-path", type=Path)
    visualize.add_argument("--source", type=Path)
    visualize.add_argument("--target", type=Path)
    visualize.add_argument("--transform-path", type=Path)
    visualize.add_argument("--output", type=Path, required=False)

    train = subparsers.add_parser("train-pointnet", help="train optional PointNet demo")
    _add_common_options(train, include_save_flags=False)
    train.add_argument("--output", type=Path)
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--samples-per-class", type=int)
    train.add_argument("--points-per-sample", type=int)

    infer = subparsers.add_parser("infer-pointnet", help="run optional PointNet inference")
    _add_common_options(infer, include_save_flags=False)
    infer.add_argument("--model", type=Path)
    infer.add_argument("--input", help="input point cloud")
    infer.add_argument("--points-per-sample", type=int)

    reconstruct = subparsers.add_parser("reconstruct", help="reconstruct a surface mesh")
    _add_common_options(reconstruct, include_save_flags=False)
    reconstruct.add_argument("--input", help="input point cloud")
    reconstruct.add_argument("--method", choices=["poisson", "ball_pivoting", "bpa", "alpha_shape"])
    reconstruct.add_argument("--output", type=Path)
    reconstruct.add_argument("--normal-radius", type=float)
    reconstruct.add_argument("--poisson-depth", type=int)
    reconstruct.add_argument("--alpha", type=float)

    verify = subparsers.add_parser("verify-portfolio", help="run portfolio smoke checks")
    _add_common_options(verify, include_save_flags=False)
    verify.add_argument("--quick", action=argparse.BooleanOptionalAction, default=None)

    benchmark = subparsers.add_parser("benchmark", help="run built-in benchmarks")
    _add_common_options(benchmark, include_save_flags=False)
    benchmark.add_argument(
        "benchmark_name",
        nargs="?",
        choices=["kdtree", "icp", "ransac", "registration", "segmentation", "all"],
        help="benchmark suite",
    )
    benchmark.add_argument(
        "--suite",
        choices=["kdtree", "icp", "ransac", "registration", "segmentation", "all"],
    )
    benchmark.add_argument("--output", dest="output_dir", type=Path)
    benchmark.add_argument("--quick", action=argparse.BooleanOptionalAction, default=None)
    benchmark.add_argument("--full", action=argparse.BooleanOptionalAction, default=None)
    benchmark.add_argument("--save-json", type=Path)
    benchmark.add_argument("--save-md", type=Path)
    benchmark.add_argument("--queries", type=int)
    benchmark.add_argument("--points", nargs="*", type=int)

    pipeline = subparsers.add_parser(
        "pipeline",
        help="run the portfolio demo pipeline",
        description="run the portfolio demo pipeline",
    )
    _add_common_options(pipeline, include_save_flags=False)
    pipeline.add_argument(
        "--input",
        type=Path,
        help="input point cloud file or directory; examples/demo_data falls back to data/",
    )
    pipeline.add_argument("--output", dest="output_dir", type=Path, help="portfolio output dir")
    pipeline.add_argument("--voxel-size", type=float)
    pipeline.add_argument("--eps", type=float, help="DBSCAN radius; defaults to an auto value")
    pipeline.add_argument("--min-points", type=int, help="DBSCAN min_points")

    return parser


def _add_batch_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", type=Path, help="run jobs from a YAML manifest")
    parser.add_argument("--config", type=Path, help="load task parameters from YAML")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--save-results", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--log-level", default="WARNING")


def _add_common_options(parser: argparse.ArgumentParser, include_save_flags: bool = True) -> None:
    parser.add_argument("--config", type=Path, help="load task parameters from YAML")
    parser.add_argument("--output-dir", "--results-dir", dest="output_dir", type=Path)
    parser.add_argument("--format", choices=["text", "json"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--log-level", default="WARNING")
    if include_save_flags:
        parser.add_argument("--save-results", action=argparse.BooleanOptionalAction, default=None)
        parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=None)


def _main_legacy(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="PointCloud-GeoLab legacy --mode interface. Prefer subcommands for new usage."
    )
    parser.add_argument("--mode", required=True, choices=["icp", "plane", "geometry", "preprocess"])
    parser.add_argument("--source")
    parser.add_argument("--target")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", "--results-dir", dest="output_dir", type=Path)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--voxel-size", type=float)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-results", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--statistical-nb-neighbors", type=int)
    parser.add_argument("--statistical-std-ratio", type=float)
    parser.add_argument("--radius", type=float)
    parser.add_argument("--min-neighbors", type=int)
    parser.add_argument("--estimate-normals", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-correspondence-distance", type=float)
    parser.add_argument("--log-level", default="WARNING")

    args = parser.parse_args(argv)
    _configure_logging(args.log_level)
    print(
        "Compatibility mode: --mode is deprecated; use subcommands such as "
        "`pointcloud-geolab icp` instead.",
        file=sys.stderr,
    )
    try:
        config = _load_yaml(args.config) if args.config else {}
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    params = _merge_parameters(args.mode, config, _namespace_values(args))
    result = _execute_task(args.mode, params)
    _emit_result(result, args.format)
    return 0 if result.success else 1


def _run_batch(args: argparse.Namespace) -> int:
    try:
        manifest = _load_yaml(args.batch)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    jobs = manifest.get("jobs", manifest if isinstance(manifest, list) else None)
    if not isinstance(jobs, list):
        print("Error: batch manifest must contain a top-level jobs list", file=sys.stderr)
        return 1

    root_overrides = {
        key: value
        for key, value in _namespace_values(args).items()
        if key in {"save_results", "visualize", "seed"} and value is not None
    }
    base_output_dir = Path(args.output_dir) if args.output_dir else Path("results") / "batch"

    results: list[TaskResult] = []
    for index, raw_job in enumerate(jobs, start=1):
        if not isinstance(raw_job, dict):
            results.append(
                TaskResult(
                    task=f"job:{index}",
                    success=False,
                    error="batch job must be a mapping",
                    parameters={"job_index": index},
                )
            )
            continue

        job = dict(raw_job)
        task = job.pop("task", job.pop("mode", None))
        if task == "benchmark":
            job["benchmark_name"] = job.pop(
                "benchmark_name",
                job.pop("benchmark", job.pop("name", DEFAULTS["benchmark"]["benchmark_name"])),
            )
        if task not in DEFAULTS:
            results.append(
                TaskResult(
                    task=str(task),
                    success=False,
                    error=(
                        "batch job task must be one of: icp, plane, geometry, preprocess, "
                        "register, fit-primitive, extract-primitives, segment, visualize, "
                        "reconstruct, verify-portfolio, train-pointnet, infer-pointnet, "
                        "benchmark, pipeline"
                    ),
                    parameters=job,
                )
            )
            continue

        job_name = str(raw_job.get("name") or f"{index:02d}_{task}")
        job.setdefault("output_dir", base_output_dir / job_name)
        params = _merge_parameters(task, {}, {**job, **root_overrides})
        results.append(_execute_task(task, params))

    payload = {
        "success": all(result.success for result in results),
        "jobs": [result.to_dict() for result in results],
    }
    if args.format == "json":
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print("Batch Summary")
        print("-------------")
        for index, result in enumerate(results, start=1):
            status = "ok" if result.success else "failed"
            print(f"{index}. {result.task}: {status}")
            if result.error:
                print(f"   Error: {result.error}")
    return 0 if payload["success"] else 1


def _execute_task(task: str, params: dict[str, Any]) -> TaskResult:
    if task == "icp":
        return run_icp(
            source=params.get("source"),
            target=params.get("target"),
            output_dir=params["output_dir"],
            voxel_size=params["voxel_size"],
            max_iterations=params["max_iterations"],
            tolerance=params["tolerance"],
            max_correspondence_distance=params["max_correspondence_distance"],
            save_results=params["save_results"],
            visualize=params["visualize"],
        )
    if task == "plane":
        return run_plane_segmentation(
            input_path=params.get("input"),
            output_dir=params["output_dir"],
            voxel_size=params["voxel_size"],
            threshold=params["threshold"],
            max_iterations=params["max_iterations"],
            seed=params["seed"],
            save_results=params["save_results"],
            visualize=params["visualize"],
        )
    if task == "geometry":
        return run_geometry_analysis(
            input_path=params.get("input"),
            output_dir=params["output_dir"],
            save_results=params["save_results"],
            visualize=params["visualize"],
        )
    if task == "preprocess":
        return run_preprocessing(
            input_path=params.get("input"),
            output=params.get("output"),
            output_dir=params["output_dir"],
            voxel_size=params["voxel_size"],
            statistical_nb_neighbors=params["statistical_nb_neighbors"],
            statistical_std_ratio=params["statistical_std_ratio"],
            radius=params["radius"],
            min_neighbors=params["min_neighbors"],
            estimate_normals_flag=params["estimate_normals"],
            normalize=params["normalize"],
            crop_min=params["crop_min"],
            crop_max=params["crop_max"],
            sample_count=params["sample_count"],
            sample_method=params["sample_method"],
            seed=params["seed"],
            save_results=params["save_results"],
            visualize=params["visualize"],
        )
    if task == "register":
        return run_global_registration(
            source=params.get("source"),
            target=params.get("target"),
            output=params["output"],
            save_transform=params["save_transform"],
            output_dir=params["output_dir"],
            voxel_size=params["voxel_size"],
            method=params["method"],
            icp_method=params["icp_method"],
            threshold=params["threshold"],
            seed=params["seed"],
            save_results=params["save_results"],
            export_html=params["export_html"],
            multiscale=params["multiscale"],
            voxel_sizes=params["voxel_sizes"],
            robust_kernel=params["robust_kernel"],
            trim_ratio=params["trim_ratio"],
            save_diagnostics=params["save_diagnostics"],
        )
    if task == "fit-primitive":
        return run_primitive_fitting(
            input_path=params.get("input"),
            model=params["model"],
            output=params["output"],
            output_dir=params["output_dir"],
            threshold=params["threshold"],
            max_iterations=params["max_iterations"],
            min_inliers=params["min_inliers"],
            seed=params["seed"],
            save_results=params["save_results"],
            export_html=params["export_html"],
        )
    if task == "extract-primitives":
        return run_extract_primitives(
            input_path=params.get("input"),
            models=params["models"],
            output=params["output"],
            output_dir=params["output_dir"],
            threshold=params["threshold"],
            max_models=params["max_models"],
            min_inliers=params["min_inliers"],
            max_iterations=params["max_iterations"],
            seed=params["seed"],
            export_html=params["export_html"],
        )
    if task == "segment":
        return run_segmentation(
            input_path=params.get("input"),
            output=params["output"],
            output_dir=params["output_dir"],
            method=params["method"],
            eps=params["eps"],
            min_points=params["min_points"],
            tolerance=params["tolerance"],
            radius=params["radius"],
            angle_threshold=params["angle_threshold"],
            export_html=params["export_html"],
            remove_ground=params["remove_ground"],
            ground_axis=params["ground_axis"],
            ground_angle_threshold=params["ground_angle_threshold"],
            export_report=params["export_report"],
        )
    if task == "visualize":
        return run_visualization(
            input_path=params.get("input"),
            output=params.get("output") or Path(params["output_dir"]) / "visualization.html",
            mode=params["mode"],
            labels_path=params["labels_path"],
            source=params["source"],
            target=params["target"],
            transform_path=params["transform_path"],
            output_dir=params["output_dir"],
        )
    if task == "reconstruct":
        return run_reconstruction(
            input_path=params.get("input"),
            output=params["output"],
            output_dir=params["output_dir"],
            method=params["method"],
            normal_radius=params["normal_radius"],
            poisson_depth=params["poisson_depth"],
            alpha=params["alpha"],
        )
    if task == "verify-portfolio":
        return run_portfolio_verification(
            output_dir=params["output_dir"],
            quick=params["quick"],
        )
    if task == "train-pointnet":
        return run_train_pointnet(
            output=params["output"],
            output_dir=params["output_dir"],
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            samples_per_class=params["samples_per_class"],
            points_per_sample=params["points_per_sample"],
            seed=params["seed"],
        )
    if task == "infer-pointnet":
        return run_infer_pointnet(
            model=params.get("model"),
            input_path=params.get("input"),
            output_dir=params["output_dir"],
            points_per_sample=params["points_per_sample"],
        )
    if task == "benchmark":
        full = bool(params["full"])
        quick = bool(params["quick"]) and not full
        benchmark_name = params["suite"] or params["benchmark_name"]
        return run_benchmark(
            benchmark=benchmark_name,
            output_dir=params["output_dir"],
            quick=quick,
            full=full,
            save_json=params["save_json"],
            save_md=params["save_md"],
            seed=params["seed"],
            queries=params["queries"],
            points=params["points"],
        )
    if task == "pipeline":
        return run_portfolio_pipeline(
            input_path=params["input"],
            output_dir=params["output_dir"],
            voxel_size=params["voxel_size"],
            eps=params["eps"],
            min_points=params["min_points"],
            seed=params["seed"],
        )
    return TaskResult(
        task=task,
        success=False,
        error=f"unsupported task: {task}",
        parameters=params,
    )


def _merge_parameters(
    task: str,
    config: dict[str, Any],
    cli_values: dict[str, Any],
) -> dict[str, Any]:
    params = dict(DEFAULTS[task])
    task_config = _task_config(task, config)
    params.update(task_config)
    for key, value in cli_values.items():
        if key in {"command", "mode", "batch", "config", "format", "log_level"}:
            continue
        if value is not None:
            params[key] = value
    return params


def _task_config(task: str, config: dict[str, Any]) -> dict[str, Any]:
    if not config:
        return {}
    nested = config.get(task)
    if isinstance(nested, dict):
        return nested
    ignored = {"jobs", "tasks"}
    return {key: value for key, value in config.items() if key not in ignored}


def _namespace_values(args: argparse.Namespace) -> dict[str, Any]:
    return {key: value for key, value in vars(args).items() if value is not None}


def _load_yaml(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as f:
        if file_path.suffix.lower() == ".json":
            data = json.load(f) or {}
        else:
            data = yaml.safe_load(f) or {}
    if not isinstance(data, dict) and not isinstance(data, list):
        raise ValueError(f"{file_path} must contain a YAML mapping or list")
    return data


def _emit_result(result: TaskResult, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return
    print(_format_text_result(result))


def _format_text_result(result: TaskResult) -> str:
    data = result.to_dict()
    metrics = data["metrics"]
    details = data["data"]
    artifacts = data["artifacts"]
    task = data["task"]

    if not result.success:
        return f"Error: {result.error}"

    if task == "icp":
        lines = [
            "ICP Registration Result",
            "-----------------------",
            f"Iterations: {metrics['iterations']}",
            f"Initial RMSE: {metrics['initial_rmse']:.6f}",
            f"Final RMSE: {metrics['final_rmse']:.6f}",
            f"Fitness: {metrics['fitness']:.4f}",
            f"Converged: {metrics['converged']}",
            "",
            "Rotation Matrix:",
            _json_matrix(details["rotation"]),
            "",
            "Translation Vector:",
            _json_vector(details["translation"]),
            "",
            "Transformation Matrix:",
            _json_matrix(details["transformation"]),
        ]
    elif task == "plane":
        a, b, c, d = details["plane_model"]
        lines = [
            "RANSAC Plane Fitting Result",
            "---------------------------",
            f"Best plane: {a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0",
            f"Inliers: {metrics['inliers']}",
            f"Outliers: {metrics['outliers']}",
            f"Inlier Ratio: {metrics['inlier_ratio']:.2%}",
        ]
    elif task == "geometry":
        lines = [
            "Point Cloud Geometry",
            "--------------------",
            f"Center: {_json_vector(metrics['center'])}",
            f"AABB Extent: {_json_vector(metrics['aabb_extent'])}",
            f"OBB Extent: {_json_vector(metrics['obb_extent'])}",
            "",
            "PCA Eigenvalues:",
            _json_vector(metrics["pca_eigenvalues"]),
            "",
            "Main Direction:",
            _json_vector(metrics["main_direction"]),
        ]
    elif task == "preprocess":
        lines = [
            "Preprocessing Result",
            "--------------------",
            f"Original points: {metrics['original_points']}",
            f"After voxel downsample: {metrics['after_voxel_downsample']}",
            f"After statistical filter: {metrics['after_statistical_filter']}",
            f"After radius filter: {metrics['after_radius_filter']}",
            f"Kept statistical inliers: {metrics['kept_statistical_inliers']}",
        ]
        if metrics["estimated_normals"]:
            lines.append("Estimated normals: true")
    elif task.startswith("benchmark:"):
        lines = [
            f"Benchmark Result: {metrics['benchmark']}",
            "-----------------",
            details["markdown"],
        ]
    elif task == "register":
        lines = [
            "Global Registration Result",
            "--------------------------",
            f"Coarse Fitness: {metrics['coarse_fitness']:.4f}",
            f"Coarse Inlier RMSE: {metrics['coarse_inlier_rmse']:.6f}",
            f"Refined Fitness: {metrics['refined_fitness']:.4f}",
            f"Final RMSE: {metrics['final_rmse']:.6f}",
            "",
            "Refined Transformation:",
            _json_matrix(details["refined_transform"]),
        ]
    elif task == "fit-primitive":
        lines = [
            "Primitive Fitting Result",
            "------------------------",
            f"Model: {metrics['model']}",
            f"Inliers: {metrics['inliers']}",
            f"Outliers: {metrics['outliers']}",
            f"Inlier Ratio: {metrics['inlier_ratio']:.2%}",
            f"Mean Residual: {metrics['residual_mean']:.6f}",
            "",
            "Parameters:",
            json.dumps(details["model_params"], indent=2, ensure_ascii=False),
        ]
    elif task == "extract-primitives":
        lines = [
            "Sequential Primitive Extraction Result",
            "--------------------------------------",
            f"Models: {metrics['model_count']}",
            f"Remaining Points: {metrics['remaining_points']}",
            "",
            "Extracted:",
        ]
        for primitive in details.get("primitives", []):
            lines.append(
                "- {model}: inlier ratio {ratio:.2%}, residual {residual:.6f}".format(
                    model=primitive["model_type"],
                    ratio=primitive["inlier_ratio"],
                    residual=primitive["residual_mean"],
                )
            )
    elif task == "segment":
        lines = [
            "Segmentation Result",
            "-------------------",
            f"Clusters: {metrics['cluster_count']}",
            f"Noise Points: {metrics['noise_points']}",
            "",
            "Cluster Stats:",
            json.dumps(details["clusters"], indent=2, ensure_ascii=False),
        ]
    elif task == "visualize":
        lines = [
            "Visualization Result",
            "--------------------",
            f"Mode: {metrics['mode']}",
        ]
    elif task == "train-pointnet":
        lines = [
            "PointNet Training Result",
            "------------------------",
            f"Loss: {metrics.get('loss')}",
            f"Accuracy: {metrics.get('accuracy')}",
        ]
    elif task == "reconstruct":
        lines = [
            "Surface Reconstruction Result",
            "-----------------------------",
            f"Method: {metrics['method']}",
            f"Vertices: {metrics['vertices']}",
            f"Triangles: {metrics['triangles']}",
        ]
    elif task == "verify-portfolio":
        lines = [
            "Portfolio Verification Result",
            "-----------------------------",
            f"Passed Commands: {metrics['passed_commands']}",
            f"Failed Commands: {metrics['failed_commands']}",
            f"Generated Artifacts: {metrics['generated_artifacts']}",
            f"Missing README Artifacts: {metrics['missing_readme_artifacts']}",
        ]
    elif task == "infer-pointnet":
        lines = [
            "PointNet Inference Result",
            "-------------------------",
            f"Class: {metrics.get('class')}",
            f"Confidence: {metrics.get('confidence')}",
        ]
    elif task == "pipeline":
        lines = [
            "Portfolio Pipeline Result",
            "-------------------------",
            f"Input Points: {metrics['input']['num_points']}",
            f"Processed Points: {metrics['preprocessing']['num_points_after']}",
            f"ICP RMSE: {metrics['registration']['rmse_before']:.6f} -> "
            f"{metrics['registration']['rmse_after']:.6f}",
            f"Clusters: {metrics['segmentation']['num_clusters']}",
            f"Noise Ratio: {metrics['segmentation']['noise_ratio']:.2%}",
        ]
    else:
        lines = [json.dumps(data, indent=2, ensure_ascii=False)]

    if artifacts:
        lines.extend(["", "Artifacts:"])
        for name, path in sorted(artifacts.items()):
            lines.append(f"- {name}: {path}")
    return "\n".join(lines)


def _json_vector(values: list[Any]) -> str:
    return json.dumps(values, ensure_ascii=False)


def _json_matrix(values: list[list[Any]]) -> str:
    return "\n".join(_json_vector(row) for row in values)


def _configure_logging(level: str | None) -> None:
    logging.basicConfig(level=getattr(logging, (level or "WARNING").upper(), logging.WARNING))


def _contains_legacy_mode(argv: list[str]) -> bool:
    return any(arg == "--mode" or arg.startswith("--mode=") for arg in argv)


if __name__ == "__main__":
    raise SystemExit(main())
