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
    run_geometry_analysis,
    run_icp,
    run_plane_segmentation,
    run_preprocessing,
)

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
        "save_results": False,
        "visualize": False,
    },
    "benchmark": {
        "output_dir": "results",
        "benchmark_name": "kdtree",
        "quick": True,
        "full": False,
        "save_json": None,
        "save_md": None,
        "seed": 42,
        "queries": 100,
        "points": None,
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
    preprocess.add_argument(
        "--estimate-normals",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="estimate normals after filtering",
    )

    benchmark = subparsers.add_parser("benchmark", help="run built-in benchmarks")
    _add_common_options(benchmark, include_save_flags=False)
    benchmark.add_argument(
        "benchmark_name",
        nargs="?",
        choices=["kdtree", "icp"],
        help="benchmark suite",
    )
    benchmark.add_argument("--quick", action=argparse.BooleanOptionalAction, default=None)
    benchmark.add_argument("--full", action=argparse.BooleanOptionalAction, default=None)
    benchmark.add_argument("--save-json", type=Path)
    benchmark.add_argument("--save-md", type=Path)
    benchmark.add_argument("--queries", type=int)
    benchmark.add_argument("--points", nargs="*", type=int)

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
                        "batch job task must be one of: icp, plane, geometry, "
                        "preprocess, benchmark"
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
            save_results=params["save_results"],
            visualize=params["visualize"],
        )
    if task == "benchmark":
        full = bool(params["full"])
        quick = bool(params["quick"]) and not full
        return run_benchmark(
            benchmark=params["benchmark_name"],
            output_dir=params["output_dir"],
            quick=quick,
            full=full,
            save_json=params["save_json"],
            save_md=params["save_md"],
            seed=params["seed"],
            queries=params["queries"],
            points=params["points"],
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
