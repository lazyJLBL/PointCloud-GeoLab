"""Register external Stanford Bunny scans with feature RANSAC plus ICP.

This example expects files prepared under ``data/external/stanford/bunny_pair``.
It exits with a clear message when the dataset is not present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from pointcloud_geolab.api import run_global_registration
from pointcloud_geolab.io.pointcloud_io import load_point_cloud, save_point_cloud
from pointcloud_geolab.io.visualization import save_point_cloud_projection
from pointcloud_geolab.registration import evaluate_registration, point_to_point_icp
from pointcloud_geolab.utils.transform import apply_homogeneous_transform


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "external" / "stanford" / "bunny_pair",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "real_bunny")
    parser.add_argument(
        "--method",
        choices=["iss_descriptor_ransac_icp", "fpfh_ransac_icp", "icp"],
        default="iss_descriptor_ransac_icp",
        help="registration method; fpfh_ransac_icp requires Open3D",
    )
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.15)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        source_path, target_path = resolve_bunny_pair(args.data_dir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    source = load_point_cloud(source_path)
    target = load_point_cloud(target_path)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "icp":
        result = point_to_point_icp(source, target, max_iterations=80, tolerance=1e-7)
        aligned = result.aligned_points
        transform = result.transformation
        metrics = {
            "method": args.method,
            "initial_rmse": result.initial_rmse,
            "final_rmse": result.final_rmse,
            "fitness": result.fitness,
            "iterations": result.iterations,
        }
    else:
        task = run_global_registration(
            source_path,
            target_path,
            output=output_dir / "aligned_bunny.ply",
            save_transform=output_dir / "transform.txt",
            output_dir=output_dir,
            voxel_size=args.voxel_size,
            method=args.method,
            threshold=args.threshold,
            save_results=False,
        )
        if not task.success:
            print(f"Registration failed: {task.error}", file=sys.stderr)
            return 1
        transform = np.asarray(task.data["refined_transform"], dtype=float)
        aligned = apply_homogeneous_transform(source, transform)
        initial = evaluate_registration(source, target, np.eye(4), threshold=args.threshold)
        metrics = {
            "method": args.method,
            "initial_rmse": initial["rmse"],
            "final_rmse": task.metrics["final_rmse"],
            "fitness": task.metrics["final_fitness"],
            "coarse_fitness": task.metrics["coarse_fitness"],
            "refined_fitness": task.metrics["refined_fitness"],
            "source_downsampled": task.metrics["source_downsampled"],
            "target_downsampled": task.metrics["target_downsampled"],
        }

    save_point_cloud(output_dir / "aligned_bunny.ply", aligned)
    save_point_cloud_projection(
        output_dir / "bunny_before.png",
        [source, target],
        labels=["source", "target"],
        title="Stanford Bunny before registration",
    )
    save_point_cloud_projection(
        output_dir / "bunny_after.png",
        [aligned, target],
        labels=["aligned source", "target"],
        title="Stanford Bunny after registration",
    )
    save_point_cloud_projection(
        output_dir / "bunny_registration.png",
        [source, target, aligned],
        labels=["source before", "target", "source after"],
        title="Stanford Bunny registration",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "source": str(source_path),
                "target": str(target_path),
                **metrics,
                "transformation": transform.tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Method: {args.method}")
    print(f"Initial RMSE: {metrics['initial_rmse']:.6f}")
    print(f"Final RMSE: {metrics['final_rmse']:.6f}")
    print(f"Fitness: {metrics['fitness']:.4f}")
    print(f"Artifacts: {output_dir}")
    return 0


def resolve_bunny_pair(data_dir: Path) -> tuple[Path, Path]:
    source = data_dir / "bunny_source.ply"
    target = data_dir / "bunny_target.ply"
    if source.exists() and target.exists():
        return source, target

    scan_source = data_dir / "bun000.ply"
    scan_target = data_dir / "bun045.ply"
    if scan_source.exists() and scan_target.exists():
        return scan_source, scan_target

    raise FileNotFoundError(
        "Stanford Bunny pair not found. Expected `bunny_source.ply` and "
        "`bunny_target.ply` or Stanford scans `bun000.ply` and `bun045.ply` under "
        f"{data_dir}. See docs/datasets.md and run "
        "`python scripts/prepare_datasets.py make-bunny-pair --input bunny.ply "
        "--output-dir data/external/stanford/bunny_pair`."
    )


if __name__ == "__main__":
    raise SystemExit(main())
