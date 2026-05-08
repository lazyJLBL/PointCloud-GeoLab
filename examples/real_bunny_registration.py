"""Register external Stanford Bunny scans with custom ICP.

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

from pointcloud_geolab.io.pointcloud_io import load_point_cloud, save_point_cloud
from pointcloud_geolab.io.visualization import save_point_cloud_projection
from pointcloud_geolab.registration import point_to_point_icp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "external" / "stanford" / "bunny_pair",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "real_bunny")
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
    result = point_to_point_icp(source, target, max_iterations=80, tolerance=1e-7)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_point_cloud(output_dir / "aligned_bunny.ply", result.aligned_points)
    save_point_cloud_projection(
        output_dir / "bunny_registration.png",
        [source, target, result.aligned_points],
        labels=["source before", "target", "source after"],
        title="Stanford Bunny registration",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "source": str(source_path),
                "target": str(target_path),
                "initial_rmse": result.initial_rmse,
                "final_rmse": result.final_rmse,
                "iterations": result.iterations,
                "transformation": result.transformation.tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Initial RMSE: {result.initial_rmse:.6f}")
    print(f"Final RMSE: {result.final_rmse:.6f}")
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
