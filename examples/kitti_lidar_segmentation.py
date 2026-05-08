"""Segment a KITTI Velodyne frame after ground removal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pointcloud_geolab.datasets import load_velodyne_frame
from pointcloud_geolab.segmentation import ground_object_segmentation, write_cluster_report
from pointcloud_geolab.visualization import save_colored_point_cloud


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frame",
        type=Path,
        default=ROOT / "data" / "external" / "kitti" / "velodyne" / "000000.bin",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "kitti_segmentation")
    parser.add_argument("--eps", type=float, default=0.65)
    parser.add_argument("--min-points", type=int, default=12)
    parser.add_argument("--max-points", type=int, default=12000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.frame.exists():
        print(
            "KITTI frame not found. Expected a Velodyne `.bin` file at "
            f"{args.frame}. See docs/datasets.md for the download layout.",
            file=sys.stderr,
        )
        return 2

    points = load_velodyne_frame(args.frame)
    if len(points) > args.max_points:
        points = points[: args.max_points]
    result = ground_object_segmentation(
        points,
        ground_threshold=0.18,
        ground_axis="z",
        ground_angle_threshold=35.0,
        cluster_method="euclidean",
        eps=args.eps,
        min_points=args.min_points,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_colored_point_cloud(output_dir / "kitti_clusters.ply", points, result.labels)
    write_cluster_report(result, output_dir / "cluster_report.md")
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "frame": str(args.frame),
                "points": len(points),
                "ground_points": len(result.ground.ground_indices),
                "clusters": len(result.clusters),
                "noise_points": len(result.noise_indices),
                "cluster_sizes": [cluster.point_count for cluster in result.clusters],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Points: {len(points)}")
    print(f"Ground points: {len(result.ground.ground_indices)}")
    print(f"Object clusters: {len(result.clusters)}")
    print(f"Artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
