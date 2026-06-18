"""Run a lightweight point-cloud scale benchmark with repeat and memory metadata."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pointcloud_geolab.kdtree import KDTree
from pointcloud_geolab.spatial import VoxelHashGrid

QUICK_SIZES = (1_000, 10_000)
FULL_SIZES = (1_000, 10_000, 100_000, 1_000_000)


@dataclass(frozen=True, slots=True)
class ScaleRun:
    """One repeated scale-benchmark measurement."""

    points: int
    generate_time: float
    bounds_time: float
    kdtree_build_time: float
    kdtree_query_time: float
    voxel_build_time: float
    voxel_radius_time: float
    peak_bytes: int


def run_scale_benchmark(
    output_dir: str | Path = "outputs/scale_benchmark",
    quick: bool = True,
    repeat: int = 2,
    seed: int = 42,
    sizes: list[int] | None = None,
    queries: int = 64,
) -> dict[str, Any]:
    """Run deterministic scale measurements and write CSV/JSON/Markdown/PNG artifacts."""

    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    point_sizes = tuple(sizes or (QUICK_SIZES if quick else FULL_SIZES))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_runs: list[list[ScaleRun]] = []
    started_tracing = not tracemalloc.is_tracing()
    if started_tracing:
        tracemalloc.start()
    try:
        for repeat_index in range(repeat):
            all_runs.append(
                [
                    _run_one_size(size, seed=seed + repeat_index, queries=queries)
                    for size in point_sizes
                ]
            )
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        if started_tracing and tracemalloc.is_tracing():
            tracemalloc.stop()

    rows = _aggregate_runs(all_runs)
    metadata = {
        "benchmark": "scale",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "parameters": {
            "quick": quick,
            "seed": seed,
            "sizes": list(point_sizes),
            "queries": queries,
        },
        "data_scale": {"points": list(point_sizes), "queries": [queries]},
        "repeat": {
            "count": repeat,
            "base_seed": seed,
            "seed_strategy": "base seed + zero-based repeat index",
            "deterministic_parameters_recorded": True,
            "timing_fields": _timing_fields(),
            "statistics": {
                "enabled": repeat > 1,
                "aggregates": ["mean", "std", "min", "max"] if repeat > 1 else [],
            },
        },
        "memory": {
            "available": True,
            "method": "tracemalloc",
            "current_bytes": int(current_bytes),
            "peak_bytes": int(peak_bytes),
            "note": "Local Python allocation peak; not a portable performance claim.",
        },
    }
    payload = {
        "benchmark": "scale",
        "metadata": metadata,
        "conclusion": (
            "Scale benchmark records local runtime and memory reference points for "
            "synthetic point counts; full 1M runs belong in manual gates."
        ),
        "rows": rows,
    }
    _write_csv(out_dir / "scale_benchmark.csv", rows)
    (out_dir / "scale_benchmark.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out_dir / "scale_benchmark.md").write_text(_format_markdown(rows, metadata), encoding="utf-8")
    _write_plot(out_dir / "scale_benchmark.png", rows)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", "--output", type=Path, default=Path("outputs/scale_benchmark")
    )
    parser.add_argument("--quick", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--points", nargs="*", type=int, help="Override benchmark point counts.")
    parser.add_argument("--queries", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = run_scale_benchmark(
            args.output_dir,
            quick=args.quick,
            repeat=args.repeat,
            seed=args.seed,
            sizes=args.points,
            queries=args.queries,
        )
    except Exception as exc:
        print(f"Scale benchmark failed: {exc}", file=sys.stderr)
        return 1
    print(
        "Scale benchmark wrote "
        f"{len(payload['rows'])} rows to {Path(args.output_dir).resolve()}",
    )
    return 0


def _run_one_size(points_count: int, seed: int, queries: int) -> ScaleRun:
    local_trace = not tracemalloc.is_tracing()
    if local_trace:
        tracemalloc.start()
    try:
        rng = np.random.default_rng(seed + points_count)
        start = time.perf_counter()
        points = rng.random((points_count, 3), dtype=np.float64)
        query_points = rng.random((min(queries, max(1, points_count)), 3), dtype=np.float64)
        generate_time = time.perf_counter() - start

        start = time.perf_counter()
        _ = points.min(axis=0), points.max(axis=0), points.mean(axis=0)
        bounds_time = time.perf_counter() - start

        start = time.perf_counter()
        tree = KDTree(points)
        kdtree_build_time = time.perf_counter() - start
        start = time.perf_counter()
        for query in query_points:
            tree.nearest_neighbor(query)
        kdtree_query_time = time.perf_counter() - start

        radius = 0.06
        start = time.perf_counter()
        voxel = VoxelHashGrid.build(points, voxel_size=radius)
        voxel_build_time = time.perf_counter() - start
        start = time.perf_counter()
        for query in query_points[: min(len(query_points), 32)]:
            voxel.radius_search(query, radius)
        voxel_radius_time = time.perf_counter() - start
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        if local_trace and tracemalloc.is_tracing():
            tracemalloc.stop()
    return ScaleRun(
        points=points_count,
        generate_time=generate_time,
        bounds_time=bounds_time,
        kdtree_build_time=kdtree_build_time,
        kdtree_query_time=kdtree_query_time,
        voxel_build_time=voxel_build_time,
        voxel_radius_time=voxel_radius_time,
        peak_bytes=int(peak_bytes),
    )


def _aggregate_runs(all_runs: list[list[ScaleRun]]) -> list[dict[str, Any]]:
    if not all_runs:
        return []
    rows: list[dict[str, Any]] = []
    for row_index, first in enumerate(all_runs[0]):
        row: dict[str, Any] = {"points": first.points, "repeat_count": len(all_runs)}
        for field in _timing_fields() + ["peak_bytes"]:
            values = [float(getattr(run_rows[row_index], field)) for run_rows in all_runs]
            row[field] = values[0]
            if len(values) > 1:
                row[f"{field}_mean"] = statistics.fmean(values)
                row[f"{field}_std"] = statistics.pstdev(values)
                row[f"{field}_min"] = min(values)
                row[f"{field}_max"] = max(values)
        rows.append(row)
    return rows


def _timing_fields() -> list[str]:
    return [
        "generate_time",
        "bounds_time",
        "kdtree_build_time",
        "kdtree_query_time",
        "voxel_build_time",
        "voxel_radius_time",
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _format_markdown(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
    lines = [
        "# Scale Benchmark",
        "",
        "| Points | KDTree build mean (s) | KDTree query mean (s) | Peak memory mean (bytes) |",
        "|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {points:,} | {build:.6f} | {query:.6f} | {memory:.0f} |".format(
                points=row["points"],
                build=row.get("kdtree_build_time_mean", row["kdtree_build_time"]),
                query=row.get("kdtree_query_time_mean", row["kdtree_query_time"]),
                memory=row.get("peak_bytes_mean", row["peak_bytes"]),
            )
        )
    lines.extend(
        [
            "",
            "## Run Metadata",
            "",
            f"- Repeat count: `{metadata['repeat']['count']}`",
            f"- Parameters: `{json.dumps(metadata['parameters'], sort_keys=True)}`",
            f"- Memory: `{metadata['memory']['peak_bytes']} peak bytes via tracemalloc`",
            "- Timing and memory numbers are local references, not portable claims.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    xs = [row["points"] for row in rows]
    ys = [row.get("kdtree_query_time_mean", row["kdtree_query_time"]) for row in rows]
    ax.plot(xs, ys, marker="o", label="KDTree nearest queries")
    ax.set_xscale("log")
    ax.set_xlabel("points")
    ax.set_ylabel("seconds")
    ax.set_title("Scale benchmark quick run")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
