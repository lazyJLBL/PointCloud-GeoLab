"""Benchmark custom KD-Tree nearest-neighbor search against brute force."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pointcloud_geolab.kdtree import KDTree


def brute_force_nearest(points: np.ndarray, queries: np.ndarray) -> list[int]:
    result = []
    for query in queries:
        distances = np.linalg.norm(points - query, axis=1)
        result.append(int(np.argmin(distances)))
    return result


def benchmark_case(points_count: int, queries_count: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    points = rng.random((points_count, 3))
    queries = rng.random((queries_count, 3))

    start = time.perf_counter()
    tree = KDTree(points)
    build_time = time.perf_counter() - start

    start = time.perf_counter()
    brute_indices = brute_force_nearest(points, queries)
    brute_time = time.perf_counter() - start

    start = time.perf_counter()
    kd_indices = [tree.nearest_neighbor(query)[0] for query in queries]
    kd_time = time.perf_counter() - start

    correct = brute_indices == kd_indices
    speedup = brute_time / kd_time if kd_time > 0 else float("inf")
    return {
        "points": points_count,
        "queries": queries_count,
        "build_time": build_time,
        "brute_time": brute_time,
        "kd_time": kd_time,
        "speedup": speedup,
        "correct": correct,
    }


def format_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Points | Queries | Build Time (s) | Brute Force (s) | KD-Tree (s) | Speedup | Correct |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            "| {points:,} | {queries:,} | {build_time:.4f} | {brute_time:.4f} | "
            "{kd_time:.4f} | {speedup:.2f}x | {correct} |".format(
                points=row["points"],
                queries=row["queries"],
                build_time=row["build_time"],
                brute_time=row["brute_time"],
                kd_time=row["kd_time"],
                speedup=row["speedup"],
                correct="yes" if row["correct"] else "no",
            )
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--points", type=int, nargs="*", default=[1000, 5000, 10000, 50000])
    parser.add_argument("--save", type=Path, help="optional markdown output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = [
        benchmark_case(points_count, args.queries, args.seed + i)
        for i, points_count in enumerate(args.points)
    ]
    table = format_table(rows)
    print(table)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(table + "\n", encoding="utf-8")
    return 0 if all(bool(row["correct"]) for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
