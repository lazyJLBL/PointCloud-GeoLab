"""Benchmark custom Generalized ICP against point-to-point ICP."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pointcloud_geolab.api import run_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmarks/gicp"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    result = run_benchmark("gicp", output_dir=args.output, quick=args.quick, seed=args.seed)
    print(result.to_dict()["data"]["markdown"])
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
