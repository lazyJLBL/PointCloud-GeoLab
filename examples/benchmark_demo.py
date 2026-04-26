"""Demo: run the quick benchmark suite."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pointcloud_geolab.api import run_benchmark


def main() -> int:
    out = ROOT / "outputs" / "benchmarks"
    result = run_benchmark("all", output_dir=out, quick=True, queries=20, points=[500, 1000])
    print(result.to_dict()["metrics"])
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
