"""Demo: point-to-point ICP registration."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import main as cli_main


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(ROOT / "data" / "bunny_source.ply"))
    parser.add_argument("--target", default=str(ROOT / "data" / "bunny_target.ply"))
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    return cli_main(
        [
            "--mode",
            "icp",
            "--source",
            args.source,
            "--target",
            args.target,
            "--max-iterations",
            "60",
            "--tolerance",
            "1e-7",
            "--save-results",
        ]
        + (["--visualize"] if args.visualize else [])
    )


if __name__ == "__main__":
    raise SystemExit(main())

