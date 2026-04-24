"""Demo: AABB, PCA-based OBB, and principal directions."""

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
    parser.add_argument("--input", default=str(ROOT / "data" / "object.ply"))
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    return cli_main(
        ["--mode", "geometry", "--input", args.input, "--save-results"]
        + (["--visualize"] if args.visualize else [])
    )


if __name__ == "__main__":
    raise SystemExit(main())

