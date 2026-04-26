"""Demo: DBSCAN clustering on a synthetic multi-object scene."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pointcloud_geolab.api import run_segmentation
from pointcloud_geolab.io import save_point_cloud


def main() -> int:
    rng = np.random.default_rng(120)
    scene = np.vstack(
        [
            rng.normal([0, 0, 0], 0.04, size=(120, 3)),
            rng.normal([0.8, 0.1, 0.0], 0.04, size=(120, 3)),
            rng.normal([-0.5, 0.7, 0.2], 0.04, size=(120, 3)),
            rng.uniform(-1.5, 1.5, size=(25, 3)),
        ]
    )
    out = ROOT / "outputs" / "segmentation"
    out.mkdir(parents=True, exist_ok=True)
    input_path = out / "multi_object_scene.ply"
    save_point_cloud(input_path, scene)
    result = run_segmentation(
        input_path,
        output=out / "segmented_scene.ply",
        output_dir=out,
        method="dbscan",
        eps=0.12,
        min_points=8,
        export_html=out / "segmented_scene.html",
    )
    print(result.to_dict()["data"]["clusters"])
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
