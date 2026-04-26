"""Demo: LiDAR-like preprocessing with downsampling, crop, sampling, and normals."""

from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pointcloud_geolab.api import run_preprocessing
from pointcloud_geolab.io import save_point_cloud


def main() -> int:
    rng = np.random.default_rng(130)
    ground_xy = rng.uniform(-3, 3, size=(1200, 2))
    ground = np.column_stack([ground_xy, rng.normal(0, 0.02, size=1200)])
    object_points = rng.normal([0.5, -0.4, 0.8], [0.25, 0.25, 0.2], size=(350, 3))
    scene = np.vstack([ground, object_points, rng.uniform(-4, 4, size=(80, 3))])
    out = ROOT / "outputs" / "preprocessing"
    out.mkdir(parents=True, exist_ok=True)
    input_path = out / "lidar_raw.xyz"
    save_point_cloud(input_path, scene)

    start = time.perf_counter()
    result = run_preprocessing(
        input_path,
        output=out / "lidar_clean.ply",
        output_dir=out,
        voxel_size=0.08,
        radius=0.18,
        min_neighbors=3,
        crop_min=[-3, -3, -0.5],
        crop_max=[3, 3, 2.0],
        estimate_normals_flag=True,
        save_results=True,
    )
    print({"elapsed_s": time.perf_counter() - start, **result.to_dict()["metrics"]})
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
