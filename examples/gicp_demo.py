"""Run the custom Generalized ICP implementation on a deterministic surface."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pointcloud_geolab.io.visualization import save_point_cloud_projection
from pointcloud_geolab.registration import generalized_icp
from pointcloud_geolab.utils.transform import apply_transform, rotation_matrix_from_euler


def make_surface(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-1.0, 1.0, size=(280, 2))
    z = 0.14 * np.sin(2.5 * xy[:, 0]) + 0.06 * np.cos(3.0 * xy[:, 1])
    return np.column_stack([xy, z])


def main() -> int:
    target = make_surface()
    rotation = rotation_matrix_from_euler(0.04, -0.025, 0.035)
    translation = np.asarray([0.05, -0.035, 0.025])
    source = apply_transform(target, rotation, translation)
    result = generalized_icp(
        source,
        target,
        max_iterations=45,
        max_correspondence_distance=0.3,
        k_neighbors=16,
    )

    output_dir = ROOT / "outputs" / "gicp_demo"
    save_point_cloud_projection(
        output_dir / "gicp_before_after.png",
        [source, target, result.aligned_points],
        labels=["source before", "target", "source after"],
        title="Generalized ICP demo",
    )
    print("Generalized ICP")
    print(f"Initial RMSE: {result.initial_rmse:.6f}")
    print(f"Final RMSE: {result.final_rmse:.6f}")
    print(f"Iterations: {result.iterations}")
    print(f"Figure: {output_dir / 'gicp_before_after.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
