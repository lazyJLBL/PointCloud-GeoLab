"""Demonstrate VoxelHashGrid radius, kNN, box, and downsampling queries."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pointcloud_geolab.spatial import VoxelHashGrid


def main() -> int:
    rng = np.random.default_rng(43)
    points = rng.normal(size=(500, 3))
    query = np.asarray([0.1, -0.2, 0.05])
    grid = VoxelHashGrid.build(points, voxel_size=0.25)

    nearest = grid.nearest_neighbor(query)
    knn = grid.knn_search(query, k=5)
    radius_neighbors = grid.radius_search(query, radius=0.35)
    box_indices = grid.box_query([-0.3, -0.3, -0.3], [0.3, 0.3, 0.3])
    centroids, _ = grid.voxel_downsample()

    print("Voxel Hash Grid")
    print(f"Input points: {len(points)}")
    print(f"Voxel buckets: {len(grid.buckets)}")
    print(f"Nearest: index={nearest[0]} distance={nearest[1]:.6f}")
    print(f"kNN indices: {[index for index, _ in knn]}")
    print(f"Radius neighbors: {len(radius_neighbors)}")
    print(f"Box query points: {len(box_indices)}")
    print(f"Downsampled centroids: {len(centroids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
