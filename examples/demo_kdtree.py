"""Demo: custom KD-Tree nearest-neighbor queries."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pointcloud_geolab.kdtree.kdtree import KDTree


def main() -> int:
    rng = np.random.default_rng(7)
    points = rng.random((500, 3))
    query = np.asarray([0.52, 0.13, 0.88])
    tree = KDTree(points)

    idx, distance = tree.nearest_neighbor(query)
    knn = tree.knn_search(query, k=5)
    radius = tree.radius_search(query, radius=0.10)

    brute_distances = np.linalg.norm(points - query, axis=1)
    brute_idx = int(np.argmin(brute_distances))
    brute_knn = np.argsort(brute_distances)[:5].tolist()

    print(f"Query point: {query.tolist()}")
    print("\nNearest neighbor:")
    print(f"Index: {idx}")
    print(f"Distance: {distance:.6f}")
    print("\nKNN result:")
    print([item[0] for item in knn])
    print("\nRadius search result:")
    print(f"Found {len(radius)} points within radius 0.10")
    passed = idx == brute_idx and [item[0] for item in knn] == brute_knn
    print(f"\nValidation with brute force: {'Passed' if passed else 'Failed'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

