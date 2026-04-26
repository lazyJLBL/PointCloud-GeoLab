# Benchmarking

Benchmarks are designed to make tradeoffs visible rather than just report that
an algorithm exists.

## Suites

- `kdtree`: custom KDTree vs brute force, Open3D, SciPy cKDTree, and sklearn KDTree when available.
- `icp`: convergence, RMSE, rotation error, and translation error under perturbations.
- `ransac`: inlier precision/recall under increasing outlier ratios.
- `registration`: ICP vs FPFH+RANSAC+ICP under large initial rotations.

Run all quick suites:

```bash
python -m pointcloud_geolab benchmark --suite all --quick --output outputs/benchmarks
```

Each run writes CSV, Markdown, JSON, PNG, and `metrics.json`.

Interpretation examples:

- The custom KDTree is useful for explaining nearest-neighbor search and is fine
  at small scale; optimized SciPy/sklearn implementations are expected to win at
  large scale.
- Plain ICP is accurate near the correct pose but can fail under large initial
  rotations; FPFH+RANSAC+ICP is slower but more robust.
