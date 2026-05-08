# Benchmarking

Benchmarks are designed to make tradeoffs visible rather than just report that
an algorithm exists.

## Suites

- `kdtree`: custom KDTree and VoxelHashGrid vs brute force, Open3D, SciPy
  cKDTree, and sklearn KDTree when available.
- `icp`: custom ICP, Huber ICP, trimmed ICP, and optional Open3D ICP under
  perturbations and source outliers.
- `ransac`: custom RANSAC primitive fitting vs NumPy PCA plane and optional
  Open3D plane segmentation under increasing outlier ratios.
- `registration`: ICP vs FPFH+RANSAC+ICP under large initial rotations.
- `gicp`: custom covariance-weighted GICP vs point-to-point ICP.
- `segmentation`: Euclidean clustering runtime over synthetic cluster sizes.

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
- Robust ICP improves correspondence weighting when source-only outliers are
  present, but it still needs enough overlap.
- GICP spends more time per iteration because it estimates and uses local
  covariance matrices.
