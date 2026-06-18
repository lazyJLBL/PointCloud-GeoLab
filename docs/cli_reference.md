# CLI Reference

The CLI entry point is available through:

```bash
python -m pointcloud_geolab --help
```

## Stable Commands

- `pipeline`: run the deterministic portfolio demo pipeline.
- `benchmark`: run built-in quick benchmarks and write CSV, JSON, Markdown, and
  PNG artifacts.
- `icp`: run point-to-point ICP.
- `plane`: fit a dominant plane with RANSAC.
- `geometry`: compute AABB, OBB, and PCA metrics.
- `preprocess`: run point-cloud preprocessing filters.
- `segment`: run DBSCAN, Euclidean, or region-growing segmentation.
- `fit-primitive`: fit plane, sphere, or cylinder primitives.
- `extract-primitives`: sequentially extract primitives.
- `verify-portfolio`: run portfolio smoke checks.

## Experimental Or Optional Commands

- `register`: feature-based global registration with diagnostics.
- `visualize`: optional HTML visualization output.
- `reconstruct`: optional Open3D-backed surface reconstruction.
- `train-pointnet` and `infer-pointnet`: optional synthetic PointNet demos.

PointNet paths are not a PointNet training release and do not install CUDA by
default.

## Error Behavior

Bad input should return a non-zero exit code. With `--format json`, failures
should remain machine-readable through the `TaskResult` envelope:

```json
{
  "task": "icp",
  "success": false,
  "metrics": {},
  "artifacts": {},
  "parameters": {},
  "data": {},
  "error": "path/to/file: missing point cloud file"
}
```

The deprecated `--mode` interface still exists for compatibility, but new
review commands should use subcommands.
