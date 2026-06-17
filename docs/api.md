# Stable API

This page lists the APIs intentionally exported from `pointcloud_geolab` and
`pointcloud_geolab.api`. Experimental feature registration, optional Open3D
reconstruction, optional ML, and visualization helpers remain importable from
their modules, but they are not part of the stable public surface.

## Result Envelope

All stable task functions return `TaskResult`:

```python
from pointcloud_geolab import TaskResult
```

Fields:

- `success`
- `metrics`
- `artifacts`
- `parameters`
- `data`
- `error`

Use `result.to_dict()` for JSON-friendly output.

## Stable Task Functions

| Function | Status | Purpose |
|---|---|---|
| `run_icp` | Core-tested | Point-to-point ICP from two point-cloud files. |
| `run_robust_icp` | Core-tested | Huber, Tukey, or trimmed ICP wrapper. |
| `run_multiscale_icp` | Core-tested | Coarse-to-fine ICP over voxel scales. |
| `run_plane_segmentation` | Core-tested | Dominant-plane RANSAC segmentation. |
| `run_geometry_analysis` | Core-tested | AABB, OBB, and PCA metrics. |
| `run_preprocessing` | Core-tested | Crop, downsample, sampling, outlier removal, normalization, normals. |
| `run_primitive_fitting` | Core-tested | RANSAC plane, sphere, or cylinder fitting. |
| `run_extract_primitives` | Core-tested | Sequential primitive extraction. |
| `run_segmentation` | Core-tested | DBSCAN, Euclidean clustering, region growing, optional ground removal. |
| `run_ground_object_segmentation` | Core-tested | Ground removal plus object cluster reporting. |
| `run_benchmark` | Demo-ready | Built-in quick/full benchmark suites with CSV/JSON/Markdown/PNG outputs. |
| `run_portfolio_verification` | Documented workflow | Portfolio smoke-check report used by `scripts/verify_portfolio.py`. |

Example:

```python
from pointcloud_geolab import run_icp, run_primitive_fitting, run_segmentation

icp = run_icp("source.ply", "target.ply")
primitive = run_primitive_fitting("scene.ply", model="sphere", threshold=0.02)
clusters = run_segmentation("scene.ply", method="dbscan", eps=0.05, min_points=20)
```

## Not Stable API

These are useful, but intentionally not exported from `pointcloud_geolab.__all__`:

- `run_global_registration` and `run_feature_registration`: optional or
  experimental coarse registration paths. Descriptor fallback is not equivalent
  to descriptor registration success.
- `run_iss_keypoints`: feature research helper.
- `run_reconstruction`: Open3D-backed optional workflow.
- `run_visualization`: optional HTML visualization workflow.
- `run_train_pointnet` and `run_infer_pointnet`: optional PyTorch demo.

The CLI remains the preferred reviewer interface:

```bash
python -m pointcloud_geolab pipeline --input examples/demo_data --output outputs/portfolio_demo
```
