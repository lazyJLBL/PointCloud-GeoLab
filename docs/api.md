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
- `path`

Use `result.to_dict()` for JSON-friendly output or `result.to_json()` when a
serialized envelope is needed.

Error results keep the same outer contract:

- `task`: task or command that failed.
- `success`: `False`.
- `error`: human-readable reason.
- `parameters`: the input parameters that reached the task, when available.
- `path`: the explicit failed path, or a best-effort path inferred from
  parameters such as `input_path`, `input`, `source`, `target`, `output`, or
  `output_dir`.

`Path` values and NumPy scalars/arrays are converted to JSON-friendly values.
Non-finite numeric values such as `NaN` and `Inf` are emitted as JSON `null`
so strict JSON parsers can consume the envelope.

## Stable Task Functions

Core-tested:

- `run_icp`: point-to-point ICP from two point-cloud files.
- `run_robust_icp`: Huber, Tukey, or trimmed ICP wrapper.
- `run_multiscale_icp`: coarse-to-fine ICP over voxel scales.
- `run_plane_segmentation`: dominant-plane RANSAC segmentation.
- `run_geometry_analysis`: AABB, OBB, and PCA metrics.
- `run_preprocessing`: crop, downsample, sampling, outlier removal,
  normalization, and normals.
- `run_primitive_fitting`: RANSAC plane, sphere, or cylinder fitting.
- `run_extract_primitives`: sequential primitive extraction.
- `run_segmentation`: DBSCAN, Euclidean clustering, region growing, and
  optional ground removal.
- `run_ground_object_segmentation`: ground removal plus object cluster
  reporting.

Demo-ready:

- `run_benchmark`: built-in quick/full benchmark suites with CSV, JSON,
  Markdown, PNG, repeat statistics, and memory metadata outputs.

Documented workflow:

- `run_portfolio_verification`: portfolio smoke-check report used by
  `scripts/verify_portfolio.py`.

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
- GICP-style covariance-weighted ICP internals: experimental, not a full
  nonlinear GICP optimizer, and not part of the v1.0.0 stable API promise.
- User-provided real-data workflow helpers: documented as examples plus
  verifier scripts, not stable public API. Use
  `examples/kitti_lidar_segmentation.py` and
  `scripts/verify_realdata_workflow.py` for that workflow.

The CLI remains the preferred reviewer interface:

```bash
python -m pointcloud_geolab pipeline \
  --input examples/demo_data \
  --output outputs/portfolio_demo
```
