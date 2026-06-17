# API Overview

This page lists the Python APIs that are stable enough to use in examples,
tests, and interviews. They are intentionally small wrappers around the core
algorithms so the implementation remains easy to inspect.

## Task API

Primary task APIs live in `pointcloud_geolab.api` and return `TaskResult`, a
JSON-friendly envelope with:

- `success`
- `metrics`
- `artifacts`
- `parameters`
- `data`
- `error`

Stable task functions:

| Function | Purpose |
|---|---|
| `run_icp` | Point-to-point ICP from two point-cloud files. |
| `run_global_registration` | FPFH/Open3D or ISS/custom-descriptor coarse registration plus ICP. |
| `run_feature_registration` | Self-implemented ISS descriptor RANSAC plus ICP convenience wrapper. |
| `run_primitive_fitting` | RANSAC plane/sphere/cylinder fitting. |
| `run_extract_primitives` | Sequential primitive extraction. |
| `run_segmentation` | DBSCAN, Euclidean clustering, region growing, optional ground removal. |
| `run_ground_object_segmentation` | Ground plane removal plus object cluster report. |
| `run_geometry_analysis` | AABB, OBB, PCA metrics. |
| `run_preprocessing` | Crop, downsample, outlier removal, normalization, normals. |
| `run_benchmark` | Built-in benchmark suites with CSV/JSON/Markdown/PNG outputs. |
| `run_portfolio_verification` | Portfolio smoke-check report. |

Example:

```python
from pointcloud_geolab.api import run_global_registration, run_primitive_fitting, run_segmentation

registration = run_global_registration("source.ply", "target.ply", voxel_size=0.05)
primitive = run_primitive_fitting("scene.ply", model="sphere", threshold=0.02)
segmentation = run_segmentation("scene.ply", method="dbscan", eps=0.05, min_points=20)
```

## Core Algorithm API

Spatial indexes:

- `pointcloud_geolab.kdtree.KDTree`
- `pointcloud_geolab.spatial.VoxelHashGrid`

Registration:

- `pointcloud_geolab.registration.estimate_rigid_transform`
- `pointcloud_geolab.registration.point_to_point_icp`
- `pointcloud_geolab.registration.point_to_plane_icp`
- `pointcloud_geolab.registration.robust_icp`
- `pointcloud_geolab.registration.multiscale_icp`
- `pointcloud_geolab.registration.generalized_icp`
- `pointcloud_geolab.registration.evaluate_registration`

Geometry:

- `pointcloud_geolab.geometry.pca_analysis`
- `pointcloud_geolab.geometry.compute_aabb`
- `pointcloud_geolab.geometry.compute_obb`
- `pointcloud_geolab.geometry.ransac_fit_primitive`
- `pointcloud_geolab.geometry.extract_primitives`

Segmentation:

- `pointcloud_geolab.segmentation.dbscan_clustering`
- `pointcloud_geolab.segmentation.euclidean_clustering`
- `pointcloud_geolab.segmentation.region_growing_segmentation`
- `pointcloud_geolab.segmentation.remove_ground_plane`
- `pointcloud_geolab.segmentation.ground_object_segmentation`
- `pointcloud_geolab.segmentation.write_cluster_report`

Preprocessing and IO:

- `pointcloud_geolab.preprocessing.voxel_downsample`
- `pointcloud_geolab.preprocessing.estimate_normals`
- `pointcloud_geolab.preprocessing.remove_statistical_outliers`
- `pointcloud_geolab.preprocessing.remove_radius_outliers`
- `pointcloud_geolab.io.load_point_cloud`
- `pointcloud_geolab.io.save_point_cloud`

## Optional APIs

These paths are importable but depend on optional extras or are intended as
demonstrations:

- `pointcloud_geolab.reconstruction.reconstruct_surface` requires Open3D.
- `pointcloud_geolab.ml.*` requires PyTorch and is an optional PointNet demo.
- `pointcloud_geolab.visualization.export_point_cloud_html` benefits from Plotly.
- LAS/LAZ IO requires `laspy`.

## Compatibility Notes

The CLI is the preferred reviewer interface:

```bash
python -m pointcloud_geolab pipeline --input examples/demo_data --output outputs/portfolio_demo
```

The legacy `main.py --mode ...` interface remains for older examples but should
not be used for new documentation.
