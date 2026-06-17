# Architecture

PointCloud-GeoLab is organized as small layers rather than one monolithic
pipeline. Most modules accept and return NumPy arrays so the core algorithms
remain easy to test without optional visualization or ML dependencies.

## Spatial Index Layer

The spatial layer provides reusable neighborhood search primitives:

- `pointcloud_geolab.kdtree`: custom KDTree nearest, kNN, and radius search.
- `pointcloud_geolab.spatial`: VoxelHashGrid fixed-radius, nearest, kNN, box
  query, and voxel downsampling helpers.

Registration, normals, clustering, feature extraction, and benchmarks reuse
these primitives.

## Geometry Layer

The geometry layer contains shape and summary computations:

- PCA and principal axes.
- AABB and OBB summaries.
- Plane, sphere, and cylinder primitive fitting.
- Sequential primitive extraction under outliers.

This layer is intentionally independent from the CLI so it can be tested with
small fixed arrays.

## Registration Layer

The registration layer handles local and coarse alignment:

- SVD rigid alignment.
- Point-to-point ICP and point-to-plane ICP.
- Robust and multi-scale ICP variants.
- GICP-style covariance-weighted ICP.
- Feature registration through ISS, local descriptors, RANSAC, and ICP
  refinement.

The GICP-style path is not a full nonlinear GICP optimizer. Feature fallback is
diagnostic and must not be described as descriptor registration success.

## Segmentation Layer

The segmentation layer groups and labels point clouds:

- DBSCAN clustering.
- Euclidean clustering.
- Region growing with normal-angle checks.
- RANSAC ground removal followed by object clustering.
- Cluster statistics and Markdown report writing.

The current segmentation methods use global thresholds and are most appropriate
for small scenes and reviewer demos.

## Benchmark And Report Layer

Benchmark and report code creates evidence artifacts:

- CSV, JSON, Markdown, and PNG benchmark outputs.
- Metadata for parameters, random seed, data scale, platform, Python, and
  optional dependencies.
- Portfolio metrics, report, figures, processed point clouds, and transform
  JSON.
- Verifier scripts that reject missing or malformed artifacts.

Generated outputs live under ignored directories such as `outputs/` and
`benchmark_results/`.

## CLI And Pipeline Layer

`pointcloud_geolab.cli` maps commands to structured API task functions. The
pipeline command runs the deterministic portfolio demo and writes a reviewable
bundle under `outputs/portfolio_demo`.

`pointcloud_geolab.api` is the stable task surface used by the CLI, examples,
and tests.

## Optional Baseline Layer

Optional integrations are isolated from core correctness:

- Open3D baselines and visualization.
- Plotly HTML export.
- PyTorch/PointNet demos.
- LAS/LAZ I/O through `laspy`.
- SciPy, scikit-learn, and pandas benchmark comparisons.

When these dependencies are absent, tests should skip or the command should
return a clear dependency message.
