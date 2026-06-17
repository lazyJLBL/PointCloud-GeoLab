# Project Boundary

PointCloud-GeoLab is a portfolio and learning project. It is meant to make
point-cloud geometry algorithms readable, runnable, and reviewable in a compact
Python codebase.

It is not intended to replace mature libraries such as PCL or Open3D. Those
projects remain the right baseline for production systems, broad hardware
coverage, and optimized large-scene processing.

## Core-tested

Core-tested parts are covered by unit tests, deterministic fixtures, and CI:

- KDTree nearest, kNN, radius, batch, duplicate, empty, and boundary queries.
- VoxelHashGrid nearest, kNN, radius, box query, and voxel downsampling.
- Preprocessing utilities such as crop, normalization, sampling, downsampling,
  outlier filtering, and local PCA normal estimation.
- SVD rigid alignment, point-to-point ICP, point-to-plane ICP, robust ICP, and
  multi-scale ICP.
- RANSAC plane, sphere, cylinder, and sequential primitive extraction.
- PCA, AABB, OBB, DBSCAN, Euclidean clustering, region growing, and simple
  ground/object segmentation.

## Demo-ready

Demo-ready workflows are designed for reviewer smoke checks:

- The portfolio pipeline generates a Markdown report, metrics JSON, key figures,
  processed cloud artifacts, and transform JSON from deterministic demo data.
- The benchmark CLI emits CSV, JSON, Markdown, PNG, parameters, seed, platform,
  and dependency metadata.
- Verifier scripts check generated artifact content, not only file existence.

Synthetic demo outputs are smoke evidence. They should not be presented as
real-data validation.

## Optional baseline

Optional paths are useful comparisons or demonstrations, but they are not the
core correctness claim:

- Open3D registration, segmentation, visualization, and reconstruction helpers.
- Plotly HTML visualization.
- PyTorch/PointNet demo paths.
- LAS/LAZ I/O through optional `laspy`.
- SciPy, scikit-learn, pandas, and Open3D benchmark baselines.

These paths should skip cleanly or report missing dependencies when the optional
packages are unavailable.

## Documented workflow

Real-data workflows for Stanford Bunny, KITTI, and ModelNet are documented, but
the datasets are not committed. Reviewers should prepare local files under
`data/external/` and regenerate outputs locally.

## Not This Project

PointCloud-GeoLab is:

- not a PCL or Open3D replacement;
- not a SLAM backend;
- not a full nonlinear GICP optimizer;
- not a production LiDAR stack;
- not a claim that synthetic demo success proves real-world performance.

The current GICP path is GICP-style covariance-weighted ICP. It uses
covariance-derived scalar weights with weighted SVD updates, and it records
`full_nonlinear_gicp: false` in diagnostics.
