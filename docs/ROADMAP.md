# PointCloud-GeoLab Roadmap

This roadmap keeps the project focused on a practical progression: first make
the existing algorithms easy to run and reproduce, then expand algorithm depth,
real-data validation, advanced processing, visualization, and machine learning
demos.

## v0.2 Usability and Engineering

Status: in progress.

- Provide a stable Python API in `pointcloud_geolab.api`.
- Replace the demo-style `--mode` entrypoint with subcommands:
  `icp`, `plane`, `geometry`, `preprocess`, and `benchmark`.
- Keep `python main.py --mode ...` working as a compatibility path.
- Support YAML configuration, batch manifests, JSON output, metrics files, and
  reproducible artifact directories.
- Integrate KDTree and ICP benchmarks into the CLI.
- Expand tests for CLI smoke paths, config overrides, batch execution, API
  calls, and failure reporting.

## v0.3 Core Algorithm Improvements

- KDTree: support higher-dimensional input, batch queries, and parallel query
  execution for large point sets.
- ICP: add point-to-plane ICP, convergence diagnostics, and robustness tests
  under noise and outliers.
- RANSAC: generalize model fitting beyond planes to spheres and cylinders.
- PCA: add covariance and principal-axis visualization, plus chunked analysis
  for large point clouds.
- Bounding volumes: add dynamic update helpers for AABB, OBB, and bounding
  spheres, then use them in spatial query acceleration examples.

## v0.4 Real Data and Benchmarking

- Add documented dataset import workflows for Bunny, Armadillo, KITTI,
  Semantic3D, and ModelNet without storing large datasets in the repository.
- Standardize preprocessing pipelines for denoising, downsampling, and normal
  estimation.
- Track RMSE, accuracy, runtime, and memory usage for reproducible comparisons.
- Compare selected workflows against Open3D and PCL where those libraries are
  available in the local environment.

## v0.5 Advanced Point Cloud Processing

- Add feature descriptors and matching demos for FPFH, ISS, and SHOT.
- Implement global registration with feature matching, RANSAC, and ICP
  refinement.
- Add surface reconstruction examples for Poisson reconstruction and ball
  pivoting.
- Add segmentation and clustering demos based on voxel grids, Euclidean
  distance, and region growing.

## v0.6 ML and Presentation

- Add PointNet and PointNet++ examples as optional machine learning demos.
- Provide semantic segmentation or detection notebooks using small reproducible
  samples.
- Demonstrate hybrid pipelines that combine traditional geometry algorithms
  with learned features.
- Add Open3D, PyVista, or Plotly viewers for interactive inspection.
- Publish notebooks, demo videos, and technical writeups that explain algorithm
  ideas, implementation details, and performance tradeoffs.
