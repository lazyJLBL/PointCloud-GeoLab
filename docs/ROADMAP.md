# PointCloud-GeoLab Roadmap

This roadmap reflects the current repository state. Completed items are kept
visible because the project is used as a hiring portfolio and reviewers need to
separate implemented work from planned work.

## Completed

- Package metadata and optional extras in `pyproject.toml` for `dev`, `vis`,
  `bench`, `open3d`, `io`, `ml`, and `all`.
- Single GitHub Actions workflow, `tests.yml`, covering Python 3.10, 3.11, and
  3.12 with install, Ruff, Black check, demo-data generation, portfolio
  pipeline smoke test, and pytest coverage.
- Stable CLI through `python -m pointcloud_geolab ...` plus legacy
  `python main.py --mode ...` compatibility.
- Portfolio pipeline that generates `report.md`, `metrics.json`, five figures,
  `processed_cloud.ply`, and `transformation.json`.
- Deterministic synthetic demo-data generator in `examples/generate_demo_data.py`.
- Custom KDTree with nearest-neighbor, kNN, radius, batch, high-dimensional, and
  optional parallel query support.
- VoxelHashGrid with radius, bounded nearest-neighbor, kNN, box query, and voxel
  downsampling.
- Preprocessing: voxel downsampling, AABB crop, normalization, random/farthest
  sampling, statistical and radius outlier removal, and local PCA normals.
- Registration: SVD rigid transform, point-to-point ICP, point-to-plane ICP,
  robust Huber/Tukey/trimmed ICP, multiscale ICP, and compact custom GICP.
- Feature registration: ISS keypoints, local covariance-spectrum descriptors,
  descriptor matching, RANSAC transform estimation, and ICP refinement.
- Open3D FPFH/RANSAC/ICP path as an optional industrial baseline.
- RANSAC primitive fitting for plane, sphere, cylinder, and sequential primitive
  extraction.
- Geometry: PCA, AABB, OBB, distances, and primitive residuals.
- Segmentation: DBSCAN, Euclidean clustering, region growing, ground removal,
  object cluster summaries, and Markdown cluster reports.
- Benchmarks that emit CSV, JSON, Markdown, and PNG for KDTree, ICP, RANSAC,
  registration, GICP, and segmentation.
- Real-data preparation docs and examples for Stanford Bunny/Armadillo, KITTI
  Velodyne, and ModelNet small samples.
- Interview-oriented docs for algorithms, registration case study, and common
  Q&A.
- Optional C++17 KDTree demo under `cpp/`.

## Next Milestones

### v0.3.1 Benchmark Hardening

- Add memory profiling to benchmark JSON.
- Add repeat-count statistics with mean/std instead of single-run timing.
- Store benchmark environment metadata: CPU, OS, Python, NumPy, SciPy,
  scikit-learn, and Open3D versions.

### v0.3.2 Real Data Coverage

- Add tiny checksum-verified real-data fixtures that are safe to keep in git.
- Add CI smoke tests for real-data examples using generated miniature KITTI-like
  `.bin` and ModelNet-like `.off` files.
- Add SemanticKITTI or nuScenes documentation as optional LiDAR extensions.

### v0.4 Algorithm Depth

- Improve GICP from scalar Mahalanobis weighting to a fuller nonlinear
  covariance objective.
- Add Octree as a second spatial index for hierarchical range queries.
- Add SHOT-like descriptor experiments and compare against Open3D FPFH.
- Add adaptive DBSCAN or range-image clustering for uneven LiDAR density.

### v0.5 Visualization and Reporting

- Add a static docs asset gallery generated from the portfolio pipeline.
- Add an HTML report mode for the portfolio demo.
- Add benchmark comparison plots with confidence intervals.

### v0.6 Optional Systems Integration

- Expand the C++ demo into a pybind11 optional extension or a standalone CLI.
- Add a ROS2 perception-node wrapper design for streaming LiDAR frames.
- Add Docker/devcontainer instructions for reproducible reviewers' setup.
