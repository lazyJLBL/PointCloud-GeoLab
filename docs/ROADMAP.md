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
- Reviewer-facing CLI through `python -m pointcloud_geolab ...` plus legacy
  `python main.py --mode ...` compatibility.
- Portfolio pipeline that generates `report.md`, `metrics.json`, five figures,
  `processed_cloud.ply`, and `transformation.json`.
- Portfolio pipeline static HTML report generated next to `report.md`.
- Deterministic synthetic demo-data generator in `examples/generate_demo_data.py`.
- Custom KDTree with nearest-neighbor, kNN, radius, batch, high-dimensional, and
  optional parallel query support.
- VoxelHashGrid with radius, bounded nearest-neighbor, kNN, box query, and voxel
  downsampling.
- Preprocessing: voxel downsampling, AABB crop, normalization, random/farthest
  sampling, statistical and radius outlier removal, and local PCA normals.
- Registration: SVD rigid transform, point-to-point ICP, point-to-plane ICP,
  robust Huber/Tukey/trimmed ICP, multiscale ICP, and GICP-style
  covariance-weighted ICP.
- Feature registration: ISS keypoints, local covariance-spectrum descriptors,
  descriptor matching, RANSAC transform estimation, and ICP refinement.
- Open3D FPFH/RANSAC/ICP path as an optional comparison baseline.
- RANSAC primitive fitting for plane, sphere, cylinder, and sequential primitive
  extraction.
- Geometry: PCA, AABB, OBB, distances, and primitive residuals.
- Segmentation: DBSCAN, Euclidean clustering, region growing, ground removal,
  object cluster summaries, and Markdown cluster reports.
- Benchmarks that emit CSV, JSON, Markdown, and PNG for KDTree, ICP, RANSAC,
  registration, GICP-style ICP, and segmentation.
- Benchmark JSON/Markdown metadata for parameters, data scale, seed, Python,
  platform, and optional baseline package versions.
- Benchmark repeat statistics for local repeated runs, plus lightweight
  `tracemalloc` memory metadata.
- Tiny synthetic KITTI-like `.bin` and ModelNet-like `.off` format fixtures
  with checksum validation in `verify-core`.
- v0.1.1 hardening release notes, artifact manifest, and release-ready checker.
- Manual v0.1.1 tag and GitHub release.
- v1.0.0 portfolio-stable release notes and artifact manifest.
- User-provided KITTI-like LiDAR segmentation workflow with reports, metrics,
  figures, timing, and memory metadata. This is not an official KITTI
  benchmark.
- Real-data workflow verifier with a CI-safe synthetic dry-run.
- Scale benchmark quick gate with repeat statistics and local memory metadata.
- API stability, CLI reference, versioning, gallery, and v1 readiness docs.
- Portfolio pipeline implementation split into smaller input, metrics, figure,
  report, and runner modules while preserving the compatibility import path.
- Lightweight artifact schema checker for release, portfolio, and benchmark
  JSON.
- Repository audit script for local and optional GitHub release-review
  snapshots.
- Manual release gate workflow for benchmark and release-readiness checks.
- Verification scripts for benchmark artifacts and portfolio demo artifacts.
- Real-data preparation docs and examples for Stanford Bunny/Armadillo, KITTI
  Velodyne, and ModelNet small samples.
- Interview-oriented docs for algorithms, registration case study, and common
  Q&A.
- Optional C++17 KDTree demo under `cpp/`.

## Next Milestones

### v1.0.1 Web-Ready Cleanup

- Keep public API error contracts and documentation aligned.
- Keep path-aware IO error tests for supported point-cloud formats.
- Keep artifact schema checks lightweight and dependency-free.
- Keep repository audit output useful without requiring `gh` for local checks.
- Keep generated `outputs/`, benchmark bundles, demo data, and coverage reports
  out of Git.
- Keep Python support explicit at 3.10-3.12 and make ZIP downloads usable for
  local checks that do not require Git metadata.
- Keep the experimental Web Console isolated from the core Python package.

### v1.1 Experimental Web Console MVP

- Add a FastAPI backend that calls only stable public task API entry points for
  point-cloud processing tasks.
- Add a Vue 3 reviewer console for uploads, previews, task status, metrics, and
  artifact downloads.
- Keep benchmark timing labels clear that values are local machine references.

### v1.2 Web Visualization and Report Gallery

- Add confidence-interval style summaries once enough repeated local runs are
  available.
- Compare benchmark output schemas across releases.
- Expand static and Web Console visualization/report gallery coverage while
  keeping generated artifacts out of Git.

### v1.x Real Data Coverage

- Add checksum-verified public-domain micro fixtures if licensing allows, kept
  separate from benchmark claims.
- Add SemanticKITTI or nuScenes documentation as optional LiDAR extensions.
- Add an official real KITTI benchmark report only as a separate future task.

### v1.x Algorithm Depth

- Improve GICP-style covariance weighting from scalar Mahalanobis weights to a
  fuller nonlinear covariance objective.
- Add Octree as a second spatial index for hierarchical range queries.
- Add SHOT-like descriptor experiments and compare against Open3D FPFH.
- Add adaptive DBSCAN or range-image clustering for uneven LiDAR density.

### v1.x Optional Systems Integration

- Expand the C++ demo into a pybind11 optional extension or a standalone CLI.
- Add a ROS2 perception-node wrapper design for streaming LiDAR frames.
- Expand DevContainer notes only when the reviewer workflow needs it.
