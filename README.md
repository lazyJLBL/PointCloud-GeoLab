# PointCloud-GeoLab

[![Tests][tests-badge]][tests-workflow]
![Python][python-badge]
![License][license-badge]
![Coverage threshold][coverage-badge]

PointCloud-GeoLab is a compact point-cloud geometry portfolio project. It keeps
the core math visible in Python and NumPy while using SciPy, scikit-learn,
Open3D, Plotly, and PyTorch only as optional baselines or demos.

The goal is not to replace Open3D or PCL. The goal is to make KDTree search,
ICP, RANSAC primitive fitting, PCA/OBB, GICP-style covariance-weighted ICP, and
LiDAR segmentation understandable, runnable, and testable.

## One Command Demo

```bash
python -m pip install -e ".[dev,vis,bench]"
python examples/generate_demo_data.py --output examples/demo_data
python -m pointcloud_geolab pipeline \
  --input examples/demo_data \
  --output outputs/portfolio_demo
```

Open `outputs/portfolio_demo/report.md` or `outputs/portfolio_demo/report.html`.

Expected files:

```text
outputs/portfolio_demo/
|-- report.md
|-- report.html
|-- metrics.json
|-- figures/
|   |-- raw_pointcloud.png
|   |-- downsampled.png
|   |-- registration_before_after.png
|   |-- segmentation_result.png
|   `-- bounding_box_or_normals.png
`-- artifacts/
    |-- processed_cloud.ply
    `-- transformation.json
```

Representative portfolio-pipeline figures:

![Raw point cloud](docs/assets/portfolio_raw_pointcloud.png)
![Registration before and after](docs/assets/portfolio_registration_before_after.png)
![Segmentation result](docs/assets/portfolio_segmentation_result.png)

## Implementation Status

### Core-tested

- **KDTree**: nearest, kNN, radius, batch, high-dimensional, duplicate, empty,
  and boundary tests.
- **VoxelHashGrid**: radius, nearest, kNN, box query, voxel downsampling, empty
  input, and brute-force consistency tests.
- **Preprocessing**: voxel downsampling, cropping, normalization, sampling,
  outlier filtering, and local PCA normals.
- **ICP variants**: SVD, point-to-point ICP, point-to-plane ICP, robust ICP, and
  multi-scale ICP tests.
- **RANSAC primitives**: plane, sphere, cylinder, and sequential extraction
  with fixed-seed outlier tests.
- **PCA / OBB**: principal axes, degenerate geometry, and rotation-stability
  tests.
- **Segmentation**: DBSCAN, Euclidean clustering, region growing, ground
  removal, and object reports.

### Demo-ready

- **Portfolio pipeline**: one command creates Markdown and HTML reports,
  metrics, figures, PLY artifacts, and transform JSON.
- **Benchmarks**: CLI emits CSV, JSON, Markdown, PNG, parameters, seed,
  platform, repeat statistics, memory metadata, and dependency metadata.

### Experimental

- **GICP-style covariance-weighted ICP**: uses covariance-derived scalar weights
  and weighted SVD. This is not a full nonlinear GICP optimizer.
- **Feature registration**: ISS keypoints, local descriptors, transform RANSAC,
  and ICP refinement. Fallback diagnostics do not mean descriptor registration
  succeeded.

### Optional

- **Open3D / ML / reconstruction**: Open3D and PointNet paths are isolated from
  core tests and skip or report cleanly when unavailable.

### Documented workflow

- **Real data workflows**: Stanford Bunny, KITTI, and ModelNet instructions
  expect local files under `data/external/`.

See [AUDIT.md](AUDIT.md) for the detailed truthfulness audit.

## Core Commands

Run the portfolio pipeline:

```bash
python examples/generate_demo_data.py --output examples/demo_data
python -m pointcloud_geolab pipeline \
  --input examples/demo_data \
  --output outputs/portfolio_demo
python scripts/verify_portfolio.py --quick
```

Run benchmarks and verify the output bundle:

```bash
python -m pointcloud_geolab benchmark \
  --suite all \
  --quick \
  --output outputs/benchmarks
python scripts/verify_benchmarks.py --output-dir outputs/benchmarks
```

Run real-data case studies after preparing local data:

```bash
python examples/real_bunny_registration.py \
  --data-dir data/external/stanford/bunny_pair \
  --output-dir outputs/real_bunny

python examples/kitti_lidar_segmentation.py \
  --frame data/external/kitti/velodyne/000000.bin \
  --output-dir outputs/kitti_segmentation
```

If the real data is missing, these scripts exit with preparation instructions.
Synthetic demos are smoke tests only and should not be described as real-data
results.

## Benchmark Notes

The benchmark entry point is:

```bash
python -m pointcloud_geolab benchmark \
  --suite all \
  --quick \
  --output outputs/benchmarks
```

Each suite writes CSV, JSON, Markdown, PNG, and `metrics.json`. JSON reports
include parameters, random seed, data scale, Python/platform metadata, and
optional dependency versions. Timing numbers are machine-specific, so fixed
results are not committed as claims.

Use `--repeat` when you want local timing aggregates:

```bash
python -m pointcloud_geolab benchmark \
  --suite kdtree \
  --quick \
  --repeat 3 \
  --output outputs/benchmarks/kdtree-repeat
```

For `--repeat > 1`, JSON and CSV rows include mean, standard deviation, minimum,
and maximum for timing fields. Benchmark JSON also records lightweight
`tracemalloc` peak-memory metadata as a local reference, not a portable
performance promise.

Baseline coverage:

- KDTree: brute force, optional SciPy `cKDTree`, optional sklearn `KDTree`, and
  optional Open3D KDTree.
- ICP: custom variants and optional Open3D ICP.
- RANSAC: custom primitive RANSAC, NumPy PCA plane baseline, and optional
  Open3D plane segmentation.
- GICP-style covariance-weighted ICP: compared with point-to-point ICP; not a
  full nonlinear GICP optimizer.

## Verification

Direct commands:

```bash
python -m pip install -e ".[dev,vis,bench]"
python -m compileall -q main.py pointcloud_geolab tests examples scripts benchmarks
python -m ruff check .
python -m black --check .
python -m pytest --cov=pointcloud_geolab
python scripts/check_repo_hygiene.py
python examples/generate_demo_data.py --output examples/demo_data
python -m pointcloud_geolab pipeline \
  --input examples/demo_data \
  --output outputs/portfolio_demo
python scripts/verify_portfolio.py --quick
python -m pointcloud_geolab benchmark \
  --suite all \
  --quick \
  --output outputs/benchmarks
python scripts/verify_benchmarks.py --output-dir outputs/benchmarks
```

Make targets:

```bash
make verify-core
make verify-portfolio
make verify-benchmarks
make verify-full
```

`verify-core` runs compile, lint, format, tests with coverage, and repository
hygiene checks. CI runs `verify-core` and `verify-portfolio`.

## Limitations

- ICP, robust ICP, multi-scale ICP, and the GICP-style implementation are local
  optimizers.
- GICP-style covariance-weighted ICP uses scalar covariance-derived weights;
  this is not a full nonlinear GICP optimizer.
- Feature registration is educational and benchmarkable, but not a replacement
  for mature descriptors in Open3D/PCL.
- Fallback output is diagnostic only. It must not be described as descriptor
  registration success.
- DBSCAN and Euclidean clustering use global radius thresholds and are sensitive
  to LiDAR density changes.
- Large LiDAR scenes need streaming/chunking and more careful memory profiling.

## Documentation

- [Algorithms](docs/algorithms.md)
- [Public API](docs/api.md)
- [Project Boundary](docs/project_boundary.md)
- [Architecture](docs/architecture.md)
- [Testing Strategy](docs/testing_strategy.md)
- [Demo Walkthrough](docs/demo_walkthrough.md)
- [Limitations](docs/limitations.md)
- [Benchmarking](docs/benchmarking.md)
- [Datasets](docs/datasets.md)
- [Registration Case Study](docs/case_study_registration.md)
- [Stanford Bunny Case Study](docs/case_study_bunny.md)
- [KITTI LiDAR Case Study](docs/case_study_kitti.md)
- [Coverage](docs/coverage.md)
- [Interview Notes](docs/interview_notes.md)
- [Reviewer Checklist](docs/reviewer_checklist.md)
- [Portfolio Review Template](docs/portfolio_report_template.md)
- [Roadmap](docs/ROADMAP.md)
- [Changelog](CHANGELOG.md)

## Resume Description

Built PointCloud-GeoLab, a point-cloud geometry portfolio project with custom
KDTree and VoxelHashGrid spatial indexes, ICP and GICP-style registration
variants, RANSAC primitive fitting, PCA/OBB geometry analysis, LiDAR
segmentation, fixed-seed tests, reproducible benchmarks, and documented
real-data workflows.

[tests-badge]: https://github.com/lazyJLBL/PointCloud-GeoLab/actions/workflows/tests.yml/badge.svg
[tests-workflow]: https://github.com/lazyJLBL/PointCloud-GeoLab/actions/workflows/tests.yml
[python-badge]: https://img.shields.io/badge/python-3.10%2B-blue
[license-badge]: https://img.shields.io/badge/license-MIT-green
[coverage-badge]: https://img.shields.io/badge/coverage%20threshold-70%25-informational
