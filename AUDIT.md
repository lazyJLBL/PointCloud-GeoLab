# PointCloud-GeoLab Audit

Audit date: 2026-06-17

Scope checked: `README.md`, `docs/`, `pyproject.toml`, `pointcloud_geolab/`,
`tests/`, `examples/`, `benchmarks/`, `scripts/`, `Makefile`, and CI.

## Verification Commands

The intended verification set is:

```bash
python -m pip install -e ".[dev,vis,bench]"
python -m compileall -q main.py pointcloud_geolab tests examples scripts benchmarks
python -m ruff check .
python -m black --check .
python -m pytest --cov=pointcloud_geolab
python examples/generate_demo_data.py --output examples/demo_data
python -m pointcloud_geolab pipeline --input examples/demo_data --output outputs/portfolio_demo
python scripts/verify_portfolio.py --quick
python -m pointcloud_geolab benchmark --suite all --quick --output outputs/benchmarks
python scripts/verify_benchmarks.py --output-dir outputs/benchmarks
```

CI runs `make verify-core` and `make verify-portfolio`.

## Status Summary

| Area | Status | Evidence and boundary |
|---|---|---|
| KDTree and VoxelHashGrid | Core-tested | Unit tests cover nearest, kNN, radius, batch, empty, duplicate, high-dimensional, and brute-force consistency cases. |
| Preprocessing, PCA/AABB/OBB, primitive RANSAC, segmentation | Core-tested | Deterministic tests cover geometry outputs, outlier cases, ground/object reports, and fixed-seed primitive fitting. |
| ICP variants | Core-tested | Point-to-point, point-to-plane, robust, trimmed, and multi-scale ICP have convergence and failure-mode tests. Diagnostics include initial/final RMSE, fitness, correspondence count, residual history, and step norms. |
| GICP-style covariance-weighted ICP | Experimental | Uses local covariances to compute scalar Mahalanobis weights and solves weighted SVD updates. This is not a full nonlinear GICP optimizer. |
| Feature registration | Experimental | ISS keypoints, local descriptors, RANSAC transform estimation, and ICP refinement exist. Geometry fallback is diagnostic only and does not mean descriptor registration succeeded. |
| Portfolio pipeline | Demo-ready | Generates report, metrics, figures, processed cloud, and transform JSON from deterministic demo data. |
| Benchmarks | Demo-ready | Benchmark outputs include CSV, JSON, Markdown, PNG, parameters, seed, platform, and dependency metadata. |
| Real data workflows | Documented workflow | Stanford Bunny, KITTI, and ModelNet workflows require local data under `data/external/`; large datasets are not committed. |
| Open3D, reconstruction, PointNet, Plotly HTML | Optional | Useful demos or baselines, but not required for core geometry correctness. |

## Fixes Applied

- Stopped tracking generated `outputs/` artifacts upstream and added ignore
  rules for generated outputs, benchmark results, external data, coverage
  artifacts, logs, and local demo data.
- Added benchmark artifact verification that parses CSV/JSON/Markdown and
  validates PNG structure.
- Added portfolio verification for report sections, metrics schema, key PNGs,
  and transform JSON.
- Added coverage threshold configuration with `fail_under = 65`.
- Added explicit public API exports and `tests/test_public_api.py`.
- Updated README and API docs to avoid overstating experimental or optional work.
- Added ICP/GICP diagnostics and explicit `full_nonlinear_gicp: false`.
- Made descriptor geometry fallback opt-in and diagnostic rather than silent.
- Unified local and CI validation around `verify-core`, `verify-portfolio`,
  `verify-benchmarks`, and `verify-full`.

## Remaining Risks

- Benchmark timing remains machine-dependent and should be regenerated locally.
- Feature registration needs broader real-data and perturbation tests before it
  can be described as robust.
- The GICP-style implementation is useful for explaining covariance weighting,
  but it is still not a full nonlinear optimizer.
- Optional dependency behavior can vary by Open3D, PyTorch, SciPy, sklearn, and
  Plotly versions.
- KITTI-scale processing still needs streaming, range-aware clustering, and
  memory profiling before it should be presented as production LiDAR tooling.
