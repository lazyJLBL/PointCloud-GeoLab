# Changelog

All notable changes for PointCloud-GeoLab are recorded here.

## Unreleased

### Added

- Repository hygiene checks for tracked generated paths, README links,
  overclaim wording, text-file shape, and release metadata consistency.
- Project boundary, architecture, and testing strategy documentation for
  reviewer orientation.

### Changed

- Markdown and TOML formatting cleanup for easier review.
- `verify-core` now includes the repository hygiene check.
- Coverage improved with additional CLI and point-cloud I/O error-path tests.

## v0.1.0 Portfolio Release - 2026-06-17

### Added

- GitHub Actions CI for Python 3.10, 3.11, and 3.12.
- Makefile verification targets: `verify-core`, `verify-portfolio`,
  `verify-benchmarks`, and `verify-full`.
- Public API exports through `pointcloud_geolab.api` and package `__all__`,
  with `docs/api.md` limited to stable entry points.
- Portfolio pipeline that emits a Markdown report, metrics JSON, figures,
  processed point cloud, and transform JSON from deterministic demo data.
- Benchmark verification for CSV, JSON, Markdown, PNG, parameters, seed, and
  platform metadata.
- Portfolio verification for reports, metrics schema, key images, transform
  JSON, and README-linked evidence.
- Coverage gate with `fail_under = 65`.
- Interview notes and reviewer checklist for release review.

### Changed

- README and AUDIT now use bounded statuses: Core-tested, Demo-ready,
  Experimental, Optional, and Documented workflow.
- GICP wording is consistently framed as GICP-style covariance-weighted ICP.
  This is not a full nonlinear GICP optimizer.
- Feature registration fallback is opt-in and diagnostic. Fallback output must
  not be described as descriptor registration success.
- CI runs `verify-core` and `verify-portfolio` using dependency-light portfolio
  demos.

### Removed

- Generated `outputs/` and `results/` artifacts were removed from the tracked
  repository state and covered by `.gitignore`.

### Known Limits

- Benchmark timings are machine-specific and should be regenerated locally.
- Real Stanford Bunny, KITTI, and ModelNet workflows are documented workflows;
  datasets are not committed.
- Open3D, PyTorch, Plotly, and LAS/LAZ support remain optional paths rather
  than core correctness evidence.
