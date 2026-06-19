# Changelog

All notable changes for PointCloud-GeoLab are recorded here.

## Unreleased

### Fixed

- Post-release v1.0.0 wording now consistently describes the latest stable
  release instead of a release candidate or future target.
- Repository hygiene checks now guard current release docs against stale
  release-candidate and issue-state wording.
- KITTI-like workflow, scale benchmark, artifact schema, documented command,
  and CLI JSON-error edge cases now have tighter validation.

## v1.0.0 - 2026-06-18

### Added

- v1.0.0 release notes and artifact manifest for portfolio-stable review.
- User-provided KITTI-like LiDAR segmentation workflow with JSON, Markdown,
  HTML, PLY, PNG, timing, and memory outputs.
- Real-data workflow verifier with a CI-safe synthetic KITTI-like dry-run.
- Scale benchmark with 1k/10k quick sizes, optional larger manual sizes,
  repeat statistics, memory metadata, machine info, JSON, CSV, Markdown, and
  PNG outputs.
- API and CLI stability documentation for stable, experimental, deprecated,
  and optional surfaces.
- Gallery documentation and small synthetic/tiny-fixture assets for reviewer
  navigation.
- v1 readiness checker and `verify-v1-candidate` Makefile target.
- v0.1.2 planning notes for API, IO, artifact schema, pipeline structure,
  release gate, and repository audit follow-up work.
- Lightweight artifact schema checker for release manifests, portfolio metrics,
  and benchmark JSON.
- Manual `release-gate.yml` workflow for benchmark and release-gate
  verification without publishing artifacts.
- Repository audit script for local version, tag, release, issue, CI, coverage,
  generated-artifact, optional dependency, and limitation summaries.
- Optional dependency diagnostics helper for Open3D, Plotly, laspy, PyTorch,
  SciPy, scikit-learn, and pandas.

### Changed

- Current package, citation, release-ready metadata, and README version wording
  are set to v1.0.0 while v0.1.0 and v0.1.1 remain historical releases.
- GICP-style covariance-weighted ICP remains Experimental and outside the
  stable API commitment; it is not a full nonlinear GICP optimizer.
- KITTI documentation now separates the user-provided single-frame workflow
  from any official real KITTI benchmark claim.
- v0.1.1 wording now describes the published hardening release rather than
  candidate wording.
- Portfolio pipeline internals were split into input, metrics, figures, report,
  and runner modules while preserving the public `pointcloud_geolab.pipeline`
  import path.
- Point-cloud IO error messages now include the path and reason for missing,
  unsupported, empty, bad-header, and bad-numeric-data cases.
- CLI configuration and batch errors emit a structured JSON envelope when
  `--format json` is requested.
- Repository hygiene checks cover workflow files, Makefile, docs, and script
  text shape.

## v0.1.1 - 2026-06-18

### Added

- Repository hygiene checks for tracked generated paths, README links,
  overclaim wording, text-file shape, and release metadata consistency.
- Project boundary, architecture, and testing strategy documentation for
  reviewer orientation.
- CI status helper for checking the latest Tests workflow with the GitHub CLI.
- Benchmark `--repeat` support with timing mean/std/min/max fields for repeated
  local runs.
- Lightweight benchmark memory metadata using Python `tracemalloc`.
- Static HTML portfolio report generated next to `report.md`.
- Benchmarking documentation for repeat statistics and local memory metadata.
- DevContainer configuration for dependency-light reviewer reproduction.
- Packaging sanity checker for pyproject metadata and optional temporary
  sdist/wheel builds.
- Release checklist covering local validation, DevContainer use, packaging, and
  bounded v0.1.1 scope.
- Tiny synthetic KITTI-like `.bin` and ModelNet-like `.off` fixtures with
  SHA256 manifest validation.
- Release notes and artifact manifest for reviewer handoff.

### Changed

- Markdown and TOML formatting cleanup for easier review.
- `verify-core` now includes the repository hygiene check, and CI keeps running
  that target on Python 3.10, 3.11, and 3.12.
- GitHub Actions now use newer official checkout/setup-python actions with
  read-only repository contents permission.
- Coverage improved with additional CLI, verifier, public API, visualization,
  reconstruction, and point-cloud I/O tests.
- Benchmark and portfolio verifiers now check repeat, memory, and HTML report
  structure in addition to artifact presence.
- Coverage gate raised from 65% to 70% after local coverage reached 78.61%.
- `verify-core` now includes DevContainer and packaging sanity checks without
  requiring Docker or the `build` module to be installed.
- `verify-core` now checks tiny dataset fixtures without downloading external
  datasets.
- Release metadata now identifies v0.1.1 as the current package version while
  preserving v0.1.0 as a historical portfolio release.

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
