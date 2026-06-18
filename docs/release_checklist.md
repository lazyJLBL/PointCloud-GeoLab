# Release Checklist

This checklist is for v0.1.1 pre-release sanity checks. It prepares the
repository for review; it does not create a tag or publish a release.

## Local Environment

Install the dependency-light reviewer environment:

```bash
python -m pip install -e ".[dev,vis,bench]"
```

Run core checks:

```bash
make verify-core
```

If `make` is unavailable on Windows, run the same target with GNU Make for
Windows:

```bash
mingw32-make verify-core
```

## DevContainer

The repository includes:

- `.devcontainer/devcontainer.json`
- `.devcontainer/Dockerfile`

The container uses a Python slim image and installs `.[dev,vis,bench]`. It does
not install heavy optional Open3D/ML extras or download real datasets.

After opening the repository in a DevContainer, these commands should work:

```bash
python -m pytest
make verify-core
make verify-portfolio
```

Check the DevContainer setup without requiring a Docker daemon:

```bash
python scripts/check_devcontainer.py
```

If Docker is unavailable, the script reports a warning and still validates the
static configuration.

## Packaging

Check package metadata and the console entry point:

```bash
python scripts/check_packaging.py
```

If the `build` module is installed, the script builds an sdist and wheel in a
temporary copy of the repository. It does not write committed `dist/`, `build/`,
or `*.egg-info` artifacts.

If `build` is not installed, the script reports a warning and keeps the metadata
checks active.

## Tiny Dataset Fixtures

Validate committed synthetic format fixtures:

```bash
python scripts/check_dataset_fixtures.py
```

This checks the tiny KITTI-like `.bin`, ModelNet-like `.off`, and SHA256
manifest under `tests/fixtures/datasets/`. These files are format smoke tests
only and are not real dataset benchmarks.

## Portfolio And Benchmarks

Regenerate ignored demo artifacts locally:

```bash
make verify-portfolio
make verify-benchmarks
```

Expected generated reports:

- `outputs/portfolio_demo/report.md`
- `outputs/portfolio_demo/report.html`
- `outputs/benchmarks/benchmark_summary.md`

These outputs are ignored and should not be committed.

## Boundary Checks

Before release, confirm the public wording stays bounded:

- GICP is described as GICP-style covariance-weighted ICP, not a full nonlinear
  GICP optimizer.
- Synthetic demo outputs are smoke evidence, not real-data validation.
- Tiny KITTI-like and ModelNet-like fixtures are format smoke tests, not real
  dataset benchmarks.
- KITTI, Stanford Bunny, and ModelNet remain documented workflows requiring
  local files under `data/external/`.
- The repository is not a SLAM backend, CUDA stack, or Open3D/PCL replacement.

Do not create a `v0.1.1` tag unless that release action is explicitly requested.
