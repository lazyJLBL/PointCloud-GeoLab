# Reviewer Checklist

This checklist is for a release reviewer who wants to verify the portfolio
without trusting committed generated artifacts.

The current package version is `0.1.1`. It is a portfolio, learning, and
reviewer-oriented release candidate, not a production LiDAR stack or a real
KITTI benchmark release.

## 1. Install

```bash
git clone https://github.com/lazyJLBL/PointCloud-GeoLab.git
cd PointCloud-GeoLab
python -m pip install -e ".[dev,vis,bench]"
```

Optional extras for additional demos:

```bash
python -m pip install -e ".[open3d,ml,io]"
```

## 2. Run Core Checks

```bash
make verify-core
```

Equivalent commands:

```bash
python -m compileall -q main.py pointcloud_geolab tests examples scripts benchmarks
python -m ruff check .
python -m black --check .
python -m pytest --cov=pointcloud_geolab
python scripts/check_repo_hygiene.py
python scripts/check_devcontainer.py
python scripts/check_packaging.py
python scripts/check_dataset_fixtures.py
```

Expected result: formatting, lint, tests, the 70% coverage gate, and repository
hygiene, DevContainer, packaging, and tiny fixture checks pass.

## 3. Optional DevContainer Reproduction

Open the repository with VS Code Dev Containers or another compatible tool.
The container uses a Python slim image and installs `.[dev,vis,bench]`.

Inside the container:

```bash
python -m pytest
make verify-core
make verify-portfolio
```

Open3D/ML extras and real datasets are optional and not installed in the default
review container.

## 4. Run Portfolio Smoke

```bash
make verify-portfolio
```

Equivalent commands:

```bash
python examples/generate_demo_data.py --output examples/demo_data
python -m pointcloud_geolab pipeline \
  --input examples/demo_data \
  --output outputs/portfolio_demo
python scripts/verify_portfolio.py --quick --output-dir outputs
```

Inspect:

```text
outputs/portfolio_demo/report.md
outputs/portfolio_demo/report.html
outputs/portfolio_demo/metrics.json
outputs/portfolio_demo/artifacts/transformation.json
```

The main reviewer-facing reports are:

```text
outputs/portfolio_demo/report.md
outputs/portfolio_demo/report.html
```

## 5. Run Benchmark Verification

```bash
make verify-benchmarks
```

Equivalent commands:

```bash
python -m pointcloud_geolab benchmark \
  --suite all \
  --quick \
  --output outputs/benchmarks
python scripts/verify_benchmarks.py --output-dir outputs/benchmarks
```

Expected result: CSV, JSON, Markdown, and PNG benchmark artifacts are parsed or
structurally validated, not just checked for existence. JSON metadata should
include repeat configuration and lightweight `tracemalloc` memory metadata.

Tiny dataset fixtures can be checked separately:

```bash
python scripts/check_dataset_fixtures.py
```

Expected result: the synthetic KITTI-like `.bin`, ModelNet-like `.off`, and
manifest checksums validate. These are format smoke tests, not real benchmarks.

For local repeat statistics:

```bash
python -m pointcloud_geolab benchmark \
  --suite kdtree \
  --quick \
  --repeat 3 \
  --output outputs/benchmarks/kdtree-repeat
python scripts/verify_benchmarks.py \
  --output-dir outputs/benchmarks/kdtree-repeat \
  --suite kdtree
```

## 6. Check Packaging Sanity

```bash
python scripts/check_packaging.py
```

If the `build` module is available, this builds an sdist and wheel in a
temporary copy. Generated `dist/`, `build/`, and `*.egg-info` files should not be
committed.

## 7. Read Truthfulness Documents

- [AUDIT.md](../AUDIT.md)
- [Limitations](limitations.md)
- [Coverage](coverage.md)
- [Interview Notes](interview_notes.md)
- [Release Checklist](release_checklist.md)

Review especially:

- GICP-style covariance-weighted ICP is not a full nonlinear GICP optimizer.
- Feature fallback diagnostics do not mean descriptor registration succeeded.
- Synthetic demo outputs are smoke evidence, not broad real-data validation.
- Tiny KITTI-like and ModelNet-like fixtures are synthetic format smoke tests,
  not real dataset benchmarks.
- Stanford Bunny, KITTI, and ModelNet are documented workflows requiring local
  datasets.
- v0.1.1 pre-release work still does not include real KITTI benchmark results,
  a full nonlinear GICP optimizer, a SLAM backend, CUDA acceleration, or a new
  release tag.

## 8. Check Repository Hygiene

```bash
python scripts/check_repo_hygiene.py
git status --short --untracked-files=all
git ls-files \
  outputs results benchmark_results examples/demo_data \
  generated_demo_artifacts portfolio_demo
```

Expected result: no tracked generated outputs.

## 9. Check Latest CI Status

After pushing a review branch or main update:

```bash
python scripts/check_ci_status.py --branch main --workflow tests.yml
```

This helper uses the GitHub CLI when available. If `gh` is missing or not
authenticated, it reports that clearly instead of printing a traceback.

## 10. Release Candidate Gate

Before creating a manual tag or GitHub release, run:

```bash
make verify-release-candidate
```

Expected result: core checks, portfolio verification, benchmark verification,
and release-ready metadata checks pass. The release artifact manifest is
`docs/releases/v0.1.1_artifacts.json`.
