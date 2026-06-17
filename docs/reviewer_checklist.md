# Reviewer Checklist

This checklist is for a release reviewer who wants to verify the portfolio
without trusting committed generated artifacts.

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
```

Expected result: formatting, lint, tests, and the 65% coverage gate pass.

## 3. Run Portfolio Smoke

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
outputs/portfolio_demo/metrics.json
outputs/portfolio_demo/artifacts/transformation.json
```

The main reviewer-facing report is:

```text
outputs/portfolio_demo/report.md
```

## 4. Run Benchmark Verification

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
structurally validated, not just checked for existence.

## 5. Read Truthfulness Documents

- [AUDIT.md](../AUDIT.md)
- [Limitations](limitations.md)
- [Coverage](coverage.md)
- [Interview Notes](interview_notes.md)

Review especially:

- GICP-style covariance-weighted ICP is not a full nonlinear GICP optimizer.
- Feature fallback diagnostics do not mean descriptor registration succeeded.
- Synthetic demo outputs are smoke evidence, not broad real-data validation.
- Stanford Bunny, KITTI, and ModelNet are documented workflows requiring local
  datasets.

## 6. Check Repository Hygiene

```bash
git status --short --untracked-files=all
git ls-files \
  outputs results benchmark_results examples/demo_data \
  generated_demo_artifacts portfolio_demo
```

Expected result: no tracked generated outputs.
