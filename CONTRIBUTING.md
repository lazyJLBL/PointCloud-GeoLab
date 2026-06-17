# Contributing

Thanks for taking a look at PointCloud-GeoLab. This is a learning and portfolio
project, so contributions should keep the algorithms inspectable and the claims
honest.

## Setup

```bash
git clone https://github.com/lazyJLBL/PointCloud-GeoLab.git
cd PointCloud-GeoLab
python -m pip install -e ".[dev,vis,bench]"
```

Optional extras:

```bash
python -m pip install -e ".[open3d,ml,io]"
```

## Local Checks

Run the core checks before opening a pull request:

```bash
python -m compileall -q main.py pointcloud_geolab tests examples scripts benchmarks
python -m ruff check .
python -m black --check .
python -m pytest --cov=pointcloud_geolab
```

On systems with GNU Make:

```bash
make verify-core
make verify-portfolio
make verify-benchmarks
```

## Pull Requests

- Keep changes scoped and explain the evidence you ran.
- Add or update tests for behavior changes.
- Do not commit generated outputs such as `outputs/`, `results/`,
  `benchmark_results/`, or `examples/demo_data/`.
- Do not describe synthetic demos, fallbacks, or approximate algorithms as
  real-data or production successes.
- Keep GICP wording precise: this project has GICP-style covariance-weighted
  ICP, not a full nonlinear GICP optimizer.

## Documentation

When changing public behavior, update the closest relevant docs:

- `README.md` for user-facing workflow changes.
- `AUDIT.md` for truthfulness and status changes.
- `docs/limitations.md` for boundaries and known failure modes.
- `docs/api.md` for stable public API updates.
