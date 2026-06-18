# Coverage

The repository uses `pytest-cov` rather than an external hosted coverage badge.
This keeps the project self-contained for reviewers.

Run:

```bash
python -m pytest --cov=pointcloud_geolab
```

`pyproject.toml` enforces `fail_under = 65`. The threshold is intentionally
modest because optional Open3D/PyTorch/Plotly paths are separated from the core
geometry evidence, but it prevents coverage from silently collapsing.

For v0.1.1 pre-release work, the practical target is to keep moving total
coverage above 74% with useful tests for CLI errors, verifier rejection cases,
public API imports, fallback diagnostics, and optional-dependency boundaries.
The gate remains 65% so optional demos do not make routine review brittle.

Generate an HTML report:

```bash
python -m pytest --cov=pointcloud_geolab --cov-report=html
```

Open `htmlcov/index.html` after the command finishes.

## What Coverage Means Here

High-value tests compare algorithm outputs against known references:

- KDTree and VoxelHashGrid queries are compared with brute force.
- ICP and GICP-style tests use known rigid transforms, noise, outliers, low-overlap
  cases, bad initialization, and degenerate planar systems.
- RANSAC tests cover plane, sphere, and cylinder fitting with fixed seeds and
  controlled outlier ratios.
- PCA/OBB tests check extents, volume, and rotation stability.
- Segmentation tests check cluster counts, noise, and ground-removal reports.

Optional paths such as Open3D reconstruction and PyTorch PointNet are smoke
tested or skipped when dependencies are missing. They are not the core
correctness evidence for the geometry implementation.

## Current Expectation

Core geometry modules should stay meaningfully covered, and the CLI pipeline
should keep smoke coverage. Repository hygiene now runs as part of
`verify-core`, next to the coverage gate. Raising the threshold is future work
after optional dependency paths are split into their own coverage view.
