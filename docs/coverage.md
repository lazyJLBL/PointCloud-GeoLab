# Coverage

The repository uses `pytest-cov` rather than an external hosted coverage badge.
This keeps the project self-contained for reviewers.

Run:

```bash
python -m pytest --cov=pointcloud_geolab
```

Generate an HTML report:

```bash
python -m pytest --cov=pointcloud_geolab --cov-report=html
```

Open `htmlcov/index.html` after the command finishes.

## What Coverage Means Here

High-value tests compare algorithm outputs against known references:

- KDTree and VoxelHashGrid queries are compared with brute force.
- ICP/GICP tests use known rigid transforms, noise, outliers, low-overlap
  cases, bad initialization, and degenerate planar systems.
- RANSAC tests cover plane, sphere, and cylinder fitting with fixed seeds and
  controlled outlier ratios.
- PCA/OBB tests check extents, volume, and rotation stability.
- Segmentation tests check cluster counts, noise, and ground-removal reports.

Optional paths such as Open3D reconstruction and PyTorch PointNet are smoke
tested or skipped when dependencies are missing. They are not the core
correctness evidence for the geometry implementation.

## Current Expectation

The project does not enforce a hard coverage threshold yet. A reviewer should
expect the core geometry modules to have meaningful unit coverage and the CLI
pipeline to have smoke coverage. Future work can add a minimum threshold after
the optional dependency paths are separated more cleanly from core modules.
