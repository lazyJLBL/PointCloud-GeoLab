# TODO Review

These items are intentionally not described as completed features.

## High Value

- Add repeated benchmark runs with mean/std, confidence intervals, and memory
  measurements.
- Add tiny checksum-verified real-data fixtures for CI smoke tests.
- Separate core coverage reporting from optional Open3D/PyTorch paths.
- Add an HTML version of the portfolio pipeline report.
- Add a stricter public API stability policy before publishing beyond portfolio
  use.

## Algorithm Improvements

- Upgrade GICP from scalar covariance-derived weights to a fuller nonlinear
  covariance objective.
- Add adaptive or range-aware clustering for uneven LiDAR density.
- Add a clearer feature-registration failure analysis for symmetric/repetitive
  objects.
- Compare custom descriptors with Open3D FPFH on prepared real data.
- Add more degeneracy tests for collinear registration and nearly spherical PCA.

## Engineering Improvements

- Add benchmark memory profiling and optional CPU metadata.
- Add Docker/devcontainer instructions for reviewer reproducibility.
- Move generated benchmark artifacts out of git or make their update workflow
  explicit.
- Add type-checking with mypy or pyright after public interfaces stabilize.
- Add a small release checklist for resume/demo updates.

## Presentation Improvements

- Regenerate README figures from `examples/gallery_demo.py` as part of a
  documented release process.
- Add a short narrated walkthrough script for interview demos.
- Add one concise architecture diagram in docs that links CLI, API, algorithms,
  examples, and benchmarks.
