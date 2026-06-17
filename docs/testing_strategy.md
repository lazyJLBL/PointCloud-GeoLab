# Testing Strategy

PointCloud-GeoLab uses tests as release evidence. The goal is not only to make
the package importable, but also to keep algorithm claims bounded by repeatable
checks.

## Unit Tests

Unit tests cover the small algorithmic pieces directly:

- KDTree and VoxelHashGrid query behavior.
- SVD rigid alignment and ICP convergence on controlled inputs.
- RANSAC primitives with known inlier/outlier structure.
- PCA, AABB, OBB, preprocessing, segmentation, and transform utilities.
- CLI/API task envelopes and error paths.

Most unit tests use small NumPy arrays so failures are easy to inspect.

## Property-style Checks

Several tests compare custom code against brute-force or invariant behavior:

- KDTree and VoxelHashGrid results are checked against brute-force search.
- RANSAC tests use fixed outlier ratios and expected inlier recovery.
- Transform tests verify shape, invertibility, and metric consistency.

These are not exhaustive formal proofs, but they catch common implementation
regressions better than one golden output file.

## Fixed Seeds

Randomized tests and demos use explicit seeds. This keeps CI deterministic while
still exercising noisy data, outliers, and synthetic scenes.

## Synthetic Fixtures

Synthetic fixtures are used for smoke tests and small algorithm checks. They are
valuable because the expected geometry is known, but they are not real-data
validation.

Real Stanford Bunny, KITTI, and ModelNet workflows are documented separately and
require local files under `data/external/`.

## Benchmark Verifier

`scripts/verify_benchmarks.py` checks that benchmark outputs contain valid CSV,
JSON, Markdown, and PNG content. It rejects empty files, malformed JSON, missing
metadata, missing rows, invalid Markdown structure, and invalid PNG headers.

This prevents benchmark evidence from becoming a set of placeholder files.

## Portfolio Verifier

`scripts/verify_portfolio.py` checks the generated portfolio bundle:

- Markdown report sections.
- Metrics JSON schema.
- Key PNG figures.
- Transform JSON shape and numeric values.
- Expected gallery and pipeline artifacts.

The verifier supports reviewer confidence without committing generated outputs.

## Optional Dependency Skips

Optional Open3D, Plotly, PyTorch, LAS/LAZ, SciPy, scikit-learn, and pandas paths
are treated as baselines or demos. Tests either skip clearly when a dependency
is missing or verify that commands return a useful message.

Core tests should not require heavyweight optional packages.

## Coverage Gate

The current coverage gate is 65%. That threshold is intentionally modest because
some optional visualization, ML, and integration paths are expensive or
environment-sensitive.

The project still aims to move coverage upward with useful tests:

- CLI and API error paths.
- Verifier rejection cases.
- Public API metadata.
- Fallback diagnostics.
- I/O fallback readers and writers.

The rule for future coverage work is simple: add tests that would catch a real
regression, not tests that only execute lines.
