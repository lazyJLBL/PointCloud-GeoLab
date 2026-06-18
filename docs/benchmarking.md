# Benchmarking

Benchmarks are designed to make tradeoffs visible rather than to claim portable
performance. Timing and memory numbers should be regenerated on the reviewer
machine.

## Suites

- `kdtree`: custom KDTree and VoxelHashGrid vs brute force, Open3D, SciPy
  cKDTree, and sklearn KDTree when available.
- `icp`: custom ICP, Huber ICP, trimmed ICP, and optional Open3D ICP under
  perturbations and source outliers.
- `ransac`: custom RANSAC primitive fitting vs NumPy PCA plane and optional
  Open3D plane segmentation under increasing outlier ratios.
- `registration`: ICP vs FPFH+RANSAC+ICP under large initial rotations.
- `gicp`: GICP-style covariance-weighted ICP vs point-to-point ICP. This is not
  a full nonlinear GICP optimizer.
- `segmentation`: Euclidean clustering runtime over synthetic cluster sizes.

Run all quick suites:

```bash
python -m pointcloud_geolab benchmark --suite all --quick --output outputs/benchmarks
```

Each run writes CSV, Markdown, JSON, PNG, and `metrics.json`. JSON and Markdown
include run metadata: parameters, data scale, random seed, Python/platform,
repeat configuration, memory metadata, and optional package versions.

Verify generated artifacts:

```bash
python scripts/verify_benchmarks.py --output-dir outputs/benchmarks
```

Validate the benchmark JSON schema directly:

```bash
python scripts/check_artifact_schema.py \
  --benchmark-json outputs/benchmarks/all_benchmark.json
```

## Repeat Statistics

The benchmark CLI accepts `--repeat`, with default `1`.

```bash
python -m pointcloud_geolab benchmark \
  --suite kdtree \
  --quick \
  --repeat 3 \
  --output outputs/benchmarks/kdtree-repeat
```

When `--repeat > 1`, rows keep the first run's existing timing fields and add
aggregate fields for each timing metric:

- `<field>_mean`
- `<field>_std`
- `<field>_min`
- `<field>_max`

Metadata records the base seed and seed strategy: base seed plus the zero-based
repeat index. This keeps the run reproducible while still avoiding a single
timing sample as the only evidence.

## Memory Metadata

Benchmark JSON records lightweight memory metadata from Python `tracemalloc`:

- `available`
- `method`
- `current_bytes`
- `peak_bytes`
- `note`

These values describe Python allocations observed during the local benchmark
process. They are useful for spotting large regressions on the same machine, but
they are not a portable memory-performance claim.

## Dataset Fixture Boundary

`tests/fixtures/datasets/` contains tiny synthetic KITTI-like and ModelNet-like
format fixtures. They are checksum-validated by:

```bash
python scripts/check_dataset_fixtures.py
```

These fixtures only prove that small `.bin` and `.off` files can be parsed and
validated in CI. They are not real KITTI or ModelNet benchmark data. Real
benchmark inputs still need to be prepared locally under `data/external/`.

## Interpretation Examples

- The custom KDTree is useful for explaining nearest-neighbor search and is fine
  at small scale; optimized SciPy/sklearn implementations are expected to win at
  large scale.
- Plain ICP is accurate near the correct pose but can fail under large initial
  rotations; FPFH+RANSAC+ICP is slower but more robust.
- Robust ICP improves correspondence weighting when source-only outliers are
  present, but it still needs enough overlap.
- GICP-style covariance-weighted ICP spends more time per iteration because it
  estimates and uses local covariance matrices. This is not a full nonlinear
  GICP optimizer.
