# Artifact Schema

PointCloud-GeoLab uses lightweight schema checks to keep reviewer artifacts
machine-readable without adding a `jsonschema` dependency.

Run:

```bash
python scripts/check_artifact_schema.py
```

The default command validates `docs/releases/v1.0.0_artifacts.json`. Optional
arguments can validate generated portfolio and benchmark files:

```bash
python scripts/check_artifact_schema.py \
  --portfolio-metrics outputs/portfolio_demo/metrics.json \
  --benchmark-json outputs/benchmarks/all_benchmark.json
```

## Release Manifest

The release manifest must include:

- `version`
- `release_date`
- `commit`
- `local_verification_commands`
- `expected_generated_artifacts.portfolio`
- `expected_generated_artifacts.benchmarks`
- optional extra sections such as `expected_generated_artifacts.realdata`
- `ignored_artifact_paths`
- `limitations`
- `open_roadmap_items`

This file describes expected local artifacts. It is not a generated benchmark
result and does not make performance claims.

## Portfolio Metrics

Portfolio metrics must include:

- `input`
- `preprocessing`
- `features`
- `registration`
- `segmentation`
- `runtime`

The schema checks basic non-negative counts, registration RMSE, fitness range,
segmentation noise ratio, and total runtime.

## Benchmark JSON

Benchmark JSON must include:

- `benchmark`
- `metadata`
- `conclusion`
- `rows`
- `metadata.parameters`
- `metadata.repeat`
- `metadata.memory`

Repeat and memory metadata are local-run diagnostics. They are useful for
reviewing reproducibility but are not portable performance promises.

The scale benchmark uses the same benchmark JSON shape and can be checked with:

```bash
python scripts/check_artifact_schema.py \
  --benchmark-json outputs/scale_benchmark/scale_benchmark.json
```

## Limits

These schema checks are intentionally small. They catch missing or malformed
release, portfolio, and benchmark evidence, but they do not validate real KITTI
benchmark quality or implement a full nonlinear GICP optimizer.
