# Case Study: Stanford Bunny Registration

This case study documents the Stanford Bunny workflow without committing large
Stanford dataset files.

## Data Preparation

Download the Bunny data from the Stanford 3D Scanning Repository and keep it
outside git:

```text
data/external/stanford/bunny/
  bunny.ply
```

Create a deterministic source/target pair:

```bash
python scripts/prepare_datasets.py make-bunny-pair \
  --input data/external/stanford/bunny/bunny.ply \
  --output-dir data/external/stanford/bunny_pair
```

The script writes `bunny_target.ply` and `bunny_source.ply`.

## Run

```bash
python examples/real_bunny_registration.py \
  --data-dir data/external/stanford/bunny_pair \
  --method icp \
  --output-dir outputs/real_bunny
```

Optional feature paths:

```bash
python examples/real_bunny_registration.py \
  --data-dir data/external/stanford/bunny_pair \
  --method iss_descriptor_ransac_icp \
  --threshold 0.15 \
  --output-dir outputs/real_bunny
```

The ISS descriptor path is experimental. If it fails, the script reports the
failure instead of silently treating a fallback as descriptor registration
success.

## Outputs

- `outputs/real_bunny/aligned_bunny.ply`
- `outputs/real_bunny/bunny_before.png`
- `outputs/real_bunny/bunny_after.png`
- `outputs/real_bunny/bunny_registration.png`
- `outputs/real_bunny/metrics.json`

For feature-based methods, `transform.txt` is also written by the task API.

## Boundaries

- ICP aligns local nearest-neighbor correspondences and is sensitive to initial pose.
- Fitness and RMSE depend on thresholds, point density, and overlap.
- This is a documented workflow for local real data, not a bundled real-data benchmark.
