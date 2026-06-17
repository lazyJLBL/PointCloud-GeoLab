# Demo Walkthrough

This walkthrough regenerates the local portfolio demo from deterministic
synthetic data. It is meant for quick review, not for proving real-data
performance.

## 1. Clone

```bash
git clone https://github.com/lazyJLBL/PointCloud-GeoLab.git
cd PointCloud-GeoLab
```

## 2. Install

```bash
python -m pip install -e ".[dev,vis,bench]"
```

The core demo does not require Open3D or PyTorch.

## 3. Generate Demo Data

```bash
python examples/generate_demo_data.py --output examples/demo_data
```

This creates small deterministic point clouds for review. The generated files
are ignored by Git and should not be committed.

## 4. Run The Portfolio Pipeline

```bash
python -m pointcloud_geolab pipeline \
  --input examples/demo_data \
  --output outputs/portfolio_demo
```

## 5. Open The Report

Open:

```text
outputs/portfolio_demo/report.md
```

The report summarizes preprocessing, registration, segmentation, generated
figures, and current limitations.

## 6. Inspect The Output Bundle

Important files:

- `outputs/portfolio_demo/metrics.json`: machine-readable input,
  preprocessing, registration, segmentation, and runtime metrics.
- `outputs/portfolio_demo/figures/`: PNG figures for the raw cloud,
  downsampled cloud, registration before/after, segmentation, and geometry
  view.
- `outputs/portfolio_demo/artifacts/transformation.json`: the estimated 4x4
  transform plus registration metrics such as RMSE before and after alignment.
- `outputs/portfolio_demo/artifacts/processed_cloud.ply`: processed point cloud
  artifact for inspection in a local viewer.

## Synthetic Data Boundary

The demo data is synthetic and deterministic. It is useful for smoke testing,
reviewing artifact generation, and checking that the pipeline can be rerun.

It should not be described as real-data validation. Real Stanford Bunny and
KITTI workflows are documented separately and require local datasets under
`data/external/`.
