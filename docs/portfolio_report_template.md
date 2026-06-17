# Portfolio Review Template

Use this checklist when presenting PointCloud-GeoLab in a resume review or
technical interview.

## 1. Start With the One-Command Demo

Run:

```bash
python examples/generate_demo_data.py --output examples/demo_data
python -m pointcloud_geolab pipeline --input examples/demo_data --output outputs/portfolio_demo
```

Open:

- `outputs/portfolio_demo/report.md`
- `outputs/portfolio_demo/metrics.json`
- `outputs/portfolio_demo/figures/registration_before_after.png`
- `outputs/portfolio_demo/figures/segmentation_result.png`

Review question: can the project turn point-cloud input into a reproducible
report with metrics, figures, and artifacts?

## 2. Check Algorithm Evidence

Run:

```bash
python -m pytest tests/test_kdtree.py tests/test_spatial_index.py tests/test_algorithm_edge_cases.py
```

Look for:

- brute-force checks for nearest-neighbor queries;
- known-transform recovery for ICP/GICP;
- outlier and low-overlap behavior;
- RANSAC primitive recovery under seeded outliers;
- PCA/OBB invariance under rigid rotation.

Review question: are tests validating geometry behavior rather than only
checking that functions return something?

## 3. Check Benchmark Evidence

Run:

```bash
python -m pointcloud_geolab benchmark --suite all --quick --output outputs/benchmarks
python scripts/verify_benchmarks.py --output-dir outputs/benchmarks
```

Open:

- `outputs/benchmarks/benchmark_summary.md`
- `outputs/benchmarks/kdtree/kdtree_benchmark.md`
- `outputs/benchmarks/ransac/ransac_benchmark.md`

Review question: do the benchmarks compare against simple baselines and record
parameters, scale, seed, and environment?

## 4. Check Real-Data Workflows

Run without data first:

```bash
python examples/real_bunny_registration.py
python examples/kitti_lidar_segmentation.py
```

Expected behavior: each script exits with a clear preparation hint instead of a
traceback.

Then prepare data following `docs/datasets.md` and rerun:

- Stanford Bunny registration should emit before/after imagery, metrics, and an
  aligned PLY.
- KITTI segmentation should emit a colored cluster PLY, metrics, and a cluster
  Markdown report.

Review question: does the project handle missing large datasets honestly?

## 5. Check Presentation Assets

Run:

```bash
python examples/gallery_demo.py
python scripts/verify_portfolio.py --quick
```

Expected gallery files:

- `outputs/gallery/registration_before_after.png`
- `outputs/gallery/icp_convergence_curve.png`
- `outputs/gallery/kdtree_benchmark.png`
- `outputs/gallery/ransac_outlier_benchmark.png`
- `outputs/gallery/segmentation_result.png`
- `outputs/gallery/primitive_extraction.html`

Review question: are the README/demo claims backed by generated files?

## 6. Suggested Interview Talking Points

- Why KDTree pruning helps and when it degrades.
- Why ICP is local and how feature RANSAC expands the convergence basin.
- How robust/trimmed ICP behaves under source-only outliers and partial overlap.
- Why RANSAC iteration count grows quickly with outlier ratio.
- Why PCA/OBB axes become unstable near equal eigenvalues.
- Why fixed-radius clustering is sensitive to LiDAR density.

See `docs/interview_notes.md` for compact answers.
