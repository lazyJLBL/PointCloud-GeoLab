# Portfolio Check Report

## Passed Commands

- `D:\anaconda-\python.exe examples/generate_demo_data.py`
- `D:\anaconda-\python.exe examples/global_registration_demo.py`
- `D:\anaconda-\python.exe examples/primitive_fitting_demo.py`
- `D:\anaconda-\python.exe examples/segmentation_demo.py`
- `D:\anaconda-\python.exe examples/benchmark_demo.py`
- `D:\anaconda-\python.exe examples/visualization_demo.py`
- `D:\anaconda-\python.exe examples/gallery_demo.py`
- `D:\anaconda-\python.exe -m pointcloud_geolab benchmark --suite all --quick --output D:\PointCloud-GeoLab\outputs\benchmarks`
- `D:\anaconda-\python.exe -m pointcloud_geolab benchmark --suite kdtree --quick --output D:\PointCloud-GeoLab\outputs\benchmarks\kdtree_smoke`
- `D:\anaconda-\python.exe -m pointcloud_geolab extract-primitives --input data/synthetic_scene.ply --models plane sphere cylinder --threshold 0.04 --max-models 3 --min-inliers 20 --output D:\PointCloud-GeoLab\outputs\primitives\verify_primitives.ply`

## Failed Commands

- None

## Generated Artifacts

- `outputs\benchmarks\all_benchmark.csv`
- `outputs\benchmarks\all_benchmark.json`
- `outputs\benchmarks\all_benchmark.md`
- `outputs\benchmarks\all_benchmark.png`
- `outputs\benchmarks\benchmark_summary.json`
- `outputs\benchmarks\benchmark_summary.md`
- `outputs\benchmarks\gicp\gicp_benchmark.csv`
- `outputs\benchmarks\gicp\gicp_benchmark.json`
- `outputs\benchmarks\gicp\gicp_benchmark.md`
- `outputs\benchmarks\gicp\gicp_benchmark.png`
- `outputs\benchmarks\gicp\metrics.json`
- `outputs\benchmarks\icp\icp_benchmark.csv`
- `outputs\benchmarks\icp\icp_benchmark.json`
- `outputs\benchmarks\icp\icp_benchmark.md`
- `outputs\benchmarks\icp\icp_benchmark.png`
- `outputs\benchmarks\icp\metrics.json`
- `outputs\benchmarks\kdtree\kdtree_benchmark.csv`
- `outputs\benchmarks\kdtree\kdtree_benchmark.json`
- `outputs\benchmarks\kdtree\kdtree_benchmark.md`
- `outputs\benchmarks\kdtree\kdtree_benchmark.png`
- `outputs\benchmarks\kdtree\metrics.json`
- `outputs\benchmarks\kdtree_smoke\kdtree_benchmark.csv`
- `outputs\benchmarks\kdtree_smoke\kdtree_benchmark.json`
- `outputs\benchmarks\kdtree_smoke\kdtree_benchmark.md`
- `outputs\benchmarks\kdtree_smoke\kdtree_benchmark.png`
- `outputs\benchmarks\kdtree_smoke\metrics.json`
- `outputs\benchmarks\metrics.json`
- `outputs\benchmarks\ransac\metrics.json`
- `outputs\benchmarks\ransac\ransac_benchmark.csv`
- `outputs\benchmarks\ransac\ransac_benchmark.json`
- `outputs\benchmarks\ransac\ransac_benchmark.md`
- `outputs\benchmarks\ransac\ransac_benchmark.png`
- `outputs\benchmarks\registration\metrics.json`
- `outputs\benchmarks\registration\registration_benchmark.csv`
- `outputs\benchmarks\registration\registration_benchmark.json`
- `outputs\benchmarks\registration\registration_benchmark.md`
- `outputs\benchmarks\registration\registration_benchmark.png`
- `outputs\benchmarks\segmentation\metrics.json`
- `outputs\benchmarks\segmentation\segmentation_benchmark.csv`
- `outputs\benchmarks\segmentation\segmentation_benchmark.json`
- `outputs\benchmarks\segmentation\segmentation_benchmark.md`
- `outputs\benchmarks\segmentation\segmentation_benchmark.png`
- `outputs\benchmarks_smoke\kdtree_benchmark.csv`
- `outputs\benchmarks_smoke\kdtree_benchmark.json`
- `outputs\benchmarks_smoke\kdtree_benchmark.md`
- `outputs\benchmarks_smoke\kdtree_benchmark.png`
- `outputs\benchmarks_smoke\metrics.json`
- `outputs\gallery\README_gallery.md`
- `outputs\gallery\cluster_report.md`
- `outputs\gallery\ground_segmentation\metrics.json`
- `outputs\gallery\kdtree_benchmark.png`
- `outputs\gallery\kdtree_benchmark\kdtree_benchmark.csv`
- `outputs\gallery\kdtree_benchmark\kdtree_benchmark.json`
- `outputs\gallery\kdtree_benchmark\kdtree_benchmark.md`
- `outputs\gallery\kdtree_benchmark\kdtree_benchmark.png`
- `outputs\gallery\kdtree_benchmark\metrics.json`
- `outputs\gallery\multiscale_icp_curve.png`
- `outputs\gallery\primitive_extraction\metrics.json`
- `outputs\gallery\primitive_extraction\primitive_models.json`
- `outputs\gallery\primitive_extraction_scene.html`
- `outputs\gallery\primitive_extraction_scene.ply`
- `outputs\gallery\primitive_scene.ply`
- `outputs\gallery\ransac_benchmark\metrics.json`
- `outputs\gallery\ransac_benchmark\ransac_benchmark.csv`
- `outputs\gallery\ransac_benchmark\ransac_benchmark.json`
- `outputs\gallery\ransac_benchmark\ransac_benchmark.md`
- `outputs\gallery\ransac_benchmark\ransac_benchmark.png`
- `outputs\gallery\ransac_outlier_benchmark.png`
- `outputs\gallery\registration_before_after.png`
- `outputs\gallery\robust_icp_outlier_comparison.png`
- `outputs\gallery\segmentation_ground_objects.html`
- `outputs\gallery\segmentation_ground_objects.ply`
- `outputs\gallery\segmentation_ground_objects_colored.ply`
- `outputs\gicp_demo\gicp_before_after.png`
- `outputs\metrics.json`
- `outputs\portfolio_check_report.md`
- `outputs\portfolio_demo\artifacts\processed_cloud.ply`
- `outputs\portfolio_demo\artifacts\transformation.json`
- `outputs\portfolio_demo\figures\bounding_box_or_normals.png`
- `outputs\portfolio_demo\figures\downsampled.png`
- ... 34 more

## Missing README Artifacts

- None

## Next Actions

- Re-run failed commands after dependency installation if optional extras are missing.
- Regenerate gallery assets before updating README image references.
