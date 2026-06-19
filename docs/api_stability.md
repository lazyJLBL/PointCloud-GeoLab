# API Stability

The stable public API is intentionally small. Use `pointcloud_geolab.api` or
the package-level exports from `pointcloud_geolab` for reviewer exercises.

## Stable

- `TaskResult`
- `run_icp`
- `run_robust_icp`
- `run_multiscale_icp`
- `run_preprocessing`
- `run_plane_segmentation`
- `run_geometry_analysis`
- `run_segmentation`
- `run_ground_object_segmentation`
- `run_primitive_fitting`
- `run_extract_primitives`
- `run_benchmark`
- `run_portfolio_verification`

`TaskResult.to_dict()` and `TaskResult.to_json()` keep these fields stable:

- `task`
- `success`
- `metrics`
- `artifacts`
- `parameters`
- `data`
- `error`
- `path`

Error results must include `task`, `success`, `error`, `parameters`, and
`path` so CLI JSON output can be checked by scripts. `Path` values and NumPy
scalars/arrays are serialized into JSON-friendly values. Non-finite numeric
values such as `NaN` and `Inf` become JSON `null` rather than non-standard JSON
tokens.

## Experimental

- GICP-style covariance-weighted ICP
- feature registration and fallback diagnostics
- large-scene scale benchmark interpretation
- user-provided KITTI-like case study workflow

The GICP-style path is not a full nonlinear GICP optimizer and is not part of a
strong stable API promise for v1.0.0.

The real-data workflow is intentionally documented as an example plus verifier
script, not a stable public API entry point. Reviewers should use
`examples/kitti_lidar_segmentation.py` and
`scripts/verify_realdata_workflow.py`; missing real data should produce a
friendly skip or error envelope, not a traceback.

## Optional

Open3D, Plotly, laspy, PyTorch, SciPy, scikit-learn, and pandas paths are
optional comparisons or demos. Missing optional dependencies should report a
clear unavailable or skip message rather than failing core tests.
