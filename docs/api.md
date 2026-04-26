# API Overview

Primary task APIs live in `pointcloud_geolab.api` and return `TaskResult`, a
JSON-friendly envelope with:

- `success`
- `metrics`
- `artifacts`
- `parameters`
- `data`
- `error`

Examples:

```python
from pointcloud_geolab.api import run_global_registration, run_primitive_fitting, run_segmentation

registration = run_global_registration("source.ply", "target.ply", voxel_size=0.05)
primitive = run_primitive_fitting("scene.ply", model="sphere", threshold=0.02)
segmentation = run_segmentation("scene.ply", method="dbscan", eps=0.05, min_points=20)
```

Lower-level modules remain directly importable:

- `pointcloud_geolab.registration.global_registration`
- `pointcloud_geolab.geometry.primitive_fitting`
- `pointcloud_geolab.segmentation.clustering`
- `pointcloud_geolab.segmentation.region_growing`
- `pointcloud_geolab.visualization.export`
