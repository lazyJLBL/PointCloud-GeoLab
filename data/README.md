# Demo Data

Run:

```bash
python examples/generate_demo_data.py
```

The generated files are small synthetic point clouds intended for repeatable demos and tests:

- `bunny_source.ply` and `bunny_target.ply`: misaligned object-like point clouds for ICP.
- `room.pcd` and `room.xyz`: room-like scene with floor, wall, tabletop, and outliers for RANSAC.
- `object.ply`: rotated box-like point cloud for AABB, OBB, and PCA analysis.

No external dataset license is required because the data is generated procedurally.

