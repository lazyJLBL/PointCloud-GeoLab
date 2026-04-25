# Demo Data

Run:

```bash
python examples/generate_demo_data.py
```

The generated files are small synthetic point clouds intended for repeatable demos and tests:

- `bunny_source.ply` and `bunny_target.ply`: misaligned object-like point clouds for ICP.
- `room.pcd` and `room.xyz`: room-like scene with floor, wall, tabletop, and outliers for RANSAC.
- `object.ply`: rotated box-like point cloud for AABB, OBB, and PCA analysis.

No external dataset license is required because the included demo data is generated procedurally.

## Real-World Experiments

Real point cloud datasets can be large, so this repository keeps only small synthetic data for reproducible tests and demos. For real-world experiments, place your own licensed `.ply`, `.pcd`, or `.xyz` files under `data/` and run the same CLI pipeline:

```bash
python main.py --mode icp --source data/your_source.ply --target data/your_target.ply
python main.py --mode plane --input data/your_scene.pcd
python main.py --mode geometry --input data/your_object.ply
```

If a public dataset is added later, document the source URL, license, and any preprocessing steps in this file.

