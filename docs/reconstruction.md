# Surface Reconstruction

This module is intentionally an Open3D-backed demo, not a from-scratch mesh
reconstruction implementation. The project core remains spatial indexing,
registration, robust estimation, primitive fitting, and segmentation.

## Point Cloud vs Mesh

A point cloud samples surfaces without explicit connectivity. A mesh adds
vertices, edges, and triangles, which enables rendering, simulation, collision,
and surface analysis.

## Why Normals Matter

Poisson and Ball Pivoting depend on oriented local surface normals. Bad normals
create flipped surfaces, holes, or inflated geometry. The wrapper estimates
normals before reconstruction.

## Methods

Poisson Reconstruction:

- good for watertight smooth surfaces;
- can hallucinate surfaces in sparse regions;
- controlled by octree depth.

Ball Pivoting:

- rolls balls over oriented points to form triangles;
- works well for fairly uniform sampling;
- sensitive to radius choice.

Alpha Shape:

- forms a shape controlled by alpha radius;
- useful for small demos and quick tests;
- can fragment with sparse or noisy points.

## Command

```bash
python -m pointcloud_geolab reconstruct \
  --input data/object.ply \
  --method poisson \
  --output outputs/reconstruction/object_mesh.ply
```

## Implementation

File: `pointcloud_geolab/reconstruction/surface.py`.

Tests skip automatically when Open3D is not installed.
