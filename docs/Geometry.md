# 3D Geometry Processing

The geometry module provides compact building blocks used in point cloud analysis.

## AABB

Axis-Aligned Bounding Box:

```text
min_bound = min(points)
max_bound = max(points)
extent = max_bound - min_bound
center = (min_bound + max_bound) / 2
```

An AABB is fast to compute and aligned with the world coordinate axes. It is useful for rough spatial range checks, broad-phase collision detection, and quick scene statistics.

## OBB

An Oriented Bounding Box can rotate with the object. It is usually tighter than an AABB for objects that are not axis-aligned.

PointCloud-GeoLab computes OBBs with PCA:

1. Compute the point cloud center.
2. Center the points.
3. Compute the covariance matrix.
4. Compute eigenvalues and eigenvectors.
5. Use eigenvectors as local axes.
6. Project points into the PCA frame.
7. Compute local min/max bounds and transform corners back to world coordinates.

## AABB vs OBB

- AABB is faster and simpler but can be loose after rotation.
- OBB is tighter for oriented objects but requires PCA and can be unstable for symmetric shapes.

## PCA Interpretation

PCA eigenvectors are the principal directions of the point distribution. Eigenvalues measure variance along those directions.

- One dominant eigenvalue: line-like point cloud.
- Two dominant eigenvalues: plane-like point cloud.
- Three similar eigenvalues: volume-like point cloud.

This is useful for shape analysis, object orientation, and local geometric feature estimation.

## Distance Utilities

Point-to-plane distance:

```text
distance = |ax + by + cz + d| / sqrt(a^2 + b^2 + c^2)
```

Point-to-line distance:

```text
distance = ||(p - a) x u||
```

where `a` is a point on the line and `u` is a unit line direction.

## Applications

- Collision detection and spatial filtering with bounding boxes.
- Object dimension estimation in 3D detection.
- Principal direction estimation for grasping and pose analysis.
- Plane and line distances for segmentation, fitting, and geometric validation.

