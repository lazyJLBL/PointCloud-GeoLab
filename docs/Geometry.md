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

## PCA

PCA computes the point cloud center, covariance matrix, eigenvalues, and eigenvectors. The eigenvectors are sorted by descending eigenvalue and represent principal directions.

## OBB

The oriented bounding box is computed from PCA:

1. Compute PCA axes.
2. Project points into the PCA coordinate frame.
3. Compute local min/max bounds.
4. Transform the local box center and corners back into world coordinates.

## Distance Utilities

- `point_to_plane_distances(points, plane_model)`
- `point_to_line_distances(points, line_point, line_direction)`

