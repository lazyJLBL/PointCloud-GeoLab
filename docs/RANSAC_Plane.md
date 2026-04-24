# RANSAC Plane Fitting

RANSAC plane fitting extracts a dominant plane from noisy point clouds with outliers.

## Algorithm

1. Randomly sample 3 points.
2. Estimate a plane using the cross product of two sample edges.
3. Normalize the plane model `[a, b, c, d]`.
4. Compute distances from all points to the plane.
5. Count points with distance below the threshold as inliers.
6. Keep the plane with the largest inlier set.

## Plane Equation

```text
ax + by + cz + d = 0
```

## Point-to-Plane Distance

```text
distance = |ax + by + cz + d| / sqrt(a^2 + b^2 + c^2)
```

## Outputs

- `plane_model`: normalized `[a, b, c, d]`.
- `inliers`: indices of points on the plane.
- `outliers`: indices of remaining points.
- `inlier_ratio`: fraction of points supporting the plane.

