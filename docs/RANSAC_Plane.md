# RANSAC Plane Fitting

RANSAC plane fitting extracts a dominant plane from noisy point clouds with outliers.

## What Problem It Solves

Real point clouds often contain sensor noise, moving objects, clutter, and mixed surfaces. A least-squares plane fit can be strongly affected by outliers. RANSAC is robust because it repeatedly proposes candidate models from minimal samples and chooses the model supported by the largest inlier set.

## Algorithm

1. Randomly sample 3 points.
2. Estimate a plane using the cross product of two sample edges.
3. Skip degenerate samples where the 3 points are collinear.
4. Normalize the plane model `[a, b, c, d]`.
5. Compute distances from all points to the plane.
6. Count points with distance below the threshold as inliers.
7. Keep the plane with the largest inlier set.

## Plane Equation

```text
ax + by + cz + d = 0
```

Three non-collinear points define a plane because two independent vectors on the plane can be crossed to produce the plane normal.

## Point-to-Plane Distance

```text
distance = |ax + by + cz + d| / sqrt(a^2 + b^2 + c^2)
```

The implementation normalizes the plane normal, but the full formula is still used to make the distance computation robust.

## Threshold

The threshold controls how far a point can be from the model and still count as an inlier.

- Too small: noisy true plane points are rejected.
- Too large: unrelated nearby structures may be included.

The right value depends on sensor noise, scene scale, and preprocessing.

## Max Iterations

More iterations increase the probability of sampling an all-inlier triplet, especially when the inlier ratio is low. They also increase runtime because every candidate plane is scored against all points.

## RANSAC vs Least Squares

Least squares uses all points and minimizes aggregate residual, so outliers can pull the plane away from the true surface. RANSAC first identifies a robust inlier set, then can optionally be followed by least-squares refinement on those inliers.

## Point Cloud Applications

RANSAC plane fitting is commonly used for:

- Ground extraction in robotics and autonomous driving.
- Wall extraction in indoor mapping.
- Tabletop segmentation in manipulation scenes.
- Removing dominant planes before object clustering.

## Outputs

- `plane_model`: normalized `[a, b, c, d]`.
- `inliers`: indices of points on the plane.
- `outliers`: indices of remaining points.
- `inlier_ratio`: fraction of points supporting the plane.

