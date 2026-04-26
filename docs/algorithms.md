# Algorithms

PointCloud-GeoLab is geometry-first: the project keeps the core numerical
steps visible instead of hiding all behavior behind library calls.

## KDTree

The custom KDTree recursively splits points by median coordinate and prunes
subtrees when the splitting-plane distance is larger than the current best
distance. It supports nearest-neighbor, kNN, and radius search.

## ICP

Point-to-point ICP alternates between nearest-neighbor correspondence search
and SVD rigid transform estimation. Given matched points `P` and `Q`, it solves
`R, t` from the covariance matrix `H = (P - mean(P))^T (Q - mean(Q))`.

Point-to-plane ICP uses the linearized residual:

```text
n^T ((w x p) + t + p - q) = 0
```

where `w` is a small rotation vector, `t` is translation, and `n` is the target
normal.

## RANSAC

RANSAC repeatedly samples minimal point sets, builds a candidate model, scores
all points by geometric residual, and keeps the model with the largest inlier
set. This makes primitive fitting robust under outliers.

## Primitive Fitting

- Plane residual: `|n^T x + d|`
- Sphere residual: `abs(||x - c|| - r)`
- Cylinder residual: `abs(distance_to_axis(x) - r)`

## Segmentation

DBSCAN and Euclidean clustering use radius neighborhoods from the custom KDTree.
Region growing adds a normal-angle condition so adjacent points only merge when
their local surface orientation is compatible.
