# Interview Notes

## What problem does this project solve?

It turns raw point clouds into measurable geometric outputs: registration
transforms, fitted primitives, clusters, bounding boxes, filtered point sets,
and benchmark reports.

## Why does ICP need a good initial pose?

ICP uses nearest neighbors as correspondences. If the initial pose is far away,
nearest neighbors can be geometrically wrong, so SVD optimizes the wrong pairs
and converges to a local minimum.

## Why is RANSAC robust to outliers?

RANSAC scores models by inlier count under a residual threshold. Outliers can
damage individual samples, but repeated random sampling eventually finds a
mostly inlier sample when the inlier ratio is sufficient.

## What is KDTree complexity?

For balanced trees, build time is roughly `O(N log N)`. Average nearest-neighbor
query time is often close to `O(log N)` in low dimensions, but degrades in high
dimensions.

## How does PCA produce an OBB?

PCA computes covariance eigenvectors. The point cloud is projected into this
orthonormal basis, an AABB is computed in PCA coordinates, and the box corners
are transformed back to world coordinates.

## Point-to-point vs point-to-plane ICP

Point-to-point minimizes Euclidean distance between matched points. Point-to-
plane minimizes distance along the target normal, which often converges faster
for smooth surfaces because it models local surface geometry.

## Why is global registration more robust than ICP alone?

FPFH descriptors create geometry-based candidate correspondences before local
optimization. RANSAC filters those correspondences into a coarse transform,
giving ICP a much better starting point.

## How should benchmark results be interpreted?

Benchmarks explain tradeoffs: readable educational implementations are useful,
but specialized libraries may be faster. Robust pipelines may cost more runtime
but survive noise, outliers, and poor initial poses.
