# Interview Notes

## 1. Why is KDTree faster than brute force? When is it not?

Brute force computes all `N` distances per query: `O(ND)`. KDTree recursively
splits space and prunes a subtree when the splitting-plane distance is already
larger than the best distance. In low dimensions this often behaves close to
`O(log N)`.

It is not always faster. High-dimensional data weakens pruning because most
points are similarly far away. Very large radius queries, duplicate-heavy data,
or unbalanced distributions can also approach `O(N)`.

Implementation: `pointcloud_geolab/kdtree/kdtree.py`.

## 2. Why does ICP need an initial pose?

ICP assumes nearest neighbors are meaningful correspondences. With a bad initial
pose, nearest neighbors can be wrong, so the SVD step solves the wrong
least-squares problem and converges to a local minimum. Feature RANSAC or
odometry/IMU priors are typical ways to provide the initial basin.

## 3. Point-to-point vs point-to-plane ICP

Point-to-point minimizes:

```text
sum ||R p_i + t - q_i||^2
```

Point-to-plane minimizes distance along the target normal:

```text
sum (n_i^T (R p_i + t - q_i))^2
```

For small updates, `R p ~= p + w x p`, giving:

```text
n^T ((w x p) + t + p - q) = 0
```

Point-to-plane usually converges faster on smooth surfaces, but it depends on
stable normals and can be ill-conditioned on degenerate geometry.

## 4. Why is multi-scale ICP more stable?

Large voxels remove fine local minima and make the coarse geometry easier to
align. Smaller voxels then refine details. The trade-off is that coarse levels
can lose small structures, so the voxel schedule should decrease gradually.

Implementation: `multiscale_icp` in `pointcloud_geolab/registration/icp.py`.

## 5. Why do robust kernels and trimmed ICP help with outliers?

Outliers create large residuals that can dominate least squares. Trimmed ICP
keeps only the closest correspondence ratio, which is useful when overlap is
partial. Huber keeps quadratic loss near zero but becomes linear for large
residuals. Tukey eventually gives very large residuals zero weight.

Failure case: if the trim ratio is too low, valid geometry is discarded; if it
is too high, outliers still influence the transform.

## 6. How does RANSAC success depend on outlier ratio?

If the inlier ratio is `w`, minimal sample size is `s`, and iterations are `N`,
the success probability is:

```text
P = 1 - (1 - w^s)^N
```

As outliers increase, `w^s` drops quickly. Cylinder fitting needs a larger
sample than plane fitting, so it needs more iterations for the same confidence.

## 7. How does PCA compute an OBB?

PCA computes covariance eigenvectors. These eigenvectors define a local
orthonormal frame. Project points into that frame, compute min/max bounds, then
transform box corners back to world coordinates. Axes are unstable when
eigenvalues are close, such as near-spherical clouds.

## 8. What is the core idea of ISS keypoints?

ISS analyzes local covariance eigenvalues. A point is salient when the
neighborhood has strong 3D variation and eigenvalue ratios satisfy:

```text
lambda2 / lambda1 < gamma21
lambda3 / lambda2 < gamma32
```

Non-maximum suppression keeps only local maxima of the saliency score.

## 9. What do FPFH and local descriptors do?

Descriptors turn local geometry into comparable vectors. FPFH is a widely used
baseline implemented through Open3D in this project. The project also includes a
custom descriptor built from eigenvalue ratios, curvature, density, and
scattering to demonstrate the geometry behind feature matching.

## 10. DBSCAN vs Euclidean clustering

Euclidean clustering builds connected components where edges connect points
within a fixed radius. DBSCAN additionally requires enough neighbors to start a
core point and labels sparse points as noise. DBSCAN handles noise better;
Euclidean clustering is simpler and common in robotics object extraction after
ground removal.

## 11. Why is ground removal common in autonomous driving point clouds?

LiDAR scenes are dominated by ground points. Removing the ground reduces point
count and prevents road points from connecting separate objects. It also makes
bounding boxes and object clusters cleaner. Failure cases include slopes,
curbs, ramps, and scenes where the largest plane is not the road.

## 12. Which parts are custom and which use Open3D?

Custom implementations:

- KDTree and Voxel Hash Grid;
- point-to-point, point-to-plane, robust, and multi-scale ICP;
- SVD rigid alignment;
- RANSAC plane/sphere/cylinder fitting and sequential extraction;
- ISS keypoints, local descriptors, descriptor matching, transform RANSAC;
- DBSCAN, Euclidean clustering, region growing, ground removal pipeline.

Open3D-backed parts:

- FPFH descriptors and feature RANSAC baseline;
- optional mesh reconstruction demos: Poisson, Ball Pivoting, Alpha Shape;
- optional visualization/backend I/O when installed.

This split is intentional: custom code proves algorithm understanding, while
Open3D baselines show awareness of production-grade tooling.
