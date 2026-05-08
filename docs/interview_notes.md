# Interview Notes

## 中文速答

### 为什么不用纯 Open3D？

这个项目的定位不是替代 Open3D，而是展示我能把核心几何算法讲清楚并工程化。
KDTree、ICP、RANSAC、PCA/OBB、segmentation、ISS 和局部描述子都保留自研实现；
Open3D 只作为 FPFH、reconstruction、IO/visualization 或工业 baseline。这样既能
证明算法理解，也能证明我知道生产工具链应该怎样对照验证。

### KDTree 的剪枝原理是什么？

查询时先访问与 query 同侧的近分支，并维护当前最优距离 `best`。如果 query 到
分裂超平面的距离已经大于 `best`，远分支里所有点都不可能更近，可以剪掉。低维
空间效果好，高维或大半径查询会退化。

### ICP 如何求解刚体变换？

每轮先用 KDTree 建立最近邻对应，再对对应点求 SVD 刚体配准：中心化点集，构造
协方差 `H`，分解 `H = U S V^T`，令 `R = V U^T`，修正反射后计算
`t = q_bar - R p_bar`。

### RANSAC 迭代次数怎么估计？

如果内点比例是 `w`，最小样本大小是 `s`，希望成功概率是 `P`，则：

```text
N >= log(1 - P) / log(1 - w^s)
```

内点比例下降或模型需要更多样本时，迭代次数会快速增加。

### 点云分割有哪些失败场景？

固定半径聚类容易受密度变化影响；地面拟合会在斜坡、台阶、墙面主导场景中失败；
相邻物体接触会被连成一个 cluster；小物体可能被 `min_points` 过滤掉。

### 数据量扩大 100 倍怎么优化？

优先做分块/流式 IO、体素降采样、空间索引复用、批量 KDTree 查询、VoxelHashGrid
固定半径搜索、并行 RANSAC/ICP correspondence search，并把 benchmark 拆成 quick
和 full profile。需要生产部署时再引入 C++/CUDA/PCL/Open3D 后端做热点替换。

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
