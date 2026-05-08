# Benchmark Summary

| Suite | Cases | Conclusion |
|---|---:|---|
| kdtree | 3 | Custom KDTree demonstrates pruning logic; SciPy/sklearn are expected to win raw throughput at large N, while voxel hash is competitive for fixed-radius locality. |
| icp | 4 | Point-to-point ICP is accurate near the solution but remains sensitive to initialization, noise, and outliers. |
| ransac | 3 | RANSAC remains stable with moderate outliers until clean minimal samples become unlikely. |
| registration | 4 | Feature-based global registration expands ICP's basin of convergence by estimating a coarse pose first. |
| gicp | 4 | GICP uses local covariance structure to weight correspondences; it is more expensive per iteration than point-to-point ICP but exposes surface-aware residuals. |
| segmentation | 2 | Euclidean clustering runtime scales with radius-neighborhood queries and benefits directly from spatial indexing. |
