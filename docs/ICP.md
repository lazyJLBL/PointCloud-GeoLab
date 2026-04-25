# ICP Registration

PointCloud-GeoLab implements classic point-to-point ICP from scratch. ICP solves the local registration problem: given a source point cloud and a target point cloud with an approximate initial overlap, estimate the rigid transform that aligns the source to the target.

## Problem

Given source points `p_i` and target correspondence points `q_i`, point-to-point ICP minimizes:

```text
min Σ ||R p_i + t - q_i||²
```

where `R` is a 3x3 rotation matrix and `t` is a 3D translation vector.

## Pipeline

1. Build a custom KD-Tree on the target point cloud.
2. For each transformed source point, query its nearest target point.
3. Estimate the best rigid transform between matched point pairs.
4. Update the full source point cloud.
5. Repeat until RMSE change is below tolerance or the iteration limit is reached.

## Why KD-Tree Matters

The bottleneck of ICP is correspondence search. In every iteration, each source point needs a nearest target point.

Brute-force matching costs roughly:

```text
O(NM)
```

for `N` source points and `M` target points. A KD-Tree built once on the target cloud reduces average nearest-neighbor query time toward:

```text
O(log M)
```

for balanced data, which makes iterative registration practical for larger point clouds.

## SVD Solver

Given matched pairs `P = {p_i}` and `Q = {q_i}`:

```text
p_mean = mean(P)
q_mean = mean(Q)
P_centered = P - p_mean
Q_centered = Q - q_mean
H = P_centered^T Q_centered
H = U S V^T
R = V U^T
t = q_mean - R p_mean
```

The SVD step gives the least-squares rigid alignment between two matched point sets.

## Reflection Correction

Numerical SVD can produce a matrix with `det(R) < 0`, which is a reflection rather than a valid 3D rotation. The implementation flips the final singular vector and recomputes `R` so that:

```text
det(R) = 1
```

This keeps the result in the valid rotation group `SO(3)`.

## RMSE

RMSE measures correspondence residuals:

```text
RMSE = sqrt(mean(||R p_i + t - q_i||²))
```

The implementation stores `rmse_history`, `initial_rmse`, and `final_rmse` to show convergence.

## Limitations

- ICP is a local optimizer and can converge to a local minimum.
- It is sensitive to the initial pose when the clouds are far apart.
- Repeated structures or weak overlap can produce wrong correspondences.
- Point-to-point ICP works well for similar sampled surfaces but may converge slower on smooth surfaces.
- Too small a `max_correspondence_distance` can leave fewer than 3 correspondences and stop the solver.

## Point-to-Point vs Point-to-Plane

Point-to-point ICP minimizes Euclidean distances between corresponding points:

```text
min Σ ||R p_i + t - q_i||²
```

Point-to-plane ICP minimizes the distance from transformed source points to target tangent planes:

```text
min Σ ((R p_i + t - q_i) · n_i)²
```

Point-to-plane ICP often converges faster on dense surface scans, but it requires reliable target normals.

## Outputs

The ICP result contains:

- `rotation`: 3x3 rotation matrix.
- `translation`: 3D translation vector.
- `transformation`: 4x4 homogeneous matrix.
- `aligned_points`: transformed source points.
- `rmse_history`: per-iteration convergence curve.
- `initial_rmse`, `final_rmse`, `fitness`, `iterations`, and `converged`.

