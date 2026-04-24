# ICP Registration

PointCloud-GeoLab implements classic point-to-point ICP from scratch.

## Pipeline

1. Build a custom KD-Tree on the target point cloud.
2. For each transformed source point, query its nearest target point.
3. Estimate the best rigid transform between matched point pairs.
4. Update the source point cloud.
5. Repeat until RMSE change is below tolerance or the iteration limit is reached.

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

If `det(R) < 0`, the last row of `V^T` is multiplied by `-1` before recomputing `R`.

## Outputs

The ICP result contains:

- `rotation`: 3x3 rotation matrix.
- `translation`: 3D translation vector.
- `transformation`: 4x4 homogeneous matrix.
- `aligned_points`: transformed source points.
- `rmse_history`: per-iteration convergence curve.
- `initial_rmse`, `final_rmse`, `fitness`, `iterations`, and `converged`.

