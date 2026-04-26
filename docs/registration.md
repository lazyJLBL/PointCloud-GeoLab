# Registration

The registration stack demonstrates why ICP needs a good initial pose and how
feature-based global registration can provide one.

```text
Downsample -> Normal Estimation -> FPFH -> RANSAC -> ICP
```

1. Voxel downsampling reduces noise and runtime.
2. Normal estimation provides local surface orientation.
3. FPFH descriptors summarize local geometry.
4. Feature RANSAC finds a coarse rigid transform robustly.
5. ICP refines the transform with point-to-point or point-to-plane residuals.

Run:

```bash
python -m pointcloud_geolab register \
  --source data/bunny_source.ply \
  --target data/bunny_target.ply \
  --method fpfh_ransac_icp \
  --voxel-size 0.05 \
  --output outputs/registration/aligned_source.ply \
  --save-transform outputs/registration/transform.txt \
  --save-results
```
