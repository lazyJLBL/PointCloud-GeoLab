# Case Study: KITTI Ground Removal and Clustering

This case study demonstrates the LiDAR segmentation path without committing
large KITTI files.

## Data Preparation

Download a KITTI Raw Velodyne frame and place it at:

```text
data/external/kitti/velodyne/000000.bin
```

The file must contain repeated `float32` tuples:

```text
x y z intensity
```

Validate the local layout:

```bash
python scripts/prepare_datasets.py validate
```

## Run

```bash
python examples/kitti_lidar_segmentation.py \
  --frame data/external/kitti/velodyne/000000.bin \
  --output-dir outputs/kitti_segmentation \
  --eps 0.65 \
  --min-points 12
```

If the frame is missing, the script exits with code `2` and prints the expected
path plus a pointer to `docs/datasets.md`.

## Pipeline

1. Load KITTI Velodyne `.bin` as XYZ points.
2. Optionally cap the point count for laptop-friendly runs.
3. Fit a dominant ground plane with RANSAC and a normal-angle constraint.
4. Cluster non-ground points with Euclidean clustering.
5. Compute per-cluster centroid, AABB, OBB, and volume.
6. Export a colored PLY, BEV images, metrics, and a Markdown object report.

## Outputs

- `outputs/kitti_segmentation/kitti_bev.png`
- `outputs/kitti_segmentation/kitti_clusters.png`
- `outputs/kitti_segmentation/kitti_clusters.ply`
- `outputs/kitti_segmentation/cluster_report.md`
- `outputs/kitti_segmentation/metrics.json`

## Interpretation

`cluster_report.md` is the primary reviewer artifact. It lists ground point
count, non-ground noise count, object cluster count, centroid, extent, and
volume. `kitti_clusters.png` is a quick visual check that ground and object
labels are not collapsed into one component.

## Limitations

- Fixed-radius clustering is sensitive to range-dependent LiDAR density.
- Slopes, ramps, and curbs can violate the dominant-plane ground assumption.
- Touching objects may merge into one cluster.
- This script processes one frame in memory; long sequences need streaming or
  chunked processing.
