# External Datasets

PointCloud-GeoLab keeps large datasets out of git. The repository contains
small synthetic demo data for CI, while real data should live under
`data/external/` or another local path passed to the examples.

## Supported Sources

| Dataset | Official Source | Local Use | Notes |
|---|---|---|---|
| Stanford Bunny / Armadillo | [Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/) | registration, reconstruction, OBB/PCA | Bunny range data and reconstruction are small enough for laptop demos; Armadillo is larger. |
| KITTI Velodyne | [KITTI Raw Data](https://www.cvlibs.net/datasets/kitti/raw_data.php) | LiDAR ground removal and clustering | KITTI stores Velodyne frames as repeated float32 `x y z intensity` tuples in `.bin`. |
| ModelNet10/40 | [Princeton ModelNet download page](https://modelnet.cs.princeton.edu/download.html) | primitive/PCA demos and optional ML | ModelNet meshes are OFF files; convert a few samples to point clouds before running examples. |

## Recommended Layout

```text
data/external/
  stanford/
    bunny/
      bunny.ply
    bunny_pair/
      bunny_source.ply
      bunny_target.ply
    armadillo/
      Armadillo.ply
  kitti/
    velodyne/
      000000.bin
  modelnet_small/
    chair.off
    sample.xyz
  manifest.json
```

Do not commit these files. Commit only scripts, docs, and tiny deterministic
fixtures.

## Preparation Commands

Print the expected layout:

```bash
python scripts/prepare_datasets.py summary
```

Validate what is present locally:

```bash
python scripts/prepare_datasets.py validate --write-manifest data/external/manifest.json
```

Convert one KITTI Velodyne frame to PLY:

```bash
python scripts/prepare_datasets.py convert-kitti-bin \
  --input data/external/kitti/velodyne/000000.bin \
  --output data/external/kitti/velodyne/000000.ply
```

Sample a ModelNet OFF mesh to an XYZ point cloud:

```bash
python scripts/prepare_datasets.py convert-modelnet-off \
  --input data/external/modelnet_small/chair.off \
  --output data/external/modelnet_small/sample.xyz \
  --points 2048
```

Create a deterministic Bunny source/target pair from one PLY:

```bash
python scripts/prepare_datasets.py make-bunny-pair \
  --input data/external/stanford/bunny/bunny.ply \
  --output-dir data/external/stanford/bunny_pair
```

Write checksums for reproducibility:

```bash
python scripts/prepare_datasets.py checksum \
  --input data/external \
  --output data/external/manifest.json
```

## Real Data Examples

```bash
python examples/real_bunny_registration.py \
  --data-dir data/external/stanford/bunny_pair \
  --output-dir outputs/real_bunny

python examples/kitti_lidar_segmentation.py \
  --frame data/external/kitti/velodyne/000000.bin \
  --output-dir outputs/kitti_segmentation

python examples/modelnet_primitive_demo.py \
  --input data/external/modelnet_small/sample.xyz \
  --output-dir outputs/modelnet_demo
```

Each example exits with code `2` and a specific preparation hint if the expected
data is missing.

## Verification

Use checksums instead of committing large files:

```bash
python scripts/prepare_datasets.py checksum \
  --input data/external \
  --output data/external/manifest.json
```

The manifest records relative path, byte size, and SHA256 digest. This makes it
possible to reproduce benchmark inputs while keeping the repository small.
