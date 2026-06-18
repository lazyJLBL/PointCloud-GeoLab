# External Datasets

PointCloud-GeoLab keeps large datasets out of git. The repository contains
small synthetic demo data for CI, while real data should live under
`data/external/` or another local path passed to the examples.

## Tiny Format Fixtures

The repository commits two tiny synthetic format fixtures under
`tests/fixtures/datasets/`:

- `mini_kitti_like.bin`: four `float32 x y z intensity` records in KITTI-like
  Velodyne layout.
- `mini_modelnet_like.off`: five vertices and four triangular faces in
  ModelNet-like OFF layout.
- `manifest.json`: declared counts, SHA256 checksums, and fixture notes.

Validate them with:

```bash
python scripts/check_dataset_fixtures.py
```

These files are format smoke tests only. They are not real KITTI, ModelNet,
SemanticKITTI, or nuScenes data, and they are not benchmark evidence.

The shared point-cloud readers report path-aware errors for missing files,
unsupported extensions, empty files, bad headers, and bad numeric data. This
keeps tiny fixture failures and user-provided real-data preparation issues easy
to diagnose.

## Supported Sources

### Stanford Bunny / Armadillo

Official source:
[Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/).

Local use: registration, reconstruction, OBB, and PCA demos.

Notes: Bunny range data and reconstruction are small enough for laptop demos.
Armadillo is larger.

### KITTI Velodyne

Official source:
[KITTI Raw Data](https://www.cvlibs.net/datasets/kitti/raw_data.php).

Local use: LiDAR ground removal and clustering through the user-provided
single-frame workflow.

Notes: KITTI stores Velodyne frames as repeated float32 `x y z intensity`
tuples in `.bin`. The repository does not include real KITTI data and does not
claim an official KITTI benchmark.

### ModelNet10/40

Official source:
[Princeton ModelNet download page](https://modelnet.cs.princeton.edu/download.html).

Local use: primitive/PCA demos and optional ML.

Notes: ModelNet meshes are OFF files; convert a few samples to point clouds
before running examples.

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

The KITTI-like workflow writes:

- `outputs/kitti_segmentation/report.md`
- `outputs/kitti_segmentation/report.html`
- `outputs/kitti_segmentation/metrics.json`
- `outputs/kitti_segmentation/kitti_bev.png`
- `outputs/kitti_segmentation/kitti_clusters.png`
- `outputs/kitti_segmentation/kitti_height_histogram.png`

CI uses `python scripts/verify_realdata_workflow.py --dry-run` with a temporary
synthetic frame. That dry-run validates file format and artifact schema only.

## Verification

Use checksums instead of committing large files:

```bash
python scripts/prepare_datasets.py checksum \
  --input data/external \
  --output data/external/manifest.json
```

The manifest records relative path, byte size, and SHA256 digest. This makes it
possible to reproduce benchmark inputs while keeping the repository small.
