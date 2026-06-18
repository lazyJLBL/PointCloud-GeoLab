# Repository Audit Report Template

Use this template when collecting a manual reviewer snapshot before a patch
release or interview handoff.

## Version And Git

- Current package version:
- HEAD commit:
- `HEAD...origin/main`:
- Current tag lookup:
- Working tree status:

## Release And CI

- GitHub release checked:
- Latest Tests workflow status:
- Manual release gate status, if run:

## Verification

```bash
python -m compileall -q main.py pointcloud_geolab tests examples scripts benchmarks
python -m ruff check .
python -m black --check .
python -m pytest --cov=pointcloud_geolab
python scripts/check_repo_hygiene.py
python scripts/check_artifact_schema.py
make verify-core
make verify-portfolio
make verify-benchmarks
make verify-release-candidate
```

## Generated Artifact Guard

- Tracked generated paths:
- Ignored generated paths reviewed:

## Optional Dependency Policy

- Open3D:
- Plotly:
- laspy:
- PyTorch:
- SciPy / scikit-learn / pandas:

## Limitations To Preserve

- No full nonlinear GICP optimizer.
- No SLAM backend.
- No CUDA acceleration.
- No PointNet training release.
- No real KITTI benchmark report.
- Synthetic demos and tiny fixtures are smoke checks only.

## Follow-up Items

- API / IO / schema follow-up:
- Documentation follow-up:
- Roadmap issue links:
