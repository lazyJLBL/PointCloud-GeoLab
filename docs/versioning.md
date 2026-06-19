# Versioning

PointCloud-GeoLab uses semantic version labels for reviewer milestones.

## Current Version

- Current version: `1.1.0`
- Release type: experimental Web Console MVP on top of the portfolio-stable API
- Historical releases retained: `v0.1.0`, `v0.1.1`, `v1.0.0`

The `v1.1.0` release adds an experimental FastAPI + Vue Web Console while the
documented portfolio workflow, stable public API, CLI reference, artifact
schema, and reviewer verification commands remain coherent. It does not mean
the project is a replacement for Open3D or PCL, and it does not add full
nonlinear GICP, SLAM, CUDA, PointNet training, an official KITTI benchmark, or
a production web platform.

## Compatibility Rules

- Patch releases should keep stable API names and `TaskResult` fields intact.
- Minor releases may add new stable API functions after tests and docs exist.
- Breaking changes should move through a documented deprecation period when
  practical.
- Experimental modules may change faster, but docs must mark them as
  Experimental.

## Historical Notes

`v0.1.0` established the portfolio release baseline. `v0.1.1` hardened CI,
hygiene, packaging, DevContainer, benchmark metadata, and tiny dataset fixtures.
The v1.0.0 release established the portfolio-stable milestone. The v1.1.0
release keeps those records as history instead of rewriting them.
