# Versioning

PointCloud-GeoLab uses semantic version labels for reviewer milestones.

## Current Target

- Current target: `1.0.0`
- Release type: portfolio-stable / reviewer-stable candidate
- Historical releases retained: `v0.1.0`, `v0.1.1`

The `v1.0.0` target means the documented portfolio workflow, stable public API,
CLI reference, artifact schema, and reviewer verification commands are expected
to remain coherent. It does not mean the project is a replacement for Open3D or
PCL, and it does not add full nonlinear GICP, SLAM, CUDA, PointNet training, or
an official KITTI benchmark.

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
The v1.0.0 candidate keeps those records as history instead of rewriting them.
