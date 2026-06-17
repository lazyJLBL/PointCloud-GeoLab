## Summary

- 

## Verification

- [ ] `python -m compileall -q main.py pointcloud_geolab tests examples scripts benchmarks`
- [ ] `python -m ruff check .`
- [ ] `python -m black --check .`
- [ ] `python -m pytest --cov=pointcloud_geolab`
- [ ] `make verify-core`
- [ ] `make verify-portfolio`

## Release Honesty

- [ ] Generated outputs are not committed.
- [ ] Synthetic demos are not described as real-data results.
- [ ] Fallback behavior is described as diagnostic, not as descriptor success.
- [ ] GICP wording remains GICP-style covariance-weighted ICP, not full nonlinear GICP.
