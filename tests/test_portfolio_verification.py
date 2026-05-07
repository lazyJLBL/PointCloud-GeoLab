from __future__ import annotations

import subprocess

from pointcloud_geolab.api import run_portfolio_verification


class _Completed:
    returncode = 0
    stdout = "ok"
    stderr = ""


def test_portfolio_verification_report_smoke(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _Completed())
    monkeypatch.setattr("pointcloud_geolab.api._missing_readme_artifacts", lambda root: [])

    result = run_portfolio_verification(output_dir=tmp_path, quick=True)

    assert result.success
    assert result.metrics["passed_commands"] > 0
    assert (tmp_path / "portfolio_check_report.md").exists()
