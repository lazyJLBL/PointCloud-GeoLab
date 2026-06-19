from __future__ import annotations

import importlib.util

import pytest

from pointcloud_geolab.utils import optional_deps
from pointcloud_geolab.utils.optional_deps import optional_dependency_report, require_optional


def test_optional_dependency_report_contains_known_optional_packages() -> None:
    report = optional_dependency_report(["open3d", "plotly", "laspy", "torch"])

    assert set(report) == {"open3d", "plotly", "laspy", "torch"}
    assert report["open3d"]["purpose"]
    assert "install_hint" in report["torch"]


def test_optional_dependency_report_covers_all_documented_optional_packages() -> None:
    report = optional_dependency_report()

    assert {"open3d", "plotly", "laspy", "torch", "scipy", "sklearn", "pandas"} <= set(report)
    for status in report.values():
        assert {"installed", "purpose", "install_hint", "error"} <= set(status)


def test_require_optional_reports_unavailable_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str):
        if name == "open3d":
            return None
        return real_find_spec(name)

    monkeypatch.setattr(optional_deps.importlib.util, "find_spec", fake_find_spec)

    with pytest.raises(ImportError, match="Optional dependency `open3d` is unavailable"):
        require_optional("open3d")


def test_unknown_optional_dependency_is_clear() -> None:
    with pytest.raises(KeyError, match="unknown optional dependency"):
        optional_deps.optional_dependency_status("not-real")
