from __future__ import annotations

import re
from pathlib import Path

import tomllib

import pointcloud_geolab

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = "0.1.0"


def test_package_and_project_versions_match_release() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pointcloud_geolab.__version__ == EXPECTED_VERSION
    assert pyproject["project"]["version"] == EXPECTED_VERSION


def test_release_metadata_mentions_v010() -> None:
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    release_notes = (ROOT / "docs" / "releases" / "v0.1.0.md").read_text(encoding="utf-8")

    assert re.search(r'^version:\s*"0\.1\.0"$', citation, flags=re.MULTILINE)
    assert "## v0.1.0 Portfolio Release" in changelog
    assert "# v0.1.0 Portfolio Release" in release_notes
