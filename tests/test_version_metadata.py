from __future__ import annotations

import re
from pathlib import Path

import pointcloud_geolab

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = "0.1.1"


def test_package_and_project_versions_match_release() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert pointcloud_geolab.__version__ == EXPECTED_VERSION
    assert re.search(r'^version\s*=\s*"0\.1\.1"$', pyproject, flags=re.MULTILINE)


def test_release_metadata_mentions_current_release_and_history() -> None:
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    current_notes = (ROOT / "docs" / "releases" / "v0.1.1.md").read_text(encoding="utf-8")
    historical_notes = (ROOT / "docs" / "releases" / "v0.1.0.md").read_text(encoding="utf-8")

    assert re.search(r'^version:\s*"0\.1\.1"$', citation, flags=re.MULTILINE)
    assert "## v0.1.1 - 2026-06-18" in changelog
    assert "## v0.1.0 Portfolio Release" in changelog
    assert "# v0.1.1 Hardening Release" in current_notes
    assert "# v0.1.0 Portfolio Release" in historical_notes
