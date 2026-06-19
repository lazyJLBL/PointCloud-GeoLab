from __future__ import annotations

import re
from pathlib import Path

import pointcloud_geolab

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = "1.1.0"


def test_package_and_project_versions_match_release() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert pointcloud_geolab.__version__ == EXPECTED_VERSION
    assert re.search(r'^version\s*=\s*"1\.1\.0"$', pyproject, flags=re.MULTILINE)


def test_release_metadata_mentions_current_release_and_history() -> None:
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    current_notes = (ROOT / "docs" / "releases" / "v1.1.0.md").read_text(encoding="utf-8")
    historical_100_notes = (ROOT / "docs" / "releases" / "v1.0.0.md").read_text(encoding="utf-8")
    historical_011_notes = (ROOT / "docs" / "releases" / "v0.1.1.md").read_text(encoding="utf-8")
    historical_010_notes = (ROOT / "docs" / "releases" / "v0.1.0.md").read_text(encoding="utf-8")

    assert re.search(r'^version:\s*"1\.1\.0"$', citation, flags=re.MULTILINE)
    assert "## v1.1.0 - 2026-06-19" in changelog
    assert "## v1.0.0 - 2026-06-18" in changelog
    assert "## v0.1.1 - 2026-06-18" in changelog
    assert "## v0.1.0 Portfolio Release" in changelog
    assert "# v1.1.0 Experimental Web Console MVP" in current_notes
    assert "# v1.0.0 Portfolio-Stable Release" in historical_100_notes
    assert "# v0.1.1 Hardening Release" in historical_011_notes
    assert "# v0.1.0 Portfolio Release" in historical_010_notes
