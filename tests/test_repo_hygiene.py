from __future__ import annotations

from pathlib import Path

from scripts.check_repo_hygiene import (
    ROOT,
    check_readme_links,
    check_text_file_shape,
    main,
    run_hygiene,
)


def test_repo_hygiene_script_runs_on_repository() -> None:
    assert main(["--root", str(ROOT)]) == 0


def test_repo_hygiene_finds_tracked_generated_path(tmp_path: Path) -> None:
    _write_minimal_repo(tmp_path)

    result = run_hygiene(tmp_path, tracked_files=["outputs/portfolio_demo/report.md"])

    assert not result.success
    assert any("generated path is tracked" in issue for issue in result.issues)


def test_repo_hygiene_finds_bad_readme_link(tmp_path: Path) -> None:
    _write_minimal_repo(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text("[Missing](docs/missing.md)\n", encoding="utf-8")

    issues = check_readme_links(tmp_path, readme)

    assert issues == ["README link target is missing: docs/missing.md"]


def test_repo_hygiene_finds_long_single_line_text(tmp_path: Path) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text("x" * 180, encoding="utf-8")

    issues = check_text_file_shape([path], max_line_length=100)

    assert any("compressed into too few lines" in issue for issue in issues)
    assert any("line length 180 exceeds 100" in issue for issue in issues)


def test_repo_hygiene_finds_version_mismatch(tmp_path: Path) -> None:
    _write_minimal_repo(tmp_path)
    (tmp_path / "CITATION.cff").write_text('version: "0.2.0"\n', encoding="utf-8")

    result = run_hygiene(tmp_path, tracked_files=[])

    assert not result.success
    assert any("version metadata mismatch" in issue for issue in result.issues)


def test_repo_hygiene_finds_positive_overclaim(tmp_path: Path) -> None:
    _write_minimal_repo(tmp_path)
    (tmp_path / "docs" / "bad.md").write_text(
        "This is a full GICP implementation.\n",
        encoding="utf-8",
    )

    result = run_hygiene(tmp_path, tracked_files=[])

    assert not result.success
    assert any("overclaim term `full GICP`" in issue for issue in result.issues)


def _write_minimal_repo(root: Path) -> None:
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / "docs" / "releases").mkdir(parents=True)
    (root / "pointcloud_geolab").mkdir()

    (root / "README.md").write_text("[Docs](docs/ok.md)\n", encoding="utf-8")
    (root / "docs" / "ok.md").write_text("# OK\n", encoding="utf-8")
    (root / "docs" / "releases" / "v0.1.0.md").write_text(
        "# v0.1.0 Portfolio Release\n",
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    (root / ".github" / "workflows" / "tests.yml").write_text(
        "name: Tests\n",
        encoding="utf-8",
    )
    (root / "CITATION.cff").write_text('version: "0.1.0"\n', encoding="utf-8")
    (root / "CHANGELOG.md").write_text(
        "# Changelog\n\n## v0.1.0 Portfolio Release\n",
        encoding="utf-8",
    )
    (root / "pointcloud_geolab" / "__init__.py").write_text(
        '__version__ = "0.1.0"\n',
        encoding="utf-8",
    )
