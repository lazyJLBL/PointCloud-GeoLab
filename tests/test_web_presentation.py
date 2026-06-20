from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_has_web_console_hero_and_latest_release() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "PointCloud-GeoLab = point-cloud geometry core" in readme
    assert "Experimental Web Console" in readme
    assert "Latest release: `v1.1.0`" in readme
    assert "docs/assets/web_console_dashboard.svg" in readme
    assert "not a production web platform" in readme


def test_web_console_svg_assets_exist() -> None:
    for name in [
        "web_console_dashboard.svg",
        "web_console_dataset_preview.svg",
        "web_console_task_artifacts.svg",
    ]:
        path = ROOT / "docs" / "assets" / name
        text = path.read_text(encoding="utf-8")
        assert "<svg" in text


def test_gallery_links_web_console_assets() -> None:
    gallery = (ROOT / "docs" / "gallery" / "README.md").read_text(encoding="utf-8")

    assert "## Experimental Web Console" in gallery
    assert "../assets/web_console_dashboard.svg" in gallery
    assert "../assets/web_console_dataset_preview.svg" in gallery
    assert "../assets/web_console_task_artifacts.svg" in gallery


def test_datasets_copy_includes_off_uploads() -> None:
    page = (ROOT / "web" / "frontend" / "src" / "components" / "FileUploader.vue").read_text(
        encoding="utf-8"
    )

    assert ".off" in page
    assert "ModelNet-like" in page


def test_dataset_required_pages_disable_run_without_selection() -> None:
    for name in [
        "Preprocessing.vue",
        "Segmentation.vue",
        "Geometry.vue",
        "Primitives.vue",
    ]:
        page = (ROOT / "web" / "frontend" / "src" / "pages" / name).read_text(encoding="utf-8")
        assert ':disabled="!canRun"' in page
        assert "Choose a dataset" in page or "Upload a dataset" in page


def test_registration_requires_source_and_target() -> None:
    page = (ROOT / "web" / "frontend" / "src" / "pages" / "Registration.vue").read_text(
        encoding="utf-8"
    )

    assert ':disabled="!canRun"' in page
    assert "Choose both source and target datasets" in page


def test_benchmark_page_mentions_quick_local_timing() -> None:
    page = (ROOT / "web" / "frontend" / "src" / "pages" / "Benchmark.vue").read_text(
        encoding="utf-8"
    )
    compact = " ".join(page.split())

    assert "always use <strong>quick</strong> mode" in compact
    assert "local machine references" in compact
    assert ':max="3"' in page
