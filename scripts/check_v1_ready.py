"""Check current v1 portfolio-stable release readiness."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_repo_hygiene import (
    display_path,
    git_tracked_generated_files_for_check,
    regex_value,
)

CURRENT_VERSION = "1.1.0"
RELEASE_DATE = "2026-06-19"
GENERATED_PREFIXES = (
    "outputs/",
    "results/",
    "examples/demo_data/",
    "benchmark_results/",
    "outputs/kitti_segmentation/",
)

REQUIRED_DOCS = (
    "docs/releases/v1.1.0.md",
    "docs/releases/v1.1.0_artifacts.json",
    "docs/versioning.md",
    "docs/api_stability.md",
    "docs/cli_reference.md",
    "docs/limitations.md",
    "docs/project_boundary.md",
    "docs/gallery/README.md",
    "docs/scale_benchmark.md",
    "docs/case_studies/kitti_lidar_result.md",
    "docs/web_console.md",
    "docs/web_api.md",
    "web/README.md",
    "web/backend/tests/test_web_backend.py",
    "web/frontend/package.json",
)

REQUIRED_ASSETS = (
    "docs/assets/portfolio_raw_pointcloud.png",
    "docs/assets/portfolio_downsampled.png",
    "docs/assets/portfolio_registration_before_after.png",
    "docs/assets/portfolio_segmentation_result.png",
    "docs/assets/portfolio_bbox_normals.png",
    "docs/assets/kitti_case_study_tiny.png",
    "docs/assets/scale_benchmark_quick.png",
)

REQUIRED_MAKE_TARGETS = (
    "verify-realdata",
    "verify-scale-benchmark",
    "verify-v1-candidate",
    "verify-web",
)


@dataclass(frozen=True, slots=True)
class V1ReadyResult:
    """Result of current v1 release checks."""

    root: Path
    issues: list[str]
    warnings: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return not self.issues


def run_v1_ready(
    root: str | Path = ROOT,
    tracked_files: list[str] | None = None,
    require_git: bool = False,
) -> V1ReadyResult:
    """Run current v1 readiness checks."""

    repo = Path(root).resolve()
    issues: list[str] = []
    warnings: list[str] = []
    issues.extend(check_version_consistency(repo))
    issues.extend(check_required_files(repo))
    issues.extend(check_release_manifest(repo))
    issues.extend(check_makefile_targets(repo))
    generated_issues, generated_warnings = check_generated_paths_for_run(
        repo,
        tracked_files=tracked_files,
        require_git=require_git,
    )
    issues.extend(generated_issues)
    warnings.extend(generated_warnings)
    issues.extend(check_boundary_wording(repo))
    return V1ReadyResult(repo, sorted(set(issues)), sorted(set(warnings)))


def check_version_consistency(root: Path) -> list[str]:
    """Return issues for current v1 metadata mismatches."""

    versions = {
        "pyproject.toml": regex_value(root / "pyproject.toml", r'^version\s*=\s*"([^"]+)"'),
        "pointcloud_geolab/__init__.py": regex_value(
            root / "pointcloud_geolab" / "__init__.py",
            r'^__version__\s*=\s*"([^"]+)"',
        ),
        "CITATION.cff": regex_value(root / "CITATION.cff", r'^version:\s*"([^"]+)"'),
        "web/frontend/package.json": regex_value(
            root / "web" / "frontend" / "package.json",
            r'^\s*"version":\s*"([^"]+)"',
        ),
        "web/backend/app/main.py": regex_value(
            root / "web" / "backend" / "app" / "main.py",
            r'^\s*version="([^"]+)"',
        ),
    }
    issues = [
        f"{name}: expected version {CURRENT_VERSION}, got {value or '<missing>'}"
        for name, value in sorted(versions.items())
        if value != CURRENT_VERSION
    ]
    changelog = root / "CHANGELOG.md"
    if not changelog.exists():
        issues.append("CHANGELOG.md: missing file")
    else:
        text = changelog.read_text(encoding="utf-8")
        expected = f"## v{CURRENT_VERSION} - {RELEASE_DATE}"
        if expected not in text:
            issues.append(f"CHANGELOG.md: missing `{expected}` section")
        for historical in [
            "## v1.0.0 - 2026-06-18",
            "## v0.1.1 - 2026-06-18",
            "## v0.1.0 Portfolio Release",
        ]:
            if historical not in text:
                issues.append(f"CHANGELOG.md: missing historical section `{historical}`")
    return issues


def check_required_files(root: Path) -> list[str]:
    """Return issues for missing v1 docs and gallery assets."""

    issues: list[str] = []
    for relative in [*REQUIRED_DOCS, *REQUIRED_ASSETS]:
        path = root / relative
        if not path.exists():
            issues.append(f"{relative}: missing required v1 file")
        elif path.is_file() and path.stat().st_size == 0:
            issues.append(f"{relative}: required v1 file is empty")
    return issues


def check_release_manifest(root: Path) -> list[str]:
    """Return issues for the v1 artifact manifest."""

    manifest = root / "docs" / "releases" / "v1.1.0_artifacts.json"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [f"{display_path(root, manifest)}: missing release artifact manifest"]
    except json.JSONDecodeError as exc:
        return [f"{display_path(root, manifest)}: invalid JSON ({exc})"]
    if not isinstance(payload, dict):
        return [f"{display_path(root, manifest)}: JSON root must be an object"]

    issues: list[str] = []
    if payload.get("version") != CURRENT_VERSION:
        issues.append(f"{display_path(root, manifest)}: version must be {CURRENT_VERSION}")
    commands = payload.get("local_verification_commands")
    if not isinstance(commands, list) or "make verify-v1-candidate" not in commands:
        issues.append(f"{display_path(root, manifest)}: missing make verify-v1-candidate command")
    artifacts = payload.get("expected_generated_artifacts")
    if not isinstance(artifacts, dict):
        issues.append(f"{display_path(root, manifest)}: expected_generated_artifacts missing")
    else:
        if not any("report.html" in item for item in artifacts.get("portfolio", [])):
            issues.append(f"{display_path(root, manifest)}: portfolio HTML report not listed")
        if not any("scale_benchmark" in item for item in artifacts.get("benchmarks", [])):
            issues.append(f"{display_path(root, manifest)}: scale benchmark artifacts not listed")
        if not any("kitti_segmentation" in item for item in artifacts.get("realdata", [])):
            issues.append(f"{display_path(root, manifest)}: KITTI workflow artifacts not listed")
        if not any("outputs/web" in item for item in artifacts.get("web", [])):
            issues.append(f"{display_path(root, manifest)}: Web Console artifacts not listed")
    limitations = " ".join(str(item).lower() for item in payload.get("limitations", []))
    for required in [
        "not a full nonlinear gicp",
        "not an official kitti benchmark",
        "not a slam backend",
        "not a cuda",
    ]:
        if required not in limitations:
            issues.append(f"{display_path(root, manifest)}: limitations missing `{required}`")
    return issues


def check_makefile_targets(root: Path) -> list[str]:
    """Return issues for missing Makefile v1 targets."""

    path = root / "Makefile"
    if not path.exists():
        return ["Makefile: missing file"]
    text = path.read_text(encoding="utf-8")
    return [
        f"Makefile: missing `{target}` target"
        for target in REQUIRED_MAKE_TARGETS
        if f"{target}:" not in text
    ]


def check_generated_paths(root: Path, tracked_files: list[str] | None = None) -> list[str]:
    """Return issues for generated or external paths tracked by Git."""

    issues, warnings = check_generated_paths_for_run(root, tracked_files=tracked_files)
    return [*issues, *warnings]


def check_generated_paths_for_run(
    root: Path,
    tracked_files: list[str] | None = None,
    require_git: bool = False,
) -> tuple[list[str], list[str]]:
    """Return generated-path issues and non-fatal Git warnings."""

    warnings: list[str] = []
    if tracked_files is None:
        try:
            tracked_files, git_warning = git_tracked_generated_files_for_check(
                root,
                require_git=require_git,
            )
            if git_warning:
                warnings.append(git_warning)
        except RuntimeError as exc:
            return [str(exc)], warnings
    issues: list[str] = []
    for item in tracked_files:
        normalized = item.replace("\\", "/")
        if any(
            normalized == prefix.rstrip("/") or normalized.startswith(prefix)
            for prefix in GENERATED_PREFIXES
        ):
            issues.append(f"generated path is tracked by git: {normalized}")
    return issues, warnings


def check_boundary_wording(root: Path) -> list[str]:
    """Return issues if v1 docs overstate unresolved roadmap items."""

    issues: list[str] = []
    paths = [
        root / "README.md",
        root / "docs" / "limitations.md",
        root / "docs" / "project_boundary.md",
        root / "docs" / "releases" / "v1.1.0.md",
        root / "docs" / "case_studies" / "kitti_lidar_result.md",
        root / "docs" / "web_console.md",
        root / "docs" / "web_api.md",
        root / "web" / "README.md",
    ]
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in paths if path.exists() and path.is_file()
    ).lower()
    required_phrases = [
        "not a full nonlinear gicp",
        "not an official kitti benchmark",
        "not a slam backend",
        "not cuda",
        "not a pointnet training release",
        "not a production web platform",
        "synthetic",
    ]
    for phrase in required_phrases:
        if phrase not in combined:
            issues.append(f"v1 docs missing boundary phrase `{phrase}`")
    risky_patterns = [
        r"\bproduction-ready\b",
        r"\bindustrial-grade\b",
        r"\bfull nonlinear gicp (is )?(implemented|complete|done)\b",
        r"\breal kitti benchmark (is )?(implemented|complete|done)\b",
    ]
    for pattern in risky_patterns:
        match = re.search(pattern, combined)
        if match:
            issues.append(f"v1 docs contain overclaim wording `{match.group(0)}`")
    if re.search(r"\bissue\s+#2\b", combined):
        issues.append(
            "v1 docs should describe official KITTI benchmark as future work, "
            "not by stale issue #2 state"
        )
    return issues


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--require-git",
        action="store_true",
        help="Fail instead of warning when Git metadata is unavailable.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_v1_ready(args.root, require_git=args.require_git)
    print(f"v{CURRENT_VERSION} readiness checked repository: {result.root}")
    for warning in result.warnings:
        print(f"Warning: {warning}")
    if result.issues:
        print(f"v{CURRENT_VERSION} readiness failed:")
        for issue in result.issues:
            print(f"- {issue}")
        return 1
    print(f"v{CURRENT_VERSION} portfolio-stable release checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
