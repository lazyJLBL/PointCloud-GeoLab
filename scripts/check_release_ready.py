"""Check v0.1.1 release-candidate metadata and repository boundaries."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_repo_hygiene import (
    CLAIM_DOC_PATTERNS,
    GENERATED_PREFIXES,
    check_overclaim_terms,
    check_tracked_generated_paths,
    collect_files,
    display_path,
    git_tracked_generated_files,
    regex_value,
)

CURRENT_VERSION = "0.1.1"
RELEASE_DATE = "2026-06-18"
ARTIFACT_MANIFEST = "v0.1.1_artifacts.json"

Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class ReleaseReadyResult:
    """Result of release-candidate checks."""

    checked_files: list[Path]
    issues: list[str]
    warnings: list[str]
    workspace_clean: bool

    @property
    def success(self) -> bool:
        return not self.issues


def run_release_ready(
    root: str | Path = ROOT,
    version: str = CURRENT_VERSION,
    tracked_files: list[str] | None = None,
    status_output: str | None = None,
    runner: Runner = subprocess.run,
) -> ReleaseReadyResult:
    """Run release-candidate checks without requiring network access."""

    repo = Path(root).resolve()
    checked: list[Path] = []
    issues: list[str] = []
    warnings: list[str] = []

    version_files = version_metadata_files(repo)
    checked.extend(version_files.values())
    issues.extend(check_version_metadata(repo, version))

    changelog = repo / "CHANGELOG.md"
    checked.append(changelog)
    issues.extend(check_changelog_section(changelog, version))

    release_doc = repo / "docs" / "releases" / f"v{version}.md"
    manifest_path = repo / "docs" / "releases" / ARTIFACT_MANIFEST
    checked.extend([release_doc, manifest_path])
    issues.extend(check_release_docs(repo, release_doc, manifest_path, version))

    claim_docs = collect_files(repo, CLAIM_DOC_PATTERNS)
    checked.extend(claim_docs)
    issues.extend(check_overclaim_terms(repo, claim_docs))
    issues.extend(check_roadmap_claims(repo, claim_docs))

    if tracked_files is None:
        try:
            tracked_files = git_tracked_generated_files(repo)
        except RuntimeError as exc:
            issues.append(str(exc))
            tracked_files = []
    issues.extend(check_tracked_generated_paths(tracked_files))

    if status_output is None:
        status_output, status_warning = git_status(repo, runner=runner)
        if status_warning:
            warnings.append(status_warning)
    workspace_clean = not status_output.strip()
    if not workspace_clean:
        warnings.append(
            "working tree has uncommitted changes; rerun after commit for final ready prompt"
        )

    return ReleaseReadyResult(
        checked_files=unique_paths(checked),
        issues=sorted(issues),
        warnings=sorted(set(warnings)),
        workspace_clean=workspace_clean,
    )


def version_metadata_files(root: Path) -> dict[str, Path]:
    """Return the files that should agree on the current package version."""

    return {
        "pyproject": root / "pyproject.toml",
        "package": root / "pointcloud_geolab" / "__init__.py",
        "citation": root / "CITATION.cff",
    }


def check_version_metadata(root: Path, expected_version: str) -> list[str]:
    """Return issues for mismatched release metadata."""

    files = version_metadata_files(root)
    values = {
        "pyproject": regex_value(files["pyproject"], r'^version\s*=\s*"([^"]+)"'),
        "package": regex_value(files["package"], r'^__version__\s*=\s*"([^"]+)"'),
        "citation": regex_value(files["citation"], r'^version:\s*"([^"]+)"'),
    }
    issues = [
        f"{display_path(root, files[name])}: missing version metadata"
        for name, value in sorted(values.items())
        if value is None
    ]
    for name, value in sorted(values.items()):
        if value is not None and value != expected_version:
            issues.append(
                f"{display_path(root, files[name])}: expected version "
                f"{expected_version}, found {value}"
            )

    citation = files["citation"]
    date_value = regex_value(citation, r'^date-released:\s*"([^"]+)"')
    if date_value != RELEASE_DATE:
        issues.append(
            f"{display_path(root, citation)}: expected date-released "
            f"{RELEASE_DATE}, found {date_value or '<missing>'}"
        )
    return issues


def check_changelog_section(path: Path, version: str) -> list[str]:
    """Return issues if CHANGELOG is not organized for the release candidate."""

    if not path.exists():
        return ["CHANGELOG.md: missing file"]
    text = path.read_text(encoding="utf-8")
    issues: list[str] = []
    if "## Unreleased" not in text:
        issues.append("CHANGELOG.md: missing Unreleased section")
    expected_heading = f"## v{version} - {RELEASE_DATE}"
    if expected_heading not in text:
        issues.append(f"CHANGELOG.md: missing `{expected_heading}` section")
    if "## v0.1.0 Portfolio Release" not in text:
        issues.append("CHANGELOG.md: missing historical v0.1.0 release section")
    return issues


def check_release_docs(
    root: Path,
    release_doc: Path,
    manifest_path: Path,
    version: str,
) -> list[str]:
    """Return issues for missing release notes or malformed artifact manifest."""

    issues: list[str] = []
    release_name = f"v{version}"
    if not release_doc.exists():
        issues.append(f"{display_path(root, release_doc)}: missing release notes")
    else:
        text = release_doc.read_text(encoding="utf-8")
        if f"# {release_name} " not in text:
            issues.append(f"{display_path(root, release_doc)}: missing {release_name} heading")

    manifest, manifest_issues = load_artifact_manifest(manifest_path)
    issues.extend(f"{display_path(root, manifest_path)}: {issue}" for issue in manifest_issues)
    if manifest:
        issues.extend(check_artifact_manifest(root, manifest_path, manifest, version))
    return issues


def load_artifact_manifest(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """Load and validate that the release artifact manifest is JSON object shaped."""

    if not path.exists():
        return None, ["missing artifact manifest"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, [f"invalid JSON ({exc})"]
    if not isinstance(payload, dict):
        return None, ["artifact manifest must be a JSON object"]
    return payload, []


def check_artifact_manifest(
    root: Path,
    path: Path,
    manifest: dict[str, Any],
    version: str,
) -> list[str]:
    """Return issues for required artifact-manifest fields."""

    issues: list[str] = []
    if manifest.get("version") != version:
        issues.append(f"expected version {version}, found {manifest.get('version')!r}")
    if not manifest.get("commit"):
        issues.append("missing commit placeholder or hash")

    commands = manifest.get("local_verification_commands")
    if not isinstance(commands, list) or not commands:
        issues.append("missing local_verification_commands list")
    else:
        required = {
            "python scripts/check_release_ready.py",
            "make verify-release-candidate",
        }
        missing = sorted(command for command in required if command not in commands)
        issues.extend(f"missing verification command `{command}`" for command in missing)

    expected = manifest.get("expected_generated_artifacts")
    if not isinstance(expected, dict):
        issues.append("missing expected_generated_artifacts object")
    else:
        issues.extend(check_expected_artifact_group(path, expected, "portfolio"))
        issues.extend(check_expected_artifact_group(path, expected, "benchmarks"))

    ignored = manifest.get("ignored_artifact_paths")
    if not isinstance(ignored, list):
        issues.append("missing ignored_artifact_paths list")
    else:
        for prefix in GENERATED_PREFIXES:
            if prefix not in ignored:
                issues.append(f"ignored_artifact_paths missing `{prefix}`")

    limitations = " ".join(str(item).lower() for item in manifest.get("limitations", []))
    if "not real" not in limitations or "not a full nonlinear gicp" not in limitations:
        issues.append("limitations must state real-data and full-GICP boundaries")

    roadmap = " ".join(str(item).lower() for item in manifest.get("open_roadmap_items", []))
    if "full nonlinear gicp" not in roadmap or "real kitti benchmark" not in roadmap:
        issues.append("open_roadmap_items must retain full GICP and real KITTI work")

    return [f"{display_path(root, path)}: {issue}" for issue in issues]


def check_expected_artifact_group(
    manifest_path: Path,
    expected: dict[str, Any],
    group: str,
) -> list[str]:
    """Return issues for one expected artifact group."""

    values = expected.get(group)
    if not isinstance(values, list) or not values:
        return [f"expected_generated_artifacts.{group} must be a non-empty list"]
    return [
        f"expected_generated_artifacts.{group} contains non-string value in {manifest_path.name}"
        for value in values
        if not isinstance(value, str)
    ]


def check_roadmap_claims(root: Path, paths: list[Path]) -> list[str]:
    """Return issues if open issue capabilities are described as completed."""

    issues: list[str] = []
    completed_words = r"(completed|done|implemented|released|finished|available|supported)"
    patterns = (
        ("full nonlinear GICP", re.compile(rf"\b{completed_words}\b.*full nonlinear GICP", re.I)),
        ("full nonlinear GICP", re.compile(rf"full nonlinear GICP.*\b{completed_words}\b", re.I)),
        ("real KITTI benchmark", re.compile(rf"\b{completed_words}\b.*real KITTI benchmark", re.I)),
        ("real KITTI benchmark", re.compile(rf"real KITTI benchmark.*\b{completed_words}\b", re.I)),
    )
    for path in paths:
        if not path.exists():
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            lower = line.lower()
            if _negated_roadmap_line(lower):
                continue
            for label, pattern in patterns:
                if pattern.search(line):
                    issues.append(
                        f"{display_path(root, path)}:{line_number}: "
                        f"roadmap item `{label}` appears completed"
                    )
    return issues


def git_status(root: Path, runner: Runner = subprocess.run) -> tuple[str, str | None]:
    """Return short status output and an optional warning."""

    command = ["git", "-C", str(root), "status", "--short", "--untracked-files=all"]
    completed = runner(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip() or "unknown git status failure"
        return "", f"git status skipped: {detail}"
    return completed.stdout, None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--version", default=CURRENT_VERSION)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_release_ready(args.root, version=args.version)
    print(f"Release readiness checked {len(result.checked_files)} files.")
    for warning in result.warnings:
        print(f"Warning: {warning}")
    if result.issues:
        print("Release readiness failed:")
        for issue in result.issues:
            print(f"- {issue}")
        return 1
    if result.workspace_clean:
        print("Release candidate is ready for manual v0.1.1 tag/release creation.")
    else:
        print("Release readiness checks passed for the current source state.")
    return 0


def _negated_roadmap_line(line: str) -> bool:
    return any(
        marker in line
        for marker in (
            "not ",
            "not a ",
            "does not ",
            "do not ",
            "still does not ",
            "future",
            "roadmap",
            "open_roadmap",
            "remain",
            "remains",
        )
    )


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


if __name__ == "__main__":
    raise SystemExit(main())
