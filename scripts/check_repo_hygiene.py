"""Check repository hygiene for release review."""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GENERATED_PREFIXES = (
    "outputs/",
    "results/",
    "examples/demo_data/",
)

BANNED_CLAIM_PATTERNS = (
    ("production-ready", re.compile(r"\bproduction-ready\b", re.IGNORECASE)),
    ("industrial-grade", re.compile(r"\bindustrial-grade\b", re.IGNORECASE)),
    ("enterprise-grade", re.compile(r"\benterprise-grade\b", re.IGNORECASE)),
    ("state-of-the-art", re.compile(r"\bstate-of-the-art\b", re.IGNORECASE)),
    ("full GICP", re.compile(r"\bfull\s+GICP\b", re.IGNORECASE)),
    ("full nonlinear GICP", re.compile(r"\bfull\s+nonlinear\s+GICP\b", re.IGNORECASE)),
)

NEGATION_MARKERS = (
    "not ",
    "not a ",
    "not an ",
    "is not ",
    "isn't ",
    "do not ",
    "does not ",
    "never ",
    "without ",
    "不是",
    "非",
)

TEXT_SHAPE_FILES = (
    "README.md",
    "pyproject.toml",
    ".github/workflows/tests.yml",
)


@dataclass(frozen=True, slots=True)
class HygieneResult:
    """Result of repository hygiene checks."""

    checked_files: list[Path]
    issues: list[str]

    @property
    def success(self) -> bool:
        return not self.issues


def run_hygiene(
    root: str | Path = ROOT,
    tracked_files: list[str] | None = None,
    max_line_length: int = 140,
) -> HygieneResult:
    """Run all repository hygiene checks."""

    repo = Path(root)
    issues: list[str] = []
    checked_files: list[Path] = []

    if tracked_files is None:
        try:
            tracked_files = _git_tracked_files(repo)
        except RuntimeError as exc:
            issues.append(str(exc))
            tracked_files = []

    issues.extend(check_tracked_generated_paths(tracked_files))

    docs = _claim_docs(repo)
    checked_files.extend(docs)
    issues.extend(check_overclaim_terms(docs))

    readme = repo / "README.md"
    checked_files.append(readme)
    issues.extend(check_readme_links(repo, readme))

    shape_files = [repo / name for name in TEXT_SHAPE_FILES]
    checked_files.extend(shape_files)
    issues.extend(check_text_file_shape(shape_files, max_line_length=max_line_length))

    version_files = [
        repo / "pointcloud_geolab" / "__init__.py",
        repo / "pyproject.toml",
        repo / "CITATION.cff",
        repo / "CHANGELOG.md",
        repo / "docs" / "releases" / "v0.1.0.md",
    ]
    checked_files.extend(version_files)
    issues.extend(check_version_consistency(repo))

    return HygieneResult(checked_files=_unique_paths(checked_files), issues=issues)


def check_tracked_generated_paths(tracked_files: list[str]) -> list[str]:
    """Return issues for generated output paths that are tracked by git."""

    issues: list[str] = []
    for raw_path in tracked_files:
        path = raw_path.replace("\\", "/").lstrip("./")
        for prefix in GENERATED_PREFIXES:
            if path == prefix.rstrip("/") or path.startswith(prefix):
                issues.append(f"generated path is tracked by git: {path}")
                break
    return issues


def check_overclaim_terms(paths: list[Path]) -> list[str]:
    """Return issues for positive overclaim language in README/docs."""

    issues: list[str] = []
    for path in paths:
        if not path.exists():
            issues.append(f"missing documentation file: {path}")
            continue
        previous = ""
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            lower = line.lower()
            for label, pattern in BANNED_CLAIM_PATTERNS:
                for match in pattern.finditer(line):
                    context_before = f"{previous} {lower[: match.start()]}".lower()
                    if _is_negated_context(context_before):
                        continue
                    issues.append(f"{path}:{line_number}: overclaim term `{label}`")
            previous = lower
    return issues


def check_readme_links(root: Path, readme: Path) -> list[str]:
    """Return issues for missing local README links and images."""

    if not readme.exists():
        return [f"missing README file: {readme}"]

    text = readme.read_text(encoding="utf-8")
    targets = re.findall(r"!?\[[^\]]+\]\(([^)]+)\)", text)
    issues: list[str] = []
    for target in targets:
        if not _is_local_target(target):
            continue
        path_part = target.split("#", 1)[0]
        if not path_part:
            continue
        candidate = (root / path_part).resolve()
        if not candidate.exists():
            issues.append(f"README link target is missing: {target}")
    return issues


def check_text_file_shape(paths: list[Path], max_line_length: int = 140) -> list[str]:
    """Return issues for important files that look minified or badly wrapped."""

    issues: list[str] = []
    for path in paths:
        if not path.exists():
            issues.append(f"missing text-shape file: {path}")
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) <= 3 and sum(len(line) for line in lines) > max_line_length:
            issues.append(f"{path}: appears to be compressed into too few lines")
        for line_number, line in enumerate(lines, start=1):
            if len(line) > max_line_length:
                issues.append(
                    f"{path}:{line_number}: line length {len(line)} exceeds {max_line_length}"
                )
    return issues


def check_version_consistency(root: Path) -> list[str]:
    """Return issues if package, citation, changelog, and release docs disagree."""

    versions = {
        "package": _regex_value(
            root / "pointcloud_geolab" / "__init__.py",
            r'__version__\s*=\s*"([^"]+)"',
        ),
        "pyproject": _regex_value(root / "pyproject.toml", r'^version\s*=\s*"([^"]+)"'),
        "citation": _regex_value(root / "CITATION.cff", r'^version:\s*"([^"]+)"'),
    }
    issues = [
        f"missing version metadata: {name}" for name, value in versions.items() if value is None
    ]
    present_versions = {value for value in versions.values() if value is not None}
    if len(present_versions) > 1:
        rendered = ", ".join(f"{name}={value}" for name, value in sorted(versions.items()))
        issues.append(f"version metadata mismatch: {rendered}")
        return issues
    if not present_versions:
        return issues

    version = present_versions.pop()
    release_name = f"v{version}"
    changelog = root / "CHANGELOG.md"
    if not changelog.exists() or f"## {release_name} " not in changelog.read_text(encoding="utf-8"):
        issues.append(f"CHANGELOG.md does not contain a {release_name} section")

    release_doc = root / "docs" / "releases" / f"{release_name}.md"
    if not release_doc.exists():
        issues.append(f"missing release notes: {release_doc}")
    elif f"# {release_name} " not in release_doc.read_text(encoding="utf-8"):
        issues.append(f"{release_doc}: missing {release_name} heading")
    return issues


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root to check.")
    parser.add_argument(
        "--max-line-length",
        type=int,
        default=140,
        help="Maximum line length for README, pyproject, and workflow files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_hygiene(args.root, max_line_length=args.max_line_length)
    print(f"Checked {len(result.checked_files)} repository hygiene files")
    if result.issues:
        print("Repository hygiene issues:")
        for issue in result.issues:
            print(f"- {issue}")
        return 1
    print("Repository hygiene checks passed.")
    return 0


def _git_tracked_files(root: Path) -> list[str]:
    command = ["git", "-C", str(root), "ls-files", "--", *GENERATED_PREFIXES]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"git ls-files failed: {detail}")
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _claim_docs(root: Path) -> list[Path]:
    docs_root = root / "docs"
    docs = sorted(docs_root.rglob("*.md")) if docs_root.exists() else []
    return [root / "README.md", *docs]


def _is_negated_context(context_before: str) -> bool:
    return any(marker in context_before for marker in NEGATION_MARKERS)


def _is_local_target(target: str) -> bool:
    return not (
        target.startswith("http://")
        or target.startswith("https://")
        or target.startswith("mailto:")
        or target.startswith("#")
    )


def _regex_value(path: Path, pattern: str) -> str | None:
    if not path.exists():
        return None
    match = re.search(pattern, path.read_text(encoding="utf-8"), flags=re.MULTILINE)
    return match.group(1) if match else None


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


if __name__ == "__main__":
    raise SystemExit(main())
