"""Check repository hygiene for release review and CI."""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GENERATED_PREFIXES = (
    "outputs/",
    "results/",
    "examples/demo_data/",
    "benchmark_results/",
)

CLAIM_DOC_PATTERNS = (
    "README.md",
    "docs/**/*.md",
    "web/README.md",
    "web/frontend/src/**/*.vue",
)

TEXT_SHAPE_PATTERNS = (
    "README.md",
    "CHANGELOG.md",
    "pyproject.toml",
    "Makefile",
    ".github/workflows/*.yml",
    "docs/**/*.md",
    "web/README.md",
    "web/frontend/src/**/*.vue",
    "web/frontend/src/**/*.ts",
    "scripts/**/*.py",
)

POST_RELEASE_WORDING_FILES = (
    "README.md",
    "CHANGELOG.md",
    "docs/releases/v1.0.0.md",
    "docs/releases/v1.1.0.md",
    "docs/ROADMAP.md",
    "docs/coverage.md",
    "docs/reviewer_checklist.md",
    "docs/web_console.md",
    "docs/web_api.md",
    "web/README.md",
)

POST_RELEASE_WORDING_PATTERNS = (
    (
        "v1.0.0 release candidate",
        re.compile(r"\bv1\.0\.0\s+release\s+candidate\b", re.IGNORECASE),
    ),
    (
        "v1.1.0 release candidate",
        re.compile(r"\bv1\.1\.0\s+release\s+candidate\b", re.IGNORECASE),
    ),
    ("v1.0.0 target", re.compile(r"\bv1\.0\.0\s+target\b", re.IGNORECASE)),
    ("v1.1.0 target", re.compile(r"\bv1\.1\.0\s+target\b", re.IGNORECASE)),
    (
        "v1.0.0 release readiness",
        re.compile(r"\bv1\.0\.0\s+release\s+readiness\b", re.IGNORECASE),
    ),
    (
        "v1.1.0 release readiness",
        re.compile(r"\bv1\.1\.0\s+release\s+readiness\b", re.IGNORECASE),
    ),
    (
        "uncreated v1.0.0 release",
        re.compile(r"do\s+not\s+create\s+a\s+v1\.0\.0\s+tag", re.IGNORECASE),
    ),
    (
        "uncreated v1.1.0 release",
        re.compile(r"do\s+not\s+create\s+a\s+v1\.1\.0\s+tag", re.IGNORECASE),
    ),
    ("stale issue #2 reference", re.compile(r"\bissue\s+#2\b", re.IGNORECASE)),
)

BANNED_CLAIM_PATTERNS = (
    ("production-ready", re.compile(r"\bproduction-ready\b", re.IGNORECASE)),
    ("industrial-grade", re.compile(r"\bindustrial-grade\b", re.IGNORECASE)),
    ("enterprise-grade", re.compile(r"\benterprise-grade\b", re.IGNORECASE)),
    ("state-of-the-art", re.compile(r"\bstate-of-the-art\b", re.IGNORECASE)),
    ("full GICP", re.compile(r"\bfull\s+GICP\b", re.IGNORECASE)),
    ("full nonlinear GICP", re.compile(r"\bfull\s+nonlinear\s+GICP\b", re.IGNORECASE)),
    ("SLAM backend", re.compile(r"\bSLAM\s+backend\b", re.IGNORECASE)),
)

NEGATION_MARKERS = (
    "no ",
    "not ",
    "not a ",
    "not an ",
    "is not ",
    "isn't ",
    "do not ",
    "does not ",
    "never ",
    "without ",
)


@dataclass(frozen=True, slots=True)
class HygieneResult:
    """Result of repository hygiene checks."""

    checked_files: list[Path]
    issues: list[str]
    warnings: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return not self.issues


def run_hygiene(
    root: str | Path = ROOT,
    tracked_files: list[str] | None = None,
    max_line_length: int = 140,
    require_git: bool = False,
) -> HygieneResult:
    """Run all repository hygiene checks."""

    repo = Path(root).resolve()
    checked_files: list[Path] = []
    issues: list[str] = []
    warnings: list[str] = []

    if tracked_files is None:
        try:
            tracked_files, git_warning = git_tracked_generated_files_for_check(
                repo,
                require_git=require_git,
            )
            if git_warning:
                warnings.append(git_warning)
        except RuntimeError as exc:
            issues.append(str(exc))
            tracked_files = []

    issues.extend(check_tracked_generated_paths(tracked_files))

    claim_docs = collect_files(repo, CLAIM_DOC_PATTERNS)
    checked_files.extend(claim_docs)
    issues.extend(check_overclaim_terms(repo, claim_docs))
    issues.extend(check_local_markdown_links(repo, claim_docs))

    text_shape_files = collect_files(repo, TEXT_SHAPE_PATTERNS)
    checked_files.extend(text_shape_files)
    issues.extend(
        check_text_file_shape(
            repo,
            text_shape_files,
            max_line_length=max_line_length,
        )
    )
    post_release_files = [repo / relative for relative in POST_RELEASE_WORDING_FILES]
    checked_files.extend(post_release_files)
    issues.extend(check_post_release_wording(repo, post_release_files))

    current_version = regex_value(repo / "pyproject.toml", r'^version\s*=\s*"([^"]+)"')
    version_files = [
        repo / "pyproject.toml",
        repo / "CITATION.cff",
        repo / "CHANGELOG.md",
    ]
    if current_version:
        version_files.append(repo / "docs" / "releases" / f"v{current_version}.md")
    checked_files.extend(version_files)
    issues.extend(check_version_consistency(repo))

    return HygieneResult(
        checked_files=unique_paths(checked_files),
        issues=sorted(issues),
        warnings=sorted(set(warnings)),
    )


def git_tracked_generated_files(root: Path) -> list[str]:
    """Return tracked files under generated-output prefixes."""

    command = ["git", "-C", str(root), "ls-files", "--", *GENERATED_PREFIXES]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"git ls-files failed: {detail}")
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def git_tracked_generated_files_for_check(
    root: Path,
    require_git: bool = False,
) -> tuple[list[str], str | None]:
    """Return tracked generated files or a warning for non-Git source trees."""

    if not has_git_metadata(root):
        message = (
            "git-only generated-path check skipped: .git metadata not found; "
            "pass --require-git for strict release gates"
        )
        if require_git:
            raise RuntimeError(message)
        return [], message
    try:
        return git_tracked_generated_files(root), None
    except RuntimeError as exc:
        if require_git:
            raise
        return [], f"git-only generated-path check skipped: {exc}"


def has_git_metadata(root: Path) -> bool:
    """Return True when root has enough Git metadata for git-only checks."""

    if (root / ".git").exists():
        return True
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0 and completed.stdout.strip() == "true"


def check_tracked_generated_paths(tracked_files: list[str]) -> list[str]:
    """Return issues for generated output paths that are tracked by git."""

    issues: list[str] = []
    for raw_path in sorted(tracked_files):
        path = normalize_repo_path(raw_path)
        for prefix in GENERATED_PREFIXES:
            if path == prefix.rstrip("/") or path.startswith(prefix):
                issues.append(f"generated path is tracked by git: {path}")
                break
    return issues


def check_overclaim_terms(root: Path, paths: list[Path]) -> list[str]:
    """Return issues for positive overclaim language in README/docs."""

    issues: list[str] = []
    for path in paths:
        if not path.exists():
            issues.append(f"{display_path(root, path)}: missing documentation file")
            continue
        previous_line = ""
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            lower = line.lower()
            for label, pattern in BANNED_CLAIM_PATTERNS:
                for match in pattern.finditer(line):
                    context = f"{previous_line} {lower[: match.start()]}".lower()
                    if is_negated_context(context):
                        continue
                    issues.append(
                        f"{display_path(root, path)}:{line_number}: overclaim term `{label}`"
                    )
            previous_line = lower
    return issues


def check_local_markdown_links(root: Path, paths: list[Path]) -> list[str]:
    """Return issues for missing local Markdown links and images."""

    issues: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for target in sorted(extract_markdown_targets(text)):
            if not is_local_target(target):
                continue
            target_path = target.split("#", 1)[0].strip()
            if not target_path:
                continue
            candidate = (path.parent / target_path).resolve()
            if not is_inside(candidate, root) or not candidate.exists():
                issues.append(f"{display_path(root, path)}: missing local link target `{target}`")
    return issues


def check_readme_links(root: Path, readme: Path) -> list[str]:
    """Compatibility wrapper for README link tests."""

    return check_local_markdown_links(root.resolve(), [readme])


def check_text_file_shape(
    root: Path,
    paths: list[Path],
    max_line_length: int = 140,
) -> list[str]:
    """Return issues for important files that look minified or badly wrapped."""

    issues: list[str] = []
    for path in paths:
        label = display_path(root, path)
        if not path.exists():
            issues.append(f"{label}: missing text file")
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        text_size = sum(len(line) for line in lines)
        if len(lines) <= 3 and text_size > max_line_length:
            issues.append(f"{label}: appears compressed into {len(lines)} line(s)")
        for line_number, line in enumerate(lines, start=1):
            if len(line) > max_line_length:
                issues.append(
                    f"{label}:{line_number}: line length {len(line)} exceeds " f"{max_line_length}"
                )
    return issues


def check_post_release_wording(root: Path, paths: list[Path]) -> list[str]:
    """Return issues for stale release-candidate or issue-state wording."""

    issues: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for label, pattern in POST_RELEASE_WORDING_PATTERNS:
                if pattern.search(line):
                    issues.append(
                        f"{display_path(root, path)}:{line_number}: stale post-release "
                        f"wording `{label}`"
                    )
    return issues


def check_version_consistency(root: Path) -> list[str]:
    """Return issues if release metadata disagrees."""

    versions = {
        "pyproject": regex_value(root / "pyproject.toml", r'^version\s*=\s*"([^"]+)"'),
        "citation": regex_value(root / "CITATION.cff", r'^version:\s*"([^"]+)"'),
    }
    issues = [
        f"missing version metadata: {name}"
        for name, value in sorted(versions.items())
        if value is None
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
    release_doc = root / "docs" / "releases" / f"{release_name}.md"

    if not changelog.exists():
        issues.append("CHANGELOG.md: missing file")
    elif f"## {release_name} " not in changelog.read_text(encoding="utf-8"):
        issues.append(f"CHANGELOG.md: missing {release_name} section")

    if not release_doc.exists():
        issues.append(f"{display_path(root, release_doc)}: missing release notes")
    elif f"# {release_name} " not in release_doc.read_text(encoding="utf-8"):
        issues.append(f"{display_path(root, release_doc)}: missing {release_name} heading")

    return issues


def collect_files(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    """Collect files matching glob patterns in deterministic order."""

    files: list[Path] = []
    for pattern in patterns:
        matches = sorted(path for path in root.glob(pattern) if path.is_file())
        files.extend(matches)
    return unique_paths(files)


def extract_markdown_targets(text: str) -> set[str]:
    """Extract inline/image and reference-style Markdown link targets."""

    targets = set(re.findall(r"!?\[[^\]]+\]\(([^)]+)\)", text))
    for match in re.finditer(r"^\[[^\]]+\]:\s*(\S+)", text, flags=re.MULTILINE):
        targets.add(match.group(1))
    return targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root to check.")
    parser.add_argument(
        "--max-line-length",
        type=int,
        default=140,
        help="Maximum line length for README, CHANGELOG, pyproject, workflow, and docs.",
    )
    parser.add_argument(
        "--require-git",
        action="store_true",
        help="Fail instead of warning when Git metadata is unavailable.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_hygiene(
        args.root,
        max_line_length=args.max_line_length,
        require_git=args.require_git,
    )
    print(f"Repository hygiene checked {len(result.checked_files)} files.")
    for warning in result.warnings:
        print(f"Warning: {warning}")
    if result.issues:
        print("Repository hygiene failed:")
        for issue in result.issues:
            print(f"- {issue}")
        return 1
    print("Repository hygiene passed.")
    return 0


def normalize_repo_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def is_negated_context(context_before: str) -> bool:
    window = context_before[-80:]
    return any(marker in window for marker in NEGATION_MARKERS)


def is_local_target(target: str) -> bool:
    return not (
        target.startswith("http://")
        or target.startswith("https://")
        or target.startswith("mailto:")
        or target.startswith("#")
    )


def is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def regex_value(path: Path, pattern: str) -> str | None:
    if not path.exists():
        return None
    match = re.search(pattern, path.read_text(encoding="utf-8"), flags=re.MULTILINE)
    return match.group(1) if match else None


def display_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


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
