"""Print a release-review audit snapshot for the repository."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_repo_hygiene import GENERATED_PREFIXES, regex_value

Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class RepositoryAudit:
    """Repository audit data suitable for text rendering."""

    version: str | None
    git_status: str
    ahead_behind: str
    current_tag: str
    release_summary: str
    issue_summary: str
    ci_summary: str
    coverage_gate: str | None
    generated_tracked: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return not self.generated_tracked


def collect_audit(
    root: str | Path = ROOT,
    runner: Runner = subprocess.run,
    gh_available: bool | None = None,
) -> RepositoryAudit:
    """Collect local and optional GitHub repository audit information."""

    repo = Path(root).resolve()
    warnings: list[str] = []
    version = regex_value(repo / "pyproject.toml", r'^version\s*=\s*"([^"]+)"')
    git_status = _run_text(
        ["git", "-C", str(repo), "status", "--short", "--untracked-files=all"],
        runner,
    )
    ahead_behind = _run_text(
        ["git", "-C", str(repo), "rev-list", "--left-right", "--count", "HEAD...origin/main"],
        runner,
    )
    tag_name = f"v{version}" if version else ""
    current_tag = (
        _run_text(["git", "-C", str(repo), "tag", "-l", tag_name], runner) if tag_name else ""
    )
    generated = _run_text(["git", "-C", str(repo), "ls-files", "--", *GENERATED_PREFIXES], runner)
    generated_tracked = [line for line in generated.splitlines() if line.strip()]
    coverage_gate = regex_value(repo / "pyproject.toml", r"fail_under\s*=\s*(\d+)")

    if gh_available is None:
        gh_available = shutil.which("gh") is not None
    if gh_available:
        release_summary = _run_text(["gh", "release", "view", tag_name], runner, limit=800)
        issue_summary = _run_text(["gh", "issue", "list", "--state", "open"], runner, limit=800)
        ci_summary = _run_text(
            ["gh", "run", "list", "--branch", "main", "--workflow", "tests.yml", "--limit", "5"],
            runner,
            limit=800,
        )
    else:
        release_summary = "skipped: GitHub CLI unavailable"
        issue_summary = "skipped: GitHub CLI unavailable"
        ci_summary = "skipped: GitHub CLI unavailable"
        warnings.append("GitHub CLI unavailable; release, issue, and CI checks were skipped.")

    return RepositoryAudit(
        version=version,
        git_status=git_status,
        ahead_behind=ahead_behind,
        current_tag=current_tag,
        release_summary=release_summary,
        issue_summary=issue_summary,
        ci_summary=ci_summary,
        coverage_gate=coverage_gate,
        generated_tracked=generated_tracked,
        warnings=warnings,
    )


def format_audit(audit: RepositoryAudit) -> str:
    """Render an audit snapshot as Markdown-like text."""

    generated = "\n".join(f"- `{path}`" for path in audit.generated_tracked) or "- None"
    warnings = "\n".join(f"- {warning}" for warning in audit.warnings) or "- None"
    return "\n".join(
        [
            "# Repository Audit Snapshot",
            "",
            f"- Version: `{audit.version or 'unknown'}`",
            f"- Current tag lookup: `{audit.current_tag or 'not found'}`",
            f"- Ahead/behind HEAD...origin/main: `{audit.ahead_behind or 'unknown'}`",
            f"- Working tree: `{'clean' if not audit.git_status.strip() else 'dirty'}`",
            f"- Coverage gate: `{audit.coverage_gate or 'unknown'}%`",
            "",
            "## Generated Artifact Guard",
            "",
            generated,
            "",
            "## Optional Dependencies Policy",
            "",
            "- Open3D, Plotly, laspy, PyTorch, SciPy, scikit-learn, and pandas are optional.",
            "- Core tests skip unavailable optional paths instead of installing CUDA or ML stacks.",
            "",
            "## GitHub Release",
            "",
            _indent(audit.release_summary or "no release output"),
            "",
            "## Open Issues",
            "",
            _indent(audit.issue_summary or "no issue output"),
            "",
            "## CI",
            "",
            _indent(audit.ci_summary or "no CI output"),
            "",
            "## Limitations",
            "",
            "- No full nonlinear GICP optimizer.",
            "- No SLAM backend, CUDA acceleration, PointNet training release, "
            "or real KITTI benchmark.",
            "- Synthetic demos and tiny fixtures remain smoke checks, not real-data benchmarks.",
            "",
            "## Warnings",
            "",
            warnings,
            "",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audit = collect_audit(args.root)
    print(format_audit(audit))
    return 0 if audit.success else 1


def _run_text(command: list[str], runner: Runner, limit: int | None = None) -> str:
    completed = runner(command, text=True, capture_output=True, check=False)
    output = completed.stdout.strip() if completed.returncode == 0 else completed.stderr.strip()
    if limit is not None and len(output) > limit:
        return output[:limit].rstrip() + "\n..."
    return output


def _indent(text: str) -> str:
    return "\n".join(f"    {line}" if line else "" for line in text.splitlines())


if __name__ == "__main__":
    raise SystemExit(main())
