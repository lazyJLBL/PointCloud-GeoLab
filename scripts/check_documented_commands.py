"""Check documented command entry points without generating large artifacts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class CommandCheck:
    """Result for one documented command smoke check."""

    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.returncode == 0 and "usage:" in self.stdout.lower()


@dataclass(frozen=True, slots=True)
class StaticCommandCheck:
    """Result for documented commands that should exist but not be executed."""

    command: str
    location: str
    success: bool
    detail: str


def documented_help_commands(root: Path = ROOT) -> list[list[str]]:
    """Return stable documented commands that should expose help text."""

    python = sys.executable
    return [
        [python, "-m", "pointcloud_geolab", "--help"],
        [python, "-m", "pointcloud_geolab", "benchmark", "--help"],
        [python, "-m", "pointcloud_geolab", "pipeline", "--help"],
        [python, "-m", "pointcloud_geolab", "register", "--help"],
        [python, str(root / "examples" / "real_bunny_registration.py"), "--help"],
        [python, str(root / "examples" / "kitti_lidar_segmentation.py"), "--help"],
        [python, str(root / "examples" / "modelnet_primitive_demo.py"), "--help"],
        [python, str(root / "scripts" / "run_scale_benchmark.py"), "--help"],
        [python, str(root / "scripts" / "verify_realdata_workflow.py"), "--help"],
    ]


def documented_web_commands() -> tuple[str, ...]:
    """Return Web Console commands that docs and Makefile should expose."""

    return ("make web-backend", "make web-frontend", "make verify-web")


def run_documented_command_checks(
    root: str | Path = ROOT,
    runner: Runner = subprocess.run,
) -> list[CommandCheck]:
    """Run help-only checks for documented commands."""

    repo = Path(root).resolve()
    checks: list[CommandCheck] = []
    for command in documented_help_commands(repo):
        completed = runner(
            command,
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
        checks.append(
            CommandCheck(
                command=command,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        )
    return checks


def run_static_documented_command_checks(root: str | Path = ROOT) -> list[StaticCommandCheck]:
    """Check Web command references without starting long-running servers."""

    repo = Path(root).resolve()
    docs = [
        repo / "README.md",
        repo / "docs" / "web_console.md",
        repo / "web" / "README.md",
    ]
    makefile = repo / "Makefile"
    makefile_text = makefile.read_text(encoding="utf-8") if makefile.exists() else ""
    checks: list[StaticCommandCheck] = []
    for command in documented_web_commands():
        target = command.removeprefix("make ")
        for doc in docs:
            text = doc.read_text(encoding="utf-8") if doc.exists() else ""
            checks.append(
                StaticCommandCheck(
                    command=command,
                    location=doc.relative_to(repo).as_posix(),
                    success=command in text,
                    detail="documented" if command in text else "missing command in docs",
                )
            )
        checks.append(
            StaticCommandCheck(
                command=command,
                location="Makefile",
                success=f"{target}:" in makefile_text,
                detail=(
                    "target exists" if f"{target}:" in makefile_text else "missing Makefile target"
                ),
            )
        )
    return checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checks = run_documented_command_checks(args.root)
    static_checks = run_static_documented_command_checks(args.root)
    failures = [check for check in checks if not check.success]
    static_failures = [check for check in static_checks if not check.success]
    print(
        f"Checked {len(checks)} documented command help entries and "
        f"{len(static_checks)} static command references."
    )
    if failures or static_failures:
        print("Documented command checks failed:")
        for failure in failures:
            command = " ".join(failure.command)
            detail = (failure.stderr or failure.stdout).strip().splitlines()
            tail = detail[-1] if detail else "no output"
            print(f"- `{command}` returned {failure.returncode}: {tail}")
        for failure in static_failures:
            print(f"- `{failure.command}` in {failure.location}: {failure.detail}")
        return 1
    print("Documented command checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
