"""Check lightweight packaging metadata and optional build sanity."""

from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]

Runner = Callable[..., subprocess.CompletedProcess[str]]

REQUIRED_OPTIONAL_GROUPS = ("dev", "vis", "bench")
IGNORED_COPY_PATTERNS = (
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "outputs",
    "results",
    "examples/demo_data",
    "benchmark_results",
    "htmlcov",
    "dist",
    "build",
    "*.egg-info",
)


@dataclass(frozen=True, slots=True)
class PackagingMetadata:
    """Subset of pyproject metadata needed for release sanity checks."""

    name: str
    version: str
    requires_python: str
    scripts: dict[str, str] = field(default_factory=dict)
    optional_dependencies: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PackagingCheck:
    """Result of packaging metadata and optional build checks."""

    checked_files: list[Path]
    issues: list[str]
    warnings: list[str]
    built_artifacts: list[Path]

    @property
    def success(self) -> bool:
        return not self.issues


def run_packaging_checks(
    root: str | Path = ROOT,
    runner: Runner = subprocess.run,
    build_available: bool | None = None,
) -> PackagingCheck:
    """Run metadata checks and build sdist/wheel when the build module is available."""

    repo = Path(root).resolve()
    pyproject = repo / "pyproject.toml"
    checked = [pyproject]
    issues: list[str] = []
    warnings: list[str] = []
    built_artifacts: list[Path] = []

    if not pyproject.exists():
        return PackagingCheck(checked, ["pyproject.toml: missing file"], warnings, built_artifacts)

    metadata = load_packaging_metadata(pyproject)
    issues.extend(check_packaging_metadata(metadata))

    if build_available is None:
        build_available = importlib.util.find_spec("build") is not None
    if not build_available:
        warnings.append("Python module `build` is not installed; package build smoke skipped.")
    else:
        build_issues, artifacts = build_package_smoke(repo, runner=runner)
        issues.extend(build_issues)
        built_artifacts.extend(artifacts)

    return PackagingCheck(checked, sorted(issues), sorted(warnings), sorted(built_artifacts))


def load_packaging_metadata(path: Path) -> PackagingMetadata:
    """Load package metadata with stdlib TOML support or a small Python 3.10 fallback."""

    text = path.read_text(encoding="utf-8")
    try:
        import tomllib

        payload = tomllib.loads(text)
    except ModuleNotFoundError:
        payload = _parse_pyproject_fallback(text)

    project = payload.get("project", {})
    scripts = project.get("scripts", payload.get("project.scripts", {}))
    optional = project.get(
        "optional-dependencies",
        payload.get("project.optional-dependencies", {}),
    )
    return PackagingMetadata(
        name=str(project.get("name", "")),
        version=str(project.get("version", "")),
        requires_python=str(project.get("requires-python", "")),
        scripts={str(key): str(value) for key, value in dict(scripts).items()},
        optional_dependencies={
            str(key): list(value) if isinstance(value, list) else []
            for key, value in dict(optional).items()
        },
    )


def check_packaging_metadata(metadata: PackagingMetadata) -> list[str]:
    """Return packaging metadata issues."""

    issues: list[str] = []
    if metadata.name != "pointcloud-geolab":
        issues.append(f"pyproject.toml: unexpected project name `{metadata.name}`")
    if not re.fullmatch(r"\d+\.\d+\.\d+", metadata.version):
        issues.append("pyproject.toml: project.version must be semantic x.y.z")
    if not metadata.requires_python:
        issues.append("pyproject.toml: missing requires-python")
    if metadata.scripts.get("pointcloud-geolab") != "pointcloud_geolab.cli:main":
        issues.append("pyproject.toml: missing pointcloud-geolab console script")
    for group in REQUIRED_OPTIONAL_GROUPS:
        if group not in metadata.optional_dependencies:
            issues.append(f"pyproject.toml: missing optional dependency group `{group}`")
    return issues


def build_package_smoke(
    root: Path,
    runner: Runner = subprocess.run,
) -> tuple[list[str], list[Path]]:
    """Build sdist and wheel in a temporary copy so dist/build outputs stay out of git."""

    issues: list[str] = []
    artifacts: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="pointcloud-geolab-build-") as tmp:
        tmp_root = Path(tmp)
        source_copy = tmp_root / "src"
        output_dir = tmp_root / "dist"
        shutil.copytree(
            root,
            source_copy,
            ignore=shutil.ignore_patterns(*IGNORED_COPY_PATTERNS),
        )
        command = [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(output_dir),
        ]
        completed = runner(
            command,
            cwd=source_copy,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip() or "unknown build failure"
            return [f"python -m build failed: {detail}"], artifacts
        artifacts = sorted(path for path in output_dir.iterdir() if path.is_file())
        if not any(path.suffix == ".whl" for path in artifacts):
            issues.append("python -m build did not produce a wheel")
        if not any(path.name.endswith(".tar.gz") for path in artifacts):
            issues.append("python -m build did not produce an sdist")
    return issues, artifacts


def _parse_pyproject_fallback(text: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "project": {},
        "project.scripts": {},
        "project.optional-dependencies": {},
    }
    section = ""
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[]")
            continue
        if "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        if section == "project" and key in {"name", "version", "requires-python"}:
            payload["project"][key] = _strip_string(value)
        elif section == "project.scripts":
            payload["project.scripts"][key] = _strip_string(value)
        elif section == "project.optional-dependencies":
            payload["project.optional-dependencies"][key] = []
    return payload


def _strip_string(value: str) -> str:
    return value.strip().strip('"').strip("'")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_packaging_checks(args.root)
    print(f"Checked {len(result.checked_files)} packaging files.")
    for warning in result.warnings:
        print(f"Warning: {warning}")
    if result.built_artifacts:
        print("Built package artifacts in a temporary directory:")
        for artifact in result.built_artifacts:
            print(f"- {artifact.name}")
    if result.issues:
        print("Packaging issues:")
        for issue in result.issues:
            print(f"- {issue}")
    else:
        print("Packaging checks passed.")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
