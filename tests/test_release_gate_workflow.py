from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_release_gate_workflow_is_manual_only() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release-gate.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "push:" not in workflow
    assert "pull_request:" not in workflow
    assert "make verify-benchmarks" in workflow
    assert "make verify-v1-candidate" in workflow
    assert "make verify-web" in workflow
    assert "actions/setup-node" in workflow
    assert "permissions:" in workflow
    assert "contents: read" in workflow


def test_makefile_includes_artifact_schema_and_audit_targets() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "check-artifact-schema:" in makefile
    assert "audit-repository:" in makefile
    assert "verify-web:" in makefile
    assert "npm test" in makefile
    assert "npm run build" in makefile
    assert "check-artifact-schema" in makefile.split("verify-core:", 1)[1].splitlines()[0]
