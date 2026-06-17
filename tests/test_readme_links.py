from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_local_links_exist() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", readme)
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme)
    local_targets = [target for target in links + images if _is_local_target(target)]

    assert local_targets, "README should contain local documentation or asset links"
    missing = []
    for target in local_targets:
        path_part = target.split("#", 1)[0]
        if path_part:
            candidate = (ROOT / path_part).resolve()
            if not candidate.exists():
                missing.append(target)

    assert missing == []


def _is_local_target(target: str) -> bool:
    return not (
        target.startswith("http://")
        or target.startswith("https://")
        or target.startswith("mailto:")
        or target.startswith("#")
    )
