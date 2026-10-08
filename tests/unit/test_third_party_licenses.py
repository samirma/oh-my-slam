"""Spec §4: every third-party component and its licence is listed in ``THIRD_PARTY_LICENSES.md``.
Every package ``uv.lock`` resolves (runtime and development groups) has a row there."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _listed() -> set[str]:
    """The first word of the first cell of every table row (the package or component name)."""
    rows = set()
    for line in (REPO / "THIRD_PARTY_LICENSES.md").read_text("utf-8").splitlines():
        if line.startswith("| ") and not line.startswith("|---"):
            rows.add(_norm(line.split("|")[1].strip().split(" ")[0]))
    return rows


def test_every_locked_package_has_a_licence_row() -> None:
    lock = tomllib.loads((REPO / "uv.lock").read_text("utf-8"))
    project = tomllib.loads((REPO / "pyproject.toml").read_text("utf-8"))["project"]["name"]
    locked = {_norm(p["name"]) for p in lock["package"]} - {_norm(project)}
    assert locked, "uv.lock lists no package"
    assert sorted(locked - _listed()) == []
