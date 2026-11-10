"""``SKILL.md`` is generated from the API definitions and never stale."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from oh_my_slam.service import skill
from oh_my_slam.service.routes import ENDPOINTS


def test_skill_md_is_current() -> None:
    text = skill.FILE.read_text("utf-8")
    assert text == skill.render(), "SKILL.md is stale: run python -m oh_my_slam.service.skill"
    for e in ENDPOINTS:
        assert f"`{e.method} {e.path}`" in text


def test_the_generator_writes_the_file(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(skill, "FILE", tmp_path / "SKILL.md")
    assert skill.main([]) == 0 and (tmp_path / "SKILL.md").read_text() == skill.render()
    r = subprocess.run([sys.executable, "-m", "oh_my_slam.service.skill"], capture_output=True,
                       timeout=60)
    assert r.returncode == 0
