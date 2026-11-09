"""Model files from the Hugging Face cache (``server.models.hub``), offline."""

from __future__ import annotations

from typing import Any

import huggingface_hub
import pytest

from oh_my_slam.server.models.hub import cached_file


def test_a_cached_file_is_read_offline_at_its_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, ...]] = []

    def download(repo: str, filename: str, **kw: Any) -> str:
        calls.append((repo, filename, kw))
        return f"/cache/{repo}/{kw['revision']}/{filename}"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert cached_file("org/m", "model.pt", "abc") == "/cache/org/m/abc/model.pt"
    assert calls == [("org/m", "model.pt", {"revision": "abc", "local_files_only": True})]


def test_a_file_not_cached_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*a: Any, **kw: Any) -> str:
        raise huggingface_hub.errors.LocalEntryNotFoundError("not cached")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", missing)
    assert cached_file("org/m", "model.pt") is None
