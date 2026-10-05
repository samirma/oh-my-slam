"""``client.replay``: responses recorded with their files, replayed by request (path fields aside),
each once, with fresh copies of the files; anything not recorded is forwarded (None here)."""

from __future__ import annotations

from pathlib import Path

import pytest

from oh_my_slam.client import replay


def test_record_then_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rec = tmp_path / "rec"
    depth = tmp_path / "server" / "depth.npy"
    depth.parent.mkdir()
    depth.write_bytes(b"depth bytes")
    monkeypatch.setenv(replay.ENV_RECORD, str(rec))
    request = {"image_path": "/tmp/a1.jpg", "out_dir": "/tmp/x", "max_side": 1024}
    replay.record("/v1/geometry", request, {"width": 4, "depth_path": str(depth)})
    replay.record("/v1/geometry", request, {"width": 5, "depth_path": str(depth)})
    depth.unlink()  # the server's scratch file is gone; the recording kept a copy
    monkeypatch.delenv(replay.ENV_RECORD)
    monkeypatch.setenv(replay.ENV_REPLAY, str(rec))
    other_paths = {"image_path": "/tmp/b2.jpg", "out_dir": str(tmp_path / "out"),
                   "max_side": 1024}
    first = replay.replay("/v1/geometry", other_paths, other_paths["out_dir"])
    assert first["width"] == 4 and Path(first["depth_path"]).read_bytes() == b"depth bytes"
    assert Path(first["depth_path"]).parent == tmp_path / "out"
    second = replay.replay("/v1/geometry", other_paths, None)
    assert second is not None and second["width"] == 5  # in order, once each
    assert replay.replay("/v1/geometry", other_paths, None) is None  # used up: forwarded
    assert replay.replay("/v1/geometry", {**other_paths, "max_side": 512}, None) is None
    assert replay.replay("/v1/gravity", {"image_path": "x"}, None) is None
    monkeypatch.setenv(replay.ENV_REPLAY, str(tmp_path / "absent"))
    assert replay.replay("/v1/segment", {}, None) is None  # no recording: all forwarded
