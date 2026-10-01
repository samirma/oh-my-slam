"""How the mapper schedules its work around the inference server, without changing its results:
keyframes go to inference as ingest writes them, and a new map's SIFT features are extracted while
their inference runs, for a camera that gets its prior focal length afterwards."""

from __future__ import annotations

import shutil
import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from oh_my_slam.mapping import api, ingest
from oh_my_slam.mapping.sfm import CameraPrior, Sfm


def _textured(path: Path, seed: int, w: int = 480, h: int = 360) -> None:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (h // 8, w // 8, 3), dtype=np.uint8)
    img = Image.fromarray(small).resize((w, h), Image.Resampling.BICUBIC)
    Image.fromarray(np.asarray(img)).save(path, quality=95)


def _database(db: Path) -> dict[str, Any]:
    """A COLMAP database's content by image name: the extractor numbers the images in the order
    its threads finish them, which varies from run to run."""
    con = sqlite3.connect(db)
    try:
        def rows(sql: str) -> list[tuple[Any, ...]]:
            return sorted(con.execute(sql).fetchall(), key=repr)

        name = dict(con.execute("SELECT image_id, name FROM images").fetchall())
        rig = dict(con.execute("SELECT frame_id, rig_id FROM frames").fetchall())
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        out: dict[str, Any] = {
            "cameras": rows("SELECT * FROM cameras"),
            "rigs": rows("SELECT * FROM rigs"),
            "rig_sensors": rows("SELECT * FROM rig_sensors"),
            "images": rows("SELECT name, camera_id FROM images"),
            "frame_data": sorted((name[d], rig[f], s, t) for f, d, s, t in con.execute(
                "SELECT frame_id, data_id, sensor_id, sensor_type FROM frame_data")),
        }
        for t in ("keypoints", "descriptors"):
            out[t] = sorted((name[r[0]], *r[1:]) for r in con.execute(f"SELECT * FROM {t}"))
        for t in ("pose_priors", "matches", "two_view_geometries"):
            out[t] = rows(f"SELECT * FROM {t}")
        assert tables - {"sqlite_sequence", "frames"} == set(out), tables
        return out
    finally:
        con.close()


@pytest.mark.skipif(shutil.which("colmap") is None, reason="needs Homebrew colmap")
def test_features_before_the_prior_give_the_same_database(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    names = []
    for i in range(3):
        _textured(frames / f"f{i:06d}.jpg", i)
        names.append(f"f{i:06d}.jpg")
    prior = CameraPrior(480, 360, focal=401.23456)
    after = Sfm(tmp_path / "after.db", frames, tmp_path / "w1")
    after.extract(names, prior)
    early = Sfm(tmp_path / "early.db", frames, tmp_path / "w2")
    camera = early.extract(names, CameraPrior(480, 360, focal=480.0))
    early.set_prior(camera, prior)
    a, b = _database(tmp_path / "after.db"), _database(tmp_path / "early.db")
    assert a["cameras"] and len(a["keypoints"]) == 3
    for name in a:
        assert a[name] == b[name], name


def _keyframes(tmp_path: Path, n: int, fail_at: int | None = None) -> Iterator[ingest.Keyframe]:
    for i in range(n):
        if i == fail_at:
            raise RuntimeError("bad video")
        path = tmp_path / f"f{i:06d}.jpg"
        _textured(path, i, 64, 48)
        yield ingest.Keyframe(path.stem, i, path, f"clip@{i}", None)


def test_inference_starts_while_ingest_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch
                                              ) -> None:
    started: list[int] = []
    first_done = threading.Event()
    seen_written: list[list[str]] = []

    def fake(kf: ingest.Keyframe, work: Path, client: Any) -> tuple[Any, list[Any]]:
        started.append(kf.index)
        first_done.set()
        return kf.index, []

    def slow_keyframes() -> Iterator[ingest.Keyframe]:
        for kf in _keyframes(tmp_path, 4):
            yield kf
            if kf.index == 0:  # the next keyframe is written only once the first is in inference
                assert first_done.wait(10)

    monkeypatch.setattr(api, "reconstruct_and_detect_keyframe", fake)
    out = api._infer_frames(slow_keyframes(), "video", tmp_path, None, lambda m: None,
                            lambda w: seen_written.append([k.name for k in w]))
    assert [nf.frame for nf in out] == [0, 1, 2, 3]  # capture order
    assert seen_written == [[f"f{i:06d}" for i in range(4)]]


def test_an_ingest_error_stops_the_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch
                                             ) -> None:
    def fake(kf: ingest.Keyframe, work: Path, client: Any) -> tuple[Any, list[Any]]:
        return kf.index, []

    monkeypatch.setattr(api, "reconstruct_and_detect_keyframe", fake)
    with pytest.raises(RuntimeError, match="bad video"):
        api._infer_frames(_keyframes(tmp_path, 5, fail_at=3), "video", tmp_path, None,
                          lambda m: None)
