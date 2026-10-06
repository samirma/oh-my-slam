"""A map of fewer than three keyframes extended without a rebuild (``mapping.api._remap_small``):
SfM runs again over the stored and the new keyframes and the result is brought into the map's
frame. Real COLMAP on a one-photo map extended by a video, and the refusals with COLMAP's results
stood in for."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from oh_my_slam.core.errors import RegistrationError
from oh_my_slam.core.images import load_rgb, save_jpeg
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.mapping import api, store
from oh_my_slam.mapping.api import update
from oh_my_slam.mapping.store import MapReader
from tests.fakes.client import FakeClient
from tests.synth.mapping import add_frames, mapping_room, ring
from tests.synth.scene import look_at
from tests.unit.test_mapping_cov_api import FakeModel, nf
from tests.unit.test_mapping_e2e_video import write_video

K = Intrinsics(500.0, 500.0, 320.0, 240.0, 640, 480, "colmap")


class SmallSfm:
    def __init__(self, db: Path, connected: bool, models: list[FakeModel | None]) -> None:
        self.db = db
        self._connected = connected
        self.models = list(models)
        self.extracted: list[list[str]] = []

    def extract(self, names: list[str], prior: Any, video: bool = False) -> int:
        self.extracted.append(names)
        return 1

    def match_pairs(self, pairs: set[tuple[int, int]], names: dict[int, str]) -> int:
        return len(pairs)

    def connected(self, seeds: set[str], candidates: set[str]) -> set[str]:
        return set(candidates) if self._connected else set()

    def map_global(self, out: Path) -> FakeModel | None:
        return self.models.pop(0)

    def map_incremental(self, out: Path) -> FakeModel | None:
        return self.models.pop(0)


def _small_map(tmp_path: Path) -> tuple[SimpleNamespace, list[store.FrameRecord]]:
    root = tmp_path / "map"
    (root / "frames").mkdir(parents=True)
    (root / "per_frame" / "f000000").mkdir(parents=True)
    save_jpeg(np.full((480, 640, 3), 120, np.uint8), root / "frames" / "f000000.jpg")
    np.save(root / store.frame_file("f000000", "depth.npy"), np.full((48, 64), 2.0, np.float16))
    old = [store.FrameRecord(0, "f000000", "frames/f000000.jpg", "", 1, 640, 480, K,
                             Pose.identity(), 64, 48, pose_source="identity")]

    def stage(rel: str) -> Path:
        p = tmp_path / "staging" / rel
        p.mkdir(parents=True, exist_ok=True)
        return p

    return SimpleNamespace(root=root, stage=stage, current=lambda rel: root / rel), old


def _ctx(tmp_path: Path) -> Any:
    tx, old = _small_map(tmp_path)
    return SimpleNamespace(tx=tx, old_frames=old, new=[nf(1), nf(2)], notes={}, work=tmp_path)


def test_the_new_keyframes_must_overlap_the_small_map(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    sfm = SmallSfm(tmp_path / "db.db", connected=False, models=[])
    with pytest.raises(RegistrationError, match="none of the input frames overlaps"):
        api._remap_small(ctx, sfm, {"f000001.jpg", "f000002.jpg"}, None, lambda m: None, 0.0)  # type: ignore[arg-type]
    assert sfm.extracted == [["f000000.jpg"]]  # the stored keyframe's features, from its image
    assert (tmp_path / "staging" / "frames" / "f000000.jpg").exists()


def test_the_small_maps_keyframes_must_be_posed_again(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    only_new = FakeModel({"f000001.jpg": Pose.identity(), "f000002.jpg": Pose.identity()})
    sfm = SmallSfm(tmp_path / "db.db", connected=True, models=[None, only_new])
    with pytest.raises(RegistrationError, match="could not be re-posed"):
        api._remap_small(ctx, sfm, {"f000001.jpg", "f000002.jpg"}, None, lambda m: None, 0.0)  # type: ignore[arg-type]


def test_without_a_metric_scale_the_re_posed_map_keeps_the_sfm_units(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_scale(m: Any, f: Any) -> Any:
        raise ValueError("no scale")

    monkeypatch.setattr(api.mframe, "metric_scale", no_scale)
    ctx = _ctx(tmp_path)
    sfm_pose = look_at(np.array([1.0, 2.0, 0.5]), np.array([0.0, 0.0, 0.0]))
    step = Pose(sfm_pose.R, sfm_pose.t + sfm_pose.R @ np.array([0.3, 0.0, 0.0]))
    model = FakeModel({"f000000.jpg": sfm_pose, "f000001.jpg": step})
    sfm = SmallSfm(tmp_path / "db.db", connected=True, models=[None, model])
    msgs: list[str] = []
    out = api._remap_small(ctx, sfm, {"f000001.jpg", "f000002.jpg"}, None, msgs.append, 0.0)  # type: ignore[arg-type]
    assert out is model and ctx.notes["remap_small"] == {"old_frames": 1, "scale": 1.0}
    np.testing.assert_allclose(model.pose("f000000.jpg").matrix(), np.eye(4), atol=1e-9)
    np.testing.assert_allclose(model.pose("f000001.jpg").t, [0.3, 0.0, 0.0], atol=1e-9)
    assert msgs and "re-mapped the 1-keyframe map" in msgs[0]


def test_a_stored_keyframe_staged_and_in_the_database_is_not_extracted_again(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pycolmap

    ctx = _ctx(tmp_path)
    staged = ctx.tx.stage("frames") / "f000000.jpg"
    staged.write_bytes(b"staged before")
    db = pycolmap.Database.open(str(tmp_path / "db.db"))
    try:
        cam = db.write_camera(pycolmap.Camera.create_from_model_id(
            0, pycolmap.CameraModelId.SIMPLE_PINHOLE, 500.0, 640, 480))
        db.write_image(pycolmap.Image(name="f000000.jpg", camera_id=cam))
    finally:
        db.close()
    monkeypatch.setattr(api.mframe, "metric_scale",
                        lambda m, f: api.mframe.ScaleResult(2.0, 0.0, {}, {}))
    model = FakeModel({"f000000.jpg": Pose.identity(), "f000001.jpg": Pose.identity()})
    sfm = SmallSfm(tmp_path / "db.db", connected=True, models=[model])
    assert api._remap_small(ctx, sfm, {"f000001.jpg", "f000002.jpg"}, None, lambda m: None,  # type: ignore[arg-type]
                            0.0) is model
    assert sfm.extracted == [] and staged.read_bytes() == b"staged before"
    assert ctx.notes["remap_small"] == {"old_frames": 1, "scale": 2.0}


@pytest.mark.skipif(shutil.which("colmap") is None, reason="needs Homebrew colmap")
def test_a_one_photo_map_extended_by_a_video_is_mapped_again_in_its_frame(tmp_path: Path) -> None:
    """A video is not a rebuild's input (only photos are): the map's one keyframe and the
    video's are posed together and brought into the map frame, which stays at the photo."""
    client = FakeClient()
    poses = ring(10, span=np.pi)
    imgs = add_frames(client, mapping_room(), poses, tmp_path / "in", "p", depth_noise=0.02,
                      seed=7)
    mdir = tmp_path / "map"
    update(mdir, [imgs[0]], client=client, progress=lambda m: None)
    T0 = MapReader(mdir).frames[0].T_map_cam
    video = tmp_path / "walk.mp4"
    write_video(video, [load_rgb(p) for p in imgs[1:]], fps=1)
    res = update(mdir, [video], fps=1.0, client=client, progress=lambda m: None)
    frames = MapReader(mdir).frames
    assert len(frames) >= 8 and res.new_frames == [f.name for f in frames[1:]]
    np.testing.assert_allclose(frames[0].T_map_cam.matrix(), T0.matrix(), atol=1e-9)
    d_est = np.linalg.norm(frames[1].T_map_cam.t - frames[0].T_map_cam.t)
    d_true = np.linalg.norm(poses[1].t - poses[0].t)
    assert d_est == pytest.approx(d_true, rel=0.15)
    last = json.loads((mdir / "map.json").read_text())["updates"][-1]
    assert last["kind"] == "video" and last["notes"]["remap_small"]["old_frames"] == 1
    assert "restarted" not in last["notes"]
