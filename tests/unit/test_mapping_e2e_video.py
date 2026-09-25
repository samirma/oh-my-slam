"""A video whose global SfM collapsed a stretch of keyframes onto one camera centre — what COLMAP's
global mapper does to a part of the view graph that hangs on the rest by one weak link: the
keyframes keep their orientations, share one centre and have no triangulated points. They must not
be accepted as posed; they are placed by multi-view poses anchored on their capture-order
neighbours and refined with the matches and the depth, and the trajectory stays continuous.
Real COLMAP on a rendered room, fake inference."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oh_my_slam.core.geometry import umeyama
from oh_my_slam.core.images import load_rgb
from oh_my_slam.mapping import sfm as sfm_mod
from oh_my_slam.mapping import trajectory as traj
from oh_my_slam.mapping.api import update
from oh_my_slam.mapping.store import MapReader
from tests.fakes.client import FakeClient
from tests.synth.mapping import add_frames, mapping_room, ring

pytestmark = pytest.mark.skipif(shutil.which("colmap") is None, reason="needs Homebrew colmap")


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    import av

    h, w = frames[0].shape[:2]
    with av.open(str(path), "w") as out:
        stream = out.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        stream.bit_rate = 8_000_000
        for img in frames:
            for pkt in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(pkt)
        for pkt in stream.encode():
            out.mux(pkt)


def collapse(rec: Any, names: list[str]) -> None:
    """Shrink ``names`` onto the centre of the first (orientations kept, observations gone)."""
    import pycolmap

    centre = np.asarray(rec.find_image_with_name(names[0]).projection_center())
    for n in names:
        im = rec.find_image_with_name(n)
        for idx, p2 in enumerate(im.points2D):
            if p2.has_point3D():
                rec.delete_observation(im.image_id, idx)
        im = rec.find_image_with_name(n)
        R = np.asarray(im.cam_from_world().rotation.matrix())
        rec.frame(im.frame_id).set_cam_from_world(
            im.camera_id, pycolmap.Rigid3d(pycolmap.Rotation3d(R), -R @ centre))


def misscale(rec: Any, block: list[str], pivot: str, factor: float) -> None:
    """Scale ``block`` (keyframes and the points they see) by ``factor`` about ``pivot``'s centre,
    as the global mapper leaves a part of the view graph hanging on the rest by one keyframe: a
    point also seen outside the block splits in two (the block's observations follow the block)."""
    import pycolmap

    centre = np.asarray(rec.find_image_with_name(pivot).projection_center())
    ids = {rec.find_image_with_name(n).image_id for n in block}
    for pid in list(rec.point3D_ids()):
        p = rec.point3D(pid)
        els = [(el.image_id, el.point2D_idx) for el in p.track.elements]
        mine = [e for e in els if e[0] in ids]
        if not mine:
            continue
        xyz = centre + factor * (np.asarray(p.xyz) - centre)
        color = np.asarray(p.color)
        for e in mine:
            if rec.exists_point3D(pid):
                rec.delete_observation(*e)
        if len(mine) >= 2:
            track = pycolmap.Track()
            for e in mine:
                track.add_element(*e)
            rec.add_point3D(xyz, track, color)
    for n in block:
        im = rec.find_image_with_name(n)
        R = np.asarray(im.cam_from_world().rotation.matrix())
        c = centre + factor * (np.asarray(im.projection_center()) - centre)
        rec.frame(im.frame_id).set_cam_from_world(
            im.camera_id, pycolmap.Rigid3d(pycolmap.Rotation3d(R), -R @ c))


def angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return traj.rotation_deg(Ra, Rb)


def video_world(tmp_path: Path) -> tuple[FakeClient, list[Any], Path]:
    """A walk around a rendered room, filmed at one frame per second (fake inference)."""
    client = FakeClient(mv_noise=(2.0, 0.1))
    truth = ring(18)
    renders = add_frames(client, mapping_room(), truth, tmp_path / "renders", "r",
                         depth_noise=0.02, seed=3)
    video = tmp_path / "walk.mp4"
    write_video(video, [load_rgb(p) for p in renders], fps=1)
    return client, truth, video


def assert_follows_truth(mdir: Path, truth: list[Any]) -> None:
    """Every keyframe within 8 cm and 1.5° of the truth (after a similarity), in metres, and the
    walk continuous (consecutive steps like the rest of the ring)."""
    frames = MapReader(mdir).frames
    assert [f.name for f in frames] == [f"f{k:06d}" for k in range(len(truth))]
    sim = umeyama(np.array([f.T_map_cam.t for f in frames]), np.array([T.t for T in truth]))
    assert sim.s == pytest.approx(1.0, abs=0.1)  # metric map
    for f, T in zip(frames, truth, strict=True):
        centre = sim.apply(f.T_map_cam.t[None])[0]
        assert np.linalg.norm(centre - T.t) < 0.08, f.name
        assert angle_deg(sim.R @ f.T_map_cam.R, T.R) < 1.5, f.name
    poses = {f.name: f.T_map_cam for f in frames}
    steps = traj.steps(poses, sorted(poses))
    assert steps.min() > 0.5 * np.median(steps)
    assert steps.max() < 1.5 * np.median(steps)


@pytest.mark.parametrize("by", ["incremental", "multiview"])
def test_collapsed_video_segment_is_placed_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                 by: str) -> None:
    """``incremental``: the incremental mapper registers the keyframes again against the fixed
    global model; ``multiview``: it cannot (as with the user's kitchen, joined by one weak link),
    so they get multi-view poses anchored on their capture-order neighbours, then refined."""
    client, truth, video = video_world(tmp_path)
    victims = [f"f{k:06d}.jpg" for k in range(6, 11)]
    global_mapping = sfm_mod.Sfm.map_global

    def collapsing(self: sfm_mod.Sfm, out: Path) -> sfm_mod.SfmModel | None:
        model = global_mapping(self, out)
        assert model is not None and set(victims) <= set(model.registered)
        collapse(model.rec, victims)
        return model

    monkeypatch.setattr(sfm_mod.Sfm, "map_global", collapsing)
    if by == "multiview":
        incremental = sfm_mod.Sfm.map_incremental

        def no_completion(self: sfm_mod.Sfm, out: Path, input_path: Path | None = None,
                          fix_existing: bool = False) -> sfm_mod.SfmModel | None:
            return None if input_path else incremental(self, out)

        monkeypatch.setattr(sfm_mod.Sfm, "map_incremental", no_completion)
    mdir = tmp_path / "map"
    msgs: list[str] = []
    update(mdir, [video], fps=1.0, client=client, progress=msgs.append)

    update_ = json.loads((mdir / "map.json").read_text())["updates"][-1]
    notes = update_["notes"]
    assert notes["sfm_unsupported"]["sfm-global"]["collapsed"] == victims
    if by == "multiview":
        assert any("anchored in capture order" in m for m in msgs), msgs
        assert notes["sfm_join"]["multiview"] == victims
        assert notes["sfm_join"]["collapsed_rejected"] == []
        assert update_["sfm"] == "sfm-global+multiview"
    else:
        assert update_["sfm"] == "sfm-global+incremental"
    assert_follows_truth(mdir, truth)


def test_misscaled_video_segment_is_rescaled(tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """The global mapper returned keyframes 7-11 (and the points they see) at 1/3 of their size
    about keyframe 6: their depth ratios disagree with the rest, so the block is scaled back about
    the keyframe it hangs on and refined with the matches and the depth."""
    client, truth, video = video_world(tmp_path)
    block = [f"f{k:06d}.jpg" for k in range(7, 12)]
    global_mapping = sfm_mod.Sfm.map_global

    def misscaling(self: sfm_mod.Sfm, out: Path) -> sfm_mod.SfmModel | None:
        model = global_mapping(self, out)
        assert model is not None and set(block) <= set(model.registered)
        misscale(model.rec, block, "f000006.jpg", 1 / 3)
        return model

    monkeypatch.setattr(sfm_mod.Sfm, "map_global", misscaling)
    mdir = tmp_path / "map"
    update(mdir, [video], fps=1.0, client=client, progress=lambda m: None)
    update_ = json.loads((mdir / "map.json").read_text())["updates"][-1]
    moves = update_["notes"]["sfm_join"]["fixed_blocks"]
    assert moves[0]["keyframes"] == block
    assert moves[0]["factor"] == pytest.approx(3.0, rel=0.1)
    assert moves[0]["tilt_deg"] == 0.0
    assert update_["sfm"] == "sfm-global+realigned"
    assert_follows_truth(mdir, truth)
