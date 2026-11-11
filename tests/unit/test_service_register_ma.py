"""The MapAnything engine of an anchored map's camera-only frames (``mapping.register_ma``,
``OH_MY_SLAM_NONE_ENGINE=mapanything``): end to end through the Starlette app on the ray-cast
room (``tests.synth.room``), MapAnything played by ``FakeBackend.geometry_views`` (true relative
poses at ``mv_scale``), and its edges on ``Registration`` itself."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from oh_my_slam.core.types import Intrinsics
from oh_my_slam.mapping import register_ma
from oh_my_slam.service import prior_first
from oh_my_slam.service.errors import ApiError
from oh_my_slam.service.update import FrameWork, UpdateWork
from tests.fakes.service import FakeBackend
from tests.synth.room import base_pose
from tests.unit.test_service_api import Client
from tests.unit.test_service_prior_first import CAM, frames_for, shot
from tests.unit.test_service_prior_first import client as _client  # noqa: F401

K = Intrinsics(160.0, 160.0, 160.0, 120.0, 320, 240)


@pytest.fixture
def client(_client: Client, monkeypatch: pytest.MonkeyPatch) -> Iterator[Client]:  # noqa: F811
    monkeypatch.setenv(prior_first.NONE_ENGINE_ENV, "mapanything")
    yield _client


def ma_frames(c: Client, poses: list[np.ndarray], **kw: Any
              ) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    """Frames whose MapAnything poses are their true ones."""
    fr, files = frames_for(c, poses, **kw)
    for f, T in zip(fr, poses, strict=True):
        c.backend.mv_add(files[f["image"]], T)
    return fr, files


def test_engine_switch_and_camera_only_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(prior_first.NONE_ENGINE_ENV, raising=False)
    assert prior_first.none_engine() == prior_first.DEFAULT_NONE_ENGINE
    w = UpdateWork("m", [FrameWork(0, "a", "cam", b""),
                         FrameWork(1, "b", "cam", b"", prior={"trust": "metric"}),
                         FrameWork(2, "c", "cam", b"", prior={"trust": "none"})])
    monkeypatch.setenv(prior_first.NONE_ENGINE_ENV, "MapAnything")
    assert prior_first.camera_only(w) == 2
    monkeypatch.setenv(prior_first.NONE_ENGINE_ENV, "geometric")
    assert prior_first.camera_only(w) == 0
    monkeypatch.setenv(prior_first.NONE_ENGINE_ENV, "other")
    with pytest.raises(ValueError, match="expected one of"):
        prior_first.none_engine()


def test_a_camera_only_map_turning_in_place(client: Client) -> None:
    """The origin by the prior-first engine, the turn by unposed MapAnything calls aligned on
    the map's keyframes: its metres are not the map's (``mv_scale``). One standpoint: the frames
    are registered as a pure rotation about the origin's centre (map.md §1), MapAnything's
    centres (moved 4 cm) snapped to it, its rotations kept."""
    client.backend.mv_scale = 3.0
    poses = [base_pose(25.0 * k, offset=0.0) for k in range(6)]
    fr, files = ma_frames(client, poses, trust=None, depth=False, height_above_floor_m=0.5)
    for T in poses[1:]:  # MapAnything places them off the standpoint
        bad = T.copy()
        bad[:3, 3] += np.array([0.04, 0.0, 0.0])
        client.backend.mv_pose[hashlib.sha256(shot(T)[0]).hexdigest()] = bad
    job = client.update("m", fr, files, cameras={"cam": CAM})
    assert job["state"] == "committed", job
    assert [i["status"] for i in job["inputs"]] == ["registered"] * 6
    assert len(client.backend.mv_calls) == 2  # 5 frames, 4 per call
    assert job["timings_s"]["register"] > 0
    T0 = np.array(job["inputs"][0]["T_map_cam"])
    for i, T in zip(job["inputs"][1:], poses[1:], strict=True):
        rel_true = np.linalg.inv(poses[0]) @ T
        rel = np.linalg.inv(T0) @ np.array(i["T_map_cam"])
        cos = (np.trace(rel_true[:3, :3].T @ rel[:3, :3]) - 1) / 2
        assert np.degrees(np.arccos(min(1.0, cos))) < 0.5
        assert np.linalg.norm(rel[:3, 3] - rel_true[:3, 3]) < 0.02
    v = client.service.store.version("m")
    assert {f["method"] for f in v["frames"][1:]} == {"rotation"}
    assert job["quality"]["failed_checks"] == []
    # map.md §1: a rotation-only frame counts its rotation only, here none
    q = job["quality"]
    assert q["max_displacement_m"] is None and q["max_displacement_deg"] == 0.0
    h = client.get("/api/health", "Health")["timings"]
    assert h["registration_mapanything_per_frame"]["count"] == 5
    assert h["checks_per_frame"]["source"] == "measured"


def test_camera_only_frames_on_a_map_with_priors(client: Client) -> None:
    """Posed calls (the anchors spread) on a map anchored by metric priors and depth: the
    camera-only frames land at their true poses at the map's scale, with or without a depth
    image; one MapAnything misplaces is not confirmed by the map's depth."""
    client.backend.mv_scale = 2.5
    fr, files = ma_frames(client, [base_pose(0, -0.6), base_pose(0, 0.0), base_pose(0, 0.6)],
                          tag="a")
    assert client.update("m", fr, files, cameras={"cam": CAM})["state"] == "committed"
    poses = [base_pose(0, -0.3), base_pose(0, 0.3)]
    nf, nfiles = ma_frames(client, poses, tag="n", trust=None, depth=False)
    df, dfiles = ma_frames(client, [base_pose(5, 0.1)], tag="d", trust=None)
    bad, bfiles = frames_for(client, [base_pose(0, 0.2)], tag="x", trust=None, depth=False)
    wrong = base_pose(180, 0.2)
    client.backend.mv_pose[hashlib.sha256(bfiles["x0"]).hexdigest()] = wrong
    job = client.update("m", nf + df + bad, {**nfiles, **dfiles, **bfiles})
    assert job["state"] == "committed", job
    st = [(i["status"], i["reason"]) for i in job["inputs"]]
    assert st == [("registered", None)] * 3 + [("rejected", "no_overlap")]
    for i, T in zip(job["inputs"][:3], [*poses, base_pose(5, 0.1)], strict=True):
        assert np.linalg.norm(np.array(i["T_map_cam"])[:3, 3] - T[:3, 3]) < 0.03
        assert i["prior_residual_m"] is None
    assert job["scale_source"] == "priors"  # the metric priors span 1.2 m
    # MapAnything frames carry a displacement (0: committed at their image-only pose)
    assert job["quality"]["max_displacement_m"] == 0.0
    assert job["quality"]["max_displacement_deg"] == 0.0


def test_camera_only_maps_record_depth_residuals(client: Client) -> None:
    """map.md §2: a camera-only map with depth images registered by MapAnything records each
    depth keyframe's residual against the map, so its scale_sigma_rel is no constant."""
    from oh_my_slam.mapping.anchored import AnchoredMap

    fr, files = ma_frames(client, [base_pose(0, 0.0)], tag="a", trust=None)
    assert client.update("m", fr, files, cameras={"cam": CAM})["state"] == "committed"
    nf, nfiles = ma_frames(client, [base_pose(0, -0.3), base_pose(0, 0.3)], tag="n", trust=None)
    job = client.update("m", nf, nfiles)
    assert [i["status"] for i in job["inputs"]] == ["registered"] * 2
    am = AnchoredMap.load(client.service.store.mapper_dir("m"))
    assert len(am.scale.depth_residuals.get("sensor", [])) >= 2


def test_too_close_and_a_new_map_without_origin(client: Client, tmp_path: Path) -> None:
    fr, files = ma_frames(client, [base_pose(0, -0.6), base_pose(0, 0.6)], tag="a")
    client.update("m", fr, files, cameras={"cam": CAM})
    nf, nfiles = ma_frames(client, [base_pose(0, 0.0)], tag="n", trust=None, depth=False)
    h = hashlib.sha256(nfiles["n0"]).hexdigest()
    client.backend.mono_depth[h] = np.full_like(client.backend.mono_depth[h], 0.1)
    job = client.update("m", nf, nfiles)
    assert (job["inputs"][0]["status"], job["inputs"][0]["reason"]) == ("rejected", "too_close")
    # nothing confirmed to register against: every camera-only frame is no_overlap
    from oh_my_slam.mapping.anchored import AnchoredMap

    am = AnchoredMap(tmp_path / "am")
    am.root.mkdir()
    work = type("W", (), {"name": "f", "index": 0})()
    got, reg_s, fuse_s = prior_first._register_none(
        None, am, [type("F", (), {"work": work})()], tmp_path, print)  # type: ignore[arg-type]
    assert [(p.status, p.reason) for p in got] == [("rejected", "no_overlap")] and fuse_s == 0.0


def test_out_of_memory_shrinks_the_calls(client: Client) -> None:
    fr, files = ma_frames(client, [base_pose(0, x) for x in (-0.9, -0.3, 0.3, 0.9)], tag="a")
    client.update("m", fr, files, cameras={"cam": CAM})
    client.backend.mv_oom = 4  # 4 anchors + 2, then + 1, then 2 anchors + 1
    nf, nfiles = ma_frames(client, [base_pose(0, -0.6), base_pose(0, 0.6)], tag="n", trust=None,
                           depth=False)
    job = client.update("m", nf, nfiles)
    assert [i["status"] for i in job["inputs"]] == ["registered"] * 2
    assert [len(c) for c in client.backend.mv_calls] == [6, 5, 3, 3]
    client.backend.mv_oom = 2  # no call fits
    nf, nfiles = ma_frames(client, [base_pose(0, 0.0)], tag="z", trust=None, depth=False)
    job = client.update("m", nf, nfiles)
    assert job["state"] == "failed" and job["error"]["code"] == "out_of_memory"


def _stored(tmp_path: Path, poses: list[np.ndarray], depth: bool = True
            ) -> list[register_ma.Stored]:
    out = []
    for i, T in enumerate(poses):
        img, _, z = shot(T)
        p = tmp_path / f"s{i}.jpg"
        p.write_bytes(img)
        out.append(register_ma.Stored(f"s{i}", p, K, T, i,
                                      (lambda z=z: z.astype(np.float32)) if depth
                                      else (lambda: None)))
    return out


def test_registration_edges(tmp_path: Path) -> None:
    fb = FakeBackend()
    stored = _stored(tmp_path, [base_pose(0, -0.6), base_pose(0, 0.6)], depth=False)
    new = []
    for i, T in enumerate([base_pose(0, 0.0), base_pose(0, 0.2)]):
        img, _, z = shot(T)
        p = tmp_path / f"n{i}.jpg"
        p.write_bytes(img)
        h = hashlib.sha256(img).hexdigest()
        fb.mv_pose[h], fb.mono_depth[h] = T, z
        new.append(register_ma.New(i, p, K, features=10 if i else 400))
    for s in stored:
        h = hashlib.sha256(s.image.read_bytes()).hexdigest()
        fb.mv_pose[h], fb.mono_depth[h] = s.T, shot(s.T)[2]
    fb.mv_pose[hashlib.sha256(new[1].image.read_bytes()).hexdigest()] = base_pose(120, 0.2)

    def mv(images: list[Path], Ks: list[Intrinsics], poses: list[Any]) -> Any:
        return fb.geometry_views(images, Ks, poses, tmp_path)

    reg = register_ma.Registration(mv, new, stored)
    res = reg.run()
    # no stored depth: MapAnything's metres, its own depths of the call as references
    assert res.notes["calls"][0]["scale_source"] == "mapanything"
    assert res.placements[0].status == "registered"
    # few_features is only the checks' verdict (api.md §4): unplaced, no_overlap
    assert (res.placements[1].status, res.placements[1].reason) == ("rejected", "no_overlap")
    assert res.calls == 1 and res.views == 4
    # tested again from its first output, never inferred again: nothing new confirms it; a
    # reference that sees its depth where it put it does (another engine's keyframe)
    assert reg.recheck() == [] and set(reg.waiting) == {1}
    _, T, _, depth, Kd, _ = reg.waiting[1]
    assert reg.recheck([register_ma.Ref(T, depth, Kd)]) == [1]
    assert reg.placements[1].status == "registered" and not reg.waiting and reg.calls == 1

    def down(images: list[Path], Ks: list[Intrinsics], poses: list[Any]) -> Any:
        raise ApiError("inference_unavailable", "down (test)")

    with pytest.raises(ApiError):
        register_ma.Registration(down, new[:1], stored).run()
    assert register_ma.median_depth(np.zeros((2, 2), np.float32)) is None
    assert register_ma.depth_ratio(np.zeros((4, 4), np.float32), np.ones((4, 4), np.float32)) \
        is None
    assert register_ma._thumb(tmp_path / "missing.jpg") is None
    import cv2

    flat = tmp_path / "flat.png"
    cv2.imwrite(str(flat), np.full((24, 32), 90, np.uint8))
    assert float(np.abs(register_ma._thumb(flat)).sum()) == 0.0  # type: ignore[arg-type]
    r = register_ma.Registration(mv, [register_ma.New(9, tmp_path / "missing.jpg", K)], stored)
    assert len(r.anchors(r.new)) == 2  # no thumbnail: by recency
    assert register_ma.baseline([np.eye(4)]) == 0.0


def test_camera_only_rotations_that_disagree_with_the_images_fail_the_hard_limit(
        client: Client) -> None:
    """map.md §4 rotation_inconsistent: an update without priors whose committed relative
    rotations of overlapping neighbours disagree with their two-view rotations (MapAnything
    turns frames that moved, not one standpoint) is rejected by the hard limit."""
    poses = [base_pose(0, 0.25 * k - 0.5, 0.0) for k in range(4)]
    fr, files = ma_frames(client, poses, trust=None, depth=False, height_above_floor_m=0.5)
    for k, T in enumerate(poses[1:], 1):
        bad = T.copy()
        bad[:3, :3] = cv2.Rodrigues(np.array([0.0, 0.0, 0.3 * (-1) ** k]))[0] @ T[:3, :3]
        client.backend.mv_pose[hashlib.sha256(shot(T)[0]).hexdigest()] = bad
    job = client.update("m", fr, files, cameras={"cam": CAM})
    assert job["state"] == "rejected", job
    assert job["quality"]["failed_checks"] == ["rotation_inconsistent"]
