"""§2.3 "the result is the same whether a sequence is mapped in one update or split across several in
the same order", for a split whose first update is weakly posed: real COLMAP on rendered views,
fake inference.

The first update is four views panned from one spot towards a wall whose middle is a window: the
scenery seen through it carries the texture of the far wall, while monocular depth places it on
the pane, 30 % nearer (as `office_sequence`'s trees behind the glass). Panned from one spot, the
four views triangulate no metric scale and are posed by the multi-view fallback, without SfM
support. An update that only extended them would freeze those poses: the map is rebuilt with the
next update's views instead, as one update of them all (``mapping.api._restart_weak_map``), and
ends as the one-update map does: the same objects, ids, labels and boxes.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from oh_my_slam.core.types import Pose
from oh_my_slam.mapping import api
from oh_my_slam.mapping.api import update
from oh_my_slam.segmentation.obb import obb_iou_upright
from tests.fakes.client import FakeClient, FakeFrame, FakeInstance
from tests.synth.mapping import K, mapping_room, ring
from tests.synth.scene import Room, look_at, render
from tests.unit.test_mapping_e2e_synth import cuboid_obb, objects_by_label

pytestmark = pytest.mark.skipif(shutil.which("colmap") is None, reason="needs Homebrew colmap")

PANE = 0.7  # monocular depth of the scenery behind the window: on the pane, 30 % nearer


def _scenery_on_the_pane(depth: np.ndarray, pose: Pose) -> np.ndarray:
    """``depth`` with the window in the wall at x = -3 m (|y| < 1 m, 0.6-1.9 m high) placed
    ``PANE`` times nearer."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    rays = np.stack([(u - K.cx) / K.fx, (v - K.cy) / K.fy, np.ones_like(depth)], axis=-1)
    pts = (rays * depth[..., None]) @ pose.R.T + pose.t
    window = ((np.abs(pts[..., 0] + 3.0) < 0.03) & (np.abs(pts[..., 1]) < 1.0)
              & (pts[..., 2] > 0.6) & (pts[..., 2] < 1.9) & (depth > 0))
    out = depth.copy()
    out[window] *= PANE
    return out


def _add(client: FakeClient, room: Room, poses: list[Pose], folder: Path, prefix: str
         ) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, pose in enumerate(poses):
        r = render(room, pose, K)
        inst = [FakeInstance(box.label, 0.85, r.ids == k + 2)
                for k, box in enumerate(room.boxes) if (r.ids == k + 2).sum() > 150]
        up = pose.R.T @ np.array([0.0, 0.0, 1.0])
        path = folder / f"{prefix}_{i:03d}.png"
        depth = _scenery_on_the_pane(r.depth, pose).astype(np.float32)
        client.add(path, r.rgb, FakeFrame(depth, K, up, inst, pose=pose))
        paths.append(path)
    return paths


def _steps() -> list[Pose]:
    """Four views of the window wall, panned from one spot."""
    eye = np.array([2.3, 0.2, 1.5])
    return [look_at(eye, np.array([-3.0, y, 0.9])) for y in (-0.9, -0.3, 0.3, 0.9)]


@pytest.fixture(scope="module")
def views(tmp_path_factory: pytest.TempPathFactory) -> tuple[FakeClient, list[Path], list[Path]]:
    base = tmp_path_factory.mktemp("restart")
    client = FakeClient(mv_noise=(1.0, 0.03))
    room = mapping_room()
    first = _add(client, room, _steps(), base / "a", "a")
    rest = _add(client, room, ring(12, start=0.2), base / "b", "b")
    return client, first, rest


def _quiet(msg: str) -> None:
    pass


def test_a_weak_first_update_is_rebuilt_with_the_next(
        views: tuple[FakeClient, list[Path], list[Path]], tmp_path: Path) -> None:
    client, first, rest = views
    one = json.loads(update(tmp_path / "one", first + rest, client=client,
                            progress=_quiet).payload)
    split = tmp_path / "split"
    published = json.loads(update(split, first, client=client, progress=_quiet).payload)
    frames = json.loads((split / "frames.json").read_text())["frames"]
    assert all(api._weak_keyframe(api.store.FrameRecord.from_dict(f)) for f in frames)
    two = json.loads(update(split, rest, client=client, progress=_quiet).payload)
    meta = json.loads((split / "map.json").read_text())
    assert meta["updates"][-1]["notes"]["restarted"]["stored_keyframes"] == len(first)
    assert meta["updates"][0]["frames_added"] == [f"f{k:06d}" for k in range(4)]
    a, b = objects_by_label(one), objects_by_label(two)
    assert sorted(a) == sorted(b) == ["box", "cabinet", "sofa"], (a.keys(), b.keys())
    _ids_persist(split, published, two)
    for label in a:
        (oa,), (ob,) = a[label], b[label]
        iou = obb_iou_upright(cuboid_obb(oa), cuboid_obb(ob))
        centre = np.linalg.norm(cuboid_obb(oa).center - cuboid_obb(ob).center)
        assert iou > 0.7 and centre < 0.05, (label, iou, centre)
    # nothing of the first map's derived files is left behind
    ids = {o["id"] for o in json.loads((split / "objects.json").read_text())["objects"]}
    files = {int(p.stem.split("_")[1]) for p in (split / "objects").glob("*.npy")}
    assert files <= ids


def test_a_map_posed_by_sfm_is_extended(
        views: tuple[FakeClient, list[Path], list[Path]], tmp_path: Path) -> None:
    """The ring's views are posed by SfM: the next update extends them, held fixed."""
    client, first, rest = views
    m = tmp_path / "m"
    update(m, rest, client=client, progress=_quiet)
    update(m, first, client=client, progress=_quiet)
    notes = json.loads((m / "map.json").read_text())["updates"][-1]["notes"]
    assert "restarted" not in notes


def _resolve(mdir: Path, oid: int) -> int:
    """``oid`` through the map's merges, permanent and provisional (``rebuild_merged``)."""
    d = json.loads((mdir / "objects.json").read_text())
    merged = {int(k): int(v) for k, v in
              {**d.get("rebuild_merged", {}), **d.get("merged_into", {})}.items()}
    seen = set()
    while oid in merged and oid not in seen:
        seen.add(oid)
        oid = merged[oid]
    return oid


def _ids_persist(mdir: Path, before: dict, after: dict) -> None:  # type: ignore[type-arg]
    """Every id ``before`` published resolves (itself, or through the merges) to an object of
    ``after`` with a compatible label, the one of that label nearest the published box (the same
    object, not another one the id moved to: a weakly posed map placed it decimetres off, but
    not nearer another), or an update reported it removed (or kept as a candidate only:
    ``unpublished``)."""
    from oh_my_slam.segmentation.detect import compatible

    now = {int(k): o for k, o in after["openlabel"]["objects"].items()}
    removed = {oid for u in json.loads((mdir / "map.json").read_text())["updates"]
               for oid in u["objects"]["removed"] + u["objects"].get("unpublished", [])}
    for k, o in before["openlabel"]["objects"].items():
        oid = _resolve(mdir, int(k))
        assert oid in now or int(k) in removed, (k, o["type"], sorted(now))
        if oid in now:
            assert compatible(o["type"], now[oid]["type"]), (k, o["type"], now[oid]["type"])
            c = cuboid_obb(o).center
            near = min((j for j, x in now.items() if compatible(o["type"], x["type"])),
                       key=lambda j: float(np.linalg.norm(cuboid_obb(now[j]).center - c)))
            assert near == oid, (k, o["type"], oid, near)


def test_published_ids_survive_a_rebuild_whatever_the_server_says_again(
        views: tuple[FakeClient, list[Path], list[Path]], tmp_path: Path) -> None:
    """A rebuild maps the stored keyframes from what the map holds of them (their detections and
    the ids they got): what the server would say about them now does not matter. Here it would
    detect nothing in them any more."""
    from tests.fakes.client import FakeClient as Fake

    client, first, rest = views
    m = tmp_path / "m"
    published = json.loads(update(m, first, client=client, progress=_quiet).payload)
    assert published["openlabel"]["objects"]
    from dataclasses import replace

    forgetful = Fake(mv_noise=client.mv_noise)
    forgetful.frames = dict(client.frames)
    for p in first:  # the same images, no detection: a re-inference would find other objects
        key = str(p.resolve())
        forgetful.frames[key] = replace(forgetful.frames[key], instances=[])
    after = json.loads(update(m, rest, client=forgetful, progress=_quiet).payload)
    assert json.loads((m / "map.json").read_text())["updates"][-1]["notes"]["restarted"]
    _ids_persist(m, published, after)


def test_a_rebuild_that_would_leave_out_a_stored_keyframe_extends_the_map(
        views: tuple[FakeClient, list[Path], list[Path]], tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A rebuild that cannot place every stored keyframe again is abandoned in the same update:
    the map is extended instead, and keeps every keyframe it had."""
    client, first, rest = views
    m = tmp_path / "m"
    update(m, first, client=client, progress=_quiet)
    before = {f["name"] for f in json.loads((m / "frames.json").read_text())["frames"]}
    real = api._place

    def losing_one(ctx, *a, **k):  # type: ignore[no-untyped-def]
        model = real(ctx, *a, **k)
        if ctx.rebuild is not None:
            stored = [nf for nf in ctx.new if nf.kf.name in ctx.rebuild.uids]
            stored[1].record = None
        return model

    monkeypatch.setattr(api, "_place", losing_one)
    update(m, rest, client=client, progress=_quiet)
    notes = json.loads((m / "map.json").read_text())["updates"][-1]["notes"]
    assert "restarted" not in notes and notes["restart_abandoned"]["left_out"] == ["f000001"]
    after = {f["name"] for f in json.loads((m / "frames.json").read_text())["frames"]}
    assert before < after


def test_a_pan_from_one_spot_is_not_rebuilt_again_and_again(tmp_path: Path) -> None:
    """A map of a head turning in place stays weakly posed whatever is added (its multi-view
    poses are not SfM's): once a rebuild left it so, the next updates extend it, and say why
    (``restart_skipped``), instead of rebuilding it every time at a cost growing with the map."""
    from tests.synth.mapping import add_frames
    from tests.unit.test_mapping_e2e_rotation import turning

    client = FakeClient()
    room = mapping_room()
    parts = [turning(8, 0.0, 12.0), turning(5, 6.0, 25.0), turning(5, 18.0, 25.0)]
    m = tmp_path / "m"
    for k, poses in enumerate(parts):
        imgs = add_frames(client, room, poses, tmp_path / f"p{k}", f"p{k}", depth_noise=0.03,
                          seed=k)
        update(m, imgs, client=client, progress=_quiet)
    notes = [u["notes"] for u in json.loads((m / "map.json").read_text())["updates"]]
    assert ["restarted" in n for n in notes] == [False, True, False]
    assert "rotation-dominant" in notes[2]["restart_skipped"]


def test_an_object_an_update_removed_does_not_come_back_with_a_rebuild(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The chair the pan saw is gone when the ring looks (update 2, a rebuild of the weak pan);
    the next update rebuilds the map again (every keyframe taken as weak here), with the pan's
    stored keyframes, which still detect the chair: their pixels of it stay retired, and it does
    not come back, under any id."""
    from tests.synth.scene import Box

    monkeypatch.setattr(api, "_weak_keyframe", lambda rec: True)
    client = FakeClient(mv_noise=(1.0, 0.03))
    room = mapping_room()
    chair = Box(np.array([0.0, 0.1, 0.35]), np.array([0.45, 0.45, 0.7]), 0.2, (230, 200, 40),
                "chair")
    first = _add(client, Room(boxes=[*room.boxes, chair]), _steps(), tmp_path / "a", "a")
    second = _add(client, room, ring(12, start=0.2), tmp_path / "b", "b")
    third = _add(client, room, ring(6, start=0.45, radius=2.0), tmp_path / "c", "c")
    m = tmp_path / "m"
    one = json.loads(update(m, first, client=client, progress=_quiet).payload)
    labels = {int(k): o["type"] for k, o in one["openlabel"]["objects"].items()}
    (chair_id,) = [k for k, t in labels.items() if t == "chair"]
    update(m, second, client=client, progress=_quiet)
    doc = json.loads(update(m, third, client=client, progress=_quiet).payload)
    updates = json.loads((m / "map.json").read_text())["updates"]
    assert ["restarted" in u["notes"] for u in updates] == [False, True, True]
    assert chair_id in updates[1]["objects"]["removed"]
    assert "chair" not in {o["type"] for o in doc["openlabel"]["objects"].values()}
    assert _resolve(m, chair_id) not in {int(k) for k in doc["openlabel"]["objects"]}
    _ids_persist(m, one, doc)


def test_no_rebuild_when_the_last_two_left_the_same_keyframes_weak(
        views: tuple[FakeClient, list[Path], list[Path]], tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Rebuilt twice, a map whose same keyframes stay weak (here the first one, taken as weak
    whatever its pose) is extended by the next update, which says why."""
    monkeypatch.setattr(api, "_weak_keyframe", lambda rec: rec.name == "f000000")
    client, first, rest = views
    m = tmp_path / "m"
    for part in (first, rest[:4], rest[4:8], rest[8:]):
        update(m, part, client=client, progress=_quiet)
    notes = [u["notes"] for u in json.loads((m / "map.json").read_text())["updates"]]
    assert ["restarted" in n for n in notes] == [False, True, True, False]
    assert notes[1]["restarted"]["weak"] == notes[2]["restarted"]["weak"] == ["f000000"]
    assert "no progress" in notes[3]["restart_skipped"]


def test_an_abandoned_rebuild_extends_the_map_as_an_extension_does(
        views: tuple[FakeClient, list[Path], list[Path]], tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The rebuild's focal re-run and alignment of the new keyframes are undone when it is
    abandoned: the map is extended exactly as without a rebuild (one thread: COLMAP repeats)."""
    from oh_my_slam.mapping import sfm as sfm_mod

    monkeypatch.setattr(sfm_mod, "SFM_THREADS", 1)
    client, first, rest = views
    base = tmp_path / "base"
    update(base, first, client=client, progress=_quiet)
    plain, abandoned = tmp_path / "plain", tmp_path / "abandoned"
    shutil.copytree(base, plain)
    shutil.copytree(base, abandoned)
    with monkeypatch.context() as mp:
        mp.setattr(api, "RESTART_MAX_KEYFRAMES", 0)
        update(plain, rest, client=client, progress=_quiet)
    real = api._place

    def losing_one(ctx, *a, **k):  # type: ignore[no-untyped-def]
        model = real(ctx, *a, **k)
        if ctx.rebuild is not None:
            next(nf for nf in ctx.new if nf.kf.name in ctx.rebuild.uids).record = None
        return model

    monkeypatch.setattr(api, "_place", losing_one)
    update(abandoned, rest, client=client, progress=_quiet)
    a = json.loads((plain / "frames.json").read_text())["frames"]
    b = json.loads((abandoned / "frames.json").read_text())["frames"]
    assert [f["name"] for f in a] == [f["name"] for f in b]
    for fa, fb in zip(a, b, strict=True):  # up to floating-point noise
        ta, tb = fa["T_map_cam"], fb["T_map_cam"]
        assert np.allclose(ta["translation"], tb["translation"], atol=1e-4), fa["name"]
        assert np.allclose(ta["quaternion_xyzw"], tb["quaternion_xyzw"], atol=1e-4), fa["name"]
        assert fa["K"] == fb["K"], fa["name"]
