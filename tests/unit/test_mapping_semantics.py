"""Mapping semantics (spec §2.3) on rendered rooms with known poses (no SfM, no server):

* the keyframes of one update are one observation — permuting them changes nothing but
  bookkeeping (keyframe names, and the ids that follow capture order);
* a later update wins over an earlier one;
* persistent identity: one id and one colour per object for the map's lifetime, OBBs refined as
  evidence accumulates, merges keep the lower id;
* a sequence mapped in one update or split over several gives the same objects and ids;
* capture timestamps are never read.

Updates run the production integration step (``mapping.api.integrate``: latest wins, objects,
cloud) on keyframes whose poses and depth come from the renderer.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oh_my_slam.core.images import load_rgb, save_jpeg
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.mapping import api, export, ingest, objects, store, validity
from oh_my_slam.mapping.objects import MapObject, ObjectState, canonical_points
from oh_my_slam.reconstruction.api import FrameReconstruction
from oh_my_slam.segmentation.api import OBB, Detection, obb_iou_upright
from oh_my_slam.segmentation.colors import color_for_id, color_hex_for_id
from tests.synth.mapping import mapping_room, ring
from tests.synth.scene import Box, Room, look_at, render

K = Intrinsics(300.0, 300.0, 200.0, 150.0, 400, 300)
SRC = Path(__file__).resolve().parents[2] / "src" / "oh_my_slam"


# --- harness --------------------------------------------------------------------------------------


@dataclass
class Shot:
    """A keyframe with its true pose, rendered image and depth, and its detections."""

    pose: Pose
    rgb: np.ndarray
    depth: np.ndarray
    dets: list[tuple[str, np.ndarray, float]] = field(default_factory=list)


def shoot(room: Room, poses: list[Pose], depth_noise: float = 0.0, seed: int = 0,
          detect: Any = None) -> list[Shot]:
    """Render ``poses``; each box seen by > 150 px is detected with its label (score 0.85) unless
    ``detect(k, label, mask)`` returns None (missed) or a replacement mask."""
    rng = np.random.default_rng(seed)
    out = []
    for k, pose in enumerate(poses):
        r = render(room, pose, K)
        scale = 1.0 + (rng.uniform(-depth_noise, depth_noise) if depth_noise else 0.0)
        dets = []
        for b, box in enumerate(room.boxes):
            m = r.ids == b + 2
            if detect is not None:
                m = detect(k, box.label, m)
            if m is not None and m.sum() > 150:
                dets.append((box.label, m, 0.85))
        out.append(Shot(pose, r.rgb, (r.depth * scale).astype(np.float32), dets))
    return out


def _detection(label: str, mask: np.ndarray, score: float) -> Detection:
    ys, xs = np.nonzero(mask)
    return Detection(label, score, "fake", mask,
                     (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)))


@dataclass
class Result:
    objs: ObjectState
    cloud: Any
    records: list[store.FrameRecord]
    names: list[str]  # keyframe name of each shot, in the order given


def known_pose_update(mdir: Path, shots: list[Shot], work: Path) -> Result:
    """One ``mapper.sh update`` of ``shots`` with their true poses (placed, well registered)."""
    work.mkdir(parents=True, exist_ok=True)
    with store.MapTransaction(mdir) as tx:
        meta = store.read_meta_or_default(tx)
        old = store.read_frames(tx)
        uid = int(meta.get("update_count", 0)) + 1
        start = int(meta.get("next_frame_index", 0))
        new = []
        for k, sh in enumerate(shots):
            idx = start + k
            name = store.frame_name(idx)
            img = tx.stage("frames") / f"{name}.jpg"
            save_jpeg(sh.rgb, img, quality=95)
            h, w = sh.depth.shape
            frame = FrameReconstruction(img, load_rgb(img, max_side=max(w, h)), sh.depth,
                                        sh.depth > 0, K, K)
            nf = api.NewFrame(ingest.Keyframe(name, idx, img, f"shot {k}", None), frame,
                              [_detection(*d) for d in sh.dets], (w, h))
            nf.record = store.FrameRecord(
                idx, name, f"frames/{name}.jpg", f"shot {k}", 1, w, h, K, sh.pose, w, h,
                pose_source="sfm-global", update_id=uid,
                stats={"observations": 500.0, "reproj_error": 0.5})
            nf.depth = sh.depth
            new.append(nf)
        ctx = api.UpdateContext(tx, meta, old, new, uid, work)
        records, objs, geo = api.integrate(ctx, lambda m: None)
        meta.update(update_count=uid, next_frame_index=start + len(shots),
                    next_object_id=objs.next_id)
        tx.write_json(store.SCENE_JSON, export.full_scene(tx.root, meta, records,
                                                          objs.exported()))
        tx.commit(meta)
    return Result(objs, geo.cloud, records, [store.frame_name(start + k)
                                              for k in range(len(shots))])


def exported(res: Result) -> dict[int, Any]:
    return {o.id: o for o in res.objs.exported()}


def match_ids(a: Result, b: Result) -> dict[int, int]:
    """Bijection between the objects of two maps of the same scene (same label, nearest box)."""
    out: dict[int, int] = {}
    ob = {o.id: o for o in b.objs.objects}
    for o in a.objs.objects:
        cands = [p for p in ob.values() if p.label == o.label and p.id not in out.values()]
        assert cands, (o.id, o.label)
        best = min(cands, key=lambda p: float(np.linalg.norm(p.centroid - o.centroid)))
        out[o.id] = best.id
    assert len(out) == len(ob)
    return out


def truth_obb(box: Box) -> OBB:
    from oh_my_slam.core.geometry import rot_z

    return OBB(np.asarray(box.center, float), rot_z(box.yaw), np.asarray(box.size, float))


# --- one update is one observation ---------------------------------------------------------------


def _noisy_shots() -> tuple[Room, list[Shot]]:
    """Fourteen views of the mapping room (more than one keyframe neighbourhood) with ±3 %
    depth noise, a detector that misses the sofa in two views and one flicker false positive (a
    "box" on the wall in one view)."""
    room = mapping_room()

    def detect(k: int, label: str, m: np.ndarray) -> np.ndarray | None:
        return None if (label == "sofa" and k in (2, 5)) else m

    shots = shoot(room, ring(14), depth_noise=0.03, seed=11, detect=detect)
    wall = render(room, shots[7].pose, K).ids == 1
    flick = np.zeros_like(wall)
    flick[40:70, 180:220] = True  # a patch of wall well inside the image
    assert (flick & wall).sum() > 1000
    shots[7].dets.append(("box", flick & wall, 0.6))
    return room, shots


def test_keyframe_order_within_an_update_does_not_matter(tmp_path: Path) -> None:
    room, shots = _noisy_shots()
    perm = [6, 2, 13, 9, 0, 11, 4, 7, 1, 12, 8, 5, 10, 3]
    a = known_pose_update(tmp_path / "a", shots, tmp_path / "wa")
    b = known_pose_update(tmp_path / "b", [shots[i] for i in perm], tmp_path / "wb")
    # the same cloud, colours and object membership (ids follow capture order: bookkeeping)
    np.testing.assert_array_equal(a.cloud.xyz, b.cloud.xyz)
    np.testing.assert_array_equal(a.cloud.rgb, b.cloud.rgb)
    ids = match_ids(a, b)
    relabel = np.vectorize(lambda x: ids.get(int(x), 0) if x else 0)
    np.testing.assert_array_equal(relabel(a.cloud.label), b.cloud.label)
    # the same objects: labels, evidence, confirmation, points and boxes
    ob = b.objs.by_id()
    for o in a.objs.objects:
        p = ob[ids[o.id]]
        assert (o.label, o.confirmed, o.observations, o.views_in_frustum, o.label_votes) == (
            p.label, p.confirmed, p.observations, p.views_in_frustum, p.label_votes)
        np.testing.assert_array_equal(o.points, p.points)
        np.testing.assert_allclose(o.obb.center, p.obb.center, atol=1e-12)
        np.testing.assert_allclose(o.obb.size, p.obb.size, atol=1e-12)
        # the keyframes that detected it are the same shots
        name_a = dict(zip(a.names, range(len(shots)), strict=True))
        name_b = {n: perm[k] for k, n in enumerate(b.names)}
        shots_a = {name_a[store.frame_name(f)] for f in o.frames}
        shots_b = {name_b[store.frame_name(f)] for f in p.frames}
        assert shots_a == shots_b
    # the three boxes are confirmed once each; the flicker detection stays unconfirmed
    for res in (a, b):
        labels = sorted(o.label for o in res.objs.exported())
        assert labels == ["box", "cabinet", "sofa"], labels
        flicker = [o for o in res.objs.objects if not o.confirmed]
        assert len(flicker) == 1 and flicker[0].observations == 1
        # new ids follow the earliest keyframe that detected each object: the number of its
        # first detection, counting the detections of the keyframes in capture order
        new = sorted(res.objs.objects, key=lambda o: o.id)
        assert [o.frames[0] for o in new] == sorted(o.frames[0] for o in new)
        assert res.objs.next_id == 1 + sum(len(sh.dets) for sh in shots)
        order = [shots[i] for i in (perm if res is b else range(len(shots)))]
        before = np.cumsum([0] + [len(sh.dets) for sh in order])
        for o in new:
            assert before[o.frames[0]] < o.id <= before[o.frames[0] + 1], (o.id, o.frames)


def test_attribution_latest_update_wins_and_finest_view_within_an_update() -> None:
    """Colour and id of a point come from the latest update that sees it; within that update the
    finest view gives the colour and a vote gives the id, whatever the keyframes' order."""
    from oh_my_slam.mapping.geometry import FrameData, attribute_points, fused_cloud_points
    from tests.synth.scene import default_room, orbit_poses

    room = default_room()
    Kc = Intrinsics(260.0, 260.0, 160.0, 120.0, 320, 240)
    frames = []
    for i, pose in enumerate(orbit_poses(8)):
        r = render(room, pose, Kc)
        uid = 1 if i < 7 else 2
        rec = store.FrameRecord(i, store.frame_name(i), "", "", 1, 320, 240, Kc, pose, 320, 240,
                                update_id=uid)
        rgb = r.rgb if uid == 1 else np.full_like(r.rgb, (10, 200, 30))  # update 2: repainted
        frames.append(FrameData(rec, r.depth.astype(np.float32), r.depth > 0, rgb,
                                np.where(r.ids >= 2, r.ids - 1, 0).astype(np.int32), uid == 2))
    xyz = fused_cloud_points(frames, voxel=0.01, depth_max=6.0)
    rgb, label, seen_new = attribute_points(xyz, frames)
    assert set(np.unique(label)) <= {0, 1, 2, 3} and (label > 0).mean() > 0.05
    # a later update wins wherever it sees a point, although seven older keyframes see it too
    assert seen_new.any() and not seen_new.all()
    assert (rgb[seen_new] == (10, 200, 30)).all()
    assert not (rgb[~seen_new] == (10, 200, 30)).all(axis=1).any()
    # ...and nowhere else: points it cannot see keep the older update's colours
    assert (rgb[~seen_new] != 128).any(axis=1).mean() > 0.9
    # within update 1 the keyframes' order is irrelevant
    rev = frames[:7][::-1] + frames[7:]
    rgb2, label2, seen2 = attribute_points(xyz, rev)
    np.testing.assert_array_equal(rgb, rgb2)
    np.testing.assert_array_equal(label, label2)
    np.testing.assert_array_equal(seen_new, seen2)
    # the finest view (smallest footprint) gives the colour (symmetric views may tie exactly)
    from oh_my_slam.mapping.geometry import _visible

    old = frames[:7]
    rgb1, _, _ = attribute_points(xyz, old)
    seen_by = [_visible(fd, xyz) for fd in old]
    best = np.full(len(xyz), np.inf)
    for idx, _, _, fp in seen_by:
        best[idx] = np.minimum(best[idx], fp)
    hit = np.zeros(len(xyz), bool)
    for fd, (idx, vv, uu, fp) in zip(old, seen_by, strict=True):
        finest = fp == best[idx]
        i = idx[finest]
        hit[i] |= (fd.rgb[vv[finest], uu[finest]] == rgb1[i]).all(axis=1)
    assert hit[np.isfinite(best)].all()


def test_object_id_vote_needs_a_third_of_the_views() -> None:
    """A mask that bleeds onto the floor in one of six views does not label the floor."""
    from oh_my_slam.mapping.geometry import FrameData, _attribute_update

    pts = np.array([[0.0, 0.0, 2.0]])
    frames = []
    for k in range(6):
        rec = store.FrameRecord(k, store.frame_name(k), "", "", 1, 5, 5,
                                Intrinsics(5.0, 5.0, 2.0, 2.0, 5, 5), Pose.identity(), 5, 5)
        lab = np.full((5, 5), 7 if k == 0 else 0, np.int32)
        frames.append(FrameData(rec, np.full((5, 5), 2.0, np.float32), np.ones((5, 5), bool),
                                np.full((5, 5, 3), 40 + k, np.uint8), lab, True))
    _, _, label = _attribute_update(pts, frames)
    assert label[0] == 0  # 1 vote of 6
    frames[1].labels[:] = 7
    _, _, label = _attribute_update(pts, frames)
    assert label[0] == 7  # 2 of 6 = a third


# --- latest wins across updates ------------------------------------------------------------------


def test_later_update_wins_over_an_earlier_one(tmp_path: Path) -> None:
    """Update 1 sees a room with a box; update 2, with far fewer keyframes, sees it without the
    box and with repainted walls: the box is removed, and wherever update 2 sees the room its
    colours replace update 1's."""
    box = Box(np.array([0.0, 0.2, 0.4]), np.array([0.7, 0.6, 0.8]), 0.2, (220, 40, 40), "cabinet")
    before = Room(boxes=[box])
    after = Room(boxes=[], wall_color=(40, 60, 200))
    mdir = tmp_path / "m"
    r1 = known_pose_update(mdir, shoot(before, ring(10)), tmp_path / "w1")
    (cab,) = r1.objs.exported()
    r2 = known_pose_update(mdir, shoot(after, ring(3, start=0.4)), tmp_path / "w2")
    assert r2.objs.exported() == [] and cab.id in r2.objs.summary["removed"]
    assert not (mdir / objects.points_file(cab.id)).exists()
    assert (r2.cloud.label == 0).all()
    # a wall point seen by update 2 carries update 2's (blue) colour
    new_ids = {r.index for r in r2.records if r.update_id == 2}
    reader = store.MapReader(mdir)
    assert {f.index for f in reader.frames if f.update_id == 2} == new_ids
    blue = (r2.cloud.rgb[:, 2] > 120) & (r2.cloud.rgb[:, 0] < 90)
    assert blue.mean() > 0.2
    # ids are never reused: the next object gets a fresh id
    r3 = known_pose_update(mdir, shoot(before, ring(4, start=1.0)), tmp_path / "w3")
    (again,) = [o for o in r3.objs.objects if o.label == "cabinet"]
    assert again.id > cab.id


def _absence_view(room: Room, pose: Pose) -> tuple[Any, validity.View]:
    from types import SimpleNamespace

    r = render(room, pose, K)
    rec = SimpleNamespace(stats={"observations": 500.0, "reproj_error": 0.5},
                          pose_source="sfm-global")
    return SimpleNamespace(record=rec), validity.View(r.depth, r.depth > 0, K, pose)


def test_removal_needs_the_update_as_a_whole() -> None:
    """Three keyframes that see through an object do not remove it when most of the update's
    keyframes see it in place (monocular depth errors); an update that sees it in place clears
    its strikes, one that sees through it with most keyframes removes it."""
    box = Box(np.array([0.0, 0.0, 0.4]), np.array([0.6, 0.6, 0.8]), 0.0, (200, 60, 60), "box")
    with_box, without = Room(boxes=[box]), Room(boxes=[])
    rng = np.random.default_rng(0)
    local = rng.uniform(-0.5, 0.5, (4000, 3)) * box.size
    face = np.argmax(np.abs(local) / box.size, axis=1)
    local[np.arange(len(local)), face] = np.sign(local[np.arange(len(local)), face]) \
        * box.size[face] / 2
    pts = (local + box.center).astype(np.float32)
    pts = pts[pts[:, 2] > 0.35]  # near the floor, seeing through is within the margin anyway

    def obj() -> MapObject:
        return MapObject(4, "box", {"box": 4.0}, [0.9] * 4, pts, frames=[0, 1, 2, 3],
                         confirmed=True)

    poses = ring(12)
    through = {i: _absence_view(without, p) for i, p in enumerate(poses[:3])}
    in_place = {i + 3: _absence_view(with_box, p) for i, p in enumerate(poses[3:])}
    o = obj()
    o.strikes = 1
    assert objects._absence([o], {**through, **in_place}) == []
    assert o.strikes == 0  # most keyframes saw it in place
    o = obj()
    assert objects._absence([o], through) == [4]  # >= 3 keyframes, all through
    # a single keyframe: an established object (>= 4 detections) goes, a weak one gets a strike
    one = {0: through[0]}
    assert objects._absence([obj()], one) == [4]
    weak = obj()
    weak.frames = [0, 1]
    assert objects._absence([weak], one) == [] and weak.strikes == 1
    assert objects._absence([weak], one) == [4]  # a second strike (from a later update)


def test_pixels_are_invalidated_only_when_the_update_as_a_whole_contradicts_them() -> None:
    box = Box(np.array([0.0, 0.0, 0.4]), np.array([0.6, 0.6, 0.8]))
    with_box, without = Room(boxes=[box]), Room(boxes=[])
    old_pose = look_at(np.array([2.2, 0.3, 1.4]), np.array([0.0, 0.0, 0.4]))
    old = validity.View(render(with_box, old_pose, K).depth,
                        render(with_box, old_pose, K).depth > 0, K, old_pose)
    near = [look_at(np.array([2.1, y, 1.5]), np.array([0.0, 0.0, 0.4]))
            for y in (-0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7)]

    def view(room: Room, pose: Pose) -> validity.View:
        d = render(room, pose, K).depth
        return validity.View(d, d > 0, K, pose)

    box_px = render(with_box, old_pose, K).ids == 2
    two_through = [view(without, near[0]), view(without, near[1])]
    cells = validity.contradicted_cells(old, two_through)
    assert validity.cells_to_pixels(cells, old.depth.shape)[box_px].mean() > 0.6
    # the same two keyframes in an update whose five others still see the box: kept
    mixed = two_through + [view(with_box, p) for p in near[2:]]
    cells = validity.contradicted_cells(old, mixed)
    assert validity.cells_to_pixels(cells, old.depth.shape)[box_px].mean() < 0.02
    # order does not matter
    np.testing.assert_array_equal(cells, validity.contradicted_cells(old, mixed[::-1]))


# --- persistent identity -------------------------------------------------------------------------


def test_identity_colour_and_obb_refinement_across_updates(tmp_path: Path) -> None:
    """Objects keep their id and colour over three updates; a box seen from one side first gets
    a better OBB once the other side is observed."""
    room = mapping_room()
    mdir = tmp_path / "m"
    first = known_pose_update(mdir, shoot(room, ring(5, start=0.0, span=np.pi / 2)),
                              tmp_path / "w1")
    ids = {o.label: o.id for o in first.objs.exported()}
    assert set(ids) == {"cabinet", "box", "sofa"}
    iou1 = {o.label: obb_iou_upright(o.obb, truth_obb(b))
            for o in first.objs.exported() for b in room.boxes if b.label == o.label}
    later = [known_pose_update(mdir, shoot(room, ring(5, start=s, span=np.pi / 2)),
                               tmp_path / f"w{k}") for k, s in ((2, np.pi), (3, 1.5 * np.pi))]
    final = later[-1]
    for o in final.objs.exported():
        assert o.id == ids[o.label] and o.color_hex == color_hex_for_id(ids[o.label])
    assert sorted(o.id for o in final.objs.exported()) == sorted(ids.values())
    iou3 = {o.label: obb_iou_upright(o.obb, truth_obb(b))
            for o in final.objs.exported() for b in room.boxes if b.label == o.label}
    assert all(iou3[k] >= iou1[k] - 0.02 for k in ids), (iou1, iou3)
    assert np.mean(list(iou3.values())) > np.mean(list(iou1.values())) + 0.05, (iou1, iou3)
    assert min(iou3.values()) > 0.6, iou3
    # the colour contract on the map cloud: each labelled point has its object's colour
    doc = json.loads((mdir / store.SCENE_JSON).read_text())
    for oid, od in doc["openlabel"]["objects"].items():
        assert od["object_data"]["text"][0]["val"] == color_hex_for_id(int(oid))


def test_merge_keeps_the_lower_id(tmp_path: Path) -> None:
    """A sofa seen only in halves by update 1 becomes two objects; update 2 sees it whole and
    the two merge into the lower id, which old keyframes' instances now resolve to."""
    sofa = Box(np.array([0.0, 0.0, 0.4]), np.array([2.4, 0.8, 0.8]), 0.0, (50, 70, 210), "sofa")
    room = Room(boxes=[sofa])
    left = [look_at(np.array([x, -2.2, 1.2]), np.array([x, 0.0, 0.4])) for x in (-1.4, -1.1)]
    right = [look_at(np.array([x, -2.2, 1.2]), np.array([x, 0.0, 0.4])) for x in (1.1, 1.4)]

    def half(poses: list[Pose], keep_x: Any) -> Any:
        """Detector that only segments the part of the sofa where ``keep_x(x_map)``."""
        def detect(k: int, label: str, m: np.ndarray) -> np.ndarray:
            v, u = np.nonzero(m)
            z = render(room, poses[k], K).depth[v, u]
            cam = np.stack([(u - K.cx) / K.fx * z, (v - K.cy) / K.fy * z, z], 1)
            keep = keep_x(poses[k].apply(cam)[:, 0])
            out = np.zeros_like(m)
            out[v[keep], u[keep]] = True
            return out
        return detect

    mdir = tmp_path / "m"
    r1 = known_pose_update(mdir, shoot(room, left, detect=half(left, lambda x: x < -0.35))
                           + shoot(room, right, detect=half(right, lambda x: x > 0.35)),
                           tmp_path / "w1")
    halves = sorted(o.id for o in r1.objs.exported())
    assert len(halves) == 2 and len(r1.objs.objects) == 2
    whole = [look_at(np.array([x, -2.3, 1.4]), np.array([x, 0.0, 0.4])) for x in (-0.3, 0.0, 0.3)]
    shots = shoot(room, whole)
    assert all(len(sh.dets) == 1 for sh in shots)
    r2 = known_pose_update(mdir, shots, tmp_path / "w2")
    (merged,) = r2.objs.objects
    assert merged.id == halves[0] and r2.objs.merged_into == {halves[1]: halves[0]}
    assert r2.objs.resolve(halves[1]) == halves[0]
    assert not (mdir / objects.points_file(halves[1])).exists()
    assert merged.obb.size[0] > 2.0 and merged.observations == 7
    # every labelled cloud point, including those attributed from update 1's keyframes whose
    # instances name the merged id, carries the kept id and its colour
    assert set(np.unique(r2.cloud.label)) == {0, halves[0]}
    reader = store.MapReader(mdir)
    _, objs_ = export.map_objects(reader)
    src = export.reader_source(reader, objs_)
    assert src.labels is not None and set(np.unique(src.labels)) == {0, halves[0]}
    assert [o.id for o in objs_] == [halves[0]] and objs_[0].color == color_for_id(halves[0])
    r1_insts = [i["object_id"] for r in reader.frames if r.update_id == 1
                for i in reader.instances(r)]
    assert sorted(set(r1_insts)) == halves  # old instance files keep the merged id...


# --- one update vs several -----------------------------------------------------------------------


def test_split_updates_give_the_same_objects_and_ids(tmp_path: Path) -> None:
    """The same twelve views mapped in one update and in three give the same objects, labels,
    ids and (nearly) the same boxes; an object detected in two views at the end of one split and
    two at the start of the next is confirmed in both cases."""
    room = mapping_room()

    def detect(k: int, label: str, m: np.ndarray) -> np.ndarray | None:
        # the green box is only detected in views 2-5 (it straddles the first split)
        return m if label != "box" or 2 <= k <= 5 else None

    shots = shoot(room, ring(12, span=1.5 * np.pi), depth_noise=0.02, seed=5, detect=detect)
    one = known_pose_update(tmp_path / "one", shots, tmp_path / "w1")
    split_dir = tmp_path / "split"
    for k, part in enumerate((shots[:4], shots[4:8], shots[8:])):
        three = known_pose_update(split_dir, part, tmp_path / f"w3{k}")
    a, b = exported(one), exported(three)
    assert sorted(a) == sorted(b), (a, b)
    for oid, o in a.items():
        p = b[oid]
        assert o.label == p.label and o.color == p.color and o.observations == p.observations
        assert obb_iou_upright(o.obb, p.obb) > 0.9, (o.label, o.obb, p.obb)
        assert np.linalg.norm(o.obb.center - p.obb.center) < 0.05
    assert {o.label for o in a.values()} == {"cabinet", "box", "sofa"}


def test_split_ids_do_not_depend_on_earlier_updates_bookkeeping(tmp_path: Path) -> None:
    """A sofa seen only in halves by the first split becomes two objects there (merged when the
    second split sees it whole), and the first split's flicker detection stays an unconfirmed
    candidate: neither shifts the ids of later objects — the one-update map and the split map
    give every object the same id."""
    sofa = Box(np.array([0.0, 0.0, 0.4]), np.array([2.4, 0.8, 0.8]), 0.0, (50, 70, 210), "sofa")
    cab = Box(np.array([0.2, 1.6, 0.4]), np.array([0.6, 0.5, 0.8]), 0.0, (220, 40, 40),
              "cabinet")
    room = Room(boxes=[sofa, cab])
    left = [look_at(np.array([x, -2.2, 1.2]), np.array([x, 0.0, 0.4])) for x in (-1.4, -1.1)]
    right = [look_at(np.array([x, -2.2, 1.2]), np.array([x, 0.0, 0.4])) for x in (1.1, 1.4)]
    whole = [look_at(np.array([x, -2.3, 1.4]), np.array([x, 0.0, 0.4])) for x in (-0.3, 0.0, 0.3)]
    behind = [look_at(np.array([x, 2.3, 1.4]), np.array([0.2, 1.6, 0.4])) for x in (-0.3, 0.2, 0.7)]

    def half(poses: list[Pose], keep_x: Any) -> Any:
        def detect(k: int, label: str, m: np.ndarray) -> np.ndarray | None:
            if label != "sofa":
                return None
            v, u = np.nonzero(m)
            z = render(room, poses[k], K).depth[v, u]
            cam = np.stack([(u - K.cx) / K.fx * z, (v - K.cy) / K.fy * z, z], 1)
            keep = keep_x(poses[k].apply(cam)[:, 0])
            out = np.zeros_like(m)
            out[v[keep], u[keep]] = True
            return out
        return detect

    first = (shoot(room, left, detect=half(left, lambda x: x < -0.35))
             + shoot(room, right, detect=half(right, lambda x: x > 0.35)))
    wall = render(room, first[1].pose, K).ids == 1
    flick = np.zeros_like(wall)
    flick[40:70, 180:220] = True
    first[1].dets.append(("box", flick & wall, 0.6))  # a one-off false positive
    second = shoot(room, whole, detect=lambda k, lab, m: m if lab == "sofa" else None)
    third = shoot(room, behind, detect=lambda k, lab, m: m if lab == "cabinet" else None)
    assert all(sh.dets for sh in first + second + third)
    one = known_pose_update(tmp_path / "one", first + second + third, tmp_path / "w1")
    for k, part in enumerate((first, second, third)):
        split = known_pose_update(tmp_path / "split", part, tmp_path / f"w3{k}")
    assert len(split.objs.merged_into) == 1  # the halves merged in the second split
    ids = {o.id: (o.label, o.confirmed) for o in one.objs.objects}
    assert ids == {o.id: (o.label, o.confirmed) for o in split.objs.objects}, ids
    assert sorted(lab for lab, ok in ids.values() if ok) == ["cabinet", "sofa"]
    assert sorted(exported(one)) == sorted(exported(split))
    assert one.objs.next_id == split.objs.next_id


# --- inputs ----------------------------------------------------------------------------------------


def _png(path: Path, rgb: np.ndarray) -> Path:
    from oh_my_slam.core.images import png_bytes

    path.write_bytes(png_bytes(rgb))
    return path


@pytest.mark.parametrize("kind", ["missing", "empty", "hidden-only"])
def test_update_creates_a_map_in_a_missing_or_empty_folder(tmp_path: Path, kind: str) -> None:
    from oh_my_slam.mapping.api import update
    from tests.fakes.client import FakeClient
    from tests.synth.mapping import add_frames

    client = FakeClient()
    (img,) = add_frames(client, mapping_room(), ring(1), tmp_path / "in", "p")
    mdir = tmp_path / "map"
    if kind != "missing":
        mdir.mkdir()
    if kind == "hidden-only":
        (mdir / ".DS_Store").write_bytes(b"x")
    res = update(mdir, [img], client=client, progress=lambda m: None)
    assert store.classify(mdir) == "map" and res.new_frames == ["f000000"]
    assert store.MapReader(mdir).meta["update_count"] == 1


def test_update_refuses_a_non_empty_folder_that_is_not_a_map(tmp_path: Path) -> None:
    from oh_my_slam.core.errors import NotAMapError
    from oh_my_slam.mapping.api import update
    from tests.fakes.client import FakeClient
    from tests.synth.mapping import add_frames

    client = FakeClient()
    (img,) = add_frames(client, mapping_room(), ring(1), tmp_path / "in", "p")
    other = tmp_path / "notes"
    other.mkdir()
    (other / "todo.txt").write_text("keep me")
    with pytest.raises(NotAMapError):
        update(other, [img], client=client, progress=lambda m: None)
    assert [p.name for p in other.iterdir()] == ["todo.txt"]
    assert (other / "todo.txt").read_text() == "keep me"
    assert client.calls["geometry"] == 0  # refused before any inference


def test_capture_timestamps_are_not_used(tmp_path: Path) -> None:
    """Keyframes follow the input order whatever the files' EXIF capture times and mtimes say,
    and the mapping code never reads either."""
    from PIL import Image

    d = tmp_path / "caps"
    d.mkdir()
    for k, name in enumerate(("a.jpg", "b.jpg", "c.jpg")):
        exif = Image.Exif()
        exif[0x0132] = f"2026:01:0{3 - k} 12:00:00"  # DateTime, in reverse order
        exif.get_ifd(0x8769)[0x9003] = f"2026:01:0{3 - k} 12:00:00"  # DateTimeOriginal
        Image.new("RGB", (16, 12), (40 * k, 0, 0)).save(d / name, exif=exif)
        os.utime(d / name, (1_000_000 - k * 1000, 1_000_000 - k * 1000))  # mtimes reversed
    kfs = list(ingest.keyframes(ingest.resolve_inputs([d]), 2.0, tmp_path / "frames", 0))
    assert [Path(k.source).name for k in kfs] == ["a.jpg", "b.jpg", "c.jpg"]
    assert [k.index for k in kfs] == [0, 1, 2]
    pattern = re.compile(r"DateTime|0x9003|0x0132|getmtime|st_mtime|st_ctime|st_birthtime")
    hits = [p.name for p in (SRC / "mapping").glob("*.py") if pattern.search(p.read_text())]
    assert hits == []


def test_canonical_points_compose() -> None:
    rng = np.random.default_rng(3)
    a = rng.uniform(-1, 1, (40000, 3)).astype(np.float32)
    b = rng.uniform(-1, 1, (30000, 3)).astype(np.float32)
    whole = canonical_points(np.concatenate([a, b]))
    assert len(whole) == objects.POINT_CAP
    np.testing.assert_array_equal(canonical_points(np.concatenate([canonical_points(a), b])),
                                  whole)
    np.testing.assert_array_equal(canonical_points(np.concatenate([b, a])), whole)
    np.testing.assert_array_equal(canonical_points(rng.permutation(np.concatenate([a, b]))),
                                  whole)
    keys = np.floor(whole.astype(np.float64) / objects.POINT_VOXEL)
    assert len(np.unique(keys, axis=0)) == len(whole)
