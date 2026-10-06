"""Objects of whole updates on rendered rooms with known poses (``mapping.api.integrate``, no SfM,
no server): a keyframe that could not be placed, a rebuild that carries no published id, a cup
that moves and is then gone within one update, and a cup that moves onto a place the map already
held a cup at."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from oh_my_slam.core.images import load_rgb, save_jpeg
from oh_my_slam.mapping import api, ingest, objects, store
from oh_my_slam.reconstruction.api import FrameReconstruction
from tests.synth.scene import Box, Room, look_at
from tests.unit.test_mapping_latest_wins import CABINET, EYE
from tests.unit.test_mapping_semantics import K, Result, Shot, _detection, shoot

HERE = np.array([-0.4, 0.1, 0.08])
THERE = np.array([0.2, -0.45, 0.08])
CUP = Box(HERE, np.array([0.14, 0.14, 0.16]), 0.3, (240, 240, 240), "cup")
MOVED = replace(CUP, center=THERE)
TARGET = np.array([0.1, 0.1, 0.2])
FIRST = [look_at(EYE + [0.0, dy, 0.0], TARGET) for dy in (0.0, 0.06, -0.06)]
LATER = [look_at(EYE + [0.0, dy, 0.0], TARGET) for dy in (0.03, -0.03, 0.09)]
LATEST = [look_at(EYE + [0.0, dy, 0.0], TARGET) for dy in (0.015, -0.045, 0.075)]


WELL = {"observations": 500.0, "reproj_error": 0.5}
# a multi-view pose its matches do not support: low confidence, and not well registered enough
# to invalidate what older keyframes saw (latest wins)
UNSUPPORTED = {"pose_matches": 5, "pose_residual_deg": 2.0}


def update_of(mdir: Path, shots: list[Shot], work: Path, unplaced: tuple[int, ...] = (),
              rebuild: api.Rebuild | None = None, unsupported: bool = False) -> Result:
    """``test_mapping_semantics.known_pose_update`` with keyframes that could not be placed
    (``unplaced``: their detections are numbered, nothing else of them is used), optionally the
    context of a rebuild, and keyframes whose poses the matches do not support
    (``unsupported``)."""
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
            if k not in unplaced:
                nf.record = store.FrameRecord(
                    idx, name, f"frames/{name}.jpg", f"shot {k}", 1, w, h, K, sh.pose, w, h,
                    pose_source="multiview" if unsupported else "sfm-global", update_id=uid,
                    stats=dict(UNSUPPORTED if unsupported else WELL),
                    low_confidence=unsupported)
                nf.depth = sh.depth
            new.append(nf)
        ctx = api.UpdateContext(tx, meta, old, new, uid, work, rebuild=rebuild)
        records, objs, geo = api.integrate(ctx, lambda m: None)
        meta.update(update_count=uid, next_frame_index=start + len(shots),
                    next_object_id=objs.next_id)
        tx.commit(meta)
    return Result(objs, geo.cloud, records, [store.frame_name(start + k)
                                              for k in range(len(shots))])


def labelled(res: Result) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for o in res.objs.exported():
        out.setdefault(o.label, []).append(o.id)
    return out


def test_the_detections_of_a_keyframe_that_could_not_be_placed_are_counted(
        tmp_path: Path) -> None:
    shots = shoot(Room(boxes=[CABINET, CUP]), FIRST + LATER)
    lost = shots[0]
    assert len(lost.dets) == 2
    plain = update_of(tmp_path / "plain", shots[1:], tmp_path / "w0")
    with_lost = update_of(tmp_path / "lost", shots, tmp_path / "w1", unplaced=(0,))
    assert len(with_lost.records) == len(shots) - 1  # the lost keyframe is not stored
    a, b = labelled(plain), labelled(with_lost)
    assert sorted(a) == sorted(b) == ["cabinet", "cup"]
    for label in a:  # the same objects, numbered after the lost keyframe's two detections
        assert [i + 2 for i in a[label]] == b[label]


def test_a_rebuild_carrying_no_published_id_numbers_its_objects_as_an_update_does(
        tmp_path: Path) -> None:
    shots = shoot(Room(boxes=[CABINET, CUP]), FIRST + LATER)
    plain = update_of(tmp_path / "plain", shots, tmp_path / "w0")
    rebuilt = update_of(tmp_path / "rebuilt", shots, tmp_path / "w1",
                        rebuild=api.Rebuild({}, {}, {}, 1, {}, {}, []))
    assert labelled(rebuilt) == labelled(plain)
    assert rebuilt.objs.rebuild_merged == {} and not rebuilt.objs.summary["removed"]


def test_a_cup_that_moves_and_is_then_gone_within_one_update_is_gone(tmp_path: Path) -> None:
    """The latest views see the cup's new place empty: the move is not kept, and its first
    place, which the later views saw empty, is judged as before — the cup is gone."""
    shots = (shoot(Room(boxes=[CABINET, CUP]), FIRST) + shoot(Room(boxes=[CABINET, MOVED]), LATER)
             + shoot(Room(boxes=[CABINET]), LATEST))
    res = update_of(tmp_path / "m", shots, tmp_path / "w")
    assert labelled(res) == {"cabinet": labelled(res)["cabinet"]}
    assert res.objs.summary["moved"] == [] and res.objs.summary["withdrawn"] == 2
    assert not any(o.label == "cup" for o in res.objs.objects)


def test_a_cup_moving_onto_a_cup_the_map_held_keeps_its_first_id(tmp_path: Path) -> None:
    """Update 1 maps a cup here; update 2 (keyframes whose poses its matches do not support, so
    they invalidate nothing update 1 saw) looks only at the place there, where it finds a cup;
    update 3 sees both places: the cup is there and not here, and update 1's views saw there
    empty. It moved: the cup keeps the id update 1 gave it, and the id update 2 gave the cup
    there resolves to it."""
    mdir = tmp_path / "m"
    r1 = update_of(mdir, shoot(Room(boxes=[CABINET, CUP]), FIRST), tmp_path / "w1")
    (first,) = labelled(r1)["cup"]
    away = THERE + np.array([-0.3, 0.3, 0.0])  # between the places, looking away from here
    close = [look_at(away + [dx, dx, 0.47], THERE) for dx in (0.0, 0.03, -0.03)]
    r2 = update_of(mdir, shoot(Room(boxes=[CABINET, MOVED]), close), tmp_path / "w2",
                   unsupported=True)
    cups = sorted(o.id for o in r2.objs.objects if o.label == "cup" and o.confirmed)
    assert len(cups) == 2 and first in cups  # here is out of view: nothing judges it
    (there,) = [c for c in cups if c != first]
    r3 = update_of(mdir, shoot(Room(boxes=[CABINET, MOVED]), LATER), tmp_path / "w3")
    assert r3.objs.summary["moved"] == [first]
    assert labelled(r3)["cup"] == [first]
    (cup,) = [o for o in r3.objs.objects if o.id == first]
    assert np.linalg.norm(cup.obb.center[:2] - THERE[:2]) < 0.05  # type: ignore[union-attr]
    assert r3.objs.merged_into[there] == first
    assert not (mdir / objects.points_file(there)).exists()
