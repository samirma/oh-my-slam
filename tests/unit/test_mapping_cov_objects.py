"""Persistent objects (``mapping.objects``) at their edges, piece by piece: empty point sets and
masks, sightings of keyframes the map no longer holds, retired pixels, loop copies, pieces of a
surface, parts named on their own, the places latest wins judges, moved objects' colours and the
ids a rebuild carries over. Keyframes are analytic views of a wall or a room (no SfM, no server)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from oh_my_slam.core import rle
from oh_my_slam.core.geometry import unproject_pixels
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.mapping import objects as ob
from oh_my_slam.mapping.objects import MapObject, ObjectState, Sighting, Verdict
from oh_my_slam.mapping.validity import View
from oh_my_slam.reconstruction.depth import DepthCorrection
from oh_my_slam.segmentation.api import OBB
from tests.synth.scene import look_at

K = Intrinsics(100.0, 100.0, 40.0, 30.0, 80, 60)


def wall(z: float = 3.0, pose: Pose | None = None, valid: bool = True) -> View:
    """A camera ``z`` m in front of a flat wall."""
    depth = np.full((60, 80), z, np.float32)
    return View(depth, np.full(depth.shape, valid), K, pose or Pose.identity())


def on_wall(view: View, mask: np.ndarray) -> np.ndarray:
    """Map points of the wall under ``mask``."""
    v, u = np.nonzero(mask)
    return view.T_map_cam.apply(unproject_pixels(u, v, view.depth[v, u].astype(float), K.K()))


def box_mask(r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    m = np.zeros((60, 80), bool)
    m[r0:r1, c0:c1] = True
    return m


def sighting(frame: int, pts: np.ndarray, confident: bool = True) -> Sighting:
    lo, hi = np.percentile(pts, [2, 98], axis=0)
    c = pts.mean(0)
    return Sighting(frame, len(pts), 0.0, (float(c[0]), float(c[1]), float(c[2])),
                    (float(lo[0]), float(lo[1]), float(lo[2])),
                    (float(hi[0]), float(hi[1]), float(hi[2])), confident)


def obj(oid: int, label: str, pts: np.ndarray, frames: list[int], obs_depth: float = 3.0,
        obb: OBB | None = None, sightings: list[Sighting] | None = None) -> MapObject:
    return MapObject(oid, label, {label: 1.0}, [0.9], np.asarray(pts, np.float32), obb,
                     frames=sorted(frames), confirmed=True, obs_depth=obs_depth,
                     sightings=sightings if sightings is not None else
                     [sighting(f, np.asarray(pts)) for f in frames])


def rec(i: int, pose: Pose, **kw: Any) -> SimpleNamespace:
    d = dict(index=i, name=f"f{i:06d}", image=f"frames/f{i:06d}.jpg", K_grid=K, T_map_cam=pose,
             low_confidence=False, update_id=1, pose_source="sfm-global",
             stats={"observations": 500.0, "reproj_error": 0.5})
    d.update(kw)
    return SimpleNamespace(**d)


def views_of(views: dict[int, View], records: list[Any] | None = None) -> ob._Views:
    records = records if records is not None else [rec(i, v.T_map_cam) for i, v in views.items()]
    return ob._Views(None, views, records)


# --- point sets, bookkeeping --------------------------------------------------------------------


def test_canonical_points_of_nothing_are_nothing() -> None:
    p, s = ob.canonical_sources(np.zeros((0, 3)), np.zeros(0))
    assert p.shape == (0, 3) and s.shape == (0,)


def test_absorbing_an_object_without_detections_keeps_the_viewing_distance() -> None:
    a = MapObject(1, "cup", {"cup": 1.0}, [0.9], np.zeros((0, 3), np.float32), obs_depth=2.0)
    b = MapObject(2, "cup", {"cup": 2.0}, [0.8], np.ones((3, 3), np.float32), obs_depth=5.0)
    a.absorb(b)
    assert a.obs_depth == 2.0 and a.label_votes == {"cup": 3.0} and len(a.points) == 1


def test_no_objects_file_no_vacated_places(tmp_path: Path) -> None:
    assert ob.load_vacated(lambda rel: tmp_path / rel) == []


def test_projected_iou_of_an_empty_mask_and_without_dilation() -> None:
    view = wall()
    m = box_mask(20, 40, 30, 50)
    pts = on_wall(view, m)
    assert ob.projected_iou(view, np.zeros((60, 80), bool), pts) == 0.0
    assert ob.projected_iou(view, m, pts, dilate=0) == pytest.approx(1.0)


def test_small_point_sets_have_no_extent_scale_or_box() -> None:
    one = np.zeros((1, 3))
    assert ob._extent(one) == 0.0
    o = obj(1, "cup", np.zeros((1, 3)), [0])
    assert ob._scale(o) == 0.0
    ob.refit(o, None)
    assert o.obb is None


def test_a_keyframe_without_usable_depth_shares_no_depth_ratio() -> None:
    assert ob.depth_ratio(wall(valid=False), wall()) is None


def test_contact_and_view_tests_of_nothing() -> None:
    view = wall()
    m = box_mask(10, 20, 10, 20)
    good = np.ones((60, 80), bool)
    assert not ob.continuous_contact(np.zeros((60, 80), bool), m, view.depth, good)
    assert not ob.in_view(np.zeros((0, 3)), K, Pose.identity())
    assert ob._agreeing([], 2.0) == []
    i, j = ob._near(np.zeros((0, 3)), np.zeros(0), np.ones((2, 3)), np.ones(2))
    assert len(i) == len(j) == 0
    assert ob._fits(obj(1, "cup", np.zeros((3, 3)), [0]), sighting(0, np.ones((4, 3))))


def test_rescaling_moves_sightings_of_held_keyframes_only() -> None:
    T = Pose.identity()
    s0 = Sighting(0, 100, 0.0, (0.0, 0.0, 2.0), (-0.1, -0.1, 1.9), (0.1, 0.1, 2.1))
    s5 = Sighting(5, 100, 0.0, (1.0, 0.0, 2.0), (0.9, -0.1, 1.9), (1.1, 0.1, 2.1))
    o = MapObject(3, "cup", {"cup": 1.0}, [0.9], np.zeros((0, 3), np.float32), frames=[0, 5],
                  obs_depth=2.0, sightings=[s0, s5])
    corr = DepthCorrection(float(np.log(1.1)), 0.0, float(np.log(2.0)))
    moved = ob.rescale_objects(ObjectState([o], 4), {0: corr}, [rec(0, T)])
    assert moved == {3}
    assert o.obs_depth == pytest.approx(2.2)  # keyframe 5 is not held: only keyframe 0 counts
    by_frame = {s.frame: s for s in o.sightings}
    assert by_frame[0].centroid == pytest.approx((0.0, 0.0, 2.2))
    assert by_frame[5] == s5 and len(o.points) == 0


def test_label_maps_skip_masks_of_another_grid() -> None:
    state = ObjectState([obj(4, "cup", np.zeros((3, 3)), [0])], 5)
    insts = [{"object_id": 4, "mask": rle.encode(np.ones((10, 10), bool))},
             {"object_id": 4, "mask": rle.encode(box_mask(0, 5, 0, 5))}]
    lab = ob.label_map_for(insts, (60, 80), state)
    assert lab.sum() == 4 * 25


# --- retired pixels and forgotten detections ----------------------------------------------------


def test_retiring_pixels_skips_keyframes_without_a_view_mask_or_valid_pixel() -> None:
    m = box_mask(20, 30, 20, 30)
    views = views_of({1: wall(), 2: wall(valid=False)},
                     [rec(0, Pose.identity()), rec(1, Pose.identity()), rec(2, Pose.identity())])
    masks = ob._Masks(None, views, {1: [(99, m)], 2: [(7, m)]}, {}, {})
    written: dict[str, bytes] = {}
    ctx = SimpleNamespace(tx=SimpleNamespace(write_bytes=written.__setitem__), update_id=3)
    o = obj(7, "cup", on_wall(wall(), m), [0, 1, 2])
    out, places = ob.retire_pixels(ctx, views, masks, [o], witnesses={})
    # keyframe 0 has no view, 1 no mask of the object, 2 no valid pixel under it; no witness
    assert out == {} and places == [] and written == {}


def test_forgetting_moved_objects_skips_keyframes_without_their_detections(
        tmp_path: Path) -> None:
    (tmp_path / "per_frame" / "f000002").mkdir(parents=True)
    (tmp_path / "per_frame" / "f000002" / "instances.json").write_text(json.dumps(
        {"instances": [{"object_id": 99, "mask": rle.encode(box_mask(0, 2, 0, 2))}]}))
    views = views_of({}, [rec(1, Pose.identity()), rec(2, Pose.identity())])
    written: dict[str, Any] = {}
    ctx = SimpleNamespace(tx=SimpleNamespace(current=lambda rel: tmp_path / rel,
                                             write_json=written.__setitem__))
    masks = ob._Masks(ctx, views, {}, {}, {})
    ob._forget_detections(ctx, views, masks, [obj(7, "cup", np.zeros((3, 3)), [0, 1, 2])], {})
    assert written == {}  # 0: no record, 1: no instances, 2: none of the object's


def test_earlier_masks_skip_keyframes_without_a_view_or_a_mask(tmp_path: Path) -> None:
    (tmp_path / "per_frame" / "f000002").mkdir(parents=True)
    (tmp_path / "per_frame" / "f000002" / "instances.json").write_text(json.dumps(
        {"instances": [{"object_id": 99, "mask": rle.encode(box_mask(0, 2, 0, 2))}]}))
    records = [rec(1, Pose.identity()), rec(2, Pose.identity())]
    ctx = SimpleNamespace(old_frames=records,
                          tx=SimpleNamespace(current=lambda rel: tmp_path / rel))
    views = views_of({2: wall()}, records)
    o = obj(7, "cup", np.zeros((3, 3)), [1, 2])
    early = ob._Earlier(ctx, ObjectState([o], 8), views)
    probe = SimpleNamespace(view=SimpleNamespace(T_map_cam=Pose.identity()), depth=3.0)
    assert early.masks(o, probe) == []  # type: ignore[arg-type]


# --- loop copies --------------------------------------------------------------------------------


def _cup(oid: int, at: tuple[float, float, float], frames: list[int], label: str = "cup"
         ) -> MapObject:
    rng = np.random.default_rng(oid)
    pts = np.asarray(at) + rng.uniform(-0.05, 0.05, (200, 3))
    return obj(oid, label, pts, frames, obs_depth=1.0)


def _loop_world() -> tuple[list[MapObject], ob._Views]:
    """Three cups seen from keyframes 0-1, each mapped again 0.3 m off by keyframes 10-11 (more
    than ``LOOP_VIEW_DIST`` away, their depth unknown); a fourth pair of copies seen elsewhere
    (keyframes 20-21 and 30-31); and a lamp seen from a keyframe half-way (5) and far away
    (7)."""
    x = {0: 0.0, 1: 0.05, 10: 1.07, 11: 1.12, 5: 0.56, 7: 9.0, 20: 20.0, 21: 20.05, 30: 21.1,
         31: 21.15}
    records = [rec(f, look_at(np.array([cx, -3.0, 1.0]), np.array([cx, 0.0, 1.0])))
               for f, cx in x.items()]
    shift = (0.3, 0.0, 0.0)
    objs = []
    for k in range(3):
        at = (3.0 * k, 0.0, 0.5)
        objs.append(_cup(1 + 2 * k, at, [0, 1]))
        objs.append(_cup(2 + 2 * k, tuple(np.add(at, shift)), [10, 11]))
    objs.append(_cup(20, (40.0, 0.0, 0.5), [20, 21]))
    objs.append(_cup(21, (40.3, 0.0, 0.5), [30, 31]))
    objs.append(_cup(30, (60.0, 0.0, 0.5), [5, 7], label="lamp"))
    return objs, views_of({}, records)


def test_copies_of_a_group_displaced_alike_across_a_loop_are_found() -> None:
    objs, views = _loop_world()
    copies = ob._loop_copies(objs, {o.id for o in objs}, views)
    # the fourth pair is alone where it is: nothing agrees with its offset
    assert copies == {frozenset((1, 2)), frozenset((3, 4)), frozenset((5, 6))}


def test_loop_copies_are_merged_once_no_other_merge_is_left() -> None:
    objs, views = _loop_world()
    state = ObjectState(objs, 40)
    alias: dict[int, int] = {}
    assert ob._merge(state, {o.id for o in objs}, alias, views) == 3
    assert alias == {2: 1, 4: 3, 6: 5}
    assert sorted(o.id for o in state.objects) == [1, 3, 5, 20, 21, 30]


def test_keyframes_whose_depth_agrees_nowhere_tell_nothing() -> None:
    _, views = _loop_world()
    s = Sighting(0, 10, 0.0, (0, 0, 0), (0, 0, 0), (1, 1, 1))
    assert ob._disagree(views, [(s, s)]) is None


def test_objects_without_posed_sightings_are_not_depth_explained() -> None:
    a, b = _cup(1, (0.0, 0.0, 0.5), [0, 1]), _cup(2, (0.3, 0.0, 0.5), [2, 3])
    assert ob._depth_explained(a, b, views_of({}, [])) == 0.0


# --- pieces of one surface ----------------------------------------------------------------------


def _slab(x0: float, x1: float, z: float, step: float = 0.01) -> np.ndarray:
    gx, gy = np.meshgrid(np.arange(x0, x1, step), np.arange(0.0, 0.4, step))
    return np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)])


def _piece(oid: int, pts: np.ndarray) -> MapObject:
    return obj(oid, "table", pts, [oid], obs_depth=1.0)


def test_the_fused_surface_joins_pieces_only_on_one_horizontal_patch_off_the_floor() -> None:
    a, b = _piece(1, _slab(0.0, 0.3, 0.75)), _piece(2, _slab(0.5, 0.8, 0.75))
    bridge = _slab(-0.1, 0.9, 0.75)
    assert ob._Surfaces(bridge, None).joined(a, b)
    assert not ob._Surfaces(bridge, floor_z=0.7).joined(a, b)  # at floor height
    far = _piece(3, _slab(2.0, 2.3, 0.75))
    assert not ob._Surfaces(bridge, None).joined(a, far)  # more than BRIDGE_GAP_M apart
    assert not ob._Surfaces(bridge[:40], None).joined(a, b)  # too little surface
    clump = np.full((80, 3), [0.4, 0.2, 0.75]) + np.random.default_rng(0).uniform(0, 0.005,
                                                                                  (80, 3))
    assert not ob._Surfaces(clump, None).joined(a, b)  # one centimetre of it
    gy, gz = np.meshgrid(np.arange(0.0, 0.4, 0.01), np.arange(0.72, 0.78, 0.005))
    upright = np.column_stack([np.full(gy.size, 0.4), gy.ravel(), gz.ravel()])
    assert not ob._Surfaces(upright, None).joined(a, b)  # not horizontal
    beside = _slab(-0.1, 0.9, 0.75) + [0.0, 0.0, 0.04]  # off the pieces' own points
    assert not ob._Surfaces(beside, None).joined(a, b)


def test_pieces_spread_over_two_patches_are_not_joined_by_either() -> None:
    a = _piece(1, np.vstack([_slab(0.0, 0.12, 0.75), _slab(0.5, 0.8, 0.75)]))
    b = _piece(2, _slab(0.0, 0.3, 0.75))
    two = np.vstack([_slab(-0.1, 0.32, 0.75), _slab(0.45, 0.9, 0.75)])  # a gap between them
    assert not ob._Surfaces(two, None).joined(a, b)


# --- detections, seen twice, parts --------------------------------------------------------------


def test_merge_chains_that_loop_end_where_they_started() -> None:
    masks = ob._Masks(None, views_of({}, []), {}, {1: 2, 2: 1}, {})
    assert masks.owner(1) == 1
    assert ob._grow(np.zeros((60, 80), bool), wall()) is None


def test_judging_needs_the_keyframe_and_its_mask() -> None:
    o = obj(7, "cup", np.zeros((3, 3)), [4])
    masks = ob._Masks(None, views_of({}, []), {}, {}, {})
    assert ob._judged_in(o, 4, np.zeros((5, 3)), masks).tolist() == [0.0, 0.0, 0.0, 0.0]
    assert ob._ray_origin(o, views_of({}, [])) is None


def test_objects_seen_from_keyframes_the_map_does_not_hold_are_not_seen_as_one() -> None:
    view = wall()
    ma, mb = box_mask(20, 30, 20, 30), box_mask(20, 30, 30, 40)
    grow = box_mask(15, 35, 15, 45)
    a, b = obj(1, "lamp", on_wall(view, ma), [1]), obj(2, "vase", on_wall(view, mb), [2])
    views = ob._Views(None, {1: view, 2: view}, [])  # views, but no poses
    masks = ob._Masks(None, views, {1: [(1, grow)], 2: [(2, grow)]}, {}, {})
    assert ob._judged(a, ob._seen_samples(b), masks)[3] >= ob.SEEN_EVIDENCE
    assert ob._seen_twice(a, b, masks) == 0.0


def test_a_keyframe_sees_points_only_in_its_frame() -> None:
    view = wall()
    assert not ob._sees(view, np.zeros((0, 3)))
    assert not ob._sees(view, np.array([[0.0, 0.0, -2.0]] * 20))  # behind it


def _panel_world() -> dict[str, Any]:
    """A whole (a 1.5 m panel on the wall 3 m ahead) and a part of it (a third of its size), each
    detected in two keyframes looking at it from where the cameras stand."""
    eye = look_at(np.array([0.5, -3.0, 0.5]), np.array([0.5, 0.0, 0.5]))
    view = wall(3.0, eye)
    whole_mask, part_mask = box_mask(10, 50, 15, 65), box_mask(22, 38, 32, 48)
    whole_pts, part_pts = on_wall(view, whole_mask), on_wall(view, part_mask)
    box = OBB(whole_pts.mean(0), np.eye(3), np.ptp(whole_pts, axis=0) + 0.1)
    part = obj(1, "knob", part_pts, [1, 2])
    whole = obj(2, "cabinet", whole_pts, [3, 4], obb=box)
    far = wall(10.0, look_at(np.array([0.5, -10.0, 0.5]), np.array([0.5, 0.0, 0.5])))
    views = {1: view, 2: view, 3: view, 4: view, 5: far}
    records = [rec(f, v.T_map_cam) for f, v in views.items()]
    masks = {f: [(1, part_mask)] for f in (1, 2)} | {f: [(2, whole_mask)] for f in (3, 4, 5)}
    return dict(part=part, whole=whole, views=views, records=records, masks=masks, box=box)


def test_a_part_named_on_its_own_lies_on_the_whole_its_keyframes_saw() -> None:
    w = _panel_world()
    views = ob._Views(None, w["views"], w["records"])
    assert ob._part_of(w["part"], w["whole"], views) >= 1.0  # without the detections
    masks = ob._Masks(None, views, w["masks"], {}, {})
    assert ob._part_of(w["part"], w["whole"], views, masks) >= 1.0


@pytest.mark.parametrize("change", ["no box", "no poses", "part unseen", "whole far",
                                    "one judge", "beside"])
def test_a_part_is_not_merged_without_the_evidence(change: str) -> None:
    w = _panel_world()
    part, whole, masks = w["part"], w["whole"], dict(w["masks"])
    views, records = dict(w["views"]), list(w["records"])
    if change == "no box":
        whole.obb = None
    elif change == "no poses":
        records = [r for r in records if r.index not in (1, 2)]
    elif change == "part unseen":
        del views[2]
    elif change == "whole far":
        whole.frames = [3, 5]
    elif change == "one judge":
        masks[4] = []
    else:  # the whole's detection in keyframe 4 lies elsewhere
        masks[4] = [(2, box_mask(1, 10, 1, 10))]
    v = ob._Views(None, views, records)
    assert ob._part_of(part, whole, v, ob._Masks(None, v, masks, {}, {})) == 0.0


def test_merge_strength_of_loop_copies_and_of_objects_seen_twice() -> None:
    a, b = _cup(1, (0.0, 0.0, 0.5), [0, 1]), _cup(2, (0.3, 0.0, 0.5), [2, 3])
    a.obb = None
    assert ob._merge_strength(a, b) < 1.0
    assert ob._merge_strength(a, b, loops={frozenset((1, 2))}) == 1.0
    masks = ob._Masks(None, views_of({}, []), {}, {}, {})
    assert ob._merge_strength(a, b, masks=masks) < 1.0  # seen by nothing: not seen as one


# --- the places latest wins judges --------------------------------------------------------------


def _place_world(cuts: dict[int, float] | None = None, **judges: Any
                 ) -> tuple[ob._Places, MapObject]:
    """A cup on the wall 3 m ahead detected by keyframes 0-1 (mask in the image's middle) and
    judging keyframes 10+ (``judges``: index -> View, or (View, record keywords))."""
    eye = Pose.identity()
    view = wall(3.0, eye)
    m = box_mask(25, 35, 35, 45)
    cup = obj(7, "cup", on_wall(view, m), [0, 1])
    views, records = {0: view, 1: view}, [rec(0, eye), rec(1, eye)]
    for name, j in judges.items():
        i = int(name[1:])
        v, kw = j if isinstance(j, tuple) else (j, {})
        if v is not None:
            views[i] = v
        records.append(rec(i, kw.pop("pose", v.T_map_cam if v is not None else eye), **kw))
    v = ob._Views(None, views, records)
    masks = ob._Masks(None, v, {0: [(7, m)], 1: [(7, m)]}, {}, {})
    return ob._Places(v, masks, [cup], cuts=cuts), cup


def test_transparency_and_foot_of_keyframes_that_cannot_judge() -> None:
    away = wall(3.0, Pose(np.diag([-1.0, 1.0, -1.0]), np.zeros(3)))  # looking the other way
    places, cup = _place_world(j10=away)
    cup.frames = [0, 99]  # 99: a keyframe the map does not hold
    assert places.transparency(cup) == (0.0, 0.0)
    places, cup = _place_world()
    cup.frames = [10, 0]
    places.views.views[10] = away
    places.views.records[10] = rec(10, away.T_map_cam)
    assert places.transparency(cup) == (0.0, 0.0)
    empty = obj(8, "cup", np.zeros((0, 3)), [0], sightings=[])
    assert places.foot(empty) == (0.0, 0.0)


def test_rings_around_detections_need_a_view_and_valid_surroundings() -> None:
    places, cup = _place_world()
    assert places.ring(cup, 5) is None  # no such keyframe
    places.masks.frames[1] = [(7, box_mask(28, 31, 38, 41))]  # a small detection
    ring = places.ring(cup, 1)
    assert ring is not None and ob.RING_MIN <= len(ring) <= ob.RING_SAMPLES
    blind = wall(valid=False)
    blind.valid[25:35, 35:45] = True  # valid only under the detection
    places.views.prior[0] = blind
    assert places.ring(cup, 0) is None


def test_occupancy_footprints_and_overlaps_of_nothing() -> None:
    places, cup = _place_world()
    assert not places.occupied(cup, 0, np.zeros((0, 2)))
    other = obj(8, "vase", np.zeros((3, 3)), [0])
    assert places._footprint(other, 0) is None
    assert not places._overlap(cup, other, 0)


def test_a_judge_whose_depth_disagrees_around_the_object_gives_no_local_ratio() -> None:
    away = wall(3.0, Pose(np.diag([-1.0, 1.0, -1.0]), np.zeros(3)))
    places, cup = _place_world(j98=(None, {}))  # keyframe 98: posed, but no view
    cup.frames = [0, 1, 98, 99]
    assert places.local_ratio(cup, 10, away) is None  # sees none of the ring
    assert places.local_ratio(cup, 10, wall(3.8)) is None  # 27 % deeper around it


def test_fused_needs_a_posed_sighting_within_its_keyframes_cut() -> None:
    places, cup = _place_world(cuts={0: 10.0})
    cup.sightings = [Sighting(99, 10, 0.0, (0, 0, 3), (0, 0, 3), (0, 0, 3)),
                     *cup.sightings]
    assert places.fused(cup)  # keyframe 0's sighting; 99 has no pose, 1 no cut


def test_verdicts_need_the_judges_view_and_a_detectable_distance() -> None:
    far = wall(9.0, Pose(np.eye(3), np.array([0.0, 0.0, -6.0])))  # 9 m from the cup
    places, cup = _place_world(j10=(None, {}), j11=far)
    assert places.verdict(cup, 10) is None  # no view of its own
    assert places.verdict(cup, 11, detectable=True) is None  # three times as far as detected


def test_a_keyframe_that_does_not_see_the_object_sees_through_none_of_it() -> None:
    away = wall(3.0, Pose(np.diag([-1.0, 1.0, -1.0]), np.zeros(3)))
    places, cup = _place_world(j10=away)
    assert not places.through_pixels(cup, Verdict(10, 1.0, True)).any()


# --- rebuilt ids ----------------------------------------------------------------------------------


def test_identity_carried_over_keeps_the_maps_merges_of_ids_now_gone() -> None:
    state = ObjectState([obj(5, "cup", np.zeros((3, 3)), [0]),
                         obj(6, "vase", np.zeros((3, 3)), [0])], 10)
    rb = SimpleNamespace(merged_into={9: 5, 6: 5}, created={5: 1})
    gone = ob._carry_identity(state, rb, {}, {}, set())
    assert state.merged_into == {9: 5} and gone == []


# --- moved objects ------------------------------------------------------------------------------


def test_objects_of_very_different_sizes_are_no_move() -> None:
    rng = np.random.default_rng(4)
    box = OBB(np.array([0.0, 0.0, 0.5]), np.eye(3), np.array([0.3, 0.3, 0.3]))
    big = obj(1, "cup", rng.uniform(-0.15, 0.15, (400, 3)) + (0.0, 0.0, 0.5), [0], obb=box)
    small = obj(2, "cup", rng.uniform(-0.02, 0.02, (400, 3)) + (2.0, 0.0, 0.5), [5], obb=box)
    places = SimpleNamespace(views=SimpleNamespace(records={}))
    assert ob._moves(ObjectState([big, small], 10), {2}, places, None) == []  # type: ignore[arg-type]


def test_colours_of_detections_without_a_record_image_or_pixel(tmp_path: Path) -> None:
    records = [rec(1, Pose.identity()), rec(2, Pose.identity()), rec(3, Pose.identity())]
    views = views_of({}, records)
    m = box_mask(20, 30, 20, 30)
    masks = ob._Masks(None, views, {1: [(7, m)], 2: [(7, m)], 3: [(7, np.zeros((60, 80), bool))]},
                      {}, {})
    unplaced = SimpleNamespace(record=None, frame=SimpleNamespace(rgb=None))
    placed = SimpleNamespace(record=SimpleNamespace(index=3),
                             frame=SimpleNamespace(rgb=np.zeros((60, 80, 3), np.uint8)))
    ctx = SimpleNamespace(new=[unplaced, placed],
                          tx=SimpleNamespace(current=lambda rel: tmp_path / rel))
    colours = ob._Colours(ctx, views, masks)
    assert list(colours.rgb) == [3]
    # 0: no record; 1, 2: no image on disk; 3: an empty mask
    o = obj(7, "cup", np.zeros((3, 3)), [0, 1, 3])
    assert colours.of(o) is None and colours.of(o) is None  # cached
    assert ob._Colours(None, views, masks)._image(1, (60, 80)) is None


class _FakePlaces:
    """``_Places`` as ``_moves`` uses it: the keyframes' order and fixed verdicts."""

    def __init__(self, frames: list[int], share: float) -> None:
        self.views = SimpleNamespace(records={f: None for f in frames})
        self.share = share

    def verdicts(self, o: MapObject, frames: list[int], detectable: bool = False
                 ) -> list[Verdict]:
        return [Verdict(f, self.share, True) for f in frames]


class _FakeColours:
    def __init__(self, by_id: dict[int, Any]) -> None:
        self.by_id = by_id

    def of(self, o: MapObject) -> Any:
        return self.by_id.get(o.id)


def _moving() -> tuple[ObjectState, list[MapObject]]:
    box = OBB(np.zeros(3), np.eye(3), np.array([0.1, 0.1, 0.1]))
    objs = []
    for oid, x, frames in ((1, 0.0, [0, 1, 2]), (2, 2.0, [5, 6]), (3, 4.0, [7, 8])):
        pts = np.random.default_rng(oid).uniform(-0.05, 0.05, (100, 3)) + [x, 0.0, 0.0]
        o = obj(oid, "cup", pts, frames, obb=OBB(np.array([x, 0.0, 0.0]), box.R, box.size))
        o.sightings = [Sighting(f, 100, 0.0, (x, 0.0, 0.0), (x - .05,) * 3, (x + .05,) * 3)
                       for f in frames]
        objs.append(o)
    return ObjectState(objs, 10), objs


def test_an_object_moves_once_to_the_best_coloured_place_seen_empty_before() -> None:
    state, (a, b, c) = _moving()
    grey = np.array([60.0, 0.0, 0.0])
    places = _FakePlaces(list(range(10)), share=0.9)
    # a would move to b or c: b's colour agrees better, so a moves there; c finds a taken
    colours = _FakeColours({1: grey, 2: grey + [0, 1.0, 0], 3: grey + [0, 5.0, 0]})
    moves = ob._moves(state, {2, 3}, places, colours)  # type: ignore[arg-type]
    assert [(m.src.id, m.dst.id) for m in moves] == [(1, 2)]
    assert ob._moves(state, {2, 3}, places, _FakeColours({3: grey})) == []  # type: ignore[arg-type]
    still = _FakePlaces(list(range(10)), share=0.1)  # both places seen as they were
    assert ob._moves(state, {2, 3}, still, colours) == []  # type: ignore[arg-type]


def test_centroids_too_far_apart_pair_nothing() -> None:
    far = ob._near(np.zeros((1, 3)), np.full(1, 0.1), np.array([[5.0, 0.0, 0.0]]), np.full(1, 0.1))
    assert all(len(x) == 0 for x in far)
