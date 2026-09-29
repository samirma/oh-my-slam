"""Mapping semantics of §2.3, "later wins", on a rendered scene with known poses (no SfM, no
server): a cup that is gone when the camera comes back to it.

The camera stands in one place and pans: it looks at a cup and a cabinet (``A``), looks away at a
wall (``B``), and looks at them again (``C``), now without the cup. The cup is removed from the map
— its object and its points in the cloud — whether the views are mapped in two updates (``A + B``,
then ``C``) or in one (``A + B + C``: the order of addition is the only sign of "latest"), and the
cabinet, which never changed, keeps its id, label, colour and box. Keyframes that watch a place
continuously are one observation of it: the same views without the look away change nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from oh_my_slam.core.types import Pose
from oh_my_slam.mapping import objects, validity
from oh_my_slam.mapping.objects import MapObject
from oh_my_slam.segmentation.api import obb_iou_upright
from tests.synth.scene import Box, Room, look_at, render
from tests.unit.test_mapping_semantics import (
    K,
    Result,
    exported,
    known_pose_update,
    shoot,
)

CUP = Box(np.array([-0.4, 0.1, 0.08]), np.array([0.14, 0.14, 0.16]), 0.3, (240, 240, 240), "cup")
CABINET = Box(np.array([0.9, 0.9, 0.4]), np.array([0.9, 0.5, 0.8]), 0.2, (220, 40, 40), "cabinet")
EYE = np.array([-1.6, 0.3, 1.2])
WITH_CUP, WITHOUT = Room(boxes=[CABINET, CUP]), Room(boxes=[CABINET])
# looking at the cup and the cabinet; looking away at the wall behind the camera
AT = [look_at(EYE + [0.0, dy, 0.0], np.array([0.3, 0.4, 0.3])) for dy in (0.0, 0.06, -0.06)]
AWAY = [look_at(EYE, np.array([-2.9, 0.3 + dy, 1.0])) for dy in (-0.3, 0.0, 0.3)]


def _cup_points(res: Result) -> int:
    """Cloud points on the cup: above the floor, within the cup's footprint."""
    xyz = res.cloud.xyz
    near = (np.abs(xyz[:, :2] - CUP.center[:2]).max(axis=1) < 0.15) & (xyz[:, 2] > 0.03)
    return int(near.sum())


def _same_object(a: Any, b: Any) -> None:
    assert (a.id, a.label, a.color) == (b.id, b.label, b.color)
    assert obb_iou_upright(a.obb, b.obb) > 0.9
    assert np.linalg.norm(a.obb.center - b.obb.center) < 0.05


def _scene() -> tuple[list[Any], list[Any]]:
    return shoot(WITH_CUP, AT + AWAY), shoot(WITHOUT, AT)


def test_a_cup_gone_in_a_later_update_leaves_the_object_and_the_cloud(tmp_path: Path) -> None:
    before, after = _scene()
    mdir = tmp_path / "m"
    r1 = known_pose_update(mdir, before, tmp_path / "w1")
    assert {o.label for o in r1.objs.exported()} == {"cabinet", "cup"} and _cup_points(r1) > 100
    (cup,) = [o for o in r1.objs.objects if o.label == "cup"]
    (cabinet,) = [o for o in r1.objs.exported() if o.label == "cabinet"]
    r2 = known_pose_update(mdir, after, tmp_path / "w2")
    assert [o.label for o in r2.objs.exported()] == ["cabinet"]
    assert r2.objs.summary["removed"] == [cup.id] and cup.id not in r2.objs.by_id()
    assert not (mdir / objects.points_file(cup.id)).exists()
    assert _cup_points(r2) == 0 and not (r2.cloud.label == cup.id).any()
    # what never changed keeps its id, label, colour and box
    _same_object(cabinet, exported(r2)[cabinet.id])


def test_a_cup_gone_in_the_last_views_of_one_update_is_not_in_the_map(tmp_path: Path) -> None:
    before, after = _scene()
    kept = known_pose_update(tmp_path / "kept", shoot(WITH_CUP, AT + AWAY + AT), tmp_path / "wk")
    assert {o.label for o in kept.objs.exported()} == {"cabinet", "cup"}
    assert _cup_points(kept) > 100
    one = known_pose_update(tmp_path / "one", before + after, tmp_path / "w1")
    assert [o.label for o in one.objs.exported()] == ["cabinet"]
    assert one.objs.summary["withdrawn"] == 1 and not one.objs.summary["removed"]
    assert _cup_points(one) == 0
    (cabinet,) = [o for o in one.objs.exported()]
    assert set(np.unique(one.cloud.label)) <= {0, cabinet.id}
    (was,) = [o for o in kept.objs.exported() if o.label == "cabinet"]
    _same_object(was, cabinet)


def test_one_update_and_two_end_in_the_same_map(tmp_path: Path) -> None:
    before, after = _scene()
    one = known_pose_update(tmp_path / "one", before + after, tmp_path / "w1")
    split = tmp_path / "split"
    known_pose_update(split, before, tmp_path / "w2")
    two = known_pose_update(split, after, tmp_path / "w3")
    a, b = exported(one), exported(two)
    assert sorted(a) == sorted(b) == [1]
    for res in (one, two):  # every confirmed object is exported: none is lost to the cloud
        assert {o.id for o in res.objs.objects if o.confirmed} == {1}
    _same_object(a[1], b[1])
    assert one.objs.next_id == two.objs.next_id  # ids count detections, not what was removed


def _place_of_the_cup(res: Result) -> tuple[np.ndarray, np.ndarray]:
    """(floor points where the cup stood, floor points of a ring around it), with colours."""
    xyz = res.cloud.xyz
    d = np.abs(xyz[:, :2] - CUP.center[:2]).max(axis=1)
    floor = np.abs(xyz[:, 2]) < 0.02
    return (d < 0.06) & floor, (d >= 0.12) & (d < 0.2) & floor


def test_the_place_of_a_removed_cup_is_drawn_from_the_latest_views(tmp_path: Path) -> None:
    """The cup, detected in 4 views, is removed by the 2 latest views, which see the empty floor
    where it stood. Its pixels are retired from the 4 views, so only the 2 latest see that floor:
    it is drawn from them anyway, as densely as the floor around it and in the floor's colour,
    and nothing of the cup is left, in one update or in two."""
    at4 = AT + [look_at(EYE + [0.0, 0.12, 0.0], np.array([0.3, 0.4, 0.3]))]
    late = [look_at(EYE + [0.0, dy, 0.0], np.array([0.3, 0.4, 0.3])) for dy in (0.03, -0.03)]
    before, after = shoot(WITH_CUP, at4 + AWAY), shoot(WITHOUT, late)
    one = known_pose_update(tmp_path / "one", before + after, tmp_path / "w1")
    split = tmp_path / "split"
    known_pose_update(split, before, tmp_path / "w2")
    two = known_pose_update(split, after, tmp_path / "w3")
    for res, mdir in ((one, tmp_path / "one"), (two, split)):
        assert [o.label for o in res.objs.exported()] == ["cabinet"]
        assert _cup_points(res) == 0
        place, ring = _place_of_the_cup(res)
        density = place.sum() / 0.12 ** 2, ring.sum() / (0.4 ** 2 - 0.24 ** 2)
        assert density[0] > 0.8 * density[1] > 0
        floor = np.array(WITHOUT.floor_color, float)
        colours = res.cloud.rgb[place].astype(float)
        assert np.abs(colours.mean(axis=0) - floor).max() < 40  # the floor, not the cup
        (vacated,) = json.loads((mdir / objects.OBJECTS_JSON).read_text())["vacated"]
        assert sorted(vacated["witnesses"]) == sorted(res.names[-2:])


def test_a_cup_gone_while_the_camera_watches_it_is_gone(tmp_path: Path) -> None:
    """The same views without the look away in between: the cup vanishes while the camera watches
    its place. The latest views win whether or not the camera looked away."""
    before, after = _scene()
    res = known_pose_update(tmp_path / "m", before[:3] + after, tmp_path / "w")
    assert [o.label for o in res.objs.exported()] == ["cabinet"]
    assert res.objs.summary["withdrawn"] == 1 and _cup_points(res) == 0


def test_views_whose_depth_disagrees_do_not_remove_a_cup_that_stays(tmp_path: Path) -> None:
    """Monocular depth of two keyframes disagrees by a few percent (here each view's depth is
    scaled by up to ±8 %): views that watch a cup that stays, or see it again from elsewhere, do
    not see through it."""
    far = [look_at(np.array([1.9, -1.8, 1.4 + dz]), np.array([-0.3, 0.1, 0.1]))
           for dz in (0.0, 0.05, 0.1)]
    shots = shoot(WITH_CUP, AT + AWAY + AT + far, depth_noise=0.08, seed=3)
    res = known_pose_update(tmp_path / "m", shots, tmp_path / "w")
    assert {o.label for o in res.objs.exported()} == {"cabinet", "cup"}
    assert not res.objs.summary["removed"] and not res.objs.summary["withdrawn"]


def test_a_cup_gone_when_the_camera_comes_back_from_another_side_is_gone(tmp_path: Path) -> None:
    """After the camera looks away it comes back to the room from a different corner, 3.5 m away
    instead of 1.9 m: it sees the cup's place, and the floor there, from another viewpoint and
    distance. Its depth is compared with the cup's detections where both see the same
    surroundings, so the latest views win from there too, in one update or in two."""
    far = [look_at(np.array([1.9, -1.8, 1.4 + dz]), np.array([-0.3, 0.1, 0.1]))
           for dz in (0.0, 0.05, 0.1)]
    before, _ = _scene()
    one = known_pose_update(tmp_path / "one", before + shoot(WITHOUT, far), tmp_path / "w1")
    assert "cup" not in {o.label for o in one.objs.objects} and _cup_points(one) == 0
    split = tmp_path / "split"
    first = known_pose_update(split, before, tmp_path / "w2")
    (cup,) = [o for o in first.objs.objects if o.label == "cup"]
    two = known_pose_update(split, shoot(WITHOUT, far), tmp_path / "w3")
    assert two.objs.summary["removed"] == [cup.id] and _cup_points(two) == 0


# --- the pieces -------------------------------------------------------------------------------------


def _record(index: int, pose: Pose) -> Any:
    return SimpleNamespace(index=index, K_grid=K, T_map_cam=pose, low_confidence=False,
                           stats={"observations": 500.0, "reproj_error": 0.5},
                           pose_source="sfm-global")


def _cup_object(frames: list[int]) -> MapObject:
    rng = np.random.default_rng(0)
    local = rng.uniform(-0.5, 0.5, (2000, 3)) * CUP.size
    face = np.argmax(np.abs(local) / CUP.size, axis=1)
    local[np.arange(len(local)), face] = np.sign(local[np.arange(len(local)), face]) \
        * CUP.size[face] / 2
    pts = (local + CUP.center).astype(np.float32)
    pts = pts[pts[:, 2] > CUP.center[2]]  # the upper half: the surface a camera above sees
    return MapObject(7, "cup", {"cup": 3.0}, [0.9] * 3, pts, frames=frames, confirmed=True,
                     obs_depth=1.5)


def _depth(room: Room, pose: Pose, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    d = render(room, pose, K).depth * scale
    return d, d > 0


def _places(judges: dict[int, tuple[Room, Pose, float]], detected: float = 1.0
            ) -> tuple[Any, MapObject]:
    """``_Places`` over the cup's three detecting views (``AT``, the cup in place: keyframes 0-2;
    their depth scaled by ``detected``) and the judging views ``judges`` (index -> room, pose,
    depth scale)."""
    cup = _cup_object([0, 1, 2])
    views, records, instances = {}, [], {}
    for i, pose in enumerate(AT):
        r = render(WITH_CUP, pose, K)
        views[i] = validity.View(r.depth * detected, r.depth > 0, K, pose)
        records.append(_record(i, pose))
        instances[i] = [(cup.id, r.ids == 3)]
    for i, (room, pose, scale) in judges.items():
        views[i] = validity.View(*_depth(room, pose, scale), K, pose)
        records.append(_record(i, pose))
    v = objects._Views(None, views, records)
    return objects._Places(v, objects._Masks(None, v, instances, {}, {}), [cup]), cup


def test_a_keyframe_judges_a_place_it_sees_as_the_detections_saw_its_surroundings() -> None:
    far = look_at(np.array([1.9, -1.8, 1.4]), np.array([-0.3, 0.1, 0.1]))
    wall = Room(boxes=[CABINET, Box(np.array([-1.0, 0.2, 0.6]), np.array([0.1, 1.0, 1.0]), 0.0,
                                    (60, 60, 200), "screen")])
    places, cup = _places({10: (WITHOUT, AT[0], 1.0), 11: (WITH_CUP, AT[0], 1.0),
                           12: (WITHOUT, AT[0], 1.12), 13: (WITH_CUP, AT[0], 1.12),
                           14: (WITHOUT, far, 1.0), 15: (wall, AT[0], 1.0),
                           16: (WITHOUT, AWAY[1], 1.0), 17: (WITHOUT, AT[0], 1.6)})
    gone, here = places.verdict(cup, 10), places.verdict(cup, 11)
    assert gone is not None and gone.share > 0.8 and here is not None and here.share < 0.2
    # a keyframe whose depth is 12 % deeper everywhere: its depth is divided by that ratio
    deeper, deeper_here = places.verdict(cup, 12), places.verdict(cup, 13)
    assert deeper is not None and deeper.share > 0.8 and abs(deeper.ratio - 1.12) < 0.03
    assert deeper_here is not None and deeper_here.share < 0.2
    far_verdict = places.verdict(cup, 14)  # another corner, 3.5 m away
    assert far_verdict is not None and far_verdict.share >= objects.REMOVE_FRACTION
    # hidden behind a screen, out of view, or a depth that disagrees with the detections' about
    # the cup's surroundings by more than the depth noise: no verdict
    assert places.verdict(cup, 15) is None and places.verdict(cup, 16) is None
    assert places.verdict(cup, 17) is None


def test_a_small_object_is_seen_through_by_a_margin_of_its_size() -> None:
    z = np.array([1.5])
    assert objects.absence_tau(z, 0.2)[0] < 0.5 * objects.absence_tau(z)[0]
    assert objects.absence_tau(z, 2.0)[0] == objects.absence_tau(z)[0]  # a large object: as before


def test_a_keyframe_must_see_through_more_than_the_detections_do() -> None:
    """The cup's detecting keyframes see 15 % farther than the points it was given (its lifted
    points disagree with their depth, as monocular depth of a small object, or of a see-through
    one, varies between keyframes), so they see through most of it themselves: a keyframe that
    sees the cup where they do is no evidence that it is gone, though it sees behind its points
    once its depth is divided by its ratio to those keyframes."""
    places, cup = _places({10: (WITH_CUP, AT[0], 1.0), 11: (WITHOUT, AT[0], 1.0)}, detected=1.15)
    assert places.transparency(cup) > 0.5
    here = places.verdict(cup, 10)
    assert here is not None and abs(here.ratio - 1 / 1.15) < 0.03
    assert here.share < objects.REMOVE_FRACTION
    clean, cup = _places({10: (WITH_CUP, AT[0], 1.0)})
    assert clean.transparency(cup) < 0.05


def test_a_pose_uncertain_by_a_few_degrees_judges_only_what_is_large_against_the_error() -> None:
    """A multi-view keyframe whose feature matches leave its pose 2° uncertain is not well
    registered, but 2° is 2.5 cm at a cup 0.7 m away: it may judge the cup, not a 5 cm object."""
    rec = _record(0, AT[0])
    rec.pose_source = "multiview"
    rec.stats = {"pose_matches": 1200, "pose_residual_deg": 2.0}
    cup = _cup_object([0, 1, 2, 3])
    cup.obs_depth = 0.7
    assert not validity.well_registered(rec.stats, rec.pose_source)
    size = float(np.linalg.norm(np.ptp(cup.points, axis=0)))
    assert objects._judges(rec, cup, size)
    assert not objects._judges(rec, cup, 0.05)
    rec.pose_source = "sfm-incremental+multiview"  # a map that joins SfM and multi-view keyframes
    assert objects._judges(rec, cup, size)
    rec.stats = {"pose_matches": 5, "pose_residual_deg": 0.3}  # too few matches: never
    assert not objects._judges(rec, cup, size)
    rec.stats = {"observations": 29.0, "reproj_error": 1.8}  # a weak SfM pose: never
    assert not objects._judges(rec, cup, size)


def test_a_map_of_a_few_keyframes_panned_from_one_spot_has_no_sfm_scale(monkeypatch: Any) -> None:
    """Keyframes panned from one spot triangulate too few points to fix the metric scale: their
    SfM units are arbitrary, so the mapper takes the multi-view poses, which are in metres. The
    unscaled map made every later keyframe misaligned and low confidence, and the map exported
    none of its objects."""
    from oh_my_slam.mapping import api

    ctx = SimpleNamespace(new=[])

    def too_few(model: Any, frames: Any) -> None:
        raise ValueError("no frame had enough triangulated points to fix the metric scale")

    monkeypatch.setattr(api.mframe, "metric_scale", too_few)
    assert not api._metric_scale_known(SimpleNamespace(), ctx)
    monkeypatch.setattr(api.mframe, "metric_scale", lambda model, frames: SimpleNamespace(scale=1.0))
    assert api._metric_scale_known(SimpleNamespace(), ctx)
