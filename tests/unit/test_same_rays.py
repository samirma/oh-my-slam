"""Objects on the same rays (``mapping.objects._same_rays``): a camera turning in place cannot tell
an object's depth along its rays; keyframes whose detections lifted a chair and keyframes whose
masks held only the wall seen through its back (``Lifted.through``) make two objects of one. They
are one when each lies on the other's detections from keyframes that cannot tell their depths
apart, and the merged object is placed by the detections that show its own surface
(``shaping_sightings``). Randomised over image size, focal length, distance and the depth ratio
between the two placements."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy import ndimage

from oh_my_slam.core.geometry import project, rot_z
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.mapping import objects as ob
from oh_my_slam.mapping.objects import MapObject, ObjectState, Sighting
from oh_my_slam.mapping.validity import View
from tests.unit.test_mapping_cov_objects import rec

BASE = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # camera looks along +x


def camera(rng: np.random.Generator) -> Intrinsics:
    w = int(rng.choice([96, 128, 160]))
    h = w * 3 // 4
    f = (w / 2) / np.tan(np.radians(rng.uniform(50.0, 90.0)) / 2)
    return Intrinsics(f, f, w / 2, h / 2, w, h)


def pose(yaw_deg: float, centre: np.ndarray) -> Pose:
    return Pose(rot_z(np.radians(yaw_deg)) @ BASE, centre)


def sighting(frame: int, pts: np.ndarray, through: bool = False) -> Sighting:
    lo, hi = np.percentile(pts, [2, 98], axis=0)
    c = pts.mean(0)
    t = lambda x: (float(x[0]), float(x[1]), float(x[2]))  # noqa: E731
    return Sighting(frame, len(pts), 0.0, t(c), t(lo), t(hi), True, through)


def thing(oid: int, pts: np.ndarray, frames: list[int], depth: float, through: bool = False,
          label: str = "chair") -> MapObject:
    o = MapObject(oid, label, {label: 1.0}, [0.9], pts.astype(np.float32), frames=frames,
                  confirmed=True, obs_depth=depth,
                  sightings=[sighting(f, pts, through) for f in frames] if len(pts) else [])
    o.sources = np.full(len(pts), ob.source_of(True, True, through), np.uint8)
    ob.refit(o, None)
    return o


def world(rng: np.random.Generator, ratio: float, baseline: float = 0.0, aside: float = 0.0
          ) -> dict[str, Any]:
    """A chair 1.5-4 m ahead of a camera turning about one centre: object 1 holds its surface
    (keyframes 0-1), object 2 the same points ``ratio`` times farther along the rays (keyframes
    2-4, whose centres lie ``baseline`` m aside; ``aside``: object 2 turned that many degrees
    away about the centre instead)."""
    K = camera(rng)
    C = np.array([0.0, 0.0, 1.0])
    z = rng.uniform(1.5, 4.0)
    size = rng.uniform(0.4, 0.7)
    yaw = rng.uniform(-180.0, 180.0)
    ahead = rot_z(np.radians(yaw)) @ np.array([1.0, 0.0, 0.0])
    side = np.cross([0.0, 0.0, 1.0], ahead)
    u, v = rng.uniform(-0.5, 0.5, (2, 1500)) * size
    chair = C + z * ahead + u[:, None] * side + v[:, None] * np.array([0.0, 0.0, 1.5])
    copy = C + ratio * (chair - C)
    if aside:
        copy = C + (copy - C) @ rot_z(np.radians(aside)).T
    views: dict[int, View] = {}
    records = []
    masks: dict[int, list[tuple[int, np.ndarray]]] = {}
    for f in range(5):
        centre = C + (np.array([0.0, baseline, 0.0]) if f >= 2 else 0.0)
        T = pose(yaw + rng.uniform(-6.0, 6.0), centre)
        views[f] = View(np.full((K.height, K.width), ratio * z + 1.0, np.float32),
                        np.ones((K.height, K.width), bool), K, T)
        records.append(rec(f, T, K_grid=K))
        pts = chair if f < 2 else copy
        uv, d = project(T.inverse().apply(pts), K.K())
        m = np.zeros((K.height, K.width), bool)
        ok = (d > 0) & np.all(np.isfinite(uv), axis=1)
        cols = np.clip(np.rint(uv[ok, 0]).astype(int), 0, K.width - 1)
        rows = np.clip(np.rint(uv[ok, 1]).astype(int), 0, K.height - 1)
        m[rows, cols] = True
        m = ndimage.binary_closing(m, iterations=2)
        if f >= 2:  # what is seen through the chair's back: its inside only
            m = ndimage.binary_erosion(m, iterations=2)
        masks[f] = [(1 if f < 2 else 2, m)]
    v = ob._Views(None, views, records)
    a = thing(1, chair, [0, 1], z)
    b = thing(2, copy, [2, 3, 4], ratio * z, through=True)
    return {"views": v, "masks": ob._Masks(None, v, masks, {}, {}), "a": a, "b": b, "z": z}


@pytest.mark.parametrize("seed", range(8))
def test_one_object_placed_twice_along_the_same_rays_is_one(seed: int) -> None:
    rng = np.random.default_rng(seed)
    w = world(rng, rng.uniform(1.3, 2.5))
    a, b, masks = w["a"], w["b"], w["masks"]
    assert ob._same_rays(a, b, masks) >= 1.0
    assert ob._same_rays(a, b, masks) == ob._same_rays(a, b, masks)  # cached
    assert set(ob._ray_pairs([a, b], {2}, masks)) == {(1, 2)}
    assert ob._ray_pairs([a, b], set(), masks) == {}  # neither touched by this update
    # merged, it is placed by the detections that show its own surface, not the wall behind
    state = ObjectState([a, b], 3)
    alias: dict[int, int] = {}
    assert ob._merge(state, {1, 2}, alias, w["views"], None, masks) == 1
    assert alias == {2: 1} and [o.id for o in state.objects] == [1]
    keep = state.objects[0]
    assert keep.obb is not None
    origin = np.array([0.0, 0.0, 1.0])
    assert np.linalg.norm(keep.obb.center - origin) == pytest.approx(w["z"], rel=0.1)


@pytest.mark.parametrize("seed", range(4))
def test_objects_the_keyframes_can_tell_apart_stay_two(seed: int) -> None:
    rng = np.random.default_rng(50 + seed)
    ratio = rng.uniform(1.3, 2.5)
    # the second object's keyframes stand far enough aside for the parallax to show its depth
    w = world(rng, ratio, baseline=1.0)
    assert ob._same_rays(w["a"], w["b"], w["masks"]) < 1.0
    # judged from a centre far from the keyframes, nothing is judged
    far = np.array([100.0, 0.0, 1.0])
    assert (ob._rays_judged(w["a"], w["b"], far, w["masks"]) == 0).all()
    # side by side: not on each other's rays
    w = world(rng, ratio, aside=40.0)
    assert ob._same_rays(w["a"], w["b"], w["masks"]) < 1.0
    assert ob._ray_pairs([w["a"], w["b"]], {1, 2}, w["masks"]) == {}


def test_the_same_rays_need_compatible_labels_separate_keyframes_and_two_detections() -> None:
    rng = np.random.default_rng(7)
    w = world(rng, 1.8)
    a, b, masks = w["a"], w["b"], w["masks"]
    lamp = thing(3, b.points.astype(float), b.frames, b.obs_depth, label="lamp")
    assert ob._same_rays(a, lamp, masks) == 0.0
    both = thing(4, b.points.astype(float), [1, 2, 3], b.obs_depth)
    assert ob._same_rays(a, both, masks) == 0.0  # a keyframe detected both
    once = thing(5, b.points.astype(float), [3], b.obs_depth)
    assert ob._same_rays(a, once, masks) == 0.0
    empty = thing(6, np.zeros((0, 3)), [2, 3], b.obs_depth)
    assert ob._same_rays(a, empty, masks) == 0.0
    # sightings of keyframes the map no longer holds: no centre to judge from
    gone = thing(0, b.points.astype(float), [8, 9], b.obs_depth)
    assert ob._same_rays(a, gone, masks) == 0.0
    assert ob._ray_pairs([gone, a], {0}, masks) == {}
    # a keyframe without a view of its own, or without the object's detection, judges nothing
    assert (ob._rays_judged(gone, a, np.zeros(3), masks) == 0).all()
    unmasked = thing(11, b.points.astype(float), [2, 3], b.obs_depth)
    assert (ob._rays_judged(unmasked, a, np.zeros(3), masks) == 0).all()
    # an object at the centre of the other's sightings has no direction to compare
    centre = thing(10, np.tile([0.0, 0.0, 1.0], (20, 1)), [2, 3], 1.0)
    assert ob._ray_pairs([a, centre], {10}, masks) == {}


def test_a_detection_seen_through_shapes_a_box_only_while_nothing_better_saw_it() -> None:
    pts = np.random.default_rng(0).uniform(0.0, 1.0, (50, 3))
    seen = [sighting(f, pts, through=True) for f in range(3)]
    own = [sighting(f, pts) for f in (3, 4)]
    o = MapObject(1, "chair", {"chair": 1.0}, [0.9], pts.astype(np.float32), frames=[0, 1, 2],
                  sightings=seen)
    assert ob.shaping_sightings(o) == seen
    o.sightings = sorted(seen + own, key=Sighting.key)
    assert ob.shaping_sightings(o) == own
    # recorded with the object, and read back
    s = seen[0]
    assert Sighting.from_list(s.to_list()) == s
    assert not Sighting.from_list(s.to_list()[:13]).through  # a map written before the flag
    assert ob.source_of(True, True, through=True) == ob.SRC_THROUGH
    assert ob.source_of(True, False, through=True) == ob.SRC_LOW_CONFIDENCE
