"""Object identity (spec §2.3, one id per physical object): copies of one object placed by
keyframes whose monocular depth disagrees (a loop closed by keyframes whose depth scale drifted)
are merged, pieces of one horizontal surface are one object — within a keyframe and across
keyframes — and objects without points in the map cloud are not exported."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.mapping import objects as mo
from oh_my_slam.mapping.objects import MapObject, ObjectState, Sighting
from oh_my_slam.mapping.validity import View
from oh_my_slam.segmentation.api import Detection, LiftedInstance, join_instances
from oh_my_slam.segmentation.lift import Lifted

K = Intrinsics(260.0, 260.0, 160.0, 120.0, 320, 240)
SHAPE = (240, 320)


def camera(yaw_deg: float = 0.0) -> Pose:
    """Camera-to-map pose at the origin (OpenCV camera axes, z-up map) looking along ``yaw``."""
    y = np.radians(yaw_deg)
    f = np.array([np.cos(y), np.sin(y), 0.0])
    right = np.cross(f, [0.0, 0.0, 1.0])
    return Pose(np.stack([right, np.cross(f, right), f], axis=1), np.zeros(3))


def wall(distance: float, yaw_deg: float = 0.0) -> View:
    """A keyframe that sees a fronto-parallel wall ``distance`` metres ahead."""
    depth = np.full(SHAPE, distance, np.float32)
    return View(depth, np.ones(SHAPE, bool), K, camera(yaw_deg))


def blob(rng: np.random.Generator, centre: tuple[float, float, float],
         size: tuple[float, float, float] = (0.08, 0.3, 0.2), n: int = 400) -> np.ndarray:
    return rng.uniform(-0.5, 0.5, (n, 3)) * np.asarray(size) + np.asarray(centre)


def sighting(frame: int, pts: np.ndarray) -> Sighting:
    lo, hi = np.percentile(pts, [2, 98], axis=0)
    c = pts.mean(0)

    def t(x: np.ndarray) -> tuple[float, float, float]:
        return (float(x[0]), float(x[1]), float(x[2]))
    return Sighting(frame, len(pts), 0.0, t(c), t(lo), t(hi))


def obj(oid: int, label: str, parts: dict[int, np.ndarray], depth: float,
        votes: dict[str, float] | None = None) -> MapObject:
    """An object detected in the keyframes ``parts`` (keyframe -> its lifted points)."""
    pts = np.concatenate(list(parts.values()))
    o = MapObject(oid, label, votes or {label: 0.8 * len(parts)}, [0.8] * len(parts),
                  mo.canonical_points(pts), frames=sorted(parts), obs_depth=depth)
    o.sightings = sorted((sighting(f, p) for f, p in parts.items()), key=Sighting.key)
    mo.refit(o, None)
    return o


def keyframes(views: dict[int, View]) -> mo._Views:
    records = [SimpleNamespace(index=i, T_map_cam=v.T_map_cam) for i, v in views.items()]
    return mo._Views(None, dict(views), records)


# --- depth-explained duplicates (loop closure) --------------------------------------------------


def test_copies_placed_by_keyframes_whose_depth_disagrees_are_merged() -> None:
    """The faucet of a sink seen by the keyframes that start the loop (0, 2) at 2.5 m and by
    those that close it (1, 3), whose depth scale drifted by 12 %, at 2.8 m: two copies 0.3 m
    apart along the same viewing rays, too far apart for the overlap tests."""
    rng = np.random.default_rng(0)
    near = {0: blob(rng, (2.5, 0.0, 0.0)), 2: blob(rng, (2.5, 0.0, 0.0))}
    far = {f: p * 1.12 for f, p in ((1, blob(rng, (2.5, 0.0, 0.0))),
                                    (3, blob(rng, (2.5, 0.0, 0.0))))}
    views = keyframes({0: wall(2.5), 1: wall(2.8), 2: wall(2.5), 3: wall(2.8)})
    assert mo.depth_ratio(views.get(0), views.get(1)) == pytest.approx(1.12, abs=1e-3)

    def merged(a: MapObject, b: MapObject, v: mo._Views | None) -> dict[int, int]:
        alias: dict[int, int] = {}
        mo._merge(ObjectState([a, b], 400), {a.id, b.id}, alias, v)
        return alias

    assert merged(obj(60, "faucet", near, 2.5), obj(257, "faucet", far, 2.8), views) == {257: 60}
    # without the keyframes' depths the copies are two objects (no overlap test holds) ...
    assert merged(obj(60, "faucet", near, 2.5), obj(257, "faucet", far, 2.8), None) == {}
    # ... and they stay two when the offset is not explained: keyframes that agree in depth
    agree = keyframes({0: wall(2.5), 1: wall(2.5), 2: wall(2.5), 3: wall(2.5)})
    assert merged(obj(60, "faucet", near, 2.5), obj(257, "faucet", far, 2.8), agree) == {}
    # a keyframe that detected both saw two things; other labels are other objects
    both = {**far, 0: far[1]}
    assert merged(obj(60, "faucet", near, 2.5), obj(257, "faucet", both, 2.8), views) == {}
    assert merged(obj(60, "faucet", near, 2.5), obj(257, "cup", far, 2.8), views) == {}


# --- split surfaces --------------------------------------------------------------------------------


def piece(label: str, mask: np.ndarray, depth: np.ndarray, score: float = 0.7) -> LiftedInstance:
    v, u = np.nonzero(mask)
    z = depth[v, u].astype(np.float64)
    pts = np.stack([(u - K.cx) / K.fx * z, (v - K.cy) / K.fy * z, z], 1)
    box = (float(u.min()), float(v.min()), float(u.max()), float(v.max()))
    return LiftedInstance(Detection(label, score, "yoloe", mask, box), mask,
                          Lifted(pts, (v * SHAPE[1] + u).astype(np.int64), int(mask.sum())))


def halves(gap: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two masks side by side over the lower image half, ``gap`` px apart."""
    left, right = np.zeros(SHAPE, bool), np.zeros(SHAPE, bool)
    left[140:, :160] = True
    right[140:, 160 + gap:] = True
    return left, right


def test_touching_pieces_of_one_surface_are_one_instance() -> None:
    """A counter top split by the detector into two desks around a book lying on it."""
    view = wall(1.2)
    left, right = halves()
    desks = [piece("desk", left, view.depth, 0.78), piece("desk", right, view.depth, 0.65)]
    assert mo.surface_pieces(view, desks) == [[0, 1]]
    joined = join_instances(desks)
    assert joined.detection.label == "desk" and joined.detection.score == 0.78
    assert np.array_equal(joined.mask, left | right)
    assert len(joined.lifted.points) == int((left | right).sum())
    # a table and a kitchen island may be pieces of one surface, a desk and a bed may not; side
    # by side items of other classes (cabinets, books) are separate objects
    for a, b, one in (("dining table", "kitchen island", True), ("rug", "carpet", True),
                      ("desk", "bed", False), ("cabinet", "cabinet", False)):
        pair = [piece(a, left, view.depth), piece(b, right, view.depth)]
        assert (mo.surface_pieces(view, pair) == [[0, 1]]) is one, (a, b)
    # pieces that do not touch, or meet at a depth step (one surface in front of another)
    apart = halves(gap=8)
    pair = [piece("desk", apart[0], view.depth), piece("desk", apart[1], view.depth)]
    assert mo.surface_pieces(view, pair) == [[0], [1]]
    step = View(np.where(right, 1.5, 1.2).astype(np.float32), np.ones(SHAPE, bool), K, camera())
    pair = [piece("desk", left, step.depth), piece("desk", right, step.depth)]
    assert mo.surface_pieces(step, pair) == [[0], [1]]


def test_pieces_of_one_surface_seen_from_different_keyframes_are_merged() -> None:
    """A counter top around the camera: detected as a desk from one side and as a bed (once as
    a desk) from another; the two pieces overlap on a 10 cm strip at the same height."""
    rng = np.random.default_rng(1)
    front = rng.uniform(-0.5, 0.5, (6000, 3)) * (0.8, 0.6, 0.02) + (0.5, 0.35, -0.3)
    side = rng.uniform(-0.5, 0.5, (6000, 3)) * (0.8, 0.6, 0.02) + (0.5, -0.15, -0.27)

    def merge(a: MapObject, b: MapObject) -> dict[int, int]:
        alias: dict[int, int] = {}
        mo._merge(ObjectState([a, b], 400), {a.id, b.id}, alias)
        return alias

    desk = obj(2, "desk", {0: front[:3000], 1: front[3000:]}, 1.0)
    bed = obj(152, "bed", {56: side[:3000], 60: side[3000:]}, 1.0, {"bed": 1.6, "desk": 0.7})
    assert merge(desk, bed) == {152: 2}
    # at another height (a shelf under the counter), seen together, or without a surface label
    # in common, the pieces stay apart
    low = obj(152, "bed", {56: side - (0, 0, 0.3)}, 1.0, {"bed": 0.8, "desk": 0.7})
    assert merge(obj(2, "desk", {0: front}, 1.0), low) == {}
    seen = obj(152, "bed", {0: side[:3000], 60: side[3000:]}, 1.0, {"bed": 1.6, "desk": 0.7})
    assert merge(obj(2, "desk", {0: front[:3000], 1: front[3000:]}, 1.0), seen) == {}
    only_bed = obj(152, "bed", {56: side}, 1.0)
    assert merge(obj(2, "desk", {0: front}, 1.0), only_bed) == {}


# --- export ------------------------------------------------------------------------------------------


def test_objects_without_map_cloud_points_are_not_exported() -> None:
    rng = np.random.default_rng(2)
    objs = []
    for oid, cloud in ((1, 5000), (2, 0), (3, None), (4, mo.EXPORT_MIN_CLOUD_POINTS)):
        o = obj(oid, "cup", {0: blob(rng, (2.0, 0.1 * oid, 0.0))}, 2.0)
        o.confirmed, o.cloud_points = True, cloud
        objs.append(o)
    # a pendant lamp whose surface did not survive the fusion has no point in the cloud: its box
    # would have no points in segments.ply; a map written before the counts keeps its objects
    assert [o.id for o in ObjectState(objs, 10).exported()] == [1, 3, 4]
