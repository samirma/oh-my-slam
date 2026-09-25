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
    # at another height (a shelf under the counter), or seen together, the pieces stay apart
    low = obj(152, "bed", {56: side - (0, 0, 0.3)}, 1.0, {"bed": 0.8, "desk": 0.7})
    assert merge(obj(2, "desk", {0: front}, 1.0), low) == {}
    seen = obj(152, "bed", {0: side[:3000], 60: side[3000:]}, 1.0, {"bed": 1.6, "desk": 0.7})
    assert merge(obj(2, "desk", {0: front[:3000], 1: front[3000:]}, 1.0), seen) == {}
    # never detected as a desk: both are still labelled as horizontal surfaces
    only_bed = obj(152, "bed", {56: side}, 1.0)
    assert merge(obj(2, "desk", {0: front}, 1.0), only_bed) == {152: 2}
    # a piece that is not labelled as a surface (a cabinet top at that height) stays apart
    top = obj(152, "cabinet", {56: side[:600]}, 1.0)
    assert merge(obj(2, "desk", {0: front}, 1.0), top) == {}


def test_a_counter_corner_labelled_rug_joins_the_counter() -> None:
    """The kitchen island: a desk (detected as desk, bed, kitchen island) and, from the keyframes
    that close the loop, its marble corner labelled rug. They continue one surface at one height
    (the corner overlaps the counter's edge by a few centimetres) and no keyframe saw both. A rug
    on the floor under the counter is another surface."""
    rng = np.random.default_rng(3)
    top = rng.uniform(-0.5, 0.5, (8000, 3)) * (1.5, 0.7, 0.02) + (0.46, -0.42, -0.3)
    corner = rng.uniform(-0.5, 0.5, (1500, 3)) * (0.45, 0.25, 0.02) + (-0.46, -0.42, -0.31)
    desk = obj(2, "desk", {0: top[:4000], 51: top[4000:]}, 1.0,
               {"desk": 7.5, "bed": 3.6, "kitchen island": 1.1})
    rug = obj(79, "rug", {27: corner[:700], 72: corner[700:]}, 1.2, {"rug": 1.1, "carpet": 0.8})
    alias: dict[int, int] = {}
    state = ObjectState([desk, rug], 400)
    assert mo._merge(state, {2, 79}, alias) == 1 and alias == {79: 2}
    (kept,) = state.objects
    assert kept.label == "desk" and "rug" in kept.label_votes
    floor = obj(81, "carpet", {27: corner - (0, 0, 1.0), 74: corner - (0, 0, 1.0)}, 1.5)
    alias = {}
    assert mo._merge(ObjectState([obj(2, "desk", {0: top}, 1.0), floor], 400), {2, 81},
                     alias) == 0


def test_pieces_the_fused_surface_joins_are_one_surface() -> None:
    """The kitchen island's near-field corner, placed 9 cm lower and 10 cm beyond the counter's
    points by the keyframes that close the loop (their depth of a surface 0.5 m away disagrees),
    is joined to the counter by the map's fused surface: one horizontal patch reaches both. A
    step between them (another surface at another height), or the floor, joins nothing."""
    rng = np.random.default_rng(4)
    top = rng.uniform(-0.5, 0.5, (8000, 3)) * (1.0, 0.8, 0.02) + (0.5, 0.0, -0.31)
    corner = rng.uniform(-0.5, 0.5, (1500, 3)) * (0.2, 0.3, 0.02) + (-0.2, 0.0, -0.40)
    desk = obj(2, "desk", {0: top[:4000], 51: top[4000:]}, 0.5, {"desk": 7.5, "bed": 3.6})
    rug = obj(79, "rug", {72: corner[:700], 75: corner[700:]}, 0.7)
    assert mo._one_surface(desk, rug) < 1.0  # their own points neither meet nor share a height

    def fused(z_of_x: object) -> np.ndarray:
        """A 5 mm cloud of the counter top from x = -0.35 to 1.05 at height ``z_of_x(x)``."""
        x, y = np.meshgrid(np.arange(-0.35, 1.05, 0.005), np.arange(-0.45, 0.45, 0.005))
        x, y = x.ravel(), y.ravel()
        return np.stack([x, y, z_of_x(x)], 1)  # type: ignore[operator]

    # the fused surface slopes from the counter's height down to the corner's, without a step
    sloped = fused(lambda x: np.interp(x, [-0.35, -0.1, 0.1, 1.05], [-0.4, -0.39, -0.32, -0.31]))
    surfaces = mo._Surfaces(sloped, floor_z=-1.3)
    assert mo._one_surface(desk, rug, surfaces) >= 1.0
    alias: dict[int, int] = {}
    mo._merge(ObjectState([desk, rug], 400), {2, 79}, alias, surfaces=surfaces)
    assert alias == {79: 2}
    # a step: the corner's surface ends in a vertical face 9 cm below the counter's
    step = fused(lambda x: np.where(x < 0.0, -0.40, -0.31))
    face = np.stack(np.meshgrid(np.array([0.0]), np.arange(-0.45, 0.45, 0.005),
                                np.arange(-0.40, -0.31, 0.005)), -1).reshape(-1, 3)
    walled = mo._Surfaces(np.concatenate([step[step[:, 0] < -0.02], step[step[:, 0] > 0.02],
                                          face]), floor_z=-1.3)
    assert mo._one_surface(desk, rug, walled) < 1.0
    # pieces at floor height: the floor would join anything
    low = mo._Surfaces(sloped, floor_z=-0.42)
    assert mo._one_surface(desk, rug, low) < 1.0
    # a piece labelled as something else is never joined
    box = obj(79, "cabinet", {72: corner[:700], 75: corner[700:]}, 0.7)
    assert mo._one_surface(desk, box, surfaces) == 0.0


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
