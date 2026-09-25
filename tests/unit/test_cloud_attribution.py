"""Object ids of the map cloud (``mapping.geometry``): a keyframe's vote for an object counts only
inside the object's attribution gate (its box grown by the depth noise at its viewing distance),
so a generous detection mask (a "carpet" mask over a counter top and the floor beyond it) cannot
paint the floor in the counter's colour; a confirmed object also takes the unlabelled cloud points
nearest its own lifted points (its surface beyond what the vote gave it), and is exported only
with enough cloud points for its size."""

from __future__ import annotations

import numpy as np
import pytest

from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.mapping import objects as mo
from oh_my_slam.mapping.geometry import (
    ATTRIBUTE_MARGIN_M,
    ATTRIBUTE_MARGIN_REL,
    FrameData,
    attribute_points,
    attribution_gates,
    attribution_margin,
    support_labels,
)
from oh_my_slam.mapping.objects import MapObject, ObjectState
from oh_my_slam.mapping.store import FrameRecord
from oh_my_slam.segmentation.obb import OBB

K = Intrinsics(260.0, 260.0, 160.0, 120.0, 320, 240)
SHAPE = (240, 320)
WALL = 2.0
VOXEL = 0.005  # the cloud's point spacing in these tests


def camera(yaw_deg: float) -> Pose:
    y = np.radians(yaw_deg)
    f = np.array([np.cos(y), np.sin(y), 0.0])
    right = np.cross(f, [0.0, 0.0, 1.0])
    return Pose(np.stack([right, np.cross(f, right), f], axis=1), np.zeros(3))


def wall_points(step: float = 0.01) -> np.ndarray:
    """The plane x = WALL in front of the cameras (y, z within ±0.6 m)."""
    y, z = np.meshgrid(np.arange(-0.6, 0.6, step), np.arange(-0.45, 0.45, step))
    return np.stack([np.full(y.size, WALL), y.ravel(), z.ravel()], 1)


def frame(index: int, yaw: float, labels: np.ndarray) -> FrameData:
    T = camera(yaw)
    v, u = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    # z-depth of the plane x = WALL seen from the origin
    d = np.stack([(u - K.cx) / K.fx, (v - K.cy) / K.fy, np.ones(SHAPE)], -1) @ T.R.T
    depth = (WALL / d[..., 0]).astype(np.float32)
    rec = FrameRecord(index, f"f{index:06d}", "", "", 1, 320, 240, K, T, 320, 240)
    return FrameData(rec, depth, depth > 0, np.full(SHAPE + (3,), 200, np.uint8),
                     labels.astype(np.int32), False)


def test_votes_outside_the_object_box_are_rejected() -> None:
    """Object 5 is a small patch in the middle of the wall; every keyframe's mask for it covers
    the whole left half of the image. Only the points in its grown box take its id."""
    left = np.zeros(SHAPE, np.int32)
    left[:, :160] = 5
    frames = [frame(k, yaw, left) for k, yaw in enumerate((-2.0, 0.0, 2.0))]
    pts = wall_points()
    box = OBB(np.array([WALL, 0.25, 0.0]), np.eye(3), np.array([0.02, 0.2, 0.2]))
    _, loose, _ = attribute_points(pts, frames)
    _, gated, _ = attribute_points(pts, frames, {5: (box, ATTRIBUTE_MARGIN_M)})
    assert (loose == 5).sum() > 4 * (gated == 5).sum() > 0
    inside = box.contains(pts, ATTRIBUTE_MARGIN_M)
    assert np.all(inside[gated == 5]) and (gated[~inside] == 0).all()
    # the points of the box that the masks cover keep the id
    assert (gated[inside & (loose == 5)] == 5).all()
    # an object without a box takes no point
    _, none, _ = attribute_points(pts, frames, {})
    assert not none.any()


def switch(confirmed: bool = True, offset: float = 0.04) -> MapObject:
    """A light switch on the wall, its lifted points ``offset`` in front of the fused wall (the
    depth of the four keyframes that detected it disagrees with the fused surface)."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.5, 0.5, (300, 3)) * (0.005, 0.06, 0.1) + (WALL - offset, -0.3, 0.1)
    o = MapObject(106, "power outlet", {"power outlet": 1.6}, [0.6] * 4,
                  mo.canonical_points(pts), frames=[35, 37, 40, 44], confirmed=confirmed,
                  obs_depth=WALL)
    mo.refit(o, None)
    return o


def test_an_object_that_wins_no_vote_takes_the_points_near_its_own() -> None:
    pts = wall_points(0.005)
    label = np.zeros(len(pts), np.int32)
    o = switch()
    assert support_labels(pts, label, ObjectState([o], 200)) == 1
    mine = pts[label == o.id]
    assert len(mine) >= 30
    # on the wall behind the switch, within its grown box: the box and the colour coincide
    assert o.obb is not None and o.obb.contains(mine, ATTRIBUTE_MARGIN_M).all()
    assert np.abs(mine[:, 1] + 0.3).max() < 0.06 and np.abs(mine[:, 2] - 0.1).max() < 0.08


def test_support_leaves_labelled_points_and_other_objects_alone() -> None:
    pts = wall_points(0.005)
    # an object that already has cloud points keeps them and adds the unlabelled ones of its own
    # surface
    o = switch()
    assert o.obb is not None
    label = np.zeros(len(pts), np.int32)
    voted = np.flatnonzero(o.obb.contains(pts, ATTRIBUTE_MARGIN_M))[:20]
    label[voted] = o.id
    label[:5] = 9  # another object's points elsewhere on the wall
    before = label.copy()
    assert support_labels(pts, label, ObjectState([o], 200)) == 1
    assert (label[voted] == o.id).all() and (label[:5] == 9).all()
    assert (label == o.id).sum() > len(voted) and (label[before > 0] == before[before > 0]).all()
    # unconfirmed objects take nothing; points of another object are never taken
    label = np.zeros(len(pts), np.int32)
    assert support_labels(pts, label, ObjectState([switch(confirmed=False)], 200)) == 0
    assert not label.any()
    o = switch()
    assert o.obb is not None
    label = np.where(o.obb.contains(pts, ATTRIBUTE_MARGIN_M), 7, 0).astype(np.int32)
    assert support_labels(pts, label, ObjectState([o], 200)) == 0
    assert not (label == o.id).any()
    # lifted points far from any cloud point (a pendant lamp the fusion did not keep): nothing
    label = np.zeros(len(pts), np.int32)
    assert support_labels(pts, label, ObjectState([switch(offset=0.5)], 200)) == 0


def test_a_point_two_objects_pick_goes_to_the_nearest_whatever_their_order() -> None:
    """Two switches side by side whose own points both reach the wall between them: each shared
    point goes to the object whose lifted point is nearest, in either order of the objects."""
    pts = wall_points(0.005)
    a = switch()
    b = switch()
    b.id = 107
    b.points = (b.points + np.array([0.0, 0.05, 0.0], np.float32)).astype(np.float32)
    mo.refit(b, None)
    out = []
    for order in ([a, b], [b, a]):
        label = np.zeros(len(pts), np.int32)
        assert support_labels(pts, label, ObjectState(list(order), 200)) == 2
        out.append(label)
    assert np.array_equal(out[0], out[1])
    ya = pts[out[0] == a.id][:, 1].mean()
    yb = pts[out[0] == b.id][:, 1].mean()
    assert ya < yb


def dishwasher(seen_from: float = 3.0) -> MapObject:
    """A dishwasher front on the wall (0.54 x 0.78 m), its box 6 cm deep, its lifted points 3 cm
    in front of the fused wall, seen from ``seen_from`` metres."""
    rng = np.random.default_rng(1)
    pts = rng.uniform(-0.5, 0.5, (6000, 3)) * (0.06, 0.54, 0.4) + (WALL - 0.03, 0.0, 0.0)
    o = MapObject(74, "dishwasher", {"dishwasher": 1.2}, [0.7, 0.6], mo.canonical_points(pts),
                  frames=[40, 44], confirmed=True, obs_depth=seen_from)
    mo.refit(o, None)
    return o


def test_a_thin_object_with_a_token_vote_takes_its_surface() -> None:
    """Detected in 2 of the many keyframes that see it, the dishwasher wins 1 point of its front
    in the vote, far below what its box's face holds in the cloud; it takes the wall points behind
    its own lifted points (its box is 6 cm deep, the wall 3 cm behind its points) and is exported.
    With only the token point, it would not be."""
    pts = wall_points(VOXEL)
    o = dishwasher()
    assert o.obb is not None
    need = mo.min_cloud_points(o.obb, VOXEL)
    face = np.prod(np.sort(o.obb.size)[1:]) / VOXEL ** 2
    assert need == int(np.ceil(mo.EXPORT_MIN_SUPPORT * face)) > 100
    label = np.zeros(len(pts), np.int32)
    label[np.flatnonzero(o.obb.contains(pts))[:1]] = o.id  # the vote's single point
    assert support_labels(pts, label, ObjectState([o], 200)) == 1
    mine = pts[label == o.id]
    assert len(mine) >= need
    assert o.obb.contains(mine, attribution_margin(o.obs_depth)).all()
    written: dict[str, object] = {}
    from types import SimpleNamespace
    tx = SimpleNamespace(write_json=lambda rel, obj: written.__setitem__(rel, obj))
    state = ObjectState([o], 200)
    mo.set_cloud_counts(tx, state, label, VOXEL)
    assert o.cloud_min == need and [x.id for x in state.exported()] == [74]
    # with only the token point it would not be exported
    mo.set_cloud_counts(tx, state, np.where(np.arange(len(pts)) == 0, o.id, 0), VOXEL)
    assert state.exported() == []
    stored = written[mo.OBJECTS_JSON]["objects"][0]  # type: ignore[index]
    assert stored["cloud_min_points"] == need
    assert MapObject.from_dict(stored, o.points).cloud_min == need


def test_the_attribution_gate_grows_with_the_viewing_distance() -> None:
    near, far = dishwasher(1.0), dishwasher(3.0)
    assert attribution_margin(near.obs_depth) == ATTRIBUTE_MARGIN_M
    assert attribution_margin(far.obs_depth) == ATTRIBUTE_MARGIN_REL * 3.0 > ATTRIBUTE_MARGIN_M
    gates = attribution_gates(ObjectState([far], 200))
    box, margin = gates[74]
    # a point 8 cm in front of the thin box counts at 3 m, not at 1 m
    axis = int(np.argmin(box.size))
    p = box.center.copy()
    p += box.R[:, axis] * (box.size[axis] / 2 + 0.08)
    assert box.contains(p[None], margin)[0]
    assert not box.contains(p[None], attribution_margin(near.obs_depth))[0]


# ------------------------------------------------------------------------------------------------
# outdoor distances: the export minimum at the map's sampling, support only for fused objects

STREET_VOXEL = 0.02  # the cloud voxel of a street map (median depth ~12 m)
STREET_FOCAL = 660.0  # focal length (px) of the 768-px depth grid of a 1080p video


def parked_car(distance: float, oid: int = 58) -> MapObject:
    """A parked car's side (4 m long, 1.4 m high, its box 1.5 m deep) seen from ``distance``."""
    rng = np.random.default_rng(2)
    pts = rng.uniform(-0.5, 0.5, (4000, 3)) * (1.5, 4.0, 1.4) + (distance, 0.0, 0.0)
    o = MapObject(oid, "car", {"car": 1.6}, [0.7, 0.7], mo.canonical_points(pts),
                  frames=[3, 9], confirmed=True, obs_depth=distance)
    mo.refit(o, None)
    return o


def _counts(objs: list[MapObject], label: np.ndarray, focal: float = 0.0) -> list[int]:
    """Export ids after recording ``label`` as the cloud's object ids."""
    from types import SimpleNamespace
    state = ObjectState(objs, 600)
    tx = SimpleNamespace(write_json=lambda rel, obj: None)
    mo.set_cloud_counts(tx, state, label, STREET_VOXEL, focal)
    return [x.id for x in state.exported()]


def test_a_distant_object_with_sparse_points_is_exported() -> None:
    """A car seen from 40 m: a depth pixel covers 6 cm there (9 cloud voxels of 2 cm), and its
    detections give it about one point per pixel of its side. 150 points are well drawn at that
    distance; a fixed share of its face's 2 cm voxels (~750) would drop it."""
    o = parked_car(40.0)
    assert o.obb is not None
    face = float(np.prod(np.sort(o.obb.size)[1:]))
    cell = 40.0 / STREET_FOCAL
    assert mo.sample_spacing(STREET_VOXEL, 40.0, STREET_FOCAL) == cell
    need = mo.min_cloud_points(o.obb, STREET_VOXEL, 40.0, STREET_FOCAL)
    assert need == int(np.ceil(mo.EXPORT_MIN_SUPPORT * face / cell ** 2)) < 100
    assert mo.min_cloud_points(o.obb, STREET_VOXEL) > 5 * need  # the voxel-only share
    label = np.zeros(1000, np.int32)
    label[:150] = o.id
    assert _counts([o], label, STREET_FOCAL) == [o.id]
    assert o.cloud_min == need
    assert _counts([o], label) == []  # without the sampling at its distance it would not be
    # near objects keep the voxel: at 3 m a depth pixel covers 4.5 mm, less than a voxel
    near = parked_car(3.0)
    assert near.obb is not None
    assert mo.min_cloud_points(near.obb, STREET_VOXEL, 3.0, STREET_FOCAL) == \
        mo.min_cloud_points(near.obb, STREET_VOXEL)


def test_a_tiny_or_unsupported_object_is_not_exported() -> None:
    """At 40 m a traffic sign's face (0.6 x 0.6 m) holds ~100 depth pixels; with a handful of
    points it is not drawn (the floor of EXPORT_MIN_CLOUD_POINTS), nor is a car with a tenth of
    what its side's pixels would give."""
    rng = np.random.default_rng(3)
    pts = rng.uniform(-0.5, 0.5, (500, 3)) * (0.05, 0.6, 0.6) + (40.0, 3.0, 2.0)
    sign = MapObject(61, "traffic sign", {"traffic sign": 1.6}, [0.7, 0.7],
                     mo.canonical_points(pts), frames=[3, 9], confirmed=True, obs_depth=40.0)
    mo.refit(sign, None)
    assert sign.obb is not None
    assert mo.min_cloud_points(sign.obb, STREET_VOXEL, 40.0, STREET_FOCAL) == \
        mo.EXPORT_MIN_CLOUD_POINTS
    car = parked_car(40.0)
    assert car.obb is not None
    need = mo.min_cloud_points(car.obb, STREET_VOXEL, 40.0, STREET_FOCAL)
    label = np.zeros(1000, np.int32)
    label[:6] = sign.id
    label[10:10 + need // 10] = car.id
    assert _counts([sign, car], label, STREET_FOCAL) == []
    label[20:40] = sign.id
    assert _counts([sign, car], label, STREET_FOCAL) == [sign.id]


def _sighting(frame: int, pts: np.ndarray) -> mo.Sighting:
    lo, hi = np.percentile(pts, [2, 98], axis=0)
    c = pts.mean(0)
    return mo.Sighting(frame, len(pts), 0.0, (float(c[0]), float(c[1]), float(c[2])),
                       (float(lo[0]), float(lo[1]), float(lo[2])),
                       (float(hi[0]), float(hi[1]), float(hi[2])))


def test_only_objects_their_keyframes_fused_take_support() -> None:
    """The switch's detecting keyframe fused the wall it is on (2 m, within the fused depth):
    it takes the wall points near its own. Detected only from beyond the fused depth (a car
    100 m down a street, placed by far monocular depth), an object's own surface is not in the
    cloud: it takes no support, however near its lifted points come to other surfaces."""
    from oh_my_slam.mapping.geometry import fused_objects

    pts = wall_points(0.005)
    o = switch()
    o.sightings = [_sighting(0, o.points.astype(np.float64))]
    fd = frame(0, 0.0, np.zeros(SHAPE, np.int32))
    assert fused_objects(ObjectState([o], 200), [fd], 3.0) == {o.id}
    assert fused_objects(ObjectState([o], 200), [fd], 1.5) == set()  # the wall beyond the cut
    assert fused_objects(ObjectState([o], 200), [frame(1, 0.0, fd.labels)], 3.0) == set()
    label = np.zeros(len(pts), np.int32)
    assert support_labels(pts, label, ObjectState([o], 200), set()) == 0 and not label.any()
    assert support_labels(pts, label, ObjectState([o], 200), {o.id}) == 1
    assert (label == o.id).sum() >= 30


def test_the_sampling_is_that_of_the_nearest_detection() -> None:
    """A car driving towards the camera: one keyframe detects it 11 m away (2 cm sampling, its
    mask ~8,000 lifted points), two others 43-52 m away, where it had driven off; its mean
    viewing distance is 35 m. Its cloud points are those of its nearest detection's surface, so
    it needs a share of its face's 2 cm voxels, not of the 5 cm cells of 35 m: 325 points (the
    road under where it was) do not export it."""
    from oh_my_slam.mapping.geometry import nearest_detections

    o = parked_car(35.0, oid=286)
    assert o.obb is not None
    near = mo.min_cloud_points(o.obb, STREET_VOXEL, 11.0, STREET_FOCAL)
    assert near == mo.min_cloud_points(o.obb, STREET_VOXEL)  # 11 m / 660 px < 2 cm
    assert mo.min_cloud_points(o.obb, STREET_VOXEL, 35.0, STREET_FOCAL) < 325 < near
    label = np.zeros(1000, np.int32)
    label[:325] = o.id
    assert _counts([o], label, STREET_FOCAL) == [o.id]  # at its mean distance it would be
    from types import SimpleNamespace
    state = ObjectState([o], 600)
    tx = SimpleNamespace(write_json=lambda rel, obj: None)
    mo.set_cloud_counts(tx, state, label, STREET_VOXEL, STREET_FOCAL, {o.id: 11.0})
    assert o.cloud_min == near and state.exported() == []
    # the nearest detection: the depth of its sighting's centre in the detecting keyframe
    frames = [frame(k, 0.0, np.zeros(SHAPE, np.int32)) for k in (0, 1)]
    frames[1].rec.T_map_cam = Pose(frames[1].rec.T_map_cam.R, np.array([-30.0, 0.0, 0.0]))
    pts = o.points.astype(np.float64) - np.array([24.0, 0.0, 0.0])  # 11 m ahead of both
    o.sightings = [_sighting(0, pts), _sighting(1, pts)]
    got = nearest_detections(ObjectState([o], 600), frames)
    assert set(got) == {o.id} and got[o.id] == pytest.approx(11.0, abs=0.05)
    assert nearest_detections(ObjectState([o], 600), frames[1:])[o.id] == \
        pytest.approx(41.0, abs=0.05)
