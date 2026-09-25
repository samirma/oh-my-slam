"""Object ids of the map cloud (``mapping.geometry``): a keyframe's vote for an object counts only
inside the object's grown box, so a generous detection mask (a "carpet" mask over a counter top
and the floor beyond it) cannot paint the floor in the counter's colour; a confirmed object that
wins no point in the vote takes the cloud points nearest its own lifted points."""

from __future__ import annotations

import numpy as np

from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.mapping import objects as mo
from oh_my_slam.mapping.geometry import (
    ATTRIBUTE_MARGIN_M,
    FrameData,
    attribute_points,
    support_labels,
)
from oh_my_slam.mapping.objects import MapObject, ObjectState
from oh_my_slam.mapping.store import FrameRecord
from oh_my_slam.segmentation.obb import OBB

K = Intrinsics(260.0, 260.0, 160.0, 120.0, 320, 240)
SHAPE = (240, 320)
WALL = 2.0


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
    _, gated, _ = attribute_points(pts, frames, {5: box})
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


def test_support_fallback_leaves_labelled_points_and_other_objects_alone() -> None:
    pts = wall_points(0.005)
    # an object that already has cloud points keeps exactly them
    o = switch()
    label = np.zeros(len(pts), np.int32)
    label[:10] = o.id
    before = label.copy()
    assert support_labels(pts, label, ObjectState([o], 200)) == 0
    assert np.array_equal(label, before)
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
