"""§2.3 "a later update wins" for objects that change between observations, on randomised scenes.

Each scene is a room with random furniture and one random object (size 0.15-0.5 m, random colour,
on the floor), photographed by a first set of views from one side of the room and a later set
from another side (60-180° around, from a different distance), each view's depth scaled by up to
±4 % (monocular depth disagrees between views). Between the two sets the object is removed, added
or moved, or nothing changes; the sets are mapped in one update (order of addition is the only
sign of "latest") or in two. The map must end as the later views show the room: a removed object
gone with its points, an added one exported and drawn, a moved one exported once where it now
stands with the id it had, and nothing else touched — no removal, no move — when nothing changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from oh_my_slam.core.types import Pose
from tests.synth.scene import Box, Room, look_at
from tests.unit.test_mapping_semantics import Result, known_pose_update, shoot

SEEDS = range(4)
FURNITURE = ("cabinet", "sofa")


@dataclass
class Scene:
    furniture: list[Box]
    here: Box  # the object where the first views see it
    there: Box  # the same object elsewhere
    first: list[Pose]
    later: list[Pose]

    def room(self, *objects: Box) -> Room:
        return Room(boxes=[*self.furniture, *objects])


def _scene(seed: int) -> Scene:
    rng = np.random.default_rng(seed)
    furniture = []
    for k, label in enumerate(FURNITURE):
        size = rng.uniform([0.6, 0.4, 0.5], [1.4, 0.7, 1.0])
        corner = np.array([1.9 if k == 0 else -1.9, rng.uniform(-1.4, 1.4)])
        furniture.append(Box(np.array([*corner, size[2] / 2]), size, rng.uniform(-0.3, 0.3),
                             tuple(int(c) for c in rng.integers(40, 220, 3)), label))
    side = rng.uniform(0.15, 0.5)
    size = np.array([side * rng.uniform(0.7, 1.3), side * rng.uniform(0.7, 1.3),
                     rng.uniform(0.2, 0.5)])
    colour = tuple(int(c) for c in rng.integers(30, 230, 3))
    here_xy = rng.uniform([-0.6, -0.6], [0.0, 0.6])
    there_xy = here_xy + rng.uniform(0.7, 1.1) * np.array([np.cos(a := rng.uniform(0, 2 * np.pi)),
                                                          np.sin(a)])
    there_xy = np.clip(there_xy, [-1.1, -1.4], [1.1, 1.4])
    if np.linalg.norm(there_xy - here_xy) < 0.6:
        there_xy = here_xy + np.array([0.8, 0.0])
    here = Box(np.array([*here_xy, size[2] / 2]), size, rng.uniform(-0.8, 0.8), colour, "bin")
    there = Box(np.array([*there_xy, size[2] / 2]), size, rng.uniform(-0.8, 0.8), colour, "bin")
    middle = np.array([*(here_xy + there_xy) / 2, 0.2])

    def views(azimuth: float, radius: float) -> list[Pose]:
        out = []
        for k in range(4):
            a = azimuth + (k - 1.5) * 0.12
            eye = middle + np.array([radius * np.cos(a), radius * np.sin(a),
                                     rng.uniform(1.1, 1.5)])
            eye[:2] = np.clip(eye[:2], [-2.8, -2.3], [2.8, 2.3])
            out.append(look_at(eye, middle + rng.normal(0, 0.05, 3)))
        return out

    az = rng.uniform(0, 2 * np.pi)
    first = views(az, rng.uniform(1.6, 2.0))
    later = views(az + rng.choice([-1, 1]) * rng.uniform(np.pi / 3, np.pi), rng.uniform(1.9, 2.4))
    return Scene(furniture, here, there, first, later)


def _map(tmp_path: Path, scene: Scene, before: Room, after: Room, split: bool, seed: int
         ) -> tuple[Result, Result]:
    """(the map after the first views, the map after all of them)."""
    first = shoot(before, scene.first, depth_noise=0.04, seed=seed)
    later = shoot(after, scene.later, depth_noise=0.04, seed=seed + 100)
    alone = known_pose_update(tmp_path / "first", first, tmp_path / "w0")
    if not split:
        return alone, known_pose_update(tmp_path / "one", first + later, tmp_path / "w1")
    mdir = tmp_path / "two"
    known_pose_update(mdir, first, tmp_path / "w2")
    return alone, known_pose_update(mdir, later, tmp_path / "w3")


def _objects(res: Result, label: str = "bin") -> list:  # type: ignore[type-arg]
    return [o for o in res.objs.exported() if o.label == label]


def _points_at(res: Result, box: Box) -> int:
    """Cloud points in the upper part of the box (above 60 % of its height, up to 3 cm over it;
    its footprint grown by 3 cm): the floor, drawn up to ~6 cm high by the views' ±4 % depth
    noise, is not counted."""
    xyz = res.cloud.xyz
    half = np.asarray(box.size[:2]) / 2 + 0.03
    c, s = np.cos(-box.yaw), np.sin(-box.yaw)
    local = xyz[:, :2] - box.center[:2]
    local = np.stack([c * local[:, 0] - s * local[:, 1], s * local[:, 0] + c * local[:, 1]],
                     axis=1)
    h = float(box.size[2])
    return int(((np.abs(local) <= half).all(axis=1) & (xyz[:, 2] > 0.6 * h)
                & (xyz[:, 2] < h + 0.03)).sum())


def _furniture_kept(first: Result, res: Result) -> None:
    """The furniture the first views mapped keeps its ids (the later views may add some)."""
    for label in FURNITURE:
        now = {o.id for o in _objects(res, label)}
        assert {o.id for o in _objects(first, label)} <= now


def _dense(box: Box) -> int:
    """Points the upper part of a drawn box carries at the least: a fifth of its top face at 1 cm
    (the cloud samples it at 0.5-1 cm)."""
    sx, sy, _ = box.size
    return int(0.2 * sx * sy / 0.01 ** 2)


@pytest.mark.parametrize("split", [False, True], ids=["one-update", "two-updates"])
@pytest.mark.parametrize("seed", SEEDS)
def test_nothing_changed(tmp_path: Path, seed: int, split: bool) -> None:
    scene = _scene(seed)
    room = scene.room(scene.here)
    first, res = _map(tmp_path, scene, room, room, split, seed)
    summary = res.objs.summary
    assert not summary["removed"] and not summary["withdrawn"] and not summary["moved"]
    assert [o.id for o in _objects(res)] == [o.id for o in _objects(first)] and _objects(res)
    assert _points_at(res, scene.here) > _dense(scene.here)
    _furniture_kept(first, res)


@pytest.mark.parametrize("split", [False, True], ids=["one-update", "two-updates"])
@pytest.mark.parametrize("seed", SEEDS)
def test_removed(tmp_path: Path, seed: int, split: bool) -> None:
    scene = _scene(seed)
    first, res = _map(tmp_path, scene, scene.room(scene.here), scene.room(), split, seed)
    assert _objects(first) and not _objects(res)
    assert _points_at(res, scene.here) == 0
    _furniture_kept(first, res)


@pytest.mark.parametrize("split", [False, True], ids=["one-update", "two-updates"])
@pytest.mark.parametrize("seed", SEEDS)
def test_added(tmp_path: Path, seed: int, split: bool) -> None:
    scene = _scene(seed)
    first, res = _map(tmp_path, scene, scene.room(), scene.room(scene.here), split, seed)
    assert not _objects(first)
    (obj,) = _objects(res)
    assert np.linalg.norm(obj.obb.center[:2] - scene.here.center[:2]) < 0.1
    assert _points_at(res, scene.here) > _dense(scene.here)  # drawn, not only listed
    assert (res.cloud.label == obj.id).sum() > 0
    _furniture_kept(first, res)


@pytest.mark.parametrize("split", [False, True], ids=["one-update", "two-updates"])
@pytest.mark.parametrize("seed", SEEDS)
def test_moved(tmp_path: Path, seed: int, split: bool) -> None:
    scene = _scene(seed)
    first, res = _map(tmp_path, scene, scene.room(scene.here), scene.room(scene.there), split,
                      seed)
    (was,) = _objects(first)
    (now,) = _objects(res)
    assert now.id == was.id  # the object keeps its id
    assert np.linalg.norm(now.obb.center[:2] - scene.there.center[:2]) < 0.1
    assert _points_at(res, scene.there) > _dense(scene.there)
    assert _points_at(res, scene.here) == 0
    _furniture_kept(first, res)


@pytest.mark.parametrize("beyond", ["detections", "later views"])
def test_a_place_beyond_the_fused_depth_is_not_judged(tmp_path: Path, beyond: str,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """The map draws no surface beyond a keyframe's fused depth, and monocular depth places an
    object there too loosely to judge its place (a car 40-100 m down a street): an object only
    detected beyond its keyframes' fused depth, or seen by later keyframes only beyond theirs,
    is kept however they see its place."""
    from oh_my_slam.mapping import geometry

    def cuts(ctx: object, records: list) -> dict[int, float]:  # type: ignore[type-arg]
        first = beyond == "detections"
        return {r.index: (0.3 if (r.index < 4) == first else 100.0) for r in records}

    monkeypatch.setattr(geometry, "keyframe_depth_cuts", cuts)
    scene = _scene(0)
    first, res = _map(tmp_path, scene, scene.room(scene.here), scene.room(), False, 0)
    assert _objects(first) and _objects(res)
    assert not res.objs.summary["removed"] and not res.objs.summary["withdrawn"]
