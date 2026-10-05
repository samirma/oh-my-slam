"""Mapping semantics of §2.3, "later wins" and persistent identity, for an object that moved: a cup
stands at one place in the first views and at another in the latest ones.

The map shows it once, where the latest views see it — its box and its points there, its first
place drawn as the latest views see it (the floor) — and it keeps its id, whether the views are
mapped in one update (the order of addition is the only sign of "latest") or in two. A cup of
another colour that appears elsewhere is another object, and a second cup beside one that stays is
a second object.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tests.synth.scene import Box, Room, look_at
from tests.unit.test_mapping_latest_wins import CABINET, EYE, _same_object
from tests.unit.test_mapping_semantics import Result, exported, known_pose_update, shoot

HERE = np.array([-0.4, 0.1, 0.08])
THERE = np.array([0.2, -0.45, 0.08])
CUP = Box(HERE, np.array([0.14, 0.14, 0.16]), 0.3, (240, 240, 240), "cup")
MOVED = replace(CUP, center=THERE)
RED = replace(MOVED, color=(200, 30, 30))
TARGET = np.array([0.1, 0.1, 0.2])
FIRST = [look_at(EYE + [0.0, dy, 0.0], TARGET) for dy in (0.0, 0.06, -0.06)]
LATER = [look_at(EYE + [0.0, dy, 0.0], TARGET) for dy in (0.03, -0.03, 0.09)]


def _points_at(res: Result, where: np.ndarray) -> int:
    """Cloud points above the floor within the cup's footprint at ``where``."""
    xyz = res.cloud.xyz
    near = (np.abs(xyz[:, :2] - where[:2]).max(axis=1) < 0.15) & (xyz[:, 2] > 0.03)
    return int(near.sum())


def _cups(res: Result) -> list:  # type: ignore[type-arg]
    return [o for o in res.objs.exported() if o.label == "cup"]


def _at_its_latest_place(res: Result, cup_id: int) -> None:
    (cup,) = _cups(res)
    assert cup.id == cup_id
    assert np.linalg.norm(cup.obb.center[:2] - THERE[:2]) < 0.05
    assert _points_at(res, THERE) > 100 and (res.cloud.label == cup_id).sum() > 100
    assert _points_at(res, HERE) == 0  # its first place shows the floor


def test_a_cup_moved_within_one_update_is_mapped_once_where_the_last_views_see_it(
        tmp_path: Path) -> None:
    first = known_pose_update(tmp_path / "first", shoot(Room(boxes=[CABINET, CUP]), FIRST),
                              tmp_path / "w0")
    (was,) = _cups(first)
    (cabinet,) = [o for o in first.objs.exported() if o.label == "cabinet"]
    one = known_pose_update(tmp_path / "one", shoot(Room(boxes=[CABINET, CUP]), FIRST)
                            + shoot(Room(boxes=[CABINET, MOVED]), LATER), tmp_path / "w1")
    assert one.objs.summary["moved"] == [was.id]
    assert not one.objs.summary["removed"] and not one.objs.summary["withdrawn"]
    _at_its_latest_place(one, was.id)  # the id its first detection gave it
    _same_object(cabinet, exported(one)[cabinet.id])


def test_a_published_cup_stays_published_at_its_new_place(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Published objects stay published, wherever they move: the cup update 1 published moves in
    update 2, whose cloud draws too few points at its new place for a new object to be exported
    there; the cup keeps its id, published and exported, at its new place."""
    from oh_my_slam.mapping import objects as mo

    mdir = tmp_path / "m"
    r1 = known_pose_update(mdir, shoot(Room(boxes=[CABINET, CUP]), FIRST), tmp_path / "w1")
    (was,) = _cups(r1)
    monkeypatch.setattr(mo, "_enough_cloud", lambda cloud, least: False)
    r2 = known_pose_update(mdir, shoot(Room(boxes=[CABINET, MOVED]), LATER), tmp_path / "w2")
    assert r2.objs.summary["moved"] == [was.id] and not r2.objs.summary["removed"]
    (cup,) = [o for o in r2.objs.objects if o.id == was.id]
    assert cup.published and cup.confirmed
    (now,) = _cups(r2)
    assert now.id == was.id and np.linalg.norm(now.obb.center[:2] - THERE[:2]) < 0.05


def test_a_cup_moved_between_updates_keeps_its_id_at_its_new_place(tmp_path: Path) -> None:
    mdir = tmp_path / "m"
    r1 = known_pose_update(mdir, shoot(Room(boxes=[CABINET, CUP]), FIRST), tmp_path / "w1")
    (was,) = _cups(r1)
    (cabinet,) = [o for o in r1.objs.exported() if o.label == "cabinet"]
    r2 = known_pose_update(mdir, shoot(Room(boxes=[CABINET, MOVED]), LATER), tmp_path / "w2")
    assert r2.objs.summary["moved"] == [was.id] and not r2.objs.summary["removed"]
    _at_its_latest_place(r2, was.id)
    _same_object(cabinet, exported(r2)[cabinet.id])
    # its first place is drawn from the latest views: the floor's colour, not the cup's
    xyz = r2.cloud.xyz
    place = (np.abs(xyz[:, :2] - HERE[:2]).max(axis=1) < 0.05) & (np.abs(xyz[:, 2]) < 0.02)
    assert place.sum() > 0
    floor = np.array(Room().floor_color, float)
    assert np.abs(r2.cloud.rgb[place].astype(float).mean(axis=0) - floor).max() < 40


def test_a_cup_of_another_colour_is_another_object(tmp_path: Path) -> None:
    mdir = tmp_path / "m"
    r1 = known_pose_update(mdir, shoot(Room(boxes=[CABINET, CUP]), FIRST), tmp_path / "w1")
    (was,) = _cups(r1)
    r2 = known_pose_update(mdir, shoot(Room(boxes=[CABINET, RED]), LATER), tmp_path / "w2")
    assert not r2.objs.summary["moved"] and r2.objs.summary["removed"] == [was.id]
    (red,) = _cups(r2)
    assert red.id != was.id and _points_at(r2, HERE) == 0


def test_a_second_cup_beside_one_that_stays_is_a_second_object(tmp_path: Path) -> None:
    both = Room(boxes=[CABINET, CUP, MOVED])
    res = known_pose_update(tmp_path / "m", shoot(Room(boxes=[CABINET, CUP]), FIRST)
                            + shoot(both, LATER), tmp_path / "w")
    assert not res.objs.summary["moved"]
    assert len(_cups(res)) == 2
    assert _points_at(res, HERE) > 100 and _points_at(res, THERE) > 100
