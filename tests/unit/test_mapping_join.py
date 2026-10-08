"""Joining a new map's keyframes onto its SfM model (``mapping.api._join_unplaced`` and its
helpers): which SfM poses are judged wrong, which lose their support with them, and which
re-placed poses are left out."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.mapping import api, validity
from oh_my_slam.mapping import frame as mframe
from oh_my_slam.mapping import trajectory as traj
from oh_my_slam.reconstruction import depth as rdepth
from oh_my_slam.reconstruction.depth import ScaleFit
from tests.synth.scene import look_at

K = Intrinsics(500.0, 500.0, 320.0, 240.0, 640, 480, "colmap")
GOOD = ScaleFit(1.0, 200, 0.05)
NONE = ScaleFit(float("nan"), 0, float("inf"))


class TrackModel:
    """An SfM model reduced to its tracks (sets of keyframe names) and poses: deregistering a
    keyframe drops it from every track, and a track left with one view is gone (as COLMAP)."""

    def __init__(self, tracks: list[set[str]], poses: dict[str, Pose] | None = None) -> None:
        self.tracks = [set(t) for t in tracks]
        self.reg = set().union(*self.tracks)
        self.poses = poses or {}

    @property
    def registered(self) -> list[str]:
        return sorted(self.reg)

    def point_counts(self) -> dict[str, int]:
        return {n: sum(n in t for t in self.tracks) for n in self.reg}

    def covisibility(self) -> dict[frozenset[str], int]:
        out: dict[frozenset[str], int] = {}
        for t in self.tracks:
            for a in t:
                for b in t:
                    if a < b:
                        out[frozenset((a, b))] = out.get(frozenset((a, b)), 0) + 1
        return out

    def deregister(self, names: set[str]) -> None:
        self.reg -= names
        self.tracks = [t - names for t in self.tracks]
        self.tracks = [t for t in self.tracks if len(t) >= 2]

    def pose(self, name: str) -> Pose:
        return self.poses[name]

    def intrinsics(self, name: str) -> Intrinsics:
        return K


def tracks(pairs: dict[tuple[str, str], int]) -> list[set[str]]:
    return [{a, b} for (a, b), n in pairs.items() for _ in range(n)]


# ------------------------------------------------------------------------ support lost with others

def test_weakened_splits_unsupported_from_changed() -> None:
    before = {"a": 100, "b": 30, "c": 40, "d": 10}
    after = {"a": 100, "b": 12, "c": 35, "d": 10}
    unsupported, changed = traj.weakened(before, after, {"a", "b", "c", "d"})
    # d never had 20 points but lost none: vetting it is not this function's business
    assert unsupported == {"b"} and changed == {"c"}


def test_a_deregistration_takes_the_keyframes_it_alone_supported() -> None:
    # f1's points are two-view tracks with f0, except 5 with f2; f3 shares 30 with f0
    model = TrackModel(tracks({("f0", "f1"): 25, ("f1", "f2"): 5, ("f2", "f3"): 100,
                               ("f0", "f3"): 30, ("f2", "f4"): 60}))
    unsupported, changed = api._deregister_cascade(model, {"f0"}, {"f1", "f2", "f3", "f4"})
    assert unsupported == {"f1"}
    assert changed == {"f2", "f3"}  # f2 lost f1's 5 points, f3 f0's 30; f4 lost nothing
    assert model.registered == ["f2", "f3", "f4"]


def test_a_deregistration_cascades_through_chains_of_weak_keyframes() -> None:
    model = TrackModel(tracks({("f0", "f1"): 15, ("f1", "f2"): 10, ("f2", "f3"): 15,
                               ("f3", "f4"): 200, ("f4", "f5"): 200}))
    unsupported, _ = api._deregister_cascade(model, {"f0"}, {"f1", "f2", "f3", "f4", "f5"})
    assert unsupported == {"f1", "f2"}  # f2 keeps 15 with f3 < 20 once f1 is gone
    assert model.registered == ["f3", "f4", "f5"]


def test_keyframes_that_lost_points_are_judged_again(monkeypatch: pytest.MonkeyPatch) -> None:
    model = TrackModel(tracks({("f0.jpg", "f1.jpg"): 25, ("f0.jpg", "f2.jpg"): 30,
                               ("f2.jpg", "f3.jpg"): 100, ("f3.jpg", "f4.jpg"): 100}))
    calls: list[set[str]] = []

    def judged(ctx: Any, m: Any, names: set[str], photos: bool,
               judge: set[str] | None = None) -> dict[str, float]:
        calls.append(set(names if judge is None else judge))
        return {"f0.jpg": 30.0} if "f0.jpg" in (judge or names) else {}

    monkeypatch.setattr(api, "_depth_contradicted", judged)
    off, unsupported = api._deregister_contradicted(None, model, set(model.registered), True)  # type: ignore[arg-type]
    assert off == {"f0.jpg": 30.0} and unsupported == {"f1.jpg"}
    # the first pass judges every posed keyframe, the second only f2, which lost f0's points
    assert calls == [{"f0.jpg", "f1.jpg", "f2.jpg", "f3.jpg", "f4.jpg"}, {"f2.jpg"}]


def test_photos_hanging_on_one_keyframe_are_joined_like_unplaced_ones() -> None:
    """Two hallway photos whose points are tracks through them and the first photo of the room
    only: their SfM scale is free, so they are deregistered whatever their depth check says, and
    so is a photo whose support they held."""
    room = tracks({("f2.jpg", "f3.jpg"): 100, ("f3.jpg", "f4.jpg"): 100,
                   ("f2.jpg", "f4.jpg"): 60, ("f4.jpg", "f5.jpg"): 100,
                   ("f3.jpg", "f5.jpg"): 60})
    hall = [{"f0.jpg", "f1.jpg", "f2.jpg"} for _ in range(140)]
    side = tracks({("f0.jpg", "f6.jpg"): 12, ("f6.jpg", "f3.jpg"): 12})  # f6: 24 points
    model = TrackModel(room + hall + side)
    posed = {n: Pose.identity() for n in model.registered}
    hanging, weak = api._deregister_hanging(model, posed)  # type: ignore[arg-type]
    assert hanging == {"f0.jpg", "f1.jpg"} and weak == {"f6.jpg"}
    assert model.registered == ["f2.jpg", "f3.jpg", "f4.jpg", "f5.jpg"]
    # a model where nothing hangs is left alone
    model = TrackModel(room)
    assert api._deregister_hanging(model, {n: Pose.identity() for n in model.registered}) == (  # type: ignore[arg-type]
        set(), set())
    assert len(model.registered) == 4


# ------------------------------------------------------------------ depth judgement of the poses

def depth_fixture(monkeypatch: pytest.MonkeyPatch, sparse: dict[str, ScaleFit],
                  dense: ScaleFit) -> tuple[Any, TrackModel]:
    names = sorted(sparse)
    poses = {n: look_at(np.array([0.3 * i, 0.0, 1.5]), np.array([0.3 * i, 3.0, 1.0]))
             for i, n in enumerate(names)}
    ctx = SimpleNamespace(new=[SimpleNamespace(
        kf=SimpleNamespace(name=n.removesuffix(".jpg")),
        frame=SimpleNamespace(depth=np.ones((4, 4), np.float32), grid_size=(4, 4)), lens=None)
        for n in names])
    model = TrackModel([set(names)], poses)
    monkeypatch.setattr(api, "_frame_depths", lambda c: [])
    monkeypatch.setattr(api.mframe, "metric_scale",
                        lambda m, f: mframe.ScaleResult(1.0, 0.0, {}, {}))
    monkeypatch.setattr(api, "_sparse_scale", lambda nf, m, n: sparse[n])
    monkeypatch.setattr(rdepth, "dense_scale", lambda *a, **k: dense)
    return ctx, model


OFF = ScaleFit(4.0, 5000, 0.05)  # a well measured dense scale 4x the metric one


def test_a_photo_without_sparse_scale_is_judged_by_its_dense_scale(
        monkeypatch: pytest.MonkeyPatch) -> None:
    sparse = {"a.jpg": GOOD, "b.jpg": GOOD, "c.jpg": NONE}
    ctx, model = depth_fixture(monkeypatch, sparse, dense=OFF)
    assert api._depth_contradicted(ctx, model, set(sparse), photos=True) == {"c.jpg": 4.0}


@pytest.mark.parametrize("dense", [
    ScaleFit(1.05, 5000, 0.1),         # supports the pose
    NONE,                              # too little overlap with the references: judges nothing
    ScaleFit(4.0, 5000, 0.5),          # too spread to judge by
])
def test_a_photo_the_dense_test_supports_or_cannot_judge_is_kept(
        monkeypatch: pytest.MonkeyPatch, dense: ScaleFit) -> None:
    sparse = {"a.jpg": GOOD, "b.jpg": GOOD, "c.jpg": NONE}
    ctx, model = depth_fixture(monkeypatch, sparse, dense=dense)
    assert api._depth_contradicted(ctx, model, set(sparse), photos=True) == {}


def test_a_photo_without_references_is_not_judged(monkeypatch: pytest.MonkeyPatch) -> None:
    lonely = {"c.jpg": NONE}
    ctx, model = depth_fixture(monkeypatch, lonely, dense=OFF)
    assert api._depth_contradicted(ctx, model, set(lonely), photos=True) == {}


def test_video_keyframes_without_sparse_scale_are_not_judged(
        monkeypatch: pytest.MonkeyPatch) -> None:
    sparse = {"a.jpg": GOOD, "b.jpg": GOOD, "c.jpg": NONE}
    ctx, model = depth_fixture(monkeypatch, sparse, dense=OFF)
    assert api._depth_contradicted(ctx, model, set(sparse), photos=False) == {}


def test_only_the_keyframes_to_judge_are_judged_the_others_are_references(
        monkeypatch: pytest.MonkeyPatch) -> None:
    sparse = {"a.jpg": GOOD, "b.jpg": ScaleFit(5.0, 200, 0.05), "c.jpg": NONE}
    ctx, model = depth_fixture(monkeypatch, sparse, dense=OFF)
    assert set(api._depth_contradicted(ctx, model, set(sparse), photos=True)) == {"b.jpg", "c.jpg"}
    assert set(api._depth_contradicted(ctx, model, set(sparse), photos=True,
                                       judge={"c.jpg"})) == {"c.jpg"}


# --------------------------------------------------------------------- re-placed poses, origin

@pytest.mark.parametrize(("matches", "residual", "rejected"), [
    (367, 23.3, True),    # the hallway photo anchored on a keyframe 11 km off
    (1513, 7.6, False),   # poorly placed, low confidence: kept
    (300, 0.5, False),
    (10, 40.0, False),    # too few matches to judge by
    (300, None, False),
])
def test_re_placed_poses_their_matches_contradict(matches: int, residual: float | None,
                                                  rejected: bool) -> None:
    stats = {"pose_matches": matches, "pose_residual_deg": residual}
    assert validity.pose_contradicted(stats) is rejected
    if rejected:
        assert not validity.pose_supported(stats)
