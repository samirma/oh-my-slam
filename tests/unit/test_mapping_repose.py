"""A new map posed by the multi-view fallback is refined again on its aligned depth
(``mapping.api._repose_on_aligned_depth``): centres placed with each keyframe's depth at its own
scale move to where the aligned depth puts them, the first keyframe holds the gauge, keyframes of
low confidence take part by their rays only, and the depth is aligned again to the new poses."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from oh_my_slam.core.types import Pose
from oh_my_slam.mapping import api, panorama, store
from tests.synth.turning import K, head_pose, perturb, rot_err_deg, turning_rig
from tests.unit.test_mapping_cov_api import FakeModel, ctx_of, nf


def _names(n: int) -> list[str]:
    return [f"f{k:06d}.jpg" for k in range(n)]


class _Reposed(FakeModel):
    def __init__(self, poses: dict[str, Pose], method: str = "multiview") -> None:
        super().__init__(poses, method=method)
        self.reposed: dict[str, Pose] = {}

    def repose(self, poses: dict[str, Pose], db: Path, image_dir: Path, work: Path) -> None:
        self.reposed = dict(poses)
        self.poses.update(poses)


def _placed(truth: dict[str, Pose], init: dict[str, Pose], rig: Any, low: set[str]
            ) -> SimpleNamespace:
    new = []
    for k, name in enumerate(sorted(truth)):
        f = nf(k)
        f.full_size = (K.width, K.height)
        f.record = store.FrameRecord(k, name[:-4], "", "", 1, K.width, K.height, K, init[name],
                                     K.width, K.height, pose_source="multiview",
                                     low_confidence=name in low)
        f.depth = rig.depth[name]
        new.append(f)
    tx = SimpleNamespace(current=lambda rel: Path("/db"), stage=lambda rel: Path("/frames"))
    return ctx_of(new, tx=tx)


@pytest.mark.parametrize("seed", range(3))
def test_centres_move_to_where_the_aligned_depth_puts_them(
        seed: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """A sweep and a second one 12 cm lower, its centres placed 6 cm off (depth at another
    scale): refined on the aligned depth they come within a centimetre, the gauge keyframe
    stays, and a low-confidence keyframe is refined by its rays alone."""
    rng = np.random.default_rng(seed)
    names = _names(18)
    truth = {n: head_pose(20.0 * k, -12.0 + (0, 14, -14)[k % 3]) for k, n in enumerate(names)}
    for k in range(6):
        truth[f"f{18 + k:06d}.jpg"] = head_pose(60.0 * k + 10, -20.0,
                                                step=np.array([0.0, 0.0, -0.12]))
    rig = turning_rig(truth, seed=seed)
    init = {n: T if n == names[0] else perturb(T, 0.5, 0.06, rng) for n, T in truth.items()}
    low = {names[5]}
    ctx = _placed(truth, init, rig, low)
    ctx.notes["pose_refinement"] = {"screen_gate_deg": 12.5}  # the first screening's gate
    monkeypatch.setattr(panorama, "verified_matches", lambda db, ns: rig.pairs)
    seen: dict[str, Any] = {}
    refine = panorama.refine_poses

    def spy(pairs: Any, views: Any, free: Any, **kw: Any) -> Any:
        seen.update(views=views, free=set(free), kw=kw)
        return refine(pairs, views, free, **kw)

    monkeypatch.setattr(panorama, "refine_poses", spy)
    model = _Reposed(init)
    assert api._repose_on_aligned_depth(ctx, model)  # type: ignore[arg-type]
    assert names[0] not in model.reposed and seen["free"] == set(truth) - {names[0]}
    assert seen["views"][names[5]].depth is None and seen["views"][names[6]].depth is not None
    assert not seen["kw"].get("refine_focal")  # the lens is held
    err = {n: float(np.linalg.norm(model.reposed[n].t - truth[n].t)) for n in model.reposed}
    assert float(np.median(list(err.values()))) < 0.01
    second = sorted(truth)[18:]  # the lower sweep: its drop is the depth's
    assert max(err[n] for n in second) < 0.02 < 0.06  # from 6 cm off
    assert float(np.median([rot_err_deg(model.reposed[n], truth[n]) for n in model.reposed])) < 0.25
    notes = ctx.notes["pose_refinement_aligned"]
    assert notes["pairs"] == len(rig.pairs) and notes["screened_pairs"] == 0
    assert notes["screen_gate_deg"] == 12.5
    assert set(ctx.pose_support) == set(model.reposed)


def test_without_pairs_nothing_is_reposed(monkeypatch: pytest.MonkeyPatch) -> None:
    names = _names(3)
    truth = {n: head_pose(20.0 * k) for k, n in enumerate(names)}
    rig = turning_rig(truth)
    ctx = _placed(truth, truth, rig, set())
    monkeypatch.setattr(panorama, "verified_matches", lambda db, ns: [])
    model = _Reposed(truth)
    assert not api._repose_on_aligned_depth(ctx, model)  # type: ignore[arg-type]
    assert model.reposed == {} and "pose_refinement_aligned" not in ctx.notes


@pytest.mark.parametrize(("method", "notes", "reposed", "aligned"), [
    ("multiview", {}, True, 2),
    ("multiview", {}, False, 1),
    ("multiview", {"one_centre": {"taken": True}}, True, 1),  # one centre: not moved
    ("multiview", {"one_centre": {"taken": False}}, True, 2),
    ("sfm-global", {}, True, 1),
])
def test_placing_a_new_multiview_map_aligns_its_depth_again_once_reposed(
        monkeypatch: pytest.MonkeyPatch, method: str, notes: dict[str, Any], reposed: bool,
        aligned: int) -> None:
    calls: list[str] = []
    model = FakeModel({"f000000.jpg": Pose.identity()}, method=method)
    monkeypatch.setattr(api, "_run_sfm", lambda c, v, cl, p: model)
    for name in ("_rerun_focal", "_define_map_frame", "_level_with_floor"):
        monkeypatch.setattr(api, name, lambda *a: None)
    monkeypatch.setattr(api, "_adjust_depth_scales", lambda c, p: None)

    def align(ctx: Any, m: Any) -> None:
        calls.append("align")
        ctx.new[0].record = SimpleNamespace()

    def repose(ctx: Any, m: Any) -> bool:
        calls.append("repose")
        return reposed

    monkeypatch.setattr(api, "_align_depths", align)
    monkeypatch.setattr(api, "_repose_on_aligned_depth", repose)
    ctx = ctx_of([nf(0)])
    ctx.notes.update(notes)
    api._place(ctx, False, None, lambda m: None)  # type: ignore[arg-type]
    assert calls.count("align") == aligned
    assert ("repose" in calls) == (method == "multiview"
                                   and not notes.get("one_centre", {}).get("taken"))
