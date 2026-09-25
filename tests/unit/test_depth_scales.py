"""Global adjustment of per-keyframe depth scales (``reconstruction.depth.adjust_log_scales``,
``mapping.api._adjust_depth_scales``, ``mapping.objects.rescale_objects``): a head turning a full
circle in a box room, with the scale error that aligning each keyframe to its predecessors
accumulates around the loop."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.mapping import api
from oh_my_slam.mapping.store import FrameRecord, frame_file
from oh_my_slam.reconstruction.depth import (
    DepthView,
    adjust_log_scales,
    pair_log_ratio,
    pair_weight,
)
from tests.synth.turning import depth_grid, head_pose

# a half-resolution grid keeps the ray casting cheap
KG = Intrinsics(217.5, 217.5, 160.0, 120.0, 320, 240)
N = 30  # keyframes, 12° apart: the last one overlaps the first


def loop_poses() -> list[Pose]:
    return [head_pose(360.0 * k / N, pitch_deg=-12.0 + 8.0 * (k % 3 == 1)) for k in range(N)]


def drift(seed: int = 0) -> np.ndarray:
    """Log scale error per keyframe: a random walk (the first keyframe exact) reaching ~15 %."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.005, 0.003, N)
    steps[0] = 0.0
    return np.cumsum(steps)


def overlapping(poses: list[Pose], max_deg: float = 40.0) -> list[tuple[int, int]]:
    F = np.array([T.R[:, 2] for T in poses])
    cos = F @ F.T
    return [(i, j) for i in range(N) for j in range(i + 1, N)
            if cos[i, j] > np.cos(np.radians(max_deg))]


def test_loop_scale_drift_is_removed_by_the_global_adjustment() -> None:
    poses = loop_poses()
    true = [depth_grid(T, KG).astype(np.float64) for T in poses]
    err = drift()
    views = [DepthView(d * np.exp(e), KG.K(), T.matrix()) for d, e, T in
             zip(true, err, poses, strict=True)]
    pairs = overlapping(poses)
    assert (0, N - 1) in pairs  # the loop closure
    closing = pair_log_ratio(views[N - 1], views[0])
    assert closing is not None and closing.log_ratio > 0.1  # 10 %+ disagreement where it closes
    x = np.zeros(N)
    for _ in range(2):
        meas = [(i, j, p.log_ratio, pair_weight(p)) for i, j in pairs
                if (p := pair_log_ratio(views[i], views[j], np.exp(x[i]), np.exp(x[j])))
                is not None]
        adj = adjust_log_scales(N, meas, {0})
        x += adj.log_scale
    # every keyframe's error is undone (the first one holds the gauge) ...
    np.testing.assert_allclose(x + err, 0.0, atol=0.01)
    # ... and the keyframes that close the loop agree with those that opened it
    after = pair_log_ratio(views[N - 1], views[0], np.exp(x[N - 1]), np.exp(x[0]))
    assert after is not None and abs(after.log_ratio) < 0.01
    assert float(np.median(adj.residuals_after)) < 0.005


def test_robust_loss_ignores_an_inconsistent_pair() -> None:
    # a chain 0-1-2-3 measured consistently (true log scales 0, .1, .2, .3) plus one wrong pair
    truth = np.array([0.0, 0.1, 0.2, 0.3])
    pairs = [(i, j, truth[i] - truth[j], 1.0) for i in range(4) for j in range(i + 1, 4)]
    pairs.append((0, 3, 0.5, 1.0))  # a pair whose ratio no scale explains
    x = adjust_log_scales(4, [(i, j, -m, w) for i, j, m, w in pairs], {0}).log_scale
    np.testing.assert_allclose(x, truth, atol=0.01)
    plain = adjust_log_scales(4, [(i, j, -m, w) for i, j, m, w in pairs], {0},
                              scales=(float("inf"),)).log_scale
    assert abs(plain[3] - truth[3]) > 0.05  # plain least squares is pulled by it


def test_fixed_keyframes_hold_and_isolated_ones_keep_their_scale() -> None:
    pairs = [(0, 1, 0.2, 1.0), (1, 2, -0.1, 1.0)]
    x = adjust_log_scales(4, pairs, {0}).log_scale
    assert x[0] == 0.0 and x[3] == pytest.approx(0.0, abs=1e-9)
    assert x[1] == pytest.approx(0.2, abs=1e-6) and x[2] == pytest.approx(0.1, abs=1e-6)
    assert not adjust_log_scales(3, pairs, {0, 1, 2}).log_scale.any()


# ------------------------------------------------------------------------------------------------
# the mapper's use of it


def record(k: int, T: Pose, method: str, low: bool = False, update: int = 1) -> FrameRecord:
    K = Intrinsics(KG.fx * 2, KG.fy * 2, KG.cx * 2, KG.cy * 2, KG.width * 2, KG.height * 2)
    return FrameRecord(k, f"f{k:06d}", f"frames/f{k:06d}.jpg", "synthetic", 1, K.width, K.height,
                       K, T, KG.width, KG.height, "multiview", update, 1.0, low,
                       {"depth_scale_method": method})


def new_frame(rec: FrameRecord, raw: np.ndarray, scale: float) -> Any:
    rec.depth_scale = scale
    frame = SimpleNamespace(depth=raw.astype(np.float32), valid=raw > 0)
    return SimpleNamespace(record=rec, frame=frame, depth=(raw * scale).astype(np.float32))


class _Tx:
    """The part of a map transaction the adjustment uses: current files and staged writes."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def current(self, rel: str) -> Path:
        return self.root / rel

    def save_npy(self, rel: str, arr: np.ndarray) -> None:
        np.save(self.root / rel, arr)


def test_an_update_that_closes_the_loop_re_scales_the_map_too(tmp_path: Path) -> None:
    """A map of the first two thirds of the loop (their scales drifted), extended by the last
    third, which closes the loop: the stored keyframes are adjusted with the new ones, as if the
    whole loop had come in one update — except the map's seed and sparse-scaled keyframes, which
    hold the metric scale. Re-scaled stored keyframes have their depth rewritten."""
    poses = loop_poses()
    err = drift(1)
    raw = [depth_grid(T, KG).astype(np.float64) / 1.2 for T in poses]  # the model's own scale
    chain = 1.2 * np.exp(err)  # sequential alignment: correct scale times the accumulated error
    old = [record(k, poses[k], "seed" if k == 0 else "sparse" if k == 5 else "dense", update=1)
           for k in range(20)]
    for k, rec in enumerate(old):
        rec.depth_scale = float(chain[k])
        d = tmp_path / frame_file(rec.name, "depth.npy")
        d.parent.mkdir(parents=True, exist_ok=True)
        np.save(d, (raw[k] * chain[k]).astype(np.float16))
    new = [new_frame(record(k, poses[k], "dense", low=(k == 24), update=2), raw[k], chain[k])
           for k in range(20, N)]
    ctx = SimpleNamespace(old_frames=old, new=new, notes={}, rescaled={}, tx=_Tx(tmp_path))
    msgs: list[str] = []
    api._adjust_depth_scales(ctx, msgs.append)  # type: ignore[arg-type]
    assert msgs and "depth scales adjusted" in msgs[0] and "stored keyframes re-scaled" in msgs[0]
    note = ctx.notes["depth_scale_adjustment"]
    assert note["fixed"] == 2 and note["adjusted"] == N - 2
    assert note["pair_ratio_p90_after"] < note["pair_ratio_p90_before"]
    # the seed and the sparse-scaled keyframe hold; the others are re-scaled, their depth too
    assert 0 not in ctx.rescaled and 5 not in ctx.rescaled and len(ctx.rescaled) >= 10
    for k, c in ctx.rescaled.items():
        assert old[k].depth_scale == pytest.approx(chain[k] * c)
        stored = np.load(tmp_path / frame_file(old[k].name, "depth.npy")).astype(np.float64)
        np.testing.assert_allclose(stored, raw[k] * chain[k] * c, rtol=2e-3)
    # every keyframe (up to what the anchored sparse keyframe carries) agrees with the others:
    # the closing keyframe with the first
    scales = np.array([r.depth_scale for r in old] + [nf.record.depth_scale for nf in new])
    closing = pair_log_ratio(DepthView(raw[-1] * scales[-1], KG.K(), poses[-1].matrix()),
                             DepthView(raw[0] * scales[0], KG.K(), poses[0].matrix()))
    assert closing is not None and abs(closing.log_ratio) < 0.015
    for nf in new:
        np.testing.assert_allclose(nf.depth, nf.frame.depth * nf.record.depth_scale, rtol=1e-6)
        assert "depth_scale_adjusted" in nf.record.stats


def test_mapper_holds_the_seed_of_a_new_map() -> None:
    poses = loop_poses()
    err = drift(2)
    raw = [depth_grid(T, KG).astype(np.float64) for T in poses]
    new = [new_frame(record(k, poses[k], "seed" if k == 0 else "dense"), raw[k],
                     float(np.exp(err[k]))) for k in range(N)]
    ctx = SimpleNamespace(old_frames=[], new=new, notes={}, rescaled={}, tx=None)
    api._adjust_depth_scales(ctx, lambda m: None)  # type: ignore[arg-type]
    scales = np.array([nf.record.depth_scale for nf in new])
    assert scales[0] == 1.0 and "depth_scale_adjusted" not in new[0].record.stats
    np.testing.assert_allclose(np.log(scales), 0.0, atol=0.012)


def test_stored_objects_move_with_their_rescaled_keyframes() -> None:
    from oh_my_slam.mapping import objects as mo

    T0, T1 = head_pose(0.0), head_pose(10.0)
    recs = [record(0, T0, "seed"), record(1, T1, "dense"), record(2, head_pose(20.0), "dense")]
    rng = np.random.default_rng(0)
    centre = T0.t + 2.0 * T0.R[:, 2]
    pts = rng.uniform(-0.1, 0.1, (400, 3)) + centre
    o = mo.MapObject(7, "cup", {"cup": 1.6}, [0.8, 0.8], mo.canonical_points(pts), frames=[0, 1],
                     obs_depth=2.0)

    def sighting(frame: int, p: np.ndarray) -> mo.Sighting:
        lo, hi = np.percentile(p, [2, 98], axis=0)
        return mo.Sighting(frame, len(p), 0.0, tuple(p.mean(0)), tuple(lo), tuple(hi))
    o.sightings = sorted([sighting(0, pts[:200]), sighting(1, pts[200:])], key=mo.Sighting.key)
    mo.refit(o, None)
    other = mo.MapObject(9, "lamp", {"lamp": 0.8}, [0.8], mo.canonical_points(pts + 1.0),
                         frames=[2], obs_depth=2.0)
    other.sightings = [sighting(2, pts + 1.0)]
    state = mo.ObjectState([o, other], 20)
    before = o.obb.center.copy() if o.obb is not None else None
    moved = mo.rescale_objects(state, {1: 0.9}, recs)
    assert moved == {7}  # the lamp's keyframe kept its scale
    s1 = next(s for s in o.sightings if s.frame == 1)
    np.testing.assert_allclose(s1.centroid, T1.t + 0.9 * (pts[200:].mean(0) - T1.t), atol=1e-6)
    s0 = next(s for s in o.sightings if s.frame == 0)
    np.testing.assert_allclose(s0.centroid, pts[:200].mean(0), atol=1e-6)  # not re-scaled
    # the points move by the sightings' mean factor (half the points at 0.9, half at 1)
    assert o.obs_depth == pytest.approx(2.0 * np.sqrt(0.9), rel=1e-3)
    assert o.obb is not None and before is not None
    dist_before = np.linalg.norm(before - T0.t)
    assert np.linalg.norm(o.obb.center - T0.t) == pytest.approx(dist_before * np.sqrt(0.9),
                                                                rel=0.02)
    assert not mo.rescale_objects(state, {}, recs)
