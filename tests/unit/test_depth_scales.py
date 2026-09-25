"""Global adjustment of the keyframes' depth (``reconstruction.depth.adjust_depth_corrections``,
``mapping.api._adjust_depth_scales``, ``mapping.objects.rescale_objects``): a head turning a full
circle in a box room, with the scale error that aligning each keyframe to its predecessors
accumulates around the loop, and per-keyframe near/far errors that no single scale removes."""

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
    MAX_FACTOR,
    BinRatio,
    DepthCorrection,
    DepthView,
    adjust_depth_corrections,
    pair_bins,
    pair_log_ratio,
    transfer_log_ratios,
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


def tilts(seed: int = 0, size: float = 0.12) -> np.ndarray:
    """Per-keyframe near/far error (slope of the log error in log depth), the first one exact."""
    t = np.random.default_rng(seed).uniform(-size, size, N)
    t[0] = 0.0
    return t


def distorted(true: np.ndarray, log_scale: float, slope: float) -> np.ndarray:
    """``true`` with a log error ``log_scale + slope · log(true / 2 m)``."""
    return true * np.exp(log_scale + slope * np.log(true / 2.0))


def overlapping(poses: list[Pose], max_deg: float = 45.0) -> list[tuple[int, int]]:
    F = np.array([T.R[:, 2] for T in poses])
    cos = F @ F.T
    return [(i, j) for i in range(N) for j in range(i + 1, N)
            if cos[i, j] > np.cos(np.radians(max_deg))]


def solve(views: list[DepthView], pairs: list[tuple[int, int]], fixed: set[int],
          rounds: int = 2) -> list[DepthCorrection]:
    """The mapper's rounds: measure at the corrected depths, solve, compose."""
    corr = [DepthCorrection() for _ in views]
    for _ in range(rounds):
        cur = [v.corrected(c) for v, c in zip(views, corr, strict=True)]
        piv = np.array([v.log_median() for v in cur])
        meas = [b for i, j in pairs for b in pair_bins(i, j, cur[i], cur[j])]
        adj = adjust_depth_corrections(len(views), meas, piv, fixed)
        corr = [c.then(d) for c, d in zip(corr, adj.corrections, strict=True)]
    return corr


def worst_pair(views: list[DepthView], corr: list[DepthCorrection],
               pairs: list[tuple[int, int]]) -> float:
    """The largest median |log ratio| over the pairs, at the corrected depths."""
    out = 0.0
    for i, j in pairs:
        r = transfer_log_ratios(views[i].corrected(corr[i]), views[j].corrected(corr[j]))
        if len(r) >= 300:
            out = max(out, float(np.median(np.abs(r))))
    return out


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
    corr = solve(views, pairs, {0})
    # every keyframe's error is undone (the first one holds the gauge), without a tilt ...
    for d, v, c in zip(true, views, corr, strict=True):
        ok = d > 0
        np.testing.assert_allclose(np.log(c.apply(v.depth)[ok] / d[ok]), 0.0, atol=0.012)
        assert abs(c.slope) < 0.02
    # ... and the keyframes that close the loop agree with those that opened it
    after = pair_log_ratio(views[N - 1].corrected(corr[N - 1]), views[0])
    assert after is not None and abs(after.log_ratio) < 0.01


def test_near_far_errors_are_removed_where_one_scale_cannot() -> None:
    """Each keyframe places near surfaces too close and far ones too far (or the reverse) by up
    to ~12 % either way, differently from its neighbours: one scale per keyframe leaves pairs
    disagreeing; a scale and a tilt per keyframe restore the true depth."""
    poses = loop_poses()
    true = [depth_grid(T, KG).astype(np.float64) for T in poses]
    err, tilt = drift(3), tilts(3)
    views = [DepthView(distorted(d, e, t), KG.K(), T.matrix()) for d, e, t, T in
             zip(true, err, tilt, poses, strict=True)]
    pairs = overlapping(poses)
    corr = solve(views, pairs, {0})
    # the pairs agree: within 2.5 % where one scale per keyframe leaves 6 % and more
    scale_only = [DepthCorrection(c.log_scale, 0.0, c.pivot) for c in corr]
    assert worst_pair(views, [DepthCorrection() for _ in views], pairs) > 0.1
    assert worst_pair(views, scale_only, pairs) > 0.05
    assert worst_pair(views, corr, pairs) < 0.025
    # and each keyframe is closer to the true depth (the prior "no tilt" holds back part of the
    # tilts, which the pairs observe only as differences between keyframes)
    for d, v, c in zip(true, views, corr, strict=True):
        ok = d > 0
        before = np.abs(np.log(v.depth[ok] / d[ok]))
        resid = np.abs(np.log(c.apply(v.depth)[ok] / d[ok]))
        assert float(np.median(resid)) < 0.02 and float(np.percentile(resid, 95)) < 0.05
        assert float(np.percentile(resid, 95)) <= float(np.percentile(before, 95)) + 0.01


def _bins(pairs: list[tuple[int, int, float, float]]) -> list[BinRatio]:
    """One bin per pair at the keyframes' pivot (no lever on the slopes)."""
    return [BinRatio(i, j, m, 0.0, 0.0, w) for i, j, m, w in pairs]


def test_robust_loss_ignores_an_inconsistent_pair() -> None:
    # a chain 0-1-2-3 measured consistently (true log scales 0, .1, .2, .3) plus one wrong pair
    truth = np.array([0.0, 0.1, 0.2, 0.3])
    pairs = [(i, j, truth[i] - truth[j], 1.0) for i in range(4) for j in range(i + 1, 4)]
    pairs.append((0, 3, 0.8, 1.0))  # a pair whose ratio no scale explains
    meas = _bins([(i, j, -m, w) for i, j, m, w in pairs])
    x = np.array([c.log_scale for c in
                  adjust_depth_corrections(4, meas, np.zeros(4), {0}).corrections])
    np.testing.assert_allclose(x, truth, atol=0.02)
    plain = adjust_depth_corrections(4, meas, np.zeros(4), {0}, scales=(float("inf"),))
    assert abs(plain.corrections[3].log_scale - truth[3]) > 0.1  # least squares is pulled by it


def test_fixed_keyframes_hold_and_isolated_ones_keep_their_scale() -> None:
    meas = _bins([(0, 1, 0.2, 1.0), (1, 2, -0.1, 1.0)])
    corr = adjust_depth_corrections(4, meas, np.zeros(4), {0}).corrections
    assert corr[0].identity and corr[3].log_scale == pytest.approx(0.0, abs=1e-9)
    assert corr[1].log_scale == pytest.approx(0.2, abs=1e-6)
    assert corr[2].log_scale == pytest.approx(0.1, abs=1e-6)
    assert all(c.identity for c in
               adjust_depth_corrections(3, meas, np.zeros(3), {0, 1, 2}).corrections)


def test_a_scale_fixed_keyframe_keeps_its_scale_but_tilts() -> None:
    """Keyframe 0 holds the scale; its near field (1 log unit below the pivot) is 10 % too deep
    against keyframes 1 and 2, its far field agrees: it tilts, its scale stays."""
    meas = [BinRatio(0, j, r, lever, lever, 1.0) for j in (1, 2)
            for r, lever in ((0.1, -1.0), (0.0, 0.0), (-0.1, 1.0)) for _ in range(20)]
    fixed = adjust_depth_corrections(3, meas, np.zeros(3), {0}).corrections
    held = adjust_depth_corrections(3, meas, np.zeros(3), set(), scale_fixed={0}).corrections
    assert fixed[0].identity and held[0].log_scale == 0.0
    # the tilt between them is found either way; held, keyframe 0 takes its share of it
    for c in (fixed, held):
        assert c[0].slope - c[1].slope == pytest.approx(0.1, abs=0.03)
    assert held[0].slope > 0.04 and abs(held[1].slope) < abs(fixed[1].slope)


def test_depth_correction_composes_and_is_clamped() -> None:
    d = np.array([0.3, 0.7, 1.5, 3.0, 6.0, 0.0])
    a = DepthCorrection(0.05, -0.1, float(np.log(1.5)))
    b = DepthCorrection(-0.02, 0.08, float(np.log(1.4)))
    np.testing.assert_allclose(a.then(b).apply(d), b.apply(a.apply(d)), rtol=1e-9)
    assert a.apply(d)[-1] == 0.0  # invalid depth stays invalid
    steep = DepthCorrection(0.0, 0.3, 0.0)
    assert steep.factor(np.array([1e4]))[0] == pytest.approx(MAX_FACTOR)
    assert steep.factor(np.array([1e-4]))[0] == pytest.approx(1.0 / MAX_FACTOR)
    assert DepthCorrection().identity and not a.identity
    assert a.scale == pytest.approx(np.exp(0.05)) and a.exponent == pytest.approx(0.9)


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
    # the seed and the sparse-scaled keyframe hold their scale (they may tilt); the others are
    # re-scaled, their depth too
    assert all(ctx.rescaled[k].log_scale == 0.0 for k in (0, 5) if k in ctx.rescaled)
    assert len(ctx.rescaled) >= 10
    for k, c in ctx.rescaled.items():
        assert old[k].depth_scale == pytest.approx(chain[k] * c.scale)
        stored = np.load(tmp_path / frame_file(old[k].name, "depth.npy")).astype(np.float64)
        np.testing.assert_allclose(stored, c.apply(raw[k] * chain[k]), rtol=2e-3)
    # every keyframe (up to what the anchored sparse keyframe carries) agrees with the others:
    # the closing keyframe with the first
    first = np.load(tmp_path / frame_file(old[0].name, "depth.npy")).astype(np.float64)
    closing = pair_log_ratio(DepthView(new[-1].depth, KG.K(), poses[-1].matrix()),
                             DepthView(first, KG.K(), poses[0].matrix()))
    assert closing is not None and abs(closing.log_ratio) < 0.015
    for nf in new:
        assert "depth_scale_adjusted" in nf.record.stats and "depth_exponent" in nf.record.stats


def test_mapper_holds_the_seed_of_a_new_map() -> None:
    poses = loop_poses()
    err = drift(2)
    raw = [depth_grid(T, KG).astype(np.float64) for T in poses]
    new = [new_frame(record(k, poses[k], "seed" if k == 0 else "dense"), raw[k],
                     float(np.exp(err[k]))) for k in range(N)]
    ctx = SimpleNamespace(old_frames=[], new=new, notes={}, rescaled={}, tx=None)
    api._adjust_depth_scales(ctx, lambda m: None)  # type: ignore[arg-type]
    scales = np.array([nf.record.depth_scale for nf in new])
    assert scales[0] == 1.0 and new[0].record.stats.get("depth_scale_adjusted", 1.0) == 1.0
    np.testing.assert_allclose(np.log(scales), 0.0, atol=0.012)
    for nf, d in zip(new, raw, strict=True):  # the aligned depth is the true depth again
        ok = d > 0
        assert float(np.median(np.abs(np.log(nf.depth[ok] / d[ok])))) < 0.012


def test_held_keyframes_of_a_large_scene_keep_their_median_depth() -> None:
    """A street-sized scene (the room scaled 5×: median depths 7-12 m) in which every keyframe
    holds its scale (sparse-scaled, as in an outdoor video) and has a near/far error about its
    median depth: the adjustment tilts each keyframe about its own median, which stays where the
    SfM points put it. (Composed about 1 m, the held scale placed each keyframe median^b too deep
    or too shallow: 25-35 % at 10 m.)"""
    s = 5.0
    poses = [Pose(T.R, s * T.t) for T in loop_poses()]
    true = [s * depth_grid(T, KG).astype(np.float64) for T in loop_poses()]
    tilt = tilts(4)
    raw = []
    for d, t in zip(true, tilt, strict=True):
        ok = d > 0
        piv = float(np.log(np.median(d[ok])))
        raw.append(np.where(ok, d * np.exp(t * (np.log(np.where(ok, d, 1.0)) - piv)), 0.0))
    new = [new_frame(record(k, poses[k], "sparse"), raw[k], 1.0) for k in range(N)]
    ctx = SimpleNamespace(old_frames=[], new=new, notes={}, rescaled={}, tx=None)
    api._adjust_depth_scales(ctx, lambda m: None)  # type: ignore[arg-type]
    assert ctx.notes["depth_scale_adjustment"]["fixed"] == N
    exps = ctx.notes["depth_scale_adjustment"]["exponent_range"]
    assert exps[1] - exps[0] > 0.05  # the keyframes were tilted
    medians = [float(np.median(nf.depth[d > 0]) / np.median(r[d > 0]))
               for nf, r, d in zip(new, raw, true, strict=True)]
    np.testing.assert_allclose(medians, 1.0, atol=2e-3)
    for nf, r, d in zip(new, raw, true, strict=True):
        ok = d > 0
        err = np.abs(np.log(nf.depth[ok] / d[ok]))
        assert float(np.median(err)) < 0.03
        assert float(np.median(err)) <= float(np.median(np.abs(np.log(r[ok] / d[ok])))) + 0.005


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
    moved = mo.rescale_objects(state, {1: DepthCorrection(float(np.log(0.9)))}, recs)
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
