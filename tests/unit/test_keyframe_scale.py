"""A keyframe's own monocular depth scale by motion stereo (map.md §1 "Keyframe depth scale", §2
Floor check; api.md §4 ``stereo_scale``; oms/2.14) on AiNex-like views (``tests.synth.stereo``):
the humanoid's head camera 0.33 m above the floor walking straight ahead, the fake model's depth
the truth in its own metres (0.8 x, its shape stretched and bent: ``model_depth``), so the
keyframe's true scale is about 1.25 raw depth to metres."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oh_my_slam.core.types import Intrinsics
from oh_my_slam.mapping import anchored as an
from oh_my_slam.mapping import motion
from oh_my_slam.mapping.anchored import AnchoredMap, FrameIn, Keyframe
from oh_my_slam.service.prior_first import stereo_scale
from tests.synth import stereo as syn

K = Intrinsics(syn.F, syn.F, syn.CX, syn.CY, syn.W, syn.H, "model")


@pytest.fixture(scope="module")
def walk() -> list[syn.Shot]:
    """The head at the origin, 0.25 m and 0.5 m straight ahead, then turned 90 degrees."""
    room = syn.ainex_room()
    return [syn.render(room, syn.head(0.0), 0), syn.render(room, syn.head(0.25), 1),
            syn.render(room, syn.head(0.5), 2), syn.render(room, syn.head(0.5, yaw_deg=90.0), 3)]


def prior(T: np.ndarray, trust: str = "drifting", sigma: float = 0.05) -> dict[str, Any]:
    return {"trust": trust, "T_map_cam": T.tolist(), "sigma_pos_m": sigma, "sigma_rot_deg": 2.0}


def frame(name: str, s: syn.Shot, trust: str = "drifting", sigma: float = 0.05,
          camera: str = "cam") -> FrameIn:
    d = syn.model_depth(s.depth)
    return FrameIn(name, camera, s.rgb, K, prior(s.T, trust, sigma), None, None, syn.HEAD_M, 0.01,
                   1000, lambda d=d: (d, K, None, None))


def mapped(tmp_path: Path, walk: list[syn.Shot], n: int = 3, **kw: Any) -> AnchoredMap:
    """The walk sent one keyframe per job."""
    m = AnchoredMap(tmp_path / "m")
    for i, s in enumerate(walk[:n]):
        m.update([frame(f"f{i}", s, **kw)], lambda _: None)
    return m


# -- the measurement -------------------------------------------------------------------------------


def test_a_straight_move_measures_each_keyframe_s_scale(tmp_path: Path,
                                                        walk: list[syn.Shot]) -> None:
    """Three keyframes 0.25 m apart along the view, drifting priors exact: each keyframe's own
    scale (the model's 0.8 x metres: about 1.25) within 10 % with its sigma, against its
    neighbours (the first one measured again once the second arrives), persisted and reported;
    registration is unchanged by it (map.md §1)."""
    m = mapped(tmp_path, walk)
    kfs = m.confirmed()
    assert [k.name for k in kfs] == ["f0", "f1", "f2"]
    for k in kfs:
        assert k.stereo_scale is not None and k.stereo_sigma is not None
        assert k.stereo_scale == pytest.approx(1.25, rel=0.12), k.name
        assert 0.1 <= k.stereo_sigma < an.MONO_FLOOR_BOUND_REL
        assert k.confident_scale() == k.stereo_scale
        assert k.stereo_pairs and set(k.stereo_pairs) <= {"f0", "f1", "f2"} - {k.name}
        assert not k.stereo_judged and k.lifts()
    m.save()
    again = AnchoredMap.load(m.root)
    k1 = again.keyframe(kfs[1].id)
    assert k1 is not None and k1.stereo_scale == kfs[1].stereo_scale
    assert k1.stereo_pairs == kfs[1].stereo_pairs
    row = stereo_scale(k1)
    assert row is not None and row["scale"] == round(kfs[1].stereo_scale, 5)
    assert row["neighbours"] == kfs[1].stereo_pairs and row["judged"] is False
    assert stereo_scale(None) is None
    # the registration lifts at k as before (oms/2.14 measures, it does not lift)
    assert m._depth_factor(kfs[1]) == pytest.approx(m.scale.k / kfs[1].k_used)


def test_a_map_written_before_oms_2_14_reads_without_a_scale(tmp_path: Path,
                                                             walk: list[syn.Shot]) -> None:
    m = mapped(tmp_path, walk, 1)
    d = m.confirmed()[0].to_json()
    for key in ("stereo_scale", "stereo_sigma", "stereo_pairs", "stereo_judged"):
        d.pop(key)
    k = Keyframe.from_json(d)
    assert (k.stereo_scale, k.stereo_sigma, k.stereo_pairs, k.stereo_judged) == (
        None, None, None, False)
    assert k.confident_scale() is None and stereo_scale(k) is None


def test_no_baseline_no_scale(tmp_path: Path, walk: list[syn.Shot]) -> None:
    """A turn in place (the neighbour 90 degrees away: no common view), a step across a client's
    fix (sqrt(|σa² − σb²|) too large for its length), and a keyframe with metric priors of 0.05 m 0.25 m
    apart (σ_t / |t| 0.28): none measured. With metric priors of 0.01 m the same walk measures
    (σ_t the root sum of squares of both)."""
    m = AnchoredMap(tmp_path / "turn")
    for i in (2, 3):
        m.update([frame(f"t{i}", walk[i])], lambda _: None)
    assert all(k.stereo_scale is None for k in m.confirmed())
    fixed = AnchoredMap(tmp_path / "fixed")  # a client fix between them: sigma 0.4, then 0.05
    fixed.update([frame("g0", walk[0], sigma=0.4)], lambda _: None)
    fixed.update([frame("g1", walk[1], sigma=0.05)], lambda _: None)
    assert all(k.stereo_scale is None for k in fixed.confirmed())
    metric = mapped(tmp_path / "metric", walk, 2, trust="metric")
    assert all(k.stereo_scale is None for k in metric.confirmed())
    tight = mapped(tmp_path / "tight", walk, 2, trust="metric", sigma=0.01)
    assert all(k.stereo_scale is not None for k in tight.confirmed())


def test_neighbours_are_near_in_the_order_of_addition_of_one_camera(
        tmp_path: Path, walk: list[syn.Shot]) -> None:
    m = mapped(tmp_path, walk, 2)
    a, b = m.confirmed()
    assert m._near(a, b)
    b.camera_id = "other"
    assert not m._near(a, b)
    b.camera_id = a.camera_id
    m.keyframes = [a] + [Keyframe.from_json({**a.to_json(), "id": f"x{i}"})
                         for i in range(an.KEYFRAME_PAIR_MAX_GAP)] + [b]
    assert not m._near(a, b) and m._scale_neighbours(a, [a, b]) == []
    m.keyframes = [a, b]
    c = Keyframe.from_json({**b.to_json(), "id": "kc"})
    assert c.prior is not None
    c.prior = {**c.prior, "trust": "metric"}
    m.keyframes.append(c)
    assert [n[0] for n in m._scale_neighbours(a, [a, b, c])] == [b]


def test_the_odometry_scale_sigma(tmp_path: Path) -> None:
    m = AnchoredMap(tmp_path / "m")
    assert m._odo_sigma_rel() == 0.0  # no correction: the priors' metres are the map's
    m.drift.ref_T = np.eye(4)
    m.drift.evidence = [[1.0, 1.0], [1.0, 1.1]]
    assert m._odo_sigma_rel() == an.DRIFT_POS_REL
    m.drift.evidence = [[1.0, 1.0], [1.0, 1.1], [1.0, 0.9], [1.0, 1.0], [0.0, 1.0]]
    r = np.log([1.0, 1.1, 0.9, 1.0])
    want = 1.4826 * float(np.median(np.abs(r - np.median(r)))) / 2.0
    assert m._odo_sigma_rel() == pytest.approx(want)


def test_a_keyframe_without_features_or_depth_has_no_scale(tmp_path: Path,
                                                           walk: list[syn.Shot]) -> None:
    m = mapped(tmp_path, walk, 2)
    a, b = m.confirmed()
    nbrs = m._scale_neighbours(a, [a, b])
    assert nbrs
    a.grid = None
    a.K_grid = None
    m._measure_scale(a, nbrs)
    assert a.stereo_scale is None and a.stereo_pairs is None
    a2, b2 = mapped(tmp_path / "2", walk, 2).confirmed()
    m2 = AnchoredMap(tmp_path / "2" / "m")
    m2.keyframes = [a2, b2]
    nb = m2._scale_neighbours(a2, [a2, b2])
    b2.feats = None
    m2.root = tmp_path / "nowhere"  # its features are not on disk either
    m2._measure_scale(a2, nb)
    assert a2.stereo_scale is None


# -- the floor check about the measured scale ------------------------------------------------------


def test_an_unjudged_keyframe_is_judged_about_its_confident_measured_scale(
        tmp_path: Path, walk: list[syn.Shot]) -> None:
    """A job keyframe the floor check did not judge about the map's k (its own floor-check
    scale beyond the monocular bound) whose measured scale is confident and agrees with its
    floor: judged (``stereo_judged``), its content at its floor-check scale, ``depth_scale``
    reported; its depth still serves later frames as an unjudged keyframe's (``lifts``). A
    measured scale its floor disagrees with judges nothing."""
    m = mapped(tmp_path, walk, 2)
    a, b = m.confirmed()
    m.begin_job()
    m._stored_ids = {a.id}  # b is the job's
    m.scale.k = a.k_used = b.k_used = (b.stereo_scale or 1.0) / 2.5  # k far from its own scale
    b.floor_unjudged, b.floor_factor = True, None
    judged = m._measure_scales()
    assert judged == {b.id}
    assert b.stereo_judged and not b.floor_unjudged and b.floor_factor is not None
    assert b.depth_scale is not None and b.depth_scale == pytest.approx(b.stereo_scale, rel=0.3)
    assert not b.lifts() and not b.floor_withheld()
    assert stereo_scale(b)["judged"] is True  # type: ignore[index]
    # a measured scale its floor disagrees with: not judged
    m._stored_ids = {a.id}
    b.floor_unjudged, b.floor_factor = True, None
    b.stereo_judged = False
    real = motion.keyframe_scale
    try:
        motion.keyframe_scale = lambda pairs, k: motion.KeyframeScale(  # type: ignore[assignment]
            3.0 * (b.stereo_scale or 1.0), 0.1, [0])
        assert m._measure_scales() == set()
    finally:
        motion.keyframe_scale = real  # type: ignore[assignment]
    assert b.floor_unjudged and not b.stereo_judged
    # a loose measurement judges nothing
    b.stereo_sigma = an.MONO_FLOOR_BOUND_REL
    assert b.confident_scale() is None


def test_a_judged_job_keyframe_fuses_its_content_at_its_floor_scale(
        tmp_path: Path, walk: list[syn.Shot], monkeypatch: pytest.MonkeyPatch) -> None:
    """Through a job: the floor check, about k, does not judge the second keyframe (forced: its
    own scale 2.5 x k), its measured scale does, and the job's fusion takes its content at its
    floor-check scale (``content_grid``)."""
    m = mapped(tmp_path, walk, 1)
    real = AnchoredMap.floor_check
    calls: list[float | None] = []

    def check(self: AnchoredMap, *a: Any, **kw: Any) -> an.FloorCheck:
        ref = a[8] if len(a) > 8 else kw.get("ref_scale")
        calls.append(ref)
        got = real(self, *a, **kw)
        if ref is None and a[0] == "mono" and self._defer is not None:
            return an.FloorCheck(judged=False)  # as if beyond the bound about k
        return got

    monkeypatch.setattr(AnchoredMap, "floor_check", check)
    fused: list[np.ndarray] = []
    real_fuse = AnchoredMap._fuse
    monkeypatch.setattr(AnchoredMap, "_fuse", lambda self, img, ig, kf, depth, Kg: (
        fused.append(depth), real_fuse(self, img, ig, kf, depth, Kg)))
    m.update([frame("f1", walk[1])], lambda _: None)
    k1 = next(k for k in m.confirmed() if k.name == "f1")
    assert k1.stereo_judged and k1.floor_factor is not None
    assert any(r is not None for r in calls)
    assert fused and np.allclose(fused[-1], m.content_grid(k1))  # type: ignore[arg-type]


# -- the pairs ---------------------------------------------------------------------------------------


def _feats(s: syn.Shot) -> an.Features:
    return an.features(s.rgb)


def test_pair_scale_statuses(walk: list[syn.Shot]) -> None:
    """``pair_scale``: a 0.25 m move known to 0.03 m (σ_t / |t| 0.12) is ``used`` at about 1.25
    with σ = hypot(σ_t / |t|, the fit's error, its spread); known to 0.1 m (0.4) it is
    ``baseline_short``; no common view is ``no_overlap``; a depth without any value gives no
    fit (``too_few_points``)."""
    ref, other = _feats(walk[0]), _feats(walk[1])
    T = syn.relative(walk[0].T, walk[1].T)
    raw = syn.model_depth(walk[0].depth)
    o = motion.Other(walk[1].rgb, None, T, 0.03, 2.0)
    got = motion.pair_scale(0, ref, other, K, o, raw, K)
    assert got.status == "used" and got.scale == pytest.approx(1.25, rel=0.12)
    assert got.sigma_rel is not None and got.sigma_rel >= 0.12 and got.points >= 30
    loose = motion.pair_scale(1, ref, other, K, motion.Other(walk[1].rgb, None, T, 0.1, 2.0),
                              raw, K)
    assert loose.status == "baseline_short" and loose.scale is None
    far = motion.pair_scale(2, ref, _feats(walk[3]), K,
                            motion.Other(walk[3].rgb, None, syn.relative(walk[0].T, walk[3].T),
                                         0.03, 2.0), raw, K)
    assert far.status in ("no_overlap", "too_few_points", "pose_inconsistent")
    empty = motion.pair_scale(3, ref, other, K, o, np.zeros_like(raw), K)
    assert empty.status == "too_few_points" and empty.scale is None


def test_pair_scale_with_too_few_points(walk: list[syn.Shot],
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    ref, other = _feats(walk[0]), _feats(walk[1])
    T = syn.relative(walk[0].T, walk[1].T)
    monkeypatch.setattr(motion, "BASELINE_MIN_POINTS", 10 ** 6)
    got = motion.pair_scale(0, ref, other, K, motion.Other(walk[1].rgb, None, T, 0.03, 2.0),
                            syn.model_depth(walk[0].depth), K)
    assert got.status == "too_few_points" and got.points > 0


def P(i: int, s: float | None, sig: float | None = 0.1, status: str = "used") -> motion.PairScale:
    return motion.PairScale(i, status, s, sig, 50)


def test_the_pairs_combined() -> None:
    """``keyframe_scale``: pairs outside the plausible range of k dropped; more than half must
    agree within 3 of their sigmas of the weighted median (two pairs: both); the agreeing pairs'
    inverse-variance mean in ln scale, its sigma never below the best pair's, at least their
    weighted spread."""
    ks = motion.keyframe_scale
    assert ks([], 1.0) is None
    assert ks([P(0, None, None, "no_overlap"), P(1, 5.0)], 1.0) is None  # beyond 3 x k
    one = ks([P(0, 1.5, 0.12)], 1.0)
    assert one is not None and (one.scale, one.sigma_rel, one.used) == (pytest.approx(1.5), 0.12,
                                                                          [0])
    assert ks([P(0, 1.0, 0.1), P(1, 2.0, 0.1)], 1.0) is None  # two pairs that disagree
    got = ks([P(0, 1.0, 0.1), P(1, 1.1, 0.2), P(2, 2.5, 0.1)], 1.0)
    assert got is not None and got.used == [0, 1]
    w = np.array([100.0, 25.0])
    mean = float((w * np.log([1.0, 1.1])).sum() / w.sum())
    assert got.scale == pytest.approx(math.exp(mean))
    assert got.sigma_rel == pytest.approx(0.1)
    wide = ks([P(0, 1.0, 0.01), P(1, 1.02, 0.01)], 1.0)
    assert wide is not None and wide.sigma_rel == pytest.approx(0.01)
    spread = ks([P(0, 1.0, 0.02), P(1, 1.1, 0.04)], 1.0)
    assert spread is not None and spread.sigma_rel > 0.02


def test_confident_scale_rules() -> None:
    k = Keyframe("k", "n", "cam", K, np.eye(4), 0.1, 1.0, "fused", None, "sensor",
                 stereo_scale=1.0, stereo_sigma=0.1)
    assert k.confident_scale() is None  # a depth image's scale is its sensor's
    k.depth_kind = "mono"
    assert k.confident_scale() == 1.0
    k.stereo_sigma = None
    assert k.confident_scale() is None
