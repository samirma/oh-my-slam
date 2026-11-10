"""Prior-first registration (``mapping.anchored``) on a ray-cast textured room
(``tests.synth.room``): cold starts, sweeps with and without depth images, provisional and
conflicting frames, drifting odometry, the depth scale's sources and the incremental cloud."""

from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from oh_my_slam.core.types import Intrinsics
from oh_my_slam.mapping import anchored as an
from oh_my_slam.mapping.anchored import AnchoredMap, Cloud, Drift, FrameIn, ScaleState
from tests.synth.room import Room, base_pose, look, render

K = Intrinsics(160.0, 160.0, 160.0, 120.0, 320, 240)
ROOM = Room()


@functools.lru_cache(maxsize=256)
def _render(key: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    T = np.array(key).reshape(4, 4)
    return render(ROOM, T, K)


def shot(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _render(tuple(np.round(T, 9).ravel().tolist()))


def prior(T: np.ndarray, trust: str = "metric", sp: float = 0.02, sr: float = 1.0
          ) -> dict[str, Any]:
    return {"trust": trust, "T_map_cam": np.asarray(T).tolist(), "sigma_pos_m": sp,
            "sigma_rot_deg": sr}


def mono_of(z: np.ndarray, scale: float = 1.0, up: Any = (0.0, -1.0, 0.0),
            descriptor: Any = None) -> Any:
    d = (z * scale).astype(np.float32)
    return lambda: (d, K, descriptor, None if up is None else np.array(up))


def frame(name: str, T: np.ndarray, *, p: dict[str, Any] | str | None = "metric",
          depth: bool = True, mono: float | None = None, height: float | None = None,
          ignore: np.ndarray | None = None, descriptor: Any = None, up: Any = (0.0, -1.0, 0.0),
          rgb: np.ndarray | None = None) -> FrameIn:
    img, z = shot(T)
    pr = prior(T) if p == "metric" else (prior(T, "drifting") if p == "drifting" else p)
    assert not isinstance(pr, str)
    return FrameIn(name, "cam", img if rgb is None else rgb, K, pr, z if depth else None,
                   ignore, height, None, 1000,
                   None if mono is None else mono_of(z, mono, up, descriptor))


def sweep(step: float, n: int, start: float = 0.0, **kw: Any) -> list[FrameIn]:
    return [frame(f"s{k}", base_pose(start + k * step), **kw) for k in range(n)]


def err(T: np.ndarray, G: np.ndarray) -> tuple[float, float]:
    return an.pose_residual(T, G)


# -- small pieces ----------------------------------------------------------------------------------


def test_geometry_helpers() -> None:
    R = an.exp_so3(np.array([0.1, -0.2, 0.3]))
    assert np.allclose(an.exp_so3(an.log_so3(R)), R)
    a = np.random.default_rng(0).normal(size=(10, 3))
    assert np.allclose(an.kabsch(a, a @ R.T), R)
    T = an.first_camera((0.0, -1.0, 0.0))
    assert np.allclose(T[:3, 2], [1.0, 0.0, 0.0]) and np.allclose(T[:3, 0], [0.0, -1.0, 0.0])
    assert np.allclose(an.first_camera(None), T)
    assert an.lowest_layer(np.zeros(10), 1.0) is None
    z = np.concatenate([np.full(500, -0.5), np.full(300, 0.4)])
    assert an.lowest_layer(z, 1.0) == pytest.approx(-0.5)


def test_features_and_matching() -> None:
    img, _ = shot(base_pose(0))
    f = an.features(img)
    assert len(f.kps) > 500 and f.desc.dtype == np.uint8
    ign = np.zeros(img.shape[:2], bool)
    ign[:, :160] = True
    assert (an.features(img, ign).kps[:, 0] >= 150).all()
    blank = np.full_like(img, 128)
    assert len(an.features(blank).kps) == 0
    ia, ib = an.match(f.desc, f.desc)
    assert len(ia) > 400 and (f.kps[ia] == f.kps[ib]).all()
    e = an.match(f.desc[:1], f.desc)
    assert len(e[0]) == 0


def test_sample_depth() -> None:
    z = np.full((24, 32), 2.0, np.float32)
    z[:, 16:] = 4.0  # an edge
    z[0, 0] = 0.0
    Kg = K.resized(32, 24)
    uv = np.array([[40.0, 120.0], [-50.0, 0.0], [160.0, 120.0], [300.0, 120.0], [5.0, 5.0]])
    d = an.sample_depth(z, Kg, K, uv)
    assert d[0] == 2.0 and d[3] == 4.0 and np.isnan(d[1]) and np.isnan(d[2]) and np.isnan(d[4])


def _refs(c: np.ndarray, sz: float = 0.03, med: float = 2.0) -> an.Refs:
    return an.Refs(np.asarray(c, float).reshape(1, 3), np.array([sz]), np.array([med]))


def test_refine_pose_and_pnp() -> None:
    rng = np.random.default_rng(1)
    T = base_pose(20)
    X = T[:3, 3] + (rng.uniform(-1, 1, (200, 3)) * [1, 1, 0.3] + [0, 0, 2]) @ T[:3, :3].T
    uv = an._project(T[:3, :3], T[:3, 3], X, K, False) + rng.normal(0, 0.3, (200, 2))
    got = an.pnp(X, uv, K, 2.0)
    assert got is not None
    d = an.unit(X - T[:3, 3])
    ref = np.zeros(200, np.intp)
    s, sc = an.refine_pose(got[0], got[1], uv, X, d, ref, _refs(T[:3, 3]), K)
    assert err(s.T, T)[0] < 0.01 and err(s.T, T)[1] < 0.2 and abs(sc[0] - 1) < 0.01
    sp, sr = s.sigmas()
    assert 0 < sp < 0.05 and 0 < sr < 0.5
    assert an.pnp(X[:5], uv[:5], K, 2.0) is None
    assert an.pnp(rng.normal(size=(50, 3)) + [0, 0, 5], rng.uniform(0, 300, (50, 2)), K,
                  0.5) is None
    # without depth: directions, the centre held
    nan = np.full_like(X, np.nan)
    r, _ = an.refine_pose(an.exp_so3(np.array([0.02, 0, 0])) @ T[:3, :3], T[:3, 3], uv, nan, d,
                          ref, _refs(T[:3, 3]), K, fix_centre=True)
    assert err(r.T, T)[1] < 0.2
    # a prior pulls within its sigmas; points scaled 10 % about their reference: the scale
    # parameter takes it, the rotation stays
    C = T[:3, 3] + np.array([0.3, 0.0, 0.0])
    Xs = C + 1.1 * (X - C)
    P = T.copy()
    P[:3, 3] += [0.005, 0.0, 0.0]
    f, sc = an.refine_pose(T[:3, :3], T[:3, 3], uv, Xs, d, ref, _refs(C, 0.3), K,
                           prior=(P, 0.01, 1.0))
    assert err(f.T, T)[1] < 0.2 and err(f.T, T)[0] < 0.02 and sc[0] < 1.0


def test_guided_matching() -> None:
    img, _ = shot(base_pose(0))
    f = an.features(img)
    pred = f.kps.astype(np.float64) + 2.0
    ia, ib = an.guided_match(f, f, pred, np.full(len(pred), 8.0))
    assert len(ia) > 0.8 * len(f.kps) and (ia == ib).mean() > 0.95
    nanp = np.full_like(pred, np.nan)
    assert len(an.guided_match(f, f, nanp, np.full(len(pred), 8.0))[0]) == 0
    one = an.Features(f.kps[:1], f.desc[:1])
    assert len(an.guided_match(one, f, pred, np.full(len(pred), 8.0))[0]) == 0


def test_rotation_ransac() -> None:
    rng = np.random.default_rng(2)
    R = an.exp_so3(np.array([0.0, 0.3, 0.1]))
    b = an.unit(rng.normal(size=(60, 3)))
    got = an.rotation_ransac(b @ R.T, b, 0.01)
    assert got is not None and an.rot_deg(got[0].T @ R) < 0.01
    assert an.rotation_ransac(b[:2], b[:2], 0.01) is None
    same = np.tile([[0.0, 0.0, 1.0]], (5, 1))
    assert an.rotation_ransac(same, same, 0.01) is None  # every pair is degenerate
    assert an.rotation_ransac(an.unit(rng.normal(size=(40, 3))), b[:40], 1e-4) is None


def test_cloud_roundtrip(tmp_path: Path) -> None:
    c = Cloud()
    c.add(np.zeros((0, 3)), np.zeros((0, 3)))
    assert len(c.keys) == 0
    c.add(np.array([[0.001, 0.0, 0.0], [0.003, 0.0, 0.0], [1.0, 1.0, 1.0]]), np.ones((3, 3)))
    c.add(np.array([[0.002, 0.001, 0.001]]), np.ones((1, 3)) * 4)
    assert len(c.keys) == 2 and sorted(c.count.tolist()) == [1, 3]
    c.save(tmp_path / "c.npz")
    d = Cloud.load(tmp_path / "c.npz")
    assert np.allclose(d.points(), c.points())
    assert len(Cloud.load(tmp_path / "none.npz").keys) == 0


def test_drift() -> None:
    d = Drift()
    P = base_pose(0, 1.0, 0.0)
    T0, sp, sr = d.predict(P, 0.02, 1.0)
    assert np.allclose(T0, P) and (sp, sr) == (0.02, 1.0)
    d.confirm(base_pose(0), P)
    T1, sp1, sr1 = d.predict(base_pose(90, 2.0, 0.0), 0.02, 1.0)
    assert np.allclose(T1[:3, 3], base_pose(90, 1.0, 0.0)[:3, 3]) and sp1 > 0.1 and sr1 > 5
    d.confirm(base_pose(0, 0.05), base_pose(0, 1.05))  # too short to tell the odometry's scale
    assert d.evidence == []
    d.confirm(base_pose(0, 1.0), base_pose(0, 2.13))
    assert d.odo_scale == pytest.approx(0.95 / 1.08, rel=1e-6)
    assert Drift.from_json(d.to_json()).odo_scale == d.odo_scale
    assert Drift.from_json(Drift().to_json()).ref_T is None


def test_scale_state() -> None:
    s = ScaleState()
    s.estimate(1.0)
    assert (s.k, s.source) == (1.0, "monocular")
    s.priors = [[0.5, 0.45], [1.0, 0.9], [0.8, 0.72], [0.05, 0.01]]
    s.estimate(0.2)  # the priors span too little yet
    assert s.source == "monocular"
    s.estimate(1.0)
    assert s.source == "priors" and s.k == pytest.approx(1 / 0.9, rel=1e-6)
    s.heights = [[0.04, 0.5], [0.04, 0.52, 0.9]]  # camera heights come first
    s.estimate(1.0)
    assert s.source == "camera_height" and s.k == pytest.approx(math.sqrt(0.5 * 0.52), rel=1e-3)
    assert ScaleState.from_json(s.to_json()).k == s.k


# -- the map ---------------------------------------------------------------------------------------


def new_map(tmp_path: Path) -> AnchoredMap:
    return AnchoredMap(tmp_path / "m")


def test_a_camera_only_map_turning_in_place(tmp_path: Path) -> None:
    """No priors: the first frame is the origin, level by its gravity; a sweep registers by
    the images alone and the camera height sets the scale."""
    m = new_map(tmp_path)
    T0 = base_pose(0, pitch_deg=0.0)
    first = m.update([frame("a", T0, p=None, depth=False, mono=1.3, height=0.5)])[0]
    assert first.status == "registered" and first.prior_residual is None
    assert m.created_with == "none" and np.allclose(m.keyframes[0].T[:3, 2], [1, 0, 0])
    to_map = m.keyframes[0].T @ np.linalg.inv(T0)
    pls = m.update([frame(f"s{k}", base_pose(30.0 * k, pitch_deg=0.0), p=None, depth=False,
                          mono=1.3, height=0.5) for k in range(1, 7)])
    assert all(p.status == "registered" for p in pls)
    for k, p in enumerate(pls, start=1):
        assert err(p.T, to_map @ base_pose(30.0 * k, pitch_deg=0.0))[1] < 1.0
    assert m.scale.source == "camera_height" and m.scale.k == pytest.approx(1 / 1.3, rel=0.05)


def test_cold_start_sweep_with_depth(tmp_path: Path) -> None:
    m = new_map(tmp_path)
    first = m.update([frame("a0", base_pose(0))])
    assert first[0].status == "registered" and first[0].method == "origin"
    assert m.created_with == "metric" and m.up_check_deg is None  # depth image: no gravity asked
    pls = m.update(sweep(30.0, 11))
    assert [p.status for p in pls] == ["registered"] * 11
    for k, p in enumerate(pls):
        assert p.T is not None
        e = err(p.T, base_pose(30.0 * k))
        assert e[0] < 0.02 and e[1] < 0.5, (k, e)
    assert m.scale_summary() == ("depth", an.DEPTH_SIGMA_REL)
    floor, pts = m.floor_and_bounds()
    assert floor is not None and abs(floor) < 0.03 and len(pts) > 1000
    m.save()
    again = AnchoredMap.load(m.root)
    assert len(again.keyframes) == 12 and again.keyframes[0].feats is None
    more = again.update([frame("b", base_pose(45))])
    assert more[0].status == "registered" and err(more[0].T, base_pose(45))[1] < 0.5


def test_sweep_with_monocular_depth_and_camera_height(tmp_path: Path) -> None:
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0), depth=False, mono=1.2, height=0.5)])
    assert m.up_check_deg is not None and m.up_check_deg < 15
    pls = m.update(sweep(30.0, 6, depth=False, mono=1.2, height=0.5))
    assert all(p.status == "registered" for p in pls)
    src, sig = m.scale_summary()
    assert src == "camera_height" and sig < 0.05
    assert m.scale.k == pytest.approx(1 / 1.2, rel=0.05)
    floor, _ = m.floor_and_bounds()
    assert floor is not None and abs(floor) < 0.05  # the cloud fused at the corrected scale


def test_metric_priors_fix_the_monocular_depth_scale(tmp_path: Path) -> None:
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0, -1.0, 0.0), depth=False, mono=1.3)])
    walk = [frame(f"w{k}", base_pose(0, -1.0 + 0.15 * k, 0.1 * (k % 2)), depth=False, mono=1.3)
            for k in range(1, 8)]
    pls = m.update(walk)
    assert all(p.status == "registered" for p in pls)
    assert m.prior_span() > an.PRIOR_SPAN_M and m.scale.source == "priors"
    assert m.scale.k == pytest.approx(1 / 1.3, rel=0.08)
    assert m.scale_summary()[0] == "priors"
    # the stored monocular keyframes' depth follows the new scale when they are matched
    k0 = m.keyframes[0]
    assert m._depth_factor(k0) == pytest.approx(m.scale.k / k0.k_used)


def test_provisional_and_conflicting_frames(tmp_path: Path) -> None:
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0)), frame("a1", base_pose(30))])
    blank = np.full((K.height, K.width, 3), 120, np.uint8)
    # no features at all: kept at its prior
    p0 = m.update([frame("blank", base_pose(10), rgb=blank)])[0]
    assert (p0.status, p0.reason, p0.method) == ("provisional", "few_features", "prior")
    assert p0.prior_residual is not None and p0.prior_residual[1] < 1e-3
    # its prior points away from anything mapped: kept at its prior
    p1 = m.update([frame("away", base_pose(180))])[0]
    assert (p1.status, p1.reason) == ("provisional", "no_overlap")
    # well supported by the images, 0.5 m from its prior: a conflict
    T = base_pose(20)
    wrong = T.copy()
    wrong[:3, 3] += [0.5, 0.0, 0.0]
    p2 = m.update([frame("off", T, p=prior(wrong))])[0]
    assert (p2.status, p2.reason) == ("rejected", "prior_conflict")
    assert p2.prior_residual is not None and p2.prior_residual[0] > 0.3
    # few matches, beyond its prior: not confirmed, kept at its prior
    an.STRONG_INLIERS, strong = 10 ** 6, an.STRONG_INLIERS
    try:
        p3 = m.update([frame("weak", base_pose(21), p=prior(wrong))])[0]
    finally:
        an.STRONG_INLIERS = strong
    assert p3.status == "provisional" and p3.inliers > 0
    # provisional frames at metric priors are references; nothing stored moved
    assert [k.state for k in m.keyframes] == ["fused", "fused", "provisional", "provisional",
                                              "provisional"]
    assert np.allclose(m.keyframes[0].T, base_pose(0))
    # a metric prior is called a conflict only on two references: one keyframe sees it
    m2 = new_map(tmp_path / "one")
    m2.update([frame("a0", base_pose(0))])
    assert m2.update([frame("off", T, p=prior(wrong))])[0].status == "provisional"


def test_frames_without_prior_in_an_anchored_map(tmp_path: Path) -> None:
    m = new_map(tmp_path)
    desc = np.ones(8, np.float32)
    m.update([frame("a0", base_pose(0), descriptor=desc, depth=False, mono=1.0),
              frame("a1", base_pose(30), descriptor=desc * 2, depth=False, mono=1.0)])
    p = m.update([frame("n", base_pose(15), p=None, depth=False, mono=1.0, descriptor=desc)])[0]
    assert p.status == "registered" and p.method == "pnp" and err(p.T, base_pose(15))[1] < 1.0
    away = m.update([frame("x", base_pose(180), p=None, depth=False, mono=1.0,
                           descriptor=desc)])[0]
    assert (away.status, away.reason) == ("rejected", "no_overlap")
    # without a descriptor anywhere: the most recent keyframes
    m2 = new_map(tmp_path / "2")
    m2.update([frame("a0", base_pose(0))])
    q = m2.update([frame("n", base_pose(10), p=None)])[0]
    assert q.status == "registered"
    # a descriptor while no keyframe has one: the most recent keyframes
    q2 = m2.update([frame("n2", base_pose(12), p=None, depth=False, mono=1.0,
                          descriptor=desc)])[0]
    assert q2.status == "registered"


def test_rotation_only_without_depth(tmp_path: Path) -> None:
    m = new_map(tmp_path)
    m.update([frame("a0", look(0, (0.0, 0.0, 0.5)), depth=False)])  # no depth at all
    assert m.keyframes[0].depth_kind == "none" and len(m.cloud.keys) == 0
    p = m.update([frame("r", look(25, (0.0, 0.0, 0.5)), depth=False)])[0]
    assert p.status == "registered" and p.method == "rotation"
    assert err(p.T, look(25, (0.0, 0.0, 0.5)))[1] < 0.3
    # too far from any keyframe for rotation alone: kept at its prior
    far = m.update([frame("f", look(25, (0.6, 0.0, 0.5)), depth=False)])[0]
    assert far.status == "provisional"
    # without a prior: rotation at the best-matched keyframe's centre
    q = m.update([frame("q", look(-20, (0.0, 0.0, 0.5)), p=None, depth=False)])[0]
    assert q.status == "registered" and q.method == "rotation"
    assert q.sigma_pos_m == an.UNANCHORED_SIGMA_POS_M


def test_rotation_failing_is_unconfirmed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch
                                         ) -> None:
    m = new_map(tmp_path)
    m.update([frame("a0", look(0, (0.0, 0.0, 0.5)), depth=False)])
    monkeypatch.setattr(an, "rotation_ransac", lambda *a, **k: None)
    assert m.update([frame("r", look(25, (0.0, 0.0, 0.5)), depth=False)])[0].status \
        == "provisional"
    assert m.update([frame("q", look(5, (0.0, 0.0, 0.5)), p=None, depth=False)])[0].status \
        == "rejected"
    monkeypatch.setattr(an, "rotation_ransac", lambda d, b, t, **k: (
        np.eye(3), np.arange(len(d)) < 5))
    assert m.update([frame("r2", look(25, (0.0, 0.0, 0.5)), depth=False)])[0].status \
        == "provisional"


def test_init_edge_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0))])
    f = frame("b", base_pose(20))
    feats = an.features(f.rgb)
    cands = m._candidates(f, base_pose(20), None)
    corr = m._correspondences(feats, cands)
    monkeypatch.setattr(an, "pnp", lambda *a, **k: None)
    got = m._init(f, corr, cands, base_pose(20))
    assert got is not None and got[3]  # rotation-only at the prior's centre
    monkeypatch.setattr(an, "pnp", lambda X, uv, K_, t, **k: (
        np.eye(3), np.zeros(3), np.arange(len(X)) < 5))
    monkeypatch.setattr(an, "rotation_ransac", lambda *a, **k: None)
    assert m._init(f, corr, cands, base_pose(20)) is None
    far = base_pose(20, 3.0, 0.0)
    assert m._init(f, corr, cands, far) is None
    T = base_pose(20)
    monkeypatch.setattr(an, "pnp", lambda X, uv, K_, t, **k: (
        T[:3, :3], T[:3, 3], np.ones(len(X), bool)))
    monkeypatch.setattr(an, "refine_pose", lambda R, c, uv, *a, **k: (an.Solve(
        R, c, np.eye(6), np.zeros(len(uv), bool), 1.0), np.ones(1)))
    p = m.update([frame("c", base_pose(25))])[0]
    assert p.status == "provisional"


def test_drifting_walk_with_odometry_overcount(tmp_path: Path) -> None:
    m = new_map(tmp_path)

    def odo(T: np.ndarray) -> np.ndarray:  # 8 % over-counted travel from the start
        P = T.copy()
        P[:2, 3] = (T[:2, 3] - [-1.0, 0.0]) * 1.08 + [-1.0, 0.0]
        return P

    truth = [base_pose(0, -1.0 + 0.2 * k, 0.0) for k in range(9)]
    m.update([frame("w0", truth[0], p=prior(odo(truth[0]), "drifting"))])
    assert m.created_with == "drifting"
    pls = m.update([frame(f"w{k}", truth[k], p=prior(odo(truth[k]), "drifting"))
                    for k in range(1, 9)])
    assert all(p.status == "registered" for p in pls)
    assert err(pls[-1].T, truth[-1])[0] < 0.03  # the depth images fix the over-count
    assert m.drift.odo_scale == pytest.approx(1 / 1.08, rel=0.03)
    assert pls[-1].prior_residual is not None and pls[-1].prior_residual[0] < 0.05
    # an unconfirmed drifting frame goes where the corrected odometry puts it
    blank = np.full((K.height, K.width, 3), 120, np.uint8)
    T = base_pose(0, 0.9, 0.0)
    p = m.update([frame("b", T, p=prior(odo(T), "drifting"), rgb=blank)])[0]
    assert p.status == "provisional" and err(p.T, T)[0] < 0.03
    assert m.keyframes[-1].usable() is False  # a provisional drifting frame is no reference


def test_scale_evidence_needs_monocular_reference(tmp_path: Path) -> None:
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0, -1.0, 0.0))])
    m.update([frame("w1", base_pose(0, -0.8, 0.0))])
    assert m.scale.priors == []  # references with depth images say nothing of k


def test_height_evidence_edges(tmp_path: Path) -> None:
    m = new_map(tmp_path)
    T = base_pose(0)
    _, z = shot(T)
    f = frame("a", T, depth=False, mono=1.0, height=0.0)
    m.add_height_evidence(f, T, z, K)
    f.height_m = 0.5
    m.add_height_evidence(f, T, np.zeros_like(z), K)
    up = look(0, (0.0, 0.0, 0.5), 80.0)  # looking at the ceiling: no floor below
    m.add_height_evidence(f, up, shot(up)[1], K)
    assert m.scale.heights == []
    m.add_height_evidence(f, T, z, K)
    assert m.scale.heights[0][0] == pytest.approx(an.DEFAULT_HEIGHT_SIGMA_M / 0.5)


def test_fusion_ignores_masked_pixels(tmp_path: Path) -> None:
    m = new_map(tmp_path)
    ign = np.ones((K.height, K.width), bool)
    m.update([frame("a0", base_pose(0), ignore=ign)])
    assert len(m.cloud.keys) == 0
    ign[:, :160] = False
    m.update([frame("a1", base_pose(10), ignore=ign)])
    n = len(m.cloud.keys)
    assert n > 0
    m.update([frame("a2", base_pose(10, 0.01), p=prior(base_pose(10, 0.01)))])
    assert len(m.cloud.keys) > n


def test_up_check_without_gravity(tmp_path: Path) -> None:
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0), depth=False, mono=1.0, up=None)])
    assert m.up_check_deg is None
    floor, pts = AnchoredMap(tmp_path / "e").floor_and_bounds()
    assert floor is None and len(pts) == 0


def test_summary_and_candidates(tmp_path: Path) -> None:
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0), depth=False, mono=1.0)])
    assert m.scale_summary() == ("monocular", an.MONOCULAR_SIGMA_REL)
    assert m.prior_span() == 0.0
    m.update([frame("a1", base_pose(0, 0.5, 0.0), depth=False, mono=1.0)])
    src, sigma = m.scale_summary()
    assert src == "priors" and sigma == pytest.approx(math.sqrt(2) * 0.02 / 0.5)
    # a candidate far away or facing elsewhere is not one
    assert m._candidates(frame("x", base_pose(180)), base_pose(180, 10.0, 0.0), None) == []


def test_kf_without_features(tmp_path: Path) -> None:
    m = new_map(tmp_path)
    blank = np.full((K.height, K.width, 3), 120, np.uint8)
    m.update([frame("a0", base_pose(0), rgb=blank)])
    assert len(m.keyframes[0].kdepth) == 0
    m.save()
    m2 = AnchoredMap.load(m.root)
    p = m2.update([frame("a1", base_pose(10))])[0]
    assert p.status == "provisional"


def test_rejection_of_unmatched_prior_free_frames_with_few_features(tmp_path: Path) -> None:
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0))])
    blank = np.full((K.height, K.width, 3), 120, np.uint8)
    p = m.update([frame("x", base_pose(0), p=None, rgb=blank)])[0]
    assert (p.status, p.reason) == ("rejected", "few_features")


def test_progress_and_timings(tmp_path: Path) -> None:
    m = new_map(tmp_path)
    seen: list[str] = []
    m.update([frame("a0", base_pose(0))], seen.append)
    assert seen and set(m.timings) == {"register", "fuse", "total"}


def test_cv2_rng_seeded() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 3)) + [0, 0, 4]
    T = np.eye(4)
    uv = an._project(T[:3, :3], T[:3, 3], X, K, False)
    a = an.pnp(X, uv, K, 1.0, seed=7)
    b = an.pnp(X, uv, K, 1.0, seed=7)
    assert a is not None and b is not None and np.allclose(a[0], b[0])
    assert cv2.__version__


def test_refine_pose_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(4)
    T = base_pose(0)
    X = T[:3, 3] + (rng.uniform(-1, 1, (50, 3)) + [0, 0, 3]) @ T[:3, :3].T
    uv = an._project(T[:3, :3], T[:3, 3], X, K, False)
    d = an.unit(X - T[:3, 3])
    ref = np.zeros(50, np.intp)
    monkeypatch.setattr(an, "GN_ITERATIONS", 1)
    s, _ = an.refine_pose(T[:3, :3], T[:3, 3] + 0.05, uv, X, d, ref, _refs(T[:3, 3]), K)
    assert err(s.T, T)[0] < 0.05
    monkeypatch.undo()
    s, _ = an.refine_pose(T[:3, :3], T[:3, 3], rng.uniform(0, 300, (50, 2)), X, d, ref,
                          _refs(T[:3, 3]), K)
    assert s.inliers.sum() < 10


def test_an_ill_determined_image_pose_confirms_nothing(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0))])
    monkeypatch.setattr(an.Solve, "sigmas", lambda self: (0.5, 0.3))
    p = m.update([frame("b", base_pose(20))])[0]
    assert p.status == "provisional" and p.inliers > 0


def test_suspicious_frames_too_close_by_their_monocular_depth(tmp_path: Path) -> None:
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0))])
    near = frame("near", base_pose(10), depth=False, mono=0.05)
    near.check_close = True
    ok = frame("ok", base_pose(20), depth=False, mono=1.0)
    ok.check_close = True
    pls = m.update([near, ok])
    assert (pls[0].status, pls[0].reason) == ("rejected", "too_close")
    assert pls[1].status == "registered"


def test_drifting_priors_scale_monocular_depth_without_heights(tmp_path: Path) -> None:
    m = new_map(tmp_path)
    truth = [base_pose(0, -1.0 + 0.15 * k, 0.05 * (k % 2)) for k in range(8)]
    m.update([frame("w0", truth[0], p="drifting", depth=False, mono=1.4)])
    pls = m.update([frame(f"w{k}", truth[k], p="drifting", depth=False, mono=1.4)
                    for k in range(1, 8)])
    assert all(p.status == "registered" for p in pls)
    assert m.scale.source == "priors" and m.scale.sigma_rel >= an.DRIFT_POS_REL
    assert m.scale.k == pytest.approx(1 / 1.4, rel=0.1)
    s = ScaleState(odometry=[[0.5, 0.4]] * 3)
    s.estimate(0.0, 0.1)  # the odometry spans too little
    assert s.source == "monocular"


def test_odometry_evidence_needs_a_drifting_reference(tmp_path: Path) -> None:
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0, -1.0, 0.0), depth=False, mono=1.0)])
    p = m.update([frame("d", base_pose(0, -0.8, 0.0), p="drifting", depth=False, mono=1.0)])[0]
    assert p.status == "registered" and m.scale.odometry == [] and m.scale.priors == []


def test_depth_alignment() -> None:
    T = base_pose(0)
    _, z = shot(T)
    uv = np.stack(np.meshgrid(np.arange(20, 300, 20.0), np.arange(20, 220, 20.0)), -1).reshape(-1, 2)
    X = T[:3, 3] + (an.rays(K, uv) * an.sample_depth(z, K, K, uv)[:, None]) @ T[:3, :3].T
    ok = np.isfinite(X).all(axis=1)
    assert an.depth_alignment(T, X[ok], uv[ok], z * 0.8, K, K) == pytest.approx(1.25, rel=1e-3)
    assert an.depth_alignment(T, X[ok][:5], uv[ok][:5], z, K, K) is None
    assert an.depth_alignment(T, X[ok], uv[ok], np.zeros_like(z), K, K) is None
    assert an.depth_alignment(T, X[ok], uv[ok], z * 0.2, K, K) is None


def test_a_disagreeing_monocular_depth_is_kept_as_it_is(tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0), depth=False, mono=1.0)])
    monkeypatch.setattr(an, "depth_alignment", lambda *a: None)
    assert m.update([frame("b", base_pose(20), depth=False, mono=1.0)])[0].status == "registered"


def test_metric_priors_with_a_scale_bias(tmp_path: Path) -> None:
    """Metric priors over-counting 10 % against depth images: the bias is measured and the
    later predictions corrected (and a sensor map reports depth, not the biased priors)."""
    m = new_map(tmp_path)
    o = np.array([-1.5, 0.0])

    def biased(T: np.ndarray) -> dict[str, Any]:
        P = T.copy()
        P[:2, 3] = o + (T[:2, 3] - o) * 1.10
        return prior(P, sp=0.05, sr=1.0)

    truth = [base_pose(5 * (k % 3), -1.5 + 0.2 * k, 0.05 * (k % 2)) for k in range(12)]
    m.update([frame("w0", truth[0], p=biased(truth[0]))])
    pls = m.update([frame(f"w{k}", truth[k], p=biased(truth[k])) for k in range(1, 12)])
    assert all(p.status == "registered" for p in pls)
    assert m.prior_bias() == pytest.approx(1 / 1.10, rel=0.03)
    assert m.scale_summary()[0] == "depth"
    assert err(pls[-1].T, truth[-1])[0] < 0.08


def test_floor_candidates_and_the_vote() -> None:
    """Each frame offers its horizontal planes; the frames vote for the floor: a counter seen
    by one frame, or a lower floor seen through a door, does not win."""
    rng = np.random.default_rng(5)

    def plane(h: float, n: int) -> np.ndarray:
        return np.column_stack([rng.uniform(-2, 2, n), rng.uniform(0.5, 4, n),
                                np.full(n, -h) + rng.normal(0, 0.002, n)])

    wall = np.column_stack([rng.uniform(-2, 2, 3000), np.full(3000, 3.0),
                            rng.uniform(-0.6, 1.0, 3000)])
    pts = np.vstack([plane(0.5, 4000), plane(0.2, 1500), wall])
    c = an.floor_candidates(pts)
    assert c[0] == pytest.approx(0.5, abs=0.01) and any(abs(h - 0.2) < 0.01 for h in c)
    assert an.floor_candidates(wall[:100]) == []
    assert an._horizontal_plane(plane(0.5, 50), 0.5) is None
    assert an.floor_candidates(np.column_stack([wall[:, :2], -np.abs(wall[:, 2]) - 0.1])) == []
    s = ScaleState(heights=[[0.04, 1.0, 2.5], [0.04, 1.02], [0.04, 0.98, 0.4], [0.04, 3.0]])
    s.estimate(0.0)
    assert s.source == "camera_height" and s.k == pytest.approx(1.0, abs=0.03)


def test_a_metric_prior_dominates_the_image_pose(tmp_path: Path) -> None:
    """A metric prior 3 cm / 1.5 deg off truth with sigma 2 cm / 1 deg: the frame stays within
    its prior's reach (the images refine it, they do not replace it)."""
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0))])
    T = base_pose(20)
    P = T.copy()
    P[:3, 3] += [0.03, 0.0, 0.0]
    p = m.update([frame("b", T, p=prior(P))])[0]
    assert p.status == "registered" and p.prior_residual is not None
    assert 0.001 < p.prior_residual[0] < 0.02


def test_keyframe_images_and_depth_are_kept(tmp_path: Path) -> None:
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0), depth=False, mono=1.0)])
    m.save()
    again = AnchoredMap.load(m.root)
    kid = again.keyframes[0].id
    p = again.image_path(kid)
    assert p is not None and cv2.imread(str(p)).shape == (K.height, K.width, 3)
    d = again.depth_of(kid)
    assert d is not None and d.shape == (K.height, K.width) and again.keyframes[0].K_grid == K
    again.scale.k *= 2.0  # a monocular keyframe's depth follows the map's depth scale
    assert np.allclose(again.depth_of(kid), 2.0 * d, rtol=1e-3)
    assert again.depth_of("nope") is None and again.image_path("nope") is None
    (again.root / an.KF_DIR / f"{kid}.depth.npy").unlink()
    assert again.depth_of(kid) is None


def test_the_odometry_drifts_with_its_path() -> None:
    d = Drift()
    d.confirm(base_pose(0), base_pose(0))
    away, back = base_pose(0, 1.0, 0.0), base_pose(5, 0.02, 0.0)
    straight = d.predict(back, 0.02, 1.0)
    d.observe(away)
    T, sp, sr = d.predict(back, 0.02, 1.0)
    assert np.allclose(T, straight[0]) and sp > straight[1] + 0.15 and sr > straight[2]
    d.observe(back)
    assert d.path_m == pytest.approx(2.0, abs=0.05)
    again = Drift.from_json(d.to_json())
    assert again.path_m == d.path_m and np.allclose(again.last_prior, back)  # type: ignore[arg-type]
    # a state written before the path was kept: the straight line
    old = {k: v for k, v in d.to_json().items() if k not in ("last_prior", "path_m", "turn_deg")}
    assert Drift.from_json(old).predict(back, 0.02, 1.0)[1] == pytest.approx(straight[1])
    d.confirm(back, back)
    assert (d.path_m, d.turn_deg) == (0.0, 0.0)


def _slip(T: np.ndarray, deg: float) -> np.ndarray:
    a = math.radians(deg)
    Rz = np.eye(4)
    Rz[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    return Rz @ T


def test_a_slipped_odometry_is_reanchored_on_two_agreeing_frames(tmp_path: Path) -> None:
    m = new_map(tmp_path)
    m.update([frame(f"a{k}", base_pose(30 * k), p="drifting") for k in range(4)])
    truth = [base_pose(y) for y in (80, 65, 50)]
    pls = m.update([frame(f"s{k}", T, p=prior(_slip(T, 25.0), "drifting"))
                    for k, T in enumerate(truth)])
    assert [p.status for p in pls] == ["registered"] * 3
    for p, T in zip(pls, truth, strict=True):
        assert err(p.T, T)[1] < 1.0  # type: ignore[arg-type]
    # a single strong conflict stays one; placements without an image pose never agree
    m2 = new_map(tmp_path / "one")
    m2.update([frame(f"a{k}", base_pose(30 * k), p="drifting") for k in range(4)])
    p = m2.update([frame("s", truth[0], p=prior(_slip(truth[0], 25.0), "drifting"))])[0]
    assert (p.status, p.reason) == ("rejected", "prior_conflict") and p.align_T is not None
    f = frame("s", truth[0], p=prior(_slip(truth[0], 25.0), "drifting"))
    assert not m2._same_drift(f, an.Placement("x", "rejected"), f, p)


def test_an_unconfirmed_frame_against_the_map_s_depth(tmp_path: Path) -> None:
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0)), frame("a1", base_pose(30))])
    m.save()
    m = AnchoredMap.load(tmp_path / "m")  # the keyframes' depth from disk
    blank = np.full((K.height, K.width, 3), 120, np.uint8)
    T = base_pose(10)
    far = T.copy()
    far[:3, 3] += [0.8, 0.0, 0.0]
    p = m.update([frame("far", T, p=prior(far), rgb=blank)])[0]
    assert (p.status, p.reason) == ("rejected", "prior_conflict")
    assert p.map_residual is not None and p.map_residual[0] > 0.5
    # contradicted within its sigmas: kept, its depth residual reported
    near = T.copy()
    near[:3, 3] += [0.45, 0.0, 0.0]
    p = m.update([frame("near", T, p=prior(near, sp=0.4), rgb=blank)])[0]
    assert p.status == "provisional" and p.map_residual is not None
    assert 0.2 < p.map_residual[0] < 0.6
    # where it agrees with the map: no residual
    p = m.update([frame("same", T, p=prior(T), rgb=blank)])[0]
    assert p.status == "provisional" and p.map_residual is None
    # a keyframe without a depth grid is no reference
    for k in m.keyframes:
        k.grid, k.K_grid = None, None
    p = m.update([frame("none", T, p=prior(far), rgb=blank)])[0]
    assert p.status == "provisional" and p.map_residual is None


def test_guided_matching_without_candidates_in_reach() -> None:
    img, _ = shot(base_pose(0))
    f = an.features(img)
    pred = np.full((len(f.kps), 2), -1000.0)
    assert len(an.guided_match(f, f, pred, np.full(len(pred), 8.0))[0]) == 0
