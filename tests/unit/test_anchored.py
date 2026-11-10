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
    A, B = an.pose(np.eye(3), np.zeros(3)), an.pose(R, np.ones(3))
    T, sp, sr = an.fuse_poses(A, 1.0, 1.0, B, 1.0, 1.0)
    assert np.allclose(T[:3, 3], 0.5) and sp == pytest.approx(math.sqrt(0.5))
    assert an.rot_deg(A[:3, :3].T @ T[:3, :3]) == pytest.approx(an.rot_deg(R) / 2, rel=1e-6)
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


def test_refine_and_pnp() -> None:
    rng = np.random.default_rng(1)
    T = base_pose(20)
    X = T[:3, 3] + (rng.uniform(-1, 1, (200, 3)) * [1, 1, 0.3] + [0, 0, 2]) @ T[:3, :3].T
    uv = an._project(T[:3, :3], T[:3, 3], X, K, False) + rng.normal(0, 0.3, (200, 2))
    got = an.pnp(X, uv, K, 2.0)
    assert got is not None
    s = an.refine(got[0], got[1], X, uv, K)
    assert err(s.T, T)[0] < 0.01 and err(s.T, T)[1] < 0.2
    sp, sr = s.sigmas()
    assert 0 < sp < 0.05 and 0 < sr < 0.5
    assert an.pnp(X[:5], uv[:5], K, 2.0) is None
    assert an.pnp(rng.normal(size=(50, 3)) + [0, 0, 5], rng.uniform(0, 300, (50, 2)), K,
                  0.5) is None
    d = an.unit(X - T[:3, 3])
    r = an.refine(an.exp_so3(np.array([0.02, 0, 0])) @ T[:3, :3], T[:3, 3], d, uv, K,
                  directions=True, fix_centre=True)
    assert err(r.T, T)[1] < 0.2


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
    s.heights = [[0.5, 0.04], [0.52, 0.04]]
    s.estimate(0.0)
    assert s.source == "camera_height" and s.k == pytest.approx(math.sqrt(0.5 * 0.52), rel=1e-3)
    s.priors = [[0.5, 0.45], [1.0, 0.9], [0.8, 0.72], [0.05, 0.01]]
    s.estimate(0.2)  # the priors span too little yet
    assert s.source == "camera_height"
    s.estimate(1.0)
    assert s.source == "priors" and s.k == pytest.approx(1 / 0.9, rel=1e-6)
    assert ScaleState.from_json(s.to_json()).k == s.k


# -- the map ---------------------------------------------------------------------------------------


def new_map(tmp_path: Path) -> AnchoredMap:
    return AnchoredMap(tmp_path / "m")


def test_new_map_needs_a_prior(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        new_map(tmp_path).update([frame("a", base_pose(0), p=None)])


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
    m.update([frame("a0", base_pose(0))])
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
    crop = shot(T)[0].copy()
    crop[:, : K.width - 31] = 120
    p3 = m.update([frame("weak", T, p=prior(wrong), rgb=crop)])[0]
    assert p3.status == "provisional" and p3.inliers < an.STRONG_INLIERS
    # provisional frames at metric priors are references; nothing stored moved
    assert [k.state for k in m.keyframes] == ["fused", "provisional", "provisional",
                                              "provisional"]
    assert np.allclose(m.keyframes[0].T, base_pose(0))


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


def test_pnp_edge_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0))])
    f = frame("b", base_pose(20))
    feats = an.features(f.rgb)
    cands = m._candidates(f, base_pose(20), None)
    corr = m._correspondences(feats, cands)
    monkeypatch.setattr(an, "pnp", lambda *a, **k: None)
    assert m._solve_pnp(f, corr) is None
    monkeypatch.setattr(an, "pnp", lambda X, uv, K_, t, **k: (
        np.eye(3), np.zeros(3), np.arange(len(X)) < 5))
    assert m._solve_pnp(f, corr) is None
    T = base_pose(20)
    monkeypatch.setattr(an, "pnp", lambda X, uv, K_, t, **k: (
        T[:3, :3], T[:3, 3], np.ones(len(X), bool)))
    monkeypatch.setattr(an, "refine", lambda *a, **k: an.Solve(
        T[:3, :3], T[:3, 3], np.eye(6), np.zeros(len(a[2]), bool), 1.0))
    assert m._solve_pnp(f, corr) is None
    empty = an._Corr(np.zeros((0, 2)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, np.intp))
    assert m._solve_pnp(f, empty) is None


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
    m._height_evidence(f, T, z, K)
    f.height_m = 0.5
    m._height_evidence(f, T, np.zeros_like(z), K)
    up = look(0, (0.0, 0.0, 0.5), 80.0)  # looking at the ceiling: no floor below
    m._height_evidence(f, up, shot(up)[1], K)
    assert m.scale.heights == []
    m._height_evidence(f, T, z, K)
    assert m.scale.heights[0][1] == pytest.approx(an.DEFAULT_HEIGHT_SIGMA_M / 0.5)


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


def test_refine_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(4)
    T = base_pose(0)
    X = T[:3, 3] + (rng.uniform(-1, 1, (50, 3)) + [0, 0, 3]) @ T[:3, :3].T
    uv = an._project(T[:3, :3], T[:3, 3], X, K, False)
    monkeypatch.setattr(an, "GN_ITERATIONS", 1)
    assert err(an.refine(T[:3, :3], T[:3, 3] + 0.05, X, uv, K).T, T)[0] < 0.05
    monkeypatch.undo()
    s = an.refine(T[:3, :3], T[:3, 3], X, rng.uniform(0, 300, (50, 2)), K)
    assert s.inliers.sum() < 3 and s.rms_px == float("inf")


def test_an_ill_determined_image_pose_confirms_nothing(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0))])
    monkeypatch.setattr(AnchoredMap, "_pnp_sigmas", lambda self, s, c: (0.5, 0.3))
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
