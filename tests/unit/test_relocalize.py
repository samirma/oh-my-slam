"""Locate's core (``mapping.relocalize``) on synthetic keyframes: points with random descriptors seen
from known poses, a symmetric twin for ambiguity, keyframes from one spot for the turn fit."""

from __future__ import annotations

import math

import numpy as np
import pytest

from oh_my_slam.mapping import relocalize as rl
from oh_my_slam.mapping import sigmas
from tests.synth import rooms

K = np.array([[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]])


def _rz(deg: float, about: np.ndarray, dz: float = 0.0) -> np.ndarray:
    a = math.radians(deg)
    S = np.eye(4)
    S[:3, :3] = [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
    S[:3, 3] = about - S[:3, :3] @ about + [0, 0, dz]
    return S


def _apply(S: np.ndarray, P: np.ndarray) -> np.ndarray:
    return P @ S[:3, :3].T + S[:3, 3]


def _scene(seed: int = 0, n: int = 600) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    P = np.column_stack([rng.uniform(1.5, 4.0, n), rng.uniform(-1.5, 1.5, n),
                         rng.uniform(0.0, 2.0, n)])
    D = rng.integers(0, 256, (n, 128)).astype(np.uint8)
    return P, D


def _view(T: np.ndarray, P: np.ndarray, D: np.ndarray, noise: float = 0.0, seed: int = 1
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    uv, z = rl.project(T, K, P)
    ok = (z > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < 320) & (uv[:, 1] >= 0) & (uv[:, 1] < 240)
    uv = uv + np.random.default_rng(seed).normal(0, noise, uv.shape)
    return uv[ok].astype(np.float32), D[ok], np.flatnonzero(ok)


def _kf(name: str, T: np.ndarray, P: np.ndarray, D: np.ndarray, depth: bool = True,
        sigma: float = 0.01) -> rl.Keyframe:
    uv, d, idx = _view(T, P, D)
    xyz = P[idx].astype(np.float32) if depth else np.full((len(idx), 3), np.nan, np.float32)
    return rl.Keyframe(name, T, K, uv, d, xyz, sigma, 0.1, depth)


def _query(T: np.ndarray, P: np.ndarray, D: np.ndarray, noise: float = 0.5, **kw: object
           ) -> rl.Query:
    uv, d, _ = _view(T, P, D, noise)
    return rl.Query(uv, d, K, 320, 240, **kw)  # type: ignore[arg-type]


A = rooms.camera(0.0, 0.0, 1.0, 0.0, -10.0)


def test_a_frame_near_a_keyframe_is_located_with_sigmas_from_its_fit(
        monkeypatch: pytest.MonkeyPatch) -> None:
    P, D = _scene()
    idx = rl.Index([_kf("a", A, P, D), _kf("b", rooms.camera(0.3, 0.2, 1.0, -8.0, -10.0), P, D)])
    T = rooms.camera(0.15, 0.1, 1.05, -4.0, -12.0)
    o = rl.locate(_query(T, P, D), idx)
    assert o.located and o.method == "points" and o.ambiguity is None
    assert o.T is not None
    dp, dr = rl.pose_distance(o.T, T)
    assert dp < 0.01 and dr < 0.2
    # the depth keyframes' calibrated floor (mapping.sigmas): 12 mm with the keyframes' 1 cm
    assert o.sigma_pos_m is not None and 0.012 <= o.sigma_pos_m < 0.03
    noisy = rl.locate(_query(T, P, D, noise=2.0), idx)
    assert noisy.sigma_pos_m is not None and noisy.sigma_pos_m >= o.sigma_pos_m
    # without the noise floors the fit's own covariance shows: noisier, looser
    monkeypatch.setenv("OH_MY_SLAM_SIGMA_FLOORS", '{"depth/none": [0, 0]}')
    a = rl.locate(_query(T, P, D), idx)
    b = rl.locate(_query(T, P, D, noise=2.0), idx)
    assert a.sigma_pos_m is not None and b.sigma_pos_m is not None
    assert b.sigma_pos_m > a.sigma_pos_m
    assert o.inlier_ratio is not None and o.inlier_ratio > 0.9
    assert o.consistent_with_prior is None and o.nearest in (0, 1)


def test_locate_sigmas_are_image_only_and_the_map_sigma_is_a_switch(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """api.md §6 "Locate sigmas": trust class none, the prior never changes them; the
    reference keyframes' map sigma joins only with ``OH_MY_SLAM_LOCATE_MAP_SIGMA``."""
    P, D = _scene()
    idx = rl.Index([_kf("a", A, P, D, sigma=0.05),
                    _kf("b", rooms.camera(0.3, 0.2, 1.0, -8.0, -10.0), P, D, sigma=0.05)])
    T = rooms.camera(0.15, 0.1, 1.05, -4.0, -12.0)
    monkeypatch.delenv(sigmas.LOCATE_MAP_SIGMA_ENV, raising=False)
    bare = rl.locate(_query(T, P, D), idx)
    with_prior = rl.locate(_query(T, P, D, prior=T.copy(), prior_sigma_pos_m=0.01,
                                  prior_sigma_rot_deg=0.5), idx)
    assert bare.located and with_prior.located
    assert (with_prior.sigma_pos_m, with_prior.sigma_rot_deg) == (bare.sigma_pos_m,
                                                                  bare.sigma_rot_deg)
    assert bare.sigma_pos_m is not None and bare.sigma_pos_m < 0.05  # not the keyframes' 5 cm
    monkeypatch.setenv(sigmas.LOCATE_MAP_SIGMA_ENV, "1")
    combined = rl.locate(_query(T, P, D), idx)
    assert combined.sigma_pos_m is not None and combined.sigma_pos_m >= 0.05
    monkeypatch.setenv(sigmas.LOCATE_MAP_SIGMA_ENV, "maybe")
    with pytest.raises(ValueError, match=sigmas.LOCATE_MAP_SIGMA_ENV):
        sigmas.locate_map_sigma()


def test_a_symmetric_twin_makes_the_pose_ambiguous_and_the_prior_picks() -> None:
    P, D = _scene()
    S = _rz(180.0, np.array([2.75, 0.0, 0.0]))
    twin = _apply(S, P)
    idx = rl.Index([_kf("a", A, P, D), _kf("twin", S @ A, twin, D)])
    T = rooms.camera(0.1, 0.05, 1.0, -3.0, -10.0)
    o = rl.locate(_query(T, P, D), idx)
    assert o.located and o.ambiguity is not None
    assert o.ambiguity["rot_deg"] == pytest.approx(180.0, abs=1.0)
    assert o.ambiguity["pos_m"] == pytest.approx(
        float(np.linalg.norm(T[:3, 3] - (S @ T)[:3, 3])), abs=0.05)
    assert o.rivals and o.rivals[0]["support"] <= 1.0
    for prior in (T, S @ T):
        p = rl.locate(_query(T, P, D, prior=prior), idx)
        assert p.T is not None and rl.pose_distance(p.T, prior)[0] < 0.05
        assert p.consistent_with_prior is True and p.ambiguity is not None
    # the camera's height picks too, when the twin is not at the same height
    Sz = _rz(180.0, np.array([2.75, 0.0, 0.0]), dz=0.5)
    idx2 = rl.Index([_kf("a", A, P, D), _kf("twin", Sz @ A, _apply(Sz, P), D)])
    for z, want in ((1.0, T), (1.5, Sz @ T)):
        h = rl.locate(_query(T, P, D, camera_z=z), idx2)
        assert h.T is not None and rl.pose_distance(h.T, want)[0] < 0.05


def test_a_prior_far_from_every_hypothesis_is_a_prior_conflict() -> None:
    """No supported hypothesis within 3 x the prior's sigmas: rejected prior_conflict, not
    located with consistent_with_prior false (api.md §6); the frame keeps its nearest keyframe."""
    P, D = _scene()
    idx = rl.Index([_kf("a", A, P, D)])
    far = A.copy()
    far[0, 3] += 1.0
    o = rl.locate(_query(A, P, D, prior=far), idx)
    assert not o.located and o.reason == "prior_conflict" and o.nearest == 0
    wide = rl.locate(_query(A, P, D, prior=far, prior_sigma_pos_m=0.5), idx)
    assert wide.located and wide.consistent_with_prior is True and wide.reason is None


def test_keyframes_from_one_spot_place_a_turn_without_depth() -> None:
    P, D = _scene(n=800)
    C = np.array([0.0, 0.0, 1.0])
    kfs = [_kf(f"k{y}", rooms.camera(*C, y, -10.0), _apply(_rz(0, C), P), D, depth=False)
           for y in (-20.0, 0.0, 20.0)]
    idx = rl.Index(kfs)
    T = rooms.camera(*C, 10.0, -10.0)
    o = rl.locate(_query(T, P, D, camera_z=1.02), idx)
    assert o.located and o.method == "turn" and o.T is not None
    assert rl.pose_distance(o.T, T)[1] < 0.3 and o.T[2, 3] == pytest.approx(1.02)
    assert o.ambiguity is None and o.sigma_rot_deg is not None and o.sigma_rot_deg > 0
    # keyframes apart (not one spot): no turn fit
    apart = [_kf("p", rooms.camera(0.0, 0.0, 1.0, 0.0, -10.0), P, D, depth=False),
             _kf("q", rooms.camera(0.0, 1.0, 1.0, 0.0, -10.0), P, D, depth=False)]
    assert not rl.locate(_query(T, P, D), rl.Index(apart)).located


def test_a_turn_with_a_twin_view_is_ambiguous() -> None:
    P, D = _scene(n=800)
    C = np.array([0.0, 0.0, 1.0])
    a = _kf("a", rooms.camera(*C, 0.0, -10.0), P, D, depth=False)
    T90 = rooms.camera(*C, 90.0, -10.0)
    b = rl.Keyframe("b", T90, K, a.uv, a.desc, a.xyz, 0.01, 0.1)  # sees the same, turned 90 deg
    o = rl.locate(_query(rooms.camera(*C, 3.0, -10.0), P, D), rl.Index([a, b]))
    assert o.located and o.ambiguity is not None
    assert o.ambiguity["rot_deg"] == pytest.approx(90.0, abs=2.0)


def test_no_overlap_and_few_matches() -> None:
    P, D = _scene()
    idx = rl.Index([_kf("a", A, P, D)])
    rng = np.random.default_rng(9)
    q = rl.Query(rng.uniform(0, 300, (300, 2)).astype(np.float32),
                 rng.integers(0, 256, (300, 128)).astype(np.uint8), K, 320, 240)
    o = rl.locate(q, idx)
    assert not o.located and o.nearest is None
    few = _query(A, P[:10], D[:10])
    assert not rl.locate(few, idx).located
    empty = rl.Query(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8), K, 320, 240)
    assert not rl.locate(empty, idx).located
    assert rl.Index([]).votes(D).size == 0


def test_many_keyframes_are_matched_by_votes_and_the_prior() -> None:
    P, D = _scene()
    rng = np.random.default_rng(3)
    others = []
    for i in range(rl.MATCH_ALL + 2):
        Pi = rng.uniform(-20, -10, (200, 3))
        Di = rng.integers(0, 256, (200, 128)).astype(np.uint8)
        others.append(rl.Keyframe(f"o{i}", rooms.camera(-15.0, 0.0, 1.0, 0.0), K,
                                  rng.uniform(0, 300, (200, 2)).astype(np.float32), Di,
                                  Pi.astype(np.float32), 0.01, 0.1))
    idx = rl.Index([*others, _kf("a", A, P, D)])
    chosen, votes = rl.candidates(_query(A, P, D), idx)
    assert len(idx.keyframes) - 1 in chosen and votes[-1] == votes.max()
    prior = rooms.camera(-15.0, 0.0, 1.0, 0.0)
    chosen, _ = rl.candidates(_query(A, P, D, prior=prior), idx)
    assert len(chosen) > rl.TOP_K or any(i < len(others) for i in chosen)
    o = rl.locate(_query(A, P, D), idx)
    assert o.located and o.nearest == len(idx.keyframes) - 1


def test_nearest_view_turns_toward_where_the_keyframe_looks() -> None:
    P, D = _scene()
    kf = _kf("a", A, P, D)
    idx = rl.Index([kf])
    q = _query(A, P, D)
    k, turn = rl.nearest_view(q, idx, 0)
    assert k is kf and turn is None  # no prior: the frame's heading is unknown
    q.prior = A
    _, turn = rl.nearest_view(q, idx, 0)
    assert turn is not None and abs(turn) < 15.0
    q.prior = rooms.camera(0.0, 0.0, 1.0, 90.0)  # looking 90 deg left of the keyframe's view
    _, turn = rl.nearest_view(q, idx, 0)
    assert turn == pytest.approx(-90.0, abs=15.0)
    blind = rl.Keyframe("b", A, K, kf.uv, kf.desc, np.full_like(kf.xyz, np.nan), 0.0, 0.0)
    assert np.allclose(blind.view_point(), A[:3, 3] + A[:3, 2])
    above = A.copy()
    above[:3, 3] = blind.view_point() - [0, 0, 1]
    q.prior = above
    _, turn = rl.nearest_view(q, rl.Index([blind]), 0)
    assert turn == pytest.approx(0.0, abs=1e-6)


def test_helpers() -> None:
    assert rl.wrap_deg(270.0) == -90.0 and rl.yaw_deg(rooms.camera(0, 0, 0, 30.0)) == \
        pytest.approx(30.0)
    pos, rot = rl.covariance(A, K, np.zeros((3, 2)), np.zeros((3, 3)))  # degenerate
    assert math.isinf(pos) or pos >= 0
    assert rl.pnp(np.zeros((3, 2)), np.zeros((3, 3)), K, 4.0) is None
    rng = np.random.default_rng(0)
    assert rl.pnp(rng.uniform(0, 300, (40, 2)), rng.uniform(1, 5, (40, 3)), K, 4.0) is None
    img = np.zeros((64, 64), np.uint8)
    uv, desc = rl.extract(img)
    assert uv.shape == (0, 2) and desc.shape == (0, 128)
    big = (rng.integers(0, 255, (1500, 2100))).astype(np.uint8)
    mask = np.zeros(big.shape, bool)
    mask[:, :1000] = True
    uv, _ = rl.extract(big, mask)
    assert len(uv) and uv[:, 0].min() > 900 and uv[:, 0].max() < 2100
    assert rl.keyframe_points(uv[:0], K, None, A).shape == (0, 3)
    pts = rl.keyframe_points(np.array([[160.0, 120.0], [0.0, 0.0]], np.float32), K,
                             np.array([[2.0, 0.0], [2.0, 2.0]], np.float32), np.eye(4))
    assert np.allclose(pts[0], [0, 0, 2]) or np.isnan(pts[0]).all()


def test_choosing_and_rivals() -> None:
    """``choose``: supported hypotheses only; with a prior the nearest within 3 prior sigmas, else
    none; without, the best-supported at the camera's height. ``rivals_of``: every supported
    hypothesis beyond 0.10 m or 5 deg, whatever it explains, strongest first, a repeat of a
    stronger rival listed once."""
    def at(x: float, deg: float = 0.0) -> np.ndarray:
        T = _rz(deg, np.zeros(3))
        T[0, 3] = x
        return T

    poses = [at(0.0), at(1.0), at(0.05), at(2.0), at(1.02)]
    inl = [100, 60, 90, 20, 50]
    q = rl.Query(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8), K, 320, 240)
    assert rl.choose(q, poses, inl) == 0 and rl.prior_distance(q, A) == 0.0
    assert rl.within_prior(q, A)
    q.camera_z = 0.0
    lifted = [at(0.0), at(1.0)]
    lifted[0][2, 3] = 1.0  # the best is off the camera's height: the other is level
    assert rl.choose(q, lifted, [100, 60]) == 1
    q.prior, q.prior_sigma_pos_m, q.prior_sigma_rot_deg = at(0.9), 0.05, 2.0
    assert rl.choose(q, poses, inl) == 1  # 1.0 is 2 sigmas off; 1.02 is farther
    q.prior = at(2.0)
    assert rl.choose(q, poses, inl) is None  # the only one there has support 0.2
    q.prior = at(0.0, 10.0)
    assert rl.choose(q, poses, inl) is None  # turned 10 deg: beyond 3 x 2 deg
    # the combined bound (api.md §4, §6): 3 x sqrt(sigma_prior^2 + sigma_hypothesis^2), each
    # hypothesis's own calibrated sigma; 10 deg off a 2 deg prior is within 3 x hypot(2, 3)
    sig = [(0.01, 3.0)] * len(poses)
    assert rl.choose(q, poses, inl, sig) == 0
    assert rl.within_prior(q, at(0.0, 10.0), (0.01, 3.0))
    assert not rl.within_prior(q, at(0.3, 10.0), (0.01, 3.0))  # 0.3 m: beyond 3 x hypot(.05,.01)
    assert rl.within_prior(q, at(0.3, 10.0), (0.01, 3.0), position=False)
    # ranked over the combined sigmas: a loose hypothesis farther off can be nearer
    q.prior = at(0.0)
    near = rl.prior_distance(q, at(0.1), (0.5, 1.0))
    assert near < rl.prior_distance(q, at(0.05), (0.0, 1.0))
    rivals = rl.rivals_of(poses[0], poses[1:], inl[1:], 100)
    assert [r["support"] for r in rivals] == [0.6]  # 0.05 m is the pose; 1.02 repeats 1.0
    assert rivals[0]["pos_m"] == pytest.approx(1.0) and rivals[0]["rot_deg"] == 0.0
    turned = rl.rivals_of(poses[0], [at(0.0, 6.0)], [25], 100)
    assert turned == [{"pos_m": 0.0, "rot_deg": pytest.approx(6.0, abs=1e-3), "support": 0.25}]
    assert rl.rivals_of(poses[0], [at(1.0)], [10], 0) == []
    P, D = _scene()
    small = _kf("s", rooms.camera(0.2, 0.0, 1.0, 0.0, -10.0), P[:12], D[:12])
    o = rl.locate(_query(A, P, D), rl.Index([_kf("a", A, P, D), small]))
    assert o.located and o.ambiguity is None and o.rivals == []
    assert rl.Index([rl.Keyframe("e", A, K, np.zeros((0, 2), np.float32),
                                 np.zeros((0, 128), np.uint8), np.zeros((0, 3), np.float32))]
                    ).votes(D).tolist() == [0]


def test_a_weak_turn_rival_is_no_ambiguity() -> None:
    P, D = _scene(n=800)
    C = np.array([0.0, 0.0, 1.0])
    a = _kf("a", rooms.camera(*C, 0.0, -10.0), P, D, depth=False)
    T90 = rooms.camera(*C, 90.0, -10.0)
    b = rl.Keyframe("b", T90, K, a.uv[:60], a.desc[:60], a.xyz[:60], 0.01, 0.1)
    o = rl.locate(_query(rooms.camera(*C, 3.0, -10.0), P, D), rl.Index([a, b]))
    assert o.located and o.method == "turn" and o.ambiguity is None


def test_triangulated_points_follow_the_keyframe_poses_and_rescale_the_depth() -> None:
    P, D = _scene(n=800)
    a = _kf("a", A, P, D)
    b = _kf("b", rooms.camera(0.0, 0.6, 1.0, -10.0, -10.0), P, D)
    far_away = _kf("c", rooms.camera(0.0, 0.0, 1.0, 180.0, -10.0), P, D)  # looks the other way
    for k in (a, b):  # depth 20 % too far, and half the keypoints without a match elsewhere
        k.xyz = (k.centre + 1.2 * (k.xyz - k.centre)).astype(np.float32)
    lone = rl.Keyframe("d", A, K, a.uv[:5], a.desc[:5], a.xyz[:5], 0.0, 0.0)
    assert rl.triangulate([a])[0] is not a.xyz and np.allclose(rl.triangulate([a])[0], a.xyz)
    pts = rl.triangulate([a, b, far_away, lone])
    _, _, idx = _view(A, P, D)
    err = np.linalg.norm(pts[0] - P[idx], axis=1)
    assert np.nanmedian(err) < 0.01  # triangulated, or depth rescaled to them
    assert np.allclose(pts[2], far_away.xyz, equal_nan=True)  # no neighbour to triangulate with
    assert np.allclose(pts[3], lone.xyz, equal_nan=True)
    calls: list[tuple[int, int]] = []

    def cached(i: int, j: int) -> np.ndarray:
        calls.append((i, j))
        return np.zeros((0, 2), np.int64)

    out = rl.triangulate([a, b], cached)
    assert calls == [(0, 1), (1, 0)] and np.allclose(out[0], a.xyz, equal_nan=True)


def test_a_triangulation_that_disagrees_with_the_depth_is_not_taken() -> None:
    """Depth twice too far (or keyframe poses the images did not confirm): the triangulation's
    scale against the depth leaves the range, and the depth points stand; single points far from
    their depth are dropped."""
    P, D = _scene(n=800)
    a = _kf("a", A, P, D)
    b = _kf("b", rooms.camera(0.0, 0.6, 1.0, -10.0, -10.0), P, D)
    a2 = rl.Keyframe("a2", a.T, a.K, a.uv, a.desc,
                     (a.centre + 2.0 * (a.xyz - a.centre)).astype(np.float32))
    pts = rl.triangulate([a2, b])
    assert np.allclose(pts[0], a2.xyz, equal_nan=True)
    a3 = rl.Keyframe("a3", a.T, a.K, a.uv, a.desc, a.xyz.copy())
    a3.xyz[:20] = (a.centre + 3.0 * (a.xyz[:20] - a.centre)).astype(np.float32)
    pts = rl.triangulate([a3, b])
    _, _, idx = _view(A, P, D)
    assert np.nanmedian(np.linalg.norm(pts[0] - P[idx], axis=1)) < 0.01


def test_an_implausible_solve_is_not_located(monkeypatch: pytest.MonkeyPatch) -> None:
    P, D = _scene()
    monkeypatch.setattr(rl, "plausible", lambda h, kfs, index: False)
    o = rl.locate(_query(A, P, D), rl.Index([_kf("a", A, P, D)]))
    assert not o.located


def test_repeated_rooms_with_noisy_keyframes_are_flagged_and_the_prior_picks() -> None:
    """Three copies of one structure 4 m apart (a house of repeated rooms), the frame's own copy
    mapped by one keyframe, the others by two, every keyframe 2 cm / 0.5 deg off: without a prior
    the pose may be any copy, and the others are its rivals (never silent); with a prior the copy
    that agrees with it is picked although it has less support, and a prior that no hypothesis
    agrees with is a prior conflict."""
    P, D = _scene(n=500)
    rng = np.random.default_rng(4)

    def off(T: np.ndarray) -> np.ndarray:
        out = _rz(float(rng.normal(0, 0.5)), T[:3, 3]) @ T
        out[:3, 3] += rng.normal(0, 0.02, 3)
        return out

    kfs = []
    for k, n in ((0, 1), (1, 2), (2, 2)):
        S = np.eye(4)
        S[1, 3] = 4.0 * k
        for j in range(n):
            T = S @ rooms.camera(0.0, 0.15 * j, 1.0, 4.0 * j, -10.0)
            kf = _kf(f"c{k}{j}", T, _apply(S, P), D)
            kf.T = off(kf.T)
            kfs.append(kf)
    idx = rl.Index(kfs)
    T = rooms.camera(0.1, 0.05, 1.0, -3.0, -10.0)
    o = rl.locate(_query(T, P, D), idx)
    assert o.T is not None and o.ambiguity is not None and len(o.rivals) >= 2
    assert o.ambiguity == {"pos_m": o.rivals[0]["pos_m"], "rot_deg": o.rivals[0]["rot_deg"]}
    assert [r["support"] for r in o.rivals] == sorted((r["support"] for r in o.rivals),
                                                      reverse=True)
    assert all(0.25 <= r["support"] <= 1.0 for r in o.rivals)
    p = rl.locate(_query(T, P, D, prior=T), idx)
    assert p.T is not None and rl.pose_distance(p.T, T)[0] < 0.1 and p.consistent_with_prior
    lost = T.copy()
    lost[0, 3] += 2.0  # a prior 2 m off every copy
    q = rl.locate(_query(T, P, D, prior=lost), idx)
    assert q.T is None and q.reason == "prior_conflict"


def test_a_weak_twin_neither_flags_nor_widens() -> None:
    """A twin view mapped by a keyframe that sees a sixth of the structure: a rival with less
    than a quarter of the pose's support is no ambiguity and leaves the sigmas as they are."""
    P, D = _scene()
    S = _rz(180.0, np.array([2.75, 0.0, 0.0]))
    twin = _kf("twin", S @ A, _apply(S, P)[:100], D[:100])
    idx = rl.Index([_kf("a", A, P, D), twin])
    T = rooms.camera(0.1, 0.05, 1.0, -3.0, -10.0)
    o = rl.locate(_query(T, P, D), idx)
    assert o.T is not None and o.ambiguity is None and rl.pose_distance(o.T, T)[0] < 0.02
    assert o.sigma_pos_m is not None and o.sigma_pos_m < 0.05


def test_a_frame_with_a_prior_always_gets_a_nearest_keyframe() -> None:
    """No keyframe looks like the frame (no votes): with a prior, the keyframe nearest it."""
    P, D = _scene()
    idx = rl.Index([_kf("a", A, P, D), _kf("b", rooms.camera(3.0, 0.0, 1.0, 0.0), P, D)])
    rng = np.random.default_rng(9)
    q = rl.Query(rng.uniform(0, 300, (300, 2)).astype(np.float32),
                 rng.integers(0, 256, (300, 128)).astype(np.uint8), K, 320, 240,
                 prior=rooms.camera(2.8, 0.0, 1.0, 0.0))
    o = rl.locate(q, idx)
    assert not o.located and o.nearest == 1


def test_a_turn_with_a_prior_picks_its_hypothesis_or_conflicts() -> None:
    P, D = _scene(n=800)
    C = np.array([0.0, 0.0, 1.0])
    a = _kf("a", rooms.camera(*C, 0.0, -10.0), P, D, depth=False)
    T90 = rooms.camera(*C, 90.0, -10.0)
    b = rl.Keyframe("b", T90, K, a.uv, a.desc, a.xyz, 0.01, 0.1)  # sees the same, turned 90 deg
    T = rooms.camera(*C, 3.0, -10.0)
    for want in (T, _rz(90.0, C) @ T):
        o = rl.locate(_query(T, P, D, prior=want), rl.Index([a, b]))
        assert o.located and o.method == "turn" and o.T is not None
        assert rl.pose_distance(o.T, want)[1] < 2.0 and o.consistent_with_prior is True
        assert o.ambiguity is not None and o.rivals[0]["rot_deg"] == pytest.approx(90.0, abs=2.0)
    o = rl.locate(_query(T, P, D, prior=_rz(45.0, C) @ T), rl.Index([a, b]))
    assert not o.located and o.reason == "prior_conflict"


def test_a_refinement_that_leaves_the_prior_keeps_the_solve(monkeypatch: pytest.MonkeyPatch
                                                             ) -> None:
    P, D = _scene()
    idx = rl.Index([_kf("a", A, P, D)])

    def away(h: rl.Hypothesis, *a: object) -> tuple[rl.Hypothesis, None]:
        T = h.T.copy()
        T[0, 3] += 1.0
        return rl.Hypothesis(T, h.inliers, h.mask), None

    monkeypatch.setattr(rl, "_refined", away)
    o = rl.locate(_query(A, P, D, prior=A), idx)
    assert o.T is not None and rl.pose_distance(o.T, A)[0] < 0.05


def test_a_turn_whose_leftover_rays_hold_no_second_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    P, D = _scene(n=800)
    C = np.array([0.0, 0.0, 1.0])
    a = _kf("a", rooms.camera(*C, 0.0, -10.0), P, D, depth=False)
    b = rl.Keyframe("b", rooms.camera(*C, 90.0, -10.0), K, a.uv, a.desc, a.xyz, 0.01, 0.1)
    real = rl.turn_ransac
    calls = [0]

    def once(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        calls[0] += 1
        R, inl = real(x, y)
        return (R, inl) if calls[0] == 1 else (R, np.zeros(len(x), bool))

    monkeypatch.setattr(rl, "turn_ransac", once)
    o = rl.locate(_query(rooms.camera(*C, 3.0, -10.0), P, D), rl.Index([a, b]))
    assert o.located and o.method == "turn" and o.rivals == [] and calls[0] == 2


def test_a_keyframe_located_with_its_own_tight_prior_locates_and_a_far_prior_conflicts() -> None:
    """R20: a fused keyframe's own image located with its own committed pose as a metric prior of
    its committed (tight) sigmas is located: the bound combines the prior's sigma with the
    hypothesis's calibrated one, so the image noise alone never makes a conflict. A prior 1 m off
    still conflicts."""
    P, D = _scene()
    B = rooms.camera(0.3, 0.2, 1.0, -8.0, -10.0)
    idx = rl.Index([_kf("a", A, P, D, sigma=0.002), _kf("b", B, P, D, sigma=0.002)])
    tight = {"prior_sigma_pos_m": 0.002, "prior_sigma_rot_deg": 0.05}
    o = rl.locate(_query(A, P, D, noise=2.0, prior=A.copy(), **tight), idx)
    assert o.located and o.consistent_with_prior is True and o.reason is None
    assert o.T is not None and rl.pose_distance(o.T, A)[0] > 0.0  # the image noise moved it
    off = A.copy()
    off[0, 3] += 1.0
    far = rl.locate(_query(A, P, D, noise=2.0, prior=off, **tight), idx)
    assert not far.located and far.reason == "prior_conflict"
