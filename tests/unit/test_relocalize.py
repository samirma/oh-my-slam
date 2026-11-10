"""Locate's core (``mapping.relocalize``) on synthetic keyframes: points with random descriptors seen
from known poses, a symmetric twin for ambiguity, keyframes from one spot for the turn fit."""

from __future__ import annotations

import math

import numpy as np
import pytest

from oh_my_slam.mapping import relocalize as rl
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
    return rl.Keyframe(name, T, K, uv, d, xyz, sigma, 0.1)


def _query(T: np.ndarray, P: np.ndarray, D: np.ndarray, noise: float = 0.5, **kw: object
           ) -> rl.Query:
    uv, d, _ = _view(T, P, D, noise)
    return rl.Query(uv, d, K, 320, 240, **kw)  # type: ignore[arg-type]


A = rooms.camera(0.0, 0.0, 1.0, 0.0, -10.0)


def test_a_frame_near_a_keyframe_is_located_with_sigmas_from_its_fit() -> None:
    P, D = _scene()
    idx = rl.Index([_kf("a", A, P, D), _kf("b", rooms.camera(0.3, 0.2, 1.0, -8.0, -10.0), P, D)])
    T = rooms.camera(0.15, 0.1, 1.05, -4.0, -12.0)
    o = rl.locate(_query(T, P, D), idx)
    assert o.located and o.method == "points" and o.ambiguity is None
    assert o.T is not None
    dp, dr = rl.pose_distance(o.T, T)
    assert dp < 0.01 and dr < 0.2
    assert o.sigma_pos_m is not None and 0.0005 < o.sigma_pos_m < 0.03  # not floored at 1 cm
    noisy = rl.locate(_query(T, P, D, noise=2.0), idx)
    assert noisy.sigma_pos_m is not None and noisy.sigma_pos_m > o.sigma_pos_m
    assert o.inlier_ratio is not None and o.inlier_ratio > 0.9
    assert o.consistent_with_prior is None and o.nearest in (0, 1)


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


def test_a_prior_far_from_the_pose_is_inconsistent() -> None:
    P, D = _scene()
    idx = rl.Index([_kf("a", A, P, D)])
    far = A.copy()
    far[0, 3] += 1.0
    o = rl.locate(_query(A, P, D, prior=far), idx)
    assert o.located and o.consistent_with_prior is False


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


def test_rivals_need_distinct_evidence_and_small_keyframes_give_no_hypothesis() -> None:
    m1 = np.zeros(16, bool)
    m1[:6] = True
    iq = np.r_[np.arange(10), np.arange(6)]  # correspondences 10-15: keypoints 0-5 again
    xyz = np.zeros((16, 3))
    xyz[10:] = [2.0, 0.0, 0.0]  # elsewhere: a twin
    best = rl.Hypothesis(np.eye(4), 6, m1)
    far = np.eye(4)
    far[0, 3] = 1.0
    same = rl.Hypothesis(far, 6, m1.copy())
    near = rl.Hypothesis(np.eye(4), 4, ~m1)
    m2 = np.zeros(16, bool)
    m2[6:10] = True
    other = rl.Hypothesis(far, 4, m2)
    close = np.eye(4)
    close[0, 3] = 0.2
    m3 = np.zeros(16, bool)
    m3[10:] = True
    twin = rl.Hypothesis(close, 6, m3)
    assert rl._rival(best, [same, near], iq, xyz) is None
    assert rl._rival(best, [same, other, rl.Hypothesis(far, 2, m2)], iq, xyz) is other
    assert rl._rival(best, [rl.Hypothesis(close, 4, m2)], iq, xyz) is None  # near, other points
    assert rl._rival(best, [twin], iq, xyz) is twin
    assert rl.spread(best, [same], iq, xyz) == pytest.approx((1.0 / math.sqrt(2), 0.0))
    P, D = _scene()
    small = _kf("s", rooms.camera(0.2, 0.0, 1.0, 0.0, -10.0), P[:12], D[:12])
    o = rl.locate(_query(A, P, D), rl.Index([_kf("a", A, P, D), small]))
    assert o.located and o.ambiguity is None
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
