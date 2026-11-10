"""Edges of the nav2 changes: the volume's small grids and floor fallback, wall growth without
band tops, door objects' chords, and locate's refinement and rival filtering."""

from __future__ import annotations

import numpy as np
import pytest

from oh_my_slam.mapping import navgrid as ng
from oh_my_slam.mapping import relocalize as rl
from tests.synth import rooms
from tests.unit.test_relocalize import A, _apply, _kf, _query, _rz, _scene


def test_a_tiny_depth_grid_has_no_normals_and_the_floor_falls_back_on_layers() -> None:
    T = rooms.camera(2.0, 2.0, 1.0, 0.0, -30.0)
    K = np.array([[2.0, 0.0, 1.5], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])
    d = np.full((2, 3), 2.0, np.float32)  # two rows: no room for the normals' stencil
    v = ng.Volume()
    assert v.integrate("a", d, K, T, d > 0)
    assert v.flat_z.sum() == 0  # no flat test on so few points
    full = ng.Volume()
    d2 = rooms.two_rooms().depth(T)
    full.integrate("b", d2, rooms.K640, T, d2 > 0)
    assert full.flat_z.sum() > 0
    # nothing flat below -5 m: the layers of flat hits decide
    assert full.estimate_floor(-5.0) == pytest.approx(-0.025, abs=0.03)


def _lay(state: np.ndarray, top: bool) -> ng.Layers:
    R, C = state.shape
    return ng.Layers(np.zeros(2), 0.1, state.astype(np.uint8), np.ones((R, C), np.uint16),
                     np.full((R, C), np.nan, np.float32), np.zeros((R, C), np.uint8),
                     np.where(state == 2, 1.0, np.nan).astype(np.float32) if top else None)


def test_walls_grow_without_band_tops() -> None:
    st = np.ones((20, 30), np.uint8)
    st[10, 2:28] = 2  # a 2.6 m wall
    st[3, 3] = 2  # a leg: not a wall
    out = ng.inflate_walls(_lay(st, top=False))
    assert out.top is None and out.state[9, 10] == 2 and out.state[11, 10] == 2
    assert out.state[2, 3] == 1
    withtop = ng.inflate_walls(_lay(st, top=True))
    assert withtop.top is not None and withtop.top[9, 10] == pytest.approx(1.0)


def test_a_door_object_finds_the_gap_of_a_wall_seen_partly() -> None:
    st = np.zeros((40, 70), np.uint8)
    st[5:35, 2:34] = 1
    st[5:35, 36:68] = 1
    st[18:35, 34:36] = 2  # the wall below the gap goes on
    st[7:9, 34:36] = 2  # above it only a 0.2 m stub: no walled chord
    st[9:18, 34:36] = 1  # the gap, 0.9 m
    lay = _lay(st, top=False)
    plain = ng.features(lay)
    assert not plain.passages
    door = [3.5, 1.35]
    f = ng.features(lay, doors=[door, [20.0, 20.0]])  # the second lies off the grid's free space
    (p,) = f.passages
    assert p.kind == "door" and abs(p.centre[1] - 1.35) < 0.1 and 0.8 <= p.width <= 1.1
    again = ng.features(lay, doors=[door, door])  # a door already given its chord
    assert len(again.passages) == 1


def test_refinement_survives_singular_systems(monkeypatch: pytest.MonkeyPatch) -> None:
    P, D = _scene()
    kf = _kf("a", A, P, D)
    ok = np.isfinite(kf.xyz).all(axis=1)
    xyz = kf.xyz[ok].astype(np.float64)[:50]
    uv = kf.uv[ok].astype(np.float64)[:50]
    origin = np.repeat(A[:3, 3][None], len(xyz), axis=0)

    def boom(*a: object, **k: object) -> None:
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(np.linalg, "solve", boom)
    monkeypatch.setattr(np.linalg, "inv", boom)
    T, cov, rms = rl.refine_keyframes(A, kf.K, uv, xyz, origin, 1.0)
    assert np.allclose(T, A) and not np.isfinite(cov).all() and rms < 1.0


def test_keyframe_sigmas_without_inliers_and_a_rival_in_the_same_basin(
        monkeypatch: pytest.MonkeyPatch) -> None:
    P, D = _scene()
    idx = rl.Index([_kf("a", A, P, D)])
    h = rl.Hypothesis(A, 0, np.zeros(3, bool))
    assert rl._keyframe_sigmas(h, np.zeros(3, np.int64), idx) == (0.0, 0.0)
    S = _rz(180.0, np.array([2.75, 0.0, 0.0]))
    twin = rl.Index([_kf("a", A, P, D), _kf("twin", S @ A, _apply(S, P), D)])
    T = rooms.camera(0.1, 0.05, 1.0, -3.0, -10.0)
    monkeypatch.setattr(rl, "_same_basin", lambda *a: True)  # every rival: the same place
    o = rl.locate(_query(T, P, D), twin)
    assert o.located and o.ambiguity is None


def test_a_1m_repeated_structure_inflates_the_sigmas_and_the_prior_picks() -> None:
    """e051 loc091: two copies of one structure 1 m apart, each mapped by its own keyframe; the
    frame sees one of them. Its rival lies 1 m away with comparable support: the sigmas cover
    it (half its distance at least), and a prior that agrees with the rival picks the rival."""
    P, D = _scene()
    shift = np.eye(4)
    shift[0, 3] = 1.0
    idx = rl.Index([_kf("a", A, P, D), _kf("b", shift @ A, _apply(shift, P), D)])
    T = rooms.camera(0.05, 0.02, 1.0, -2.0, -10.0)
    o = rl.locate(_query(T, P, D), idx)
    assert o.located and o.ambiguity is not None
    assert o.ambiguity["pos_m"] == pytest.approx(1.0, abs=0.05)
    assert o.sigma_pos_m is not None and o.sigma_pos_m >= 0.5 * o.ambiguity["pos_m"] - 1e-3
    for prior in (T, shift @ T):
        p = rl.locate(_query(T, P, D, prior=prior), idx)
        assert p.T is not None and rl.pose_distance(p.T, prior)[0] < 0.05
        assert p.consistent_with_prior is True and p.sigma_pos_m is not None
        assert p.sigma_pos_m >= 0.49


def test_a_low_inlier_ratio_floors_the_sigmas() -> None:
    P, D = _scene()
    idx = rl.Index([_kf("a", A, P, D)])
    T = rooms.camera(0.05, 0.02, 1.0, -2.0, -10.0)
    q = _query(T, P, D)
    rng = np.random.default_rng(3)  # most of the frame's matches point nowhere
    kf = idx.keyframes[0]
    bad = np.isfinite(kf.xyz).all(axis=1)
    kf.xyz[bad] += rng.normal(0, 1.0, (int(bad.sum()), 3)).astype(np.float32) * \
        (rng.random(int(bad.sum())) < 0.7)[:, None]
    o = rl.locate(q, idx)
    assert o.located and o.inlier_ratio is not None and o.inlier_ratio < rl.LOW_RATIO
    assert o.sigma_pos_m is not None and o.sigma_pos_m >= rl.LOW_RATIO_SIGMA_POS_M
    assert o.sigma_rot_deg is not None and o.sigma_rot_deg >= rl.LOW_RATIO_SIGMA_ROT_DEG
