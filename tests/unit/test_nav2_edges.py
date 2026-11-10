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
