"""One standpoint and rotation consistency without priors (``mapping.standpoint``, map.md §1,
§4) on the ray-cast textured room: the two-view fit's parallax and rotation, standpoint groups
registered as a pure rotation about a common centre, and the ``rotation_inconsistent`` limit."""

from __future__ import annotations

import numpy as np
import pytest

from oh_my_slam.mapping import anchored as an
from oh_my_slam.mapping import standpoint as sp
from tests.synth.room import look
from tests.unit.test_anchored import K, shot

C0 = (0.0, 0.0, 0.5)


def view(name: str, T: np.ndarray, order: int = 0, stored: bool = False,
         R: np.ndarray | None = None) -> sp.View:
    img, _ = shot(T)
    return sp.View(name, K, an.features(img), (T[:3, :3] if R is None else R).copy(),
                   T[:3, 3].copy(), stored, order)


def _yaw(R: np.ndarray, deg: float) -> np.ndarray:
    return an.exp_so3(np.array([0.0, 0.0, np.radians(deg)])) @ R


def test_two_view_parallax_and_rotation() -> None:
    A, B = look(0, C0, -10), look(25, C0, -10)
    tv = sp.two_view(view("a", A), view("b", B))
    assert tv is not None and tv.standpoint and tv.parallax < sp.PARALLAX_MAX_PX and tv.share > 0.3
    assert an.rot_deg(tv.R_ab.T @ (A[:3, :3].T @ B[:3, :3])) < 0.2
    moved = look(25, (0.4, 0.2, 0.5), -10)
    tm = sp.two_view(view("a", A), view("m", moved))
    assert tm is not None and not tm.standpoint and tm.parallax > sp.PARALLAX_MAX_PX
    assert an.rot_deg(tm.R_ab.T @ (A[:3, :3].T @ moved[:3, :3])) < 3.0  # the epipolar fit's
    away = look(180, C0, -10)
    assert sp.two_view(view("a", A), view("w", away)) is None  # nothing shared
    pairs = sp.Pairs()
    a, b = view("a", A), view("b", B)
    one = pairs.get(a, b)
    back = pairs.get(b, a)  # the same fit, reversed
    assert one is not None and back is not None and np.allclose(back.R_ab, one.R_ab.T)
    assert pairs.get(a, b) is one and len(pairs.fits) == 2
    assert pairs.get(a, view("w", away)) is None and pairs.get(view("w", away), a) is None


def test_a_turn_in_place_is_registered_as_a_pure_rotation_about_the_stored_keyframe() -> None:
    """The engine's centres scattered: the group of one standpoint takes the stored keyframe's
    centre and keeps the engine's rotations; rotations 60 deg off fail the rotation check."""
    truth = [look(y, C0, -10) for y in (0, 25, 50, 75)]
    views = [view("k", truth[0], stored=True)]
    rng = np.random.default_rng(0)
    for i, T in enumerate(truth[1:], 1):
        v = view(f"n{i}", T, i, R=_yaw(T[:3, :3], 60.0))
        v.c = v.c + rng.normal(0, 0.03, 3)
        views.append(v)
    pairs = sp.Pairs()
    med, mx = sp.consistency(views, pairs)
    assert sp.inconsistent(med, mx)  # the engine's rotations disagree with the images'
    got = sp.groups(views, pairs, 0.5)
    assert set(got) == {"n1", "n2", "n3"} and len({g.group for g in got.values()}) == 1
    for v in views[1:]:
        g = got[v.name]
        assert np.allclose(g.R, v.R)  # the engine's rotation, kept
        assert np.allclose(g.c, truth[0][:3, 3]) and 0.0 < g.sigma_pos_m < 0.1
        assert g.sigma_rot_deg > 0.0
    for v, T in zip(views[1:], truth[1:], strict=True):
        v.R = T[:3, :3].copy()
    med, mx = sp.consistency(views, pairs)
    assert med is not None and mx is not None and not sp.inconsistent(med, mx)


def test_a_new_only_group_centres_on_its_mean_and_nothing_measurable_passes() -> None:
    truth = [look(y, C0, -10) for y in (0, 25)]
    views = [view("a", truth[0], 0), view("b", truth[1], 1, R=_yaw(truth[1][:3, :3], 5.0))]
    views[1].c = views[1].c + np.array([0.02, 0.0, 0.0])
    got = sp.groups(views, sp.Pairs())
    assert np.allclose(got["a"].c, truth[0][:3, 3] + [0.01, 0.0, 0.0])
    assert got["a"].sigma_pos_m == pytest.approx(0.01)
    assert an.rot_deg(got["a"].R.T @ truth[0][:3, :3]) < 1e-9  # the first frame: the reference
    assert np.allclose(got["b"].R, views[1].R)  # the engine's own rotation
    lone = [view("a", truth[0], 0), view("w", look(180, C0, -10), 1)]
    assert sp.groups(lone, sp.Pairs()) == {}
    assert sp.consistency(lone, sp.Pairs()) == (None, None)
    assert not sp.inconsistent(None, None)
    assert sp.inconsistent(4.0, 4.0) and sp.inconsistent(1.0, 11.0)
    assert not sp.inconsistent(2.0, 9.0)


def test_two_view_and_group_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    A, B = look(0, C0, -10), look(25, C0, -10)
    a, b = view("a", A), view("b", B)
    monkeypatch.setattr(an.cv2, "findEssentialMat", lambda *x, **k: (None, None))
    assert sp.two_view(a, b) is None  # no epipolar fit
    monkeypatch.undo()
    monkeypatch.setattr(sp, "rotation_ransac", lambda *x, **k: None)
    assert sp.two_view(a, b) is None  # no rotation fit
    monkeypatch.undo()
    # every view that can share a view is a neighbour, in canonical name order whatever the
    # request order; one facing away (axes 180 deg apart) or too far is not
    far = [view(f"v{i}", look(5 * i, C0, -10), i) for i in (2, 10, 1)]
    far.append(view("w", look(180, C0, -10)))
    far.append(view("x", look(0, (9.0, 0.0, 0.5), -10)))
    assert sp.neighbours(far, 0) == [2, 1]  # v1 before v10
    assert sp.neighbours(list(reversed(far)), 4) == [2, 3]  # the same views, by name
    # a new frame reached only through a stored keyframe that is not the group's root
    s1 = view("s1", A, stored=True)
    s2 = view("s2", look(0, C0, -10), stored=True)
    n = view("n", B, 1)
    pairs = sp.Pairs()
    pairs.fits[("n", "s1")] = sp.TwoView(np.eye(3), 0.0, 100, 0.5, 0.01)
    pairs.fits[("n", "s2")] = None
    got = sp.groups([s1, s2, n], pairs)
    assert set(got) == {"n"}
    pairs2 = sp.Pairs()
    pairs2.fits[("n", "s1")] = sp.TwoView(np.eye(3), 0.0, 100, 0.5, 0.01)
    pairs2.fits[("n", "s2")] = sp.TwoView(np.eye(3), 0.0, 300, 0.5, 0.01)
    pairs2.fits[("m", "s1")] = sp.TwoView(np.eye(3), 0.0, 150, 0.5, 0.01)
    pairs2.fits[("m", "s2")] = None
    pairs2.fits[("m", "n")] = None
    pairs2.fits[("n", "m")] = None
    m = view("m", B, 2)
    got = sp.groups([s1, s2, n, m], pairs2)
    assert "n" in got and "m" not in got  # m joins the group only through s1, not its root


def test_the_parallax_limit_on_measured_cases() -> None:
    """map.md §1, 1 px: the worst pure turn measured on this room (+30 deg from 10 deg, about
    0.57 px at fx 160) is one standpoint; a 2 cm move (about 1.5 px) is not."""
    turn = sp.two_view(view("a", look(10, C0, -10)), view("b", look(40, C0, -10)))
    assert turn is not None and 0.3 < turn.parallax < sp.PARALLAX_MAX_PX and turn.standpoint
    got = sp.groups([view("a", look(10, C0, -10), 0), view("b", look(40, C0, -10), 1)],
                    sp.Pairs())
    assert set(got) == {"a", "b"} and got["a"].group == got["b"].group
    A = look(0, C0, -10)
    move = sp.two_view(view("a", A), view("m", look(10, (0.02, 0.0, 0.5), -10)))
    assert move is not None and move.parallax > sp.PARALLAX_MAX_PX and not move.standpoint
    assert sp.groups([view("a", A, 0), view("m", look(10, (0.02, 0.0, 0.5), -10), 1)],
                     sp.Pairs()) == {}


def test_the_standpoint_centre_ties_go_to_the_lowest_keyframe_id() -> None:
    """map.md §1 (P3): two overlapped stored keyframes with as many pair inliers: the centre is
    the lowest keyframe id's, whatever their order."""
    A, B = look(0, C0, -10), look(25, C0, -10)
    s_hi = view("s_hi", look(0, (0.2, 0.0, 0.5), -10), stored=True)
    s_hi.kid = "k000007"
    s_lo = view("s_lo", A, stored=True)
    s_lo.kid = "k000003"
    n = view("n", B, 1)
    for order in ([s_hi, s_lo, n], [s_lo, s_hi, n]):
        pairs = sp.Pairs()
        pairs.fits[("n", "s_hi")] = sp.TwoView(np.eye(3), 0.0, 100, 0.5, 0.01)
        pairs.fits[("n", "s_lo")] = sp.TwoView(np.eye(3), 0.0, 100, 0.5, 0.01)
        got = sp.groups(order, pairs)
        assert np.allclose(got["n"].c, s_lo.c)
