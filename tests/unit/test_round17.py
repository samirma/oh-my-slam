"""Round 17 (spec P30, P30a, P30b, P30c): the depth-consistency hard limit and
``quality.depth_pairs`` (oms/2.6); the parallax choice for rotated hypotheses when the measured
parallax is plausible, the prior first; the two-view parallax of a turn in place on a repeating
texture; frames kept at a prior their images contradict contribute nothing, free-by-rule cells
included; a not-judged frame's sent height gives no scale evidence from its own depth (map.md
§1, §2, §4, §6; api.md §4). The scene: an empty rectilinear room whose walls repeat every 2 m,
seen by the client's narrow camera (52 deg) turning in place in 26.5 deg steps."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
from typing import Any

import numpy as np
import pytest

from oh_my_slam.core.types import Intrinsics
from oh_my_slam.mapping import anchored as an
from oh_my_slam.mapping import standpoint as sp
from tests.synth.room import Room, base_pose, jpeg, png_depth, render
from tests.unit.test_service_api import client as old  # noqa: F401
from tests.unit.test_service_prior_first import client  # noqa: F401

K = Intrinsics(655.8, 655.8, 320.0, 240.0, 640, 480)  # the client's camera: 52 deg wide
N = 8
CAM = {"fx": K.fx, "fy": K.fy, "cx": K.cx, "cy": K.cy, "width": K.width, "height": K.height}


def _poses(yaw0: float = -53.0) -> list[np.ndarray]:
    return [base_pose(yaw0 + 26.5 * i, -0.3, 0.37, 0.5, -0.0436, 0.0) for i in range(N)]


POSES = _poses()


@functools.cache
def _room(period: float) -> Room:
    return Room(floor="tiles", boxes=[], walls="periodic", wall_period=period)


@functools.cache
def _shot(period: float, i: int) -> tuple[an.Features, np.ndarray, np.ndarray]:
    rgb, z = render(_room(period), POSES[i], K)
    return an.features(rgb), z.astype(np.float32), rgb


def _view(period: float, i: int, depth: bool = True) -> sp.View:
    f, z, _ = _shot(period, i)
    T = POSES[i]
    return sp.View(f"f{i}", K, f, T[:3, :3].copy(), T[:3, 3].copy(), False, i,
                   depth=(lambda: (z, K)) if depth else None)


def _rel(i: int, j: int) -> np.ndarray:
    return np.linalg.inv(POSES[i]) @ POSES[j]


def _frames(c: Any, period: float, idx: list[int], prior: str | None = None
            ) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    frames, files = [], {}
    for i in idx:
        _, z, rgb = _shot(period, i)
        img = jpeg(rgb)
        c.backend.mono_depth[hashlib.sha256(img).hexdigest()] = z
        files[f"f{i}"], files[f"f{i}d"] = img, png_depth(z)
        fr: dict[str, Any] = {"name": f"f{i}", "image": f"f{i}", "depth": f"f{i}d",
                              "camera_id": "cam", "height_above_floor_m": 0.5}
        if prior is not None:
            fr["prior"] = {"trust": prior, "T_map_cam": POSES[i].tolist(),
                           "sigma_pos_m": 0.05, "sigma_rot_deg": 2.0}
        frames.append(fr)
    return frames, files


def _errors(job: dict[str, Any]) -> list[tuple[str, float, float]]:
    """Each registered frame's rotation (deg) and position (m) error relative to the first."""
    reg = [(k, i) for k, i in enumerate(job["inputs"]) if i["status"] == "registered"]
    k0, i0 = reg[0]
    T0 = np.array(i0["T_map_cam"])
    out = []
    for k, i in reg[1:]:
        rel = np.linalg.inv(T0) @ np.array(i["T_map_cam"])
        tr = np.linalg.inv(POSES[k0]) @ POSES[k]
        out.append((i["name"], an.rot_deg(tr[:3, :3].T @ rel[:3, :3]),
                    float(np.linalg.norm(tr[:3, 3] - rel[:3, 3]))))
    return out


# -- the two-view parallax of a turn in place on a repeating texture (code fix) ----------------


def test_a_turn_in_place_measures_its_parallax_on_a_repeating_wall() -> None:
    """The adjacent turns f1-f2 and f5-f6 of the 2 m room: their largest epipolar consensus
    takes the walls' period-shifted matches with it (a turn in place satisfies every essential
    matrix), whose median read 365 and 545 px; the rotation-only consensus (over 40 % of the
    inliers, one standpoint) now measures the pair: about 0.5 px, as the true pose predicts."""
    for i in (1, 5):
        tv = sp.two_view(_view(2.0, i, False), _view(2.0, i + 1, False))
        assert tv is not None and tv.parallax < sp.PARALLAX_MAX_PX, (i, tv.parallax)
    # a moving base keeps its parallax: the rotation-only consensus is no standpoint there
    room = Room()
    T = [base_pose(0.0, -0.3 + 0.25 * k, 0.37, 0.5) for k in range(2)]
    shots = [render(room, t, K) for t in T]
    a, b = (sp.View(f"m{k}", K, an.features(s[0]), t[:3, :3], t[:3, 3]) for k, (s, t) in
            enumerate(zip(shots, T, strict=True)))
    tv = sp.two_view(a, b)
    assert tv is not None and tv.parallax > 20.0


def test_turn_parallax_pieces() -> None:
    """``turn_parallax``: the whole median below the standpoint bound, or a rotation-only
    consensus too small (``CHECK_MIN_INLIERS``, ``TURN_MIN_SHARE``) or itself no standpoint,
    keeps the whole median; otherwise the consensus's."""
    assert sp.TURN_MIN_SHARE == an.SUPPORT_MIN == 0.25
    px = np.r_[np.full(300, 0.5), np.full(400, 400.0)]
    rot = np.r_[np.ones(300, bool), np.zeros(400, bool)]
    assert sp.turn_parallax(px, rot) == 0.5
    assert sp.turn_parallax(np.full(10, 0.4), np.ones(10, bool)) == 0.4  # a standpoint already
    small = np.r_[np.ones(90, bool), np.zeros(610, bool)]
    assert sp.turn_parallax(px, small) == 400.0  # below 100
    few = np.r_[np.full(150, 0.5), np.full(650, 400.0)]
    assert sp.turn_parallax(few, np.r_[np.ones(150, bool), np.zeros(650, bool)]) == 400.0
    moving = np.r_[np.full(300, 8.0), np.full(400, 50.0)]
    assert sp.turn_parallax(moving, rot) == 50.0  # its consensus is no standpoint


# -- P30: the parallax separates rotated hypotheses when plausible ---------------------------


def test_the_rotated_pick_pieces() -> None:
    """``parallax_plausible`` (README "Symmetric scenes": the measured parallax at most
    ``PARALLAX_PLAUSIBLE_MARGIN`` = 2 x the largest the hypotheses predict), ``rotated_pick``
    (plausible, then the first matching hypothesis when those matching are one geometry),
    ``one_geometry`` and ``prior_geometry`` (the prior first, within 3 x its position sigma)."""
    assert an.PARALLAX_PLAUSIBLE_MARGIN == 2.0
    assert an.parallax_plausible(0.7, [0.5, 130.0]) and an.parallax_plausible(260.0, [None, 130.0])
    assert not an.parallax_plausible(437.0, [0.5, 130.0])
    assert not an.parallax_plausible(1.0, [None, None])
    I3 = np.eye(3)
    R90 = an.exp_so3(np.array([0.0, np.radians(90.0), 0.0]))
    c = [np.zeros(3), np.array([2.0, 0.0, 0.0]), np.zeros(3)]
    assert an.rotated_pick(0.7, [0.5, 130.0, None], [I3, R90, R90], c) == 0
    assert an.rotated_pick(120.0, [0.5, 130.0, None], [I3, R90, R90], c) == 1
    assert an.rotated_pick(437.0, [0.5, 130.0, None], [I3, R90, R90], c) is None  # implausible
    assert an.rotated_pick(0.7, [0.5, 1.2, None], [I3, R90, R90], c) is None  # both match
    assert an.rotated_pick(50.0, [0.5, 130.0, None], [I3, R90, R90], c) is None  # none matches
    assert an.one_geometry([], [I3], [c[0]]) is None
    assert an.one_geometry([0, 2], [I3, R90, I3], c) == 0
    assert an.one_geometry([0, 2], [I3, R90, R90], c) is None  # same place, turned
    assert an.one_geometry([0, 1], [I3, I3, I3], c) is None  # twins
    # the prior: positions within 3 sigma, one geometry
    assert an.prior_geometry([I3, R90], c[:2], np.array([0.05, 0.0, 0.0]), 0.05) == 0
    assert an.prior_geometry([I3, R90], c[:2], np.array([2.0, 0.0, 0.0]), 0.05) == 1
    assert an.prior_geometry([I3, R90], [c[0], c[0]], np.zeros(3), 0.05) is None
    assert an.prior_geometry([I3, R90], c[:2], np.array([9.0, 0.0, 0.0]), 0.05) is None


def _rotated(monkeypatch: pytest.MonkeyPatch, poses: list[np.ndarray],
             measured: float | None = None) -> sp.TwoView | None:
    """``two_view`` of the 2 m room's turn f0-f1 whose hypotheses (its largest consensus,
    repeated) the depth images all confirm at the metric relative poses ``poses``."""
    a, b = _view(2.0, 0), _view(2.0, 1)
    e = an.epipolar(a.feats, a.K, b.feats, b.K)
    assert e is not None
    ok = an.DepthCounts(100, 100, 0, 0, 0)
    monkeypatch.setattr(sp, "depth_check", lambda *x, **k: (True, ok, ok))
    monkeypatch.setattr(sp, "epipolar_hypotheses", lambda *x, **k: iter([e] * len(poses)))
    it = iter(poses)
    monkeypatch.setattr(sp, "depth_pose",
                        lambda *x: sp.DepthPose(next(it).copy(), 50, 60, "ransac", 0.0))
    if measured is not None:
        monkeypatch.setattr(sp, "turn_parallax", lambda *x: measured)
    return sp.two_view(a, b)


def test_rotated_hypotheses_are_separated_by_a_plausible_parallax(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """map.md §1 "Symmetric scenes" (P30): two depth-confirmed hypotheses 90 deg apart (the
    turn in place and a lattice pose 2.75 m away): the pair takes the one whose predicted
    parallax matches its measured one, whichever comes first; a measurement far above what the
    hypotheses' baselines and depths allow (implausible), or one both match, leaves the pair
    ambiguous."""
    true = _rel(0, 1)
    lattice = true.copy()
    lattice[:3, :3] = an.exp_so3(np.array([0.0, np.radians(90.0), 0.0])) @ true[:3, :3]
    lattice[:3, 3] += np.array([2.2, 0.0, 1.6])
    for order, want in (([true, lattice], 1), ([lattice, true], 2)):
        tv = _rotated(monkeypatch, order)
        assert tv is not None and tv.measurable and not tv.ambiguous and tv.hyp == want
        assert an.rot_deg(tv.R_ab.T @ true[:3, :3]) < 0.01
    tv = _rotated(monkeypatch, [true, lattice], measured=5000.0)  # implausible
    assert tv is not None and tv.ambiguous and not tv.measurable
    monkeypatch.setattr(sp, "predicted_parallax", lambda *x: 1.0)
    tv = _rotated(monkeypatch, [true, lattice])  # both match
    assert tv is not None and tv.ambiguous


def test_the_real_room_pair_takes_its_true_pose() -> None:
    """The 2 m room's f0-f1 (P24 left it ambiguous: hyp 1 the turn in place, 0.7 px measured,
    0.5 predicted; hyp 2 a lattice pose 63 deg, 130 px predicted): the pair is measured at its
    true rotation."""
    tv = sp.two_view(_view(2.0, 0), _view(2.0, 1))
    assert tv is not None and tv.measurable and not tv.ambiguous
    assert an.rot_deg(_rel(0, 1)[:3, :3].T @ tv.R_ab) < 0.5


def test_mapanythings_gate_separates_rotated_candidates() -> None:
    """``standpoint.places`` (map.md §2 step 2, P30): a candidate pose 90 deg from the pair's
    confirmed fit is placed only when the parallax picks it (plausibly): never when the fit
    matches, the candidate's prediction is missing, or the measurement is implausible."""
    a, b = _view(2.0, 0), _view(2.0, 1)
    pairs = sp.Pairs()
    tv = pairs.get(a, b)
    assert tv is not None and tv.measurable and tv.t_m is not None and tv.measured_px is not None
    turned = _rel(0, 1).copy()
    turned[:3, :3] = an.exp_so3(np.array([0.0, np.radians(90.0), 0.0])) @ turned[:3, :3]
    turned[:3, 3] += np.array([2.0, 0.0, 1.0])
    assert not sp.places(a, b, turned, pairs)  # the parallax picks the fit
    key = ("f0", "f1")
    pairs.fits[key] = dataclasses.replace(tv, predicted_px=100.0, measured_px=1500.0)
    real = sp.predicted_parallax
    try:
        sp.predicted_parallax = lambda *x: 1500.0  # type: ignore[assignment]
        assert sp.places(a, b, turned, pairs)  # the candidate's prediction matches
        pairs.fits[key] = dataclasses.replace(tv, predicted_px=1.0, measured_px=5000.0)
        sp.predicted_parallax = lambda *x: 1.0  # type: ignore[assignment]
        assert not sp.places(a, b, turned, pairs)  # implausible
    finally:
        sp.predicted_parallax = real  # type: ignore[assignment]


def test_the_geometric_gate_separates_rotated_candidates(
        tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """``AnchoredMap.depth_confirms`` (map.md §2 step 2, P30): a candidate pose that a reference
    confirms with another confirmed one 90 deg away: a prior sent for the frame decides first
    (within 3 x its position sigma), else the parallax when plausible; the pose picked is
    placed (its twins then judged as before), one picked over it replaces it once refined on all
    the frame's correspondences when it holds, and nothing separating them places it in
    neither."""
    from tests.unit.test_anchored import frame, new_map

    am = new_map(tmp_path)
    am.update([frame("a", base_pose(0.0), p=None), frame("b", base_pose(20.0), p=None)])
    R90 = an.exp_so3(np.array([0.0, np.radians(90.0), 0.0]))
    state: dict[str, Any] = {"m": 0.5, "p": [0.5, 300.0], "twin": False}

    def amb(self: Any, f: Any, T: Any, own: Any, corr: Any, cands: Any, j: int, ref: Any,
            twin: Any = None, turned: Any = None) -> float | None:
        assert turned is not None
        turned.append(an.Solve(R90 @ T[:3, :3], T[:3, 3] + np.array([2.0, 0.0, 0.0]),
                               np.eye(6) * 1e-6, np.ones(len(corr.uv), bool), 0.5))
        if state["twin"] and twin is not None:
            twin.append(an.Solve(T[:3, :3].copy(), T[:3, 3] + np.array([0.0, 1.0, 0.0]),
                                 np.eye(6) * 1e-6, np.ones(len(corr.uv), bool), 0.5))
        return 90.0

    monkeypatch.setattr(an.AnchoredMap, "_ambiguous", amb)
    def parallax(self: Any, *x: Any) -> tuple[float, list[float]]:
        if state["p"] == "by rotation":  # turned candidates 300 px, the rest (twins) 0.5 px
            return state["m"], [0.5 if an.rot_deg(x[4][0].R.T @ p.R) < 10.0 else 300.0
                                for p in x[4]]
        return state["m"], list(state["p"][:len(x[4])])

    monkeypatch.setattr(an.AnchoredMap, "_twin_parallax", parallax)
    assert am.update([frame("c", base_pose(10.0), p=None)])[0].status == "registered"
    state["p"] = [0.5, 0.6]  # nothing separates them
    assert am.update([frame("d", base_pose(5.0), p=None)])[0].status == "rejected"
    state["m"], state["p"] = 2000.0, [0.5, 300.0]  # implausible
    assert am.update([frame("e", base_pose(15.0), p=None)])[0].status == "rejected"
    # the parallax picks its pose over the turned one, then its twins (both 0.5 px): neither
    state["m"], state["p"], state["twin"] = 0.5, "by rotation", True
    assert am.update([frame("i", base_pose(12.0), p=None)])[0].status == "rejected"
    # with priors (in the map's frame): the prior at the true pose decides before the parallax
    # (which would pick the turned one), for rotated candidates as for twins
    state["m"], state["p"], state["twin"] = 300.0, [0.5, 300.0], False
    pm = new_map(tmp_path / "p")
    pm.update([frame("a", base_pose(0.0)), frame("b", base_pose(20.0))])
    pl = pm.update([frame("g", base_pose(12.0))])[0]
    assert pl.status == "registered" and pl.T is not None
    assert an.rot_deg(pl.T[:3, :3].T @ base_pose(12.0)[:3, :3]) < 1.0


def test_the_prior_decides_first_for_twins(tmp_path: Any,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """map.md §2 step 2 (P30): translational twins against one reference are decided by a
    prior sent for the frame first, within 3 x its position sigma: the twin at the prior is
    placed although the parallax would pick the other."""
    from tests.unit.test_anchored import frame, new_map, prior

    am = new_map(tmp_path)
    am.update([frame("a", base_pose(0.0)), frame("b", base_pose(20.0))])
    shift = np.array([0.0, 1.0, 0.0])

    def amb(self: Any, f: Any, T: Any, own: Any, corr: Any, cands: Any, j: int, ref: Any,
            twin: Any = None, turned: Any = None) -> float | None:
        if twin is not None:
            twin.append(an.Solve(T[:3, :3].copy(), T[:3, 3] + shift, np.eye(6) * 1e-6,
                                 np.ones(len(corr.uv), bool), 0.5))
        return None

    monkeypatch.setattr(an.AnchoredMap, "_ambiguous", amb)
    monkeypatch.setattr(an.AnchoredMap, "_twin_parallax", lambda self, *x: (0.5, [300.0, 0.5]))
    T = base_pose(12.0)
    pl = am.update([frame("g", T, p=prior(T, sp=0.05))])[0]
    assert pl.status == "registered" and pl.T is not None
    assert float(np.linalg.norm(pl.T[:3, 3] - T[:3, 3])) < 0.1  # the prior's, not the twin


# -- P30, P30a: the depth-consistency hard limit and quality.depth_pairs ------------------------


def _kf(name: str, T: np.ndarray, sigma: float = 0.01, kind: str = "sensor") -> an.Keyframe:
    return an.Keyframe(name, name, "cam", K, T.copy(), sigma, 0.1, "fused", None, kind)


def test_depth_pairs_pieces() -> None:
    """``depth_pairs``: each new view with every later one that can share a view (``close``),
    the names in canonical order, the counts at the committed relative pose with the margin
    ``SENSOR_DEPTH_TOL`` x depth + 3 x the combined committed position sigma, failing under the
    depth confirmation's count and threshold in either direction; a direction with fewer than
    ``MIN_AGREEING`` counted points has a null ratio; ``depth_counts``' margin widens both the
    agreement and the see-through band."""
    z = [_shot(2.0, i)[1] for i in range(4)]
    truth = [(_kf(f"f{i}", POSES[i]), (z[i], K)) for i in range(4)]
    got = sp.depth_pairs(truth, 4)
    assert got and all(not p.fails for p in got)
    rows = [p.row() for p in got]
    assert {(r["a"], r["b"]) for r in rows} >= {("f0", "f1"), ("f1", "f2")}
    assert all(set(r) == {"a", "b", "agree_ratio_ab", "agree_ratio_ba", "fails"} for r in rows)
    # f3 committed 90 deg off about its own centre: it contradicts the views it now overlaps
    R = an.exp_so3(np.array([0.0, 0.0, np.radians(-90.0)])) @ POSES[3][:3, :3]
    off = [*truth[:3], (_kf("f3", an.pose(R, POSES[3][:3, 3])), (z[3], K))]
    bad = [p for p in sp.depth_pairs(off, 4) if p.fails]
    assert bad and all("f3" in (p.a, p.b) for p in bad)
    # in canonical order whichever comes first; the margin of the committed sigmas
    rev = sp.depth_pairs([truth[1], truth[0]], 2)
    assert [(p.a, p.b) for p in rev] == [("f0", "f1")]
    loose = sp.depth_pairs([(_kf("f0", POSES[0], 0.5), (z[0], K)), truth[1]], 2)
    assert loose[0].margin_m == pytest.approx(3.0 * np.hypot(0.5, 0.01))
    far = an.pose(POSES[1][:3, :3], POSES[1][:3, 3] + np.array([20.0, 0.0, 0.0]))
    assert sp.depth_pairs([truth[0], (_kf("f9", far), (z[1], K))], 2) == []  # not close
    # only the first ``new`` are paired with the later ones: stored ones with each other not
    assert sp.depth_pairs([truth[0], truth[1], truth[2]], 0) == []
    few = an.DepthCounts(100, 10, 5, 0, 85)
    p = sp.DepthPair("a", "b", few, an.DepthCounts(100, 50, 0, 0, 50), 0.0)
    assert p.row()["agree_ratio_ab"] is None and p.row()["agree_ratio_ba"] == 1.0 and not p.fails
    p = sp.DepthPair("a", "b", an.DepthCounts(100, 40, 0, 0, 60),
                     an.DepthCounts(100, 40, 20, 0, 40), 0.0)
    assert p.fails and p.row()["agree_ratio_ba"] == pytest.approx(0.6667, abs=1e-4)
    A = an.DepthRef(np.eye(4), np.full((8, 8), 2.0, np.float32), K.resized(8, 8).K())
    B = an.DepthRef(an.pose(np.eye(3), np.array([0.0, 0.0, 0.3])),
                    np.full((8, 8), 2.0, np.float32), K.resized(8, 8).K())
    assert an.depth_counts(A, B, 0.1).through > 0
    assert an.depth_counts(A, B, 0.1, 0.2).through == 0
    assert an.depth_counts(A, B, 0.1, 0.2).agree == an.depth_counts(A, B, 0.1, 0.2).points - \
        an.depth_counts(A, B, 0.1, 0.2).outside


def test_depth_consistency_pairs_registered_sensor_frames_only(tmp_path: Any) -> None:
    """``depth_consistency``: the update's registered frames (their keyframes) with each other
    and with the stored image-confirmed keyframes, sensor depth only; a kept or monocular frame
    is never paired, and without such a frame nothing is evaluated."""
    from tests.unit.test_anchored import frame, new_map

    am = new_map(tmp_path)
    pls = am.update([frame("a", base_pose(0.0)), frame("b", base_pose(15.0))])
    bad, rows = sp.depth_consistency(am, pls)
    assert not bad and [(r["a"], r["b"]) for r in rows] == [("a", "b")]
    more = am.update([frame("c", base_pose(30.0)), frame("m", base_pose(20.0), depth=False)])
    bad, rows = sp.depth_consistency(am, more)
    assert not bad and {(r["a"], r["b"]) for r in rows} == {("a", "c"), ("b", "c")}
    assert sp.depth_consistency(am, []) == (False, [])
    kept = dataclasses.replace(pls[0], status="kept_at_prior")
    assert sp.depth_consistency(am, [kept]) == (False, [])
    # a committed view turned 90 deg off the stored ones fails the update
    kf = am.keyframe(more[0].keyframe or "")
    assert kf is not None
    kf.T = an.pose(an.exp_so3(np.array([0.0, 0.0, np.radians(60.0)])) @ kf.T[:3, :3], kf.centre)
    bad, rows = sp.depth_consistency(am, more)
    assert bad and any(r["fails"] for r in rows)


def _no_rotated_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    """P24 as it was: two confirmed hypotheses (or candidates) 10 deg or more apart are always
    ambiguous (the parallax never separates them): the 2 m room's turn then locks onto the
    walls' lattice, 90 deg off (diag-lattice)."""
    monkeypatch.setattr(an, "rotated_pick", lambda *x: None)
    monkeypatch.setattr(sp, "rotated_pick", lambda *x: None)


def test_a_lattice_locked_turn_is_depth_inconsistent(
        client: Any, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """map.md §4 "Depth consistency" (P30, P30a): the 2 m room's turn without priors registered
    whole through a lattice pose (every measured pair agrees with it, the rotation check
    passes), its frames 90 deg off the first: their committed views contradict the first one's
    depth, so the job is rejected ``depth_inconsistent`` and commits nothing, its dry run
    predicts it, and ``quality.depth_pairs`` shows the pairs that contradict."""
    from oh_my_slam.service import prior_first
    from tests.unit.test_service_prior_first import dry

    monkeypatch.setenv(prior_first.NONE_ENGINE_ENV, "geometric")
    _no_rotated_choice(monkeypatch)
    frames, files = _frames(client, 2.0, list(range(N)))
    pred = dry(client, "m", frames, files, cameras={"cam": CAM})
    job = client.update("m", frames, files, cameras={"cam": CAM})
    q = job["quality"]
    assert job["state"] == "rejected" and "depth_inconsistent" in q["failed_checks"], q
    assert max(r for _, r, _ in _errors(job)) > 80.0  # the lattice commit it refuses
    bad = [r for r in q["depth_pairs"] if r["fails"]]
    assert bad and all(r["agree_ratio_ab"] is None or r["agree_ratio_ab"] <= 1.0 for r in bad)
    assert any(min(x for x in (r["agree_ratio_ab"], r["agree_ratio_ba"]) if x is not None)
               < an.DEPTH_CONFIRM_AGREEMENT for r in bad)
    assert "depth_inconsistent" in pred["quality"]["failed_checks"]
    assert pred["quality"]["depth_pairs"] == q["depth_pairs"]
    assert job["timings_s"]["depth_consistency"] < 1.0
    assert client.http.get("/api/maps/m").status_code == 404


def test_the_room_turn_commits_with_its_true_rotations(
        client: Any, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """The same turn with the P30 parallax choice and the turn-in-place parallax: the frames
    are registered at their true rotations, no pair contradicts (the pairs listed, none
    failing), and the job commits."""
    from oh_my_slam.service import prior_first

    monkeypatch.setenv(prior_first.NONE_ENGINE_ENV, "geometric")
    frames, files = _frames(client, 2.0, list(range(N)))
    job = client.update("m", frames, files, cameras={"cam": CAM})
    q = job["quality"]
    assert job["state"] == "committed", q
    assert "depth_inconsistent" not in q["failed_checks"]
    assert q["depth_pairs"] and not any(r["fails"] for r in q["depth_pairs"])
    assert all(r <= 1.0 for _, r, _ in _errors(job))
    assert all(r["a"] < r["b"] for r in q["depth_pairs"])
    assert any(r["agree_ratio_ab"] is not None for r in q["depth_pairs"])


def test_the_limit_applies_with_priors(client: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
                                       ) -> None:
    """map.md §4 (P30): with or without priors: a turn sent with metric priors at the true poses
    lists its pairs in ``depth_pairs`` and passes; one frame's prior turned 90 deg (kept or
    rejected by its prior checks, never paired as registered) never fails it; on the older
    mapper the list is empty."""
    frames, files = _frames(client, 3.0, list(range(4)), prior="metric")
    job = client.update("m", frames, files, cameras={"cam": CAM})
    q = job["quality"]
    assert job["state"] == "committed", q
    assert q["depth_pairs"] and not any(r["fails"] for r in q["depth_pairs"])


# -- P30b, P30c: a frame kept at a prior its images contradict contributes nothing -----------------


def test_a_failed_check_with_a_conflicting_image_pose_shows_the_conflict(
        tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """api.md §4 (P30c): a frame kept for a failed check whose image-only pose conflicts with its
    prior (the conflict test, no measurable-pair threshold) keeps the check's reason with
    ``detail`` ``conflict_unsupported``; without a conflict, or with a loose image pose
    (``loose_image_pose`` keeps precedence), the detail is null."""
    from tests.unit.test_anchored import frame, new_map, prior

    m = new_map(tmp_path)
    m.update([frame("a", base_pose(0.0)), frame("b", base_pose(20.0))])
    T = base_pose(10.0)
    off = T.copy()
    off[:3, 3] += np.array([0.0, 0.0, 0.5])
    for P, want in ((off, "conflict_unsupported"), (T, None)):
        f = dataclasses.replace(frame("x", T, p=prior(P)), unconfirmed="blur")
        pl = m.place(f, an.features(f.rgb), None)
        assert (pl.status, pl.reason, pl.detail) == ("kept_at_prior", "blur", want)
    monkeypatch.setattr(an, "CONFIRM_SIGMA_ROT_DEG", 0.0)  # every image pose loose
    f = dataclasses.replace(frame("y", T, p=prior(off)), unconfirmed="blur")
    assert m.place(f, an.features(f.rgb), None).detail is None


def test_too_close_keeps_the_conflict_of_the_placement_it_demotes(tmp_path: Any) -> None:
    """``AnchoredMap._too_close`` (api.md §4, P30c): a too-close frame whose placement conflicts
    with its prior (rejected ``prior_conflict`` or kept ``conflict_unsupported``) is kept for
    the check with ``detail`` ``conflict_unsupported``; ``kept_at`` keeps that detail on a
    failed check (and only that one); such a keyframe has no too-close exception, and the flag
    is stored with it."""
    from tests.unit.test_anchored import frame, new_map

    m = new_map(tmp_path)
    m.update([frame("a", base_pose(0.0))])
    f = frame("x", base_pose(5.0))
    for pl, want in ((an.Placement("x", "rejected", "prior_conflict"), "conflict_unsupported"),
                     (an.Placement("x", "kept_at_prior", "no_overlap",
                                   detail="conflict_unsupported"), "conflict_unsupported"),
                     (an.Placement("x", "kept_at_prior", "no_overlap",
                                   detail="loose_image_pose"), None),
                     (an.Placement("x", "registered"), None)):
        got = m._too_close(f, pl, 0.2)
        assert (got.status, got.reason, got.detail) == ("kept_at_prior", "too_close", want)
    blur = dataclasses.replace(f, unconfirmed="blur")
    got = m._too_close(blur, an.Placement("x", "rejected", "prior_conflict"), 0.2)
    assert (got.reason, got.detail) == ("blur", "conflict_unsupported")
    assert m.kept_at(f, "plain", 0, detail="loose_image_pose").detail is None
    kf = an.Keyframe("k", "x", "cam", K, np.eye(4), 0.01, 0.1, "provisional",
                     {"trust": "metric"}, "sensor", reason="too_close")
    assert kf.too_close_exception()
    kf.conflict = True
    assert not kf.too_close_exception() and not kf.detects() and not kf.contributes()
    assert an.Keyframe.from_json(kf.to_json()).conflict


def _stands(c: Any) -> list[np.ndarray]:
    assert c.service.derived is not None
    return [s.T_map_cam for s in c.service.derived.get("m", c.service.store.latest("m")).stands]


def test_a_contradicted_prior_makes_no_camera_stand(
        client: Any, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """map.md §6 "Unconfirmed frames" (P30b): a frame kept at a prior its images contradict
    (``conflict_unsupported``, short of the measurable-pair threshold) is committed (its row
    ``no_stand``) but makes no camera stand or near blind disc; a frame kept for want of an image
    pose keeps its stand."""
    from tests.unit.test_service_prior_first import frames_for

    c = client
    from tests.unit.test_service_prior_first import CAM as CAM_PF

    fr, files = frames_for(c, [base_pose(0), base_pose(20)], tag="a")
    assert c.update("m", fr, files, cameras={"cam": CAM_PF})["state"] == "committed"
    monkeypatch.setattr(an, "measurable_pose", lambda *a: False)
    T = base_pose(10)
    off = T.copy()
    off[:3, 3] += np.array([0.0, 0.0, 0.5])
    fr, files = frames_for(c, [T], tag="b", priors=[off])
    job = c.update("m", fr, files)
    assert job["state"] == "committed"
    assert (job["inputs"][0]["status"], job["inputs"][0]["detail"]) == (
        "kept_at_prior", "conflict_unsupported")
    rows = {r["name"]: r for r in c.service.store.version("m")["frames"]}
    assert rows["b0"].get("no_stand") is True and "no_stand" not in rows["a0"]
    stands = _stands(c)
    assert len(stands) == 2 and all(not np.allclose(s, off) for s in stands)
    # a frame without an image pose (far away) keeps its stand
    far = base_pose(10, 40.0, 0.0)
    fr, files = frames_for(c, [far], tag="c", depth=False)
    monkeypatch.undo()
    job = c.update("m", fr, files)
    assert job["inputs"][0]["status"] == "kept_at_prior"
    assert job["inputs"][0]["detail"] != "conflict_unsupported"
    assert len(_stands(c)) == 3


def test_the_older_mapper_withholds_a_contradicted_prior(
        old: Any, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """P30b/P30c on the older mapper: a conflicting frame short of the measurable-pair threshold
    is kept ``conflict_unsupported`` with no stand (its row ``no_stand``); a too-close frame
    whose pose conflicts keeps ``too_close`` with ``conflict_unsupported`` and no too-close
    content; the depth-consistency limit is not this engine's (``depth_pairs`` empty)."""
    from oh_my_slam.service import update
    from tests.unit.test_service_api import CAMERA, frames_of

    monkeypatch.setattr(update, "MEASURABLE_INLIERS", 10 ** 6)

    def pr(i: int, T: np.ndarray) -> dict[str, Any]:
        P = T.copy()
        if i == 2:
            P[:3, 3] += [0.0, 0.0, 0.5]
        return {"trust": "metric", "T_map_cam": P.tolist(), "sigma_pos_m": 0.02}

    frames, files, _ = frames_of(old, 6, prior=pr)
    job = old.update("m", frames, files, {"cam": CAMERA})
    assert job["state"] == "committed" and job["quality"]["depth_pairs"] == []
    assert job["inputs"][2]["detail"] == "conflict_unsupported"
    rows = old.service.store.version("m")["frames"]
    assert [r.get("no_stand", False) for r in rows] == [False, False, True, False, False, False]
    fr = update._Frame.__new__(update._Frame)  # the exception's own test
    fr.unconfirmed, fr.detail = "too_close", "conflict_unsupported"
    fr.work = type("W", (), {"trust": "metric"})()  # type: ignore[assignment]
    monkeypatch.setattr(update, "_has_depth", lambda fr: True)
    assert not update._too_close_content(fr)
    fr.detail = None
    assert update._too_close_content(fr)


def test_a_not_judged_frame_gives_no_height_evidence_from_its_own_depth(
        tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """map.md §2 (P30c): a monocular frame the floor check would not judge keeps its sent height
    for its own pose only: no camera-height evidence from its own depth."""
    from tests.unit.test_anchored import frame, new_map

    m = new_map(tmp_path)
    m.update([frame("a", base_pose(0.0, pitch_deg=0.0), p=None, depth=False, mono=1.3,
                    height=0.5)])
    n = len(m.scale.heights)
    monkeypatch.setattr(an.AnchoredMap, "floor_check",
                        lambda self, *a: an.FloorCheck(judged=False))
    m.update([frame("b", base_pose(30.0, pitch_deg=0.0), p=None, depth=False, mono=1.3,
                    height=0.5)])
    assert len(m.scale.heights) == n


def test_the_older_mapper_shows_a_too_close_conflict(
        old: Any, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """api.md §4 (P30c) on the older mapper: a frame registered within the distance limit whose
    pose conflicts with its prior is kept ``too_close`` with ``detail`` ``conflict_unsupported``
    (no too-close content, no stand)."""
    from oh_my_slam.service import update
    from tests.unit.test_service_api import CAMERA, WORLD, frames_of

    monkeypatch.setattr(update, "TOO_CLOSE_INLIERS", 10 ** 6)

    def pr(i: int, T: np.ndarray) -> dict[str, Any]:
        P = T.copy()
        if i == 1:
            P[:3, 3] += [0.0, 0.0, 0.5]
        return {"trust": "metric", "T_map_cam": P.tolist(), "sigma_pos_m": 0.01,
                "sigma_rot_deg": 0.5}

    frames, files, _ = frames_of(old, 4, prior=pr)
    old.backend.add(files["img1"], WORLD[1], depth_m=0.15)
    job = old.update("m", frames, files, {"cam": CAMERA})
    i = job["inputs"][1]
    assert (i["status"], i["reason"], i["detail"]) == ("kept_at_prior", "too_close",
                                                       "conflict_unsupported")
    rows = {r["name"]: r for r in old.service.store.version("m")["frames"]}
    assert rows[i["name"]].get("no_stand") is True and not rows[i["name"]]["content"]


def test_floor_not_judged_on_a_registered_or_provisional_frame(tmp_path: Any) -> None:
    """api.md §4 ``detail`` (oms/2.5): ``floor_not_judged`` on a ``registered`` or
    ``provisional`` frame whose keyframe the floor check did not judge; never on a kept one."""
    from oh_my_slam.service import prior_first
    from tests.unit.test_anchored import frame, new_map

    m = new_map(tmp_path)
    pl = m.update([frame("a", base_pose(0.0))])[0]
    kf = m.keyframe(pl.keyframe or "")
    assert kf is not None
    kf.floor_unjudged = True
    fr = type("F", (), {})()
    fr.work = type("W", (), {"name": "a", "trust": "metric"})()
    fr.scene_m = fr.duplicate_of = None
    for status, want in (("registered", "floor_not_judged"), ("provisional", "floor_not_judged"),
                         ("kept_at_prior", None)):
        row = prior_first._input(fr, dataclasses.replace(pl, status=status), m)  # type: ignore[arg-type]
        assert row["detail"] == want, status


# -- P30d: a not-judged keyframe's depth lifts no later frame -------------------------------------


def test_an_unjudged_keyframe_lifts_no_later_frame(tmp_path: Any,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    """map.md §2 "A floor_not_judged keyframe as a reference" (P30d): the half-scale sweep's s5
    and s6 stay references for matching and pose, but their depth lifts no later frame (no
    keypoint depth: their matches are rays), so the map's k and its evidence equal those of the
    same sweep whose s5 and s6 carry no depth at all; before, the frames lifted on their
    half-scale depth took it as their own (monocular residuals at log 0.5, a wider
    ``scale_sigma_rel``)."""
    from oh_my_slam.mapping.anchored import AnchoredMap
    from tests.unit.test_round14 import frame as f14
    from tests.unit.test_round16 import _half_scale_sweep

    def sweep(root: Any, depthless: set[str]) -> AnchoredMap:
        m = AnchoredMap(root)
        m.update([f14("a0", base_pose(0), depth=None, mono=1.3)])
        pls = m.update([f14(f"s{k}", base_pose(30.0 * k + 15.0), depth=None,
                            mono=None if f"s{k}" in depthless else
                            (0.65 if f"s{k}" in {"s5", "s6"} else 1.3)) for k in range(12)])
        assert all(p.status == "registered" for p in pls)
        return m

    a, half = _half_scale_sweep(tmp_path / "a")
    b = sweep(tmp_path / "b", half)
    assert a.scale.k == pytest.approx(b.scale.k, rel=1e-9)
    assert a.scale.depth_residuals.keys() == b.scale.depth_residuals.keys()
    for kind, got in a.scale.depth_residuals.items():
        # (the same evidence: the guided matching's radius alone differs, by the stored
        # median depth of s5 and s6, a few rays of s7 at 1e-5 in the log residual)
        assert np.allclose(got, b.scale.depth_residuals[kind], rtol=0.0, atol=1e-4)
    unjudged = [k for k in a.keyframes if k.name in half]
    assert unjudged and all(k.floor_unjudged and not k.lifts() and k.usable() for k in unjudged)
    feats = an.features(f14("q", base_pose(170.0), depth=None, mono=1.3).rgb)
    corr = a._correspondences(feats, unjudged)
    assert len(corr.uv) and not np.isfinite(corr.X).any()  # rays only
    monkeypatch.setattr(an.Keyframe, "lifts", lambda self: True)  # as before P30d
    c, _ = _half_scale_sweep(tmp_path / "c")
    # the frames lifted on s5's and s6's half-scale depth took it as their own (log 0.5): the
    # skew of the monocular residuals and of scale_sigma_rel
    assert any(r < -0.5 for r in c.scale.depth_residuals["mono"])
    assert not any(r < -0.5 for r in a.scale.depth_residuals["mono"])
    assert c.scale.sigma_rel > a.scale.sigma_rel


def test_locate_takes_no_depth_from_an_unjudged_keyframe() -> None:
    """api.md §6 (P30d): a keyframe the floor check did not judge gives locate its features
    without its depth (``depth_unused`` row: no keypoint points), so a frame whose only depth
    references are such keyframes is located from the images alone, with the without-depth
    noise floor (0.15 m), never the 8 mm floor of a depth reference."""
    from oh_my_slam.mapping import relocalize, sigmas
    from oh_my_slam.service.store import DEPTH_UNUSED, depth_unused

    assert depth_unused(True) == {DEPTH_UNUSED: True} and depth_unused(False) == {}
    room = Room()
    Kr = Intrinsics(320.0, 320.0, 320.0, 240.0, 640, 480)
    T = base_pose(0.0)
    rgb, z = render(room, T, Kr)
    uv, desc = relocalize.extract(np.asarray(rgb.mean(axis=2), np.uint8))
    none = relocalize.keyframe_points(uv, Kr.K(), None, T)
    assert not np.isfinite(none).any()
    kf = relocalize.Keyframe("u", T, Kr.K(), uv, desc, none, 0.05, 1.0, False)
    q_rgb, _ = render(room, base_pose(8.0), Kr)
    quv, qdesc = relocalize.extract(np.asarray(q_rgb.mean(axis=2), np.uint8))
    q = relocalize.Query(quv, qdesc, Kr.K(), 640, 480)
    out = relocalize.locate(q, relocalize.Index([kf]))
    assert out.T is not None and out.sigma_pos_m >= sigmas.floor_of(False, None)[0] - 1e-9
    assert sigmas.floor_of(False, None)[0] == pytest.approx(0.15)
    assert sigmas.floor_of(True, None)[0] < 0.05
    # with that keyframe's depth (a depth reference), the 8 mm floor applies
    pts = relocalize.keyframe_points(uv, Kr.K(), z, T)
    deep = relocalize.Keyframe("d", T, Kr.K(), uv, desc, pts, 0.01, 0.1, True)
    got = relocalize.locate(q, relocalize.Index([deep]))
    assert got.T is not None and got.sigma_pos_m < 0.1


def test_mapanything_anchors_take_no_depth_from_an_unjudged_keyframe(tmp_path: Any) -> None:
    """P30d on MapAnything's anchors (``prior_first._anchor_depth``): a stored keyframe the
    floor check did not judge anchors by its image and pose, without depth."""
    from oh_my_slam.service import prior_first
    from tests.unit.test_anchored import frame, new_map

    m = new_map(tmp_path)
    pl = m.update([frame("a", base_pose(0.0))])[0]
    m.save()
    kf = m.keyframe(pl.keyframe or "")
    assert kf is not None and prior_first._anchor_depth(m, kf)() is not None
    kf.floor_unjudged = True
    assert prior_first._anchor_depth(m, kf)() is None
    assert prior_first._unjudged(m, pl) == {"depth_unused": True}
    assert prior_first._unjudged(None, pl) == {}
