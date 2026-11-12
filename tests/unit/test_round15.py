"""Round 15 (spec P25a, P27, P28, P28a): the label choice never takes a label without a share;
the cycle rule on every pair, depth-confirmed ones included, with stored-to-stored pairs measured
from the stored images; frames whose pairs have no chain outside the pair limits; translational
twins separated by their parallax; the PnP refinement kept on its hypothesis (map.md §1, §2, §4,
§5). The scene: an empty rectilinear room whose walls repeat every metre (``Room(walls=
"periodic")``), seen by the client's narrow camera (52 deg) turning in place in 26.5 deg steps."""

from __future__ import annotations

import functools
import logging
import re
from typing import Any

import numpy as np
import pytest

from oh_my_slam.core.types import Intrinsics
from oh_my_slam.mapping import anchored as an
from oh_my_slam.mapping import standpoint as sp
from tests.synth.room import Room, base_pose, render
from tests.unit.test_service_prior_first import client  # noqa: F401

# -- P25a: the label choice ---------------------------------------------------------------------


def test_a_label_without_a_share_is_never_chosen() -> None:
    """map.md §5 "Label choice" (P25a): the engine's own choice stays (it need not be the best
    share), but one with no share (share 0, or no entry) gives way to the best-share label, on
    every version; then the client-label rule applies; the first ``label_scores`` entry always
    has a score above 0."""
    from oh_my_slam.service.objects import chosen_label, label_scores, with_label_scores

    votes = [{"label": "cup", "share": 0.6}, {"label": "mug", "share": 0.5},
             {"label": "bowl", "share": 0.3}, {"label": "jar", "share": 0.0}]
    assert chosen_label({"label": "jar", "detection_votes": votes}, []) == "cup"
    assert chosen_label({"label": "box", "detection_votes": votes}, []) == "cup"  # no entry
    assert chosen_label({"label": "bowl", "detection_votes": votes}, []) == "bowl"  # E stays
    assert chosen_label({"label": "box", "detection_votes": []}, ["box"]) == "box"  # no votes
    # E replaced by L, then a client label within the margin of L's share wins
    assert chosen_label({"label": "jar", "detection_votes": votes}, ["Mug"]) == "mug"
    # a client label the engine chose stays, even far behind the best share (P25a as written)
    far = [{"label": "cup", "share": 0.9}, {"label": "mug", "share": 0.1}]
    assert chosen_label({"label": "mug", "detection_votes": far}, ["mug"]) == "mug"
    # a client label with no share is never chosen either
    zero = [{"label": "cup", "share": 0.6}, {"label": "mug", "share": 0.0}]
    assert chosen_label({"label": "mug", "detection_votes": zero}, ["mug"]) == "cup"
    answer = {"objects": [{"label": "jar", "detection_votes": votes}]}
    got = with_label_scores(answer, [])
    assert got["objects"][0]["label"] == "cup" and "label_scores" not in got["objects"][0]
    for vocab in (["mug"], ["jar"], ["vase"]):
        rec = with_label_scores(answer, vocab)["objects"][0]
        assert rec["label_scores"][0]["score"] > 0.0
        assert rec["label_scores"][0]["label"] == rec["label"]
    assert label_scores({"label": "cup", "detection_votes": votes})[0]["score"] == 0.6


# -- the empty periodic room ---------------------------------------------------------------------

K = Intrinsics(655.8, 655.8, 320.0, 240.0, 640, 480)  # the client's camera: 52 deg wide
ROOM = Room(floor="tiles", boxes=[], walls="periodic", wall_period=1.0)
N = 8
POSES = [base_pose(-53.0 + 26.5 * i, -0.3, 0.37, 0.5, -0.0436, 0.0) for i in range(N)]


@functools.cache
def _shot(i: int) -> tuple[an.Features, np.ndarray, np.ndarray]:
    rgb, z = render(ROOM, POSES[i], K)
    return an.features(rgb), z.astype(np.float32), rgb


def _view(i: int, R: np.ndarray | None = None, depth: bool = True) -> sp.View:
    f, z, _ = _shot(i)
    T = POSES[i]
    return sp.View(f"f{i}", K, f, (T[:3, :3] if R is None else R).copy(), T[:3, 3].copy(),
                   False, i, depth=(lambda: (z, K)) if depth else None)


def _rel(i: int, j: int) -> np.ndarray:
    return np.linalg.inv(POSES[i]) @ POSES[j]


def _err(i: int, j: int, tv: sp.TwoView) -> float:
    return an.rot_deg(_rel(i, j)[:3, :3].T @ tv.R_ab)


def _yawed(yaw: dict[int, float]) -> list[sp.View]:
    out = []
    for i in range(N):
        R = None
        if i in yaw:
            R = an.exp_so3(np.array([0.0, 0.0, np.radians(yaw[i])])) @ POSES[i][:3, :3]
        out.append(_view(i, R))
    return out


@functools.cache
def _pairs() -> sp.Pairs:
    """The room's two-view fits, once (the tests copy them)."""
    pairs = sp.Pairs()
    sp.check(_yawed({}), pairs)
    return pairs


def _copy(pairs: sp.Pairs) -> sp.Pairs:
    out = sp.Pairs()
    out.fits.update(pairs.fits)
    return out


def test_a_symmetric_room_confirms_the_lattice_pose_by_depth() -> None:
    """The root cause (map.md §1 "Cycle consistency", P28): each view sees the floor and one
    flat wall, so the 53 deg pair's largest consensus, a pose turned by 90 deg and shifted,
    agrees with both depth images exactly (nothing seen through), while the true pose shares no
    view (fewer than ``MIN_AGREEING`` points either way): the depth test alone measures it."""
    tv = _copy(_pairs()).get(_view(0), _view(2))
    assert tv is not None and tv.measurable and tv.depth and tv.t_m is not None
    assert abs(_err(0, 2, tv) - 90.0) < 2.0
    z0, z2 = _shot(0)[1], _shot(2)[1]
    A = an.DepthRef(np.eye(4), z0, K.K())
    B = an.DepthRef(an.pose(tv.R_ab, tv.t_m), z2, K.K())
    ab, ba = an.depth_counts(A, B, an.SENSOR_DEPTH_TOL), an.depth_counts(B, A, an.SENSOR_DEPTH_TOL)
    assert ab.through == 0 and ba.through == 0 and ab.passes and ba.passes
    true = an.DepthRef(_rel(0, 2), z2, K.K())
    assert an.depth_counts(A, true, 0.1).agree < an.MIN_AGREEING
    assert an.depth_counts(true, A, 0.1).agree < an.MIN_AGREEING


def test_the_cycle_rule_judges_depth_confirmed_pairs() -> None:
    """map.md §1 "Cycle consistency" (P28): the cycle rule applies to depth-confirmed pairs
    too, so the room's lattice pair leaves the rotation check (90 deg off its chain of adjacent
    pairs) and the true rotations pass; a frame committed 15 or 90 deg off is still
    ``rotation_inconsistent``."""
    pairs = _copy(_pairs())
    bad, rows = sp.check(_yawed({}), pairs)
    assert not bad and rows and all(r["difference_deg"] <= sp.PAIR_MAX_DEG for r in rows)
    assert ("f0", "f2") not in {(r["a"], r["b"]) for r in rows}
    for deg in (15.0, 90.0):
        bad, rows = sp.check(_yawed({3: deg}), pairs)
        assert bad and max(r["difference_deg"] for r in rows) > 10.0


# -- P28a: frames whose pairs have no chain ------------------------------------------------------


def _tv(inliers: int, deg: float, depth: bool = True) -> sp.TwoView:
    return sp.TwoView(an.exp_so3(np.array([0.0, 0.0, np.radians(deg)])), 5.0, inliers, 0.5,
                      0.01, True, depth)


def _line(n: int, committed: dict[int, float], fits: dict[tuple[str, str], sp.TwoView]
          ) -> tuple[bool, list[dict[str, Any]]]:
    f, _, _ = _shot(0)
    R = base_pose(0.0)[:3, :3]
    views = [sp.View(f"v{k}", K, f, an.exp_so3(np.array([0.0, 0.0, np.radians(
        committed.get(k, 0.0))])) @ R, np.array([0.1 * k, 0.0, 0.0]), False, k) for k in range(n)]
    pairs = sp.Pairs()
    pairs.fits.update(fits)
    return sp.check(views, pairs)


def test_a_two_frame_jobs_single_unchained_pair_is_still_judged() -> None:
    """map.md §1 "Cycle consistency" (P28a): a frame's only measurable pair, without a chain,
    stays measurable and is judged against the committed rotations (the single pair of a
    two-frame job); a frame with two or more measurable pairs none of which has a chain has
    them outside the pair limits, depth-confirmed or not, and never rejects the job on its own
    (a pair one of whose frames puts it outside is outside); a frame with a chained pair keeps
    all of its pairs."""
    bad, rows = _line(2, {1: 20.0}, {("v0", "v1"): _tv(300, 0.0)})
    assert bad and [(r["a"], r["b"]) for r in rows] == [("v0", "v1")]
    three: dict[tuple[str, str], Any] = {("v0", "v1"): _tv(300, 0.0), ("v1", "v2"): _tv(250, 0.0),
                                         ("v0", "v2"): None}
    for depth in (True, False):
        fits = {k: None if v is None else _tv(v.inliers, 0.0, depth) for k, v in three.items()}
        bad, rows = _line(3, {2: 20.0}, fits)
        assert not bad and rows == []  # v1's two pairs have no chain: outside, both
    # a cycle: the weakest pair (v0-v2) has a chain, so v0 and v2 keep theirs; v1's two pairs
    # have none, so v1 puts them outside (though v0 and v2 keep them); v0-v2 is judged
    cyc = {**three, ("v0", "v2"): _tv(200, 0.0)}
    bad, rows = _line(3, {2: 20.0}, cyc)
    assert bad and [(r["a"], r["b"]) for r in rows] == [("v0", "v2")]
    bad, rows = _line(3, {}, cyc)
    assert not bad and len(rows) == 1


# -- translational twins (P28) -------------------------------------------------------------------


def _twins(monkeypatch: pytest.MonkeyPatch, poses: list[np.ndarray]) -> sp.TwoView | None:
    """``two_view`` of the turn in place f0-f1 whose hypotheses (its largest consensus, repeated)
    the depth images all confirm, at the metric relative poses ``poses``."""
    a, b = _view(0), _view(1)
    e = an.epipolar(a.feats, a.K, b.feats, b.K)
    assert e is not None
    ok = an.DepthCounts(100, 100, 0, 0, 0)
    monkeypatch.setattr(sp, "depth_check", lambda *x, **k: (True, ok, ok))
    monkeypatch.setattr(sp, "epipolar_hypotheses", lambda *x, **k: iter([e] * len(poses)))
    it = iter(poses)
    monkeypatch.setattr(sp, "depth_pose",
                        lambda *x: sp.DepthPose(next(it).copy(), 50, 60, "ransac", 0.0))
    return sp.two_view(a, b)


def test_translational_twins_are_separated_by_their_parallax(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """map.md §1 "Translational twins" (P28): two depth-confirmed hypotheses sharing the
    rotation, 1 m apart (a shift along the plain wall): the pair takes the one whose predicted
    two-view parallax matches its measured one — the turn in place, whichever comes first;
    when the parallax cannot separate them, the pair is ambiguous; two at one position are one
    geometry measured twice."""
    assert an.TWIN_MIN_M == 0.3
    true = _rel(0, 1)
    shifted = true.copy()
    shifted[:3, 3] += POSES[0][:3, :3].T @ np.array([0.0, 1.0, 0.0])  # 1 m along the wall
    for order, want in (([true, shifted], 1), ([shifted, true], 2)):
        tv = _twins(monkeypatch, order)
        assert tv is not None and tv.measurable and not tv.ambiguous and tv.hyp == want
        assert tv.t_m is not None and float(np.linalg.norm(tv.t_m - true[:3, 3])) < 0.01
        assert tv.measured_px is not None and tv.measured_px < 3.0
        assert tv.predicted_px is not None and an.parallax_matches(tv.predicted_px,
                                                                   tv.measured_px)
    monkeypatch.setattr(sp, "predicted_parallax", lambda *x: 1.0)
    tv = _twins(monkeypatch, [true, shifted])
    assert tv is not None and tv.ambiguous and not tv.measurable
    near = true.copy()
    near[:3, 3] += 0.05
    tv = _twins(monkeypatch, [true, near])
    assert tv is not None and tv.measurable and tv.hyp == 1


def test_the_twin_rules_pieces() -> None:
    """``predicted_parallax``: about 0 for a turn in place, tens of pixels for a 1 m shift
    across a wall 2-3 m away (the rotation-only fit absorbs most of a sideways shift in front
    of a plane), 0 without translation, None with fewer than three points in front;
    ``parallax_matches`` (README "Translational twins": 2 x (1 px + 25 %)); ``twin_pick``,
    ``prior_pick`` and ``one_position``."""
    a = _view(0)
    e = an.epipolar(a.feats, a.K, _view(1).feats, K)
    assert e is not None
    z = _shot(0)[1]
    true = _rel(0, 1)
    turn = an.predicted_parallax(e.ua, K, z, K, true, K.fx)
    assert turn is not None and turn < 3.0
    shifted = true.copy()
    shifted[:3, 3] += np.array([1.0, 0.0, 0.0])
    far = an.predicted_parallax(e.ua, K, z, K, shifted, K.fx)
    assert far is not None and far > 20.0 and not an.parallax_matches(far, turn)
    still = true.copy()
    still[:3, 3] = 0.0
    assert an.predicted_parallax(e.ua, K, z, K, still, K.fx) == 0.0
    behind = true.copy()
    behind[:3, 3] = [0.0, 0.0, 50.0]
    assert an.predicted_parallax(e.ua, K, z, K, behind, K.fx) is None
    assert an.parallax_matches(3.4, 1.0) and not an.parallax_matches(3.6, 1.0)
    assert an.parallax_matches(250.0, 300.0) and not an.parallax_matches(None, 1.0)
    c = [np.zeros(3), np.array([1.0, 0.0, 0.0]), np.array([1.1, 0.0, 0.0])]
    assert an.twin_pick(1.0, [1.2, 300.0, None], c) == 0
    assert an.twin_pick(300.0, [1.2, 300.0, 290.0], c) == 1  # two at one position
    assert an.twin_pick(1.0, [1.2, 1.5, None], c) is None  # both match
    assert an.twin_pick(1.0, [300.0, 300.0, None], c) is None  # none matches
    assert an.prior_pick(c, np.array([0.02, 0.0, 0.0]), 0.02) == 0
    assert an.prior_pick(c, np.array([1.05, 0.0, 0.0]), 0.05) == 1
    assert an.prior_pick(c, np.array([0.5, 0.0, 0.0]), 0.5) is None  # its sigma takes both
    assert an.prior_pick(c, np.array([5.0, 0.0, 0.0]), 0.1) is None
    assert an.one_position([], c) is None


def test_mapanythings_gate_takes_the_twin_the_parallax_picks() -> None:
    """``standpoint.places`` (map.md §2 step 2, P28): a candidate pose that the reference's
    depth confirmed is placed by it unless the pair is ambiguous, or the pair's confirmed fit
    shares its rotation ``TWIN_MIN_M`` or more away and the parallax picks the fit (or cannot
    separate the two); another rotation is the symmetric-scene rule's, and a pair without a fit
    or without a depth fit places it."""
    import dataclasses

    a, b = _view(0), _view(1)
    pairs = _copy(_pairs())
    tv = pairs.get(a, b)
    assert tv is not None and tv.measurable and tv.t_m is not None and tv.ua is not None
    true = _rel(0, 1)
    assert sp.places(a, b, true, pairs)
    far = true.copy()
    far[:3, 3] += np.array([1.0, 0.0, 0.0])
    assert not sp.places(a, b, far, pairs)  # the parallax picks the pair's fit
    turned = far.copy()
    turned[:3, :3] = an.exp_so3(np.array([0.0, np.radians(90.0), 0.0])) @ true[:3, :3]
    assert sp.places(a, b, turned, pairs)
    key = ("f0", "f1")
    pairs.fits[key] = dataclasses.replace(tv, t_m=far[:3, 3].copy(), predicted_px=400.0)
    assert sp.places(a, b, true, pairs)  # the fit is the twin: the candidate is picked
    pairs.fits[key] = dataclasses.replace(tv, t_m=far[:3, 3].copy(), predicted_px=tv.measured_px)
    assert not sp.places(a, b, true, pairs)  # both match: cannot separate
    nodepth = _view(0, depth=False)
    assert not sp.places(nodepth, b, true, pairs)  # the candidate predicts nothing
    pairs.fits[key] = dataclasses.replace(tv, ambiguous=True, confirmed=False)
    assert not sp.places(a, b, true, pairs) and sp.ambiguous(a, b, pairs)
    pairs.fits[key] = dataclasses.replace(tv, confirmed=False)
    assert sp.places(a, b, far, pairs)
    pairs.fits[key] = None
    assert sp.places(a, b, far, pairs) and not sp.ambiguous(a, b, pairs)
    assert sp.places(_view(0), _view(1), true)  # fitted on its own


def test_the_geometric_gate_separates_translational_twins(
        tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """``AnchoredMap.depth_confirms`` (map.md §2 step 2, P28): a candidate pose that a
    reference confirms with a twin sharing its rotation 1 m away: a prior sent for the frame
    decides first, else the parallax (``_twin_parallax``); the pose picked is placed, a twin
    picked over it replaces it once refined on all the frame's correspondences (when it holds),
    and nothing separating them places it in neither. Each test is logged (L4)."""
    from tests.unit.test_anchored import frame, new_map

    am = new_map(tmp_path)
    am.update([frame("a", base_pose(0.0), p=None), frame("b", base_pose(20.0), p=None)])
    shift = np.array([0.0, 1.0, 0.0])

    def amb(self: Any, f: Any, T: Any, own: Any, corr: Any, cands: Any, j: int, ref: Any,
            twin: Any = None) -> float | None:
        if twin is not None:
            twin.append(an.Solve(T[:3, :3].copy(), T[:3, 3] + shift, np.eye(6) * 1e-6,
                                 np.ones(len(corr.uv), bool), 0.5))
        return None

    picks = {"m": 0.5, "p": [0.5, 300.0]}
    monkeypatch.setattr(an.AnchoredMap, "_ambiguous", amb)
    monkeypatch.setattr(an.AnchoredMap, "_twin_parallax",
                        lambda self, *x: (picks["m"], list(picks["p"])))
    pl = am.update([frame("c", base_pose(10.0), p=None)])[0]
    assert pl.status == "registered"
    picks["p"] = [0.5, 0.5]  # nothing separates them
    assert am.update([frame("d", base_pose(5.0), p=None)])[0].status == "rejected"
    picks["p"] = [300.0, 0.5]  # the twin: refined on all the matches it does not hold
    assert am.update([frame("e", base_pose(15.0), p=None)])[0].status == "rejected"
    picks["m"] = None  # type: ignore[assignment]  # no fit to measure: in neither
    assert am.update([frame("h", base_pose(7.0), p=None)])[0].status == "rejected"
    # with priors (in the map's frame): the prior at the true pose decides before the parallax
    # (which would pick the twin)
    picks["m"], picks["p"] = 0.5, [300.0, 0.5]
    pm = new_map(tmp_path / "p")
    pm.update([frame("a", base_pose(0.0)), frame("b", base_pose(20.0))])
    pl = pm.update([frame("g", base_pose(12.0))])[0]
    assert pl.status == "registered" and pl.T is not None
    assert float(np.linalg.norm(pl.T[:3, 3] - base_pose(12.0)[:3, 3])) < 0.1


def test_a_twin_picked_over_the_pose_replaces_it(tmp_path: Any,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """``AnchoredMap._twin``: the twin the parallax picks is refined on all the frame's
    correspondences and placed when it stays that twin; one the refinement moves away, or the
    depth rejects, is no placement; ``_twin_parallax`` measures the frame's pair with the
    reference (None without features or a fit)."""
    from tests.unit.test_anchored import frame, new_map

    am = new_map(tmp_path)
    am.update([frame("a", base_pose(0.0), p=None)])
    f = frame("c", base_pose(10.0), p=None)
    own = an.depth_image(f)
    assert own is not None
    cands = list(am.keyframes)
    ref = am._grid(cands[0])
    assert ref is not None
    T = base_pose(10.0)
    n = 40
    corr = an._Corr(np.zeros((n, 2)), np.ones((n, 3)), np.ones((n, 3)), np.zeros(n, np.intp))

    def solve(c: np.ndarray, k: int = n) -> an.Solve:
        inl = np.zeros(n, bool)
        inl[:k] = True
        return an.Solve(T[:3, :3].copy(), c, np.eye(6) * 1e-6, inl, 0.5)

    s, twin = solve(T[:3, 3].copy()), solve(T[:3, 3] + np.array([1.0, 0.0, 0.0]))
    monkeypatch.setattr(an.AnchoredMap, "_twin_parallax", lambda self, *x: (0.5, [300.0, 0.5]))
    got = {"r": twin}
    monkeypatch.setattr(an, "refine_pose", lambda *x, **k: (got["r"], None))
    monkeypatch.setattr(an, "depth_confirmed", lambda *x, **k: True)
    out = am._twin(f, s, [twin], own, 0, ref, corr, cands, None, None)
    assert out is not None and out[0] is twin and out[1] is not None and len(out[1][0]) == n
    got["r"] = solve(T[:3, 3].copy())  # back at the pose: not that twin
    assert am._twin(f, s, [twin], own, 0, ref, corr, cands, None, None) is None
    got["r"] = solve(twin.c, 3)  # too few inliers
    assert am._twin(f, s, [twin], own, 0, ref, corr, cands, None, None) is None
    monkeypatch.undo()
    feats = an.features(f.rgb)
    m, preds = am._twin_parallax(f, feats, cands[0], own, [s, twin])
    assert m is not None and len(preds) == 2
    assert am._twin_parallax(f, None, cands[0], own, [s]) == (None, [])
    empty = an.Features(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32))
    assert am._twin_parallax(f, empty, cands[0], own, [s]) == (None, [])


def test_the_geometric_engine_collects_its_twin_candidates(
        tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """``AnchoredMap._ambiguous``: a sequential PnP pose within ``SYMMETRIC_MIN_DEG`` of the
    pose but ``TWIN_MIN_M`` or more from it that the depth confirms is a translational twin
    (collected, not ambiguous); only collected when asked for."""
    from tests.unit.test_anchored import frame, new_map
    from tests.unit.test_round13 import _cloud

    am = new_map(tmp_path)
    am.update([frame("a", base_pose(0.0), p=None)])
    f = frame("c", base_pose(10.0), p=None)
    T = base_pose(10.0)
    moved = T.copy()
    moved[:3, 3] += np.array([0.0, 1.0, 0.0])

    def corr(*groups: tuple[np.ndarray, np.ndarray]) -> Any:
        X = np.vstack([g[0] for g in groups])
        uv = np.vstack([g[1] for g in groups])
        d = (X - T[:3, 3]) / np.linalg.norm(X - T[:3, 3], axis=1)[:, None]
        return an._Corr(uv, X, d, np.zeros(len(X), np.intp))

    own = np.ones((4, 4), np.float32)
    ref = an.DepthRef(T, own, f.K.K())
    monkeypatch.setattr(an, "depth_confirmed", lambda *x, **k: True)
    both = corr(_cloud(T, 80, 0, f.K), _cloud(moved, 80, 1, f.K))
    twin: list[an.Solve] = []
    assert am._ambiguous(f, T, own, both, list(am.keyframes), 0, ref, twin) is None
    assert len(twin) == 1 and float(np.linalg.norm(twin[0].c - moved[:3, 3])) < 0.05
    assert am._ambiguous(f, T, own, both, list(am.keyframes), 0, ref) is None


# -- P27: stored-to-stored pairs -------------------------------------------------------------------


def test_stored_pairs_are_measured_from_their_images_and_cached(tmp_path: Any) -> None:
    """map.md §1 "Stored-to-stored pairs" (P27): a stored pair a chain needs is measured from
    the two keyframes' own features and depth (``Pairs.stored``, timed in ``stored_s``) and
    kept in the map's cache by keyframe and content: another job's ``Pairs`` reuses it without
    measuring; changed content is measured again. The map saves the cache with itself, only for
    the keyframes it still holds."""
    from tests.unit.test_anchored import frame, new_map

    s1 = sp.View("s1", K, _shot(0)[0], POSES[0][:3, :3], POSES[0][:3, 3], True, kid="k1",
                 depth=lambda: (_shot(0)[1], K))
    s2 = sp.View("s2", K, _shot(1)[0], POSES[1][:3, :3], POSES[1][:3, 3], True, kid="k2",
                 depth=lambda: (_shot(1)[1], K))
    cache: dict[str, Any] = {}
    pairs = sp.Pairs(cache)
    tv = pairs.stored(s1, s2)
    assert tv is not None and tv.measurable and pairs.stored_s > 0.0 and len(cache) == 1
    assert pairs.stored(s2, s1) is not None and len(cache) == 1  # this job's fit, reversed
    again = sp.Pairs(cache)
    got = again.stored(s1, s2)
    assert got is not None and again.stored_s == 0.0
    assert np.allclose(got.R_ab, tv.R_ab) and got.inliers == tv.inliers and got.ua is None
    other = sp.View("s2", K, _shot(2)[0], POSES[1][:3, :3], POSES[1][:3, 3], True, kid="k2")
    third = sp.Pairs(cache)
    third.stored(s1, other)
    assert len(cache) == 2 and third.stored_s > 0.0  # its content changed: measured again
    assert sp.Pairs().stored(s1, sp.View("e", K, an.Features(
        np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)), np.eye(3), np.zeros(3),
        True, kid="k3")) is None
    none = sp.Pairs(cache)
    none.fits[("s1", "x")] = None
    assert none.stored(s1, sp.View("x", K, _shot(0)[0], np.eye(3), np.zeros(3), True)) is None
    # the map's cache
    am = new_map(tmp_path)
    am.update([frame("a", base_pose(0.0), p=None), frame("b", base_pose(20.0), p=None)])
    ids = [k.id for k in am.keyframes]
    am.pair_cache[f"{ids[0]}:x|{ids[1]}:y"] = tv.to_json()
    am.pair_cache[f"{ids[0]}:x|gone:y"] = None
    am.save()
    back = an.AnchoredMap.load(am.root)
    assert list(back.pair_cache) == [f"{ids[0]}:x|{ids[1]}:y"]
    rt = sp.TwoView.from_json(back.pair_cache[f"{ids[0]}:x|{ids[1]}:y"])
    assert np.allclose(rt.R_ab, tv.R_ab) and rt.hyp == tv.hyp and rt.agreement == tv.agreement
    plain = sp.TwoView(np.eye(3), 1.0, 100, 0.5, 0.01)
    assert sp.TwoView.from_json(plain.to_json()).t_m is None


# -- fix B: the PnP refinement stays on its hypothesis -----------------------------------------------


def test_pnp_refinement_stays_on_its_hypothesis(monkeypatch: pytest.MonkeyPatch) -> None:
    """``depth_pose``: on the room's lattice-ambiguous lifted points the PnP RANSAC can lock onto
    another consensus; the pose stays within ``PNP_MAX_STEP_DEG`` of the hypothesis's Sampson
    rotation (refined locally from it, else the seed itself), never another hypothesis."""
    assert sp.PNP_MAX_STEP_DEG == 5.0
    a, b = _view(1), _view(3)
    e = an.epipolar(a.feats, a.K, b.feats, b.K)
    assert e is not None
    fit = sp._fit(a, b, e)
    assert fit is not None
    da, db = a.depth_image(), b.depth_image()
    assert da is not None and db is not None
    dp = sp.depth_pose(a, b, e, fit.R_ab, da, db)
    assert dp is not None and an.rot_deg(fit.R_ab.T @ dp.T[:3, :3]) <= sp.PNP_MAX_STEP_DEG
    # a RANSAC far off: the local refinement from the seed
    real = sp.cv2.solvePnPRansac

    def far(*x: Any, **k: Any) -> Any:
        good, rv, tv, inl = real(*x, **k)
        return good, np.array([[0.0], [np.radians(90.0)], [0.0]]), tv, inl

    monkeypatch.setattr(sp.cv2, "solvePnPRansac", far)
    dp = sp.depth_pose(a, b, e, fit.R_ab, da, db)
    assert dp is not None and dp.mode == "local" and dp.ransac_deg > 45.0
    assert an.rot_deg(fit.R_ab.T @ dp.T[:3, :3]) <= sp.PNP_MAX_STEP_DEG
    # the local refinement leaves it too: the seed
    monkeypatch.setattr(sp.cv2, "solvePnPRefineLM",
                        lambda *x: (np.array([[0.0], [np.radians(90.0)], [0.0]]), x[5]))
    dp = sp.depth_pose(a, b, e, fit.R_ab, da, db)
    assert dp is not None and dp.mode == "seed" and np.allclose(dp.T[:3, :3], fit.R_ab)
    # a seed nothing reprojects near: the seed
    off = an.exp_so3(np.array([0.0, np.radians(150.0), 0.0])) @ fit.R_ab
    dp = sp.depth_pose(a, b, e, off, da, db)
    assert dp is not None and dp.mode == "seed" and dp.inliers < sp.PNP_MIN_POINTS


# -- the logging (L1-L5) -----------------------------------------------------------------------------

L1 = re.compile(r"two-view f(\d)-f(\d) hyp (\d)/(\d): (\d+) inliers, overlap [\d.]+, sampson "
                r"[\d.]+ deg, pnp (?:[\d.]+ deg \((\d+)/(\d+) pnp inliers(?:, \w+ from [\d.]+ "
                r"deg)?\), t \[[-+\d. ]+\] m \|t\| [\d.]+(?:, a>b agree \d+ through \d+ occluded "
                r"\d+ outside \d+ of \d+ = [\d.]+, b>a agree \d+ through \d+ occluded \d+ "
                r"outside \d+ of \d+ = [\d.]+)?|none deg, t none): (confirmed|rejected agreement|"
                r"rejected agreeing|rejected no pnp|skipped overlap)$")
L3 = re.compile(r"rotation check f\d-f\d: [\d.]+ deg \(measured [\d.]+, committed [\d.]+, \d+ "
                r"inliers, overlap [\d.]+, parallax [\d.]+ px, hyp \d, \|t\| (?:[\d.]+|n/a) m, "
                r"agreement (?:[\d.]+/[\d.]+|n/a), chain (?:[\d.]+|n/a) deg, (?:depth-confirmed|"
                r"cycle rule)\)")


def test_two_view_logs_every_depth_hypothesis(caplog: pytest.LogCaptureFixture,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """The service log (README "Depth confirmation", L1-L3): one line per depth-tested
    hypothesis with its PnP, shift and both ways' agreement counts and verdict; one verdict per
    pair; the rotation check's line with the hypothesis, shift, agreement and chain angle."""
    caplog.set_level(logging.INFO, logger="oh_my_slam.mapping.standpoint")
    sp.two_view(_view(0), _view(1))
    hyp = [m for r in caplog.records for m in [L1.match(r.getMessage())] if m]
    assert hyp and hyp[0].group(8) == "confirmed"
    assert any(r.getMessage().startswith("two-view f0-f1: chose hyp 1 at ")
               for r in caplog.records)
    caplog.clear()
    sp.two_view(_view(0), _view(3))  # 79.5 deg: its hypotheses fail
    verdicts = {m.group(8) for r in caplog.records for m in [L1.match(r.getMessage())] if m}
    assert verdicts and "confirmed" not in verdicts
    caplog.clear()
    monkeypatch.setattr(sp.cv2, "solvePnPRansac", lambda *x, **k: (False, None, None, None))
    sp.two_view(_view(0), _view(1))
    msgs = [r.getMessage() for r in caplog.records]
    assert any(L1.match(m) and m.endswith("rejected no pnp") for m in msgs)
    assert "two-view f0-f1: none confirmed" in msgs
    monkeypatch.undo()
    caplog.clear()
    sp.check(_yawed({}), _copy(_pairs()))
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("rotation")]
    assert lines and all(L3.match(m) for m in lines)
    assert any("chain n/a" not in m for m in lines)


def test_a_frame_whose_pairs_have_no_chain_never_rejects_the_job() -> None:
    """The client's W2 structure (map.md §1, P28a): frame 2's adjacent pairs unconfirmed, its
    only measurable pair is the lattice pair f0-f2, which has no chain of stronger pairs and so
    survives the cycle rule; but frame 0's two pairs (f0-f1, f0-f2) have none either, so they
    are outside the pair limits: the true rotations pass. Without f0-f1, the lattice pair is
    both frames' only pair: judged, ``rotation_inconsistent`` (the accepted degenerate case)."""
    import dataclasses

    base = _copy(_pairs())
    for k in (("f1", "f2"), ("f2", "f3")):
        tv = base.get(_view(int(k[0][1])), _view(int(k[1][1])))
        assert tv is not None
        base.fits[k] = dataclasses.replace(tv, confirmed=False)
        base.fits[(k[1], k[0])] = dataclasses.replace(tv.reversed(), confirmed=False)
    bad, rows = sp.check(_yawed({}), base)
    assert not bad and ("f0", "f2") not in {(r["a"], r["b"]) for r in rows}
    alone = _copy(base)
    tv = alone.get(_view(0), _view(1))
    assert tv is not None
    alone.fits[("f0", "f1")] = dataclasses.replace(tv, confirmed=False)
    alone.fits[("f1", "f0")] = dataclasses.replace(tv.reversed(), confirmed=False)
    bad, rows = sp.check(_yawed({}), alone)
    assert bad and ("f0", "f2") in {(r["a"], r["b"]) for r in rows}


def _room_frames(c: Any, idx: list[int]) -> tuple[list[dict[str, Any]], dict[str, bytes],
                                                  dict[str, Any]]:
    import hashlib

    from tests.synth.room import jpeg, png_depth

    cam = {"fx": K.fx, "fy": K.fy, "cx": K.cx, "cy": K.cy, "width": K.width, "height": K.height}
    frames, files = [], {}
    for i in idx:
        _, z, rgb = _shot(i)
        img = jpeg(rgb)
        c.backend.mono_depth[hashlib.sha256(img).hexdigest()] = z
        files[f"f{i}"], files[f"f{i}d"] = img, png_depth(z)
        frames.append({"name": f"f{i}", "image": f"f{i}", "depth": f"f{i}d", "camera_id": "cam",
                       "height_above_floor_m": 0.5})
    return frames, files, cam


def test_a_symmetric_room_turn_without_priors_commits(
        client: Any, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """The client's turn in an empty rectilinear room with walls repeating every metre (W2-like:
    f 655.8, 640 x 480, camera 0.5 m high, 26.5 deg steps), without priors: the job and its dry
    run commit (never ``rotation_inconsistent``), every frame they register at its true
    relative rotation, every rotation pair within the pair limit and the lattice pair out of
    the check (P28), the same pairs on both, and the job reports ``timings_s.stored_pairs``
    (oms/2.4). (The registration places few of these frames, as before: their PnP consensus is
    the walls' lattice, whose reference the depth gate rejects or finds ambiguous; the pair-level
    tests above check the whole turn.)"""
    from oh_my_slam.service import prior_first
    from tests.unit.test_service_prior_first import dry

    monkeypatch.setenv(prior_first.NONE_ENGINE_ENV, "geometric")
    frames, files, cam = _room_frames(client, list(range(N)))
    pred = dry(client, "m", frames, files, cameras={"cam": cam})
    job = client.update("m", frames, files, cameras={"cam": cam})
    assert job["state"] == "committed", job["quality"]
    q = job["quality"]
    assert "rotation_inconsistent" not in q["failed_checks"]
    assert all(r["difference_deg"] <= sp.PAIR_MAX_DEG for r in q["rotation_pairs"])
    assert ("f0", "f2") not in {(r["a"], r["b"]) for r in q["rotation_pairs"]}
    assert pred["quality"]["rotation_pairs"] == q["rotation_pairs"]
    assert job["timings_s"]["stored_pairs"] >= 0.0
    reg = [(i, POSES[k]) for k, i in enumerate(job["inputs"]) if i["status"] == "registered"]
    assert reg
    T0, P0 = np.array(reg[0][0]["T_map_cam"]), reg[0][1]
    for i, P in reg[1:]:
        rel = np.linalg.inv(T0) @ np.array(i["T_map_cam"])
        assert an.rot_deg((np.linalg.inv(P0) @ P)[:3, :3].T @ rel[:3, :3]) < 2.0, i["name"]


def test_a_job_measures_the_stored_pairs_its_chains_need(
        client: Any, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """map.md §1 "Stored-to-stored pairs" (P27), api.md §4 (oms/2.4): a second update without
    priors joining frames between stored keyframes measures the stored pairs their chains need
    from the stored images, reports the seconds as ``timings_s.stored_pairs`` (part of
    ``register``), and commits them with the map (``stored_pairs.json``) for the next jobs; a
    first update has none to measure (0)."""
    import json

    from oh_my_slam.service import prior_first
    from tests.unit.test_round11 import _turn_frames

    monkeypatch.setenv(prior_first.NONE_ENGINE_ENV, "geometric")
    measured: list[tuple[str, str]] = []
    real = sp.Pairs.stored

    def stored(self: sp.Pairs, a: sp.View, b: sp.View) -> Any:
        before = self.stored_s
        out = real(self, a, b)
        if self.stored_s > before:
            measured.append((a.name, b.name))
        return out

    monkeypatch.setattr(sp.Pairs, "stored", stored)
    fr, files = _turn_frames(client, [base_pose(20.0 * k) for k in range(4)], True, 0.0)
    job = client.update("m", fr, files, cameras={"cam": _cam()})
    assert job["state"] == "committed" and job["timings_s"]["stored_pairs"] >= 0.0
    measured.clear()
    fr, files = _turn_frames(client, [base_pose(10.0 + 20.0 * k) for k in range(3)], True, 0.0)
    for f in fr:
        f["name"] = "m" + f["name"]
    job = client.update("m", fr, files, cameras={"cam": _cam()})
    assert job["state"] == "committed", job["quality"]
    tm = job["timings_s"]
    assert measured and 0.0 < tm["stored_pairs"] <= tm["register"]
    path = client.service.store.mapper_dir("m") / an.PAIRS_JSON
    assert path.exists() and len(json.loads(path.read_text())) >= len(set(measured))


def _cam() -> dict[str, Any]:
    from tests.unit.test_service_prior_first import CAM

    return CAM


def test_gate_and_pair_edges(tmp_path: Any) -> None:
    """Edges: the geometric gate without correspondences takes the confirmed pose as it is (no
    other candidate to ask); a stored pair whose keyframes cannot share a view is not measured
    for a chain; a depth agreement nothing counts in has ratio 0 (and logs its counts)."""
    from tests.unit.test_anchored import frame, new_map

    am = new_map(tmp_path)
    am.update([frame("a", base_pose(0.0))])  # (a prior: the map frame is the poses')
    f = frame("c", base_pose(5.0), p=None)
    s = an.Solve(base_pose(5.0)[:3, :3], base_pose(5.0)[:3, 3], np.eye(6) * 1e-6,
                 np.ones(40, bool), 0.5)
    got = am.depth_confirms(f, s, np.zeros(40, np.intp), list(am.keyframes))
    assert got is not None and got[0] is s and got[1] is None
    mono = frame("m", base_pose(5.0), p=None, depth=False)
    assert am.depth_confirms(mono, s, np.zeros(40, np.intp), list(am.keyframes)) == (s, None)
    c = an.DepthCounts(10, 0, 0, 0, 10)
    assert c.ratio == 0.0 and not c.passes and c.text().endswith("of 10 = 0.000")
    feats = _shot(0)[0]
    R = POSES[0][:3, :3]
    views = [sp.View("s1", K, feats, R, np.zeros(3), True, kid="k1"),
             sp.View("s2", K, feats, R, np.array([20.0, 0.0, 0.0]), True, kid="k2"),
             sp.View("n", K, feats, R, np.array([10.0, 0.0, 0.0]), False, 0)]
    pairs = sp.Pairs()
    pairs.fits.update({("n", "s1"): _tv(500, 0.0), ("n", "s2"): _tv(300, 0.0)})
    cand = [(2, 0, _tv(500, 0.0)), (2, 1, _tv(300, 0.0))]
    assert sp.stored_pairs(views, pairs, cand) == [] and ("s1", "s2") not in pairs.fits
    assert [(i, j) for i, j, _ in sp.consistent(views, cand)] == [(2, 0), (2, 1)]


def test_a_twin_the_gate_picks_is_placed(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The callers of the gate (map.md §2 step 2, P28): a twin it picks over the image pose is
    the pose placed, with its correspondences (the frame's ``place``, and the hypothesis near a
    prior, ``_near_prior``); a hypothesis near the prior that the gate rejects is none."""
    from tests.unit.test_anchored import frame, new_map

    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0)), frame("a1", base_pose(30))])
    real = an.AnchoredMap.depth_confirms
    moved: list[an.Solve] = []

    def twin(self: Any, f: Any, s: an.Solve, *a: Any, **k: Any) -> Any:
        got = real(self, f, s, *a, **k)
        if not got:
            return got
        t = an.Solve(s.R, s.c + np.array([0.0, 0.0, 0.01]), s.cov, s.inliers, s.rms_px)
        moved.append(t)
        return t, (np.zeros((len(s.inliers), 2)),) * 3 + (np.zeros(len(s.inliers), np.intp),)

    monkeypatch.setattr(an.AnchoredMap, "depth_confirms", twin)
    T = base_pose(15)
    f = frame("n", T)
    feats = an.features(f.rgb)
    cands = m._candidates(f, T, None)
    corr = m._correspondences(feats, cands, (T, 0.02, 1.0), K_ANCH)
    got = m._near_prior(f, corr, cands, T, 0.02, 1.0, 10)
    assert got is not None and got[0] is moved[-1] and len(got[2][0]) == len(moved[-1].inliers)
    pl = m.place(f, feats, None)
    assert pl.status == "registered" and pl.align_T is not None
    assert np.allclose(pl.align_T[:3, 3], moved[-1].c)
    monkeypatch.setattr(an.AnchoredMap, "depth_confirms", lambda *a, **k: None)
    assert m._near_prior(f, corr, cands, T, 0.02, 1.0, 10) is None


def _k_anch() -> Intrinsics:
    from tests.unit.test_anchored import K as KA

    return KA


K_ANCH = _k_anch()
