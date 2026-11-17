"""oms/2.16: depth consistency rejects frames, not updates (map.md §4 "Resolution",
"Whole-update rejection", "Suspect stored keyframes"; api.md §4 "Depth consistency").

The decision (``standpoint.depth_resolve``) on hand-made pairs: one contradicting frame goes
alone, two new frames contradicting each other both go, a frame registered only through rejected
frames goes with them; a stored keyframe contradicted by three anchored frames that agree with a
keyframe they were not measured against is marked suspect, while floating or lattice-registered
frames cannot outvote it. Then the mapper's rounds (``AnchoredMap.plan_depth_rerun``) and the
service end to end on frames ray-cast from a textured room, a view's depth corrupted only as the
depth-consistency limit sees it (``AnchoredMap.sensor_depth``)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from oh_my_slam.mapping import anchored as an
from oh_my_slam.mapping import standpoint as sp
from oh_my_slam.mapping.anchored import AnchoredMap, DepthCounts, Keyframe, Placement
from tests.synth.room import base_pose
from tests.unit.test_service_api import Client
from tests.unit.test_service_prior_first import CAM, client, dry, frames_for  # noqa: F401


def _counts(r: float | None) -> DepthCounts:
    """One way of a pair: 100 points agreeing or contradicting at ratio ``r``; None: too few
    counted points (an unmeasured direction)."""
    if r is None:
        return DepthCounts(100, 5, 0, 0, 95)
    return DepthCounts(100, round(100 * r), 100 - round(100 * r), 0, 0)


def P(a: str, b: str, ab: float | None = 1.0, ba: float | None = 1.0) -> sp.DepthPair:
    return sp.DepthPair(a, b, _counts(ab), _counts(ba), 0.0)


def _new(**refs: str) -> dict[str, tuple[str, frozenset[str]]]:
    """New frames by name: keyframe id ``k_<name>``, references the space-separated ids."""
    return {n: (f"k_{n}", frozenset(r.split())) for n, r in refs.items()}


STORED = {f"s{i}": f"k_s{i}" for i in range(4)}
FIXED = set(STORED.values())


def resolve(pairs: list[sp.DepthPair], new: dict[str, tuple[str, frozenset[str]]],
            marked: set[str] | None = None, vetoed: set[str] | None = None,
            stored: dict[str, str] | None = None, fixed: set[str] | None = None
            ) -> sp.DepthResolution:
    return sp.depth_resolve(pairs, new, STORED if stored is None else stored,
                            FIXED if fixed is None else fixed, marked or set(), vetoed or set())


# -- the decision ---------------------------------------------------------------------------------


def test_pair_pieces() -> None:
    """``DepthPair``: a measured agreement, the ratios from either frame's side, the row's
    ``outcome`` only on a failing pair."""
    p = P("a", "b", 0.95, None)
    assert p.agrees and not p.fails and p.ratios("b") == (None, 0.95)
    assert not P("a", "b", None, None).agrees and not P("a", "b", 0.5).agrees
    assert P("a", "b").row("frame_rejected")["outcome"] is None
    assert P("a", "b", 0.5).row("frame_rejected")["outcome"] == "frame_rejected"


def test_one_contradicting_frame_goes_alone() -> None:
    """A new frame failing its pair with a stored keyframe is rejected, naming it; the others,
    and a pair that agrees, stay."""
    new = _new(n1="k_s0", n2="k_s0", n3="k_s1")
    res = resolve([P("n1", "s0"), P("n2", "s0", 0.887, 0.96), P("n2", "s1"), P("n3", "s1"),
                   P("n1", "n2", 0.5)], new)
    # the new-new failure waits while a frame contradicts the stored map
    assert res.reject == {"n2": ("s0",)} and not res.mark and not res.veto


def test_two_new_frames_contradicting_each_other_both_go() -> None:
    res = resolve([P("n1", "s0"), P("n2", "s0"), P("n1", "n2", 0.6, 1.0)],
                  _new(n1="k_s0", n2="k_s0"))
    assert res.reject == {"n1": ("n2",), "n2": ("n1",)}


def test_a_frame_registered_only_through_rejected_frames_goes_with_them() -> None:
    """``n3`` rests on ``n2`` only (rejected): it goes, then ``n4`` resting on ``n3`` only;
    ``n5`` rests on ``n2`` and a stored keyframe: it stays; the origin (no reference) stays."""
    new = _new(n1="", n2="k_s0", n3="k_n2", n4="k_n3", n5="k_n2 k_s1")
    res = resolve([P("n2", "s0", 0.4)], new)
    assert res.reject == {"n2": ("s0",), "n3": ("n2",), "n4": ("n3",)}


def _suspect_case(**over: str) -> tuple[list[sp.DepthPair], dict[str, tuple[str, frozenset[str]]]]:
    """Three new frames contradicting stored ``s0``, each registered on ``s1`` (agreeing) and
    agreeing with ``s2`` (never measured against), agreeing with each other."""
    refs = {"n1": "k_s1", "n2": "k_s1", "n3": "k_s1 k_s0", **over}
    pairs = []
    for n in refs:
        pairs += [P(n, "s0", 0.85, 0.97), P(n, "s1"), P(n, "s2", 0.98, None)]
    pairs += [P("n1", "n2"), P("n2", "n3"), P("n1", "n3")]
    return pairs, _new(**refs)


def test_a_stored_keyframe_contradicted_by_three_anchored_frames_is_suspect() -> None:
    pairs, new = _suspect_case()
    res = resolve(pairs, new)
    assert res.reject == {} and list(res.mark) == ["k_s0"]
    assert res.mark["k_s0"] == [
        {"name": n, "agree_ratio_keyframe": 0.97, "agree_ratio_frame": 0.85}
        for n in ("n1", "n2", "n3")]
    # kept in a later round while it holds; withdrawn when it no longer does (and never again)
    again = resolve(pairs, new, marked={"k_s0"})
    assert again.mark == {} and again.veto == set() and again.reject == {}
    gone = resolve(pairs[:6], new, marked={"k_s0"})
    assert gone.veto == {"k_s0"} and set(gone.reject) == {"n1", "n2"}
    assert resolve(pairs, new, vetoed={"k_s0"}).mark == {}


def test_two_frames_do_not_mark_a_keyframe() -> None:
    pairs, new = _suspect_case()
    res = resolve([p for p in pairs if "n3" not in (p.a, p.b)], {**new, "n3": new["n3"]})
    assert res.mark == {} and set(res.reject) == {"n1", "n2"}


def test_floating_frames_cannot_outvote_a_stored_keyframe() -> None:
    """Frames registered only through each other (not anchored in the stored map) contradict
    ``s0`` while agreeing with ``s2``: no mark; they are rejected."""
    pairs, new = _suspect_case(n1="k_n2", n2="k_n3", n3="k_n1")
    res = resolve(pairs, new)
    assert res.mark == {} and set(res.reject) == {"n1", "n2", "n3"}


def test_lattice_registered_frames_cannot_outvote_a_stored_keyframe() -> None:
    """Frames that agree only with the keyframes they were measured against (their references,
    ``s1`` and ``s2``), as a lattice pose does, contradicting ``s0``: no mark."""
    pairs, new = _suspect_case(n1="k_s1 k_s2", n2="k_s1 k_s2", n3="k_s1 k_s2")
    assert resolve(pairs, new).mark == {}


def test_contradictors_that_disagree_or_contradict_others_mark_nothing() -> None:
    pairs, new = _suspect_case()
    split = [p if (p.a, p.b) != ("n1", "n2") else P("n1", "n2", 0.5) for p in pairs]
    assert resolve(split, new).mark == {}
    # n3 also contradicts s3: it supports no mark, two frames are left
    assert resolve([*pairs, P("n3", "s3", 0.5)], new).mark == {}
    # a re-registered or newly stored keyframe is never marked
    assert resolve(pairs, new, fixed=FIXED - {"k_s0"}).mark == {}


# -- the mapper's rounds ----------------------------------------------------------------------


def test_plan_depth_rerun_rounds(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """``plan_depth_rerun`` applies a round: marks and unmarks keyframes, forces the rejected
    frames for the next registration; ends without a change. A round that would cross the share
    (frames rejected for a contradiction only, dependents aside) or that is pending at the run
    limit is not applied: the update is rejected whole and the round reported ``applied``
    false (its rows, its marks in ``depth_pending``)."""
    am = AnchoredMap(tmp_path / "m")
    am.keyframes = [Keyframe(k, f"s{i}", "cam", an.Intrinsics(1, 1, 0, 0, 2, 2), np.eye(4), 0.0,
                             0.0, "fused", None, "sensor") for i, k in enumerate(("k_s0", "k_s1"))]
    plans: list[Any] = []
    monkeypatch.setattr(sp, "depth_plan", lambda m, pls: plans.pop(0))
    pls = [Placement(f"n{i}", "registered", inliers=40 + i) for i in range(4)]
    plans.append(None)
    assert not am.plan_depth_rerun(pls)  # no new frame with sensor depth
    row = {("n0", "s0"): P("n0", "s0", 0.5).row("frame_rejected")}
    plans.append((row, sp.DepthResolution(reject={"n0": ("s0",), "n3": ("n0",)},
                                          contradict={"n0"},
                                          mark={"k_s0": [{"name": "n1"}]}), 4))
    assert am.plan_depth_rerun(pls) and am.depth_first == 4 and am.depth_runs == 1
    assert am.excluded()["n0"].reason == "depth_inconsistent"
    assert am.excluded()["n0"].depth_with == ("s0",) and am.excluded()["n0"].inliers == 40
    assert am.excluded()["n3"].depth_with == ("n0",)  # a dependent: not counted for the share
    assert am.depth_contradicting == {"n0"} and set(am.depth_rejected) == {"n0", "n3"}
    assert am.keyframes[0].suspect == {"contradicted_by": [{"name": "n1"}]}
    assert not am.keyframes[0].reference() and am.depth_rows == row and "depth" in am.reran
    plans.append(({}, sp.DepthResolution(veto={"k_s0"}), 3))
    assert am.plan_depth_rerun(pls) and am.keyframes[0].suspect is None
    assert am.depth_vetoed == {"k_s0"} and am.depth_suspects == {} and am.depth_first == 4
    plans.append(({("n1", "n2"): P("n1", "n2").row()}, sp.DepthResolution(), 3))
    assert not am.plan_depth_rerun(pls) and ("n1", "n2") in am.depth_rows  # the fixed point
    # one more contradicting frame: 2 of 4, not more than half: applied
    plans.append(({}, sp.DepthResolution(reject={"n1": ("s1",)}, contradict={"n1"}), 3))
    assert am.plan_depth_rerun(pls) and am.depth_contradicting == {"n0", "n1"}
    # a third would cross the share: not applied, reported as decided
    pend = {("n2", "s1"): P("n2", "s1", 0.5).row("frame_rejected"),
            ("n1", "s0"): P("n1", "s0", 0.5).row("keyframe_suspect"),
            ("n2", "s0"): P("n2", "s0", 0.5).row("keyframe_suspect"),
            ("n1", "n2"): P("n1", "n2", 0.5).row("update_rejected")}
    plans.append((pend, sp.DepthResolution(reject={"n2": ("s1",)}, contradict={"n2"},
                                           mark={"k_s1": [{"name": "n2"}]}), 3))
    assert not am.plan_depth_rerun(pls) and am.depth_whole and "n2" not in am.excluded()
    assert am.depth_rows[("n2", "s1")]["applied"] is False
    assert am.depth_rows[("n2", "s0")]["applied"] is False  # n2 a pending rejection
    assert "applied" not in am.depth_rows[("n1", "s0")]  # n1 applied, s0 not newly marked
    assert "applied" not in am.depth_rows[("n1", "n2")]  # no decision of the round
    assert am.depth_pending == {"k_s1": [{"name": "n2"}]} and am.keyframes[1].suspect is None
    assert sp.suspect_keyframes(am) == [
        {"name": "s1", "contradicted_by": [{"name": "n2"}], "applied": False}]
    assert not am.plan_depth_rerun(pls) and not plans  # decided: no further round
    # the run limit with a decision pending: likewise
    am.depth_whole, am.depth_runs = False, sp.DEPTH_RESOLVE_RUNS
    plans.append(({}, sp.DepthResolution(veto={"k_s0"}), 3))
    assert not am.plan_depth_rerun(pls) and am.depth_whole
    am.begin_job()  # a second registration of the job keeps it; a new job starts afresh
    assert am.depth_whole
    am.reran = set()
    am.begin_job()
    assert not am.depth_whole and am.depth_rows == {} and am.depth_first is None
    assert am.depth_pending == {} and am.depth_contradicting == set()


def test_a_monocular_frame_resting_on_a_rejected_frame_goes_with_it() -> None:
    """Step 3 covers frames of any depth kind (``others``: never paired): a monocular frame
    registered only through the rejected frame is rejected naming it; one that also rests on a
    stored keyframe stays; only step 2's rejections are contradictions."""
    others = _new(m1="k_n2", m2="k_n2 k_s1", m3="k_m1")
    res = sp.depth_resolve([P("n2", "s0", 0.4)], _new(n2="k_s0"), STORED, FIXED, set(), set(),
                           others)
    assert res.reject == {"n2": ("s0",), "m1": ("n2",), "m3": ("m1",)}
    assert res.contradict == {"n2"}


def test_a_re_registered_keyframe_counts_as_stored_but_never_votes() -> None:
    """A stored keyframe the job re-registered (in ``stored``, not ``fixed``): a failing pair
    with it rejects the new frame; it is never marked, and a suspect-test frame failing with
    it still supports a mark of another keyframe."""
    pairs, new = _suspect_case()
    stored = {**STORED, "r0": "k_r0"}
    res = resolve([*pairs, P("n1", "r0", 0.5)], new, stored=stored)
    assert list(res.mark) == ["k_s0"] and res.reject == {"n1": ("r0",)}
    res = resolve([P("n1", "r0", 0.5), P("n2", "r0", 0.5), P("n3", "r0", 0.5)],
                  _new(n1="k_s1", n2="k_s1", n3="k_s1"), stored=stored)
    assert res.mark == {} and set(res.reject) == {"n1", "n2", "n3"}


# -- the service end to end -------------------------------------------------------------------


def _corrupt(monkeypatch: pytest.MonkeyPatch, names: set[str]) -> None:
    """The depth-consistency limit sees these views' depth at 0.6 x (their points in front of
    what the others see: contradictions); registration sees the true depth."""
    real: Callable[..., Any] = AnchoredMap.sensor_depth

    def fake(self: AnchoredMap, k: Keyframe) -> Any:
        got = real(self, k)
        return got if got is None or k.name not in names else (got[0] * 0.6, got[1])

    monkeypatch.setattr(AnchoredMap, "sensor_depth", fake)


def test_one_bad_frame_is_rejected_alone_and_the_rest_commits(
        client: Client, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """A look of four frames into a stored sweep, one of whose depth contradicts the stored
    views: it is rejected ``depth_inconsistent`` naming them, the job registers again and the
    other three commit; ``depth_pairs`` keeps its pairs (``frame_rejected``); the dry run
    predicts it."""
    fr, files = frames_for(client, [base_pose(30.0 * k) for k in range(6)], tag="s")
    assert client.update("m", fr, files, cameras={"cam": CAM})["state"] == "committed"
    _corrupt(monkeypatch, {"t1"})
    fr, files = frames_for(client, [base_pose(15.0 + 30.0 * k) for k in range(4)], tag="t")
    pred = dry(client, "m", fr, files)
    job = client.update("m", fr, files)
    q = job["quality"]
    assert job["state"] == "committed" and q["failed_checks"] == [], q
    st = {i["name"]: i for i in job["inputs"]}
    assert st["t1"]["status"] == "rejected" and st["t1"]["reason"] == "depth_inconsistent"
    assert st["t1"]["depth_inconsistent_with"] and all(
        n.startswith("s") for n in st["t1"]["depth_inconsistent_with"])
    assert [st[n]["status"] for n in ("t0", "t2", "t3")] == ["registered"] * 3
    bad = [r for r in q["depth_pairs"] if r["fails"]]
    assert bad and all("t1" in (r["a"], r["b"]) and r["outcome"] == "frame_rejected"
                       for r in bad)
    assert all(r["outcome"] is None for r in q["depth_pairs"] if not r["fails"])
    assert q["suspect_keyframes"] == []
    p = {r["name"]: r for r in pred["frames"]}
    assert p["t1"]["predicted_reason"] == "depth_inconsistent"
    assert p["t1"]["depth_inconsistent_with"] == st["t1"]["depth_inconsistent_with"]
    assert pred["quality"]["depth_pairs"] == q["depth_pairs"]
    names = [f["name"] for f in client.get("/api/maps/m/frames", "Frames")["frames"]]
    assert "t1" not in names and {"t0", "t2", "t3"} <= set(names)


def test_most_of_the_update_contradicting_rejects_it_whole(
        client: Client, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """Two of three frames contradict: no consistent part remains (more than half), the update
    is rejected ``depth_inconsistent`` as before oms/2.16, nothing commits."""
    fr, files = frames_for(client, [base_pose(30.0 * k) for k in range(6)], tag="s")
    assert client.update("m", fr, files, cameras={"cam": CAM})["state"] == "committed"
    _corrupt(monkeypatch, {"t0", "t1"})
    fr, files = frames_for(client, [base_pose(15.0 + 30.0 * k) for k in range(3)], tag="t")
    job = client.update("m", fr, files)
    q = job["quality"]
    assert job["state"] == "rejected" and "depth_inconsistent" in q["failed_checks"], q
    # the round that crossed the share is reported as decided, not applied
    bad = [r for r in q["depth_pairs"] if r["fails"]]
    assert bad and all(r["outcome"] == "frame_rejected" and r["applied"] is False
                       for r in bad if {"t0", "t1"} & {r["a"], r["b"]})
    assert all(i["status"] == "registered" for i in job["inputs"])
    assert client.service.store.latest("m") == 1


def test_a_stored_keyframe_contradicted_by_a_look_is_marked_suspect(
        client: Client, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """A stored view whose depth the later look contradicts from several anchored standpoints
    is marked suspect: the look commits whole, ``quality.suspect_keyframes`` lists it with its
    contradicting frames, the version records the mark (frames read
    ``depth_suspect_version``, the mapper's store), and from then on it is no reference, no
    locate reference and never paired."""
    fr, files = frames_for(client, [base_pose(20.0 * k) for k in range(9)], tag="b")
    assert client.update("m", fr, files, cameras={"cam": CAM})["state"] == "committed"
    real = AnchoredMap.sensor_depth
    _corrupt(monkeypatch, {"b4"})
    fr, files = frames_for(client, [base_pose(10.0 + 20.0 * k) for k in range(1, 7)], tag="t")
    pred = dry(client, "m", fr, files)
    job = client.update("m", fr, files)
    q = job["quality"]
    assert job["state"] == "committed", q
    assert [i["status"] for i in job["inputs"]] == ["registered"] * 6
    assert [s["name"] for s in q["suspect_keyframes"]] == ["b4"], q["suspect_keyframes"]
    by = q["suspect_keyframes"][0]["contradicted_by"]
    assert len(by) >= sp.SUSPECT_MIN_FRAMES
    assert all(r["agree_ratio_keyframe"] is None or r["agree_ratio_keyframe"] < 0.9 or
               (r["agree_ratio_frame"] is not None and r["agree_ratio_frame"] < 0.9) for r in by)
    sus = [r for r in q["depth_pairs"] if "b4" in (r["a"], r["b"]) and r["fails"]]
    assert sus and {r["outcome"] for r in sus} == {"keyframe_suspect"}
    assert pred["quality"]["suspect_keyframes"] == q["suspect_keyframes"]
    frames = client.get("/api/maps/m/frames", "Frames")["frames"]
    assert [f.get("depth_suspect_version") for f in frames if f["name"] == "b4"] == [2]
    assert all("depth_suspect_version" not in f for f in frames if f["name"] != "b4")
    store = client.service.store
    rec = store.version("m", 2)
    row = next(f for f in rec["frames"] if f["name"] == "b4")
    assert row["depth_suspect"]["version"] == 2 and row["depth_suspect"]["contradicted_by"] == by
    am = AnchoredMap.load(store.mapper_dir("m"))
    kf = next(k for k in am.keyframes if k.name == "b4")
    assert kf.suspect == {"contradicted_by": by} and not kf.reference() and kf.usable()
    client.get("/api/maps/m/grid?cell_m=0.1", "Grid")
    assert client.service.derived is not None
    meta = client.service.derived.get("m", 2).meta
    assert "b4" not in meta["keyframes"] and "b3" in meta["keyframes"]
    # a later job: the mark stays (version 3 copies it), b4 is never paired again
    monkeypatch.setattr(AnchoredMap, "sensor_depth", real)
    fr, files = frames_for(client, [base_pose(80.0)], tag="u")
    job = client.update("m", fr, files)
    assert job["state"] == "committed" and job["quality"]["suspect_keyframes"] == []
    assert all("b4" not in (r["a"], r["b"]) for r in job["quality"]["depth_pairs"])
    assert job["quality"]["depth_pairs"]
    row = next(f for f in store.version("m", 3)["frames"] if f["name"] == "b4")
    assert row["depth_suspect"]["version"] == 2


def test_a_monocular_frame_registered_through_the_rejected_frame_goes_with_it(
        client: Client, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """A mixed RGB-D and monocular job: the sensor frame ``t1`` contradicts the stored views
    and is rejected; the monocular ``m0``, turned beyond the stored views' reach and registered
    only through ``t1``, is rejected ``depth_inconsistent`` naming it; ``t0`` commits."""
    fr, files = frames_for(client, [base_pose(30.0 * k) for k in range(3)], tag="s")
    assert client.update("m", fr, files, cameras={"cam": CAM})["state"] == "committed"
    _corrupt(monkeypatch, {"t1"})
    fr, files = frames_for(client, [base_pose(45.0), base_pose(105.0)], tag="t")
    fm, fmf = frames_for(client, [base_pose(160.0)], tag="m", depth=False)
    job = client.update("m", [*fr, *fm], {**files, **fmf})
    st = {i["name"]: i for i in job["inputs"]}
    assert job["state"] == "committed", job["quality"]
    assert st["t1"]["reason"] == "depth_inconsistent" and st["t0"]["status"] == "registered"
    assert st["m0"]["status"] == "rejected" and st["m0"]["reason"] == "depth_inconsistent"
    assert st["m0"]["depth_inconsistent_with"] == ["t1"]


def test_the_last_round_is_reported_when_not_applied(
        client: Client, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """No re-run left (``DEPTH_RESOLVE_RUNS`` 0): the first round's decision is pending, so a
    contradiction remains and the update is rejected whole; its rows show the rejection it
    decided with ``applied`` false, and the frame keeps its status."""
    monkeypatch.setattr(sp, "DEPTH_RESOLVE_RUNS", 0)
    fr, files = frames_for(client, [base_pose(30.0 * k) for k in range(6)], tag="s")
    assert client.update("m", fr, files, cameras={"cam": CAM})["state"] == "committed"
    _corrupt(monkeypatch, {"t1"})
    fr, files = frames_for(client, [base_pose(15.0 + 30.0 * k) for k in range(4)], tag="t")
    pred = dry(client, "m", fr, files)
    job = client.update("m", fr, files)
    q = job["quality"]
    assert job["state"] == "rejected" and "depth_inconsistent" in q["failed_checks"], q
    assert all(i["status"] == "registered" for i in job["inputs"])
    pend = [r for r in q["depth_pairs"] if r.get("applied") is False]
    assert pend and all("t1" in (r["a"], r["b"]) and r["outcome"] == "frame_rejected"
                        for r in pend)
    assert pred["quality"]["depth_pairs"] == q["depth_pairs"]


def test_a_pending_mark_is_reported_when_not_applied(
        client: Client, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """The look that would mark ``b4`` suspect, with no re-run left: the update is rejected
    whole and ``suspect_keyframes`` shows the mark it decided with ``applied`` false; nothing
    is stored."""
    monkeypatch.setattr(sp, "DEPTH_RESOLVE_RUNS", 0)
    fr, files = frames_for(client, [base_pose(20.0 * k) for k in range(9)], tag="b")
    assert client.update("m", fr, files, cameras={"cam": CAM})["state"] == "committed"
    _corrupt(monkeypatch, {"b4"})
    fr, files = frames_for(client, [base_pose(10.0 + 20.0 * k) for k in range(1, 7)], tag="t")
    job = client.update("m", fr, files)
    q = job["quality"]
    assert job["state"] == "rejected" and "depth_inconsistent" in q["failed_checks"], q
    assert [(s["name"], s.get("applied")) for s in q["suspect_keyframes"]] == [("b4", False)]
    sus = [r for r in q["depth_pairs"] if "b4" in (r["a"], r["b"]) and r["fails"]]
    assert sus and all(r["outcome"] == "keyframe_suspect" and r["applied"] is False
                       for r in sus)
    am = AnchoredMap.load(client.service.store.mapper_dir("m"))
    assert all(k.suspect is None for k in am.keyframes)
