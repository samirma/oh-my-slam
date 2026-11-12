"""Round 13 (spec P24, P25): symmetric scenes in the two-view measurement and the registration
gate, the depth agreement's counting, the dry run's exception, masked views and the older
mapper's vote shares, standpoint joins without a chain (map.md §1, §2, §5); the label scores and
the client-label choice (oms/2.4, map.md §5)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from oh_my_slam.mapping import anchored as an
from oh_my_slam.mapping import register_ma
from oh_my_slam.mapping import standpoint as sp
from oh_my_slam.service import prior_first
from tests.synth.room import K_ROBOT, base_pose
from tests.unit.test_round11 import _err, _frame, _turn, _turn_frames, _tv, _view
from tests.unit.test_service_api import Client
from tests.unit.test_service_prior_first import (
    CAM,
    client,  # noqa: F401
    dry,
)


def _yaw(deg: float) -> np.ndarray:
    return an.exp_so3(np.array([0.0, np.radians(deg), 0.0]))


def _hypotheses(monkeypatch: pytest.MonkeyPatch, turns: list[float]) -> list[str]:
    """``two_view`` of frames 0 and 1 with every hypothesis (the pair's own largest consensus,
    repeated) confirmed by the depth images, each PnP rotation the true one turned by
    ``turns[k]`` degrees; the calls the depth confirmation got, logged."""
    a, b = _view(0), _view(1)
    e = an.epipolar(a.feats, a.K, b.feats, b.K)
    assert e is not None
    monkeypatch.setattr(sp, "epipolar_hypotheses", lambda *x, **k: iter([e] * len(turns)))
    true = np.linalg.inv(_turn(0)) @ _turn(1)
    poses = iter(turns)
    calls: list[str] = []

    def pose(*x: Any) -> np.ndarray:
        T = true.copy()
        T[:3, :3] = _yaw(next(poses)) @ true[:3, :3]
        return T

    monkeypatch.setattr(sp, "depth_pose", pose)
    monkeypatch.setattr(sp, "depth_confirmed", lambda *x, **k: calls.append("c") or True)
    return calls


def test_two_confirmed_hypotheses_apart_make_the_pair_ambiguous(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """map.md §1 "Symmetric scenes" (P24): every hypothesis is evaluated, not only up to the
    first confirmed; two confirmed ones ``SYMMETRIC_MIN_DEG`` or more apart make the pair
    ambiguous: no measurable pair (a reversed fit carries the verdict)."""
    assert sp.SYMMETRIC_MIN_DEG == 10.0
    calls = _hypotheses(monkeypatch, [0.0, 90.0])
    tv = sp.two_view(_view(0), _view(1))
    assert tv is not None and tv.ambiguous and not tv.confirmed and not tv.measurable
    assert len(calls) == 2
    pairs = sp.Pairs()
    pairs.fits[("f0", "f1")] = tv
    back = pairs.get(_view(1), _view(0))
    assert back is not None and back.ambiguous and not back.measurable
    assert sp.ambiguous(_view(0), _view(1), pairs) and sp.ambiguous(_view(1), _view(0), pairs)


def test_hypotheses_closer_than_the_bound_are_one_geometry(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Two confirmed hypotheses 3 deg apart are one geometry measured twice: the first is the
    fit, measurable; every hypothesis was still tested."""
    calls = _hypotheses(monkeypatch, [0.0, 3.0, 0.5])
    tv = sp.two_view(_view(0), _view(1))
    assert tv is not None and tv.measurable and not tv.ambiguous and _err(0, 1, tv) < 0.5
    assert len(calls) == 3


def test_the_tiled_pairs_are_not_ambiguous() -> None:
    """The furnished tiled room's lattice pairs are rejected by the depth images, not
    ambiguous: the true pairs stay measurable (the rule only removes symmetric scenes)."""
    for j in (1, 2):
        tv = sp.two_view(_view(0), _view(j))
        assert tv is not None and tv.measurable and not tv.ambiguous


def test_the_registration_gate_takes_no_ambiguous_reference(
        tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """map.md §2 step 2 (P24), geometric engine: a reference against which another candidate
    pose passes the depth confirmation too places the frame in neither pose; the next
    reference holding enough of its inliers must confirm it; with every one ambiguous the frame
    is no placement (rejected no_overlap). A pose a reference rejects asks nothing more."""
    from tests.unit.test_anchored import frame, new_map

    am = new_map(tmp_path)
    pls = am.update([frame("a", base_pose(0.0), p=None), frame("b", base_pose(20.0), p=None)])
    assert [p.status for p in pls] == ["registered", "registered"]
    asked: list[str] = []
    bad: set[str] = set()

    def amb(self: Any, f: Any, T: Any, own: Any, corr: Any, cands: Any, j: int,
            ref: Any) -> float | None:
        asked.append(cands[j].name)
        return 90.0 if cands[j].name in bad else None

    monkeypatch.setattr(an.AnchoredMap, "_ambiguous", amb)
    bad.update({"a", "b"})
    pl = am.update([frame("c", base_pose(10.0), p=None)])[0]
    assert pl.status == "rejected" and pl.reason == "no_overlap"
    assert sorted(asked) == ["a", "b"]  # both references held enough inliers
    first, other = asked
    bad.clear()
    bad.add(first)
    asked.clear()
    pl = am.update([frame("e", base_pose(10.0), p=None)])[0]
    assert pl.status == "registered" and asked == [first, other]
    monkeypatch.setattr(an, "depth_confirmed", lambda *x, **k: False)
    asked.clear()
    pl = am.update([frame("g", base_pose(10.0), p=None)])[0]
    assert pl.status == "rejected" and asked == []


def _cloud(T: np.ndarray, n: int, seed: int, K: Any) -> tuple[np.ndarray, np.ndarray]:
    """``n`` world points 2-4 m in front of camera ``T`` and their pixels there."""
    rng = np.random.default_rng(seed)
    cam = np.column_stack([rng.uniform(-1.0, 1.0, n), rng.uniform(-0.7, 0.7, n),
                           rng.uniform(2.0, 4.0, n)])
    X = cam @ T[:3, :3].T + T[:3, 3]
    uv = np.column_stack([K.fx * cam[:, 0] / cam[:, 2] + K.cx, K.fy * cam[:, 1] / cam[:, 2] + K.cy])
    return X, uv


def test_the_geometric_engines_other_candidate_poses(
        tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """``AnchoredMap._ambiguous``: sequential PnP poses on the frame's lifted matches to one
    reference that its pose leaves unexplained; one ``SYMMETRIC_MIN_DEG`` or more away that the
    depth confirms makes the reference ambiguous (its angle); one the depth rejects, or near
    the pose (one geometry), or too few matches left, or no PnP consensus: not ambiguous."""
    from tests.unit.test_anchored import frame, new_map

    am = new_map(tmp_path)
    am.update([frame("a", base_pose(0.0), p=None)])
    f = frame("c", base_pose(10.0), p=None)
    K = f.K
    T = base_pose(10.0)

    def turned(deg: float) -> np.ndarray:
        out = T.copy()
        out[:3, :3] = T[:3, :3] @ _yaw(deg)
        return out

    def corr(*groups: tuple[np.ndarray, np.ndarray]) -> Any:
        X = np.vstack([g[0] for g in groups])
        uv = np.vstack([g[1] for g in groups])
        d = (X - T[:3, 3]) / np.linalg.norm(X - T[:3, 3], axis=1)[:, None]
        return an._Corr(uv, X, d, np.zeros(len(X), np.intp))

    own = np.ones((4, 4), np.float32)
    ref = an.DepthRef(T, own, K.K())
    cands = list(am.keyframes)
    mine = _cloud(T, 80, 0, K)
    assert an.SYMMETRIC_MIN_DEG == 10.0
    monkeypatch.setattr(an, "depth_confirmed", lambda *x, **k: True)
    lattice = corr(mine, _cloud(turned(90.0), 80, 1, K))
    deg = am._ambiguous(f, T, own, lattice, cands, 0, ref)
    assert deg is not None and abs(deg - 90.0) < 1.0
    assert am._ambiguous(f, T, own, corr(mine), cands, 0, ref) is None  # nothing left
    near = corr(mine, _cloud(turned(3.0), 80, 2, K), _cloud(turned(5.0), 80, 3, K))
    assert am._ambiguous(f, T, own, near, cands, 0, ref) is None  # one geometry, measured again
    rng = np.random.default_rng(4)
    noise = (rng.uniform(-50.0, 50.0, (80, 3)), rng.uniform(0.0, 300.0, (80, 2)))
    assert am._ambiguous(f, T, own, corr(mine, noise), cands, 0, ref) is None  # no consensus
    monkeypatch.setattr(an, "depth_confirmed", lambda *x, **k: False)
    assert am._ambiguous(f, T, own, lattice, cands, 0, ref) is None  # the depth rejects it


def test_mapanything_takes_no_ambiguous_reference(tmp_path: Any) -> None:
    """``register_ma.confirms`` (P24): an ambiguous reference places the frame in neither pose;
    the next one seeing it must confirm it; with none left, no placement; ``Registration``
    hands each frame its gate by the reference's key."""
    from tests.unit.test_anchored import frame, new_map

    am = new_map(tmp_path)
    am.update([frame("a", base_pose(0.0), p=None), frame("b", base_pose(15.0), p=None)])
    refs = [an.DepthRef(g.T, g.depth, g.K, kf.id) for kf in am.keyframes
            for g in [am._grid(kf)] if g is not None]
    T = am.keyframes[0].T
    new = register_ma.New(0, tmp_path / "x.jpg", am.keyframes[0].K, refs[0].depth)
    keys: list[str] = []

    def amb(bad: set[str]) -> Any:
        return lambda r: keys.append(r.key) or r.key in bad

    assert register_ma.confirms(new, T, refs[0].depth, refs[0].K, refs, amb({refs[0].key}))
    assert keys == [refs[0].key, refs[1].key]
    assert not register_ma.confirms(new, T, refs[0].depth, refs[0].K, refs,
                                    amb({r.key for r in refs}))
    away = an.DepthRef(refs[1].T.copy(), refs[1].depth, refs[1].K, "away")
    away.T[:3, 3] += 50.0  # sees none of the frame's points: no reference of the test
    keys.clear()
    assert register_ma.confirms(new, T, refs[0].depth, refs[0].K, [away, refs[0]], amb(set()))
    assert keys == [refs[0].key]
    reg = register_ma.Registration(lambda *x: [], [new], [])
    assert reg._gate(new) is None
    seen: list[tuple[int, str]] = []
    reg = register_ma.Registration(lambda *x: [], [new], [],
                                   lambda f, key: seen.append((f.index, key)) or True)
    gate = reg._gate(new)
    assert gate is not None and gate(refs[1]) and seen == [(0, refs[1].key)]


def test_the_service_answers_mapanythings_gate(
        client: Client, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """``prior_first._MaRun._ambiguous``: a stored keyframe's pair or another frame of the job
    (``new<index>``), fitted through the job's pairs; an unknown key or a frame without
    features is no ambiguous pair."""
    monkeypatch.setenv(prior_first.NONE_ENGINE_ENV, "mapanything")
    real = an.AnchoredMap.depth_confirms
    monkeypatch.setattr(an.AnchoredMap, "depth_confirms",
                        lambda self, f, *a: f.name not in ("f2", "f3") and real(self, f, *a))
    got: list[Any] = []
    init = register_ma.Registration.__init__

    def keep(self: Any, *a: Any) -> None:
        init(self, *a)
        got.append(self.ambiguous)

    monkeypatch.setattr(register_ma.Registration, "__init__", keep)
    poses = [base_pose(0.0), base_pose(15.0), base_pose(30.0), base_pose(40.0)]
    fr, files = _turn_frames(client, poses, True, 0.0)
    job = client.update("m", fr, files, cameras={"cam": CAM})
    assert job["state"] == "committed", job["quality"]
    assert [i["status"] for i in job["inputs"]] == ["registered"] * 4
    gate = got[0]
    run = gate.__self__
    f2 = next(n for n in run.reg.new if n.index == 2)
    assert gate(f2, "new3") is False
    stored = run.am.keyframes[0].id
    assert gate(f2, stored) is False and gate(f2, "nope") is False
    run.feats.pop(2)
    assert gate(f2, stored) is False


def test_the_depth_agreement_counts_only_what_it_could_see() -> None:
    """map.md §1 "Depth agreement" (P24), ``seen_through``: a point in front of the other
    view's surface contradicts; one behind it (occluded), outside its image, behind the camera
    or where it measures no depth in range counts neither way."""
    from oh_my_slam.core.types import Intrinsics

    K = Intrinsics(100.0, 100.0, 50.0, 40.0, 100, 80)
    wall = np.full((80, 100), 2.0, np.float32)
    p = an.DepthRef(np.eye(4), wall, K.K())

    def counts(depth: np.ndarray, T: np.ndarray | None = None) -> tuple[int, int]:
        return an.seen_through(p, an.DepthRef(np.eye(4) if T is None else T, depth, K.K()),
                               0.1)

    n = len(an.backproject(wall, K.K()))
    assert counts(wall) == (n, 0)
    assert counts(np.full_like(wall, 3.0)) == (0, n)  # q sees through: contradictions
    assert counts(np.full_like(wall, 1.0)) == (0, 0)  # occluded behind q's surface
    assert counts(np.full_like(wall, 7.0)) == (0, 0)  # beyond q's depth range
    assert counts(np.zeros_like(wall)) == (0, 0)  # no depth
    away = np.eye(4)
    away[0, 3] = 50.0
    assert counts(wall, away) == (0, 0)  # outside q's image
    behind = np.eye(4)
    behind[2, 3] = 5.0
    assert counts(wall, behind) == (0, 0)  # behind q


def test_the_dry_run_differs_from_the_job_only_where_mapanything_places(
        client: Client, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """api.md §4, map.md §2 step 7 (P24): a depth frame the geometric engine leaves unplaced
    is predicted by geometric registration (rejected) while the job's MapAnything places it;
    every other frame's predicted status and reason are the job's."""
    monkeypatch.setenv(prior_first.NONE_ENGINE_ENV, "mapanything")
    real = an.AnchoredMap.depth_confirms
    monkeypatch.setattr(an.AnchoredMap, "depth_confirms",
                        lambda self, f, *a: f.name != "f2" and real(self, f, *a))
    poses = [base_pose(0.0), base_pose(15.0), base_pose(30.0), base_pose(45.0)]
    fr, files = _turn_frames(client, poses, True, 0.0)
    pred = dry(client, "m", fr, files, cameras={"cam": CAM})
    job = client.update("m", fr, files, cameras={"cam": CAM})
    assert job["state"] == "committed", job["quality"]
    differ = {f["name"] for f, p, i in zip(fr, pred["frames"], job["inputs"], strict=True)
              if (p["predicted_status"], p.get("predicted_reason")) != (i["status"],
                                                                         i.get("reason"))}
    assert differ == {"f2"}
    assert pred["frames"][2]["predicted_status"] == "rejected"
    assert job["inputs"][2]["status"] == "registered"
    assert len(client.backend.mv_calls) == 1


def test_a_standpoint_join_without_a_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """map.md §1 (P24): stored-to-stored pairs are not re-measured, so a pair joining a new
    frame to a stored one may have no chain of stronger pairs: kept as measured without depth,
    judged by the depth test alone with it, even far off what a chain would give."""
    f, _ = _frame(0)
    T = base_pose(0.0)
    views = [sp.View("s1", K_ROBOT, f, T[:3, :3], T[:3, 3], True, kid="k1"),
             sp.View("s2", K_ROBOT, f, T[:3, :3], T[:3, 3], True, kid="k2"),
             sp.View("n", K_ROBOT, f, T[:3, :3], T[:3, 3], False, 0)]
    for depth in (False, True):
        pairs = sp.Pairs()
        # s1-s2 would make n-s2 70 deg off its chain n-s1-s2: it is never measured
        pairs.fits.update({("n", "s1"): _tv(500, 20.0, depth), ("n", "s2"): _tv(300, -50.0, depth),
                           ("s1", "s2"): _tv(900, 0.0, depth)})
        got = {frozenset((views[i].name, views[j].name)) for i, j, _ in sp.evaluated(views, pairs)}
        assert got == {frozenset(("n", "s1")), frozenset(("n", "s2"))}


# -- masked views and the older mapper's vote shares (map.md §5 "Ignored pixels", P24) ----------


def test_the_older_mappers_masked_views_leave_the_vote_shares() -> None:
    """``mapping.objects.MapObject.view_votes``: each keyframe's votes are recorded
    (``frame_votes``), so a masked keyframe's leave the numerator and the denominator
    (``label_votes / observations``); a merge and a round trip keep them; nothing masked, or a
    map written before (no votes recorded): nothing to take out."""
    from oh_my_slam.mapping import objects as ob
    from oh_my_slam.service.objects import _mapper_records

    def obj(oid: int, votes: list[tuple[int, str, float]]) -> ob.MapObject:
        o = ob.MapObject(oid, votes[0][1], {}, [], np.zeros((0, 3), np.float32))
        for f, lab, sc in votes:
            o.vote(lab, sc)
            fv = o.frame_votes.setdefault(f, {})
            fv[lab] = fv.get(lab, 0.0) + sc
        o.frames = sorted({f for f, _, _ in votes})
        return o

    o = obj(1, [(0, "box", 1.0), (1, "bin", 1.0), (2, "box", 1.0)])
    assert o.view_votes() == {}
    o.masked = [1]
    assert o.view_votes() == {"view_votes": {"box": 2.0}, "view_observations": 2}
    d = o.to_dict()
    assert d["view_votes"] == {"box": 2.0} and d["frame_votes"]["1"] == {"bin": 1.0}
    back = ob.MapObject.from_dict(d, o.points)
    assert back.frame_votes == o.frame_votes and back.view_votes() == o.view_votes()
    old = ob.MapObject.from_dict({k: v for k, v in d.items() if k != "frame_votes"}, o.points)
    assert old.frame_votes == {} and old.view_votes() == {}
    other = obj(2, [(1, "bin", 0.5), (3, "bin", 1.0)])
    o.absorb(other)
    assert o.frame_votes[1] == {"bin": 1.5} and o.frame_votes[3] == {"bin": 1.0}
    from types import SimpleNamespace

    meta = {"objects": [{"id": 7, "label": "box", "score": 0.9, "created_update": 1,
                         "label_votes": {"box": 2.0, "bin": 1.0}, "observations": 3,
                         "view_votes": {"box": 2.0}, "view_observations": 2,
                         "centre_m": [0.0, 0.0, 0.0], "size_m": [0.3, 0.3, 0.3],
                         "yaw_deg": 0.0}]}
    cloud = {"object_id": np.array([7, 7]), "rgb": np.full((2, 3), 100, np.uint8),
             "xyz": np.zeros((2, 3), np.float32)}
    dv = SimpleNamespace(version=1, record={"frames": [], "update_versions": {}}, meta=meta,
                         cloud=lambda: cloud)
    rec = _mapper_records(dv)["objects"][0]  # type: ignore[arg-type]
    assert rec["detection_votes"] == [{"label": "box", "share": 1.0}]
    assert rec["detected_as"] == ["box", "bin"]


def test_add_records_each_keyframes_votes() -> None:
    from types import SimpleNamespace

    from oh_my_slam.mapping import objects as ob

    o = ob.MapObject(1, "box", {}, [], np.zeros((0, 3), np.float32))
    sighting = ob.Sighting(4, 1, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    obs = [SimpleNamespace(key=(4, k), label=lab, score=0.5, points=np.zeros((1, 3), np.float32),
                           source=ob.SRC_RELIABLE, frame=4, depth=1.0, pixel_count=10,
                           sighting=sighting) for k, lab in enumerate(["box", "box"])]
    o.add(obs, 1)  # type: ignore[arg-type]
    assert o.frame_votes == {4: {"box": 1.0}}


# -- P25: label scores (oms/2.4) -----------------------------------------------------------------


def test_label_scores_list_the_label_then_the_votes() -> None:
    """map.md §5 ``label_scores`` (oms/2.4): the record's label first, then the other labels
    with a share, highest first, at most three, each its ``detection_votes`` share; present
    whenever the version's vocabulary holds client labels (the schema's ``MapObject``)."""
    from oh_my_slam.service import schema
    from oh_my_slam.service.objects import label_scores, with_label_scores

    votes = [{"label": "cup", "share": 0.6}, {"label": "red mug", "share": 0.5},
             {"label": "bowl", "share": 0.2}, {"label": "vase", "share": 0.1},
             {"label": "jar", "share": 0.0}]
    o = {"label": "red mug", "detection_votes": votes}
    assert label_scores(o) == [{"label": "red mug", "score": 0.5}, {"label": "cup", "score": 0.6},
                               {"label": "bowl", "score": 0.2}]
    assert label_scores({"label": "cup", "detection_votes": votes[:1]}) == [
        {"label": "cup", "score": 0.6}]  # fewer labels got votes: fewer entries
    assert label_scores({"label": "box", "detection_votes": []}) == [
        {"label": "box", "score": 0.0}]
    answer = {"objects": [o]}
    assert with_label_scores(answer, []) is answer
    got = with_label_scores(answer, ["Red Mug"])
    assert got["objects"][0]["label_scores"][0] == {"label": "red mug", "score": 0.5}
    assert with_label_scores(got, ["Red Mug"]) == got  # idempotent
    item = schema.defs()["MapObject"]["properties"]["label_scores"]
    assert item["maxItems"] == 3
    assert schema.SCHEMA_VERSION == "oms/2.4"


def test_an_objects_read_carries_label_scores_with_client_labels(
        client: Client) -> None:  # noqa: F811
    """The objects read of a version with client labels: every record carries
    ``label_scores``, its first entry its label (the schema checks the body)."""
    from tests.unit.test_service_prior_first import frames_for

    fr, files = frames_for(client, [base_pose(0), base_pose(20)], tag="a")
    job = client.update("m", fr, files, cameras={"cam": CAM}, labels=["Red Mug"])
    assert job["state"] == "committed"
    objs = client.get("/api/maps/m/objects", "Objects")["objects"]
    for o in objs:
        assert o["label_scores"][0]["label"] == o["label"]


# -- the second registration reuses the first's work (map.md §2 step 6) --------------------------


def test_a_second_registration_reuses_the_first_and_equals_a_full_re_run(
        client: Client, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """A pose duplicate that served the job (its depth image's scale evidence) makes the job
    register once more without it (P16). That second registration reuses the first's work
    (``AnchoredMap._preps``, ``_placed``, ``_epi``: each keyed by everything it depends on), so
    it places fewer frames than a full re-run, and the job's statuses, poses, sigmas and
    quality are exactly a full re-run's (the caches dropped at its start)."""
    from tests.unit.test_service_prior_first import frames_for

    runs: list[str] = []
    placed: list[str] = []
    reg, place = an.AnchoredMap._register_drift, an.AnchoredMap.place

    def counted(self: an.AnchoredMap, *a: Any) -> Any:
        runs.append("r")
        return reg(self, *a)

    def placing(self: an.AnchoredMap, f: Any, *a: Any, **k: Any) -> Any:
        placed.append(f.name)
        return place(self, f, *a, **k)

    monkeypatch.setattr(an.AnchoredMap, "_register_drift", counted)
    monkeypatch.setattr(an.AnchoredMap, "place", placing)

    def job(name: str) -> tuple[dict[str, Any], int, int]:
        fr, files = frames_for(client, [base_pose(60), base_pose(30)], tag="s")
        assert client.update(name, fr, files, cameras={"cam": CAM})["state"] == "committed"
        runs.clear()
        placed.clear()
        poses = [base_pose(0), base_pose(0), base_pose(10), base_pose(45)]
        fr, files = frames_for(client, poses, tag="f")
        doc = client.update(name, fr, files, cameras={"cam": CAM})
        assert doc["state"] == "committed", doc["quality"]
        return doc, len(runs), len(placed)

    reused, n_runs, n_reused = job("a")
    assert n_runs >= 2  # registered once more
    assert {i["name"]: i.get("reason") for i in reused["inputs"]}["f1"] == "duplicate"
    begin = an.AnchoredMap.begin_job

    def fresh(self: an.AnchoredMap) -> None:  # a full re-run: nothing kept from the first
        begin(self)
        self._epi, self._placed, self._preps, self._digests = {}, {}, {}, {}

    monkeypatch.setattr(an.AnchoredMap, "begin_job", fresh)
    full, n_runs_full, n_full = job("b")
    assert n_runs_full == n_runs and n_reused < n_full
    assert reused["inputs"] == full["inputs"]
    assert {k: v for k, v in reused["quality"].items()} == full["quality"]


def test_the_job_caches_key_on_content(tmp_path: Any) -> None:
    """``AnchoredMap._digest`` (once per array, by content) and ``_kf_state`` (a keyframe
    without features or grid); ``_prepared`` once per frames, static pass and depth scale."""
    from tests.unit.test_anchored import frame, new_map

    am = new_map(tmp_path)
    a = np.arange(6.0)
    assert am._digest(None) is None
    assert am._digest(a) == am._digest(a.copy()) and am._digest(a) is am._digest(a)
    am.update([frame("a", base_pose(0.0), p=None, depth=False, mono=1.0)])
    k = am.keyframes[0]
    k.feats, k.grid, k.K_grid = None, None, None
    am._load_kf = lambda kf: None  # type: ignore[method-assign]
    assert am._kf_state(k)[10] is None
    fr = [frame("b", base_pose(10.0), p=None)]
    first = am._prepared(fr, True)
    assert am._prepared(fr, True) is first and am._prepared(fr, False) is not first


def test_a_client_label_that_is_also_a_default_wins_within_the_margin() -> None:
    """map.md §5 "Label choice" (oms/2.4): a label is a client label by the map's stored
    vocabulary, also when it equals a default (``egg``): at 0.30 it beats the default ``orange``
    at 0.40 (the gap within ``CLIENT_LABEL_MARGIN``), not one ahead by more; the best-share label
    stays when it is a client label; a map without client labels chooses as before."""
    from oh_my_slam.segmentation.detect import default_vocabulary
    from oh_my_slam.service.objects import CLIENT_LABEL_MARGIN, with_label_scores

    assert "egg" in default_vocabulary() and CLIENT_LABEL_MARGIN == 0.15
    votes = [{"label": "orange", "share": 0.4}, {"label": "egg", "share": 0.3}]
    rec = {"label": "orange", "detection_votes": votes, "detected_as": ["orange", "egg"]}
    got = with_label_scores({"objects": [rec]}, ["Egg"])["objects"][0]
    assert got["label"] == "egg" and got["detected_as"] == ["orange", "egg"]
    assert got["label_scores"] == [{"label": "egg", "score": 0.3},
                                   {"label": "orange", "score": 0.4}]
    far = {**rec, "detection_votes": [{"label": "orange", "share": 0.5}, votes[1]]}
    assert with_label_scores({"objects": [far]}, ["egg"])["objects"][0]["label"] == "orange"
    mine = {**rec, "label": "egg", "detection_votes": [{"label": "egg", "share": 0.5},
                                                       {"label": "orange", "share": 0.45}]}
    assert with_label_scores({"objects": [mine]}, ["egg", "orange"])["objects"][0]["label"] == "egg"
    assert with_label_scores({"objects": [rec]}, [])["objects"][0]["label"] == "orange"
    none = {"label": "box", "detection_votes": []}
    assert with_label_scores({"objects": [none]}, ["egg"])["objects"][0]["label"] == "box"
