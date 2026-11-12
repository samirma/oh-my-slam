"""oms/2.2 P15-P16 (round 8): drift support across passes and from kept conflicting frames, the
measurable-pair rule for prior conflicts on every engine, the drift sigma grown from the nearest
supporting frame, the pose duplicates that served a job undone by a second registration, the
cross-engine retries checked as the first engine's frames, and MapAnything re-inference when the
other engine adds anchors."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oh_my_slam.mapping import anchored as an
from oh_my_slam.mapping import register_ma
from oh_my_slam.mapping.anchored import AnchoredMap, Drift
from oh_my_slam.service import prior_first
from tests.synth.room import base_pose
from tests.unit.test_anchored import K, frame, new_map, prior
from tests.unit.test_service_api import CAMERA, Client, frames_of
from tests.unit.test_service_api import client as old  # noqa: F401
from tests.unit.test_service_prior_first import CAM, frames_for
from tests.unit.test_service_prior_first import client as _client  # noqa: F401


@pytest.fixture
def ma(_client: Client, monkeypatch: pytest.MonkeyPatch) -> Iterator[Client]:  # noqa: F811
    monkeypatch.setenv(prior_first.NONE_ENGINE_ENV, "mapanything")
    yield _client


def _status(job: dict[str, Any]) -> dict[str, tuple[str, str | None]]:
    return {i["name"]: (i["status"], i["reason"]) for i in job["inputs"]}


def _shifted(T: np.ndarray, dx: float) -> np.ndarray:
    W = T.copy()
    W[:3, 3] += [dx, 0.0, 0.0]
    return W


# -- R-9: drift support ------------------------------------------------------------------------


def test_a_left_out_frame_and_a_kept_conflicting_frame_can_still_support(tmp_path: Path
                                                                          ) -> None:
    """map.md §1 (R-9): a frame one fit left out keeps its overlap, so it counts in n and is an
    observation of the next pass; a kept ``conflict_unsupported`` frame that can support is an
    observation too (its image-only pose kept), one that cannot is not."""
    m = new_map(tmp_path)
    T = base_pose(0)
    fs = [frame(f"d{k}", base_pose(10 * k), p="drifting") for k in range(4)]
    out_fit = an._drift_outliers(Drift(an.pose(np.eye(3), np.array([3.0, 0, 0])), T), [
        an._DriftObs(0, "d0", T, T, (0.02, 1.0), (0.01, 0.5), False,
                     an.Placement("d0", "registered", T=T, inliers=150, inlier_share=0.3,
                                  measurable=True))], set())
    able = an.Placement("d1", "kept_at_prior", "no_overlap", T, 0.02, 1.0, 150,
                        detail="conflict_unsupported", align_T=T, image_sigmas=(0.01, 0.5),
                        inlier_share=0.3)
    weak = an.Placement("d2", "kept_at_prior", "no_overlap", T, 0.02, 1.0, 40,
                        detail="conflict_unsupported", align_T=T, image_sigmas=(0.01, 0.5),
                        inlier_share=0.3)
    obs = m._drift_evidence(fs, {0: out_fit[0], 1: able, 2: weak,
                                 3: an.Placement("d3", "rejected", "no_overlap")})
    assert [o.name for o in obs] == ["d0", "d1"]
    assert all(an._can_support(o.pl) for o in obs)


def test_the_measurable_pair_threshold_is_one_reference_s() -> None:
    """api.md §4 (P16): 100 inliers covering 10 % of the cells, on one reference."""
    xs, ys = np.meshgrid(np.linspace(5, 315, 20), np.linspace(5, 235, 6))
    uv = np.stack([xs.ravel(), ys.ravel()], axis=1)  # 120 inliers over the whole image
    assert an.measurable_pose(uv, np.zeros(len(uv), np.intp), K)
    halves = (np.arange(len(uv)) % 2).astype(np.intp)  # 60 on each of two references
    assert not an.measurable_pose(uv, halves, K)
    assert not an.measurable_pose(uv[:, :] * [0.05, 0.05], np.zeros(len(uv), np.intp), K)


# -- SI-13: prior conflicts on every engine ------------------------------------------------------


def test_the_older_mapper_keeps_a_conflict_below_the_measurable_threshold(
        old: Client, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """api.md §4 (P16) on the older mapper: a conflicting frame whose mapper inliers fall short
    of the measurable-pair threshold is kept at its prior, ``conflict_unsupported``, and the
    mapper runs again without it."""
    from oh_my_slam.service import update

    monkeypatch.setattr(update, "MEASURABLE_INLIERS", 10 ** 6)

    def pr(i: int, T: np.ndarray) -> dict[str, Any]:
        P = T.copy()
        if i == 2:
            P[:3, 3] += [0.0, 0.0, 0.5]
        return {"trust": "metric", "T_map_cam": P.tolist(), "sigma_pos_m": 0.02}

    frames, files, _ = frames_of(old, 6, prior=pr)
    job = old.update("m", frames, files, {"cam": CAMERA})
    assert job["state"] == "committed"
    bad = job["inputs"][2]
    assert (bad["status"], bad["reason"], bad["detail"]) == ("kept_at_prior", "no_overlap",
                                                             "conflict_unsupported")
    assert len(old.backend.calls) == 2 and len(old.backend.calls[1]) == 5


# -- SI-12: pose duplicates that served the job --------------------------------------------------


def test_a_duplicate_that_served_the_job_is_undone_by_a_second_registration(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """map.md §2 step 6 (P16): a rejected duplicate that gave scale evidence leaves none of it:
    the job registers once more without it, and ends as if it had never been sent."""
    runs: list[int] = []
    reg = an.AnchoredMap._register_drift

    def counted(self: AnchoredMap, *a: Any) -> Any:
        runs.append(1)
        return reg(self, *a)

    monkeypatch.setattr(an.AnchoredMap, "_register_drift", counted)
    m = new_map(tmp_path / "a")
    m.update([frame("s", base_pose(60)), frame("s1", base_pose(30))])
    runs.clear()
    pls = m.update([frame("f10", base_pose(0)), frame("f2", base_pose(0))])
    assert len(runs) == 2
    assert {pl.name: (pl.status, pl.reason) for pl in pls} == {
        "f2": ("registered", None), "f10": ("rejected", "duplicate")}
    assert pls[0].duplicate_of == "f2" and m.reran == set()
    ref = new_map(tmp_path / "b")
    ref.update([frame("s", base_pose(60)), frame("s1", base_pose(30))])
    ref.update([frame("f2", base_pose(0))])
    assert m.scale.depth_residuals == ref.scale.depth_residuals
    assert [k.name for k in m.keyframes] == [k.name for k in ref.keyframes]


def test_a_duplicate_that_served_nothing_needs_no_second_registration(tmp_path: Path) -> None:
    """Without any effect (no reference, drift, scale evidence or re-registration rests on
    it), the pass's rejection stands alone; the plan is made once per job."""
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0)), frame("a1", base_pose(30))])
    pls = m.update([frame("z", base_pose(15))], hold=True)
    b = frame("b", base_pose(15))
    m.add(b, an.features(b.rgb), an.Placement("b", "registered", None, base_pose(15), 0.01,
                                              0.1, 500, "mapanything"),
          b.depth, K, "sensor", None)  # type: ignore[arg-type]
    m._evidence_from.discard("z")
    assert not m._dup_effects(m._job_duplicates(), pls)
    assert not m.plan_rerun(pls)
    m._evidence_from.add("z")
    assert m.plan_rerun(pls) and "z" in m.excluded()
    assert not m.plan_rerun(pls)  # once per job
    m.end_job(pls, False)
    assert m.excluded() == {} and m.reran == set()


def test_what_a_duplicate_can_have_served(tmp_path: Path) -> None:
    """``_dup_effects``: a reference of another placement, a seeded component's seed, a
    re-registration, scale evidence, a drifting frame the drift took."""
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0))])
    m.begin_job()
    d = frame("d", base_pose(15))
    pl = an.Placement("d", "registered", None, base_pose(15), 0.01, 0.1, 500, "pnp")
    kf = m.add(d, an.features(d.rgb), pl, d.depth, K, "sensor", None)  # type: ignore[arg-type]
    dups = {"d": an.Placement("d", "rejected", "duplicate", keyframe=kf.id)}
    other = an.Placement("o", "registered", refs=(kf.id,))
    assert not m._dup_effects(dups, [pl])
    assert m._dup_effects(dups, [pl, other])
    member = m.keyframes[0]
    member.seed = kf.id
    assert m._dup_effects(dups, [pl])
    member.seed = None
    m._rereg_by["x"] = {kf.id}
    assert m._dup_effects(dups, [pl])
    m._rereg_by.clear()
    kf.prior = {"trust": "drifting", "T": base_pose(15).tolist(), "sigma_pos_m": 0.1,
                "sigma_rot_deg": 2.0}
    assert m._dup_effects(dups, [pl])


def test_a_re_registered_stored_keyframe_is_a_frame_of_the_job_for_the_pass(tmp_path: Path
                                                                            ) -> None:
    """map.md §2 step 6 (P16): a stored kept keyframe re-registered in this job is not a stored
    confirmed keyframe for the pass: named after a job frame at its pose, it is the duplicate,
    and its re-registration is undone (it stays kept as stored); the job frame registers."""
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0))])
    w = frame("w", base_pose(15))
    kept = m.add(w, an.features(w.rgb),
                 an.Placement("w", "kept_at_prior", "no_overlap", base_pose(15), 0.02, 1.0),
                 w.depth, K, "sensor", None)  # type: ignore[arg-type]
    m.begin_job()  # w is stored now
    assert not kept.usable()
    before = (kept.T.copy(), kept.state)
    m._rereg_state[kept.id] = (kept.T.copy(), kept.sigma_pos_m, kept.sigma_rot_deg, kept.state,
                               kept.method, kept.inliers, kept.rgb, kept.reason, kept.height_ok,
                               kept.seed, kept.seed_sigma,
                               (kept.floor_ok, kept.grid_scale, kept.floor_factor))
    kept.state, kept.reason = "fused", None
    m.reregistered = [{"name": "w", "keyframe": kept.id, "displacement_m": 0.0,
                       "displacement_deg": 0.0, "content": False, "objects": False}]
    v = frame("v", base_pose(15))
    pl = an.Placement("v", "registered", None, base_pose(15), 0.01, 0.1, 500, "pnp")
    m.add(v, an.features(v.rgb), pl, v.depth, K, "sensor", None)  # type: ignore[arg-type]
    m._defer = []
    m._evidence_from.add("w")  # it served the job: a second registration leaves it out
    assert m.plan_rerun([pl]) and m._no_rereg == {kept.id} and m.excluded() == {}
    dups, _ = m.end_job([pl], False)
    assert dups == {}  # the job frame v is no duplicate of the stored one
    assert m.reregistered == [] and kept.state == before[1] and np.allclose(kept.T, before[0])
    assert {k.name for k in m.keyframes} == {"a0", "w", "v"}


def test_a_planned_second_registration_skips_a_re_registration(tmp_path: Path) -> None:
    """A re-registered stored keyframe the plan leaves out (``_no_rereg``) stays kept."""
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0))])
    m.update([frame("w", base_pose(20), p=prior(base_pose(20)))], hold=False)
    kept = [k for k in m.keyframes if not k.usable()]
    m._no_rereg = {k.id for k in kept}
    assert m.reregister(kept, [m.keyframes[0]]) == []


def test_the_mapanything_path_registers_once_more_without_a_planned_frame(
        ma: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    """``run_anchored``: both engines run again from the job's start when a plan asks it, a
    MapAnything frame the plan forces taking its placement."""
    calls: list[int] = []

    def plan(self: AnchoredMap, pls: list[an.Placement]) -> bool:
        calls.append(1)
        if len(calls) > 1:
            return False
        self._exclude["n0"] = an.Placement("n0", "rejected", "duplicate", duplicate_of="a0")
        return True

    def dups(self: AnchoredMap) -> dict[str, an.Placement]:
        k = next((k for k in self.keyframes if k.name == "p0"
                  and k.id not in self._stored_ids), None)
        return {} if k is None else {"p0": an.Placement("p0", "rejected", "duplicate",
                                                        duplicate_of="a0", keyframe=k.id)}

    monkeypatch.setattr(an.AnchoredMap, "plan_rerun", plan)
    fr, files = frames_for(ma, [base_pose(0, 0.0)], tag="a")
    for f, T in zip(fr, [base_pose(0, 0.0)], strict=True):
        ma.backend.mv_add(files[f["image"]], T)
    assert ma.update("m", fr, files, cameras={"cam": CAM})["state"] == "committed"
    calls.clear()
    nposes = [base_pose(20, 0.0), base_pose(-20, 0.0)]
    nf, nfiles = frames_for(ma, nposes, tag="n", trust=None, depth=False)
    for f, T in zip(nf, nposes, strict=True):
        ma.backend.mv_add(nfiles[f["image"]], T)
    pf, pfiles = frames_for(ma, [base_pose(40, 0.0)], tag="p")
    monkeypatch.setattr(an.AnchoredMap, "_job_duplicates", dups)
    job = ma.update("m", nf + pf, {**nfiles, **pfiles})
    assert len(calls) == 2
    st = _status(job)
    assert st["n0"] == ("rejected", "duplicate") and st["p0"] == ("rejected", "duplicate")
    assert st["n1"][0] != "rejected" or st["n1"][1] != "duplicate"


# -- R-15: the cross-engine retries --------------------------------------------------------------


def test_a_retry_whose_images_now_conflict_takes_that_verdict(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """map.md §2, api.md §4 (R-15): a kept job frame whose images now give a pose beyond its
    prior is rejected ``prior_conflict`` (measurable; its keyframe dropped) or kept with
    ``conflict_unsupported`` (not), never left with a stale ``no_image_pose``."""
    for measurable in (True, False):
        m = new_map(tmp_path / str(measurable))
        m.update([frame("a0", base_pose(0))])
        T = base_pose(95)
        p = frame("p", T, p=prior(_shifted(T, 0.5)))
        pl = m.update([p])[0]
        assert (pl.status, pl.detail) == ("kept_at_prior", "no_image_pose")
        kp = next(k for k in m.keyframes if k.name == "p")
        n = frame("n", base_pose(90))
        nk = m.add(n, an.features(n.rgb),
                   an.Placement("n", "registered", None, base_pose(90), 0.01, 0.1, 500, "pnp"),
                   n.depth, K, "sensor", None)  # type: ignore[arg-type]
        m.candidates_for = lambda f, *a, nk=nk, **k: [nk]  # type: ignore[method-assign]
        if not measurable:
            monkeypatch.setattr(an, "MEASURABLE_INLIERS", 10 ** 6)
        got = m.retry_kept([(p, kp)], [nk])
        if measurable:
            assert (got["p"].status, got["p"].reason) == ("rejected", "prior_conflict")
            assert all(k is not kp for k in m.keyframes)
        else:
            assert (got["p"].status, got["p"].detail) == ("kept_at_prior",
                                                          "conflict_unsupported")
            assert got["p"].keyframe == kp.id and not kp.usable()
            # tried again: the same verdict
            assert m.retry_kept([(p, kp)], [nk])["p"].detail == "conflict_unsupported"
        monkeypatch.undo()


def test_a_retried_frame_goes_through_the_prior_checks(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    """R-15: a kept job frame confirmed by the other engine's frames is checked against the
    job's frames with a prior (``_against_priors``) and records its scale evidence."""
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0))])
    p = frame("p", base_pose(95))
    m.update([p])
    kp = next(k for k in m.keyframes if k.name == "p")
    n = frame("n", base_pose(90))
    nk = m.add(n, an.features(n.rgb),
               an.Placement("n", "registered", None, base_pose(90), 0.01, 0.1, 500, "pnp"),
               n.depth, K, "sensor", None)  # type: ignore[arg-type]
    m.candidates_for = lambda f, *a, **k: [nk]  # type: ignore[method-assign]
    checked: list[str] = []
    real = an.AnchoredMap._against_priors

    def spy(self: AnchoredMap, f: an.FrameIn, *a: Any) -> None:
        checked.append(f.name)
        real(self, f, *a)

    monkeypatch.setattr(an.AnchoredMap, "_against_priors", spy)
    far = frame("far", base_pose(200))
    m.update([far])
    kfar = next(k for k in m.keyframes if k.name == "far")
    got = m.retry_kept([(p, kp), (far, kfar)], [nk])
    assert "far" not in got  # still no image pose: as it was
    assert got["p"].status == "registered" and checked == ["p"]
    assert "p" in m._evidence_from and m._mine[-1][0] is p
    assert not m._retried_drifting  # a metric prior: no drift fit to run


def test_the_cross_engine_drift_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R-15: once a retry gave a drifting frame an image pose, the drift fit runs over the job;
    a change or a new frame left out plans one more registration from the fitted estimate, once
    per job."""
    m = new_map(tmp_path)
    fs = [frame(f"d{k}", base_pose(20 * k), p="drifting") for k in range(3)]
    pls = m.update(fs, hold=True)
    assert not m.plan_drift_rerun(fs, pls)  # no retry gave a drifting frame a pose
    m._retried_drifting = True
    assert not m.plan_drift_rerun(fs, pls)  # the fit agrees with the job's estimate
    fitted = Drift(base_pose(5), base_pose(0))
    monkeypatch.setattr(an.AnchoredMap, "_fit_drift", lambda self, *a: (fitted, {}))
    assert m.plan_drift_rerun(fs, pls) and m._start_drift is fitted
    assert not m.plan_drift_rerun(fs, pls)  # once per job
    m.reran.clear()
    left = an.Placement("d1", "rejected", "prior_conflict")
    monkeypatch.setattr(an.AnchoredMap, "_fit_drift", lambda self, *a: (None, {1: left}))
    assert m.plan_drift_rerun(fs, pls) and m.excluded()["d1"] is left
    m.reran.clear()
    same = {1: an.Placement("d1", pls[1].status, pls[1].reason, detail=pls[1].detail)}
    monkeypatch.setattr(an.AnchoredMap, "_fit_drift", lambda self, *a: (None, same))
    assert not m.plan_drift_rerun(fs, pls)  # left out already
    m._job_used = None
    assert not m.plan_drift_rerun(fs, pls)


def test_a_retried_drifting_frame_is_noted_for_the_drift_fit(tmp_path: Path) -> None:
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0), p="drifting")])
    p = frame("p", base_pose(95), p="drifting")
    m.update([p])
    kp = next(k for k in m.keyframes if k.name == "p")
    n = frame("n", base_pose(90))
    nk = m.add(n, an.features(n.rgb),
               an.Placement("n", "registered", None, base_pose(90), 0.01, 0.1, 500, "pnp"),
               n.depth, K, "sensor", None)  # type: ignore[arg-type]
    m.candidates_for = lambda f, *a, **k: [nk]  # type: ignore[method-assign]
    m.retry_kept([(p, kp)], [nk])
    assert m._retried_drifting


# -- MapAnything: the other engine's keyframes join the anchor pool -------------------------------


def test_the_other_engine_s_keyframes_become_anchors() -> None:
    """``Registration.add_anchors``: a frame inferred before is inferred again once the pool
    grew (``retry``, once per growth)."""
    reg = register_ma.Registration(lambda *a: [], [], [])
    assert reg.pool == []
    reg.add_anchors([register_ma.Stored("k1", Path("x.jpg"), K, np.eye(4), 7)])
    assert [v.key for v in reg.pool] == ["k1"] and reg.pool[0].order == 7


def test_a_drift_outlier_without_a_measurable_pose_is_kept_and_stored(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """P16 with R-9: a frame the drift fit leaves out whose image-only pose is short of the
    measurable-pair threshold is kept at its prior, ``conflict_unsupported``, stored as every
    kept frame, and still counts for the next fit."""
    from tests.unit.test_match_graph_engine import _drift_map, _rot_z

    monkeypatch.setattr(an, "measurable_pose", lambda *a: False)
    truth = [base_pose(y) for y in (40, 50, 70, 80, 100)]
    slips = [0.0, 0.0, 0.0, 25.0, 25.0]
    m = _drift_map(tmp_path)
    pls = m.update([frame(f"s{k}", T, p=prior(_rot_z(T, d), "drifting"))
                    for k, (T, d) in enumerate(zip(truth, slips, strict=True))])
    assert [(p.status, p.detail) for p in pls[3:]] == [("kept_at_prior",
                                                        "conflict_unsupported")] * 2
    assert {"s3", "s4"} <= {k.name for k in m.keyframes if not k.usable()}
