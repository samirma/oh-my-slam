"""oms/2.2 P11-P12 (round 6): the MapAnything path's distance limit and match graph, the
re-registration rules (distance limit, eligibility, height gate), the drift support rule, the
seeded component's sigmas, static content on textured frames, the measurable pairs of the
rotation check and standpoint groups, the too-close exception without a depth image, and the
older mapper's drift and kept-only updates."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oh_my_slam.mapping import anchored as an
from oh_my_slam.mapping import register_ma
from oh_my_slam.mapping.anchored import AnchoredMap
from oh_my_slam.service import prior_first
from tests.synth.room import base_pose
from tests.unit.test_anchored import K, frame, new_map, prior, shot
from tests.unit.test_service_api import Client
from tests.unit.test_service_api import client as old  # noqa: F401
from tests.unit.test_service_prior_first import CAM, frames_for
from tests.unit.test_service_prior_first import client as _client  # noqa: F401


@pytest.fixture
def ma(_client: Client, monkeypatch: pytest.MonkeyPatch) -> Iterator[Client]:  # noqa: F811
    monkeypatch.setenv(prior_first.NONE_ENGINE_ENV, "mapanything")
    yield _client


def ma_frames(c: Client, poses: list[np.ndarray], **kw: Any
              ) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    fr, files = frames_for(c, poses, **kw)
    for f, T in zip(fr, poses, strict=True):
        c.backend.mv_add(files[f["image"]], T)
    return fr, files


def _status(job: dict[str, Any]) -> dict[str, tuple[str, str | None]]:
    return {i["name"]: (i["status"], i["reason"]) for i in job["inputs"]}


def _poses(job: dict[str, Any]) -> dict[str, np.ndarray]:
    return {i["name"]: np.array(i["T_map_cam"]) for i in job["inputs"]
            if i["T_map_cam"] is not None}


# -- the MapAnything path ------------------------------------------------------------------------


def test_depth_agreement_never_overrides_too_close_on_mapanything(
        ma: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    """api.md §4: MapAnything's agreeing depth points are no image feature inliers: a frame
    whose depth agrees (thousands of points) but whose scene lies within 0.30 m is rejected
    too_close."""
    fr, files = ma_frames(ma, [base_pose(0, 0.0)], tag="a")
    assert ma.update("m", fr, files, cameras={"cam": CAM})["state"] == "committed"
    run = register_ma.Registration.run

    def close(self: register_ma.Registration) -> register_ma.Result:
        res = run(self)
        for p in res.placements.values():
            p.scene_m = 0.2
        return res

    monkeypatch.setattr(register_ma.Registration, "run", close)
    nf, nfiles = ma_frames(ma, [base_pose(20, 0.0)], tag="n", trust=None, depth=False)
    job = ma.update("m", nf, nfiles)
    assert _status(job) == {"n0": ("rejected", "too_close")}
    assert job["inputs"][0]["scene_distance_m"] == 0.2


def test_one_match_graph_over_both_engines(ma: Client) -> None:
    """map.md §2: a frame with a prior that only a MapAnything frame of the same job reaches is
    placed again once that frame registers (one fixed point over both engines), whatever the
    request order."""
    fr, files = ma_frames(ma, [base_pose(0, 0.0)], tag="a")
    nf, nfiles = ma_frames(ma, [base_pose(40, 0.0)], tag="n", trust=None, depth=False)
    pf, pfiles = ma_frames(ma, [base_pose(90, 0.0)], tag="p")
    out = []
    for name, order in (("m1", nf + pf), ("m2", pf + nf)):
        assert ma.update(name, fr, files, cameras={"cam": CAM})["state"] == "committed"
        job = ma.update(name, order, {**nfiles, **pfiles})
        assert _status(job) == {"n0": ("registered", None), "p0": ("registered", None)}, job
        out.append(_poses(job))
    for n, T in out[0].items():
        dp, dr = an.pose_residual(T, out[1][n])
        assert dp < 0.01 and dr < 0.3
    assert np.linalg.norm(out[0]["p0"][:3, 3] - base_pose(90, 0.0)[:3, 3]) < 0.03


def test_mapanything_standpoints_do_not_depend_on_the_request_order(ma: Client) -> None:
    """map.md §1, §2: a turn in place registered by MapAnything, shuffled: the same standpoint
    (centre and statuses), the same poses."""
    out = []
    poses = [base_pose(25.0 * k, offset=0.0) for k in range(5)]
    for name, order in (("s1", [0, 1, 2, 3, 4]), ("s2", [0, 3, 1, 4, 2])):
        fr, files = ma_frames(ma, [poses[k] for k in order], tag=name, trust=None, depth=False,
                              height_above_floor_m=0.5)
        for f, k in zip(fr, order, strict=True):  # names in capture order whatever the order
            f["name"] = f"t{k}"
        job = ma.update(name, fr, files, cameras={"cam": CAM})
        assert job["state"] == "committed", job
        out.append(job)
    a, b = _poses(out[0]), _poses(out[1])
    assert _status(out[0]) == _status(out[1])
    for n, T in a.items():
        dp, dr = an.pose_residual(T, b[n])
        assert dp < 0.01 and dr < 0.3, n


# -- re-registration -----------------------------------------------------------------------------


def test_re_registration_reapplies_the_distance_limit(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """map.md §2 with api.md §4: a stored frame kept too_close, overlapped by a new frame, is
    placed again with the distance limit: under 300 image inliers it stays kept."""
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0))])
    monkeypatch.setattr(an, "TOO_CLOSE_M", 100.0)
    monkeypatch.setattr(an, "TOO_CLOSE_INLIERS", 10 ** 6)
    assert m.update([frame("c", base_pose(40))])[0].reason == "too_close"
    m.save()
    m = AnchoredMap.load(m.root)
    pls = m.update([frame("b", base_pose(42), depth=False, mono=1.0)])
    assert pls[0].status == "registered" and m.reregistered == []
    kept = m.keyframes[1]
    assert (kept.state, kept.reason) == ("provisional", "too_close")


def test_re_registration_eligibility_and_the_height_gate(tmp_path: Path) -> None:
    """map.md §2: eligibility is the coverage of the kept frame's image by a new frame's matches
    (no pose test: a kept frame whose prior turned it away is matched all the same); the height
    gate runs again at the re-registered pose."""
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0))])
    T = base_pose(20)
    off = T.copy()
    off[2, 3] += 0.10
    k = frame("k", T, p=prior(off, sp=0.3, sr=2.0), depth=False, mono=1.0)
    k.unconfirmed = "blur"
    k.height_m, k.height_sigma_m = float(off[2, 3]), 0.005  # consistent at its prior
    assert m.update([k])[0].status == "kept_at_prior"
    kept = m.keyframes[-1]
    assert kept.height_ok and (kept.height_m, kept.height_sigma_m) == (float(off[2, 3]), 0.005)
    m.save()
    m = AnchoredMap.load(m.root)
    kept = m.keyframes[-1]
    assert kept.height_m == float(off[2, 3])
    turned = an.Keyframe("x", "x", "cam", K, base_pose(180), 0.01, 0.1, "fused", None, "none")
    turned.feats = an.features(shot(base_pose(22))[0])
    saved = kept.T.copy()
    kept.T = base_pose(200)  # its prior's pose faces away: still eligible by its image
    assert m.overlaps(kept, turned)
    kept.T = saved
    pls = m.update([frame("b", base_pose(25))])
    assert pls[0].status == "registered" and [e["name"] for e in m.reregistered] == ["k"]
    kept = m.keyframe(kept.id)
    assert kept is not None and kept.usable() and not kept.height_ok
    assert "k" in m.height_inconsistent and not kept.contributes() and not kept.detects()


def test_eligibility_asks_the_registration_minimum_of_inliers() -> None:
    """S-1 (map.md §2): an epipolar fit for eligibility needs 15 matches, not 30."""
    a = an.features(shot(base_pose(0))[0])
    b = an.features(shot(base_pose(10))[0])
    ib, ia = an.match(b.desc, a.desc)
    few_a = an.Features(a.kps[ia[:20]], a.desc[ia[:20]])
    few_b = an.Features(b.kps[ib[:20]], b.desc[ib[:20]])
    assert an.epipolar(few_a, K, few_b, K) is None
    e = an.epipolar(few_a, K, few_b, K, an.MIN_INLIERS)
    assert e is not None and len(e.ua) >= an.MIN_INLIERS


# -- the too-close exception without a depth image -----------------------------------------------


def test_a_too_close_frame_without_depth_image_gives_detections_only(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """api.md §4: kept only for too_close with a metric prior and no depth image: its detections
    and votes count (``detects``), its depth never reaches the cloud (``contributes``)."""
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0))])
    monkeypatch.setattr(an, "TOO_CLOSE_M", 100.0)
    monkeypatch.setattr(an, "TOO_CLOSE_INLIERS", 10 ** 6)
    n = len(m.cloud.keys)
    f = frame("c", base_pose(30), depth=False, mono=1.0)
    f.check_close = True
    assert (m.update([f])[0].status, m.keyframes[-1].reason) == ("kept_at_prior", "too_close")
    kf = m.keyframes[-1]
    assert kf.depth_kind == "mono" and kf.detects() and not kf.contributes()
    assert len(m.cloud.keys) == n
    d = frame("d", base_pose(50), p="drifting", depth=False, mono=1.0)
    d.check_close = True
    m.update([d])
    assert not m.keyframes[-1].detects()  # a drifting prior: nothing


# -- canonical order -----------------------------------------------------------------------------


def test_a_map_without_priors_starts_at_its_first_frame_by_name(tmp_path: Path) -> None:
    """map.md §1 (P11): without priors the origin is the first frame in canonical name order."""
    m = new_map(tmp_path)
    pls = m.update([frame("f10", base_pose(20, pitch_deg=0.0), p=None, depth=False, mono=1.0),
                    frame("f2", base_pose(0, pitch_deg=0.0), p=None, depth=False, mono=1.0)])
    assert pls[1].method == "origin" and pls[0].method != "origin"


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_a_job_duplicate_is_the_later_frame_by_name(tmp_path: Path, order: tuple[int, int]
                                                    ) -> None:
    """api.md §4, map.md §2 step 6 (P13): one pose-duplicate pass after the registration fixed
    point, in canonical name order: of a pair the later-named frame is the duplicate, whatever
    the request order or the round it registered in; a stored keyframe always counts. The
    duplicate's keyframe is dropped and gives no content."""
    m = new_map(tmp_path)
    m.update([frame("s", base_pose(60)), frame("s1", base_pose(30))])
    fs = [frame("f10", base_pose(0)), frame("f2", base_pose(0)), frame("a", base_pose(60))]
    pls = m.update([fs[order[0]], fs[order[1]], fs[2]])
    got = {pl.name: (pl.status, pl.reason, pl.duplicate_of) for pl in pls}
    assert got == {"f2": ("registered", None, None), "f10": ("rejected", "duplicate", "f2"),
                   "a": ("rejected", "duplicate", "s")}
    assert sorted(k.name for k in m.keyframes) == ["f2", "s", "s1"]
    assert all(pl.keyframe is None for pl in pls if pl.status == "rejected")


def test_a_held_job_ends_with_one_duplicate_pass_over_both_engines(tmp_path: Path) -> None:
    """The prior-first engine holds the job open for the other engine's frames
    (``update(hold=True)``): nothing is fused nor tested for duplicates until ``end_job``,
    which rejects the later-named frame of a pair across engines."""
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0)), frame("a1", base_pose(30))])
    n = len(m.cloud.keys)
    pls = m.update([frame("z", base_pose(15))], hold=True)
    assert pls[0].status == "registered" and len(m.cloud.keys) == n  # held
    b = frame("b", base_pose(15))  # the other engine's frame, named before z
    m.add(b, an.features(b.rgb), an.Placement("b", "registered", None, base_pose(15), 0.01,
                                              0.1, 500, "mapanything"),
          b.depth, K, "sensor", None)  # type: ignore[arg-type]
    dups, _ = m.end_job(pls, False)
    assert (dups["z"].reason, dups["z"].duplicate_of) == ("duplicate", "b")
    assert "z" not in {k.name for k in m.keyframes} and len(m.cloud.keys) > n


def test_the_drift_reference_is_the_last_frame_by_name(tmp_path: Path) -> None:
    """map.md §2 (audit r6 item 4c): the drift estimate kept for later jobs does not depend on
    the request order."""
    got = []
    for name, order in (("a", (0, 1)), ("b", (1, 0))):
        m = new_map(tmp_path / name)
        m.update([frame("a0", base_pose(0), p="drifting")])
        fs = [frame("s2", base_pose(20), p="drifting"), frame("s10", base_pose(40), p="drifting")]
        m.update([fs[k] for k in order])
        assert m.drift.ref_prior is not None
        got.append(m.drift.ref_prior)
    assert np.allclose(got[0], got[1]) and np.allclose(got[0], base_pose(40))


# -- the drift estimate --------------------------------------------------------------------------


def test_one_strong_frame_alone_never_makes_a_drift_estimate(tmp_path: Path) -> None:
    """map.md §1 (P12, S-5): a first estimate needs min(3, n) supporting frames, at least half
    of those that can support it, never one alone: a lone drifting frame 25° off on a metric
    map is left out (``prior_conflict``); three that agree make the estimate."""
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0)), frame("a1", base_pose(30))])
    T = base_pose(20)
    p = m.update([frame("d", T, p=prior(_rot_z(T, 25.0), "drifting"))])[0]
    assert (p.status, p.reason) == ("rejected", "prior_conflict")
    m2 = new_map(tmp_path / "three")
    m2.update([frame("a0", base_pose(0)), frame("a1", base_pose(30))])
    truth = [base_pose(y) for y in (10, 20, 40)]
    pls = m2.update([frame(f"d{k}", T, p=prior(_rot_z(T, 25.0), "drifting"))
                     for k, T in enumerate(truth)])
    assert [p.status for p in pls] == ["registered"] * 3


def test_who_can_support_a_drift_estimate() -> None:
    """map.md §1 (P13, SI-1): an image-only pose with 100 inliers across all its references and
    a 10 % overlap with one of them, whatever its status; an origin or a seed has no image-only
    pose."""
    T = base_pose(0)

    def pl(n: int, share: float, status: str = "registered") -> an.Placement:
        return an.Placement("x", status, T=T, inliers=n, method="pnp", inlier_share=share)

    assert an._can_support(pl(100, 0.1))
    assert an._can_support(pl(300, 0.2, "kept_at_prior"))  # whatever its final status
    assert not an._can_support(pl(99, 0.5))
    assert not an._can_support(pl(500, 0.05))
    assert not an._can_support(an.Placement("o", "registered", T=T, inliers=0, method="origin"))
    # the overlap is with one reference: two references at 6 % each do not make 12 %
    uv = np.array([[10.0, 10.0], [50.0, 10.0], [90.0, 10.0], [130.0, 10.0], [170.0, 10.0],
                   [10.0, 50.0], [50.0, 50.0], [90.0, 50.0]])
    one = an.ref_share(uv, np.zeros(8, np.intp), K)
    two = an.ref_share(uv, np.array([0, 1, 0, 1, 0, 1, 0, 1], np.intp), K)
    assert one > two > 0.0 and an.ref_share(uv[:0], np.zeros(0, np.intp), K) == 0.0


def test_the_drift_passes_end_with_the_last_fit_applied(tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """Audit r6 item 12: at the pass cap the last fit's outliers are rejected, never left
    registered under an estimate that left them out."""
    monkeypatch.setattr(an, "DRIFT_PASSES", 1)
    calls = []
    fit = an.AnchoredMap._fit_drift

    def counted(self: AnchoredMap, *a: Any) -> Any:
        calls.append(1)
        return fit(self, *a)

    monkeypatch.setattr(an.AnchoredMap, "_fit_drift", counted)
    truth = [base_pose(y) for y in (40, 50, 70, 80)]
    good = [frame(f"g{k}", T, p=prior(T, "drifting", 0.15, 7.4)) for k, T in enumerate(truth)]
    m = new_map(tmp_path)
    m.update([frame(f"a{k}", base_pose(30 * k), p="drifting") for k in range(4)])
    calls.clear()
    bad_T = base_pose(55)
    bad = frame("bad", bad_T, p=prior(_rot_z(bad_T, 86.0), "drifting", 0.15, 7.4))
    pls = m.update([*good[:2], bad, *good[2:]])
    assert (pls[2].status, pls[2].reason) == ("rejected", "prior_conflict")
    assert len(calls) == 1 and all(p.status == "registered" for p in [*pls[:2], *pls[3:]])


def _rot_z(T: np.ndarray, deg: float) -> np.ndarray:
    a = np.radians(deg)
    R = np.eye(4)
    R[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    return R @ T


# -- a seeded component --------------------------------------------------------------------------


def test_a_seeded_component_is_never_tighter_than_its_seed(tmp_path: Path) -> None:
    """map.md §2 step 2 (P11, P13): a placement whose references are all in a seeded component
    reports the seed prior's sigma combined with its own, its keyframe too, and the height gate
    runs on that; a placement resting also on a confirmed keyframe outside it links the
    component: its floor ends."""
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0))])
    a0 = m.keyframes[0]
    s = an.Keyframe("k8", "s", "cam", K, base_pose(90), 0.05, 2.0, "fused", None, "none",
                    seed="k8", seed_sigma=(0.05, 2.0))
    m.keyframes.append(s)
    f = frame("x", base_pose(95), p=None)
    f.height_m, f.height_sigma_m = float(base_pose(95)[2, 3]) + 0.05, 0.001
    pl = an.Placement("x", "registered", T=base_pose(95), sigma_pos_m=0.004, sigma_rot_deg=0.2,
                      refs=("k8",))
    kf = m.add(f, an.features(f.rgb), pl, f.depth, K, "sensor", None)  # type: ignore[arg-type]
    assert pl.sigma_pos_m == pytest.approx(round(np.hypot(0.05, 0.004), 4))
    assert pl.sigma_rot_deg >= 2.0  # type: ignore[operator]
    assert (kf.sigma_pos_m, kf.seed, kf.seed_sigma) == (pl.sigma_pos_m, "k8", (0.05, 2.0))
    assert kf.height_ok and "x" not in m.height_inconsistent  # within its looser sigma
    assert an.Keyframe.from_json(kf.to_json()).seed_sigma == (0.05, 2.0)
    # through the component and a confirmed keyframe outside it: linked, no floor any more
    other = an.Placement("y", "registered", T=base_pose(95), sigma_pos_m=0.001,
                         sigma_rot_deg=0.01, refs=("k8", a0.id))
    assert m._seed_floor(other) is None and other.sigma_pos_m == 0.001
    assert s.seed is None and kf.seed is None
    assert m._seed_floor(an.Placement("z", "registered", sigma_pos_m=0.001, sigma_rot_deg=0.01,
                                      refs=("k8",))) is None


# -- static content ------------------------------------------------------------------------------


def test_static_content_on_textured_frames() -> None:
    """map.md §1 (audit r6 item 5): frames of thousands of keypoints each (more than the old
    all-pairs bound) still lose the keypoints fixed across three frames turned pairwise beyond
    5°; the scene's keep."""
    rng = np.random.default_rng(3)
    fixed = rng.uniform(0, 300, (5, 2)).astype(np.float32)
    d = rng.random((5, 128)) + 0.01
    fd = np.round(np.sqrt(d / d.sum(1, keepdims=True)) * an.ROOTSIFT_QUANT).astype(np.uint8)
    fs, prep = [], []
    for k in range(3):
        f = frame(f"t{k}", base_pose(10 * k))
        kps = np.vstack([rng.uniform(0, [K.width, K.height], (3000, 2)).astype(np.float32),
                         fixed])
        r = rng.random((3000, 128)) + 0.01
        desc = np.vstack([np.round(np.sqrt(r / r.sum(1, keepdims=True))
                                   * an.ROOTSIFT_QUANT).astype(np.uint8), fd])
        fs.append(f)
        prep.append(an._Prep(an.Features(kps, desc), None, False, None))
    an.static_content(fs, prep)
    for p in prep:
        assert len(p.feats.kps) == 3000
        assert not any(np.allclose(q, fixed[0]) for q in p.feats.kps)
    got = an.job_features(fs)  # the whole job's (both engines): set on each frame
    assert all(f.feats is g for f, g in zip(fs, got, strict=True))


def test_near_pairs() -> None:
    a = np.array([[0.0, 0.0], [10.0, 10.0], [2.5, 2.0]])
    b = np.array([[1.0, 1.0], [10.5, 9.0], [50.0, 50.0], [-1.0, 0.0]])
    ia, ib = an.near_pairs(a, b, 2.0)
    assert sorted(zip(ia.tolist(), ib.tolist(), strict=True)) == [(0, 0), (0, 3), (1, 1),
                                                                   (2, 0)]
    assert len(an.near_pairs(a, np.zeros((0, 2)), 2.0)[0]) == 0
    assert len(an.near_pairs(a, np.array([[100.0, 100.0]]), 2.0)[0]) == 0


# -- the rotation check --------------------------------------------------------------------------


def test_the_rotation_check_measures_every_pair_that_can_share_a_view() -> None:
    """map.md §4 (audit r6 item 7): a pair beyond the six nearest views is measured too (no
    count cap), only measurable pairs count (100 inliers and 10 % overlap), and the pairs do
    not depend on the request order. (The stored pairs give the new frame's pairs their chains,
    P27, so they stay inside the pair limits, P28b.)"""
    from oh_my_slam.mapping import standpoint as sp

    empty = an.Features(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8))
    R = base_pose(0)[:3, :3]
    bad = an.exp_so3(np.array([0.0, 0.0, np.radians(20.0)]))
    new = sp.View("n", K, empty, R, np.zeros(3))
    # the farthest committed 20° off what the images measure
    stored = [sp.View(f"s{k}", K, empty, bad @ R if k == 7 else R,
                      np.array([0.3 * (k + 1), 0.0, 0.0]), True) for k in range(8)]
    pairs = sp.Pairs()
    for v in stored:
        pairs.fits[("n", v.name)] = sp.TwoView(np.eye(3), 5.0, 500 if v.name == "s0" else 200,
                                               0.5, 0.01)
        for w in stored:
            pairs.fits[(v.name, w.name)] = sp.TwoView(np.eye(3), 5.0, 300, 0.5, 0.01)
    med, mx = sp.consistency([*stored, new], pairs)
    assert mx is not None and mx > 19.0 and sp.inconsistent(med, mx)
    pairs.fits[("n", "s7")] = sp.TwoView(np.eye(3), 5.0, 99, 0.5, 0.01)  # not measurable
    assert sp.consistency([new, *reversed(stored)], pairs)[1] == pytest.approx(0.0, abs=1e-6)
    pairs.fits[("n", "s7")] = sp.TwoView(np.eye(3), 5.0, 500, 0.09, 0.01)  # overlaps too little
    assert sp.consistency([*stored, new], pairs)[1] == pytest.approx(0.0, abs=1e-6)
    assert not sp.TwoView(np.eye(3), 0.0, 99, 0.5, 0.01).standpoint  # grouping too


# -- the older mapper (map.md §1 "Every engine", P12 S-6) -----------------------------------------


def _metric(i: int, T: np.ndarray) -> dict[str, Any]:
    return {"trust": "metric", "T_map_cam": T.tolist(), "sigma_pos_m": 0.01, "sigma_rot_deg": 0.5}


def test_the_older_mapper_accepts_a_kept_only_update(old: Client) -> None:  # noqa: F811
    """map.md §4: a later update whose frames are all kept_at_prior is accepted and stores them
    (the mapper's map as it was)."""
    from tests.unit.test_service_api import CAMERA, WORLD, frames_of, textured

    frames, files, _ = frames_of(old, 3, prior=_metric)
    assert old.update("m", frames, files, {"cam": CAMERA})["state"] == "committed"
    files = {"lost": textured(91)}  # unknown to the fake mapper: not placed
    frames = [{"name": "lost", "image": "lost", "camera_id": "cam", "prior": _metric(0, WORLD[4])}]
    job = old.update("m", frames, files)
    assert job["state"] == "committed" and job["result_version"] == 2, job["quality"]
    assert (job["inputs"][0]["status"], job["inputs"][0]["reason"]) == ("kept_at_prior",
                                                                        "no_overlap")
    assert job["inputs"][0]["detail"] == "no_image_pose"
    rows = old.get("/api/maps/m/frames", "Frames")["frames"]
    assert [r["pose_state"] for r in rows] == ["fused"] * 3 + ["kept_at_prior"]
    assert job["scale_source"] == "priors"


def test_the_older_mapper_keeps_its_drift_estimate_across_versions(old: Client) -> None:  # noqa: F811
    """map.md §1 "Drift estimate", every engine: three drifting frames make the estimate (in
    the version record); a later lone frame whose odometry slipped cannot change it and is
    rejected prior_conflict; a kept drifting frame takes the prior's sigma plus the estimate's
    own."""
    from oh_my_slam.mapping.anchor import Sim3
    from tests.unit.test_service_api import CAMERA, WORLD, anchor_rot, frames_of, textured

    frames, files, _ = frames_of(old, 3, prior=_metric)
    assert old.update("m", frames, files, {"cam": CAMERA})["state"] == "committed"
    drift = Sim3(1.0, anchor_rot(-30), np.array([5.0, 5.0, 0.0]))

    def drifting(sim: Sim3) -> Any:
        return lambda i, T: {"trust": "drifting", "T_map_cam": sim.pose(T).tolist(),
                             "sigma_pos_m": 0.05, "sigma_rot_deg": 2.0}

    frames, files, truth = frames_of(old, 3, start=3, prior=drifting(drift))
    job = old.update("m", frames, files)
    assert [i["status"] for i in job["inputs"]] == ["registered"] * 3, job["inputs"]
    rec = old.service.store.version("m")
    assert rec["drift_sim"] is not None and len(rec["drift_sigma"]) == 2
    slip = Sim3(1.0, anchor_rot(-60), np.array([5.0, 5.0, 0.0]))
    frames, files, _ = frames_of(old, 1, start=1, prior=drifting(slip), seed=2)
    files["lost"] = textured(91)
    frames.append({"name": "lost", "image": "lost", "camera_id": "cam",
                   "prior": drifting(drift)(4, WORLD[4])})
    job = old.update("m", frames, files)
    st = {i["name"]: i for i in job["inputs"]}
    assert (st["f2_1"]["status"], st["f2_1"]["reason"]) == ("rejected", "prior_conflict")
    lost = st["lost"]
    assert lost["status"] == "kept_at_prior" and lost["sigma_pos_m"] > 0.05
    assert centre_error(lost["T_map_cam"], WORLD[4]) < 0.02


def centre_error(T: Any, ref: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(T)[:3, 3] - ref[:3, 3]))


def test_locate_takes_the_frame_s_own_height_sigma() -> None:
    """api.md §6 (P12, S-7): height_sigma is the locate frame's own when it carries one, even
    when the height is its camera's registered one."""
    from oh_my_slam.service import nav
    from oh_my_slam.service.cameras import Camera

    cam = Camera(160.0, 160.0, 160.0, 120.0, 320, 240, height_above_floor_m=0.5,
                 height_sigma_m=0.02)
    img = np.asarray(shot(base_pose(0))[0], np.uint8)
    fr = nav.LocateFrame("q", cam, img, None, None, None, 0.004)
    q, _ = nav._query(fr, 0.1, True)
    assert (q.camera_z, q.camera_z_sigma) == (pytest.approx(0.6), 0.004)
    fr = nav.LocateFrame("q", cam, img, None, None, None, None)
    q, _ = nav._query(fr, 0.1, True)  # P15: never the camera's; the README default meanwhile
    assert (q.camera_z_sigma, q.camera_z_sigma_given) == (0.02, False)
    fr = nav.LocateFrame("q", Camera(160.0, 160.0, 160.0, 120.0, 320, 240), img, None, None,
                         None, 0.004)
    assert nav._query(fr, 0.1, True)[0].camera_z is None  # no height at all


def test_retry_kept_places_the_job_s_kept_frames_against_new_keyframes(tmp_path: Path) -> None:
    """``AnchoredMap.retry_kept`` (map.md §2, one match graph over both engines): a kept frame
    whose candidates hold a new keyframe is placed again and registered (fused, the height gate
    run; pose duplicates are the job's last pass, ``end_job``); one not placed, kept for a
    failed check, confirmed already, or with no new candidate stays as it was."""
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0))])
    p = frame("p", base_pose(90))
    p.height_m, p.height_sigma_m = float(base_pose(90)[2, 3]) + 1.0, 0.001  # inconsistent
    p2 = frame("p2", base_pose(100), depth=False)  # no depth: nothing to fuse
    blank = frame("q", base_pose(60), rgb=np.full((K.height, K.width, 3), 120, np.uint8))
    bad = frame("r", base_pose(70))
    bad.unconfirmed = "blur"
    pls = m.update([p]) + m.update([p2, blank, bad])  # apart: no component of their own
    assert [(x.status, x.reason) for x in pls] == [("kept_at_prior", "no_overlap")] * 3 + [
        ("kept_at_prior", "blur")]
    kfs = {k.name: k for k in m.keyframes}
    n = frame("n", base_pose(45))
    nk = m.add(n, an.features(n.rgb),
               an.Placement("n", "registered", None, base_pose(45), 0.01, 0.1, 500, "pnp"),
               n.depth, K, "sensor", None)  # type: ignore[arg-type]
    zd = frame("zd", base_pose(45))
    zk = m.add(zd, an.features(zd.rgb),
               an.Placement("zd", "kept_at_prior", "no_overlap", base_pose(45), 0.02, 1.0),
               zd.depth, K, "sensor", None)  # type: ignore[arg-type]
    kept = [(p, kfs["p"]), (p2, kfs["p2"]), (blank, kfs["q"]), (bad, kfs["r"]),
            (frame("a0", base_pose(0)), kfs["a0"]), (zd, zk)]
    assert m.retry_kept(kept, []) == {}  # nothing new among their candidates
    n_cloud = len(m.cloud.keys)
    got = m.retry_kept(kept, [nk])
    assert set(got) == {"p", "p2", "zd"}
    assert got["p"].status == got["p2"].status == got["zd"].status == "registered"
    assert zk.usable() and kfs["p"].usable() and kfs["p2"].usable()
    assert not kfs["p"].height_ok and m.height_inconsistent == ["p"]
    assert not kfs["q"].usable() and not kfs["r"].usable()
    # p fails the height gate, p2 has no depth, zd's points fall in n's voxels
    assert len(m.cloud.keys) == n_cloud


def test_retry_kept_iterates_to_a_fixed_point(tmp_path: Path) -> None:
    """map.md §2 (audit r7 item 3): a kept frame X that links to the map only through a kept
    frame Y, which ``retry_kept`` confirms later in name order, is placed again once Y is
    confirmed: X ends registered."""
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0))])
    x, y = frame("a1", base_pose(100)), frame("b1", base_pose(95))
    pls = m.update([x]) + m.update([y])  # apart from a0: kept
    assert [pl.status for pl in pls] == ["kept_at_prior"] * 2
    kfs = {k.name: k for k in m.keyframes}
    n = frame("n", base_pose(90))
    nk = m.add(n, an.features(n.rgb),
               an.Placement("n", "registered", None, base_pose(90), 0.01, 0.1, 500, "pnp"),
               n.depth, K, "sensor", None)  # type: ignore[arg-type]
    cands = {"a1": [kfs["b1"]], "b1": [nk]}  # X sees only Y, Y sees the new keyframe
    m.candidates_for = lambda f, *a, **k: cands[f.name]  # type: ignore[method-assign]
    got = m.retry_kept([(x, kfs["a1"]), (y, kfs["b1"])], [nk])
    assert got["b1"].status == got["a1"].status == "registered"
    assert kfs["a1"].usable() and kfs["b1"].usable()


def test_mapanything_infers_each_frame_once_per_job(ma: Client) -> None:
    """A frame MapAnything places is inferred once per job; one it does not place is tested
    again against what registered since from its output (``Registration.recheck``) and
    inferred again only when the pool of anchors grew since its call (``retry``, round 7: the
    real sweeps lost frames their first call's anchors did not see), once per growth."""
    fr, files = ma_frames(ma, [base_pose(0, -0.6), base_pose(0, 0.0), base_pose(0, 0.6)],
                          tag="a")
    assert ma.update("m", fr, files, cameras={"cam": CAM})["state"] == "committed"
    nf, nfiles = ma_frames(ma, [base_pose(0, -0.3), base_pose(0, 0.3), base_pose(5, 0.1)],
                           tag="n", trust=None, depth=False)
    bad, bfiles = frames_for(ma, [base_pose(0, 0.2)], tag="x", trust=None, depth=False)
    import hashlib

    ma.backend.mv_pose[hashlib.sha256(bfiles["x0"]).hexdigest()] = base_pose(180, 0.2)
    ma.backend.mv_calls.clear()
    job = ma.update("m", nf + bad, {**nfiles, **bfiles})
    assert [i["status"] for i in job["inputs"]] == ["registered"] * 3 + ["rejected"]
    stored = {p.name for p in (ma.service.store.mapper_dir("m") / an.KF_DIR).glob("*.jpg")}
    new = [n for call in ma.backend.mv_calls for n in call if n not in stored]
    # the first call infers the four; the one more infers x0 alone (its anchors: the frames
    # registered since), and nothing after it (the pool did not grow again)
    assert len(set(new)) == 4 and len(ma.backend.mv_calls) == 2, ma.backend.mv_calls
    assert ma.backend.mv_calls[1][-1] not in ma.backend.mv_calls[1][:-1]
    assert all(n in stored or n in ma.backend.mv_calls[0] for n in ma.backend.mv_calls[1])


@pytest.mark.parametrize("engine", ["mapanything", "geometric"])
def test_a_pure_turn_in_45_degree_steps_passes_the_rotation_check(
        _client: Client, monkeypatch: pytest.MonkeyPatch, engine: str) -> None:  # noqa: F811
    """The replay's real sweeps (about 45° steps, 7-8 frames, the camera ahead of the robot's
    axis, a few frames rejected): at true committed rotations the image-measured relative
    rotations were 1.5-6.2° off (the essential matrix of the best RANSAC sample, the
    rotation-only fit's absorbed translation), enough for rotation_inconsistent on real noise.
    The refined relative pose (``standpoint.refined_rotation``) agrees within 2°.
    Each measured pair is logged (``standpoint.measured``). Its eight adjacent pairs close one
    cycle, so only the weakest has a chain: a pair both of whose frames' pairs have none is
    outside the pair limits (P28b); every measurable pair is checked here (``judged``)."""
    from oh_my_slam.mapping import standpoint as sp
    from tests.fakes.service import plain

    monkeypatch.setenv(prior_first.NONE_ENGINE_ENV, engine)
    got: list[sp.PairCheck] = []
    real = sp.judged

    def logged(views: Any, pairs: Any) -> list[sp.Judged]:
        out = real(views, pairs)
        if not any(v.stored for v in views):  # the rotation check's (the joins' have stored)
            for g in out:
                v, w = views[g.i], views[g.j]
                got.append(sp.PairCheck(v.name, w.name, an.rot_deg((v.R.T @ w.R).T @ g.tv.R_ab),
                                        g.tv.inliers, g.tv.share, g.tv.parallax))
        return out

    monkeypatch.setattr(sp, "judged", logged)
    poses = [base_pose(45.0 * k, offset=0.1) for k in range(8)]
    fr, files = ma_frames(_client, poses, trust=None, depth=False, height_above_floor_m=0.5)
    for k in (3, 6):  # rejected frames never enter the check
        files[f"w{k}"] = plain(320, 240)
        fr.append({"name": f"f{k}x", "image": f"w{k}", "camera_id": "cam"})
    job = _client.update("m", fr, files, cameras={"cam": CAM})
    assert job["state"] == "committed" and job["quality"]["failed_checks"] == [], job["quality"]
    assert len(got) == 8 and {c.a for c in got} | {c.b for c in got} == {f"f{k}" for k in range(8)}
    # the refined rotations vary a little with the RANSAC samples drawn (OpenCV's global random
    # state, which earlier tests advance): within 2.5 deg of the truth here, far inside the 10 deg
    # pair limit, against 3.9 and 6.2 deg before the refinement (2.3 deg on the geometric engine
    # since the floor check gives each monocular keyframe its own depth scale, oms/2.4)
    deg = sorted(c.deg for c in got)
    assert deg[-1] < 2.5 and float(np.median(deg)) < 0.8, got
