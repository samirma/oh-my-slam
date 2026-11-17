"""oms/2.18: one prior scale shared by metric and drifting priors, the re-placement of the
keyframes that rest on their priors, residuals against the prior as placed and as sent, the
references' sigma composed into a registered pose's, and odometry epochs (map.md §1 "Prior
scale", "Odometry epochs", §2 "Reference sigma"; ``mapping.anchored``)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oh_my_slam.mapping import anchored as an
from oh_my_slam.mapping import quality
from oh_my_slam.mapping.anchored import Drift, PriorScale
from tests.synth.room import base_pose
from tests.unit.test_anchored import K, err, frame, new_map, prior

ORIGIN = np.array([-1.5, 0.0])
OVER = 1.10  # the odometry over-counts its travel by 10 %


def odo(T: np.ndarray, over: float = OVER, o: np.ndarray = ORIGIN) -> np.ndarray:
    P = T.copy()
    P[:2, 3] = o + (T[:2, 3] - o) * over
    return P


def walk(n: int, step: float = 0.2) -> list[np.ndarray]:
    return [base_pose(5 * (k % 3), -1.5 + step * k, 0.05 * (k % 2)) for k in range(n)]


# -- the estimate and its decision ---------------------------------------------------------------


def test_the_estimate_needs_agreeing_baselines_and_a_span() -> None:
    ev = [[0.22, 0.20]] * 5
    assert an.prior_scale_estimate(ev, 0.4) is None  # the priors span too little
    assert an.prior_scale_estimate(ev[:4], 1.0) is None  # too few
    assert an.prior_scale_estimate([[0.05, 0.04]] * 9, 1.0) is None  # all too short
    s, sig = an.prior_scale_estimate(ev + [[0.22, 0.4]], 1.0)  # type: ignore[misc]
    assert s == pytest.approx(0.2 / 0.22) and sig == pytest.approx(0.02 / math.sqrt(5))
    # baselines of fewer than DRIFT_AGREE_MIN frames: no estimate, never from one sample
    one = [[0.22, 0.20, 1.0, 2.0]] * 4 + [[0.22, 0.20, 1.5, 2.0]] * 3
    assert an.prior_scale_estimate(one, 1.0) is None
    assert an.prior_scale_estimate(one + [[0.22, 0.20, 3.0, 2.0]], 1.0) is not None


def test_the_decision_holds_small_changes_and_needs_significance() -> None:
    p = PriorScale()
    assert p.decide(None) == p and p.in_force == 1.0
    small = p.decide((0.97, 0.01))  # within 2 x hypot(sigma, the depth images' 3 %)
    assert not small.applied and small.value == 0.97 and small.in_force == 1.0
    on = p.decide((0.91, 0.01))
    assert on.applied and on.in_force == 0.91
    assert on.decide(None) == on
    assert on.decide((0.915, 0.01)).in_force == 0.91  # within 2 sigma: held
    assert on.decide((0.88, 0.01)).in_force == 0.88
    off = on.decide((1.0, 0.01))  # no longer significant: back to 1
    assert not off.applied and off.in_force == 1.0
    assert PriorScale.from_json(None) == PriorScale()
    assert PriorScale.from_json(on.to_json()) == on
    assert quality.prior_scale(None) == {"value": 1.0, "sigma": None, "applied": False,
                                         "evidence_count": 0}
    assert quality.prior_scale(on)["applied"] is True
    assert quality.replaced([{"name": "a", "keyframe": "k1", "displacement_m": 0.1}]) == [
        {"name": "a", "displacement_m": 0.1}]


def test_drift_confirm_with_a_fixed_scale() -> None:
    d = Drift()
    d.confirm(base_pose(0, 0.0, 0.0), base_pose(0, 0.0, 0.0), 0.9)
    d.confirm(base_pose(0, 1.0, 0.0), base_pose(0, 1.1, 0.0), 0.9)
    assert d.odo_scale == 0.9 and len(d.evidence) == 1
    assert Drift.from_json(d.to_json()).epoch == 0
    assert an.prior_epoch(None) == 0 and an.prior_epoch({"odometry_epoch": 3}) == 3


# -- metric and drifting priors from one odometry ---------------------------------------------


def _metric(T: np.ndarray, sp: float = 0.05) -> dict[str, Any]:
    return prior(odo(T), sp=sp, sr=1.0)


def test_a_scale_from_metric_priors_places_drifting_ones_too(tmp_path: Path) -> None:
    """Metric priors teach the scale; the drifting priors of the same odometry are placed with
    it (the drift estimate's odometry scale is the prior scale), and every status reports its
    residual against the prior as placed and as sent."""
    m = new_map(tmp_path)
    truth = walk(14)
    m.update([frame("w00", truth[0], p=_metric(truth[0]))])
    m.update([frame(f"w{k:02d}", truth[k], p=_metric(truth[k])) for k in range(1, 7)])
    assert m.pscale.applied and m.pscale.value == pytest.approx(1 / OVER, rel=0.02)
    assert m.drift.ref_T is None  # no drifting frame yet: identity, placed as metric
    blank = np.full((K.height, K.width, 3), 120, np.uint8)

    def drifting(k: int) -> dict[str, Any]:
        return prior(odo(truth[k]), "drifting", 0.05 + 0.05 * k)

    pls = m.update([frame(f"w{k:02d}", truth[k], p=drifting(k)) for k in range(7, 11)]
                   + [frame("kept", truth[11], p=drifting(11), rgb=blank, depth=False,
                            mono=1.0)])
    reg, kept = pls[:4], pls[4]
    assert all(p.status == "registered" for p in reg)
    assert m.drift.odo_scale == m.pscale.value  # the drift estimate's scale is the prior's
    assert all(err(p.T, truth[k])[0] < 0.05 for k, p in zip(range(7, 11), reg, strict=True))
    for k, p in zip(range(7, 11), reg, strict=True):
        assert p.prior_residual is not None and p.prior_residual[0] < 0.06
        assert err(p.T, odo(truth[k]))[0] > 0.1  # against the prior as sent: the over-count
    assert kept.status == "kept_at_prior" and kept.prior_residual == (0.0, 0.0)
    assert err(kept.T, truth[11])[0] < 0.08 and err(kept.T, odo(truth[11]))[0] > 0.15
    assert m.drift_moves([drifting(11)])[1] == m.drift.odo_scale


def test_no_scale_from_one_sample(tmp_path: Path) -> None:
    m = new_map(tmp_path)
    truth = walk(3)
    m.update([frame("w0", truth[0], p=_metric(truth[0]))])
    m.update([frame(f"w{k}", truth[k], p=_metric(truth[k])) for k in range(1, 3)])
    assert 0 < len(m.bias) < an.PRIOR_SCALE_MIN and not m.pscale.applied
    assert m.pscale.sigma is None and m.replaced == []


def test_re_placement_carries_the_references_corrections(tmp_path: Path) -> None:
    """Keyframes placed before the scale was known move by their prior's share of its change
    plus their image's share of their reference's correction; a stored keyframe that gave
    content refuses the cloud."""
    m = new_map(tmp_path)
    truth = walk(12)
    m.update([frame("w00", truth[0], p=_metric(truth[0], 0.03))])
    m.update([frame(f"w{k:02d}", truth[k], p=_metric(truth[k], 0.03)) for k in range(1, 3)])
    assert not m.pscale.applied
    before = {k.name: k.centre.copy() for k in m.keyframes}
    cloud_before = len(m.cloud.keys)
    pls = m.update([frame(f"w{k:02d}", truth[k], p=_metric(truth[k], 0.03))
                    for k in range(3, 9)])
    assert m.pscale.applied and m.replaced
    names = {e["name"] for e in m.replaced}
    assert {"w02", "w08"} <= names
    for k in m.keyframes:
        if k.name in before and k.name in names:
            assert err(k.T, truth[int(k.name[1:])])[0] < float(
                np.linalg.norm(before[k.name] - truth[int(k.name[1:])][:3, 3]))
    assert all(err(p.T, truth[k])[0] < 0.05 for k, p in zip(range(3, 9), pls, strict=True))
    assert len(m.cloud.keys) > 0 and cloud_before > 0
    assert all(k.placed is None or k.placed["scale"] in (m.pscale.value, m.drift.odo_scale)
               for k in m.keyframes)


def test_replace_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A reference cycle ends; a keyframe without a record or prior never moves; the refusion
    skips keyframes without an image, grid or content."""
    m = new_map(tmp_path)
    truth = walk(3)
    m.update([frame(f"w{k}", truth[k], p=_metric(truth[k])) for k in range(3)])
    m.save()
    a, b, c = m.keyframes
    a.placed = {"lever": [1.0, 0.0, 0.0], "scale": 1.0, "share": 0.5, "ref": b.id,
                "drift": False}
    b.placed = {"lever": [1.0, 0.0, 0.0], "scale": 1.0, "share": 0.5, "ref": a.id,
                "drift": True}
    c.placed = None
    m.pscale = PriorScale(0.9, 0.01, True)
    m.drift.odo_scale = 0.8
    real = m.image_path
    monkeypatch.setattr(m, "image_path", lambda kid: None if kid == b.id else real(kid))
    pls = [an.Placement("r", "rejected"), an.Placement(c.name, "registered", keyframe=c.id)]
    # b defines the drift estimate's reference: it moves with b
    b.prior = {"trust": "drifting", "T": b.T.tolist(), "sigma_pos_m": 0.1, "sigma_rot_deg": 2.0}
    m.drift.ref_T, m.drift.ref_prior = b.T.copy(), b.T.copy()
    ref_before = m.drift.ref_T.copy()
    got = m._replace(pls, {c.id})
    assert {e["name"] for e in got} == {a.name, b.name}
    assert c.placed is None and not np.allclose(m.drift.ref_T, ref_before)


def test_lever_and_scaled_edges(tmp_path: Path) -> None:
    m = new_map(tmp_path)
    P = base_pose(0, 1.0, 0.5)
    assert np.allclose(m.scaled(P, 0), P)  # no origin
    assert np.allclose(m.lever({"trust": "metric", "T": P.tolist()})[0], 0.0)
    m.origin_c = np.zeros(3)
    m.pscale = PriorScale(0.9, 0.01, True)
    assert np.allclose(m.scaled(P, 1), P)  # another epoch than the origin's: as sent
    assert m.scaled(P, 0)[0, 3] == pytest.approx(0.9 * P[0, 3])
    moved, odo_scale = m.drift_moves([prior(P, "drifting")])  # the identity: scaled
    assert moved[0][0] == pytest.approx(0.1 * math.hypot(P[0, 3], P[1, 3])) and odo_scale == 0.9
    m.drift = Drift(base_pose(0), base_pose(0), 0.9, epoch=0)
    lv, sc = m.lever({"trust": "drifting", "T": P.tolist(), "odometry_epoch": 2})
    assert np.allclose(lv, 0.0) and sc == 1.0
    lv, sc = m.lever({"trust": "drifting", "T": P.tolist()})
    assert sc == 0.9 and np.linalg.norm(lv) > 0
    f = frame("x", P, p=prior(P, "drifting") | {"odometry_epoch": 2})
    got = m._predict(f)
    assert got is not None and np.allclose(got[0], P)  # another epoch: as sent
    assert m.fixed_odo() == 0.9


# -- the references' sigma ---------------------------------------------------------------------


def test_a_pose_on_a_loose_reference_reports_its_looseness(tmp_path: Path) -> None:
    m = new_map(tmp_path)
    truth = walk(3)
    m.update([frame("w0", truth[0], p=prior(truth[0], "drifting", 0.02))])
    m.keyframes[0].sigma_pos_m, m.keyframes[0].sigma_rot_deg = 0.42, 3.0
    p = m.update([frame("w1", truth[1], p=prior(truth[1], "drifting", 0.5, 5.0))])[0]
    assert p.status == "registered" and p.own_sigmas is not None
    assert p.own_sigmas[0] < 0.1 and p.sigma_pos_m is not None and p.sigma_pos_m > 0.3
    assert m.keyframes[-1].sigma_pos_m == p.sigma_pos_m
    assert p.own_sigma_pos() == p.own_sigmas[0]


def test_a_metric_prior_bounds_the_composed_sigma(tmp_path: Path) -> None:
    m = new_map(tmp_path)
    truth = walk(3)
    m.update([frame("w0", truth[0], p=prior(truth[0], sp=0.03))])
    m.keyframes[0].sigma_pos_m = 0.42
    p = m.update([frame("w1", truth[1], p=prior(truth[1], sp=0.03))])[0]
    assert p.status == "registered" and p.sigma_pos_m is not None
    # the prior holds most of the pose: the reference's error enters by the image's share
    a = an.prior_share(0.03, an.METRIC_IMAGE_SIGMA_POS_M)
    assert p.prior_share[0] == pytest.approx(a, rel=0.3)
    assert p.sigma_pos_m < 0.42 * (1 - a) + 0.05


# -- odometry epochs ---------------------------------------------------------------------------


def test_a_new_epoch_restarts_the_drift_estimate(tmp_path: Path) -> None:
    m = new_map(tmp_path)
    truth = walk(8)
    m.update([frame("w0", truth[0], p=prior(truth[0], "drifting", 0.03))])
    m.update([frame(f"w{k}", truth[k], p=prior(truth[k], "drifting", 0.03))
              for k in range(1, 4)])
    assert m.drift.ref_T is not None and m.reanchored is None
    jump = base_pose(0, 0.4, 0.3)  # the client re-anchored: its odometry jumped

    def re(k: int) -> dict[str, Any]:
        P = truth[k].copy()
        P[:3, 3] += jump[:3, 3]
        return prior(P, "drifting", 0.6) | {"odometry_epoch": 1}

    pls = m.update([frame(f"w{k}", truth[k], p=re(k)) for k in range(4, 8)])
    assert m.reanchored == "signalled" and m.drift.epoch == 1
    assert all(p.status == "registered" for p in pls)
    assert all(err(p.T, truth[k])[0] < 0.05 for k, p in zip(range(4, 8), pls, strict=True))
    assert m.keyframes[-1].prior is not None and m.keyframes[-1].prior["odometry_epoch"] == 1
    m.update([frame("w0b", truth[0], p=prior(truth[0], "drifting", 0.03))])
    assert m.drift.epoch == 1  # an older epoch's frame does not restart it
    # a mixed-epoch job: the stale epoch's frame is placed as sent and gives no drift evidence
    support, ref_prior = list(m.drift.support), m.drift.ref_prior
    stale = prior(base_pose(25, -1.1, 0.1), "drifting", 0.03)
    m.update([frame("w2b", base_pose(25, -1.1, 0.1), p=stale),
              frame("w7b", base_pose(-10, -0.15, 0.0), p=re(7))])
    assert np.asarray(stale["T_map_cam"])[:3, 3].tolist() not in m.drift.support
    assert ref_prior is not None and m.drift.ref_prior is not None
    assert all(c in m.drift.support for c in support)
    assert an.prior_epoch({"odometry_epoch": 1}) == m.drift.epoch


def test_a_new_map_starts_in_its_first_epoch(tmp_path: Path) -> None:
    m = new_map(tmp_path)
    truth = walk(3)
    m.update([frame(f"w{k}", truth[k], p=prior(truth[k], "drifting", 0.03)
                    | {"odometry_epoch": 2}) for k in range(3)])
    assert m.reanchored is None and m.drift.epoch == 2


def test_a_detected_restart_takes_the_prior_scale_or_one(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m = new_map(tmp_path)
    monkeypatch.setattr(m, "_drift_evidence", lambda frames, out: ["obs"])
    monkeypatch.setattr(m, "_reanchor", lambda obs, able, used: True)
    stale = Drift(base_pose(0), base_pose(0), 1.3, epoch=2)
    got, out = m._fit_drift([], {}, stale)
    assert got is not None and got.odo_scale == 1.0 and got.epoch == 2 and out == {}
    assert m.reanchored == "detected"
    m.pscale = PriorScale(0.9, 0.01, True)
    got, _ = m._fit_drift([], {}, stale)
    assert got is not None and got.odo_scale == 0.9


def test_a_scale_no_longer_applied_returns_the_drift_to_its_evidence(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The prior scale stops being applied: the drift estimate's odometry scale returns to its
    own evidence's fit, and a keyframe placed through it re-places back."""
    m = new_map(tmp_path)
    truth = walk(2)
    m.update([frame(f"w{k}", truth[k], p=_metric(truth[k])) for k in range(2)])
    k = m.keyframes[1]
    m.pscale = PriorScale(0.9, 0.01, True)
    m.drift = Drift(base_pose(0), base_pose(0), 0.9, evidence=[[1.0, 1.1]])
    k.placed = {"lever": [1.0, 0.0, 0.0], "scale": 0.9, "share": 1.0, "ref": None,
                "drift": True}
    before = k.centre.copy()
    monkeypatch.setattr(an, "prior_scale_estimate", lambda ev, span: (1.0, 0.01))
    assert m.estimate_prior_scale() and not m.pscale.applied
    assert m.drift.odo_scale == pytest.approx(1.1)
    m._replace([], set())
    assert k.centre[0] - before[0] == pytest.approx(0.2)
    assert Drift().evidence_scale() == 1.0


def test_a_correction_below_the_minimum_propagates_nothing(tmp_path: Path) -> None:
    m = new_map(tmp_path)
    truth = walk(2)
    m.update([frame(f"w{k}", truth[k], p=_metric(truth[k])) for k in range(2)])
    a, b = m.keyframes
    a.placed = {"lever": [0.08, 0.0, 0.0], "scale": 1.0, "share": 1.0, "ref": None,
                "drift": False}
    b.placed = {"lever": [0.15, 0.0, 0.0], "scale": 1.0, "share": 0.5, "ref": a.id,
                "drift": False}
    m.pscale = PriorScale(0.9, 0.01, True)
    # a moves 8 mm (not applied); b's own half is 7.5 mm: with a's 4 mm it would pass 1 cm
    assert m._replace([], {a.id, b.id}) == []


def test_an_unsignalled_re_anchor_is_detected(tmp_path: Path) -> None:
    """A stale correction: the drift estimate moves the priors 0.4 m, the images put the frames
    at their priors as sent; the estimate restarts instead of fitting a change."""
    m = new_map(tmp_path)
    truth = walk(9)
    m.update([frame("w0", truth[0], p=prior(truth[0], "drifting", 0.03))])
    m.update([frame(f"w{k}", truth[k], p=prior(truth[k], "drifting", 0.03))
              for k in range(1, 4)])
    shift = an.pose(np.eye(3), np.array([0.4, 0.0, 0.0]))
    assert m.drift.ref_T is not None
    m.drift.ref_T = shift @ m.drift.ref_T
    pls = m.update([frame(f"w{k}", truth[k], p=prior(truth[k], "drifting", 0.05))
                    for k in range(4, 9)])
    assert m.reanchored == "detected"
    assert all(err(p.T, truth[k])[0] < 0.05 for k, p in zip(range(4, 9), pls, strict=True))
    dc = quality.drift_correction([(0.0, 0.0)], 1.0, None, m.reanchored)
    assert dc is not None and dc["restarted"] == "detected"


def test_the_locate_summary_logs_depth_and_rivals() -> None:
    """oms/2.18 (log only): the locate summary line says whether the frame sent depth, its
    depth check, and how many rivals it had, how many outside the prior's bound."""
    from oh_my_slam.service.nav import LocateFrame, locate_summary

    res: dict[str, Any] = {"status": "located", "reason": None,
                           "T_map_cam": base_pose(0).tolist(), "sigma_pos_m": 0.1,
                           "sigma_rot_deg": 1.0}
    fr = LocateFrame("f", None, None, None, None, None, None)  # type: ignore[arg-type]
    got = locate_summary(fr, res)
    assert got["depth"] is False and "depth_check" not in got
    assert got["rivals"] == 0 and got["rivals_outside_prior"] == 0
    fr = LocateFrame("f", None, None, None, None, None, None,  # type: ignore[arg-type]
                     np.zeros((2, 2), np.uint16))
    res |= {"depth_check": {"pose": "confirmed", "refused": 2},
            "rivals": [{"pos_m": 1.0, "outside_prior": True},
                       {"pos_m": 0.2, "outside_prior": False}, {"pos_m": 0.3}]}
    got = locate_summary(fr, res)
    assert got["depth"] is True and got["depth_check"] == {"pose": "confirmed", "refused": 2}
    assert got["rivals"] == 3 and got["rivals_outside_prior"] == 1


# -- locate places its prior as an update does (api.md §6 "Prior placement") --------------------


def _drifted_map(root: Path) -> Any:
    from oh_my_slam.mapping.anchored import AnchoredMap

    m = AnchoredMap(root)
    m.created_with, m.origin_c = "metric", np.zeros(3)
    # the client's raw odometry, which the map's drift estimate absorbed: 0.5 m along x
    m.drift = Drift(an.pose(np.eye(3), np.array([0.5, 0.0, 0.0])), np.eye(4), 0.9,
                    support=[[0.0, 0.0, 0.0]])
    m.pscale = PriorScale(0.9, 0.01, True)
    m.save()
    return m


def test_a_locate_prior_is_placed_through_the_drift_estimate(tmp_path: Path) -> None:
    from oh_my_slam.mapping.anchored import AnchoredMap
    from oh_my_slam.service import nav
    from oh_my_slam.service.cameras import Camera

    _drifted_map(tmp_path)
    pm = AnchoredMap.priors_only(tmp_path)
    assert pm is not None and pm.pscale.applied
    P = an.pose(np.eye(3), np.array([1.0, 0.0, 0.0]))
    T, sp, sr = pm.place_prior(prior(P, "drifting", 0.1, 2.0))  # type: ignore[misc]
    assert T[0, 3] == pytest.approx(0.5 + 0.9 * 1.0) and sp > 0.1  # drift-grown
    Tm, spm, _ = pm.place_prior(prior(P, "metric", 0.03))  # type: ignore[misc]
    assert Tm[0, 3] == pytest.approx(0.9) and spm == pytest.approx(0.03)
    assert AnchoredMap.priors_only(tmp_path / "none") is None
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / an.STATE_JSON).write_text('{"engine": "mapper"}')
    assert AnchoredMap.priors_only(tmp_path / "other") is None
    cam = Camera(300.0, 300.0, 160.0, 120.0, 320, 240)
    rgb = np.random.default_rng(0).integers(0, 255, (240, 320, 3)).astype(np.uint8)
    fr = nav.LocateFrame("q", cam, rgb, None, prior(P, "drifting", 0.1, 2.0), None, None)
    q, _ = nav._query(fr, 0.0, False, pm)
    assert q.prior is not None and q.prior[0, 3] == pytest.approx(1.4)
    assert q.prior_sigma_pos_m == pytest.approx(sp)
    fr.prior = prior(P, "metric", 0.03)
    q, _ = nav._query(fr, 0.0, False, pm)
    assert q.prior is not None and q.prior[0, 3] == pytest.approx(0.9)
    assert q.prior_sigma_pos_m == pytest.approx(0.03)
    assert nav.prior_residual(fr, P.tolist(), q.prior) == (pytest.approx(0.1), 0.0)


def test_the_placer_is_read_once_per_version(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from oh_my_slam.service import nav
    from oh_my_slam.service.errors import ApiError

    _drifted_map(tmp_path / "mapper")
    calls: list[int] = []

    def mapper_for(name: str, n: int) -> Path | None:
        calls.append(n)
        if n == 2:
            raise ApiError("version_gone", "gone", {})
        return None if n == 3 else tmp_path / "mapper"

    store = SimpleNamespace(dir=lambda name: tmp_path, mapper_for=mapper_for)
    derived = SimpleNamespace(store=store)
    nav._PLACERS.clear()
    got = nav.placer(derived, SimpleNamespace(name="m", version=1))  # type: ignore[arg-type]
    assert got is not None and got.pscale.applied
    assert nav.placer(derived, SimpleNamespace(name="m", version=1)) is got  # type: ignore[arg-type]
    assert nav.placer(derived, SimpleNamespace(name="m", version=2)) is None  # type: ignore[arg-type]
    assert nav.placer(derived, SimpleNamespace(name="m", version=3)) is None  # type: ignore[arg-type]
    assert calls == [1, 2, 3]
    for v in range(4, 4 + nav.PLACERS_KEPT):
        nav.placer(derived, SimpleNamespace(name="m", version=v))  # type: ignore[arg-type]
    assert len(nav._PLACERS) == nav.PLACERS_KEPT
