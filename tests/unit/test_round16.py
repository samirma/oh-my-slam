"""Round 16 (P29, P29a, P29b, oms/2.5): drifting priors fused by calibrated sigmas, the monocular noise
floor, the sent height measured from z 0 when priors fixed the frame, the floor across versions,
the drift estimate weighed against the priors as sent, a monocular frame beyond the scale bound
not judged by the floor check (it keeps k and its pose, its content is excluded: P29b),
``quality.drift_correction``, a real plane extent for the floor,
the floor-check scale for content only, and the service log's job and locate lines."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oh_my_slam.core.log import ContextFilter, log_context
from oh_my_slam.core.types import Intrinsics
from oh_my_slam.mapping import anchor, sigmas
from oh_my_slam.mapping import anchored as an
from oh_my_slam.mapping.anchored import AnchoredMap, FrameIn
from oh_my_slam.mapping.quality import drift_correction
from tests.synth.room import base_pose, look
from tests.unit.test_round14 import H, K, _floor_points, frame, old, prior, shot  # noqa: F401
from tests.unit.test_service_api import Client
from tests.unit.test_service_nav import decode
from tests.unit.test_service_prior_first import (
    CAM,
    client,  # noqa: F401
    dry,
    frames_for,
)
from tests.unit.test_service_prior_first import shot as service_shot

# -- P29 S1: the monocular noise floor --------------------------------------------------------------


def test_the_monocular_noise_floor_is_0_15_m_and_tunable(monkeypatch: pytest.MonkeyPatch) -> None:
    """api.md §6 (oms/2.5): without depth 0.15 m whatever the prior, rotation unchanged."""
    for trust in ("metric", "drifting", None):
        assert sigmas.floor_of(False, trust) == (0.15, 2.1)
    assert sigmas.floor_of(True, "metric") == (0.008, 1.25)
    assert sigmas.floor_of(True, None) == (0.012, 1.25)
    monkeypatch.setenv(sigmas.SIGMA_FLOORS_ENV, json.dumps({"nodepth/none": [0.2, 3.0]}))
    assert sigmas.floor_of(False, None) == (0.2, 3.0)
    assert sigmas.floor_of(False, "drifting") == (0.15, 2.1)


# -- F1: one fusion of image and prior, by calibrated sigmas ----------------------------------------


def _fused_case(tmp_path: Path, trust: str) -> tuple[AnchoredMap, Any, np.ndarray, np.ndarray]:
    """A monocular map (its scale from the origin's camera height) whose origin keyframe's
    keypoints are lifted 1.3 x too far (its depth's scale wrong), and a frame 0.4 m behind it with a prior 1 cm from truth: its image-only pose
    lands well off truth."""
    m = AnchoredMap(tmp_path / trust)
    T0 = base_pose(0.0)
    m.update([frame("a0", T0, p=prior(T0, trust, 0.03, 2.0), depth=None, mono=1.0, hs=0.01)])
    assert m.scale.source == "camera_height"  # its scale measured: the image pose is tight
    k = m.keyframes[0]
    assert k.kdepth is not None
    k.kdepth = (k.kdepth * 1.3).astype(np.float32)
    G = base_pose(0.0, x=-0.4)
    P = G.copy()
    P[:3, 3] += np.array([0.0, 0.01, 0.0])
    pl = m.update([frame("b1", G, p=prior(P, trust, 0.03, 2.0), depth=None, mono=1.0,
                         height=None)])[0]
    return m, pl, G, P


@pytest.mark.parametrize("trust", ["drifting", "metric"])
def test_a_prior_is_weighed_by_calibrated_isotropic_sigmas(tmp_path: Path, trust: str) -> None:
    """Diagnosis F1 (map.md §1): the committed centre is the 1/σ²-weighted mean of the image-only
    centre (its calibrated sigma, isotropic on monocular references) and the prior as placed
    (its sigma, drift-grown for a drifting one), by one path for both trusts; never the image
    pose refined against the raw match information."""
    m, pl, G, P = _fused_case(tmp_path, trust)
    assert pl.status == "registered" and pl.align_T is not None and pl.image_sigmas is not None
    img = pl.align_T[:3, 3]
    assert np.linalg.norm(img - G[:3, 3]) >= 0.1  # the image-only pose is well off
    assert pl.prior_ref is not None
    T0 = pl.prior_ref
    sp0 = 0.03 if trust == "metric" else m.placed_drift.predict(P, 0.03, 2.0)[1]  # type: ignore[union-attr]
    rp = pl.image_sigmas[0]
    assert rp >= 0.15  # the monocular floor
    w = sp0 ** 2 / (sp0 ** 2 + rp ** 2)  # the image's share
    want = T0[:3, 3] + w * (img - T0[:3, 3])
    assert np.linalg.norm(pl.T[:3, 3] - want) < 2e-3
    assert np.linalg.norm(pl.T[:3, 3] - T0[:3, 3]) <= w * np.linalg.norm(img - T0[:3, 3]) + 0.01
    assert pl.prior_residual is not None and pl.prior_residual[0] <= 0.05
    # the reported sigma: the combination of the two real sigmas through the weights
    assert pl.sigma_pos_m == pytest.approx(math.hypot(w * rp, (1 - w) * sp0), rel=0.02)


# -- F3 / P29 S2: the height measured from z 0 when priors fixed the frame --------------------------


def _sensor_map(tmp_path: Path, name: str = "m") -> AnchoredMap:
    m = AnchoredMap(tmp_path / name)
    m.update([frame("a0", base_pose(0)), frame("a1", base_pose(30.0))])
    return m


def _lifted(m: AnchoredMap, dz: float) -> None:
    """The map's cloud moved up by ``dz`` (a floor estimate biased by its depth)."""
    m.cloud = m.cloud.rotated(np.eye(3), np.array([0.0, 0.0, dz]))


def test_a_biased_cloud_floor_does_not_lift_the_sent_height(tmp_path: Path) -> None:
    """Diagnosis F3 (map.md §2, §6, oms/2.5): on a map whose frame priors fixed, a cloud floor
    0.05 m up lies within the floor band of z 0: the floor is z 0 and the height's target the
    sent height above it, weighted by the height's sigma alone (the floor's is 0)."""
    m = _sensor_map(tmp_path)
    _lifted(m, 0.05)
    assert m._floor_estimate()[0] == 0.0 and m.floor()[0] == 0.0
    m.begin_job()
    assert (m.floor_z, m.floor_z_sigma, m.floor_z_rel) == (0.0, 0.0, 0.0)
    G = base_pose(60.0)
    f = frame("b1", G, depth=None, mono=1.0, hs=0.01)
    assert m._height_z(f) == (pytest.approx(H), pytest.approx(0.01))
    pl = m.update([f])[0]
    m2 = _sensor_map(tmp_path, "m2")
    _lifted(m2, 0.05)
    bare = m2.update([frame("b1", G, depth=None, mono=1.0, height=None)])[0]
    assert pl.status == bare.status == "registered" and pl.height_term
    assert abs(pl.T[2, 3] - G[2, 3]) < 0.01  # within 1 cm of the prior's z
    # the height term moves the frame along z only
    assert np.linalg.norm(pl.T[:2, 3] - bare.T[:2, 3]) < 0.005


def test_a_real_floor_beyond_the_band_is_still_used(tmp_path: Path) -> None:
    """An arm on a worktop (map.md §6): the observed plane 0.15 m from z 0 is the floor; the
    height's target is that floor plus the height, weighted by hypot(height sigma, its
    floor_sigma_m), and on monocular depth also scale_sigma_rel × the height."""
    m = _sensor_map(tmp_path)
    _lifted(m, 0.15)
    z, sig = m.floor()
    assert z == pytest.approx(0.15, abs=0.01)
    m.begin_job()
    assert m.floor_z == z and m.floor_z_sigma == sig and m.floor_z_rel == 0.0
    f = frame("b1", base_pose(60.0), hs=0.01)
    hz = m._height_z(f)
    assert hz is not None and hz[0] == pytest.approx(z + H)
    assert hz[1] == pytest.approx(math.hypot(0.01, sig))
    for k in m.keyframes:  # the same plane, measured on monocular depth
        k.depth_kind = "mono"
    assert m._mono_cloud()
    got = m.map_floor()
    assert got is not None and got[2] == m.scale_summary()[1] > 0
    m.floor_z_rel = got[2]
    hz = m._height_z(f)
    assert hz is not None and hz[1] == pytest.approx(math.hypot(0.01, sig, got[2] * H))
    # the z 0 convention never takes the scale term; a camera-only map's first update: z 0
    assert AnchoredMap(tmp_path / "x").map_floor() is None
    c = AnchoredMap(tmp_path / "c")
    c.created_with, c.origin_height_m = "none", H
    assert c.map_floor() == (0.0, 0.0, 0.0)


# -- P29 S3: the floor across versions --------------------------------------------------------------


def test_a_version_keeps_the_previous_floor_within_the_band(tmp_path: Path) -> None:
    """map.md §6 "Floor across versions" (oms/2.5), on every map: the previous ``floor_z_m``
    stays while the new estimate lies within max(0.06 m, 3 × its floor_sigma_m) of it, and it
    is written with the version (``state.json``)."""
    m = _sensor_map(tmp_path)
    _lifted(m, 0.10)
    est = m._floor_estimate()[0]
    assert est == pytest.approx(0.10, abs=0.01)
    m.floor_prev = 0.08  # within the band: kept
    assert m.floor()[0] == 0.08
    m.floor_prev = 0.0  # beyond it: the new estimate
    assert m.floor()[0] == est
    m.floor_prev = est  # the same: as it is
    assert m.floor()[0] == est
    m.floor_prev = 0.08
    m.floor()
    m.save()
    again = AnchoredMap.load(m.root)
    assert again.floor_prev == 0.08 and again.floor()[0] == 0.08
    # a save without an estimate made estimates it
    m2 = _sensor_map(tmp_path, "m2")
    m2._last_floor = None
    m2.save()
    assert AnchoredMap.load(m2.root).floor_prev == m2._floor_estimate()[0]
    # a camera-only map with a camera height too (its z 0 refined by the plane)
    c = AnchoredMap(tmp_path / "c")
    fs = [frame(n, base_pose(y), hs=0.01) for n, y in (("a0", 0.0), ("a1", 30.0))]
    for f in fs:
        f.prior = None
    c.update(fs)
    assert c.created_with == "none" and c.origin_height_m == H
    zc = c._floor_estimate()[0]
    assert zc is not None
    c.floor_prev = zc + 0.03
    assert c.floor()[0] == zc + 0.03


# -- F4 / P29 S4: the drift estimate weighed against the priors as sent ----------------------------


def test_the_refined_drift_is_weighed_against_the_priors_as_sent() -> None:
    """map.md §1 "Drift estimate" (oms/2.5): image poses 0.2 m off priors of 0.03 m, at a 0.15 m
    image sigma, move the estimate only by their share; a rotation-only pose's centre does not
    count."""
    P = [base_pose(30.0 * k) for k in range(3)]
    off = np.array([0.2, 0.0, 0.0])
    obs = []
    for k, Pk in enumerate(P):
        T = Pk.copy()
        T[:3, 3] += off
        obs.append(an._DriftObs(k, f"f{k}", T, Pk, (0.03, 2.0), (0.15, 2.0), k == 2,
                                an.Placement(f"f{k}", "registered")))
    fit = an._refined_drift(an.Drift(P[0], P[0]), obs)
    moved = fit.predict(P[1], 0.0, 0.0)[0][:3, 3] - P[1][:3, 3]
    # two image centres at 1/0.15², three priors at 1/0.03²
    w = 2 / 0.15 ** 2 / (2 / 0.15 ** 2 + 3 / 0.03 ** 2)
    assert moved == pytest.approx(w * off, abs=1e-6)
    # loose priors: the images win
    loose = [an._DriftObs(o.index, o.name, o.T, o.P, (5.0, 2.0), (0.01, 2.0), False, o.pl)
             for o in obs]
    fit = an._refined_drift(an.Drift(P[0], P[0]), loose)
    assert np.allclose(fit.predict(P[1], 0.0, 0.0)[0][:3, 3] - P[1][:3, 3], off, atol=1e-3)


def _ratchet_frames(rng: np.random.Generator) -> list[tuple[str, np.ndarray, float]]:
    out = []
    x, yaw = -0.6, 0.0
    for k in range(24):
        yaw = 12.0 * k if k < 12 else 144.0 - 10.0 * (k - 12)
        x += 0.03
        out.append((f"f{k:02d}", base_pose(yaw, x=x, height=H), math.exp(rng.normal(0, 0.15))))
    return out


def test_the_drift_ratchet_stops_over_six_jobs(tmp_path: Path) -> None:
    """Diagnosis F4: six jobs of exact drifting priors (σ 0.03 m / 2°) on a monocular map whose
    frames' depth scale jitters by 15 %: the placed prior stays within 3 × its drift-grown sigma
    of the prior as sent, and no job's displacement passes 0.10 m."""
    rng = np.random.default_rng(7)
    m = AnchoredMap(tmp_path / "m")
    seq = _ratchet_frames(rng)
    for j in range(6):
        fs = []
        for name, T, s in seq[4 * j:4 * j + 4]:
            img, z = shot(T)
            d = (z * 1.3 * s).astype(np.float32)
            fs.append(FrameIn(name, "cam", img, K, prior(T, "drifting", 0.03, 2.0), None, None,
                              H, 0.01, 1000, lambda d=d: (d, K, None, np.array([0.0, -1.0, 0.0]))))
        pls = m.update(fs)
        disp = []
        for (name, T, _), pl in zip(seq[4 * j:4 * j + 4], pls, strict=True):
            Tp, sp, _ = m.placed_drift.predict(T, 0.03, 2.0)  # type: ignore[union-attr]
            assert np.linalg.norm(Tp[:3, 3] - T[:3, 3]) <= 3.0 * sp, (j, name)
            if pl.status == "registered" and pl.prior_residual is not None:
                disp.append(pl.prior_residual[0])
        assert disp and max(disp) <= 0.10, (j, disp)


def test_odometry_evidence_on_a_reference_with_few_matches(tmp_path: Path) -> None:
    """A monocular frame's odometry-scale evidence (oms/2.5): its image baseline on its main
    reference's own depth when that reference holds ``MIN_INLIERS`` of its matches, else its
    pose's own."""
    m = AnchoredMap(tmp_path / "m")
    T = base_pose(0)
    s = an.Solve(T[:3, :3], T[:3, 3] + [0.3, 0.0, 0.0], np.eye(6), np.ones(3, bool), 0.5)
    sub = (np.zeros((3, 2)), np.ones((3, 3)), np.ones((3, 3)), np.array([1, 1, 1], np.intp))
    refs = an.Refs(np.zeros((2, 3)), np.ones(2), np.ones(2))
    ref = (T[:3, 3].copy(), {"trust": "drifting", "T": T.tolist()}, "mono")
    P = T.copy()
    P[0, 3] += 0.2
    got = m._scale_evidence(s, P, "drifting", 0, ref, (sub, refs, K, (0.03, 2.0)), 1.0)
    assert got == [("odometry", [pytest.approx(0.2), pytest.approx(0.3)])]


# -- P26-1: a real plane extent ---------------------------------------------------------------------


def _wall(Hc: float, noise: float, yaw_deg: float = 20.0) -> tuple[np.ndarray, Intrinsics,
                                                                    np.ndarray]:
    """A depth image holding only a vertical wall 0.95 m ahead (turned by ``yaw_deg``), with
    relative noise ``noise``: no floor in view."""
    Kw = Intrinsics(320.0, 320.0, 319.5, 239.5, 640, 480)
    rng = np.random.default_rng(0)
    T = look(0.0, (0.0, 0.0, Hc), 0.0)
    v, u = np.mgrid[0:Kw.height, 0:Kw.width]
    r = np.stack([(u - Kw.cx) / Kw.fx, (v - Kw.cy) / Kw.fy, np.ones_like(u, float)], -1)
    a = math.radians(yaw_deg)
    t = 0.95 / (r.reshape(-1, 3) @ np.array([math.sin(a), 0.0, math.cos(a)]))
    z = (t * r.reshape(-1, 3)[:, 2]).reshape(Kw.height, Kw.width)
    z = (z * (1 + noise * rng.standard_normal(z.shape))).astype(np.float32)
    return z, Kw, T


@pytest.mark.parametrize("Hc", [0.34, 0.5, 0.8])
def test_a_wall_slice_is_no_floor(Hc: float) -> None:
    """Diagnosis item 6 (map.md §2 "Floor check"): a noisy wall-only depth image holds no
    measurable floor plane (a horizontal slice of it spreads along one axis only): the frame is
    not checked, whether its depth is a sensor's or monocular."""
    z, Kw, T = _wall(Hc, 0.004)
    assert an.own_floor(z, Kw, T[:3, :3], Hc) is None
    for kind in ("sensor", "mono"):
        assert an.floor_verdict(kind, Hc, 0.01, T, 0.05, z, Kw) == an.FloorCheck()
    # a real floor still measures within 1 cm, and a tilted plane's own normal is no floor
    zf, Kg, Tf = _floor_points(Hc)
    assert an.own_floor(zf, Kg, Tf[:3, :3], Hc) == pytest.approx(Hc, abs=0.01)
    q = np.random.default_rng(1).uniform(-1.0, 1.0, (4000, 3))
    q[:, 2] = -Hc + 0.03 * q[:, 0]  # 1.7 deg: a floor
    assert an.horizontal_plane(q, Hc) is not None
    q[:, 1] *= 0.01  # a thin ribbon: no extent across
    assert an.horizontal_plane(q, Hc) is None


# -- P29 S5: a monocular frame beyond the scale bound is not judged --------------------------------


def test_a_monocular_frame_beyond_the_bound_is_not_judged() -> None:
    """map.md §2 (oms/2.5): its own scale beyond the monocular bound, it keeps k (no own scale)
    and is not excluded for it; sensor depth unchanged."""
    z, Kg, T = _floor_points(0.5)
    got = an.floor_verdict("mono", 0.5, 0.01, T, 0.05, z * 2.0, Kg)
    assert got == an.FloorCheck(True, 1.0, None, False)
    assert an.floor_verdict("mono", 0.5, 0.01, T, 0.05, z * 1.2, Kg).judged
    assert not an.floor_verdict("sensor", 0.5, 0.01, T, 0.01, z * 1.2, Kg).ok


def _half_scale_sweep(tmp_path: Path, fused: list[str] | None = None) -> tuple[AnchoredMap, set[str]]:
    """The diagnosis's sweep: a monocular map whose frames s5 and s6 have half the others' depth.
    ``fused`` collects the names of the keyframes whose depth goes into the cloud."""
    m = AnchoredMap(tmp_path / "m")
    if fused is not None:
        fuse = m._fuse

        def spy(image: Any, ignore: Any, kf: Any, depth: Any, K_grid: Any) -> None:
            fused.append(kf.name)
            fuse(image, ignore, kf, depth, K_grid)

        m._fuse = spy  # type: ignore[method-assign]
    m.update([frame("a0", base_pose(0), depth=None, mono=1.3)])
    half = {"s5", "s6"}
    pls = m.update([frame(f"s{k}", base_pose(30.0 * k + 15.0), depth=None,
                          mono=0.65 if f"s{k}" in half else 1.3) for k in range(12)])
    assert all(p.status == "registered" for p in pls)
    return m, half


def test_unjudged_monocular_keyframes_keep_k_and_their_pose_but_no_content(
        tmp_path: Path) -> None:
    """The diagnosis's half-scale frames (item 4): registered with their pose and sigmas, not
    in ``floor_inconsistent``, without a depth scale (k kept), but, since P29b, withheld exactly
    as a frame the check excluded: no cloud points, no objects; the others take their own
    scale."""
    fused: list[str] = []
    m, half = _half_scale_sweep(tmp_path, fused)
    assert m.floor_inconsistent == []
    kf = {k.name: k for k in m.keyframes}
    for n in half:
        k = kf[n]
        assert k.floor_ok and k.floor_unjudged and k.floor_withheld()
        assert k.state == "fused" and k.usable()  # a registered frame, a reference
        assert k.sigma_pos_m > 0 and k.T[2, 3] == pytest.approx(H, abs=0.05)  # its pose kept
        assert not k.contributes() and not k.detects()  # no cloud, no objects
        assert k.depth_scale is None and k.floor_factor is None
        assert n not in fused
    rescaled = [k for k in m.keyframes if k.depth_scale is not None]
    assert len(rescaled) >= 6 and not any(k.floor_unjudged for k in rescaled)
    assert all(k.contributes() and k.detects() and k.name in fused for k in rescaled)
    assert not any(k.floor_withheld() for k in rescaled)
    # stored and loaded again
    d = an.Keyframe.from_json(json.loads(json.dumps(kf["s5"].to_json())))
    assert d.floor_unjudged and d.floor_withheld() and not d.detects()
    # the floor estimate is made from the others' cloud only
    floor, sigma = m.floor()
    assert floor is not None and abs(floor) < 0.03 and sigma <= 0.02


def test_an_unjudged_frame_is_no_scale_evidence_and_no_cloud(tmp_path: Path) -> None:
    """P29b: a frame whose own scale lies beyond the monocular bound, registered last, leaves the
    map's scale, its evidence and its cloud as they are without it (what it recorded is taken
    back: ``AnchoredMap.refine_scale``)."""
    def build(name: str, odd: bool) -> AnchoredMap:
        m = AnchoredMap(tmp_path / name)
        m.update([frame("a0", base_pose(0), depth=None, mono=1.3)])
        fs = [frame(f"s{k}", base_pose(30.0 * k + 15.0), depth=None, mono=1.3)
              for k in range(5)]
        if odd:
            fs.append(frame("odd", base_pose(165.0), depth=None, mono=0.65))
        m.update(fs)
        return m

    a, b = build("a", True), build("b", False)
    assert [k.name for k in a.keyframes if k.floor_unjudged] == ["odd"]
    assert a.scale.k == b.scale.k and a.scale.source == b.scale.source
    assert a.scale.heights == b.scale.heights and len(a.scale.heights) == 6
    assert a.scale.height_yaws == b.scale.height_yaws
    assert a.scale.depth_residuals == b.scale.depth_residuals
    assert a.scale.sigma_rel == b.scale.sigma_rel
    assert np.array_equal(a.cloud.keys, b.cloud.keys) and len(a.cloud.keys) > 0
    # the same frame judged (its own scale within the bound) is evidence and content
    c = AnchoredMap(tmp_path / "c")
    c.update([frame("a0", base_pose(0), depth=None, mono=1.3)])
    c.update([frame("odd", base_pose(15.0), depth=None, mono=1.3)])
    assert len(c.scale.heights) == 2 and c.keyframes[-1].contributes()


def test_unjudged_frames_give_no_objects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """P29b: ``fuse_objects`` takes no view of a frame the check did not judge, nor of one it
    excluded (a keyframe's ``detects``)."""
    from concurrent.futures import Future

    from oh_my_slam.mapping.anchored_objects import ObjectMap, fuse_objects

    m, half = _half_scale_sweep(tmp_path)
    pls = [an.Placement(k.name, "registered", None, k.T, k.sigma_pos_m, k.sigma_rot_deg, 100,
                        keyframe=k.id) for k in m.keyframes]
    seen: list[list[str]] = []
    update = ObjectMap.update

    def spy(self: ObjectMap, views: Any, *a: Any, **k: Any) -> Any:
        seen.append([v.kid for v in views])
        return update(self, views, *a, **k)

    monkeypatch.setattr(ObjectMap, "update", spy)
    futures = []
    for _ in pls:
        fut: Future[list[Any]] = Future()
        fut.set_result([])
        futures.append(fut)
    frames = [(pl, None, Path("kf.jpg"), fut) for pl, fut in zip(pls, futures, strict=True)]
    fuse_objects(m, frames, 2, lambda path, regions, dets, size: dets)
    unjudged = {k.id for k in m.keyframes if k.name in half}
    assert seen and unjudged.isdisjoint(seen[0]) and len(seen[0]) == len(m.keyframes) - 2


# -- P26-2: the floor-check scale is for content only -----------------------------------------------


def test_the_floor_factor_applies_to_content_only(tmp_path: Path) -> None:
    """map.md §2 "Floor check": a monocular keyframe's own scale is its cloud's, grid evidence's
    and objects'; registration lifting, depth confirmation, ``depth_of`` and scale evidence keep
    k."""
    m = AnchoredMap(tmp_path / "m")
    m.update([frame("a0", base_pose(0), depth=None, mono=1.3)])
    m.update([frame("b1", base_pose(30.0), depth=None, mono=1.3)])
    k = m.keyframes[1]
    assert k.floor_factor is not None and k.depth_scale == pytest.approx(1 / 1.3, rel=0.05)
    k.floor_factor = 0.8  # an own scale 20 % below the map's
    assert m._depth_factor(k) == pytest.approx(m.scale.k / k.k_used)
    assert m._depth_factor(k, content=True) == k.floor_factor
    m.save()
    reg, content = m.depth_of(k.id), m.content_depth(k.id)
    assert reg is not None and content is not None
    ok = reg > 0
    assert np.median(content[ok] / reg[ok]) == pytest.approx(k.floor_factor / m._depth_factor(k),
                                                             rel=1e-2)
    grid = m.content_grid(k)
    assert grid is not None and k.grid is not None
    assert np.allclose(grid, k.grid * k.floor_factor)
    ref, ref_c = m._grid(k), m._grid(k, content=True)
    assert ref is not None and ref_c is not None
    assert np.allclose(ref_c.depth, k.grid * k.floor_factor)
    # its stored keypoint depths stay at k (its grid's, not its own scale's): registration lifts
    assert k.feats is not None and k.K_grid is not None and k.kdepth is not None
    at = an.sample_depth(k.grid, k.K_grid, k.K, k.feats.kps)
    both = np.isfinite(at) & np.isfinite(k.kdepth)
    assert both.sum() > 100 and np.allclose(k.kdepth[both], at[both], rtol=1e-3)
    k.floor_factor = None
    assert m.content_grid(k) is k.grid


# -- P29 S6: drift_correction, T_map_cam, prior residuals (oms/2.5) ---------------------------------


def test_drift_correction_values() -> None:
    assert drift_correction([], 1.0) is None
    assert drift_correction([(0.1, 2.0), (0.3, 1.0)], 1.02) == {
        "translation_m": 0.3, "rotation_deg": 2.0, "odo_scale": 1.02}
    m = AnchoredMap(Path("unused"))
    P = base_pose(0)
    moved, odo = m.drift_moves([prior(P, "drifting")])
    assert moved == [(0.0, 0.0)] and odo == 1.0  # the identity
    m.drift = an.Drift(P, P, 1.1)
    m.drift.ref_T = P.copy()
    m.drift.ref_T[:3, 3] += [0.2, 0.0, 0.0]
    moved, odo = m.drift_moves([prior(P, "drifting", st=0.5)])
    assert moved[0][0] == pytest.approx(0.2) and odo == 1.1
    from oh_my_slam.service.update import mapper_drift_correction

    class Work:
        def __init__(self, p: dict[str, Any] | None) -> None:
            self.prior = p
            self.trust = "none" if p is None else p["trust"]

    fr = [type("F", (), {"work": Work(prior(P, "drifting"))})(),
          type("F", (), {"work": Work(None)})()]
    assert mapper_drift_correction(fr, None) == {  # type: ignore[arg-type]
        "translation_m": 0.0, "rotation_deg": 0.0, "odo_scale": 1.0}
    sim = anchor.Sim3(1.25, np.eye(3), np.array([0.1, 0.0, 0.0]))
    got = mapper_drift_correction(fr, sim)  # type: ignore[arg-type]
    assert got is not None and got["odo_scale"] == 0.8 and got["translation_m"] > 0
    assert mapper_drift_correction(fr[1:], sim) is None  # type: ignore[arg-type]


def _drifting(c: Client, poses: list[np.ndarray], tag: str, off: np.ndarray | None = None,
              depth: bool = True) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    fr, files = frames_for(c, poses, tag=tag, depth=depth, trust="drifting")
    for f, T in zip(fr, poses, strict=True):
        P = T.copy()
        if off is not None:
            P[:3, 3] += off
        f["prior"] = {"trust": "drifting", "T_map_cam": P.tolist(), "sigma_pos_m": 0.05,
                      "sigma_rot_deg": 2.0}
    return fr, files


def test_jobs_and_dry_runs_report_the_drift_correction(client: Client) -> None:  # noqa: F811
    """api.md §4 (oms/2.5): null without a drifting prior; the estimate's correction at the job's
    drifting frames, on dry runs as on jobs, and each prior residual against the prior as
    placed."""
    fr, files = frames_for(client, [base_pose(0), base_pose(30.0)], tag="a")
    job = client.update("m", fr, files, cameras={"cam": CAM})
    assert job["state"] == "committed" and job["quality"]["drift_correction"] is None
    poses = [base_pose(y) for y in (10.0, 20.0, 40.0, 50.0)]
    off = np.array([0.3, 0.0, 0.0])  # an odometry 0.3 m off the map frame
    fr, files = _drifting(client, poses, "d", off)
    d = dry(client, "m", fr, files)
    dc = d["quality"]["drift_correction"]
    assert dc is not None and dc["translation_m"] == pytest.approx(0.3, abs=0.05)
    assert dc["odo_scale"] == 1.0
    job = client.update("m", fr, files)
    assert job["state"] == "committed"
    jc = job["quality"]["drift_correction"]
    assert jc["translation_m"] == pytest.approx(0.3, abs=0.05)
    for row, T in zip(job["inputs"], poses, strict=True):
        assert row["status"] == "registered"
        # the committed pose lies near truth, so against the prior as placed the residual is small
        assert np.linalg.norm(np.asarray(row["T_map_cam"])[:3, 3] - T[:3, 3]) < 0.05
        assert row["prior_residual_m"] < 0.05
    # a new map whose frame drifting priors fix: the identity
    fr, files = _drifting(client, [base_pose(0)], "n")
    assert dry(client, "n", fr, files, cameras={"cam": CAM})["quality"]["drift_correction"] == {
        "translation_m": 0.0, "rotation_deg": 0.0, "odo_scale": 1.0}


def test_a_gated_job_reports_the_poses_it_would_commit(client: Client) -> None:  # noqa: F811
    """api.md §4 (oms/2.5): on a job ``accept_if`` rejects, the frames keep their statuses and
    ``T_map_cam``, sigmas and prior residuals are those the job would have committed; a rejected
    frame has none. Committed, ``max_displacement_m`` is the largest prior residual and the
    poses are the version's."""
    poses = [base_pose(0), base_pose(30.0)]
    for name in ("g", "c"):
        fr, files = frames_for(client, poses[:1], tag="a")
        assert client.update(name, fr, files, cameras={"cam": CAM})["state"] == "committed"
    P = [T.copy() for T in poses]
    P[1][:3, 3] += [0.0, 0.01, 0.0]
    fr, files = frames_for(client, poses[1:], tag="b", priors=P[1:])
    files["p"] = io.BytesIO(b"not an image").getvalue()
    fr.append({"name": "p", "image": "p", "camera_id": "cam"})
    gated = client.update("g", fr, files, accept_if={"max_displacement_m": 1e-9})
    done = client.update("c", fr, files)
    assert gated["state"] == "rejected" and done["state"] == "committed"
    assert "max_displacement_m" in gated["quality"]["failed_checks"]
    for a, b in zip(gated["inputs"], done["inputs"], strict=True):
        assert a["status"] == b["status"]
        for key in ("T_map_cam", "sigma_pos_m", "sigma_rot_deg", "prior_residual_m",
                    "prior_residual_deg"):
            assert a[key] == b[key], key
    rej = [r for r in gated["inputs"] if r["status"] == "rejected"]
    assert rej and all(r["T_map_cam"] is None and r["sigma_pos_m"] is None for r in rej)
    q = done["quality"]
    res = [r["prior_residual_m"] for r in done["inputs"]
           if r["status"] == "registered" and r["prior_residual_m"] is not None]
    assert q["max_displacement_m"] == max(res)
    frames = {f["name"]: f for f in client.get("/api/maps/c/frames", "Frames")["frames"]}
    for r in done["inputs"]:
        if r["status"] == "registered":
            assert np.allclose(frames[r["name"]]["T_map_cam"], r["T_map_cam"], atol=1e-5)


def test_the_service_reports_floor_not_judged(client: Client) -> None:  # noqa: F811
    """api.md §4 (oms/2.5): a monocular frame beyond the scale bound is registered with
    ``detail`` ``floor_not_judged``, no ``depth_scale``, and is not in ``floor_inconsistent``."""
    cam = {**CAM, "height_above_floor_m": 0.5, "height_sigma_m": 0.01}
    fr, files = frames_for(client, [base_pose(0)], tag="a", depth=False)
    assert client.update("m", fr, files, cameras={"cam": cam})["state"] == "committed"
    fr, files = frames_for(client, [base_pose(15.0 + 30.0 * k) for k in range(5)], tag="s",
                           depth=False)
    key = hashlib.sha256(files["s2"]).hexdigest()
    client.backend.mono_depth[key] = client.backend.mono_depth[key] * 0.5
    job = client.update("m", fr, files)
    assert job["state"] == "committed" and job["quality"]["floor_inconsistent"] == []
    rows = {i["name"]: i for i in job["inputs"]}
    assert rows["s2"]["status"] == "registered" and rows["s2"]["detail"] == "floor_not_judged"
    assert "depth_scale" not in rows["s2"]
    assert all(rows[f"s{k}"]["detail"] is None for k in (0, 1, 3, 4))


def _behind_map(c: Client, name: str, u_scale: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """A monocular map with a camera height (``a0``), then a job of three frames turning at the
    origin and ``u``, 0.9 m behind it, whose monocular depth is ``u_scale`` times the true
    one (0.5: its own scale beyond the bound). Returns the dry run of the job and the job."""
    cam = {**CAM, "height_above_floor_m": 0.5, "height_sigma_m": 0.01}
    fr, files = frames_for(c, [base_pose(0)], tag="a", depth=False)
    assert c.update(name, fr, files, cameras={"cam": cam})["state"] == "committed"
    poses = [base_pose(15.0 * (k + 1)) for k in range(3)] + [base_pose(0.0, -0.9)]
    fr, files = frames_for(c, poses, tag="s", depth=False)
    key = hashlib.sha256(files["s3"]).hexdigest()
    c.backend.mono_depth[key] = c.backend.mono_depth[key] * u_scale
    d = dry(c, name, fr, files)
    return d, c.update(name, fr, files)


def test_the_service_withholds_an_unjudged_frames_content_but_not_its_stand(
        client: Client) -> None:  # noqa: F811
    """P29b, prior-first engine: the unjudged frame ``s3`` keeps its registration and pose, is no
    rejection and is not in ``floor_inconsistent``, but its version row has no content: it gives
    no occupancy, no free space from its depth, no floor seen; it still marks free-by-rule cells
    (its stand). A dry run predicts the same quality as the job."""
    d, job = _behind_map(client, "m", 0.5)
    _, ctrl = _behind_map(client, "c", 1.0)
    assert job["state"] == ctrl["state"] == "committed"
    rows = {i["name"]: i for i in job["inputs"]}
    assert rows["s3"]["status"] == "registered" and rows["s3"]["detail"] == "floor_not_judged"
    assert rows["s3"]["T_map_cam"] is not None and "depth_scale" not in rows["s3"]
    assert rows["s3"]["sigma_pos_m"] is not None
    assert {i["name"]: i["detail"] for i in ctrl["inputs"]}["s3"] is None  # the control: judged
    assert job["quality"]["floor_inconsistent"] == ctrl["quality"]["floor_inconsistent"] == []
    # the dry run predicts what the job reports
    assert d["quality"]["floor_inconsistent"] == []
    for key in ("height_inconsistent", "rejected_share", "reregistered"):
        assert d["quality"][key] == job["quality"][key]
    assert [r["predicted_status"] for r in d["frames"]] == ["registered"] * 4
    rec = {f["name"]: f for f in client.service.store.version("m")["frames"]}
    assert not rec["s3"]["content"] and rec["s3"]["floor_inconsistent"]
    assert rec["s2"]["content"] and not rec["s2"]["floor_inconsistent"]
    # the grid: what the control sees through s3 is missing, its stand is free by rule
    g, c = (client.get(f"/api/maps/{n}/grid", "Grid") for n in ("m", "c"))
    assert g["origin_m"] == c["origin_m"] and g["shape"] == c["shape"]
    state, cstate = (decode(x["layers"]["state"]) for x in (g, c))
    seen, cseen = (decode(x["layers"]["floor_observed"]) for x in (g, c))
    assert int((cseen == 1).sum()) > int((seen == 1).sum())  # floor it saw, not seen now
    assert int((cstate == 2).sum()) > int((state == 2).sum())  # occupancy it gave, none now
    ox, oy = g["origin_m"]
    r, col = int((rec["s3"]["T_map_cam"][1][3] - oy) / g["cell_m"]), \
        int((rec["s3"]["T_map_cam"][0][3] - ox) / g["cell_m"])
    assert state[r, col] == 1 and seen[r, col] == 2  # free by rule, floor not seen


def test_input_rows_carry_detail_by_status() -> None:
    from oh_my_slam.service.prior_first import input_row

    T = np.eye(4)
    assert input_row("a", "registered", None, 50, T, 0.1, 1.0, None, None, None,
                     "floor_not_judged")["detail"] == "floor_not_judged"
    assert input_row("a", "provisional", None, 50, T, 0.1, 1.0, None, None, None,
                     "floor_not_judged")["detail"] == "floor_not_judged"
    assert input_row("a", "registered", None, 50, T, 0.1, 1.0, None, None, None,
                     "no_image_pose")["detail"] is None
    assert input_row("a", "kept_at_prior", "no_overlap", 0, T, 0.1, 1.0, (0.0, 0.0), None, None,
                     "no_image_pose")["detail"] == "no_image_pose"
    assert input_row("a", "kept_at_prior", "no_overlap", 0, T, 0.1, 1.0, (0.0, 0.0), None, None,
                     "floor_not_judged")["detail"] is None
    assert input_row("a", "rejected", "prior_conflict", 0, T, 0.1, 1.0, (0.3, 0.0), None, None,
                     "floor_not_judged")["detail"] is None


def test_the_older_mapper_does_not_judge_a_monocular_frame_beyond_the_bound() -> None:
    from oh_my_slam.service.update import Updater

    z, Kg, T = _floor_points(0.5)

    class Cam:
        height_above_floor_m, height_sigma_m = 0.5, 0.01

        @staticmethod
        def canonical() -> Intrinsics:
            return Kg

    class Work:
        height_m, height_sigma_m = None, None

    class Fr:
        internal = type("I", (), {"name": "k"})()
        pose, sigma_pos_m, cam, work = T, 0.01, Cam(), Work()
        depth_m: Any = None
        detail: str | None = None

    internal = type("M", (), {})()
    internal.depth = lambda name: z * 2.0
    fr = Fr()
    assert not Updater._floor_off(fr, internal, anchor.Sim3())  # type: ignore[arg-type]
    assert fr.detail == "floor_not_judged"
    assert fr.floor_unjudged  # P29b: left out of the grid's evidence, though not listed


def test_the_older_mapper_withholds_an_unjudged_frames_grid_evidence(
        old: Client, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """P29b, the older mapper: a frame the floor check did not judge keeps its registration,
    pose and ``detail`` ``floor_not_judged``, is not in ``floor_inconsistent`` and not rejected,
    but its version row feeds no occupancy, heights or free space (``derived._occupies``), as an
    excluded frame's; its camera still stands (free by rule). A dry run predicts the same
    quality. (The mapper's own cloud and objects, which its process builds, still hold it:
    ``Updater._floor_off``.)"""
    from oh_my_slam.service.derived import _occupies
    from oh_my_slam.service.update import Updater
    from tests.fakes.service import CAMERA, pose, textured

    def judge(fr: Any, internal: Any, sim: Any) -> bool:
        if fr.work.name == "t2":
            fr.detail, fr.floor_unjudged = "floor_not_judged", True
        return False

    monkeypatch.setattr(Updater, "_floor_off", staticmethod(judge))
    frames, files = [], {}
    for i, a in enumerate((0, 30, 60, 90)):
        T = pose(a, 0.0, 0.0, 1.2)
        files[f"t{i}"] = old.backend.add(textured(700 + i), T)
        frames.append({"name": f"t{i}", "image": f"t{i}", "camera_id": "cam",
                       "prior": prior(T), "height_above_floor_m": 1.2,
                       "height_sigma_m": 0.01})
    d = dry(old, "m", frames, files, cameras={"cam": CAMERA})
    job = old.update("m", frames, files, {"cam": CAMERA})
    assert d["quality"]["floor_inconsistent"] == job["quality"]["floor_inconsistent"] == []
    rows = {i["name"]: i for i in job["inputs"]}
    assert rows["t2"]["status"] == "registered" and rows["t2"]["detail"] == "floor_not_judged"
    assert rows["t2"]["T_map_cam"] is not None and rows["t1"]["detail"] is None
    rec = {r["name"]: r for r in old.service.store.version("m")["frames"]}
    assert rec["t2"]["floor_inconsistent"] and not _occupies(rec["t2"])  # no grid evidence
    assert _occupies(rec["t1"]) and rec["t2"]["pose_state"] == "fused"
    from oh_my_slam.service.derived import Derived

    stands = Derived(old.service.store, old.backend.read)._stands(
        old.service.store.version("m"), old.service.store.cameras("m"))
    assert len(stands) == 4  # its stand counts for the free-by-rule cells


# -- containment logging ----------------------------------------------------------------------------


class _Lines(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []
        self.addFilter(ContextFilter())
        self.setFormatter(logging.Formatter("%(name)s%(context)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


@pytest.fixture
def lines() -> Any:
    h = _Lines()
    root = logging.getLogger("oh_my_slam")
    root.addHandler(h)
    yield h.lines
    root.removeHandler(h)


def test_log_context_tags_lines(lines: list[str]) -> None:
    log = logging.getLogger("oh_my_slam.x")
    log.info("outside")
    with log_context("job j1"):
        log.info("inside")
    assert lines == ["oh_my_slam.x: outside", "oh_my_slam.x [job j1]: inside"]


def test_the_service_log_holds_job_and_locate_summaries(client: Client,  # noqa: F811
                                                        lines: list[str]) -> None:
    """README "Service log": each update job's lines carry its id, and it ends with one summary
    line (inputs' sigmas, priors, prior residuals, drift correction); each locate answer one
    line with its sigmas and prior residual."""
    fr, files = frames_for(client, [base_pose(0), base_pose(30.0)], tag="a")
    job = client.update("m", fr, files, cameras={"cam": CAM})
    tag = f"[job {job['job_id']}]"
    summary = [ln for ln in lines if "job summary" in ln]
    assert len(summary) == 1 and tag in summary[0]
    doc = json.loads(summary[0].split("job summary ", 1)[1])
    assert doc["job_id"] == job["job_id"] and doc["drift_correction"] is None
    rows = {r["name"]: r for r in doc["inputs"]}
    assert rows["a1"]["prior"] == "metric" and rows["a1"]["prior_sigma_pos_m"] == 0.02
    assert rows["a1"]["sigma_pos_m"] == job["inputs"][1]["sigma_pos_m"]
    assert rows["a1"]["prior_residual_m"] == job["inputs"][1]["prior_residual_m"]
    assert any(tag in ln and "summary" not in ln for ln in lines)
    from tests.fakes.service import multipart

    img, _, _ = service_shot(base_pose(15.0))
    req = {"frames": [{"name": "q", "image": "q", "camera_id": "cam",
                       "prior": {"trust": "metric", "T_map_cam": base_pose(15.0).tolist(),
                                 "sigma_pos_m": 0.05, "sigma_rot_deg": 2.0}}]}
    data, ctype = multipart(req, {"q": img})
    r = client.http.post("/api/maps/m/locate", content=data, headers={"content-type": ctype})
    assert r.status_code == 200, r.text
    loc = [ln for ln in lines if "locate summary" in ln]
    assert loc and "[locate m v1]" in loc[-1]
    got = json.loads(loc[-1].split("locate summary ", 1)[1])
    res = r.json()["frames"][0]
    assert got["name"] == "q" and got["sigma_pos_m"] == res["sigma_pos_m"]
    if res["status"] == "located":
        assert got["prior_residual_m"] is not None
    # an ended job without inputs logs no summary
    from oh_my_slam.service.jobs import Job, job_summary

    assert job_summary(Job("x", "m", 0.0)) is None


def test_the_locate_summary_measures_the_prior_residual() -> None:
    from oh_my_slam.service.nav import LocateFrame, locate_summary

    T = base_pose(0)
    P = T.copy()
    P[:3, 3] += [0.03, 0.04, 0.0]
    fr = LocateFrame("q", None, None, None, {"trust": "metric", "T_map_cam": P.tolist()},  # type: ignore[arg-type]
                     None, None)
    res = {"status": "located", "reason": None, "T_map_cam": T.tolist(), "sigma_pos_m": 0.1,
           "sigma_rot_deg": 1.0}
    got = locate_summary(fr, res)
    assert got["prior_residual_m"] == pytest.approx(0.05) and got["prior_residual_deg"] == 0.0
    res["T_map_cam"] = None
    assert locate_summary(fr, res)["prior_residual_m"] is None
    fr.prior = None
    assert locate_summary(fr, res)["prior"] is None
