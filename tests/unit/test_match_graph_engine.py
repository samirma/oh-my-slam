"""The oms/2.2 P10 rules in the prior-first engine (``mapping.anchored``): the match graph's
edge (the registration minimum), ``detail`` of kept frames and their sigmas, re-registration in
the same job, the robust drift estimate, the height gate and static image content."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from oh_my_slam.mapping import anchored as an
from oh_my_slam.mapping.anchored import AnchoredMap
from tests.synth.room import base_pose
from tests.unit.test_anchored import K, err, frame, new_map, prior


def _few_matches(monkeypatch: pytest.MonkeyPatch, n: int, desc: np.ndarray) -> None:
    """At most ``n`` matches for the frame of descriptors ``desc``, no guided ones."""
    real = an.match

    def match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ia, ib = real(a, b)
        if a is desc or b is desc:
            return ia[:n], ib[:n]
        return ia, ib

    monkeypatch.setattr(an, "match", match)
    monkeypatch.setattr(an, "guided_match", lambda *a, **k: (np.zeros(0, np.intp),
                                                             np.zeros(0, np.intp)))


@pytest.mark.parametrize(("n", "registered"), [(40, True), (10, False)])
def test_an_edge_needs_the_registration_minimum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                n: int, registered: bool) -> None:
    """map.md §2: about 40 inliers to a confirmed keyframe and a good prior register the frame;
    10 (below ``MIN_INLIERS``) are no edge: it is kept, ``no_overlap``, ``no_image_pose``."""
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0))])
    f = frame("b", base_pose(10))
    feats = an.features(f.rgb)
    monkeypatch.setattr(an, "features", lambda *a, **k: feats)
    _few_matches(monkeypatch, n, feats.desc)
    p = m.update([f])[0]
    if registered:
        assert p.status == "registered" and an.MIN_INLIERS <= p.inliers <= n
        assert err(p.T, base_pose(10))[0] < 0.05  # type: ignore[arg-type]
    else:
        assert (p.status, p.reason, p.detail, p.inliers) == (
            "kept_at_prior", "no_overlap", "no_image_pose", 0)


def test_kept_frames_say_why_and_keep_the_prior_s_sigmas(tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """api.md §4 ``detail``: no_image_pose, loose_image_pose, conflict_unsupported; null for a
    failed check. A kept frame's sigmas are its prior's as given (not floored at 1 cm)."""
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0)), frame("a1", base_pose(30))])
    away = frame("away", base_pose(180), p=prior(base_pose(180), sp=0.004, sr=0.3))
    p = m.update([away])[0]
    assert (p.reason, p.detail) == ("no_overlap", "no_image_pose")
    assert (p.sigma_pos_m, p.sigma_rot_deg) == (0.004, 0.3)
    T = base_pose(20)
    wrong = T.copy()
    wrong[:3, 3] += [0.5, 0.0, 0.0]
    monkeypatch.setattr(an, "MEASURABLE_INLIERS", 10 ** 6)
    p = m.update([frame("weak", T, p=prior(wrong))])[0]
    assert (p.status, p.reason, p.detail) == ("kept_at_prior", "no_overlap",
                                              "conflict_unsupported")
    monkeypatch.setattr(an.Solve, "sigmas", lambda self: (0.5, 5.0))
    p = m.update([frame("loose", base_pose(15))])[0]
    assert (p.status, p.reason, p.detail) == ("kept_at_prior", "no_overlap", "loose_image_pose")
    bad = frame("bad", base_pose(10))
    bad.unconfirmed = "blur"
    p = m.update([bad])[0]
    assert (p.status, p.reason, p.detail) == ("kept_at_prior", "blur", None)


def _drift_map(tmp_path: Path) -> AnchoredMap:
    m = new_map(tmp_path)
    m.update([frame(f"a{k}", base_pose(30 * k), p="drifting") for k in range(4)])
    return m


def _rot_z(T: np.ndarray, deg: float) -> np.ndarray:
    a = math.radians(deg)
    R = np.eye(4)
    R[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    return R @ T


def test_one_mismatched_drifting_frame_moves_no_other_frame(tmp_path: Path) -> None:
    """map.md §1 "Drift estimate": among good drifting priors, one whose prior is 86° off the
    pose its images give is left out of the fit and rejected ``prior_conflict``, its residual
    measured against the prior as sent through the drift estimate; the others are placed as
    without it, and the drift estimate does not swing."""
    truth = [base_pose(y) for y in (40, 50, 70, 80)]
    good = [frame(f"g{k}", T, p=prior(T, "drifting", 0.15, 7.4)) for k, T in enumerate(truth)]
    alone = _drift_map(tmp_path / "alone")
    ref = alone.update(good)
    m = _drift_map(tmp_path / "m")
    bad_T = base_pose(55)
    bad = frame("bad", bad_T, p=prior(_rot_z(bad_T, 86.0), "drifting", 0.15, 7.4))
    pls = m.update([*good[:2], bad, *good[2:]])
    assert (pls[2].status, pls[2].reason) == ("rejected", "prior_conflict")
    assert pls[2].prior_residual is not None and pls[2].prior_residual[1] > 80.0
    for a, b in zip([*pls[:2], *pls[3:]], ref, strict=True):
        assert a.status == b.status == "registered"
        dp, dr = err(a.T, b.T)  # type: ignore[arg-type]
        assert dp < 0.01 and dr < 0.2
    P = np.asarray(good[0].prior["T_map_cam"])  # type: ignore[index]
    dp, dr = err(m.drift.predict(P, 0.1, 1.0)[0], alone.drift.predict(P, 0.1, 1.0)[0])
    assert dp < 0.01 and dr < 0.2


def test_a_drift_change_needs_three_frames_and_half_the_job(tmp_path: Path) -> None:
    """Two of five drifting frames agreeing on a 25° slip cannot change the estimate: they are
    rejected ``prior_conflict`` and the rest registered; three of three can. (Since oms/2.5 the
    fit weighs the images against the priors as sent: the slipped priors' 5° sigma allows the
    correction, a 1° one would not, map.md §1.)"""
    truth = [base_pose(y) for y in (40, 50, 70, 80, 100)]
    slips = [0.0, 0.0, 0.0, 25.0, 25.0]
    m = _drift_map(tmp_path)
    pls = m.update([frame(f"s{k}", T, p=prior(_rot_z(T, d), "drifting", sr=5.0 if d else 1.0))
                    for k, (T, d) in enumerate(zip(truth, slips, strict=True))])
    assert [(p.status, p.reason) for p in pls] == [("registered", None)] * 3 + [
        ("rejected", "prior_conflict")] * 2
    m2 = _drift_map(tmp_path / "two")
    pls = m2.update([frame(f"s{k}", T, p=prior(_rot_z(T, 25.0), "drifting", sr=5.0))
                     for k, T in enumerate(truth[2:])])
    assert [p.status for p in pls] == ["registered"] * 3
    for p, T in zip(pls, truth[2:], strict=True):
        assert err(p.T, T)[1] < 1.0  # type: ignore[arg-type]


def test_drift_fit_helpers() -> None:
    T = base_pose(0)
    pl = an.Placement("x", "registered", T=T, inliers=150, inlier_share=0.3, measurable=True)
    o = an._DriftObs(0, "x", T, T, (0.02, 1.0), (0.01, 0.5), False, pl)
    d = an.Drift(T, T)
    assert an._drift_inliers(d, [o]) == {0}
    assert an._drift_outliers(d, [o], {0}) == {}
    far = an.Drift(_rot_z(T, 40.0), T)
    out = an._drift_outliers(far, [o], set())
    assert out[0].reason == "prior_conflict" and out[0].prior_residual[1] > 30.0  # type: ignore[index]
    # R-9: it keeps its overlap, so it still counts and can support in the next fit
    assert out[0].inlier_share == 0.3 and an._can_support(out[0])
    # P16: without a measurable pose, a frame left out is kept, conflict_unsupported
    weak = an._DriftObs(0, "x", T, T, (0.02, 1.0), (0.01, 0.5), False,
                        an.Placement("x", "registered", T=T, inliers=150, inlier_share=0.3))
    kept = an._drift_outliers(far, [weak], set())[0]
    assert (kept.status, kept.reason, kept.detail) == ("kept_at_prior", "no_overlap",
                                                       "conflict_unsupported")
    assert kept.align_T is not None and kept.image_sigmas == (0.01, 0.5)
    assert an._can_support(kept)
    fitted = an._refined_drift(d, [o])
    assert err(fitted.ref_T, T)[0] < 1e-9  # type: ignore[arg-type]
    sp, sr = an._drift_sigma([o, o])
    assert sp == pytest.approx(math.hypot(0.02, 0.01) / math.sqrt(2))


def test_a_kept_frame_re_registers_in_the_same_job(tmp_path: Path) -> None:
    """map.md §2 step 5: a stored kept frame that a registered frame of the job overlaps (≥ 10 %
    of its view), with the registration minimum of inliers and its prior's checks passed on the
    fitted pose, is confirmed by its images in the same job, and is then a reference for the
    job's later rounds. Its depth alone never confirms it; a frame overlapping less than 10 % of
    its view, or a prior the images conflict with, leaves it kept."""
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0))])
    p = m.update([frame("k", base_pose(95))])[0]  # nothing overlaps it yet
    assert (p.status, p.detail) == ("kept_at_prior", "no_image_pose")
    m.save()
    m = AnchoredMap.load(m.root)
    # b registers on a0, re-registers k, and c (seen by k only) registers on k
    pls = m.update([frame("b", base_pose(55)), frame("c", base_pose(135))])
    assert [p.status for p in pls] == ["registered", "registered"]
    assert [e["name"] for e in m.reregistered] == ["k"]
    assert m.keyframe(m.reregistered[0]["keyframe"]).usable()  # type: ignore[union-attr]
    # less than 10 % of its view overlapped: not tried
    m2 = new_map(tmp_path / "m2")
    m2.update([frame("a0", base_pose(0))])
    m2.update([frame("k", base_pose(95))])
    m2.update([frame("b", base_pose(30))])
    assert m2.reregistered == []
    # its images conflict with its prior: stays kept
    m3 = new_map(tmp_path / "m3")
    m3.update([frame("a0", base_pose(0))])
    off = base_pose(95)
    off[:3, 3] += [0.6, 0.0, 0.0]
    m3.update([frame("k", base_pose(95), p=prior(off))])
    m3.update([frame("b", base_pose(55))])
    assert m3.reregistered == [] and not m3.keyframes[1].usable()
    # a bare stored image: its depth would place it, its images cannot
    m4 = new_map(tmp_path / "m4")
    m4.update([frame("a0", base_pose(0))])
    blank = np.full((K.height, K.width, 3), 120, np.uint8)
    m4.update([frame("k", base_pose(50), rgb=blank)])
    m4.update([frame("b", base_pose(40))])
    assert m4.reregistered == [] and not m4.keyframes[1].usable()


def test_the_height_gate_keeps_a_registered_frame_out_of_content(tmp_path: Path) -> None:
    """map.md §2: a frame whose committed camera height disagrees with the one sent by more than
    3 × sqrt(height_sigma² + σ_z²) stays registered (a reference) but adds nothing to the
    cloud, and is listed in ``height_inconsistent``."""
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0), height=0.5)])
    n = len(m.cloud.keys)
    ok = frame("ok", base_pose(20), height=0.5)
    ok.height_sigma_m = 0.01
    off = frame("off", base_pose(40), height=0.8)
    off.height_sigma_m = 0.01
    pls = m.update([ok, off])
    assert [p.status for p in pls] == ["registered", "registered"]
    assert m.height_inconsistent == ["off"]
    k_ok, k_off = m.keyframe(pls[0].keyframe), m.keyframe(pls[1].keyframe)  # type: ignore[arg-type]
    assert k_ok.height_ok and not k_off.height_ok  # type: ignore[union-attr]
    assert k_off.usable() and not k_off.contributes()  # type: ignore[union-attr]
    assert len(m.cloud.keys) > n
    # without a height sigma, or on a camera-only map without a camera height: no gate
    f = frame("x", base_pose(30), height=0.8)
    assert m.height_consistent(f, pls[1])
    m2 = new_map(tmp_path / "c")
    m2.created_with = "none"
    f.height_sigma_m = 0.01
    assert m2.height_consistent(f, pls[1])


def _with_blob(rgb: np.ndarray) -> np.ndarray:
    """A textured patch fixed in the image (the robot's own shadow, a lens mark)."""
    out = rgb.copy()
    rng = np.random.default_rng(7)
    out[180:236, 20:120] = rng.integers(0, 255, (56, 100, 1), dtype=np.uint8)
    out[180:236, 20:120] = cv2.GaussianBlur(out[180:236, 20:120], (3, 3), 0)
    return out


def _blob_feats(f: Any) -> int:
    k = f.kps
    return int(((k[:, 0] >= 20) & (k[:, 0] < 120) & (k[:, 1] >= 180) & (k[:, 1] < 236)).sum())


def test_static_content_is_left_out_of_matching_in_a_turn() -> None:
    """map.md §1 "Static image content": a blob fixed in the image across three frames turned by
    more than 5° from each other is left out of their features; a job that does not turn keeps
    all its features."""
    turn = [frame(f"t{k}", base_pose(10 * k)) for k in range(3)]
    for f in turn:
        f.rgb = _with_blob(f.rgb)
    prep = [an._Prep(an.features(f.rgb), None, False, None) for f in turn]
    before = [(_blob_feats(p.feats), len(p.feats.kps)) for p in prep]
    an.static_content(turn, prep)
    for (blob, total), p in zip(before, prep, strict=True):
        # the blob's features out (but those at its rim, whose descriptors see the scene
        # move behind them), the scene's kept
        gone = blob - _blob_feats(p.feats)
        assert gone > 0.6 * blob and total - len(p.feats.kps) <= gone + 5
    still = [frame(f"s{k}", base_pose(0, 0.02 * k)) for k in range(3)]
    for f in still:
        f.rgb = _with_blob(f.rgb)
    prep = [an._Prep(an.features(f.rgb), None, False, None) for f in still]
    n = [len(p.feats.kps) for p in prep]
    an.static_content(still, prep)
    assert [len(p.feats.kps) for p in prep] == n
    # two frames are no job to measure; without priors the rotation is the images'
    an.static_content(turn[:2], prep[:2])
    for f in turn:
        f.prior = None
    prep = [an._Prep(an.features(f.rgb), None, False, None) for f in turn]
    blob = [_blob_feats(p.feats) for p in prep]
    an.static_content(turn, prep)
    assert all(_blob_feats(p.feats) < 0.4 * b for p, b in zip(prep, blob, strict=True))


def test_pair_rotation_without_enough_matches() -> None:
    a, b = frame("a", base_pose(0), p=None), frame("b", base_pose(180), p=None)
    empty = an.Features(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8))
    assert an._pair_rotation(a, empty, b, empty) is None


def _feats(kps: list[tuple[float, float]], seed: int = 0) -> an.Features:
    rng = np.random.default_rng(seed)
    d = rng.random((len(kps), 128)) + 0.01
    d = np.sqrt(d / d.sum(axis=1, keepdims=True))
    return an.Features(np.array(kps, np.float32), np.round(d * an.ROOTSIFT_QUANT).astype(np.uint8))


def test_static_content_needs_three_frames_turned_from_each_other() -> None:
    """A keypoint fixed in frames that did not turn from each other stays; a frame with no
    keypoint near another's is passed over."""
    fs = [frame(f"t{k}", base_pose(y)) for k, y in enumerate((0, 2, 20, 40))]
    blob = _feats([(10.0, 10.0)], 1)
    far = _feats([(300.0, 200.0)], 2)
    prep = [an._Prep(blob, None, False, None), an._Prep(blob, None, False, None),
            an._Prep(far, None, False, None), an._Prep(blob, None, False, None)]
    an.static_content(fs, prep)
    # t0 and t1 turned 2° apart: one of them in the group; t0, t3 and one more frame are not
    # three frames turned pairwise beyond 5°, t2 holds nothing there
    assert [len(p.feats.kps) for p in prep] == [1, 1, 1, 1]
    fs[2].prior = prior(base_pose(20))
    prep[2] = an._Prep(blob, None, False, None)
    an.static_content(fs, prep)
    # t1 is fixed across t1, t2, t3 (turned pairwise): every frame loses it
    assert [len(p.feats.kps) for p in prep] == [0, 0, 0, 0]


def test_fusion_outside_a_job_and_a_missing_depth_grid(tmp_path: Path) -> None:
    """A keyframe added outside an update fuses at once when it carries content (MapAnything's
    frames), not at all when it does not; a keyframe whose depth file is gone has no grid."""
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0))])
    n = len(m.cloud.keys)
    f = frame("x", base_pose(10))
    kept = an.Placement("x", "kept_at_prior", "no_overlap", base_pose(10), 0.02, 1.0)
    m.add(f, an.features(f.rgb), kept, f.depth, K, "sensor", None)  # type: ignore[arg-type]
    assert len(m.cloud.keys) == n
    m.save()
    (m.root / an.KF_DIR / f"{m.keyframes[0].id}.depth.npy").unlink()
    m = AnchoredMap.load(m.root)
    assert m._grid(m.keyframes[0]) is None
