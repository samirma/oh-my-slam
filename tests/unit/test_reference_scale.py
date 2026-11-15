"""oms/2.12: the measured monocular scale sigma (map.md §1) and locate's reference scale and
reference groups (api.md §6 "Reference scale", "Ambiguity"; ``mapping.relocalize``).

The locate scenes are synthetic: points on the walls of a round room with random descriptors,
keyframes taken from one spot looking in three directions (a robot turning on the spot), each
direction's keyframes one group of references whose stored depth may be scaled wrong about
their own centres (a monocular keyframe's depth scale error)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from oh_my_slam.mapping import anchored as an
from oh_my_slam.mapping import relocalize as rl
from oh_my_slam.service import store
from tests.synth import rooms

K = np.array([[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]])
TRUTH = rooms.camera(0.6, 0.0, 1.0, 0.0, -10.0)  # the located camera, 0.6 m from the spot


def _room(seed: int = 0, n: int = 1200) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.uniform(-math.pi, math.pi, n)
    r = rng.uniform(2.5, 3.5, n)
    P = np.column_stack([r * np.cos(a), r * np.sin(a), rng.uniform(0.0, 2.0, n)])
    return P, rng.integers(0, 256, (n, 128)).astype(np.uint8)


def _view(T: np.ndarray, P: np.ndarray, D: np.ndarray, noise: float = 0.0
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    uv, z = rl.project(T, K, P)
    ok = (z > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < 320) & (uv[:, 1] >= 0) & (uv[:, 1] < 240)
    uv = uv + np.random.default_rng(1).normal(0, noise, uv.shape)
    return uv[ok].astype(np.float32), D[ok], np.flatnonzero(ok)


def _kf(name: str, T: np.ndarray, P: np.ndarray, D: np.ndarray, scale: float = 1.0,
        s: float = an.MONOCULAR_SIGMA_REL, depth: bool = False,
        lifted: bool = True) -> rl.Keyframe:
    """A keyframe whose stored depth is ``scale`` times the truth (about its centre), of scale
    sigma ``s``; ``lifted`` False: its points triangulated."""
    uv, d, idx = _view(T, P, D)
    c = T[:3, 3]
    xyz = (c + scale * (P[idx] - c)).astype(np.float32)
    return rl.Keyframe(name, T, K, uv, d, xyz, 0.01, 0.1, depth, scale_sigma_rel=s,
                       lifted=None if lifted else np.zeros(len(uv), bool))


def _spot(P: np.ndarray, D: np.ndarray, scales: tuple[float, float, float],
          s: float = an.MONOCULAR_SIGMA_REL) -> rl.Index:
    """Six keyframes looking ahead (depth scaled by ``scales[0]``), two turned left
    (``scales[1]``) and two turned right (``scales[2]``), all within 0.25 m: one place."""
    kfs = [_kf(f"a{i}", rooms.camera(0.04 * i, 0.0, 1.0, yaw, -10.0), P, D, scales[0], s)
           for i, yaw in enumerate((-5.0, -3.0, -1.0, 1.0, 3.0, 5.0))]
    kfs += [_kf(f"l{i}", rooms.camera(0.05 * i, 0.0, 1.0, yaw, -10.0), P, D, scales[1], s)
            for i, yaw in enumerate((36.0, 39.0))]
    kfs += [_kf(f"r{i}", rooms.camera(0.05 * i, 0.0, 1.0, yaw, -10.0), P, D, scales[2], s)
            for i, yaw in enumerate((-36.0, -39.0))]
    return rl.Index(kfs)


def _query(P: np.ndarray, D: np.ndarray, T: np.ndarray = TRUTH) -> rl.Query:
    uv, d, _ = _view(T, P, D, 0.5)
    return rl.Query(uv, d, K, 320, 240)


# -- monocular scale sigma (map.md §1) ------------------------------------------------------------


def test_a_monocular_scale_sigma_is_the_measured_constant_whatever_the_frame_count() -> None:
    assert an.MONOCULAR_SIGMA_REL == 0.65
    for n in (0, 3, 30, 300):
        st = an.ScaleState()
        st.depth_residuals["mono"] = [0.2 * (-1) ** i for i in range(n)]
        st.estimate(0.0)
        assert (st.source, st.sigma_rel, st.k) == ("monocular", 0.65, 1.0)
    # a state or a version the nominal 0.10 wrote reads as the constant; other sources as recorded
    old = {**an.ScaleState().to_json(), "sigma_rel": 0.1}
    assert an.ScaleState.from_json(old).sigma_rel == 0.65
    assert an.ScaleState.from_json({**old, "source": "camera_height"}).sigma_rel == 0.1
    assert an.recorded_scale_sigma("monocular", 0.1) == 0.65
    assert an.recorded_scale_sigma("camera_height", 0.061) == 0.061


def test_the_floor_check_s_monocular_bound_is_its_own_constant() -> None:
    assert an.MONO_FLOOR_BOUND_REL == 0.30
    assert an.MONO_FLOOR_BOUND_REL != 3.0 * an.MONOCULAR_SIGMA_REL


def test_a_monocular_keyframe_s_own_scale_sigma_is_its_height_s() -> None:
    k = an.Keyframe("k", "f", "cam", an.Intrinsics(270, 270, 159.5, 119.5, 320, 240), np.eye(4),
                    0.05, 1.0, "fused", None, "mono", height_m=0.355, height_sigma_m=0.03)
    assert k.own_scale_sigma is None  # no own depth scale
    k.grid_scale, k.floor_factor = 1.0, 1.2
    assert k.own_scale_sigma == pytest.approx(0.03 / 0.355)
    k.height_sigma_m = None
    assert k.own_scale_sigma == pytest.approx(an.DEFAULT_HEIGHT_SIGMA_M / 0.355)
    k.height_sigma_m = 1e-6
    assert k.own_scale_sigma == an.MIN_SCALE_SIGMA_REL
    k.height_m = None
    assert k.own_scale_sigma is None
    assert store.own_scale_sigma(None) == {}
    assert store.own_scale_sigma(0.0845070) == {store.OWN_SCALE_SIGMA: 0.08451}


# -- reference scale (api.md §6) ------------------------------------------------------------------


def test_the_reference_scale_term_is_the_points_scale_sigma_times_their_median_distance(
        ) -> None:
    T = np.eye(4)
    xyz = np.array([[0, 0, 1.0], [0, 0, 2.0], [0, 0, 3.0], [0, 0, 9.0]])
    srel = np.array([0.1, 0.0, 0.1, 0.5])
    mask = np.array([True, True, True, False])
    # s the mean over the inliers (a triangulated point counts 0), d their median distance
    assert rl.reference_scale(T, xyz, srel, mask) == pytest.approx((0.2 / 3) * 2.0)
    assert rl.reference_scale(T, xyz, srel, np.zeros(4, bool)) == 0.0
    # in a locate: hypot(the sigma without it, s x d), d from the located camera
    P, D = _room()
    q = _query(P, D)
    bare = rl.locate(q, _spot(P, D, (1.0, 1.0, 1.0), s=0.0))
    held = rl.locate(q, _spot(P, D, (1.0, 1.0, 1.0), s=0.1))
    assert bare.located and held.located and bare.T is not None
    assert bare.sigma_pos_m is not None and held.sigma_pos_m is not None
    seen = _view(TRUTH, P, D)[2]  # the points it sees, nearly all inliers
    d = float(np.median(np.linalg.norm(P[seen] - bare.T[:3, 3], axis=1)))
    assert held.sigma_pos_m == pytest.approx(math.hypot(bare.sigma_pos_m, 0.1 * d), rel=0.05)
    assert held.sigma_rot_deg == bare.sigma_rot_deg  # the position's only


def test_a_pose_on_triangulated_points_has_no_reference_scale_term() -> None:
    P, D = _room()
    q = _query(P, D)
    a = _kf("a", rooms.camera(0.0, 0.0, 1.0, -3.0, -10.0), P, D, s=0.0)
    b = _kf("b", rooms.camera(0.1, 0.0, 1.0, 3.0, -10.0), P, D, s=0.0)
    bare = rl.locate(q, rl.Index([a, b]))
    tri = rl.locate(q, rl.Index([
        _kf("a", a.T, P, D, s=0.65, lifted=False), _kf("b", b.T, P, D, s=0.65, lifted=False)]))
    assert bare.located and (tri.sigma_pos_m, tri.sigma_rot_deg) == (bare.sigma_pos_m,
                                                                       bare.sigma_rot_deg)


def test_a_sensor_depth_map_s_locate_sigma_is_unchanged_within_tolerance() -> None:
    """A depth image's scale sigma (half a percent) adds millimetres at 3 m: the depth floor's
    12 mm stays the larger part."""
    P, D = _room()
    q = _query(P, D)
    kw = {"depth": True}
    a0, b0 = (_kf(n, rooms.camera(x, 0.0, 1.0, yaw, -10.0), P, D, s=0.0, **kw)
              for n, x, yaw in (("a", 0.0, -3.0), ("b", 0.1, 3.0)))
    before = rl.locate(q, rl.Index([a0, b0]))
    after = rl.locate(q, rl.Index([_kf("a", a0.T, P, D, s=an.DEPTH_SENSOR_SIGMA_REL, **kw),
                                   _kf("b", b0.T, P, D, s=an.DEPTH_SENSOR_SIGMA_REL, **kw)]))
    assert before.sigma_pos_m is not None and after.sigma_pos_m is not None
    assert 0.012 <= before.sigma_pos_m <= after.sigma_pos_m < before.sigma_pos_m + 0.01
    assert after.sigma_pos_m < 0.03


# -- reference groups (api.md §6 "Ambiguity") -----------------------------------------------------


def test_each_group_of_references_that_disagrees_is_a_rival(monkeypatch: pytest.MonkeyPatch
                                                            ) -> None:
    """The keyframes turned left hold depth 1.4 times too far, those turned right 0.7 times: each
    group's own solve is a rival of the true pose; without the groups only one of them is (the
    second fit's)."""
    P, D = _room()
    q = _query(P, D)
    idx = _spot(P, D, (1.0, 1.4, 0.7))
    o = rl.locate(q, idx)
    assert o.located and o.T is not None and rl.pose_distance(o.T, TRUTH)[0] < 0.02
    assert len(o.rivals) == 2 and o.ambiguity is not None
    assert all(0.1 < r["pos_m"] < 0.4 and r["support"] >= rl.SUPPORT_MIN for r in o.rivals)
    monkeypatch.setattr(rl, "reference_groups", lambda per_kf, index: [])
    assert len(rl.locate(q, idx).rivals) == 1


def test_a_biased_reported_pose_meets_a_rival_and_its_sigma_covers_the_error() -> None:
    """The bigger group's depth is 1.4 times too far, so the reported pose is its biased solve:
    the true pose is its rival, and the reference scale term covers the error within 3 sigma,
    with the monocular sigma and with a camera-height map's."""
    P, D = _room()
    q = _query(P, D)
    bare = rl.locate(q, _spot(P, D, (1.4, 1.0, 1.0), s=0.0)).sigma_pos_m
    assert bare is not None
    for s in (an.MONOCULAR_SIGMA_REL, 0.061):
        o = rl.locate(q, _spot(P, D, (1.4, 1.0, 1.0), s=s))
        assert o.located and o.T is not None and o.sigma_pos_m is not None
        err = rl.pose_distance(o.T, TRUTH)[0]
        assert err > 0.1  # biased
        assert o.ambiguity is not None and o.ambiguity["pos_m"] == pytest.approx(err, abs=0.05)
        assert err <= 3.0 * o.sigma_pos_m and o.sigma_pos_m > bare


def test_reference_groups_split_the_matched_keyframes_by_place_and_direction() -> None:
    P, D = _room()
    idx = _spot(P, D, (1.0, 1.0, 1.0))
    per = [(40, i) for i in range(6)] + [(20, 6), (15, 7), (25, 8), (6, 9)]
    got = rl.reference_groups(per, idx)
    assert got == [[5, 4, 3, 2, 1, 0], [6, 7], [8, 9]]  # most correspondences first
    # a group short of GROUP_MIN_POINTS correspondences is none; one group alone is none
    assert rl.reference_groups([(40, 0), (40, 1), (10, 6), (10, 8)], idx) == []
    assert rl.reference_groups([(40, 0), (40, 6), (10, 8)], idx) == [[6], [0]]
    # at most RIVAL_GROUPS: keyframes far apart are groups of their own too
    far = rl.Index([_kf(f"f{i}", rooms.camera(2.0 * i, 0.0, 1.0, 0.0, -10.0), P, D)
                    for i in range(rl.RIVAL_GROUPS + 2)])
    many = rl.reference_groups([(40 - i, i) for i in range(rl.RIVAL_GROUPS + 2)], far)
    assert many == [[i] for i in range(rl.RIVAL_GROUPS)]


def test_triangulated_marks_the_points_still_lifted_from_the_stored_depth() -> None:
    P, D = _room(n=1500)
    a = _kf("a", rooms.camera(0.0, 0.0, 1.0, 0.0, -10.0), P, D, scale=1.2)
    b = _kf("b", rooms.camera(0.0, 0.6, 1.0, -5.0, -10.0), P, D, scale=1.2)
    far = _kf("c", rooms.camera(0.0, 0.0, 1.0, 180.0, -10.0), P, D)
    pts, lifted = rl.triangulated([a, b, far])
    assert len(pts) == len(lifted) == 3
    # a and b: triangulated, or their depth rescaled to the triangulation: none lifted as stored
    assert not lifted[0].any() and not lifted[1].any()
    assert lifted[2].all()  # no neighbour: its depth as stored
    _, single = rl.triangulated([a])
    assert single[0].all()
    # too few points to measure the depth's scale: only the triangulated ones are not lifted
    few = rl.Keyframe("few", a.T, K, a.uv, a.desc, a.xyz.copy())
    _, got = rl.triangulated([few, b], lambda i, j: rl._pair(*(
        (few, b) if i == 0 else (b, few)))[:12])
    assert got[0].any() and not got[0].all()
    assert a.lifted_mask().all()
    a.xyz[0] = np.nan
    assert not a.lifted_mask()[0]
    tri = _kf("t", a.T, P, D, lifted=False)
    assert not tri.lifted_mask().any()
