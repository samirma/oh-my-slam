"""Round 14 (P26, oms/2.4): the per-frame floor check (sensor and monocular), the sent camera
height as a z measurement of the registration, the prior's ``sigma_tilt_deg``, a robust
``floor_sigma_m`` and a floor band tested on the surface's height."""

from __future__ import annotations

import functools
import hashlib
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oh_my_slam.core.types import Intrinsics
from oh_my_slam.mapping import anchor, navgrid, sigmas
from oh_my_slam.mapping import anchored as an
from oh_my_slam.mapping.anchored import AnchoredMap, FrameIn, Keyframe
from oh_my_slam.service.derived import _occupies
from tests.synth.room import Room, base_pose, look, render
from tests.unit.test_service_api import Client
from tests.unit.test_service_prior_first import (
    CAM,
    client,  # noqa: F401
    dry,
    frames_for,
)

K = Intrinsics(160.0, 160.0, 160.0, 120.0, 320, 240)
ROOM = Room()
H = 0.5  # base_pose's camera height


@functools.lru_cache(maxsize=512)
def _render(room: str, key: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    return render(ROOMS[room], np.array(key).reshape(4, 4), K)


def shot(T: np.ndarray, room: str = "room") -> tuple[np.ndarray, np.ndarray]:
    return _render(room, tuple(np.round(T, 9).ravel().tolist()))


def prior(T: np.ndarray, trust: str = "metric", sp: float = 0.02, sr: float = 1.0,
          st: float | None = None) -> dict[str, Any]:
    p = {"trust": trust, "T_map_cam": np.asarray(T).tolist(), "sigma_pos_m": sp,
         "sigma_rot_deg": sr}
    if st is not None:
        p["sigma_tilt_deg"] = st
    return p


def frame(name: str, T: np.ndarray, *, p: dict[str, Any] | None = None, depth: float | None = 1.0,
          mono: float | None = None, height: float | None = H, hs: float | None = None,
          room: str = "room", P: np.ndarray | None = None) -> FrameIn:
    """A frame of the synthetic room: its depth image times ``depth`` (None: none), its
    monocular depth times ``mono`` (None: none), a metric prior at ``P`` (default its pose)."""
    img, z = shot(T, room)
    pr = prior(T if P is None else P) if p is None else p
    m = None if mono is None else (lambda d=(z * mono).astype(np.float32): (
        d, K, None, np.array([0.0, -1.0, 0.0])))
    return FrameIn(name, "cam", img, K, pr, None if depth is None else z * depth, None, height,
                   hs, 1000, m)


def pitch_of(T: np.ndarray) -> float:
    """The optical axis's elevation (deg)."""
    return math.degrees(math.asin(float(np.clip(T[2, 2], -1.0, 1.0))))


def tilt_err(T: np.ndarray, G: np.ndarray) -> float:
    return sigmas.yaw_tilt(T[:3, :3], G[:3, :3])[1]


# -- pieces ----------------------------------------------------------------------------------------


def test_yaw_and_tilt_of_a_rotation() -> None:
    A = base_pose(30.0)[:3, :3]
    B = an.exp_so3(np.array([0.0, 0.0, math.radians(10.0)])) @ A
    assert sigmas.yaw_tilt(B, A) == pytest.approx((10.0, 0.0), abs=1e-6)
    C = an.exp_so3(np.array([math.radians(3.0), 0.0, 0.0])) @ A
    assert sigmas.yaw_tilt(C, A) == pytest.approx((0.0, 3.0), abs=1e-6)
    assert sigmas.tilt_of(B, A, None) is None
    assert sigmas.tilt_of(C, A, 0.5) == pytest.approx((0.0, 3.0, 0.5), abs=1e-6)
    # level_toward: the yaw stays R's, the tilt goes to G's by the gain
    L = an.level_toward(B, C, 1.0)
    assert sigmas.yaw_tilt(L, B)[0] < 1e-6 and sigmas.yaw_tilt(L, C)[1] < 1e-6
    assert np.allclose(an.level_toward(B, C, 0.0), B)
    assert sigmas.axes_rms(2.0, None) == pytest.approx(2.0)
    assert sigmas.axes_rms(3.0, 0.0) == pytest.approx(math.sqrt(3.0))
    assert list(sigmas.rotation_axes(5.0, 1.0)) == [1.0, 1.0, 5.0]
    assert sigmas.tilt_sigma(None) is None and sigmas.tilt_sigma({"trust": "metric"}) is None
    assert sigmas.tilt_sigma({"sigma_tilt_deg": 0.01}, 0.1) == 0.1


def test_the_conflict_bound_per_part_with_a_tilt_sigma() -> None:
    """map.md §1 "Prior rotation sigma": yaw against sigma_rot, tilt against sigma_tilt."""
    # 8 deg of yaw against a 10 deg yaw sigma: no conflict; 3 deg of tilt against 0.5 deg: one
    assert not sigmas.conflicts(0.0, 8.0, (0.05, 10.0), (0.01, 1.0), tilt=(8.0, 0.0, 0.5))
    assert sigmas.conflicts(0.0, 3.0, (0.05, 10.0), (0.01, 0.5), tilt=(0.0, 3.0, 0.5))
    assert not sigmas.conflicts(0.0, 3.0, (0.05, 10.0), (0.01, 0.5))  # without: within 10 deg
    d0 = sigmas.distance(0.0, 3.0, (0.05, 10.0), (0.01, 0.5))
    d1 = sigmas.distance(0.0, 3.0, (0.05, 10.0), (0.01, 0.5), (0.0, 3.0, 0.5))
    assert d1 > 5 * d0
    # the anchors' chi of the older mapper
    T = base_pose(0.0)
    tilted = T.copy()
    tilted[:3, :3] = an.exp_so3(np.array([math.radians(3.0), 0.0, 0.0])) @ T[:3, :3]
    a = anchor.Anchor("a", "prior", tilted, T, 0.05, 10.0)
    assert anchor.chi(anchor.Sim3(), a) < anchor.K_SIGMA
    a.sigma_tilt_deg = 0.5
    assert anchor.chi(anchor.Sim3(), a) > anchor.K_SIGMA
    assert anchor.residual_parts(anchor.Sim3(), a) == pytest.approx((0.0, 3.0), abs=1e-6)


def test_measurement_fusion_reduces_to_the_isotropic_rules() -> None:
    """``fuse_measures``: two isotropic estimates give the weighted mean and its own sigma;
    a height measurement pulls the centre's z and, by the image's correlation, its pitch."""
    R, c = base_pose(20.0)[:3, :3], np.array([1.0, 2.0, 0.5])
    P = an.pose(an.exp_so3(np.array([0.01, -0.02, 0.03])) @ R, c + [0.03, -0.02, 0.05])
    cov = np.diag([1e-6] * 3 + [1e-4] * 3)
    w = np.linalg.pinv(an._scaled_cov(cov, 0.05, 1.0, True))
    ms = [an.Measure(np.zeros(6), w, an._scaled_cov(cov, 0.04, 0.8, True)),
          an.prior_measure(R, c, P, 0.02, 1.5, None, (0.02, 1.5, None))]
    T, sp, _ = an.fuse_measures(R, c, ms)
    wa, wb = 1 / 0.05 ** 2, 1 / 0.02 ** 2
    assert np.allclose(T[:3, 3], (c * wa + P[:3, 3] * wb) / (wa + wb))
    g = wb / (wa + wb)
    assert sp == pytest.approx(math.hypot((1 - g) * 0.04, g * 0.02))
    # a degenerate covariance: isotropic; a rotation-only pose: no centre information
    iso = an._scaled_cov(np.zeros((6, 6)), 0.05, 1.0, True)
    assert np.allclose(np.diag(iso), [math.radians(1.0) ** 2] * 3 + [0.0025] * 3)
    assert not an._scaled_cov(cov, 0.05, 1.0, False)[3:, 3:].any()
    # z and pitch correlated (a camera turning in place): the height corrects both
    C = np.eye(6) * 1e-6
    C[5, 5], C[0, 0] = 1e-4, 1e-4
    C[0, 5] = C[5, 0] = 0.9e-4
    W = np.linalg.pinv(an._scaled_cov(C, 0.05, 1.0, True))
    T2, _, _ = an.fuse_measures(R, c, [an.Measure(np.zeros(6), W, an._scaled_cov(
        C, 0.05, 1.0, True)), an.height_measure(c, 0.4, 0.01)])
    assert T2[2, 3] < c[2] - 0.05 and an.rot_deg(T2[:3, :3] @ R.T) > 0.5


def test_floor_sigma_is_a_robust_standard_deviation() -> None:
    """map.md §6 "Floor sigma": 1.4826 MAD of the floor points' heights, not a half-width: a
    sheet of a few bad frames 0.15 m up does not move it, nor shrink it."""
    rng = np.random.default_rng(0)
    z = rng.normal(0.0, 0.01, 4000)
    assert an.floor_spread(z, 0.0) == pytest.approx(0.01, rel=0.1)
    sheet = np.concatenate([z, rng.normal(0.11, 0.005, 400)])  # 10 %: outside the MAD's reach
    assert an.floor_spread(sheet, 0.0) < 0.013
    wide = np.concatenate([rng.normal(0.0, 0.01, 2000), rng.normal(0.0, 0.04, 2000)])
    assert an.floor_spread(wide, 0.0) > 0.015  # a thick floor is reported thick


def _near_occupied(lay: navgrid.Layers, r: float = 2.0) -> int:
    """Occupied cells within ``r`` of the origin (the room's walls lie 2.4 m away or more)."""
    R, C = lay.shape
    yy, xx = np.mgrid[0:R, 0:C]
    d = np.hypot(lay.origin[0] + (xx + 0.5) * lay.cell, lay.origin[1] + (yy + 0.5) * lay.cell)
    return int(((lay.state == navgrid.OCCUPIED) & (d <= r)).sum())


def _floor_volume(dz: float, floor_z: float) -> navgrid.Layers:
    """A flat floor at ``dz`` (ray-cast) seen by a camera turning in place at 0.34 m, the grid's
    floor at ``floor_z`` with a 0.01 m sigma."""
    room = Room(half=(2.6, 2.4), boxes=[])
    vol = navgrid.Volume()
    for yaw in range(0, 360, 30):
        T = look(float(yaw), (0.0, 0.0, 0.34 - dz), -20.0)
        _, z = render(room, T, K, supersample=1)
        T[2, 3] += dz  # the floor seen dz higher
        vol.integrate(f"y{yaw}", z, K.K(), T)
    return navgrid.layers(vol, floor_z, 0.05, 1.8, 0.01)


def test_the_floor_band_tests_the_surface_height_not_the_voxel_centre() -> None:
    """map.md §6 Floor band (P26, R5): a floor surface 0.055 m up lies inside the 0.06 m band
    wherever the floor falls on the voxel lattice (before: the voxel centred at 0.075 m made 300+
    obstacles); one 0.07 m up lies outside it."""
    assert _near_occupied(_floor_volume(0.055, 0.0)) == 0
    for f in (0.02, -0.02):
        assert _near_occupied(_floor_volume(0.055 + f, f)) == 0
    assert _near_occupied(_floor_volume(0.075, 0.0)) > 100


def test_surface_heights_survive_storage_and_older_volumes_use_centres() -> None:
    room = Room(half=(2.6, 2.4), boxes=[])
    T = look(0.0, (0.0, 0.0, 0.34), -20.0)
    _, z = render(room, T, K, supersample=1)
    v = navgrid.Volume()
    v.integrate("a", z, K.K(), T)
    w = navgrid.Volume.from_bytes(v.to_bytes())
    assert w.hit_dz is not None and np.allclose(w.surface_z(), v.surface_z())
    floor = v.surface_z()[(v.hit > 0) & (v.flat.astype(int) * 2 >= v.hit)]
    assert np.abs(floor[np.abs(floor) < 0.05]).max() < 0.012  # the hits' height: the floor's
    # a volume of a version derived before oms/2.4: its voxel centres stand in
    import io

    data = dict(np.load(io.BytesIO(v.to_bytes())))
    del data["hit_dz"]
    buf = io.BytesIO()
    np.savez(buf, **data)
    old = navgrid.Volume.from_bytes(buf.getvalue())
    assert old.hit_dz is None
    zc = old.origin[2] + (np.arange(old.shape[0]) + 0.5) * old.vz
    assert np.allclose(old.surface_z()[:, 0, 0], zc)
    old.integrate("b", z, K.K(), look(90.0, (0.0, 0.0, 0.34), -20.0))  # grows, no heights
    assert old.hit_dz is None
    # a hand-made volume whose hits were set without heights: their voxels' centres
    hand = navgrid.Volume()
    hand.ensure(np.zeros(3), np.ones(3))
    hand.hit_dz = np.zeros((1, 1, 1), np.float32)
    assert hand.ensure(np.array([-2.0, 0, 0]), np.ones(3))  # a mismatched one is not padded
    assert hand.surface_z().shape == hand.hit.shape


# -- the floor check -------------------------------------------------------------------------------


def _floor_points(h: float, n: int = 64) -> tuple[np.ndarray, Intrinsics, np.ndarray]:
    """A depth grid of a level floor ``h`` below a camera pitched 30 deg down."""
    T = look(0.0, (0.0, 0.0, h), -30.0)
    room = Room(half=(20.0, 20.0), boxes=[])
    Kg = K.resized(n, n * 3 // 4)
    _, z = render(room, T, Kg, supersample=1)
    return z, Kg, T


def test_floor_verdicts() -> None:
    z, Kg, T = _floor_points(0.5)
    # a depth image: within its bound, or out by 0.1 m
    assert an.floor_verdict("sensor", 0.5, 0.01, T, 0.01, z, Kg) == an.FloorCheck()
    assert not an.floor_verdict("sensor", 0.5, 0.01, T, 0.01, z * 1.2, Kg).ok
    assert an.floor_verdict("sensor", 0.5, None, T, 0.01, z * 1.06, Kg).ok  # default sigma
    # a monocular depth: its own scale, H / h_f, or out beyond the bound
    got = an.floor_verdict("mono", 0.5, 0.01, T, 0.05, z * 1.2, Kg)
    assert got.ok and got.factor == pytest.approx(1 / 1.2, rel=0.02)
    # beyond the bound: not judged, kept at k (oms/2.5, P29a)
    assert an.floor_verdict("mono", 0.5, 0.01, T, 0.05, z * 2.0, Kg) == an.FloorCheck(judged=False)
    assert not an.floor_verdict("mono", 0.5, 0.01, T, 0.05, z * 1.2, Kg, rel=1.6).judged
    # not checked: no height, no depth, no plane
    assert an.floor_verdict("mono", None, 0.01, T, 0.05, z, Kg) == an.FloorCheck()
    assert an.floor_verdict("none", 0.5, 0.01, T, 0.05, z, Kg) == an.FloorCheck()
    assert an.floor_verdict("sensor", 0.5, 0.01, T, 0.05, np.zeros_like(z), Kg) == an.FloorCheck()
    wall = np.full_like(z, 2.0)  # a frontal wall: no floor plane below
    assert an.floor_verdict("sensor", 0.5, 0.01, T, 0.05, wall, Kg) == an.FloorCheck()
    assert an.own_floor(z, Kg, T[:3, :3], 0.5) == pytest.approx(0.5, abs=0.01)


def _cloud_floor(m: AnchoredMap) -> np.ndarray:
    """The cloud's heights within 1 m of the turning camera (nothing stands there) and 0.3 m of
    the floor."""
    pts = m.cloud.points()
    near = (np.hypot(pts[:, 0], pts[:, 1]) < 1.0) & (np.abs(pts[:, 2]) < 0.3)
    return pts[near][:, 2]


def test_a_monocular_frame_with_a_wrong_scale_gives_no_content(tmp_path: Path) -> None:
    """Diagnosis test 2 (F1): a 360 deg monocular sweep with two adjacent frames whose depth is
    half the others' (as look008/009 were): they stay registered and, since oms/2.5 (P29a), are
    not judged: they keep k and their pose and are not listed, and, since P29b, give no content;
    the others take their own scale."""
    m = AnchoredMap(tmp_path / "m")
    m.update([frame("a0", base_pose(0), depth=None, mono=1.3)])
    half = {"s5", "s6"}
    pls = m.update([frame(f"s{k}", base_pose(30.0 * k + 15.0), depth=None,
                          mono=0.65 if f"s{k}" in half else 1.3) for k in range(12)])
    assert all(p.status == "registered" for p in pls)
    assert m.floor_inconsistent == []
    kf = {k.name: k for k in m.keyframes}
    for n in half:
        assert kf[n].floor_ok and kf[n].floor_unjudged and not kf[n].contributes()
        assert kf[n].floor_withheld() and kf[n].depth_scale is None
    kept = [k for k in m.keyframes if k.name not in half and k.depth_scale is not None]
    assert len(kept) >= 6
    for k in kept:  # the own scale brings the raw depth (1.3 x true) onto metres
        assert k.depth_scale == pytest.approx(1 / 1.3, rel=0.05)
    floor, sigma = m.floor()
    assert floor is not None and abs(floor) < 0.03 and sigma <= 0.02


def test_a_sensor_frame_whose_depth_disagrees_with_its_height(tmp_path: Path) -> None:
    """Diagnosis test 3 (F1, depth images): a depth image scaled 1.3 at a correct pose is left
    out of content; its status, pose and sigmas stay."""
    m = AnchoredMap(tmp_path / "m")
    pl = m.update([frame("a0", base_pose(0), hs=0.01, depth=1.3)])[0]
    assert pl.status == "registered" and m.floor_inconsistent == ["a0"]
    k = m.keyframe(pl.keyframe or "")
    assert k is not None and not k.floor_ok and not k.contributes()
    assert k.T[2, 3] == pytest.approx(H, abs=1e-6) and len(m.cloud.keys) == 0
    # within its bound: kept
    m2 = AnchoredMap(tmp_path / "m2")
    m2.update([frame("a0", base_pose(0), hs=0.01, depth=1.02)])
    assert m2.floor_inconsistent == [] and len(m2.cloud.keys) > 0


def test_the_floor_check_on_reregistered_and_unseeded_keyframes(tmp_path: Path) -> None:
    """A stored keyframe a job confirms is checked at its confirmed pose (``_recheck_floor``),
    a monocular one taking its own scale for good (``floor_factor``)."""
    m = AnchoredMap(tmp_path / "m")
    m.update([frame("a0", base_pose(0), depth=None, mono=1.3)])
    m.update([frame("b1", base_pose(30.0), depth=None, mono=1.3)])
    k = m.keyframe(m.keyframes[1].id)
    assert k is not None and k.depth_scale is not None
    # as stored before oms/2.4: no own scale, no grid scale
    k.floor_factor = k.grid_scale = None
    m.floor_inconsistent = []
    m._recheck_floor(k)
    assert k.floor_factor is not None and k.depth_scale == pytest.approx(1 / 1.3, rel=0.05)
    # its own scale is its content's only (P26-2): registration keeps k
    assert m._depth_factor(k, content=True) == k.floor_factor
    assert m._depth_factor(k) == pytest.approx(m.scale.k / k.k_used)
    # its own floor's scale far from the map's now: not judged, back at k (oms/2.5)
    m.scale.k *= 2.0
    m._recheck_floor(k)
    assert k.floor_ok and k.floor_unjudged and k.floor_factor is None
    assert m.floor_inconsistent == [] and k.floor_withheld() and not k.contributes()  # P29b
    # a depth image far from its height: out, listed once
    k.depth_kind = "sensor"
    assert k.grid is not None
    k.grid = (k.grid * 1.5).astype(np.float32)
    m._recheck_floor(k)
    m._recheck_floor(k)
    assert not k.floor_ok and m.floor_inconsistent == [k.name]
    k.depth_kind = "mono"
    # without a depth grid: nothing to check
    k.grid, k.K_grid = None, None
    m._recheck_floor(k)
    # stored and loaded again: the fields survive
    d = Keyframe.from_json(json.loads(json.dumps(m.keyframes[0].to_json())))
    assert d.floor_ok and d.grid_scale == m.keyframes[0].grid_scale
    assert an.FloorCheck() == an.FloorCheck(True, 1.0, None)
    assert AnchoredMap(tmp_path / "x").floor_check("mono", H, 0.01, base_pose(0), 0.01,
                                                   np.zeros((3, 4), np.float32), K, 1.0).ok


# -- the sent height and the tilt sigma in registration --------------------------------------------


def _warped(z: np.ndarray, s: float) -> np.ndarray:
    """Monocular depth: a scale and a smooth warp (relative depth bends planes)."""
    v, u = np.mgrid[0:K.height, 0:K.width]
    return (z * s * (1.0 + 0.04 * (v / K.height - 0.5) + 0.02 * np.sin(u / K.width * math.pi))
            ).astype(np.float32)


def _drifting_sweep(m: AnchoredMap, hs: float | None, st: float | None) -> list[Any]:
    """Diagnosis test 4: 2 x 12 monocular frames turning in place with warped depth, drifting
    priors with exact z and pitch whose rotation sigma grows from 1 to 15 deg."""
    rng = np.random.default_rng(3)
    out = []
    m.update([frame("a0", base_pose(0), depth=None, mono=1.0, hs=hs)])
    for k in range(24):
        T = base_pose(15.0 * k + (7.0 if k >= 12 else 0.0))
        img, z = shot(T)
        d = _warped(z, 1.0 + rng.normal(0, 0.05))
        p = prior(T, "drifting", 0.02 + 0.001 * k, 1.0 + 14.0 * k / 23, st)
        out.append(FrameIn(f"d{k:02d}", "cam", img, K, p, None, None, H, hs, 1000,
                           lambda d=d: (d, K, None, np.array([0.0, -1.0, 0.0]))))
    pls = []
    for i in range(0, 24, 4):
        pls += m.update(out[i:i + 4])
    return pls


def test_drifting_priors_with_exact_height_and_pitch(tmp_path: Path) -> None:
    """Diagnosis test 4 (F2a/b, P26): with the sent height and a tilt sigma, every committed
    camera height stays within 3 height sigmas and its pitch within 1 deg, the reported sigmas
    containing the errors."""
    m = AnchoredMap(tmp_path / "m")
    pls = _drifting_sweep(m, 0.01, 0.5)
    reg = [p for p in pls if p.status == "registered"]
    assert len(reg) >= 20
    for p in reg:
        G = base_pose(15.0 * int(p.name[1:]) + (7.0 if int(p.name[1:]) >= 12 else 0.0))
        assert abs(p.T[2, 3] - H) <= 0.03, p.name
        assert abs(pitch_of(p.T) - pitch_of(G)) <= 1.0, p.name
        assert np.linalg.norm(p.T[:3, 3] - G[:3, 3]) <= 3.0 * p.sigma_pos_m + 0.01, p.name
        assert p.height_term


def test_the_height_gate_tests_the_image_only_pose(tmp_path: Path) -> None:
    """Diagnosis test 5 (map.md §2): a frame whose image-only z is 0.10 m off stays listed in
    ``height_inconsistent`` though the height pulls its committed pose near the sent one."""
    m = AnchoredMap(tmp_path / "m")
    m.update([frame("a0", base_pose(0), hs=0.01)])
    pl = m.update([frame("b1", base_pose(30.0), hs=0.01, height=H + 0.10,
                         p=prior(base_pose(30.0), sp=0.2))])[0]
    assert pl.status == "registered" and pl.height_term
    assert "b1" in m.height_inconsistent
    assert pl.align_T is not None and abs(pl.align_T[2, 3] - H) < 0.02  # the image says 0.5
    assert pl.T is not None and pl.T[2, 3] > H + 0.02  # the height pulled it up
    # without a height sigma, no height term: the gate on the committed pose
    m2 = AnchoredMap(tmp_path / "m2")
    m2.update([frame("a0", base_pose(0))])
    p2 = m2.update([frame("b1", base_pose(30.0))])[0]
    assert not p2.height_term and m2.height_inconsistent == []


EMPTY = Room(half=(2.6, 2.4), boxes=[])


def test_the_sent_height_outweighs_a_wobbling_metric_prior_z(tmp_path: Path) -> None:
    """The diagnosis's run I (§0.3a): a stepping base's metric priors whose z wobbles with an SD
    of 2 cm (sigma_pos 0.02), the sent height exact (height_sigma 0.015), a camera at 0.3375 m
    turning in place. Before, the committed z followed the prior's (0.295-0.379 m, none gated);
    with the height a z measurement it stays within 2 cm of the floor plus the height."""
    rng = np.random.default_rng(5)
    Hs, hs = 0.3387, 0.015
    m = AnchoredMap(tmp_path / "m")
    frames, wob = [], []
    for k in range(24):
        yaw = 15.0 * k + (7.0 if k >= 12 else 0.0)
        a = math.radians(yaw)
        T = look(yaw, (0.15 * math.cos(a), 0.15 * math.sin(a), 0.3375), -20.0)
        img, z = render(EMPTY, T, K)
        P = T.copy()
        P[2, 3] = Hs + (rng.normal(0.0, 0.02) if k else 0.0)
        wob.append(P[2, 3])
        frames.append(FrameIn(f"w{k:02d}", "cam", img, K, prior(P, "metric", 0.02, 1.0), z,
                              None, Hs, hs, 1000))
    assert max(wob) - min(wob) > 0.05  # the priors' z spans several centimetres
    pls = m.update(frames[:1])
    for i in range(1, 24, 4):
        pls += m.update(frames[i:i + 4])
    reg = [p for p in pls if p.status == "registered" and p.method != "origin"]
    assert len(reg) >= 18
    for p in reg:
        assert p.height_term and abs(p.T[2, 3] - (m.floor_z or 0.0) - Hs) <= 0.02, (
            p.name, p.T[2, 3])


def _tilted_map(tmp_path: Path, st: float | None) -> tuple[AnchoredMap, list[Any]]:
    """A map whose stored keyframes sit 2 deg off level (an earlier chain's drift), then
    drifting frames with exact gravity, a 10 deg rotation sigma and tilt sigma ``st``."""
    m = AnchoredMap(tmp_path / "m")
    tilt = an.exp_so3(np.array([0.0, math.radians(2.0), 0.0]))
    first = []
    for k in range(4):
        T = base_pose(30.0 * k)
        B = T.copy()
        B[:3, :3] = tilt @ T[:3, :3]
        img, z = shot(T)
        first.append(FrameIn(f"a{k}", "cam", img, K, prior(B, sp=0.01, sr=0.05), z, None, None,
                             None, 1000))
    m.update(first[:1])
    m.update(first[1:])
    pls = m.update([frame(f"b{k}", base_pose(30.0 * k + 15.0),
                          p=prior(base_pose(30.0 * k + 15.0), "drifting", 0.05, 10.0, st),
                          hs=0.01) for k in range(4)])
    return m, pls


def test_a_tilt_sigma_holds_pitch_and_roll(tmp_path: Path) -> None:
    """map.md §1 "Prior rotation sigma": pitch and roll by sigma_tilt_deg, yaw by sigma_rot_deg
    in the refinement; the reported rotation sigma stays one (RMS) sigma."""
    _, pls = _tilted_map(tmp_path / "with", 0.3)
    _, base = _tilted_map(tmp_path / "without", None)
    for p, q in zip(pls, base, strict=True):
        assert p.status == q.status == "registered"
        G = base_pose(30.0 * int(p.name[1:]) + 15.0)
        assert tilt_err(p.T, G) <= 0.5 < tilt_err(q.T, G), p.name
        assert p.sigma_rot_deg < q.sigma_rot_deg


def test_a_tilt_beyond_its_sigma_conflicts(tmp_path: Path) -> None:
    """The conflict bound per part (api.md §4, map.md §1): a prior 6 deg off level with a
    0.3 deg tilt sigma conflicts though its 10 deg rotation sigma would hold it; its kept
    sigma is the RMS of the axes."""
    m = AnchoredMap(tmp_path / "m")
    m.update([frame("a0", base_pose(0))])
    m.update([frame("a1", base_pose(30.0))])
    T = base_pose(15.0)
    P = T.copy()
    P[:3, :3] = an.exp_so3(np.array([math.radians(6.0), 0.0, 0.0])) @ T[:3, :3]
    pl = m.update([frame("c", T, p=prior(P, "metric", 0.02, 10.0, 0.3))])[0]
    assert (pl.status, pl.reason) in (("rejected", "prior_conflict"), ("kept_at_prior",
                                                                        "no_overlap"))
    m2 = AnchoredMap(tmp_path / "m2")
    m2.update([frame("a0", base_pose(0))])
    m2.update([frame("a1", base_pose(30.0))])
    assert m2.update([frame("c", T, p=prior(P, "metric", 0.02, 10.0))])[0].status == \
        "registered"
    kept = m2.kept_at(frame("k", T, p=prior(P, "metric", 0.02, 10.0, 0.0)), "no_overlap", 0)
    assert kept.sigma_rot_deg == pytest.approx(10.0 / math.sqrt(3.0), abs=0.01)


def test_the_drift_estimate_keeps_a_drifting_priors_gravity() -> None:
    """map.md §1: a drifting prior predicted through a correction that tilts keeps its own pitch
    and roll by its tilt sigma (``Drift.predict``)."""
    ref = base_pose(0.0)
    tilted = ref.copy()
    tilted[:3, :3] = an.exp_so3(np.array([math.radians(2.0), 0.0, math.radians(5.0)])) @ ref[:3, :3]
    d = an.Drift(tilted, ref, 1.0)
    P = base_pose(40.0)
    T0, _, _ = d.predict(P, 0.05, 10.0)
    T1, _, sr = d.predict(P, 0.05, 10.0, 0.3)
    assert tilt_err(T0, P) > 1.5 and tilt_err(T1, P) < 0.1
    assert sigmas.yaw_tilt(T1[:3, :3], T0[:3, :3])[0] < 0.2  # its yaw is the correction's
    assert sr == pytest.approx(10.0, abs=0.01)


# -- a sensor-depth client: standpoints, a doorway ------------------------------------------------


DOOR_ROOM = Room(half=(3.0, 2.5), boxes=[
    ((0.8, -2.5, 0.0), (0.9, -0.45, 2.6)),  # a partition at x 0.8-0.9 with a doorway
    ((0.8, 0.45, 0.0), (0.9, 2.5, 2.6)),  # ... 0.9 m wide at y -0.45 to 0.45
    ((-2.2, -1.8, 0.0), (-1.4, -1.2, 1.2)),
    ((-0.5, 1.7, 0.0), (0.4, 2.5, 0.75)),
    ((2.0, -1.0, 0.0), (2.6, 1.2, 0.9))])
ROOMS = {"room": ROOM, "door": DOOR_ROOM}


def _misregistered(z: np.ndarray, deg: float) -> np.ndarray:
    """A depth image whose ray directions are turned ``deg`` in pitch against its colour (depth
    and colour misregistered): the floor it lifts tilts, the chain drifts."""
    v = np.mgrid[0:K.height, 0:K.width][0]
    return (z * (1.0 + math.tan(math.radians(deg)) * (v + 0.5 - K.cy) / K.fy)).astype(
        np.float32)


def _standpoints(m: AnchoredMap, hs: float | None, st: float | None
                 ) -> tuple[list[Any], dict[str, np.ndarray]]:
    """A camera at 0.50 m with depth turning at three standpoints of the room before a doorway
    (drifting priors: exact z and gravity, a rotation sigma that grows), its depth misregistered
    by 2 deg so the image chain drifts in height and pitch."""
    pts = [(-1.6, 0.3), (-0.6, -0.9), (-0.9, 1.1)]
    out, k, truth = [], 0, {}
    for j, (x, y) in enumerate(pts):
        for yaw in range(0, 360, 30):
            T = base_pose(float(yaw + 10 * j), x, y, 0.5, pitch_deg=-15.0)
            img, z = shot(T, "door")
            p = prior(T, "drifting" if k else "metric", 0.03 + 0.002 * k, 1.0 + 0.3 * k, st)
            out.append(FrameIn(f"p{k:02d}", "cam", img, K, p, _misregistered(z, 2.0), None,
                               0.5, hs, 1000))
            truth[f"p{k:02d}"] = T
            k += 1
    pls = m.update(out[:1])
    for i in range(1, len(out), 6):
        pls += m.update(out[i:i + 6])
    return pls, truth


def _grid(m: AnchoredMap) -> navgrid.Layers:
    """The version's grid as ``service.derived`` builds it: the contributing keyframes' depth."""
    vol = navgrid.Volume()
    for kf in m.keyframes:
        if not kf.contributes():
            continue
        ref = m._grid(kf)
        assert ref is not None
        d = ref.depth
        vol.integrate(kf.name, d, kf.K.resized(d.shape[1], d.shape[0]).K(), kf.T,
                      an.valid_depth(d))
    floor, sigma = m.floor()
    return navgrid.layers(vol, floor or 0.0, 0.05, 1.8, sigma)


def _passage_occupied(lay: navgrid.Layers) -> float:
    """The share of occupied cells in the doorway and the floor just beyond it."""
    cells = [lay.cell_of((x, y)) for x in np.arange(0.82, 1.6, 0.05)
             for y in np.arange(-0.35, 0.36, 0.05)]
    st = [lay.state[r, c] for r, c in cells if 0 <= r < lay.shape[0] and 0 <= c < lay.shape[1]]
    return float(np.mean([s == navgrid.OCCUPIED for s in st]))


def test_a_sensor_depth_client_keeps_its_height_and_level_and_the_doorway_free(
        tmp_path: Path) -> None:
    """The reporter's sensor-depth client (P26): a camera at 0.50 m with depth, three
    standpoints, a chain that drifts in height and pitch. With the sent height and a tilt sigma
    every committed height stays within 2 cm of the sent one, its level within 0.5 deg, and the
    doorway's floor stays free."""
    m = AnchoredMap(tmp_path / "m")
    pls, truth = _standpoints(m, 0.01, 0.3)
    reg = [p for p in pls if p.status == "registered" and p.method != "origin"]
    assert len(reg) >= 25
    for p in reg:
        assert abs(p.T[2, 3] - 0.5) <= 0.02, (p.name, p.T[2, 3])
        assert tilt_err(p.T, truth[p.name]) <= 0.5, p.name
    assert _passage_occupied(_grid(m)) == 0.0


# -- the service: rows, quality and input statuses -------------------------------------------------


def test_floor_inconsistent_rows_feed_no_occupancy() -> None:
    assert _occupies({"pose_state": "fused"})
    assert not _occupies({"pose_state": "fused", "floor_inconsistent": True})
    assert not _occupies({"pose_state": "fused", "height_inconsistent": True})


def test_the_service_reports_the_floor_check(client: Client) -> None:  # noqa: F811
    """api.md §4 (oms/2.4): a monocular frame the floor check rescales reports ``depth_scale``;
    one beyond the monocular bound is, since oms/2.5, not judged: in no ``floor_inconsistent``
    (a dry run's neither), ``detail`` ``floor_not_judged``, its version row without content
    (P29b)."""
    cam = {**CAM, "height_above_floor_m": 0.5, "height_sigma_m": 0.01}
    fr, files = frames_for(client, [base_pose(0)], tag="a", depth=False)
    assert client.update("m", fr, files, cameras={"cam": cam})["state"] == "committed"
    poses = [base_pose(15.0 + 30.0 * k) for k in range(5)]
    fr, files = frames_for(client, poses, tag="s", depth=False)
    key = hashlib.sha256(files["s2"]).hexdigest()
    client.backend.mono_depth[key] = client.backend.mono_depth[key] * 0.5
    d = dry(client, "m", fr, files)
    assert d["quality"]["floor_inconsistent"] == []
    job = client.update("m", fr, files)
    assert job["state"] == "committed"
    assert job["quality"]["floor_inconsistent"] == []
    rows = {i["name"]: i for i in job["inputs"]}
    assert "depth_scale" not in rows["s2"] and rows["s2"]["status"] == "registered"
    assert rows["s2"]["detail"] == "floor_not_judged"
    assert all(rows[f"s{k}"].get("depth_scale", 0) > 0 for k in (0, 1, 3, 4))
    rec = {f["name"]: f for f in client.service.store.version("m")["frames"]}
    assert not rec["s2"]["content"] and rec["s2"]["floor_inconsistent"]  # withheld, not listed
    assert all(rec[f"s{k}"]["content"] for k in (0, 1, 3, 4))


# -- the older mapper ------------------------------------------------------------------------------


def test_the_older_mapper_checks_the_floor_on_its_grid() -> None:
    """``Updater._floor_off``: the mapper's depth at the committed pose, by the same rule."""
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

    internal = type("M", (), {})()
    internal.depth = lambda name: z * 2.0
    fr = Fr()
    # beyond the monocular bound: not judged (oms/2.5)
    assert not Updater._floor_off(fr, internal, anchor.Sim3())  # type: ignore[arg-type]
    internal.depth = lambda name: z * 1.2
    assert not Updater._floor_off(fr, internal, anchor.Sim3())  # type: ignore[arg-type]
    fr.depth_m = z  # a depth image: the sensor bound
    assert Updater._floor_off(fr, internal, anchor.Sim3())  # type: ignore[arg-type]
    internal.depth = lambda name: z
    assert not Updater._floor_off(fr, internal, anchor.Sim3())  # type: ignore[arg-type]
    internal.depth = lambda name: None
    assert not Updater._floor_off(fr, internal, anchor.Sim3())  # type: ignore[arg-type]


# -- a phantom floor on monocular keyframes (the client's task_map, synthetic) ----------------------


CAM_H = 0.339  # the client's camera height
EMPTY = Room(boxes=[])


def _phantom_frames(lifts: tuple[float, ...] = (0.06, 0.065, 0.07, 0.075)
                   ) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """A turn in place of monocular keyframes in the empty room, camera 0.339 m high: every
    other frame's depth is scaled so that its own floor lies ``lifts`` high (0.06-0.075 m: a
    monocular scale 18-22 % too small, within the monocular bound; the client's 0.15-0.22 m,
    35-56 %, lie beyond it), the others at the true scale."""
    out = []
    it = iter(lifts)
    for k in range(8):
        T = base_pose(45.0 * k, 0.3, -0.2, CAM_H, 0.05, -20.0)
        _, z = render(EMPTY, T, K)
        scale = 1.0 if k % 2 == 0 else (CAM_H - next(it)) / CAM_H
        out.append((T, (z * scale).astype(np.float32), "mono"))
    return out


def _grid_of(frames: list[tuple[np.ndarray, np.ndarray, str]], check: bool, floor: float = 0.0,
             floor_sigma: float = 0.02) -> tuple[int, float, float]:
    """The grid of a version rebuilt from its keyframes' depth as ``service.derived`` does (half
    resolution), with or without the floor check (``floor_verdict``: the sent camera height,
    each frame's own floor): the occupied cells within 1 m of the cameras, and the SD and
    robust SD (1.4826 MAD) of the volume's flat points' heights within 0.25 m of the floor."""
    vol = navgrid.Volume()
    cams = []
    for k, (T, d, kind) in enumerate(frames):
        cams.append(T[:3, 3])
        if check:
            chk = an.floor_verdict(kind, CAM_H, None, T, 0.02, d, K)
            if not chk.ok or not chk.judged:  # excluded, or not judged: no content (P29b)
                continue
            d = (d * chk.factor).astype(np.float32)
        d = d[1::2, 1::2]
        Kh = K.resized(d.shape[1], d.shape[0])
        vol.integrate(f"k{k}", d, Kh.K(), T, an.valid_depth(d))
    C = np.array(cams)
    vol.ensure(C.min(0) - [0, 0, 3.0], C.max(0) + [0, 0, 2.0])
    zc = vol.origin[2] + (np.arange(len(vol.flat_z)) + 0.5) * navgrid.FLOOR_BIN_M
    sel = np.abs(zc - floor) < 0.25
    w, z = vol.flat_z[sel], zc[sel]
    mean = float((w * z).sum() / w.sum())
    sd = math.sqrt(float((w * (z - mean) ** 2).sum() / w.sum()))
    pts = np.repeat(z, np.round(w).astype(np.int64))
    robust = 1.4826 * float(np.median(np.abs(pts - np.median(pts))))
    lay = navgrid.layers(vol, floor, 0.05, 1.8, floor_sigma)
    R, Cc = lay.shape
    yy, xx = np.mgrid[0:R, 0:Cc]
    cx = lay.origin[0] + (xx + 0.5) * lay.cell
    cy = lay.origin[1] + (yy + 0.5) * lay.cell
    c0 = C[:, :2].mean(0)
    near = np.hypot(cx - c0[0], cy - c0[1]) <= 1.0
    return int(((lay.state == navgrid.OCCUPIED) & near).sum()), sd, robust


def test_monocular_keyframes_with_a_raised_floor_lose_their_phantom_floor() -> None:
    """Regression of the client's ``task_map`` (monocular keyframes, camera 0.339 m), on a
    synthetic copy: monocular frames whose own floor lies 0.06-0.075 m high make many occupied
    cells within 1 m of the cameras on the empty floor; under the floor check (map.md §2,
    oms/2.4: each takes its own scale from the sent height) about none, and the SD of the
    floor's heights about halves. The client's 0.15-0.22 m (an own scale beyond the monocular
    bound) are not judged (oms/2.5, P29a) and, since P29b, give no content: their phantom floor
    is gone too (no occupied cell; the frames at the true scale remain)."""
    frames = _phantom_frames()
    before, sd0, r0 = _grid_of(frames, False)
    after, sd1, r1 = _grid_of(frames, True)
    assert before >= 100  # 182 here
    assert after <= 10  # 0
    assert sd1 <= 0.6 * sd0 and r1 <= r0  # 0.033 m -> 0.0 m
    far = _phantom_frames((0.15, 0.17, 0.20, 0.22))
    far_before, _, _ = _grid_of(far, False)
    far_after, _, _ = _grid_of(far, True)
    assert far_before >= 100  # the phantom floor of the raised frames, unchecked
    assert far_after == 0  # not judged, no content: gone


@pytest.fixture
def old(tmp_path: Path) -> Iterator[Client]:
    """The older mapper's path for every map (``Updater(prior_first=False)``)."""
    from starlette.testclient import TestClient

    from oh_my_slam.service.app import Service, create_app
    from oh_my_slam.service.jobs import Jobs
    from oh_my_slam.service.store import Store
    from oh_my_slam.service.timings import Timings
    from oh_my_slam.service.update import Updater
    from tests.fakes.service import FakeBackend, FakeInference

    store = Store(tmp_path / "data")
    store.create()
    backend = FakeBackend()
    timings = Timings()
    updater = Updater(store, backend, timings, prior_first=False)
    jobs = Jobs(store.jobs, updater, timings.predict_update)
    jobs.start()
    service = Service(store, jobs, FakeInference(), backend.require_inference, timings=timings,
                      version="test", dry_run=updater.dry_run)
    with TestClient(create_app(service)) as http:
        yield Client(http, backend, service)
    jobs.stop(timeout=10)


def test_the_older_mapper_lists_its_floor_inconsistent_frames(
        old: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    """The mapper's engine: a frame the floor check fails is listed and its version row feeds
    no grid evidence."""
    from oh_my_slam.service.update import Updater
    from tests.fakes.service import CAMERA, pose, textured

    monkeypatch.setattr(Updater, "_floor_off",
                        staticmethod(lambda fr, internal, sim: fr.work.name == "t2"))
    frames, files = [], {}
    for i, a in enumerate((0, 30, 60, 90)):
        T = pose(a, 0.0, 0.0, 1.2)
        files[f"t{i}"] = old.backend.add(textured(700 + i), T)
        frames.append({"name": f"t{i}", "image": f"t{i}", "camera_id": "cam",
                       "prior": prior(T), "height_above_floor_m": 1.2,
                       "height_sigma_m": 0.01})
    job = old.update("m", frames, files, {"cam": CAMERA})
    assert job["quality"]["floor_inconsistent"] == ["t2"]
    rows = {r["name"]: r for r in old.service.store.version("m")["frames"]}
    assert rows["t2"]["floor_inconsistent"] and not _occupies(rows["t2"])


def test_a_kept_keyframe_placed_again_keeps_its_tilt_sigma(tmp_path: Path) -> None:
    """A stored kept keyframe's prior keeps its ``sigma_tilt_deg``: placed again
    (re-registration), its tilt is weighed by it as when it was sent."""
    m = AnchoredMap(tmp_path / "m")
    m.update([frame("a0", base_pose(0))])
    m.update([frame("a1", base_pose(30.0))])
    T = base_pose(15.0)
    w = frame("w", T, p=prior(T, "metric", 0.02, 10.0, 0.5))
    assert w.depth is not None
    kept = m.add(w, an.features(w.rgb), an.Placement("w", "kept_at_prior", "no_overlap", T,
                                                     0.02, 1.0), w.depth, K, "sensor", None)
    assert kept.prior is not None and kept.prior["sigma_tilt_deg"] == 0.5
    m.save()
    got = m._place_again(kept)
    assert got is not None and got[0].status == "registered"
    assert got[0].T is not None and tilt_err(got[0].T, T) < 0.3


def test_a_frame_without_prior_takes_the_sent_height(tmp_path: Path) -> None:
    """map.md §2: the sent height is a z measurement of a frame without a prior too (its image
    pose alone), on a camera-only map whose first frame had a camera height."""
    m = AnchoredMap(tmp_path / "m")

    def bare(name: str, T: np.ndarray, h: float) -> FrameIn:
        img, z = shot(T)
        return FrameIn(name, "cam", img, K, None, z, None, h, 0.005, 1000)

    m.update([bare("a0", base_pose(0.0), H)])
    assert m.created_with == "none" and m.origin_height_m == H
    pl = m.update([bare("b1", base_pose(30.0), H + 0.03)])[0]
    assert pl.status == "registered" and pl.height_term
    assert pl.T is not None and pl.align_T is not None and pl.T[2, 3] > pl.align_T[2, 3] + 0.01
    assert pl.image_sigmas is not None and pl.map_residual is not None
    assert pl.map_residual[0] > 0.01  # the height moved it: its displacement counts it
