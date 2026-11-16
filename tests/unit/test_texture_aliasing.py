"""oms/2.13: the texture-aliasing guards of registration (depth consistency, floor points), the
floor rule of locate and the one-view drift rule (map.md §1-§2, api.md §4, §6)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oh_my_slam.core.types import Intrinsics
from oh_my_slam.mapping import anchored as an
from oh_my_slam.mapping import relocalize as rl
from oh_my_slam.mapping.anchored import AnchoredMap, FrameIn, Placement
from oh_my_slam.mapping.quality import drift_correction
from oh_my_slam.service import nav
from oh_my_slam.service.prior_first import input_row, sent_residual
from oh_my_slam.service.schema import validate

K = Intrinsics(300.0, 300.0, 320.0, 240.0, 640, 480)
TILE = np.array([0.6, 0.0, 0.0])  # the floor texture repeats every 0.6 m along x
DATA = Path(__file__).resolve().parents[1] / "data" / "aliasing"


def cam(c: tuple[float, float, float], yaw_deg: float = 0.0, pitch_deg: float = 25.0
        ) -> np.ndarray:
    """A camera at ``c`` looking along yaw, pitched down (optical frame: x right, y down)."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    z = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), -math.sin(p)])
    x = np.array([math.sin(y), -math.cos(y), 0.0])
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    T[:3, 3] = c
    return T


A = cam((0.0, 0.0, 0.5))


def project(T: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pc = (X - T[:3, 3]) @ T[:3, :3]
    return pc[:, :2] / pc[:, 2:3] * [K.fx, K.fy] + [K.cx, K.cy], pc[:, 2]


def scene(rng: np.random.Generator, floor: int = 200, wall: int = 40
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A tiled floor and a wall seen from ``A``: the pixels, the map points they are matched to
    (each floor pixel to the same texture one tile further: a pose 0.6 m ahead explains them all)
    and which are floor."""
    F = np.column_stack([rng.uniform(1.0, 2.5, floor), rng.uniform(-0.6, 0.6, floor),
                         np.zeros(floor)])
    W = np.column_stack([np.full(wall, 3.0), rng.uniform(-0.8, 0.8, wall),
                         rng.uniform(0.2, 0.9, wall)])
    X = np.vstack([F, W])
    uv, z = project(A, X)
    ok = (z > 0) & (uv[:, 0] > 0) & (uv[:, 0] < K.width) & (uv[:, 1] > 0) & (uv[:, 1] < K.height)
    is_floor = np.arange(len(X)) < floor
    matched = np.where(is_floor[:, None], X + TILE, X)
    return uv[ok], matched[ok], is_floor[ok]


def ref_keyframe(c: tuple[float, float, float] = (-0.3, 0.0, 0.5)) -> an.Keyframe:
    return an.Keyframe("k000000", "r", "cam", K, cam(c), 0.01, 0.1, "fused", None, "mono", 1.0,
                       1.5)


def mapper(tmp_path: Path, floor: bool = True) -> AnchoredMap:
    m = AnchoredMap(tmp_path / "m")
    m.scale.source, m.scale.sigma_rel = "camera_height", 0.05
    m.floor_z, m.floor_z_sigma = (0.0, 0.0) if floor else (None, 0.0)
    return m


def corr_of(uv: np.ndarray, X: np.ndarray, ref: an.Keyframe) -> Any:
    return an._Corr(uv.astype(np.float64), X, an.unit(X - ref.centre), np.zeros(len(uv), np.intp))


def image_pose(m: AnchoredMap, uv: np.ndarray, X: np.ndarray,
               own: tuple[np.ndarray, Intrinsics, str] | None = None) -> Any:
    ref = ref_keyframe()
    f = FrameIn("q", "cam", np.zeros((K.height, K.width, 3), np.uint8), K, own_depth=own)
    feats = an.Features(uv.astype(np.float32), np.zeros((len(uv), 128), np.uint8))
    return m._image_pose(f, feats, corr_of(uv, X, ref), [ref], None, 0.0)


# -- floor points ------------------------------------------------------------------------------


def test_floor_points_band() -> None:
    X = np.array([[0.0, 0.0, 0.05], [0.0, 0.0, 0.1], [np.nan, np.nan, np.nan], [0, 0, -0.2]])
    assert not an.floor_points(X, None).any()
    assert an.floor_points(X, (0.0, 0.0)).tolist() == [True, False, False, False]
    # a point's own height sigma widens its band: 3 x 0.07 = 0.21
    assert an.floor_points(X, (0.0, 0.0), 0.07).tolist() == [True, True, False, True]
    assert rl.floor_points(X[[0, 1]], (0.0, 0.0)).tolist() == [True, False]
    assert not rl.floor_points(X, None).any()


def test_aliased_floor_never_places_a_frame_alone(tmp_path: Path) -> None:
    """A tiled floor matched one tile off places the frame 0.6 m ahead on 200 inliers; with a map
    floor that fit has no off-floor inlier and the fit on the wall alone is the frame's pose."""
    uv, X, is_floor = scene(np.random.default_rng(1))
    got = image_pose(mapper(tmp_path, floor=False), uv, X)
    assert got is not None and got[2] is None  # no map floor: no floor share
    s = got[0][0]
    assert np.linalg.norm(s.c - (A[:3, 3] + TILE)) < 0.02  # the alias wins without the guard
    got = image_pose(mapper(tmp_path), uv, X)
    assert got is not None
    s, n, _, method = got[0][:4]
    assert method == "pnp" and np.linalg.norm(s.c - A[:3, 3]) < 0.02
    assert n >= an.MIN_INLIERS and got[2] == 0.0
    assert len(got[1].uv) == int((~is_floor).sum())  # fitted on the off-floor matches


def test_a_floor_only_fit_is_no_pose(tmp_path: Path) -> None:
    uv, X, _ = scene(np.random.default_rng(2), wall=0)
    assert image_pose(mapper(tmp_path, floor=False), uv, X) is not None
    assert image_pose(mapper(tmp_path), uv, X) is None


def test_a_floor_fit_may_confirm_its_prior_never_move_it(tmp_path: Path) -> None:
    """A fit on the floor alone within the prior's sigma and the images' own noise confirms the
    prior; one tile away (the alias) it is no pose (map.md §2, oms/2.13)."""
    uv, X, is_floor = scene(np.random.default_rng(2), wall=0)
    ref = ref_keyframe()
    f = FrameIn("q", "cam", np.zeros((K.height, K.width, 3), np.uint8), K,
                prior={"trust": "drifting", "T_map_cam": A.tolist(), "sigma_pos_m": 0.5,
                       "sigma_rot_deg": 10.0})
    feats = an.Features(uv.astype(np.float32), np.zeros((len(uv), 128), np.uint8))
    m = mapper(tmp_path)
    true = corr_of(uv, X - TILE, ref)  # the floor matched to its own texture
    got = m._image_pose(f, feats, true, [ref], A, 0.5)
    assert got is not None and np.linalg.norm(got[0][0].c - A[:3, 3]) < 0.02 and got[2] == 1.0
    assert m._image_pose(f, feats, corr_of(uv, X, ref), [ref], A, 0.5) is None  # the alias
    assert m._image_pose(f, feats, true, [ref], A, 0.0) is None  # a prior sigma of 0 confirms nothing


def test_a_correct_mixed_fit_keeps_its_floor_inliers(tmp_path: Path) -> None:
    """Floor points that agree with the off-floor ones support the pose: the share reports them."""
    uv, X, is_floor = scene(np.random.default_rng(3))
    X = np.where(is_floor[:, None], X - TILE, X)  # the floor matched to its own texture
    got = image_pose(mapper(tmp_path), uv, X)
    assert got is not None and np.linalg.norm(got[0][0].c - A[:3, 3]) < 0.02
    assert got[2] is not None and got[2] > 0.7


def test_the_mapper_floor_follows_a_references_own_scale(tmp_path: Path) -> None:
    """A monocular reference lifted at k twice its own floor-check scale puts its floor at
    c_z - 2 x its height: those points are floor points."""
    m = mapper(tmp_path)
    k = ref_keyframe()
    k.k_used, k.floor_factor = 0.5, 1.0  # lifted at k / k_used = 2 x its own scale
    assert m._floor_ratio(k) == pytest.approx(2.0)
    refs = m._refs([k])
    X = np.array([[1.0, 0.0, 0.5 - 2 * 0.5], [1.0, 0.0, 0.0], [1.0, 0.0, 0.3]])
    assert m._on_floor(X, np.zeros(3, np.intp), refs).tolist() == [True, False, False]
    sensor = ref_keyframe()
    sensor.depth_kind = "sensor"
    assert m._floor_ratio(sensor) == 1.0 and m._floor_ratio(ref_keyframe()) == 1.0
    assert not mapper(tmp_path, floor=False)._on_floor(X, np.zeros(3, np.intp), refs).any()
    plain = an.Refs(refs.c, refs.scale_sigma, refs.median)  # defaults: ratio 1, point floor
    assert m._on_floor(X, np.zeros(3, np.intp), plain).tolist() == [False, True, False]


# -- depth consistency -------------------------------------------------------------------------


def test_depth_consistency_drops_and_rejects() -> None:
    pred = np.full(40, 1.0)
    keep, whole = an.depth_consistency(pred.copy(), pred, np.full(40, 0.05))
    assert keep.all() and whole
    own = pred.copy()
    own[:5] = 3.0  # five inliers far from their predicted depth
    own[5] = np.nan  # one without own depth stays
    keep, whole = an.depth_consistency(own, pred, np.full(40, 0.05))
    assert keep.tolist() == [False] * 5 + [True] * 35 and whole
    keep, whole = an.depth_consistency(pred * 2.0, pred, np.full(40, 0.05))
    assert not keep.any() and not whole  # a 2 x panel: inconsistent as a whole
    keep, whole = an.depth_consistency(pred * 2.0, pred, np.full(40, 0.05), free_scale=True)
    assert keep.all() and whole  # a monocular frame's own scale is free
    keep, whole = an.depth_consistency(pred[:5] * 2.0, pred[:5], np.full(5, 0.05))
    assert whole  # too few to judge the whole fit


def _panel(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A wall seen from ``A``, matched truly, and ``A``'s depth image of it."""
    W = np.column_stack([np.full(120, 3.0), rng.uniform(-1.2, 1.2, 120),
                         rng.uniform(0.1, 1.2, 120)])
    uv, _ = project(A, W)
    u, v = np.meshgrid(np.arange(K.width) + 0.0, np.arange(K.height) + 0.0)
    d = np.stack([(u - K.cx) / K.fx, (v - K.cy) / K.fy, np.ones_like(u)], -1) @ A[:3, :3].T
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (3.0 - A[0, 3]) / d[..., 0]
    grid = np.where(np.isfinite(t) & (t > 0), t, 0.0).astype(np.float32)
    return uv, W, grid


def test_a_frame_whose_own_depth_contradicts_the_fit_has_no_pose(tmp_path: Path) -> None:
    uv, W, grid = _panel(np.random.default_rng(4))
    m = mapper(tmp_path, floor=False)
    got = image_pose(m, uv, W, (grid, K, "sensor"))
    assert got is not None and got[0][1] == len(uv)
    assert image_pose(m, uv, W, (grid * 2.0, K, "sensor")) is None
    # a monocular frame: its depth's shape is tested, not its scale (the floor check's)
    got = image_pose(m, uv, W, (grid * 2.0, K, "mono"))
    assert got is not None and got[0][1] == len(uv)
    # a few inliers its depth contradicts are left out; too many leave no pose
    bad = grid.copy()
    left = uv[:, 0] < 120
    bad[:, :120] *= 3.0
    got = image_pose(m, uv, W, (bad, K, "sensor"))
    assert got is not None and got[0][1] <= len(uv) - int(left.sum()) + 2
    assert image_pose(m, uv[:20], W[:20], (np.where(np.arange(K.width) < 600, grid * 3,
                                                    grid).astype(np.float32), K, "sensor")) is None


def test_the_guards_also_test_a_hypothesis_near_the_prior(tmp_path: Path) -> None:
    uv, W, grid = _panel(np.random.default_rng(7))
    m = mapper(tmp_path, floor=False)
    ref = ref_keyframe()
    corr = corr_of(uv, W, ref)

    def near(own: tuple[np.ndarray, Intrinsics, str]) -> Any:
        f = FrameIn("q", "cam", np.zeros((K.height, K.width, 3), np.uint8), K,
                    prior={"trust": "metric", "T_map_cam": A.tolist(), "sigma_pos_m": 0.05,
                           "sigma_rot_deg": 1.0}, own_depth=own)
        return m._near_prior(f, corr, [ref], A, 0.05, 1.0, len(uv))

    assert near((grid, K, "sensor")) is not None
    assert near((grid * 2.0, K, "sensor")) is None


def _near(m: AnchoredMap, uv: np.ndarray, X: np.ndarray, best: int, sp: float = 0.5) -> Any:
    ref = ref_keyframe()
    f = FrameIn("q", "cam", np.zeros((K.height, K.width, 3), np.uint8), K,
                prior={"trust": "metric", "T_map_cam": A.tolist(), "sigma_pos_m": sp,
                       "sigma_rot_deg": 2.0})
    return m._near_prior(f, corr_of(uv, X, ref), [ref], A, sp, 2.0, best)


def test_a_hypothesis_near_the_prior_is_refitted_off_the_floor(tmp_path: Path) -> None:
    """The near-prior hypothesis takes the off-floor refit as any fit (map.md §2, oms/2.13)."""
    uv, X, _ = scene(np.random.default_rng(8), wall=20)
    got = _near(mapper(tmp_path), uv, X, 40)
    assert got is not None and np.linalg.norm(got[0].c - A[:3, 3]) < 0.02
    uv, X, _ = scene(np.random.default_rng(8), wall=10)
    assert _near(mapper(tmp_path), uv, X, 40) is None  # too few off the floor to refit


def _twin_of(m: AnchoredMap, keep: Any, shift: np.ndarray | None = None) -> None:
    """``depth_confirms`` picking a twin: the same pose on the correspondences ``keep`` takes."""
    def confirms(f: Any, s: Any, ref_of: Any, cands: Any, corr: Any = None, feats: Any = None,
                 prior: Any = None) -> Any:
        sel = keep(corr)
        sub = (corr.uv[sel], corr.X[sel], corr.d[sel], corr.kf[sel])
        c = s.c if shift is None else s.c + shift
        return an.dataclasses.replace(s, c=c, inliers=np.ones(int(sel.sum()), bool)), sub

    m.depth_confirms = confirms  # type: ignore[method-assign]


def test_a_twin_passes_the_guards_too(tmp_path: Path) -> None:
    uv, X, is_floor = scene(np.random.default_rng(3))
    X = np.where(is_floor[:, None], X - TILE, X)  # a correct mixed fit
    m = mapper(tmp_path)
    _twin_of(m, lambda c: np.ones(len(c.uv), bool))
    got = image_pose(m, uv, X)
    assert got is not None and got[2] is not None and got[2] > 0.7
    # a twin resting on the floor alone, a tile away from the prior
    _twin_of(m, lambda c: c.X[:, 2] < 0.05, TILE)
    assert image_pose(m, uv, X) is None
    assert _near(m, uv, X, 40, sp=0.05) is None
    _twin_of(m, lambda c: np.ones(len(c.uv), bool))
    assert _near(m, uv, X, 40, sp=0.05) is not None


def test_locate_points_are_at_each_keyframes_own_scale(tmp_path: Path) -> None:
    """The locate index lifts content depth (a monocular keyframe's own floor-check scale): a
    floor seen 0.5 m below a camera stays on the floor, where registration's lifting at k (twice
    the keyframe's own scale here) puts it 0.5 m lower (api.md §6 "Floor points")."""
    from oh_my_slam.service.backend import read_anchored

    m = AnchoredMap(tmp_path / "map")
    k = ref_keyframe((0.0, 0.0, 0.5))
    k.k_used, k.floor_factor, k.K_grid = 0.5, 1.0, K
    m.keyframes = [k]
    m.scale.k = 1.0
    m.save()
    grid = np.full((K.height, K.width), 0.5 / math.sin(math.radians(25.0)), np.float32)
    np.save(m.root / an.KF_DIR / f"{k.id}.depth.npy", grid.astype(np.float16))
    centre = np.array([[K.cx, K.cy]], np.float32)  # the optical axis: down 25 deg to the floor
    content = read_anchored(m.root).depth(k.id)
    assert content is not None
    on = rl.keyframe_points(centre, K.K(), content, k.T)
    assert abs(float(on[0, 2])) < 0.01 and rl.floor_points(on.astype(np.float64), (0.0, 0.0))[0]
    lifted = rl.keyframe_points(centre, K.K(), m.depth_of(k.id), k.T)
    assert float(lifted[0, 2]) == pytest.approx(-0.5, abs=0.01)


def test_own_depth_sources(tmp_path: Path) -> None:
    m = mapper(tmp_path)
    m.scale.k = 2.0
    rgb = np.zeros((4, 4, 3), np.uint8)
    f = FrameIn("a", "c", rgb, K)
    assert m._own_depth(f) is None
    d = np.ones((4, 4), np.float32)
    f.mono = lambda: (d, K, None, None)
    grid, _, kind = m._own_depth(f)  # type: ignore[misc]
    assert kind == "mono" and float(grid[0, 0]) == 2.0
    f.depth = d * 3
    assert m._own_depth(f)[2] == "sensor"  # type: ignore[index]
    f.own_depth = (d, K, "mono")
    assert m._own_depth(f)[0] is d  # type: ignore[index]
    assert m._kind_scale_sigma("sensor") == an.SENSOR_SCALE_SIGMA_REL
    assert m._kind_scale_sigma("mono") == 0.05
    m.scale.source = "monocular"
    assert m._kind_scale_sigma("mono") == an.MONOCULAR_REGISTRATION_SIGMA_REL


# -- locate ------------------------------------------------------------------------------------


def _index(rng: np.random.Generator, wall: int = 40, floor: bool = True, alias: bool = True
           ) -> tuple[rl.Index, rl.Query]:
    uv, X, is_floor = scene(rng, wall=wall)
    if not alias:
        X = np.where(is_floor[:, None], X - TILE, X)
    desc = rng.integers(0, 256, (len(uv), 128)).astype(np.uint8)
    kfs = []
    for j, c in enumerate(((-0.2, 0.15, 0.5), (-0.2, -0.15, 0.5))):
        T = cam(c)
        sel = np.arange(len(uv)) % 2 == j
        kuv, _ = project(T, X[sel])
        kfs.append(rl.Keyframe(f"k{j}", T, K.K(), kuv.astype(np.float32), desc[sel],
                               X[sel].astype(np.float32)))
    q = rl.Query(uv.astype(np.float32), desc, K.K(), K.width, K.height)
    return rl.Index(kfs, (0.0, 0.0) if floor else None), q


def test_locate_reports_the_off_floor_fit_and_the_alias_as_a_rival() -> None:
    idx, q = _index(np.random.default_rng(5))
    o = rl.locate(q, idx)
    assert o.located and o.T is not None and np.linalg.norm(o.T[:3, 3] - A[:3, 3]) < 0.03
    assert o.ambiguity is not None and o.ambiguity["pos_m"] == pytest.approx(0.6, abs=0.05)
    assert o.floor_share is not None and o.floor_share < 0.2
    idx, q = _index(np.random.default_rng(5), floor=False)
    o = rl.locate(q, idx)
    assert o.T is not None and np.linalg.norm(o.T[:3, 3] - (A[:3, 3] + TILE)) < 0.03
    assert o.floor_share is None


def test_locate_on_the_floor_alone_is_no_pose() -> None:
    idx, q = _index(np.random.default_rng(6), wall=0)
    o = rl.locate(q, idx)
    assert not o.located and o.reason is None
    idx, q = _index(np.random.default_rng(6), wall=0, floor=False)
    assert rl.locate(q, idx).located


def test_locate_prior_residual_against_the_prior_as_sent() -> None:
    def frame(prior: dict[str, Any] | None) -> nav.LocateFrame:
        return nav.LocateFrame("q", None, np.zeros((2, 2, 3), np.uint8), None,  # type: ignore[arg-type]
                               prior, None, None)

    assert nav.prior_residual(frame(None), A.tolist()) == (None, None)
    fr = frame({"trust": "drifting", "T_map_cam": cam((0.3, 0.4, 0.5)).tolist()})
    assert nav.prior_residual(fr, None) == (None, None)
    dm, dd = nav.prior_residual(fr, A.tolist())
    assert dm == pytest.approx(0.5) and dd == pytest.approx(0.0, abs=1e-3)


# -- the one-view drift rule -------------------------------------------------------------------


def _drift_frame(name: str, c: tuple[float, float, float], shift: np.ndarray, yaw: float = 0.0,
                 sp: float = 0.3) -> tuple[FrameIn, Placement]:
    P = cam(c, yaw)
    V = P.copy()
    V[:3, 3] += shift
    f = FrameIn(name, "cam", np.zeros((2, 2, 3), np.uint8), K,
                prior={"trust": "drifting", "T_map_cam": P.tolist(), "sigma_pos_m": sp,
                       "sigma_rot_deg": 5.0})
    pl = Placement(name, "registered", None, V, 0.05, 0.5, 200, "pnp", (0.0, 0.0), align_T=V,
                   image_sigmas=(0.05, 0.5), inlier_share=0.5, measurable=True)
    return f, pl


def _job(m: AnchoredMap, specs: list[tuple[str, tuple[float, float, float], float]],
         shift: np.ndarray) -> tuple[list[FrameIn], dict[int, Placement], Any, dict[int, Placement]]:
    frames, out = [], {}
    for i, (name, c, yaw) in enumerate(specs):
        f, pl = _drift_frame(name, c, shift, yaw)
        frames.append(f)
        out[i] = pl
    used = m.drift
    got, outliers = m._fit_drift(frames, out, used)
    return frames, out, got, outliers


SHIFT = np.array([1.0, 0.0, 0.0])  # a correction of 1 m against priors of sigma 0.3 m


def test_one_frame_never_moves_the_drift(tmp_path: Path) -> None:
    m = mapper(tmp_path)
    m.created_with = "drifting"
    _, _, got, outliers = _job(m, [("a", (2.0, 0.0, 0.5), 0.0)], SHIFT)
    assert got is None and outliers[0].reason == "prior_conflict"
    assert m.drift_rule() == {"held": True, "agreeing_frames": ["a"]}


def test_three_independent_views_move_it(tmp_path: Path) -> None:
    m = mapper(tmp_path)
    m.created_with = "drifting"
    specs = [("a", (2.0, 0.0, 0.5), 0.0), ("b", (3.0, 0.0, 0.5), 0.0), ("c", (4.0, 0.0, 0.5), 0.0)]
    _, _, got, outliers = _job(m, specs, SHIFT)
    assert got is not None and not outliers
    assert m.drift_rule() == {"held": False, "agreeing_frames": ["a", "b", "c"]}
    # within the prior's sigma nothing is tested
    m2 = mapper(tmp_path / "small")
    _job(m2, specs, np.array([0.05, 0.0, 0.0]))
    assert m2.drift_rule() == {"held": False, "agreeing_frames": []}


def test_views_from_one_place_count_once_unless_they_look_apart(tmp_path: Path) -> None:
    m = mapper(tmp_path)
    m.created_with = "drifting"
    one = [("a", (2.0, 0.0, 0.5), 0.0), ("b", (2.1, 0.0, 0.5), 3.0), ("c", (2.2, 0.0, 0.5), 6.0)]
    _, _, got, outliers = _job(m, one, SHIFT)
    assert got is None and sorted(outliers) == [0, 1, 2] and m.drift_rule()["held"]
    apart = [("a", (2.0, 0.0, 0.5), 0.0), ("b", (2.1, 0.0, 0.5), 15.0),
             ("c", (2.2, 0.0, 0.5), 30.0)]
    m2 = mapper(tmp_path / "apart")
    m2.created_with = "drifting"
    _job(m2, apart, SHIFT)
    assert not m2.drift_rule()["held"]


def test_held_views_wait_for_agreement_in_later_jobs(tmp_path: Path) -> None:
    m = mapper(tmp_path)
    m.created_with = "drifting"
    frames, _, _, outliers = _job(m, [("a", (2.0, 0.0, 0.5), 0.0)], SHIFT)
    m._close_drift(frames, outliers, m.drift)
    assert [v.name for v in m.drift.pending] == ["a"]
    saved = an.Drift.from_json(json.loads(json.dumps(m.drift.to_json())))
    assert [v.name for v in saved.pending] == ["a"] and np.allclose(saved.pending[0].T,
                                                                    m.drift.pending[0].T)
    # a view held elsewhere, disagreeing, stays kept beside it
    frames_e, _, _, out_e = _job(m, [("e", (3.5, 3.0, 0.5), 0.0)], np.array([0.0, -1.5, 0.0]))
    m._close_drift(frames_e, out_e, m.drift)
    assert [v.name for v in m.drift.pending] == ["a", "e"]
    m.begin_job()
    frames, out, got, outliers = _job(m, [("b", (3.0, 0.0, 0.5), 0.0),
                                          ("c", (4.0, 0.0, 0.5), 0.0)], SHIFT)
    assert not outliers and m.drift_rule() == {"held": False, "agreeing_frames": ["a", "b", "c"]}
    assert got is not None and [v.name for v in got.pending] == ["a", "e"]  # carried by the fit
    m._close_drift(frames, out, got)
    assert [v.name for v in m.drift.pending] == ["e"]  # "a" consumed
    # a held kept frame (no measurable pose) stays an observation in the next pass
    m.begin_job()
    f, pl = _drift_frame("d", (6.0, 0.0, 0.5), np.array([0.0, 2.0, 0.0]))
    pl.measurable = False
    got, outliers = m._fit_drift([f], {0: pl}, m.drift)
    assert outliers[0].detail == "conflict_unsupported"
    got, again = m._fit_drift([f], outliers, m.drift)
    assert sorted(again) == [0] and m.drift_rule()["held"]


def test_kept_views_support_a_one_frame_job(tmp_path: Path) -> None:
    m = mapper(tmp_path)
    m.created_with = "drifting"
    for name, c in (("a", (2.0, 0.0, 0.5)), ("b", (3.0, 0.0, 0.5))):
        frames, _, _, outliers = _job(m, [(name, c, 0.0)], SHIFT)
        m._close_drift(frames, outliers, m.drift)
        m.begin_job()
    assert [v.name for v in m.drift.pending] == ["a", "b"]
    _, _, got, outliers = _job(m, [("c", (4.0, 0.0, 0.5), 0.0)], SHIFT)
    assert got is not None and not outliers  # one frame here, two kept: three agree


def test_a_change_without_robust_support_is_still_refused(tmp_path: Path) -> None:
    """Image poses far looser than their priors move them little (no view held), yet two of three
    frames agreeing on a 2 m first estimate (a metric map) are no robust support (map.md §1)."""
    m = mapper(tmp_path)
    m.created_with = "metric"
    frames, out = [], {}
    for i, (name, x, shift) in enumerate((("a", 2.0, 2.0), ("b", 3.0, 2.0), ("c", 4.0, 0.0))):
        f, pl = _drift_frame(name, (x, 0.0, 0.5), np.array([shift, 0.0, 0.0]), sp=0.1)
        pl.image_sigmas = (0.5, 0.5)
        frames.append(f)
        out[i] = pl
    got, outliers = m._fit_drift(frames, out, m.drift)
    assert got is None and sorted(outliers) == [0, 1] and not m.drift_rule()["held"]


def test_the_rule_tests_positions_not_rotations(tmp_path: Path) -> None:
    """Repeated texture aliases where a camera is: a rotation the images measure (a robot's yaw
    27 deg off its odometry at its start look) is the conflict bound's, never held alone."""
    m = mapper(tmp_path)
    m.created_with = "drifting"
    f, pl = _drift_frame("r", (2.0, 0.0, 0.5), np.zeros(3))
    pl.align_T = cam((2.0, 0.0, 0.5), 27.0)
    m._fit_drift([f], {0: pl}, m.drift)
    assert m.drift_rule() == {"held": False, "agreeing_frames": []}
    pl.method = "rotation"  # a rotation-only view never moves its prior's position
    V = cam((2.0, 0.0, 0.5), 30.0)
    pl.align_T = V
    o = m._drift_evidence([f], {0: pl})[0]
    assert an._correction(m.drift, o) is None
    # agreement: a rotation-only view is judged by its rotation only
    v = an.DriftView("v", V, cam((2.0, 0.0, 0.5)), (0.05, 0.5), True)
    w = an.DriftView("w", cam((9.0, 0.0, 0.5), 30.0), cam((9.0, 0.0, 0.5)), (0.05, 0.5), False)
    assert [x.name for x in an._agreeing(v, [v, w], 1.0)] == ["v", "w"]


def test_no_drift_evidence_resets_the_rule(tmp_path: Path) -> None:
    m = mapper(tmp_path)
    assert m._fit_drift([], {}, m.drift) == (None, {})
    assert m.drift_rule() == {"held": False, "agreeing_frames": []}


# -- reports and schema -----------------------------------------------------------------------


def test_drift_correction_reports_the_rule() -> None:
    q = drift_correction([(0.1, 1.0)], 1.0, {"held": True, "agreeing_frames": ["a"]})
    assert q is not None and q["held"] and q["agreeing_frames"] == ["a"]
    validate("DriftCorrection", q)
    q = drift_correction([(0.1, 1.0)], 1.0)
    assert q is not None and q["held"] is False and q["agreeing_frames"] == []


def test_input_rows_carry_the_sent_residual_and_floor_share() -> None:
    prior = {"trust": "drifting", "T_map_cam": cam((0.3, 0.4, 0.5)).tolist()}
    sent = sent_residual(A, prior, "drifting")
    assert sent is not None and sent[0] == pytest.approx(0.5)
    assert sent_residual(None, prior, "drifting") is None
    assert sent_residual(A, None, "drifting") is None
    assert sent_residual(A, prior, "none") is None
    row = input_row("f", "registered", None, 40, A, 0.1, 1.0, (0.1, 1.0), None, sent=sent,
                    floor_share=0.25)
    assert row["prior_residual_sent_m"] == 0.5 and row["inlier_floor_share"] == 0.25
    validate("InputStatus", row)
    row = input_row("f", "rejected", "prior_conflict", 40, None, None, None, (0.1, 1.0), None,
                    sent=sent, floor_share=0.25)
    assert row["prior_residual_sent_m"] is None and row["inlier_floor_share"] is None
    validate("InputStatus", row)


def test_locate_results_validate() -> None:
    res = {"name": "q", "status": "located", "reason": None, "T_map_cam": A.tolist(),
           "sigma_pos_m": 0.1, "sigma_rot_deg": 1.0, "inlier_ratio": 0.5, "ambiguity": None,
           "rivals": [], "consistent_with_prior": None, "height_consistent": None,
           "prior_residual_m": 0.2, "prior_residual_deg": 1.0, "inlier_floor_share": 0.1,
           "nearest_view": None}
    validate("LocateResult", res)


# -- the client's map (oms/2.13 evidence, offline replay on small copies) ----------------------


def _replay(q: str, r: str, tag: str = "", placed: bool = False) -> Placement:
    meta = json.loads((DATA / f"{tag}meta.json").read_text())
    z = np.load(DATA / f"{tag}{q}_{r}.npz")
    m = AnchoredMap(Path("/nonexistent"))
    m.scale = an.ScaleState(float(meta["scale"]["k"]), meta["scale"]["source"],
                            float(meta["scale"]["sigma_rel"]))
    m.created_with = "drifting"
    m.floor_z, m.floor_z_sigma = float(meta["floor_z_m"]), 0.0
    rk = meta["keyframes"][r]
    ref = an.Keyframe(r, rk["name"], rk["camera_id"], Intrinsics.from_dict(rk["K"]),
                      np.asarray(rk["T"], np.float64), rk["sigma_pos_m"], rk["sigma_rot_deg"],
                      "fused", rk["prior"], "mono", rk["k_used"], rk["median_depth"],
                      floor_factor=rk["floor_factor"], grid_scale=rk["grid_scale"],
                      floor_unjudged=bool(rk.get("floor_unjudged", False)))
    ref.feats = an.Features(z["r_kps"], z["r_desc"])
    ref.kdepth = z["r_kdepth"]
    qk = meta["keyframes"][q]
    Kq = Intrinsics.from_dict(qk["K"])
    Kg = Intrinsics.from_dict(qk["K_grid"]).resized(80, 60)
    p = qk["prior"]
    prior = {"trust": "drifting", "T_map_cam": p["T"], "sigma_pos_m": p["sigma_pos_m"],
             "sigma_rot_deg": p["sigma_rot_deg"], "sigma_tilt_deg": p.get("sigma_tilt_deg")}
    own = (z["q_depth"].astype(np.float32) * float(m.scale.k / qk["k_used"]), Kg, "mono")
    f = FrameIn(qk["name"], qk["camera_id"], np.zeros((Kq.height, Kq.width, 3), np.uint8), Kq,
                prior, own_depth=own)
    # the prior as sent, or as the drift estimate placed it (a frame the map kept: its pose)
    P = np.asarray(qk["T"] if placed else p["T"], np.float64)
    sp, sr = (qk["sigma_pos_m"], qk["sigma_rot_deg"]) if placed \
        else (p["sigma_pos_m"], p["sigma_rot_deg"])
    return m.place(f, an.Features(z["q_kps"], z["q_desc"]), None, (P, sp, sr), [ref])


@pytest.mark.parametrize(("q", "r"), [("k000011", "k000002"), ("k000013", "k000007")])
def test_the_client_lap_frames_no_longer_register_at_the_start_spot(q: str, r: str) -> None:
    """k000011 matched the start spot's planks (k000002) and k000013 the start spot's floor seen
    through a reference lifted at twice its own scale (k000007): neither registers there now."""
    pl = _replay(q, r)
    start = np.array([0.0, 0.2])
    meta = json.loads((DATA / "meta.json").read_text())
    prior_c = np.asarray(meta["keyframes"][q]["prior"]["T"])[:2, 3]
    if pl.status == "registered":
        assert pl.T is not None
        assert np.linalg.norm(pl.T[:2, 3] - prior_c) < np.linalg.norm(pl.T[:2, 3] - start)
    else:
        assert pl.status == "kept_at_prior"


def test_a_correct_floor_view_near_its_prior_registers() -> None:
    """dev_ainex_j5 (oms/2.13 regression): k000001 of the start look, 90 inliers 96 % on the floor
    and 0.17 m from its prior (sigma 0.47 m), confirms its prior and registers; the j3 lap frames
    above, as far from their priors as those sigmas allow, still do not."""
    pl = _replay("k000001", "k000000", "j5_")
    assert pl.status == "registered" and pl.method == "pnp" and pl.T is not None
    assert np.linalg.norm(pl.T[:2, 3] - np.array([-0.26, -0.03])) < 0.1
    assert pl.inlier_floor_share is not None and pl.inlier_floor_share > 0.9


def test_locate_on_the_floor_may_confirm_the_prior() -> None:
    idx, q = _index(np.random.default_rng(9), wall=0, alias=False)
    q.prior, q.prior_sigma_pos_m, q.prior_sigma_rot_deg = A.copy(), 0.5, 10.0
    o = rl.locate(q, idx)
    assert o.located and o.T is not None and np.linalg.norm(o.T[:3, 3] - A[:3, 3]) < 0.03
    assert o.floor_share == 1.0
    idx, q = _index(np.random.default_rng(9), wall=0)  # the alias, a tile from the prior
    q.prior, q.prior_sigma_pos_m, q.prior_sigma_rot_deg = A.copy(), 0.5, 10.0
    assert not rl.locate(q, idx).located


# dev_ainex_j6 ground truth (base pose in the start frame; the camera adds the head transform)
J6_TRUTH = {"k000009": (-0.069, 0.732), "k000011": (0.041, 1.578)}


@pytest.mark.parametrize(("q", "r"), [("k000009", "k000008"), ("k000011", "k000009"),
                                      ("k000005", "k000004"), ("k000006", "k000005")])
def test_the_j6_frames_register_near_their_references(q: str, r: str) -> None:
    """dev_ainex_j6 (oms/2.13 regression): one frame per job with an arm ignore mask. The lap
    frames' references (k000008, and k000009 itself) are monocular keyframes the floor check did
    not judge (own scales 1.3-2.0 x k); their depth now lifts a later frame's pose, so k000009 and
    k000011 register near truth, and the look frames k000005 / k000006 near their neighbours."""
    pl = _replay(q, r, "j6_", placed=True)
    assert pl.status == "registered" and pl.method == "pnp" and pl.T is not None
    meta = json.loads((DATA / "j6_meta.json").read_text())
    if q in J6_TRUTH:
        # (one reference each, kept at its placed prior in the live map: a few decimetres)
        assert np.linalg.norm(pl.T[:2, 3] - np.array(J6_TRUTH[q])) < 0.35
    else:
        ref_c = np.asarray(meta["keyframes"][r]["T"])[:2, 3]
        assert np.linalg.norm(pl.T[:2, 3] - ref_c) < 0.3


def test_few_features_counts_what_registration_extracts() -> None:
    """api.md §6 ``few_features`` (oms/2.13): keypoints at registration's contrast threshold; a
    faint texture OpenCV's default contrast barely sees still registers (dev_ainex_j6 k000005: 3
    keypoints at the default, 105 inliers in registration). ``suspicious`` keeps the default."""
    from oh_my_slam.mapping import checks

    rng = np.random.default_rng(0)
    img = np.full((240, 320), 128.0)
    for _ in range(300):  # faint blobs
        x, y = rng.integers(10, 310), rng.integers(10, 230)
        img[y - 2:y + 3, x - 2:x + 3] += rng.choice([-6.0, 6.0])
    rgb = np.repeat(np.clip(img, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
    m = checks.measure(rgb)
    assert m.strong is not None and m.features > m.strong and m.features >= checks.FEW_FEATURES
    assert m.suspicious == (m.strong < checks.SUSPECT_FEATURES or m.texture < checks.SUSPECT_TEXTURE)
    assert checks.Measures(500, 0.2, 0.3).suspicious is False  # strong defaults to features
    assert checks.Measures(500, 0.2, 0.3, 10).suspicious is True
