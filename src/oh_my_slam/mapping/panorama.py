"""Pose refinement of multi-view keyframes from feature matches and monocular depth.

Rotation-dominant input (a camera turning in place) is posed by multi-view inference, whose
rotations are degrees and whose camera centres decimetres off. Bundle adjustment cannot repair
that: with almost no parallax, few matches become triangulated points and those few are
ill-conditioned, so the rotations stay poorly constrained, weakly observed keyframes drift, and a
map extended by a later update disagrees with its earlier part. Every verified feature match still
constrains its two keyframes, though: the keypoint of keyframe ``a``, lifted with ``a``'s monocular
depth, must reproject onto the matched keypoint of ``b`` (and vice versa). ``refine_poses``
minimises the robust (Cauchy) reprojection error of all those lifted keypoints over the rotations
and camera centres of the free keyframes — optionally also the focal length and the radial
distortion of the camera all keyframes share — with the other keyframes (the map's, when extending
it) fixed. Keypoints without depth are points at infinity (they constrain the rotations only). The
depth's own per-keyframe scale error only scales the recovered baselines (centimetres for a turning
head); it does not bias the rotations. ``refine_turning`` stages it for rotation-dominant input: rotations first, then the
centres restart where the overlapping fixed keyframes are.

Solver: Levenberg-Marquardt on left rotation increments and centres, iteratively reweighted, with
the robust scale shrinking from about a hundred pixels to a few (the initial poses can be degrees
off); weak priors keep each centre near its initial value where the matches say nothing about it
(pure rotation, points at infinity) and the focal length near its initial value.

Distortion: COLMAP's division model (``SIMPLE_DIVISION``), one coefficient ``k`` in normalised
coordinates: a keypoint ``d`` (pixels from the principal point over the focal length) is the ray
``(d / (1 + k |d|²), 1)``, and a ray ``(u, 1)`` projects to ``2 u / (1 + sqrt(1 - 4 k |u|²))``.
With ``k = 0`` it is the pinhole. A wide-angle lens bends straight lines towards the image border
(barrel distortion, ``k < 0``) by far more than the matching tolerance: on ``examples/camera``
(110° diagonally; its first, uncropped frames) a pinhole refinement of the shared focal length ran
from about 1350 px to 6700-9800 px and shrank the pan steps and the tilts to a fifth to a seventh,
the only way a pinhole bends a turn's rays like the lens. So the distortion is refined with the
focal length: 1392 px and ``k = -0.524`` on the 27 current frames (43 % of the half-diagonal at
the corners). A coefficient that moves the image corners by less than ``DISTORTION_MIN_SHARE`` of
the half-diagonal after a robust scale is set to 0 and held: the pinhole stays the pinhole. The
rendered ``ainex-captures`` fitted 1 % of it; four phone photos of trees behind a window
(``office_sequence``'s first, images the phone corrects itself) fitted ``k = +0.06`` with the focal
length 1.2 times their EXIF one, 3.5 %: what few views of a distant scene let the focal length and
the lens trade, not a lens. Kept, it gave the map's camera that focal length for good, the photos
of the next updates (their EXIF focal length) a camera of their own, and the map two cameras that
the rebuild's SfM could not join.
The focal length and the distortion trade off against each other where the views turn little: the
first 9 uncropped frames of ``examples/camera`` (three pan positions) fitted 3750 px and
``k = -0.77`` about as well (0.20° median residual) as their prior's 1350 px and ``k = -0.40``
(0.25°), and turned by half as much. A focal length more than ``FOCAL_MAX_FACTOR`` off the prior is therefore not
accepted: the refinement starts again with the prior's held.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.types import Intrinsics, Pose

USED_CONFIGS = (2, 3, 4, 5, 6)  # COLMAP TwoViewGeometry: calibrated … planar-or-panoramic
MIN_INLIERS = 15
MAX_PER_PAIR = 250
ROBUST_SCALES_PX = (100.0, 30.0, 8.0, 2.5)  # graduated Cauchy scale of the reprojection error
MAX_ITERATIONS = 12  # per robust scale
CENTRE_PRIOR_M = 0.25  # weak prior (standard deviation) of each centre around its initial value
FOCAL_PRIOR_LOG = 0.1  # weak prior (standard deviation) of the log focal-length change
# a shared focal length this far from its initial value (the camera's prior) is a wrong camera
FOCAL_MAX_FACTOR = 2.0
DISTORTION_PRIOR = 0.5  # weak prior (standard deviation) of the division coefficient around 0
# a distortion that moves the image corners by less than this share of the half-diagonal is none
DISTORTION_MIN_SHARE = 0.1
NOISE_PX = 1.5  # keypoint noise the priors are weighed against


@dataclass(frozen=True)
class PairMatches:
    """Inlier matches of one verified image pair (full-resolution pixel coordinates; for a camera
    with distortion, where its pinhole sees them: its keyframes' depth is on the pinhole's grid)."""

    a: str
    b: str
    uv_a: NDArray[np.float64]
    uv_b: NDArray[np.float64]


@dataclass
class View:
    """A keyframe taking part in the refinement."""

    pose: Pose  # camera-to-map (initial value for a free keyframe)
    K: Intrinsics  # full resolution
    depth: NDArray[Any] | None = None  # z-depth on the grid (metric); None: rays only
    K_grid: Intrinsics | None = None
    full_size: tuple[int, int] | None = None
    # the camera (with distortion) whose undistorted image the depth grid covers
    # (``Intrinsics.pinhole``); None: the grid covers the image itself
    grid_lens: Intrinsics | None = None


@dataclass
class PoseFit:
    poses: dict[str, Pose]  # refined camera-to-map poses of the free keyframes
    focal_scale: float = 1.0  # multiply the shared focal length by this
    pairs: int = 0
    matches: int = 0
    median_before_deg: float = float("nan")
    median_after_deg: float = float("nan")
    per_frame_deg: dict[str, float] = field(default_factory=dict)  # median residual per keyframe
    per_frame_matches: dict[str, int] = field(default_factory=dict)  # matches per keyframe
    distortion: float = 0.0  # the shared camera's division coefficient (module docstring)

    def summary(self) -> dict[str, Any]:
        return {"pairs": self.pairs, "matches": self.matches,
                "median_before_deg": round(self.median_before_deg, 4),
                "median_after_deg": round(self.median_after_deg, 4),
                "focal_scale": round(self.focal_scale, 6),
                "distortion": round(self.distortion, 6)}


def verified_matches(db_path: Path, names: Collection[str], min_inliers: int = MIN_INLIERS,
                     max_per_pair: int = MAX_PER_PAIR) -> list[PairMatches]:
    """Inlier matches of every verified pair among ``names`` (at most ``max_per_pair`` per pair,
    evenly subsampled), sorted by image names."""
    import pycolmap

    wanted = set(names)
    db = pycolmap.Database.open(str(db_path))
    try:
        cameras = {c.camera_id: c for c in db.read_all_cameras()}
        images = [im for im in db.read_all_images() if im.name in wanted]
        id_name = {im.image_id: im.name for im in images}
        camera_of = {im.image_id: cameras[im.camera_id] for im in images}
        pair_ids, geoms = db.read_two_view_geometries()
        keypoints: dict[int, NDArray[np.float64]] = {}
        pinhole: dict[int, NDArray[np.float64] | None] = {}
        out = []
        for pid, g in zip(pair_ids, geoms, strict=True):
            if int(g.config) not in USED_CONFIGS or len(g.inlier_matches) < min_inliers:
                continue
            a, b = pycolmap.pair_id_to_image_pair(int(pid))
            if a not in id_name or b not in id_name:
                continue
            for i in (a, b):
                if i not in keypoints:
                    keypoints[i] = np.asarray(db.read_keypoints(i), np.float64)[:, :2]
                    pinhole[i] = (pinhole_pixels(camera_of[i], keypoints[i])
                                  if distorted(camera_of[i]) else None)
            m = np.asarray(g.inlier_matches)
            if len(m) > max_per_pair:
                m = m[np.linspace(0, len(m) - 1, max_per_pair).astype(int)]
            na, nb = id_name[a], id_name[b]
            ka = keypoints[a] if pinhole[a] is None else pinhole[a]
            kb = keypoints[b] if pinhole[b] is None else pinhole[b]
            ua, ub = ka[m[:, 0]], kb[m[:, 1]]  # type: ignore[index]
            out.append(PairMatches(na, nb, ua, ub) if na < nb else PairMatches(nb, na, ub, ua))
    finally:
        db.close()
    return sorted(out, key=lambda p: (p.a, p.b))


def corner_shift(size: tuple[int, int], focal: float, distortion: float) -> float:
    """How far the division model (coefficient ``distortion`` at ``focal``) moves the corners of
    an image of ``size`` (width, height) from where its pinhole sees them, as a share of the
    half-diagonal."""
    d = float(np.hypot(*size)) / 2 / focal
    return abs(d / (1 + distortion * d * d) - d) / d


def distorted(camera: Any) -> bool:
    """Whether a COLMAP camera has a non-zero distortion parameter."""
    return bool(np.any(np.asarray(camera.params, np.float64)[list(camera.extra_params_idxs())]))


def pinhole_pixels(camera: Any, uv: NDArray[Any]) -> NDArray[np.float64]:
    """Where the pinhole of a COLMAP ``camera`` (its focal length and principal point, without its
    distortion) sees the keypoints ``uv`` (full-resolution pixels, (M, 2)) of its image."""
    n = np.asarray(camera.cam_from_img(np.asarray(uv, np.float64).reshape(-1, 2)), np.float64)
    K = np.asarray(camera.calibration_matrix(), np.float64)
    return n * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]


# ------------------------------------------------------------------------------------------------
# solver


def _exp(w: NDArray[Any]) -> NDArray[np.float64]:
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def _skew(v: NDArray[Any]) -> NDArray[np.float64]:
    """Per-row cross-product matrices, (N, 3) → (N, 3, 3)."""
    z = np.zeros(len(v))
    return np.stack([np.stack([z, -v[:, 2], v[:, 1]], 1),
                     np.stack([v[:, 2], z, -v[:, 0]], 1),
                     np.stack([-v[:, 1], v[:, 0], z], 1)], 1)


def _usable_depth(view: View) -> NDArray[np.float64] | None:
    """The view's depth with non-positive values and depth-edge pixels set to inf (no depth)."""
    if view.depth is None or view.K_grid is None or view.full_size is None:
        return None
    from oh_my_slam.core.geometry import depth_edge_mask

    d = np.asarray(view.depth, np.float64)
    ok = (d > 0) & np.isfinite(d)
    ok &= ~depth_edge_mask(np.where(ok, d, 0.0))
    return np.where(ok, d, np.inf)


def _depth_at(view: View, usable: NDArray[np.float64] | None, uv: NDArray[Any]
              ) -> NDArray[np.float64]:
    """Monocular z-depth at full-resolution keypoints (``PairMatches``: where the camera's
    pinhole sees them, at its own focal length, for a camera with distortion; the image's own
    pixels otherwise); inf where there is none (a point at infinity). A grid over the undistorted
    image of ``grid_lens`` is read where that image shows them."""
    if usable is None or view.K_grid is None or view.full_size is None:
        return np.full(len(uv), np.inf)
    from oh_my_slam.mapping.frame import grid_uv

    if view.grid_lens is not None and view.K.k:  # the undistorted image's focal length
        P, K = view.grid_lens.pinhole(), view.K
        uv = (uv - [K.cx, K.cy]) / [K.fx, K.fy] * [P.fx, P.fy] + [P.cx, P.cy]
    elif view.grid_lens is not None:  # the image's own pixels (a camera fitting its lens again)
        uv = view.grid_lens.pinhole_pixels(uv)
    g = grid_uv(uv, view.full_size, view.K_grid)
    u, v = np.rint(g[:, 0]).astype(int), np.rint(g[:, 1]).astype(int)
    h, w = usable.shape
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    out = np.full(len(uv), np.inf)
    out[inside] = usable[v[inside], u[inside]]
    return out


@dataclass
class _Problem:
    """Every match in both directions: the keypoint of ``src`` lifted with its depth and
    reprojected into ``dst``, compared with the matched keypoint there. Directions are stored in
    contiguous segments (one per pair and direction)."""

    src: NDArray[np.int64]  # (N,) view index
    dst: NDArray[np.int64]
    q: NDArray[np.float64]  # (N, 2) src keypoint, normalised at the initial focal length
    m: NDArray[np.float64]  # (N, 2) dst keypoint relative to dst's principal point (px)
    f: NDArray[np.float64]  # (N, 2) dst's initial (fx, fy)
    depth: NDArray[np.float64]  # (N,) z-depth in src (inf: point at infinity)
    starts: NDArray[np.int64]  # (S,) first match of each segment
    seg_views: list[tuple[int, int]]  # (src, dst) view index of each segment


BEHIND_PX = 1000.0  # residual of a point that projects behind the destination camera


def _evaluate(pb: _Problem, R: NDArray[Any], C: NDArray[Any], phi: float, kappa: float,
              jac: bool, lens: bool = True
              ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Reprojection residuals (N, 2) in pixels and, with ``jac``, their Jacobian (N, 2, 14) with
    respect to (δR_src, C_src, δR_dst, C_dst, φ, k); ``R`` (V, 3, 3) and ``C`` (V, 3) are the
    views' camera-to-map rotations and centres, the focal lengths the initial ones times exp(φ),
    ``kappa`` the division coefficient of the shared camera (module docstring). Without ``lens``
    (the coefficient held) its column stays 0; a pinhole (``kappa`` 0) skips the distortion."""
    Ra, Rb = R[pb.src], R[pb.dst]
    s = np.exp(phi)
    da = pb.q / s  # the src keypoint, normalised at the current focal length
    ra2 = np.sum(da * da, axis=1) if kappa or lens else None
    rho = 1.0 + kappa * ra2 if kappa and ra2 is not None else None
    ua = da if rho is None else da / rho[:, None]
    pa = np.column_stack([ua, np.ones(len(da))])  # z = 1 ray in src
    fin = np.isfinite(pb.depth)
    z = np.where(fin, pb.depth, 1.0)
    va = np.einsum("nij,nj->ni", Ra, pa)
    P = np.where(fin[:, None], (C[pb.src] - C[pb.dst]) + z[:, None] * va, va)
    X = np.einsum("nji,nj->ni", Rb, P)  # Rb^T P, in dst's camera frame
    Z = X[:, 2]
    front = 1e-3 * np.linalg.norm(X, axis=1) < Z
    Zs = np.where(front, Z, 1.0)
    x = X[:, :2] / Zs[:, None]  # the ray in dst, normalised
    fs = pb.f * s
    r2 = np.sum(x * x, axis=1) if kappa or lens else None
    if kappa and r2 is not None:
        disc = 1.0 - 4.0 * kappa * r2
        front &= disc > 0  # inside the lens
        t = np.sqrt(np.where(front, disc, 1.0))
        g = 2.0 / (1.0 + t)  # the ray's distortion: x -> g x
        pred = fs * g[:, None] * x
    else:
        t = g = np.ones(len(x))
        pred = fs * x
    r = np.where(front[:, None], pred - pb.m, BEHIND_PX)
    if not jac:
        return r, None
    # d(pred)/dX = fs (g I + c x xᵀ) d(x)/dX
    A = np.zeros((len(r), 2, 3))
    A[:, 0, 0] = 1.0 / Zs
    A[:, 1, 1] = 1.0 / Zs
    A[:, :, 2] = -x / Zs[:, None]
    if kappa:
        c = 8.0 * kappa / (t * (1.0 + t) ** 2)
        xA = x[:, 0, None] * A[:, 0, :] + x[:, 1, None] * A[:, 1, :]  # xᵀ d(x)/dX
        A = g[:, None, None] * A + (c[:, None] * x)[:, :, None] * xA[:, None, :]
    A *= fs[:, :, None]
    A[~front] = 0.0
    ARt = np.einsum("nij,nkj->nik", A, Rb)  # de/dP
    J = np.zeros((len(r), 2, 14))
    J[:, :, 0:3] = -np.einsum("nij,njk->nik", ARt, z[:, None, None] * _skew(va))
    J[:, :, 3:6] = np.where(fin[:, None, None], ARt, 0.0)
    J[:, :, 6:9] = np.einsum("nij,njk->nik", ARt, _skew(P))
    J[:, :, 9:12] = -J[:, :, 3:6]

    def through_src(dua: NDArray[Any]) -> NDArray[np.float64]:
        """The residuals' change for a change ``dua`` of the src ray (N, 2)."""
        dpa = np.column_stack([dua, np.zeros(len(dua))])
        return np.asarray(np.einsum("nij,nj->ni", ARt,
                                    z[:, None] * np.einsum("nij,nj->ni", Ra, dpa)))

    on = front[:, None]
    dua_dphi = -da if rho is None or ra2 is None else -da * ((1.0 - kappa * ra2) / rho**2)[:, None]
    J[:, :, 12] = through_src(dua_dphi) + np.where(on, pred, 0.0)
    if lens and ra2 is not None and r2 is not None:
        sq = 1.0 if rho is None else rho**2
        dg = 4.0 * r2 / (t * (1.0 + t) ** 2)  # dg/dk at fixed x
        J[:, :, 13] = through_src(-da * (ra2 / sq)[:, None]) + np.where(on, fs * dg[:, None] * x,
                                                                        0.0)
    return r, J


def _cost(r: NDArray[Any], c: float) -> float:
    return float(0.5 * c * c * np.log1p(np.sum(r * r, axis=1) / (c * c)).sum())


def _normal_equations(pb: _Problem, r: NDArray[Any], J: NDArray[Any], c: float,
                      free: dict[int, int], npar: int) -> tuple[NDArray[np.float64],
                                                                NDArray[np.float64]]:
    """Σ w JᵀJ and Σ w Jᵀr (Cauchy weights at scale ``c``) over the free views' parameters: one
    14 x 14 block per segment (its two views' rotations and centres, the focal length and the
    distortion), by a matrix product over the segment's matches, added into the free views'
    parameters (the camera's two last)."""
    H = np.zeros((npar, npar))
    g = np.zeros(npar)
    w = 1.0 / (1.0 + np.sum(r * r, axis=1) / (c * c))
    Jw = J * w[:, None, None]
    ends = np.r_[pb.starts[1:], len(r)]
    for j, (lo, hi) in enumerate(zip(pb.starts.tolist(), ends.tolist(), strict=True)):
        parts = [(6 * free[view], off, 6)
                 for view, off in zip(pb.seg_views[j], (0, 6), strict=True) if view in free]
        parts.append((npar - 2, 12, 2))
        A = J[lo:hi].reshape(-1, 14)
        Aw = Jw[lo:hi].reshape(-1, 14)
        Hs = Aw.T @ A
        gs = Aw.T @ r[lo:hi].reshape(-1)
        for ga, la, sa in parts:
            g[ga:ga + sa] += gs[la:la + sa]
            for gb, lb, sb in parts:
                H[ga:ga + sa, gb:gb + sb] += Hs[la:la + sa, lb:lb + sb]
    return H, g


def refine_poses(pairs: list[PairMatches], views: dict[str, View], free: Collection[str],
                 refine_focal: bool = False, use_depth: bool = True,
                 scales: tuple[float, ...] = ROBUST_SCALES_PX, hold_focal: bool = False,
                 hold_distortion: bool = False,
                 gravity: dict[str, tuple[NDArray[Any], float]] | None = None) -> PoseFit:
    """Refine the camera-to-map poses of the ``free`` keyframes (see the module docstring).

    ``views``: every keyframe that may take part (free and fixed); pairs with a keyframe outside
    it, or with no free keyframe, are ignored. ``refine_focal`` scales every keyframe's focal
    length by one common factor and gives them one radial distortion (one shared camera, its
    keypoints as the image shows them); a focal length that leaves its initial value by more than
    ``FOCAL_MAX_FACTOR`` is a wrong camera, and the refinement starts again with it held
    (``hold_focal``: the distortion alone refined). ``hold_distortion``: the camera's distortion
    is known (the keypoints are where its pinhole sees them), the focal length alone refined.
    ``use_depth=False`` is the pure-rotation model (centres stay put). ``gravity``: for free
    keyframes, the up direction in the camera and its uncertainty (degrees), in a map whose up is
    +z: the tilt between them is a prior (``_tilt``), which levels a block that hangs on the rest
    by a few weak matches about the axis they leave free."""
    names = sorted(n for n in free if n in views)
    used_pairs = [p for p in pairs if p.a in views and p.b in views
                  and (p.a in free or p.b in free)]
    fit = PoseFit({n: views[n].pose for n in names}, 1.0, len(used_pairs),
                  sum(len(p.uv_a) for p in used_pairs))
    if not used_pairs or not names:
        return fit
    order = sorted({n for p in used_pairs for n in (p.a, p.b)} | set(names))
    vid = {n: k for k, n in enumerate(order)}
    free_idx = {vid[n]: k for k, n in enumerate(names)}
    usable = {n: _usable_depth(views[n]) if use_depth else None for n in order}
    cols: dict[str, list[Any]] = {k: [] for k in ("src", "dst", "q", "m", "f", "depth")}
    starts, seg_views = [], []
    count = 0
    for p in used_pairs:
        for a, b, ua, ub in ((p.a, p.b, p.uv_a, p.uv_b), (p.b, p.a, p.uv_b, p.uv_a)):
            Ka, Kb = views[a].K, views[b].K
            size = len(ua)
            cols["src"].append(np.full(size, vid[a]))
            cols["dst"].append(np.full(size, vid[b]))
            cols["q"].append((ua - [Ka.cx, Ka.cy]) / [Ka.fx, Ka.fy])
            cols["m"].append(ub - [Kb.cx, Kb.cy])
            cols["f"].append(np.tile([Kb.fx, Kb.fy], (size, 1)))
            cols["depth"].append(_depth_at(views[a], usable[a], ua))
            starts.append(count)
            seg_views.append((vid[a], vid[b]))
            count += size
    pb = _Problem(*(np.concatenate(cols[k]) for k in ("src", "dst", "q", "m", "f", "depth")),
                  starts=np.asarray(starts, np.int64), seg_views=seg_views)
    R = np.stack([views[n].pose.R for n in order]).astype(np.float64)
    C = np.stack([views[n].pose.t for n in order]).astype(np.float64)
    fi = np.array([vid[n] for n in names])
    C0 = C[fi].copy()
    npar = 6 * len(names) + 2
    phi = kappa = 0.0
    prior_c = (NOISE_PX / CENTRE_PRIOR_M) ** 2
    prior_f = (NOISE_PX / FOCAL_PRIOR_LOG) ** 2
    prior_k = (NOISE_PX / DISTORTION_PRIOR) ** 2
    f_px = float(np.median(pb.f))
    image = (views[names[0]].K.width, views[names[0]].K.height)  # the shared camera's
    tilts = [(k, np.asarray(gravity[n][0], np.float64) / np.linalg.norm(gravity[n][0]),
              (NOISE_PX / np.radians(gravity[n][1])) ** 2)
             for k, n in enumerate(names) if gravity and n in gravity]

    def residuals(Rs: NDArray[Any], Cs: NDArray[Any], ph: float, ka: float
                  ) -> NDArray[np.float64]:
        return _evaluate(pb, Rs, Cs, ph, ka, jac=False, lens=False)[0]

    def total(r: NDArray[Any], Rs: NDArray[Any], Cs: NDArray[Any], ph: float, ka: float,
              c: float) -> float:
        prior = prior_c * float(np.sum((Cs[fi] - C0) ** 2))
        prior += sum(w * float(np.sum(_tilt(Rs[fi[k]] @ u)[0] ** 2)) for k, u, w in tilts)
        return _cost(r, c) + 0.5 * (prior + prior_f * ph * ph + prior_k * ka * ka)

    def degrees(r: NDArray[Any]) -> NDArray[np.float64]:
        return np.degrees(np.linalg.norm(r, axis=1) / f_px)

    fit.median_before_deg = float(np.median(degrees(residuals(R, C, phi, kappa))))
    fixed = [] if use_depth else [6 * k + j for k in range(len(names)) for j in (3, 4, 5)]
    if not refine_focal:
        fixed += [npar - 2, npar - 1]
    else:
        fixed += ([npar - 2] if hold_focal else []) + ([npar - 1] if hold_distortion else [])
    for c in scales:
        lam = 1e-3
        r = residuals(R, C, phi, kappa)
        current = total(r, R, C, phi, kappa, c)
        for _ in range(MAX_ITERATIONS):
            r, J = _evaluate(pb, R, C, phi, kappa, jac=True, lens=npar - 1 not in fixed)
            assert J is not None
            H, g = _normal_equations(pb, r, J, c, free_idx, npar)
            for k in range(len(names)):  # weak centre prior
                sl = slice(6 * k + 3, 6 * k + 6)
                H[sl, sl] += prior_c * np.eye(3)
                g[sl] += prior_c * (C[fi[k]] - C0[k])
            for k, u, w in tilts:  # gravity prior
                e, Jt = _tilt(R[fi[k]] @ u)
                sl = slice(6 * k, 6 * k + 3)
                H[sl, sl] += w * Jt.T @ Jt
                g[sl] += w * Jt.T @ e
            H[-2, -2] += prior_f
            g[-2] += prior_f * phi
            H[-1, -1] += prior_k
            g[-1] += prior_k * kappa
            for i in fixed:
                H[i, :] = 0.0
                H[:, i] = 0.0
                H[i, i] = 1.0
                g[i] = 0.0
            improved = False
            delta = np.zeros(npar)
            for _ in range(8):
                A = H + lam * np.diag(np.maximum(np.diag(H), 1e-9))
                try:
                    delta = -np.linalg.solve(A, g)
                except np.linalg.LinAlgError:
                    lam *= 10
                    continue
                Rn, Cn = R.copy(), C.copy()
                for k, v in enumerate(fi):
                    Rn[v] = _exp(delta[6 * k:6 * k + 3]) @ R[v]
                    Cn[v] = C[v] + delta[6 * k + 3:6 * k + 6]
                phn, kan = phi + float(delta[-2]), kappa + float(delta[-1])
                cn = total(residuals(Rn, Cn, phn, kan), Rn, Cn, phn, kan, c)
                if cn < current:
                    R, C, phi, kappa, current = Rn, Cn, phn, kan, cn
                    lam = max(lam / 3, 1e-7)
                    improved = True
                    break
                lam *= 10
            if not improved or float(np.abs(delta).max()) < 1e-7:
                break
        if refine_focal and npar - 1 not in fixed and corner_shift(
                image, f_px * np.exp(phi), kappa) < DISTORTION_MIN_SHARE:
            kappa = 0.0  # no lens distortion: the pinhole, as without it
            fixed.append(npar - 1)
    if refine_focal and abs(phi) > np.log(FOCAL_MAX_FACTOR):
        return refine_poses(pairs, views, free, refine_focal, use_depth, scales,
                            hold_focal=True, hold_distortion=hold_distortion, gravity=gravity)
    final = degrees(residuals(R, C, phi, kappa))
    fit.poses = {n: Pose(R[vid[n]], C[vid[n]]) for n in names}
    fit.focal_scale = float(np.exp(phi))
    fit.distortion = kappa
    fit.median_after_deg = float(np.median(final))
    for n in names:
        mine = (pb.src == vid[n]) | (pb.dst == vid[n])
        if mine.any():
            fit.per_frame_deg[n] = float(np.median(final[mine]))
            fit.per_frame_matches[n] = int(mine.sum() // 2)
    return fit


def _tilt(up: NDArray[Any]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The tilt of a camera whose up direction in the map is ``up`` (unit): ``up`` x +z (its sine
    as length), and its Jacobian with respect to a left rotation increment of the camera."""
    z = np.array([0.0, 0.0, 1.0])
    return np.cross(up, z), _skew(z[None])[0] @ _skew(np.asarray(up, np.float64)[None])[0]


def centre_hints(pairs: list[PairMatches], views: dict[str, View], free: Collection[str]
                 ) -> dict[str, NDArray[np.float64]]:
    """For each free keyframe, the median centre of the fixed keyframes it shares matches with
    (of all fixed keyframes when it shares none)."""
    fixed = [n for n in views if n not in free]
    if not fixed:
        return {n: views[n].pose.t.copy() for n in free if n in views}
    everywhere = np.median([views[n].pose.t for n in fixed], axis=0)
    near: dict[str, list[str]] = {}
    for p in pairs:
        for a, b in ((p.a, p.b), (p.b, p.a)):
            if a in free and b in views and b not in free:
                near.setdefault(a, []).append(b)
    return {n: (np.median([views[m].pose.t for m in near[n]], axis=0) if n in near
                else everywhere.copy()) for n in free if n in views}


def refine_turning(pairs: list[PairMatches], views: dict[str, View], free: Collection[str],
                   refine_focal: bool = False, hold_distortion: bool = False,
                   gravity: dict[str, tuple[NDArray[Any], float]] | None = None) -> PoseFit:
    """``refine_poses`` for rotation-dominant input, whose multi-view camera centres are noise
    (decimetres, where the head moves centimetres) that can trap the joint refinement: first the
    rotations alone (pure-rotation model, centres irrelevant), then every free centre restarts at
    the centre of the fixed keyframes it overlaps (``centre_hints``), then rotations and centres
    are refined together with the depth."""
    rot = refine_poses(pairs, views, free, use_depth=False, gravity=gravity)
    hints = centre_hints(pairs, views, free)
    staged = {n: replace(v, pose=Pose(rot.poses[n].R, hints[n])) if n in rot.poses else v
              for n, v in views.items()}
    fit = refine_poses(pairs, staged, free, refine_focal=refine_focal,
                       hold_distortion=hold_distortion, gravity=gravity)
    fit.median_before_deg = rot.median_before_deg
    return fit
