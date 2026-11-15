"""Motion stereo of frame analysis (api.md §7 "Motion stereo", oms/2.11): other frames of the
reference's camera, each at a client-given relative pose with sigmas, matched to the reference;
each relative pose refined on the images; the matches triangulated and the reference's
monocular depth brought onto them (``reconstruction.stereo``). The method and its constants are
the evaluation's (``reports/motion_stereo_eval.md`` §9; README "Frame analysis").

* **Matches** (``pair``): the frames' SIFT features and ratio-test matches as registration takes
  them (``anchored.features``, ``anchored.match``: none on ignored pixels), one per reference
  keypoint; fewer than ``MIN_MATCHES`` is ``no_overlap``.
* **Pose** (``refine``): the relative pose ``X_ref = R X_frame + t`` minimising the matches'
  Sampson distances (pixels, over ``MATCH_SIGMA_PX``; Cauchy, then Huber on the inliers within
  ``INLIER_PX``) with the sent pose as a Gaussian prior (the sent ``sigma_rot_deg`` on the
  rotation, ``sigma_pos_m`` on the translation; floors ``MIN_PRIOR_SIGMA_*``). It is run from
  the sent pose and from the matches' own essential matrix (RANSAC, ``cv2.recoverPose``: so a
  sent rotation far off is found far off, not pulled into a wrong local minimum); the start
  with more inliers wins. The images say nothing of |t|: the refined translation keeps the
  sent length, only its direction is refined. ``sigma_rot_deg_post`` is the posterior's largest
  rotation sigma (Gauss–Newton, the prior included). Fewer than ``MIN_MATCHES`` inliers is
  ``no_overlap``; a sent translation shorter than ``MIN_BASELINE_M`` gives no epipolar geometry
  and is ``baseline_short`` unrefined.
* **Status.** ``pose_inconsistent``: the refined rotation more than ``POSE_K_SIGMA`` x
  ``sigma_rot_deg`` from the sent one; else ``baseline_short``: ``sigma_pos_m`` / |t| above
  ``BASELINE_SIGMA_REL_MAX``; else ``too_few_points``: fewer than ``BASELINE_MIN_POINTS``
  triangulated points with σ(ln z) below ``BASELINE_POINT_SIGMA_MAX``; else ``used``.
  ``points`` is that number of points (for every status that triangulated).
* **Depth** (``motion_stereo``). The points of every ``used`` pair, pooled, fit the reference's
  depth (``reconstruction.stereo.stereo_depth``): metric at the sent |t| (``qualified``), with
  ``scale_sigma_rel`` = hypot(σ_t / |t|, the fit's standard error, its spread), σ_t / |t| of the
  pool being the pairs' combined by their share of the points (the pooled scale is their
  points' weighted mean). Without a used pair, the ``baseline_short`` pairs that hold enough
  points give the depth's shape alone (their |t| is not good enough for its scale)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.geometry import project
from oh_my_slam.core.types import Intrinsics
from oh_my_slam.mapping.anchored import Features, exp_so3, features, log_so3, match, rays, rot_deg
from oh_my_slam.reconstruction.stereo import (
    BASELINE_POINT_SIGMA_MAX,
    MATCH_SIGMA_PX,
    StereoDepth,
    Triangulated,
    stereo_depth,
    triangulate,
)

Array = NDArray[np.float64]

MAX_MOTION_FRAMES = 4  # api.md §7: frames[] holds at most this many (the schema's maxItems)
BASELINE_SIGMA_REL_MAX = 0.10  # sigma_pos_m / |t| of a pair that gives the scale
BASELINE_MIN_POINTS = 30  # ... and its triangulated points below BASELINE_POINT_SIGMA_MAX
POSE_K_SIGMA = 3.0  # ... and its refined rotation within this many sigma_rot_deg of the sent
MIN_MATCHES = 12  # a pair with fewer matches, or inliers once refined, has no overlap
INLIER_PX = 3.0  # Sampson distance of an inlier (pixels of the canonical image)
GATE_PX = 4.0  # the first inliers: within the prior's 3 rotation sigmas (as pixels) + this
CAUCHY_SCALE = 3.0  # in MATCH_SIGMA_PX
HUBER_SCALE = 1.5
MAX_EVALUATIONS = 60  # per least-squares pass
MIN_PRIOR_SIGMA_ROT_DEG = 0.05
MIN_PRIOR_SIGMA_POS_M = 0.0005
MIN_BASELINE_M = 1e-4  # a shorter sent translation gives no epipolar geometry

STATUSES = ("used", "baseline_short", "too_few_points", "pose_inconsistent", "no_overlap")


@dataclass
class Other:
    """Another frame of the reference's camera: its image on the canonical pinhole, its ignored
    pixels there (None: none), its pose in the reference's optical frame and that pose's
    sigmas."""

    rgb: NDArray[np.uint8]
    ignore: NDArray[np.bool_] | None
    T_ref_frame: Array
    sigma_pos_m: float
    sigma_rot_deg: float


@dataclass
class Pair:
    """A frame's report (``motion.frames[]``) and, when it triangulated, its points: their rays
    in the reference (unit z) and their triangulation."""

    index: int
    status: str
    T_refined: Array | None = None
    sigma_rot_deg_post: float | None = None
    rotation_change_deg: float | None = None
    points: int = 0
    sigma_t_rel: float = math.inf
    ra: Array = field(default_factory=lambda: np.zeros((0, 3)))
    tri: Triangulated | None = None

    def body(self) -> dict[str, Any]:
        """The frame's entry of ``motion.frames``."""
        return {"index": self.index, "status": self.status,
                "T_ref_frame_refined": None if self.T_refined is None
                else [[round(float(v), 6) for v in row] for row in self.T_refined],
                "sigma_rot_deg_post": None if self.sigma_rot_deg_post is None
                else round(self.sigma_rot_deg_post, 4),
                "rotation_change_deg": None if self.rotation_change_deg is None
                else round(self.rotation_change_deg, 4),
                "points": int(self.points)}


def sampson_px(R: Array, t: Array, ra: Array, rb: Array, f: float) -> Array:
    """The Sampson distances (pixels at focal ``f``) of matches with rays ``ra`` (reference) and
    ``rb`` (the frame), unit z, under ``X_ref = R X_frame + t`` (scale-free in t)."""
    tx = np.array([[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]])
    E = tx @ R
    Ex, Etx = rb @ E.T, ra @ E
    den = np.sqrt(Ex[:, 0] ** 2 + Ex[:, 1] ** 2 + Etx[:, 0] ** 2 + Etx[:, 1] ** 2) + 1e-15
    out: Array = np.sum(ra * Ex, axis=1) / den * f
    return out


@dataclass
class Refined:
    R: Array
    t: Array
    inliers: NDArray[np.bool_]
    sigma_rot_rad: float
    cost: float


def refine(ra: Array, rb: Array, f: float, R0: Array, t0: Array, sigma_pos_m: float,
           sigma_rot_deg: float, start: tuple[Array, Array]) -> Refined:
    """The relative pose refined on the matches from ``start`` with the sent ``R0``, ``t0`` as
    the prior (module docstring), its translation of the sent length."""
    from scipy.optimize import least_squares

    sr = math.radians(max(sigma_rot_deg, MIN_PRIOR_SIGMA_ROT_DEG))
    st = max(sigma_pos_m, MIN_PRIOR_SIGMA_POS_M)
    Rs, ts = start
    gate = 3.0 * sr * f + GATE_PX
    idx = np.flatnonzero(np.abs(sampson_px(Rs, ts, ra, rb, f)) < gate)
    if len(idx) < MIN_MATCHES:
        idx = np.arange(len(ra))

    def at(x: Array) -> tuple[Array, Array]:
        return exp_so3(x[:3]) @ Rs, ts + x[3:]

    def fun(x: Array, sel: NDArray[np.intp]) -> Array:
        R, t = at(x)
        return np.r_[sampson_px(R, t, ra[sel], rb[sel], f) / MATCH_SIGMA_PX,
                     log_so3(R @ R0.T) / sr, (t - t0) / st]

    x = np.zeros(6)
    for loss, scale in (("cauchy", CAUCHY_SCALE), ("huber", HUBER_SCALE)):
        res = least_squares(fun, x, args=(idx,), loss=loss, f_scale=scale, method="trf",
                            max_nfev=MAX_EVALUATIONS)
        x = res.x
        d = np.abs(sampson_px(*at(x), ra, rb, f))
        sel = np.flatnonzero(d < INLIER_PX)
        if len(sel) < MIN_MATCHES:
            break
        idx = sel
    cov = np.linalg.pinv(res.jac.T @ res.jac)
    srot = math.sqrt(float(np.linalg.norm(cov[:3, :3], 2)))  # its largest eigenvalue (PSD)
    R, t = at(x)
    t = t * (float(np.linalg.norm(t0)) / max(float(np.linalg.norm(t)), 1e-12))
    return Refined(R, t, d < INLIER_PX, srot, float(res.cost))


def image_start(ra: Array, rb: Array, f: float, t0: Array) -> tuple[Array, Array] | None:
    """The relative pose the matches give alone (essential matrix, RANSAC within ``INLIER_PX``,
    the chirality-consistent decomposition), its translation of the sent length; None
    without one."""
    E, mask = cv2.findEssentialMat(rb[:, :2], ra[:, :2], np.eye(3), cv2.RANSAC, 0.999,
                                   INLIER_PX / f)
    if E is None:
        return None
    # several solutions come stacked: the first
    _, R, t, _ = cv2.recoverPose(np.ascontiguousarray(E[:3]), rb[:, :2], ra[:, :2], np.eye(3),
                                 mask=mask)
    return np.asarray(R, np.float64), np.asarray(t, np.float64).ravel() * float(np.linalg.norm(t0))


def pair(index: int, ref: Features, other: Features, K: Intrinsics, o: Other) -> Pair:
    """A frame's ``Pair`` against the reference (module docstring)."""
    io, ir = match(other.desc, ref.desc)  # one per reference keypoint
    if len(ir) < MIN_MATCHES:
        return Pair(index, "no_overlap")
    T0 = np.asarray(o.T_ref_frame, np.float64)
    R0, t0 = T0[:3, :3], T0[:3, 3]
    if float(np.linalg.norm(t0)) < MIN_BASELINE_M:
        return Pair(index, "baseline_short")
    ra, rb = rays(K, ref.kps[ir]), rays(K, other.kps[io])
    f = float(K.fx)
    starts = [(R0, t0)]
    alone = image_start(ra, rb, f, t0)
    if alone is not None:
        starts.append(alone)
    best = None
    for s in starts:
        r = refine(ra, rb, f, R0, t0, o.sigma_pos_m, o.sigma_rot_deg, s)
        if best is None or (int(r.inliers.sum()), -r.cost) > (int(best.inliers.sum()), -best.cost):
            best = r
    assert best is not None
    if int(best.inliers.sum()) < MIN_MATCHES:
        return Pair(index, "no_overlap")
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = best.R, best.t
    change = rot_deg(best.R @ R0.T)
    out = Pair(index, "used", T, math.degrees(best.sigma_rot_rad), change)
    if change > POSE_K_SIGMA * o.sigma_rot_deg:
        out.status = "pose_inconsistent"
        return out
    out.ra = ra[best.inliers]
    out.tri = triangulate(out.ra, rb[best.inliers], best.R, best.t, f, best.sigma_rot_rad)
    out.points = int(np.count_nonzero(out.tri.sigma < BASELINE_POINT_SIGMA_MAX))
    out.sigma_t_rel = o.sigma_pos_m / float(np.linalg.norm(t0))
    if out.sigma_t_rel > BASELINE_SIGMA_REL_MAX:
        out.status = "baseline_short"
    elif out.points < BASELINE_MIN_POINTS:
        out.status = "too_few_points"
    return out


@dataclass
class Motion:
    """The motion stereo of an analysed frame: each frame's ``Pair``, and the reference's depth
    brought onto the triangulated points (None: none fitted, or no depth given): metric at the
    sent baselines when ``qualified`` (a used pair: ``scale_sigma_rel`` its scale's sigma),
    else the shape alone."""

    frames: list[Pair]
    stereo: StereoDepth | None = None
    qualified: bool = False
    scale_sigma_rel: float | None = None

    def body(self) -> dict[str, Any]:
        """The response's ``motion``."""
        return {"frames": [p.body() for p in self.frames]}


def motion_stereo(ref_rgb: NDArray[np.uint8], ref_ignore: NDArray[np.bool_] | None,
                  K: Intrinsics, others: list[Other], depth: NDArray[np.float32] | None,
                  K_grid: Intrinsics) -> Motion:
    """The ``Motion`` of a reference image (canonical pinhole ``K``, its ignored pixels) and
    ``others``, with the reference's monocular ``depth`` on its grid (intrinsics ``K_grid``;
    None: report the frames only, e.g. beside a depth image)."""
    ref = features(ref_rgb, ref_ignore)
    frames = [pair(i, ref, features(o.rgb, o.ignore), K, o) for i, o in enumerate(others)]
    out = Motion(frames)
    used = [p for p in frames if p.status == "used"]
    shape = [p for p in frames if p.status == "baseline_short" and p.points >= BASELINE_MIN_POINTS]
    pool = used or shape
    if depth is None or not pool:
        return out
    ra = np.concatenate([p.ra[p.tri.ok] for p in pool if p.tri is not None])
    z = np.concatenate([p.tri.z[p.tri.ok] for p in pool if p.tri is not None])
    sig = np.concatenate([p.tri.sigma[p.tri.ok] for p in pool if p.tri is not None])
    uv, _ = project(ra, K_grid.K())
    out.stereo = stereo_depth(depth, uv, z, sig)
    if out.stereo is None or not used:
        return out
    n = np.array([p.points for p in used], np.float64)
    rel = np.array([p.sigma_t_rel for p in used])
    t_rel = float(np.sqrt(np.sum((n / n.sum()) ** 2 * rel ** 2)))
    out.qualified = True
    out.scale_sigma_rel = math.hypot(t_rel, out.stereo.fit.sigma_a, out.stereo.fit.spread)
    return out
