"""One standpoint and rotation consistency without priors (map.md §1, §4; README "Frames without
priors").

**Two-view fit** (``two_view``). Two frames' epipolar inliers (``anchored.epipolar``: SIFT
matches that agree with some relative pose, essential matrix and RANSAC), as unit rays; a
rotation-only fit (RANSAC on the rays) gives the relative rotation of a turn in place. The
pair's *two-view parallax* (map.md §1, the one parallax measure) is the median, over the
epipolar inliers, of the rotation-only residual (pixels of the images as sent: its angle times
the pair's focal length) divided by the sine of the ray's angle to the epipolar fit's baseline (at least
``SIN_MIN``). About 0 when the camera only turned. Below ``PARALLAX_MAX_PX`` the pair is
translation-unconstrained: one standpoint. The pair's *overlap* is the smaller of the two images'
8 x 8 inlier-cell coverages (``anchored.cell_share``, map.md §1 "Overlap"). The image-measured
relative rotation is that of the relative pose refined on the epipolar inliers
(``refined_rotation``: Huber-robust Sampson error, from the essential matrix's pose and from the
rotation-only fit), independent of any joint solve. When both frames carry depth images, the fit
is the first of up to ``EPIPOLAR_HYPOTHESES`` sequential hypotheses that the pair's depth images
confirm (``anchored.depth_check``) after a PnP refinement on its depth-lifted inliers kept on its
hypothesis (``depth_pose``), whose rotation is then the measured one: repetitive or symmetric
content (a square tile lattice) can make the largest consensus a false geometry. Every hypothesis
that could make the pair measurable is evaluated, not only up to the first confirmed: when two
confirmed ones have rotations at least ``SYMMETRIC_MIN_DEG`` apart, the scene itself is symmetric
(depth cannot reject either), and the pair is ambiguous (``TwoView.ambiguous``): no measurable
pair, and no placement by that reference alone (map.md §1 "Symmetric scenes", §2 step 2). Two
that share the rotation but lie ``TWIN_MIN_M`` apart are translational twins, separated by their
predicted two-view parallax against the pair's measured one (``anchored.twin_pick``), ambiguous
only when it cannot separate them (map.md §1 "Translational twins", P28).

**Measurable pair** (``TwoView.measurable``, map.md §1): at least ``CHECK_MIN_INLIERS`` epipolar
inliers of a relative pose confirmed by the depth images where both frames carry them, an overlap
of at least ``VIEW_SHARE``, and, for every pair (P28), the cycle rule (``consistent``: within
``CYCLE_MAX_DEG`` of the chain of stronger pairs, stored-to-stored pairs measured from the stored
images where a chain needs them, P27). Standpoint grouping and the pair limits use it, over the
same pairs (``evaluated``). The measurement is never seeded from, nor refined toward, any
committed or joint pose. The pairs fitted are every pair of views that can share a view (``close``: centres within
``PAIR_MAX_DIST_M``, optical axes within ``PAIR_MAX_ANGLE_DEG``, the registration candidates'
rule), with no count cap and in canonical name order: they do not depend on the request order.

**Standpoint groups** (``groups``). Frames without priors whose pairwise parallax is below
``PARALLAX_MAX_PX`` over measurable pairs form one standpoint:
connected through such pairs. A group is registered as a pure rotation about a common centre:
the position of the overlapped stored image-confirmed keyframe with the most pair inliers with
the group (ties: the lowest keyframe id; stored keyframes never move), else the mean of its
frames' image-only centres; its position sigma is their spread (RMS distance to that centre). Each frame keeps the
engine's image-only rotation: a two-view rotation of a small baseline absorbs the translation (a
robot turning with its camera ahead of its axis: 1.5-2 deg per pair on the test room, adding up
along a chain), the engine's multi-view or lifted-point rotation does not. Its rotation sigma is
the noise of its path along the strongest pairs (a maximum spanning tree by inliers) from that
stored keyframe, else the group's first frame in canonical name order.

**Rotation consistency** (``consistency``). Over the measurable pairs among the update's
registered frames and the confirmed keyframes they can share a view with: the angle
between their committed relative rotation and the pair's image-measured one, but the pairs both
of whose frames have two or more measurable pairs none with a chain (``outside``, P28b); the
hard limit ``rotation_inconsistent`` fails when the median exceeds ``MEDIAN_MAX_DEG`` or any pair
``PAIR_MAX_DEG``. No measurable pair passes."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.log import get_logger
from oh_my_slam.core.types import Intrinsics
from oh_my_slam.mapping import sigmas
from oh_my_slam.mapping.anchored import (
    CANDIDATE_MAX_ANGLE_DEG,
    CANDIDATE_MAX_DIST_M,
    EPIPOLAR_HYPOTHESES,
    EPIPOLAR_MIN_MATCHES,
    MEASURABLE_INLIERS,
    MEASURABLE_OVERLAP,
    MIN_AGREEING,
    MIN_DEPTH_M,
    PARALLAX_SIN_MIN,
    REF_WIDTH,
    REPROJ_PX,
    SYMMETRIC_MIN_DEG,
    TWIN_MIN_M,
    AnchoredMap,
    DepthCounts,
    DepthRef,
    Epipolar,
    Features,
    FrameIn,
    Keyframe,
    Placement,
    cell_share,
    depth_check,
    depth_image,
    epipolar_hypotheses,
    exp_so3,
    kabsch,
    natural_key,
    pose,
    predicted_parallax,
    rot_deg,
    rotation_ransac,
    sample_depth,
    twin_pick,
    twins,
    unit,
)

Array = NDArray[np.float64]
log = get_logger("oh_my_slam.mapping.standpoint")

PARALLAX_MAX_PX = 1.0  # map.md §1: a pair below this two-view parallax (px) is one standpoint
VIEW_SHARE = MEASURABLE_OVERLAP  # map.md §1 "Measurable pair": an overlap of at least this
MEDIAN_MAX_DEG = 3.0  # rotation_inconsistent: the pairs' median disagreement above this...
PAIR_MAX_DEG = 10.0  # ... or one pair's above this
MIN_MATCHES = EPIPOLAR_MIN_MATCHES  # a pair with fewer epipolar inliers is not measured
# a measurable pair (map.md §1) has this many epipolar inliers: a weakly overlapping pair's
# two-view rotation is off by 4-8 deg on 50 inliers (test room), which no committed map could
# meet within the 10 deg pair limit
CHECK_MIN_INLIERS = MEASURABLE_INLIERS
SIN_MIN = PARALLAX_SIN_MIN  # a ray this near the baseline has its parallax scaled by 1/this at most
# the pairs measured (``close``): committed centres at most this far apart and optical axes at
# most this far apart, the registration candidates' rule (``anchored``): views further apart share
# no view. No count cap: every such pair is fitted (each once, ``Pairs``)
PAIR_MAX_DIST_M = CANDIDATE_MAX_DIST_M
PAIR_MAX_ANGLE_DEG = CANDIDATE_MAX_ANGLE_DEG
PNP_MIN_POINTS = 6  # a hypothesis's depth pose (``depth_pose``) needs this many lifted inliers
PNP_ITERATIONS = 200
# ``depth_pose`` keeps the PnP on its hypothesis: a PnP rotation further than this from the
# hypothesis's Sampson rotation left it (lattice-ambiguous lifted points: 52 -> 134 deg on the
# empty room), and is refined locally from the hypothesis instead
PNP_MAX_STEP_DEG = 5.0
# map.md §1 "Cycle consistency" (P20, P28: every pair): a pair whose two-view rotation disagrees
# by more than this with the chain of stronger measured pairs is not measurable
CYCLE_MAX_DEG = 10.0
LOG_HYPOTHESES = EPIPOLAR_HYPOTHESES  # (L1) at most this many hypothesis lines per pair


DepthImage = tuple[NDArray[np.float32], Intrinsics]  # metres on a grid, the grid's intrinsics


@dataclass
class View:
    """A frame as the two-view fits see it: its name, intrinsics, features, committed rotation
    and centre (image-only for a new frame), whether it is a stored confirmed keyframe, its
    request order (new frames), and its depth image when it carries one (loaded once, when a fit
    needs it)."""

    name: str
    K: Intrinsics
    feats: Features
    R: Array
    c: Array
    stored: bool = False
    order: int = 0
    kid: str = ""  # a stored keyframe's id (the standpoint centre's tie-break)
    depth: Callable[[], DepthImage | None] | None = None
    _depth: list[DepthImage | None] = field(default_factory=list, repr=False)
    _digest: list[str] = field(default_factory=list, repr=False)

    def depth_image(self) -> DepthImage | None:
        """Its depth image (metres on its grid, the grid's intrinsics), or None (monocular)."""
        if self.depth is None:
            return None
        if not self._depth:
            self._depth.append(self.depth())
        return self._depth[0]

    def digest(self) -> str:
        """What its two-view fits depend on (``Pairs`` cache key): its features, intrinsics and
        whether it carries a depth image."""
        if not self._digest:
            h = hashlib.sha1(np.ascontiguousarray(self.feats.kps).tobytes())
            h.update(np.ascontiguousarray(self.feats.desc).tobytes())
            h.update(repr((self.K.fx, self.K.fy, self.K.cx, self.K.cy, self.K.width,
                           self.K.height, self.depth is not None)).encode())
            self._digest.append(h.hexdigest()[:20])
        return self._digest[0]


@dataclass
class TwoView:
    R_ab: Array  # the image-measured rotation of b's camera in a's: R_a^T R_b
    parallax: float  # the two-view parallax (px, map.md §1)
    inliers: int
    share: float  # the overlap: the smaller of the two coverages
    noise_deg: float  # the rotation-only fit's residual angle over sqrt(inliers)
    # its relative pose is confirmed (map.md §1 "Measurable pair"): by both frames' depth images
    # when both carry one (``depth``); the cycle rule (``consistent``) then judges every pair
    confirmed: bool = True
    depth: bool = False  # both frames carry depth images: their test came first
    # two distinct hypotheses (``SYMMETRIC_MIN_DEG`` apart, or translational twins the parallax
    # cannot separate) both pass the depth confirmation (map.md §1 "Symmetric scenes",
    # "Translational twins"): not confirmed, and no placement by this pair alone
    ambiguous: bool = False
    hyp: int = 0  # the confirmed hypothesis (1-based, ``two_view``), 0 without the depth test
    t_m: Array | None = None  # b's centre in a's camera (m), the depth pose's; None without
    agreement: tuple[float, float] | None = None  # its depth agreement a>b, b>a
    # the pair's measured two-view parallax (its largest consensus, map.md §1) and the one the
    # fit's depth pose predicts there (``anchored.predicted_parallax``): translational twins
    measured_px: float | None = None
    predicted_px: float | None = None
    # the pair's matches: a's and b's pixels of its largest consensus (not cached on disk)
    ua: Array | None = field(default=None, repr=False)
    ub: Array | None = field(default=None, repr=False)

    @property
    def unconstrained(self) -> bool:
        """Translation-unconstrained (map.md §1): below ``PARALLAX_MAX_PX``."""
        return self.parallax < PARALLAX_MAX_PX

    @property
    def measurable(self) -> bool:
        """A measurable pair (map.md §1): ``CHECK_MIN_INLIERS`` epipolar inliers of a confirmed
        relative pose and an overlap of at least ``VIEW_SHARE``; every pair also passes the
        cycle rule among the pairs evaluated together (``consistent``). Standpoint grouping and
        the pair limits use it."""
        return self.confirmed and self.inliers >= CHECK_MIN_INLIERS and self.share >= VIEW_SHARE

    @property
    def standpoint(self) -> bool:
        return self.unconstrained and self.measurable

    def reversed(self) -> TwoView:
        """The same fit with the frames swapped (a's camera in b's)."""
        t = None if self.t_m is None else -self.R_ab.T @ self.t_m
        ag = None if self.agreement is None else (self.agreement[1], self.agreement[0])
        return dataclasses.replace(self, R_ab=self.R_ab.T, t_m=t, agreement=ag, ua=self.ub,
                                   ub=self.ua)

    def to_json(self) -> dict[str, Any]:
        """For the stored pairs' cache (``Pairs.cache``): its fields but the pixels."""
        return {"R_ab": self.R_ab.tolist(), "parallax": self.parallax, "inliers": self.inliers,
                "share": self.share, "noise_deg": self.noise_deg, "confirmed": self.confirmed,
                "depth": self.depth, "ambiguous": self.ambiguous, "hyp": self.hyp,
                "t_m": None if self.t_m is None else self.t_m.tolist(),
                "agreement": None if self.agreement is None else list(self.agreement),
                "measured_px": self.measured_px, "predicted_px": self.predicted_px}

    @staticmethod
    def from_json(d: dict[str, Any]) -> TwoView:
        ag = d.get("agreement")
        return TwoView(np.asarray(d["R_ab"], np.float64), float(d["parallax"]),
                       int(d["inliers"]), float(d["share"]), float(d["noise_deg"]),
                       bool(d["confirmed"]), bool(d["depth"]), bool(d["ambiguous"]),
                       int(d.get("hyp", 0)),
                       None if d.get("t_m") is None else np.asarray(d["t_m"], np.float64),
                       None if ag is None else (float(ag[0]), float(ag[1])),
                       d.get("measured_px"), d.get("predicted_px"))


def refined_rotation(ra: Array, rb: Array, starts: list[tuple[Array, Array]], thr: float
                     ) -> Array:
    """The relative rotation R (b's camera in a's: x_a ~ R x_b + t) of the relative pose that
    minimises the Huber-robust Sampson error of the essential matrix [t]x R over the pair's
    epipolar inliers (rays ``ra``, ``rb``, normalised image units, ``thr`` the Huber scale),
    from each start (rotation, translation direction); the lowest cost wins."""
    from scipy.optimize import least_squares

    xa, xb = np.asarray(ra, np.float64), np.asarray(rb, np.float64)

    def sampson(p: Array, R0: Array) -> Array:
        R = exp_so3(p[:3]) @ R0
        t = p[3:] / (np.linalg.norm(p[3:]) + 1e-12)
        tx = np.array([[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]])
        E = tx @ R
        Ex, Etx = xb @ E.T, xa @ E
        den = Ex[:, 0] ** 2 + Ex[:, 1] ** 2 + Etx[:, 0] ** 2 + Etx[:, 1] ** 2
        out: Array = np.sum(xa * Ex, axis=1) / np.sqrt(den + 1e-18)
        return out

    best: tuple[float, Array] | None = None
    for R0, t0 in starts:
        r = least_squares(sampson, np.r_[np.zeros(3), t0], loss="huber", f_scale=thr,
                          args=(R0,))
        if best is None or r.cost < best[0]:
            best = (float(r.cost), exp_so3(r.x[:3]) @ R0)
    assert best is not None
    return best[1]


def _fit(a: View, b: View, e: Epipolar) -> TwoView | None:
    """The two-view fit of one epipolar hypothesis ``e`` (module docstring); None without a
    rotation-only fit."""
    thr = e.thr
    da, db = unit(e.ra), unit(e.rb)
    rot = rotation_ransac(da, db, 2.0 * thr)
    if rot is None:
        return None
    R_rot = kabsch(db[rot[1]], da[rot[1]]) if int(rot[1].sum()) >= 3 else rot[0]
    ang = np.arccos(np.clip(np.sum((db @ R_rot.T) * da, axis=1), -1.0, 1.0))
    # recoverPose: x_b = R x_a + t, so R is a's camera in b's and b's centre in a's is -R^T t
    _, R_e, t_e, _ = cv2.recoverPose(e.E, e.ra[:, :2], e.rb[:, :2], np.eye(3))
    base = unit(-(np.asarray(R_e, np.float64).T @ np.asarray(t_e, np.float64).ravel()))
    # a ray's residual grows with (baseline / depth) x sin(its angle to the baseline): near the
    # epipole (a camera moving forward) it vanishes, so each residual is divided by that sine
    sin = np.sqrt(np.clip(1.0 - (da @ base) ** 2, 0.0, 1.0))
    focal = math.sqrt(a.K.fx * b.K.fx)  # the residual angle in pixels
    parallax = float(np.median(ang * focal / np.maximum(sin, SIN_MIN)))
    R_epi = np.asarray(R_e, np.float64).T
    # the relative rotation: the five-degree-of-freedom relative pose refined on the epipolar
    # inliers (``refined_rotation``). The rotation-only fit absorbs a small translation as
    # rotation (a robot turning with its camera ahead of its axis), and the essential matrix of
    # the best RANSAC sample is coarse on a small baseline: on 45 deg steps of the test room
    # both were 1.5-3.9 deg off the truth, the refined pose 0.1-0.6 deg (2.7 deg on a pair of
    # 0.55 px parallax)
    # (only a measurable pair is refined: no limit reads another's rotation)
    n = len(e.ua)
    noise = math.degrees(float(np.sqrt(np.mean(np.minimum(ang, 2.0 * thr) ** 2))))
    share = min(cell_share(e.ua, a.K), cell_share(e.ub, b.K))
    R_ab = R_epi
    if n >= CHECK_MIN_INLIERS and share >= VIEW_SHARE:
        t_a = -(np.asarray(R_e, np.float64).T @ np.asarray(t_e, np.float64).ravel())
        R_ab = refined_rotation(e.ra, e.rb, [(R_epi, t_a), (R_rot, t_a), (R_rot, -t_a)], thr)
    return TwoView(np.asarray(R_ab, np.float64), parallax, n, share,
                   noise / math.sqrt(max(n, 1)), ua=e.ua, ub=e.ub)


@dataclass
class DepthPose:
    """A hypothesis's metric relative pose (``depth_pose``): ``T_a_b``, the PnP's inliers of the
    lifted points, how it was kept on its hypothesis (``ransac``: the PnP RANSAC's own; ``local``:
    refined from the hypothesis; ``seed``: the hypothesis itself) and the PnP RANSAC's rotation
    angle (deg)."""

    T: Array
    inliers: int
    lifted: int
    mode: str
    ransac_deg: float


def depth_pose(a: View, b: View, e: Epipolar, R_ab: Array, da: DepthImage, db: DepthImage
               ) -> DepthPose | None:
    """The metric relative pose ``T_a_b`` (b's camera in a's) of a hypothesis from the pair's
    depth images: the translation from its inliers' depths (median of P_a - R P_b), then PnP
    (RANSAC at the registration threshold, seeded there) of a's depth-lifted inliers onto b's
    pixels: rotation errors of 0.02-0.3 deg where the Sampson rotation of a weak pair is off by
    up to 21 deg (test room; README "Depth confirmation"). The PnP stays on its hypothesis: when
    the RANSAC's rotation lies more than ``PNP_MAX_STEP_DEG`` from the Sampson one ``R_ab``, the
    seed is refined locally instead (Levenberg-Marquardt on the lifted inliers within the
    threshold of it), and when that still moves further, the seed itself is the pose: the
    refinement never replaces the hypothesis with another. None with fewer than six lifted
    inliers on both sides or no PnP."""
    za = sample_depth(da[0], da[1], a.K, e.ua).astype(np.float64)
    zb = sample_depth(db[0], db[1], b.K, e.ub).astype(np.float64)
    ok = np.isfinite(za) & np.isfinite(zb) & (za > MIN_DEPTH_M) & (zb > MIN_DEPTH_M)
    lifted = int(ok.sum())
    if lifted < PNP_MIN_POINTS:
        return None
    Pa, Pb, ub = e.ra[ok] * za[ok, None], e.rb[ok] * zb[ok, None], e.ub[ok]
    t = np.median(Pa - Pb @ R_ab.T, axis=0)  # x_a = R x_b + t
    r0, _ = cv2.Rodrigues(R_ab.T)
    t0 = (-R_ab.T @ t).reshape(3, 1)
    thr = REPROJ_PX * b.K.width / REF_WIDTH
    Kb = b.K.K()
    cv2.setRNGSeed(0)
    good, rv, tv, inl = cv2.solvePnPRansac(
        Pa, ub, Kb, None, r0.copy(), t0.copy(), useExtrinsicGuess=True,
        iterationsCount=PNP_ITERATIONS, reprojectionError=thr, confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not good or inl is None or len(inl) < PNP_MIN_POINTS:
        return None
    Rp = np.asarray(cv2.Rodrigues(rv)[0], np.float64)  # x_b = Rp x_a + tp
    tp = np.asarray(tv, np.float64).ravel()
    ransac_deg = rot_deg(Rp.T)
    mode, n = "ransac", len(inl)
    if rot_deg(R_ab.T @ Rp.T) > PNP_MAX_STEP_DEG:
        proj, _ = cv2.projectPoints(Pa, r0, t0, Kb, None)
        near = np.linalg.norm(proj.reshape(-1, 2) - ub, axis=1) <= thr
        mode, n = "seed", int(near.sum())
        Rp, tp = R_ab.T, t0.ravel()
        if n >= PNP_MIN_POINTS:
            rl, tl = cv2.solvePnPRefineLM(Pa[near], ub[near], Kb, None, r0.copy(), t0.copy())
            Rl = np.asarray(cv2.Rodrigues(rl)[0], np.float64)
            if rot_deg(R_ab.T @ Rl.T) <= PNP_MAX_STEP_DEG:
                mode, Rp, tp = "local", Rl, np.asarray(tl, np.float64).ravel()
    T = np.eye(4)
    T[:3, :3] = Rp.T
    T[:3, 3] = -Rp.T @ tp
    return DepthPose(T, n, lifted, mode, ransac_deg)


def _hyp_line(a: View, b: View, k: int, tv: TwoView, sampson_deg: float, dp: DepthPose | None,
              counts: tuple[DepthCounts, DepthCounts] | None, verdict: str) -> str:
    """(L1) One depth-tested hypothesis of a pair, for the service log (``{K}`` filled in once
    the pair's hypotheses are known)."""
    if dp is None:
        pnp, t = "pnp none deg", "t none"
    else:
        x, y, z = dp.T[:3, 3]
        how = "" if dp.mode == "ransac" else f", {dp.mode} from {dp.ransac_deg:.2f} deg"
        pnp = f"pnp {rot_deg(dp.T[:3, :3]):.2f} deg ({dp.inliers}/{dp.lifted} pnp inliers{how})"
        t = f"t [{x:+.3f} {y:+.3f} {z:+.3f}] m |t| {float(np.linalg.norm(dp.T[:3, 3])):.3f}"
    agree = "" if counts is None else f", a>b {counts[0].text()}, b>a {counts[1].text()}"
    return (f"two-view {a.name}-{b.name} hyp {k}/{{K}}: {tv.inliers} inliers, overlap "
            f"{tv.share:.2f}, sampson {sampson_deg:.2f} deg, {pnp}, {t}{agree}: {verdict}")


def two_view(a: View, b: View) -> TwoView | None:
    """The two-view fit of a pair (module docstring), None below ``MIN_MATCHES``. When both
    frames carry depth images, its sequential epipolar hypotheses (``epipolar_hypotheses``, in
    support order) are each refined by PnP on the depth-lifted inliers (``depth_pose``) and
    tested by the pair's depth images both ways (``depth_check``): every hypothesis that could
    make the pair measurable (``CHECK_MIN_INLIERS``), not only up to the first confirmed. The
    first confirmed is the fit (its inliers, overlap and PnP rotation), unless

    * another confirmed one lies ``SYMMETRIC_MIN_DEG`` or more from it: the pair is ambiguous
      (``TwoView.ambiguous``, map.md §1 "Symmetric scenes");
    * confirmed ones share the rotation but lie ``TWIN_MIN_M`` or more apart (translational
      twins, map.md §1, P28): the fit is the one whose predicted two-view parallax
      (``predicted_parallax`` at the pair's matches) matches the pair's measured one
      (``twin_pick``), and the pair is ambiguous when the parallax cannot separate them.

    With none confirmed, or ambiguous, the pair carries no measurement (not measurable). Each
    hypothesis and the verdict are logged (L1, L2). Without depth on either side, the largest
    consensus. The cycle rule (``consistent``) judges every pair after this."""
    # a later hypothesis only matters when it could make the pair measurable: its RANSAC on
    # fewer matches than that is not run (an adaptive RANSAC on a low inlier share is slow)
    hyps = epipolar_hypotheses(a.feats, a.K, b.feats, b.K, then=CHECK_MIN_INLIERS)
    e = next(hyps, None)
    if e is None:
        return None
    first = _fit(a, b, e)
    if first is None:
        return None
    first.measured_px = first.parallax
    da, db = a.depth_image(), b.depth_image()
    if da is None or db is None:
        return first
    first.depth = True
    e0 = e  # the pair's matches: its largest consensus
    tv: TwoView | None = first
    lines: list[str] = []
    confirmed: list[TwoView] = []
    k = 0
    # RANSAC takes the largest consensus first: once one is below the measurable threshold, so
    # are the ones the matches it left can give (``epipolar_hypotheses`` stops there)
    while e is not None and len(e.ua) >= CHECK_MIN_INLIERS:
        k += 1
        if tv is not None:
            sampson = rot_deg(tv.R_ab)
            dp = None if tv.share < VIEW_SHARE else depth_pose(a, b, e, tv.R_ab, da, db)
            counts = None
            if tv.share < VIEW_SHARE:
                verdict = "skipped overlap"
            elif dp is None:
                verdict = "rejected no pnp"
            else:
                ok, ab, ba = depth_check(DepthRef(np.eye(4), da[0], da[1].K()),
                                         DepthRef(dp.T, db[0], db[1].K()))
                counts = (ab, ba)
                verdict = "confirmed" if ok else "rejected agreeing" if min(
                    ab.agree, ba.agree) < MIN_AGREEING else "rejected agreement"
                if ok:
                    tv.R_ab, tv.t_m, tv.depth, tv.hyp = dp.T[:3, :3].copy(), \
                        dp.T[:3, 3].copy(), True, k
                    tv.agreement, tv.measured_px = (ab.ratio, ba.ratio), first.parallax
                    tv.predicted_px = predicted_parallax(e0.ua, a.K, da[0], da[1], dp.T,
                                                         math.sqrt(a.K.fx * b.K.fx))
                    confirmed.append(tv)
            lines.append(_hyp_line(a, b, k, tv, sampson, dp, counts, verdict))
        e = next(hyps, None)
        tv = None if e is None or len(e.ua) < CHECK_MIN_INLIERS else _fit(a, b, e)
    for line in lines[:LOG_HYPOTHESES]:
        log.info(line.replace("{K}", str(k)))
    return _verdict(a, b, first, confirmed)


def _px(p: float | None) -> str:
    return "none" if p is None else f"{p:.1f} px"


def _verdict(a: View, b: View, first: TwoView, confirmed: list[TwoView]) -> TwoView:
    """The pair's fit among its depth-confirmed hypotheses (``two_view``), logged (L2)."""
    pair = f"two-view {a.name}-{b.name}"
    if not confirmed:
        log.info("%s: none confirmed", pair)
        first.confirmed = False
        return first
    apart = next(((c, d) for i, c in enumerate(confirmed) for d in confirmed[i + 1:]
                  if rot_deg(c.R_ab.T @ d.R_ab) >= SYMMETRIC_MIN_DEG), None)
    if apart is not None:
        c, d = apart
        log.info("%s: ambiguous hyp %d %.1f deg / hyp %d %.1f deg (%.1f deg apart: a symmetric "
                 "scene)", pair, c.hyp, rot_deg(c.R_ab), d.hyp, rot_deg(d.R_ab),
                 rot_deg(c.R_ab.T @ d.R_ab))
        first.confirmed, first.ambiguous = False, True
        return first
    centres = [c.t_m for c in confirmed if c.t_m is not None]  # (each confirmed one has it)
    if twins(centres):
        assert first.measured_px is not None
        pick = twin_pick(first.measured_px, [c.predicted_px for c in confirmed], centres)
        about = " / ".join(f"hyp {c.hyp} at |t| {float(np.linalg.norm(p)):.2f} m predicts "
                           f"{_px(c.predicted_px)}" for c, p in zip(confirmed, centres,
                                                                    strict=True))
        if pick is None:
            log.info("%s: ambiguous translational twins (%s; measured %.1f px)", pair, about,
                     first.measured_px)
            first.confirmed, first.ambiguous = False, True
            return first
        log.info("%s: translational twins (%s; measured %.1f px): the parallax picks hyp %d",
                 pair, about, first.measured_px, confirmed[pick].hyp)
        confirmed = [confirmed[pick]]
    got = confirmed[0]
    log.info("%s: chose hyp %d at %.2f deg (%d inliers)", pair, got.hyp, rot_deg(got.R_ab),
             got.inliers)
    return got


def ambiguous(a: View, b: View, pairs: Pairs | None = None) -> bool:
    """Whether a pair is ambiguous (``TwoView.ambiguous``, map.md §1 "Symmetric scenes",
    "Translational twins"): the registration gate's rule (§2 step 2), the reference ``b`` alone
    places the frame ``a`` in neither pose. Through ``pairs`` when given (each pair fitted
    once)."""
    tv = two_view(a, b) if pairs is None else pairs.get(a, b)
    return tv is not None and tv.ambiguous


def places(a: View, b: View, T_a_b: Array, pairs: Pairs | None = None) -> bool:
    """The registration gate's pair rules (map.md §2 step 2) for a candidate pose of frame ``a``
    that reference ``b``'s depth confirmed, ``T_a_b`` (b's camera in a's): False when the pair
    is ambiguous (``ambiguous``), or when the pair's confirmed fit shares the candidate's
    rotation (within ``SYMMETRIC_MIN_DEG``) at a position ``TWIN_MIN_M`` or more away and the
    parallax picks the fit, or cannot separate the two (map.md §1 "Translational twins",
    ``twin_pick``: measured at the pair's matches, predicted by a's depth). True otherwise
    (no fit, or no depth fit to compare)."""
    tv = two_view(a, b) if pairs is None else pairs.get(a, b)
    if tv is None:
        return True
    if tv.ambiguous:
        return False
    if not tv.confirmed or tv.t_m is None or tv.ua is None or tv.measured_px is None \
            or rot_deg(T_a_b[:3, :3].T @ tv.R_ab) >= SYMMETRIC_MIN_DEG \
            or float(np.linalg.norm(T_a_b[:3, 3] - tv.t_m)) < TWIN_MIN_M:
        return True
    d = a.depth_image()
    pred = None if d is None else predicted_parallax(tv.ua, a.K, d[0], d[1], T_a_b,
                                                     math.sqrt(a.K.fx * b.K.fx))
    pick = twin_pick(tv.measured_px, [tv.predicted_px, pred], [tv.t_m, T_a_b[:3, 3]])
    log.info("%s: against %s, the candidate pose and the pair's fit are translational twins "
             "(%.2f m apart; measured %.1f px, the fit predicts %s, the candidate %s): %s",
             a.name, b.name, float(np.linalg.norm(T_a_b[:3, 3] - tv.t_m)), tv.measured_px,
             _px(tv.predicted_px), _px(pred),
             {1: "the candidate", 0: "the fit"}.get(pick if pick is not None else -1,
                                                     "ambiguous"))
    return pick == 1


class Pairs:
    """The two-view fits of an update, each pair fitted once. Stored-to-stored pairs (``stored``:
    map.md §1 "Stored-to-stored pairs", P27) are also kept in ``cache`` (the map's, by keyframe
    and content, ``AnchoredMap.pair_cache``: reused across jobs while both frames are unchanged),
    their measuring seconds in ``stored_s``."""

    def __init__(self, cache: dict[str, Any] | None = None) -> None:
        self.fits: dict[tuple[str, str], TwoView | None] = {}
        self.cache = cache
        self.stored_s = 0.0

    def get(self, a: View, b: View) -> TwoView | None:
        key = (a.name, b.name)
        if key not in self.fits:
            rev = self.fits.get((b.name, a.name), False)
            if rev is not False:
                self.fits[key] = None if rev is None else rev.reversed()
            else:
                self.fits[key] = two_view(a, b)
        return self.fits[key]

    def stored(self, a: View, b: View) -> TwoView | None:
        """A stored-to-stored pair's fit (both stored keyframes): from this job's fits, else the
        map's cache, else measured from their own images (``two_view``: never their poses),
        timed, and cached."""
        key, rev = (a.name, b.name), (b.name, a.name)
        if key in self.fits or rev in self.fits:
            return self.get(a, b)
        ck = f"{a.kid or a.name}:{a.digest()}|{b.kid or b.name}:{b.digest()}"
        if self.cache is not None and ck in self.cache:
            d = self.cache[ck]
            self.fits[key] = None if d is None else TwoView.from_json(d)
            return self.fits[key]
        t = time.perf_counter()
        got = self.get(a, b)
        self.stored_s += time.perf_counter() - t
        if self.cache is not None:
            self.cache[ck] = None if got is None else got.to_json()
        return got


def close(Ra: Array, ca: Array, Rb: Array, cb: Array) -> bool:
    """Whether two poses (rotation, centre) can share a view: centres within
    ``PAIR_MAX_DIST_M``, optical axes within ``PAIR_MAX_ANGLE_DEG``."""
    cos = float(np.clip(Ra[:, 2] @ Rb[:, 2], -1.0, 1.0))
    return float(np.linalg.norm(cb - ca)) <= PAIR_MAX_DIST_M \
        and math.degrees(math.acos(cos)) <= PAIR_MAX_ANGLE_DEG


def neighbours(views: Sequence[View], i: int) -> list[int]:
    """Every other view that view ``i`` can share a view with (``close``), in canonical name
    order: the pairs do not depend on the request order (map.md §2)."""
    v = views[i]
    near = [j for j, w in enumerate(views) if j != i and close(v.R, v.c, w.R, w.c)]
    return sorted(near, key=lambda j: (natural_key(views[j].name), views[j].stored))


@dataclass
class Judged:
    """A measurable pair the cycle rule kept (``consistent``): its views, fit, and the angle
    (deg) between its rotation and the chain of strictly stronger pairs (None: no such chain)."""

    i: int
    j: int
    tv: TwoView
    chain_deg: float | None


def judged(views: Sequence[View], pairs: Pairs) -> list[Judged]:
    """The measurable pairs among ``views`` that the standpoints and the pair limits evaluate:
    each new view with every view it can share a view with (``neighbours``), each pair once,
    after the cycle rule (``consistent``). When one of them involves a stored keyframe, the
    stored-to-stored pairs its chain may need join the rule's candidates (``stored_pairs``,
    P27), but are not returned."""
    out: list[tuple[int, int, TwoView]] = []
    seen: set[frozenset[str]] = set()
    for i, v in enumerate(views):
        if v.stored:
            continue
        for j in neighbours(views, i):
            key = frozenset((v.name, views[j].name))
            if key in seen:
                continue
            seen.add(key)
            tv = pairs.get(v, views[j])  # R_ab: j's camera in i's
            if tv is not None and tv.measurable:
                out.append((i, j, tv))
    got = _cycle(views, out + stored_pairs(views, pairs, out))
    return [g for g in got if not (views[g.i].stored and views[g.j].stored)]


def evaluated(views: Sequence[View], pairs: Pairs) -> list[tuple[int, int, TwoView]]:
    """``judged``'s pairs as (i, j, fit)."""
    return [(g.i, g.j, g.tv) for g in judged(views, pairs)]


def stored_pairs(views: Sequence[View], pairs: Pairs, cand: list[tuple[int, int, TwoView]]
                 ) -> list[tuple[int, int, TwoView]]:
    """The stored-to-stored pairs a chain of ``cand``'s pairs involving stored keyframes may
    need (map.md §1 "Stored-to-stored pairs", P27): every pair of stored keyframes that are
    ends of such a pair and can share a view (``close``), measurable, each measured from their
    own images (``Pairs.stored``: cached per version). None with fewer than two such ends."""
    ends = sorted({k for i, j, _ in cand for k in (i, j) if views[k].stored},
                  key=lambda k: natural_key(views[k].name))
    out = []
    for n, x in enumerate(ends):
        for y in ends[n + 1:]:
            v, w = views[x], views[y]
            if not close(v.R, v.c, w.R, w.c):
                continue
            tv = pairs.stored(v, w)
            if tv is not None and tv.measurable:
                out.append((x, y, tv))
    return out


def _chain(tree: dict[int, list[tuple[int, Array]]], i: int, j: int) -> Array | None:
    """The rotation of ``j``'s camera in ``i``'s composed along the forest ``tree``'s path, or
    None without one."""
    got = {i: np.eye(3)}
    todo = [i]
    while todo:
        k = todo.pop()
        for m, R in tree.get(k, []):
            if m not in got:
                got[m] = got[k] @ R
                todo.append(m)
    return got.get(j)


def consistent(views: Sequence[View], cand: list[tuple[int, int, TwoView]]
               ) -> list[tuple[int, int, TwoView]]:
    """``_cycle``'s kept pairs as (i, j, fit)."""
    return [(g.i, g.j, g.tv) for g in _cycle(views, cand)]


def _cycle(views: Sequence[View], cand: list[tuple[int, int, TwoView]]) -> list[Judged]:
    """The cycle rule (map.md §1 "Cycle consistency"; P20, P28: every pair, after the depth
    confirmation where both frames carry depth images): a pair of ``cand`` is measurable only
    when its two-view rotation agrees within ``CYCLE_MAX_DEG`` with the chain of stronger
    measured pairs, the two-view rotations composed along a path of pairs that each have
    strictly more inliers. Strongest first: each pair is judged against the pairs already kept
    that are stronger, the path that of their maximum spanning forest (ties to the canonical
    names), so the outcome depends on the pairs' inliers and names, not on the request order,
    and never on any pose. A pair with no stronger chain passes (the accepted degenerate case,
    as a lattice geometry consistent around a whole cycle), its ``chain_deg`` None: the pair
    limits read it (``outside``)."""
    parent = list(range(len(views)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree: dict[int, list[tuple[int, Array]]] = {}
    keep: dict[int, float | None] = {}
    order = sorted(range(len(cand)), key=lambda c: -cand[c][2].inliers)
    at = 0
    while at < len(order):
        n = cand[order[at]][2].inliers
        level = [c for c in order[at:] if cand[c][2].inliers == n]
        at += len(level)
        kept = []
        for c in level:
            i, j, tv = cand[c]
            R = _chain(tree, i, j)
            deg = None if R is None else rot_deg(R.T @ tv.R_ab)
            if deg is not None and deg > CYCLE_MAX_DEG:
                log.info("two-view %s-%s: %.1f deg off the chain of stronger pairs (%d "
                         "inliers, %s): not measurable", views[i].name, views[j].name, deg,
                         tv.inliers, "depth-confirmed" if tv.depth else "without depth")
                continue
            kept.append(c)
            keep[c] = deg
        # the level's pairs join the stronger ones (Kruskal, in canonical name order)
        for c in sorted(kept, key=lambda c: sorted((natural_key(views[cand[c][0]].name),
                                                    natural_key(views[cand[c][1]].name)))):
            i, j, tv = cand[c]
            if find(i) != find(j):
                parent[find(i)] = find(j)
                tree.setdefault(i, []).append((j, tv.R_ab))
                tree.setdefault(j, []).append((i, tv.R_ab.T))
    return [Judged(*cand[c], keep[c]) for c in sorted(keep)]


def outside(views: Sequence[View], got: list[Judged]) -> set[int]:
    """The pairs of ``got`` outside the pair limits (map.md §1 "Cycle consistency", P28a,
    P28b): each frame of a pair gives it a verdict over its own measurable pairs in ``got``
    (stored keyframes too; the stored-to-stored pairs a chain needed are no pair of the check):
    a frame with two or more of them, none of which has a chain of stronger pairs
    (``Judged.chain_deg`` None), puts them outside; a frame whose only pair has no chain, or
    with some chained pairs, keeps its pairs. A pair is outside only when both its frames put
    it outside, depth-confirmed or not: a sequence measured by its adjacent pairs only stays
    judged."""
    per: dict[int, list[int]] = {}
    for k, g in enumerate(got):
        for v in (g.i, g.j):
            per.setdefault(v, []).append(k)
    off = {v for v, ks in per.items()
           if len(ks) >= 2 and all(got[k].chain_deg is None for k in ks)}
    return {k for k, g in enumerate(got) if g.i in off and g.j in off}


@dataclass
class Placed:
    R: Array
    c: Array
    sigma_pos_m: float
    sigma_rot_deg: float  # the path's pair noise and the reference's (uncalibrated)
    group: int


def groups(views: Sequence[View], pairs: Pairs, ref_sigma_rot_deg: float = 0.0
           ) -> dict[str, Placed]:
    """The new frames of ``views`` that share a standpoint (module docstring), each placed as a
    pure rotation about its group's centre; frames of no group are left out."""
    n = len(views)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges: list[tuple[int, int, int, TwoView]] = []
    for i, j, tv in evaluated(views, pairs):  # R_ab: j's camera in i's
        if tv.unconstrained:
            edges.append((i, j, tv.inliers, tv))
            parent[find(i)] = find(j)
    out: dict[str, Placed] = {}
    members: dict[int, list[int]] = {}
    for i in range(n):
        members.setdefault(find(i), []).append(i)
    for gid, (_, mem) in enumerate(sorted(members.items())):
        new = [i for i in mem if not views[i].stored]
        if len(mem) < 2 or not new:
            continue
        stored = [i for i in mem if views[i].stored]
        # the overlapped stored keyframe with the most pair inliers with the group, ties to the
        # lowest keyframe id (map.md §1): its centre
        strength = {i: sum(e[2] for e in edges if i in (e[0], e[1])) for i in stored}
        root = min(stored, key=lambda i: (-strength[i], views[i].kid, views[i].name)) \
            if stored else min(new, key=lambda i: natural_key(views[i].name))
        centre = views[root].c if stored else np.mean([views[i].c for i in new], axis=0)
        spread = math.sqrt(float(np.mean([np.sum((views[i].c - centre) ** 2) for i in new])))
        # rotations along a maximum spanning tree of the group's standpoint pairs
        R = {root: views[root].R}
        var = {root: (0.0 if views[root].stored else ref_sigma_rot_deg) ** 2}
        mine = sorted((e for e in edges if e[0] in mem and e[1] in mem), key=lambda e: -e[2])
        grown = True
        while grown:
            grown = False
            for i, j, _, tv in mine:
                if j in R and i not in R:
                    R[i], var[i] = R[j] @ tv.R_ab.T, var[j] + tv.noise_deg ** 2
                    grown = True
                elif i in R and j not in R and not views[j].stored:
                    R[j], var[j] = R[i] @ tv.R_ab, var[i] + tv.noise_deg ** 2
                    grown = True
        for i in new:
            if i not in R:  # reached only through a stored keyframe other than the root
                continue
            # its rotation stays the engine's image-only one (``join_standpoint``: a two-view
            # rotation of a small baseline is biased, and the tree adds the biases up); the tree
            # gives its rotation sigma
            out[views[i].name] = Placed(views[i].R.copy(), centre.copy(), spread,
                                        math.sqrt(var[i]), gid)
    return out


@dataclass
class PairCheck:
    """One measurable pair of the rotation check (``measured``): its names, the disagreement
    (deg) between the committed and the image-measured relative rotation, its epipolar inliers,
    overlap and two-view parallax (px)."""

    a: str
    b: str
    deg: float
    inliers: int
    share: float
    parallax: float
    measured_deg: float = 0.0  # the image-measured relative rotation's angle
    committed_deg: float = 0.0  # the committed relative rotation's angle

    def row(self) -> dict[str, Any]:
        """Its ``quality.rotation_pairs`` entry (map.md §4, P14): the names in canonical
        order."""
        from oh_my_slam.mapping.anchored import natural_key

        a, b = sorted((self.a, self.b), key=natural_key)
        return {"a": a, "b": b, "inliers": int(self.inliers),
                "overlap": round(min(max(self.share, 0.0), 1.0), 4),
                "measured_deg": round(self.measured_deg, 3),
                "committed_deg": round(self.committed_deg, 3),
                "difference_deg": round(self.deg, 3)}


def consistency(views: Sequence[View], pairs: Pairs) -> tuple[float | None, float | None]:
    """The median and largest disagreement (deg) between committed and image-measured relative
    rotations over the measurable pairs of the new frames of ``views`` (with each other and
    with the stored ones, ``measured``); (None, None) without a measurable pair."""
    dis = [c.deg for c in measured(views, pairs)]
    if not dis:
        return None, None
    return float(np.median(dis)), float(max(dis))


def measured(views: Sequence[View], pairs: Pairs) -> list[PairCheck]:
    """The rotation check's measurable pairs (``TwoView.measurable``, after the cycle rule) of
    the new frames of ``views``, but those outside the pair limits (``outside``, P28b), each
    logged (L3, the service log: which pairs a ``rotation_inconsistent`` comes from, with the
    hypothesis, its shift, agreement and the angle to its chain)."""
    out: list[PairCheck] = []
    got = judged(views, pairs)
    off = outside(views, got)
    for k, g in enumerate(got):
        tv, v, w = g.tv, views[g.i], views[g.j]
        committed = v.R.T @ w.R
        c = PairCheck(v.name, w.name, rot_deg(committed.T @ tv.R_ab), tv.inliers,
                      round(tv.share, 3), round(tv.parallax, 2), rot_deg(tv.R_ab),
                      rot_deg(committed))
        shift = "n/a" if tv.t_m is None else f"{float(np.linalg.norm(tv.t_m)):.2f}"
        agree = "n/a" if tv.agreement is None else \
            f"{tv.agreement[0]:.2f}/{tv.agreement[1]:.2f}"
        chain = "n/a" if g.chain_deg is None else f"{g.chain_deg:.2f}"
        limits = ", outside the pair limits (both frames' measurable pairs have no chain)" \
            if k in off else ""
        log.info("rotation check %s-%s: %.2f deg (measured %.2f, committed %.2f, %d inliers, "
                 "overlap %.2f, parallax %.2f px, hyp %d, |t| %s m, agreement %s, chain %s deg, "
                 "%s)%s", c.a, c.b, c.deg, c.measured_deg, c.committed_deg, c.inliers, c.share,
                 c.parallax, tv.hyp, shift, agree, chain,
                 "depth-confirmed" if tv.depth else "cycle rule", limits)
        if k not in off:
            out.append(c)
    return out


def inconsistent(median_deg: float | None, max_deg: float | None) -> bool:
    """The hard limit ``rotation_inconsistent`` (map.md §4)."""
    return median_deg is not None and max_deg is not None \
        and (median_deg > MEDIAN_MAX_DEG or max_deg > PAIR_MAX_DEG)


# -- the engines' orchestration (map.md §1, §4): the stored views, the standpoint registration and
# the rotation hard limit of an update ---------------------------------------------------------


def stored_views(am: AnchoredMap, near: list[tuple[Any, Any]], exclude: list[Keyframe] | None = None
                 ) -> list[View]:
    """The map's image-confirmed keyframes (but ``exclude``) as views: those that can share a
    view (``close``) with a pose of ``near`` ((R, centre) pairs), with their features."""
    conf = [k for k in am.confirmed() if all(k is not n for n in exclude or [])]
    out = []
    for k in conf:
        if not any(close(R, c, k.T[:3, :3], k.centre) for R, c in near):
            continue
        am._load_kf(k)
        if k.feats is not None and len(k.feats.kps):
            out.append(View(k.name, k.K, k.feats, k.T[:3, :3].copy(), k.centre.copy(), True,
                            kid=k.id, depth=keyframe_depth(am, k)))
    return out


def keyframe_view(am: AnchoredMap, k: Keyframe) -> View | None:
    """A keyframe as a stored view with its features (loaded) and its depth image for the
    two-view fits; None without features."""
    am._load_kf(k)
    return None if k.feats is None else View(k.name, k.K, k.feats, k.T[:3, :3].copy(),
                                             k.centre.copy(), True, kid=k.id,
                                             depth=keyframe_depth(am, k))


def keyframe_depth(am: AnchoredMap, k: Keyframe) -> Callable[[], DepthImage | None] | None:
    """A keyframe's depth image for its two-view fits (``AnchoredMap.sensor_depth``), None for
    a keyframe without one."""
    if k.depth_kind != "sensor":
        return None
    return functools.partial(am.sensor_depth, k)


def frame_depth(f: FrameIn) -> Callable[[], DepthImage | None] | None:
    """A new frame's depth image for its two-view fits (``anchored.depth_image``: on its
    canonical pixels, its ignored pixels holding none), None without one."""
    d = depth_image(f)
    return None if d is None else (lambda: (d, f.K))


@dataclass
class NewFrame:
    """A frame without a prior registered by an engine, for ``standpoints``: its name,
    intrinsics, request index, whether it has a depth image, its image-only pose and features,
    and that depth image (metres on its grid, the grid's intrinsics) for its two-view fits."""

    name: str
    K: Intrinsics
    index: int
    depth: bool
    T: Array
    feats: Features
    image: DepthImage | None = None


def standpoints(am: AnchoredMap, new: list[NewFrame], pairs: Pairs
                ) -> dict[int, tuple[Array, float, float]]:
    """The frames of ``new`` that share a standpoint with each other or with a stored confirmed
    keyframe (``groups``, map.md §1): by request index, their pure-rotation pose about the
    group's centre and its calibrated sigmas (the position's: the group's spread)."""
    views = stored_views(am, [(n.T[:3, :3], n.T[:3, 3]) for n in new])
    views += [View(n.name, n.K, n.feats, n.T[:3, :3].copy(), n.T[:3, 3].copy(), False, n.index,
                   depth=None if n.image is None else functools.partial(lambda d: d, n.image))
              for n in new]
    got = groups(views, pairs)
    out = {}
    for n in new:
        g = got.get(n.name)
        if g is not None:
            sp, sr = sigmas.calibrated(g.sigma_pos_m, g.sigma_rot_deg, n.depth, None)
            out[n.index] = (pose(g.R, g.c), round(sp, 4), round(sr, 3))
    return out


def join_standpoint(am: AnchoredMap, pairs: Pairs, f: FrameIn, feats: Features, pl: Placement
                    ) -> Placement:
    """A frame without a prior that the prior-first engine registered, placed as one
    standpoint with the confirmed keyframe it shares one with (the most inliers, ties to the
    lowest keyframe id; map.md §1): a pure rotation about that keyframe's centre, its position
    sigma the distance of its image-only centre to it. Its rotation stays its image-only one
    (PnP on the map's lifted points): a two-view rotation of a small baseline is biased by the
    translation it absorbs (1.5-2 deg on a robot turning with its camera 10 cm ahead of its
    axis, test room), the lifted points' is not. Its displacement is the rotation between its
    committed and its image-only pose (0 here). As it was when it shares none."""
    assert pl.T is not None
    new = View(f.name, f.K, feats, pl.T[:3, :3].copy(), pl.T[:3, 3].copy(),
               depth=frame_depth(f))
    views = [*stored_views(am, [(pl.T[:3, :3], pl.T[:3, 3])]), new]
    best = None
    # its pairs with the stored keyframes after the cycle rule (map.md §1, P27, P28): the
    # stored-to-stored pairs their chains need are measured from the stored images
    for _, j, tv in evaluated(views, pairs):  # the new frame is each pair's first
        v = views[j]
        if tv.standpoint and (
                best is None or (-tv.inliers, v.kid) < (-best[1].inliers, best[0].kid)):
            best = (v, tv)
    if best is None:
        return pl
    v, tv = best
    spread = float(np.linalg.norm(pl.T[:3, 3] - v.c))
    sp = sigmas.calibrated(spread, 0.0, f.depth is not None, None)[0]
    image_T = pl.T if pl.align_T is None else pl.align_T
    T = pose(image_T[:3, :3], v.c)
    return Placement(pl.name, "registered", None, T, round(sp, 4), pl.sigma_rot_deg, pl.inliers,
                     "rotation", lifted=pl.lifted, align_T=image_T,
                     map_residual=(0.0, rot_deg(T[:3, :3].T @ image_T[:3, :3])),
                     refs=tuple(dict.fromkeys((*pl.refs, v.kid))))


def check(views: Sequence[View], pairs: Pairs) -> tuple[bool, list[dict[str, Any]]]:
    """The hard limit ``rotation_inconsistent`` (map.md §4) over the measurable pairs of the
    new frames of ``views`` (``measured``), and those pairs as ``quality.rotation_pairs``."""
    got = measured(views, pairs)
    dis = [c.deg for c in got]
    bad = bool(dis) and inconsistent(float(np.median(dis)), float(max(dis)))
    return bad, [c.row() for c in got]


def rotation_check(am: AnchoredMap, placed: list[Placement], pairs: Pairs
                   ) -> tuple[bool, list[dict[str, Any]]]:
    """The hard limit ``rotation_inconsistent`` (map.md §4) over the update's registered
    frames ``placed`` (their keyframes) and the stored confirmed keyframes they neighbour, with
    the pairs it evaluated (``check``)."""
    kfs = [k for pl in placed if pl.status == "registered" and pl.keyframe is not None
           for k in [am.keyframe(pl.keyframe)] if k is not None and k.feats is not None]
    if not kfs:
        return False, []
    views = stored_views(am, [(k.T[:3, :3], k.centre) for k in kfs], kfs)
    views += [View(k.name, k.K, k.feats, k.T[:3, :3].copy(), k.centre.copy(), False, i,
                   depth=keyframe_depth(am, k))
              for i, k in enumerate(kfs) if k.feats is not None]
    return check(views, pairs)


def views_inconsistent(new: list[View], stored: list[View], pairs: Pairs | None = None
                       ) -> tuple[bool, list[dict[str, Any]]]:
    """The hard limit ``rotation_inconsistent`` (map.md §4) for an engine that hands its
    committed poses as views (the older mapper): the new views against each other and the
    stored ones that can share a view with one of them (``close``), with the pairs evaluated
    (through ``pairs``: its stored-to-stored seconds)."""
    near = [w for w in stored if any(close(v.R, v.c, w.R, w.c) for v in new)]
    return check([*near, *new], Pairs() if pairs is None else pairs)
