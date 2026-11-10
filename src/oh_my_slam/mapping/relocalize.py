"""Locate with uncertainty (api.md §6, O5): one frame's pose in a map version, its sigmas from the
fit, and the rival hypothesis that makes it ambiguous.

The map side is a ``Index`` of keyframes: their SIFT keypoints and descriptors (``extract``), the
map-frame point of every keypoint that has a stored depth (``keyframe_points``), their poses and
sigmas. A frame is matched to the keyframes (all of them up to ``MATCH_ALL``; else the
``TOP_K`` with most nearest-neighbour votes, plus those near its prior), ratio test
``RATIO``. Then:

* **2-D/3-D fit.** Its keypoints matched to keyframe points give a RANSAC PnP
  (``PX_THRESHOLD`` px at 640 px of width) refined on the inliers. Its covariance is the
  Gauss-Newton one, ``s^2 (J^T J)^-1`` over the inliers' reprojection residuals (``s`` their RMS,
  at least ``MIN_PX_SIGMA``), for the rotation and the camera centre; ``sigma_pos_m`` and
  ``sigma_rot_deg`` are ``SIGMA_SCALE`` times the square roots of their traces (the 3-D RMS),
  combined with the matched keyframes' own sigmas.
* **Turn fit.** Without enough of those, a frame whose matched keyframes were taken from one spot
  (centres within ``ONE_SPOT_REL`` of their depth, or ``ONE_SPOT_M``) is placed at their centre with
  the rotation that turns its rays onto theirs (``locate.turn_ransac``).
* **Rival.** The best other hypothesis: a second fit without the inliers of the first, and a fit
  on each of the ``RIVAL_KEYFRAMES`` keyframes with most correspondences alone. A hypothesis is a
  rival when it lies farther than ``AMBIGUITY_POS_M`` or ``AMBIGUITY_ROT_DEG`` from the pose and
  most of its inliers are not the pose's (distinct evidence); the frame is ambiguous when the
  rival's inliers are at least ``AMBIGUITY_SUPPORT_RATIO`` of the pose's.

A prior picks between the pose and its rival when only one agrees with it (``consistent``), as
does the frame's height above the floor."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.mapping.locate import kabsch, turn_ransac

Array = NDArray[np.float64]

MAX_FEATURES = 2000
WORK_SIDE = 1024  # features on the image scaled to at most this side
SIFT_CONTRAST = 0.01  # a quarter of OpenCV's default: simulated and indoor robot frames are low
FEATURES_TAG = "c01"  # in the derived feature cache's keys: features of these parameters
RATIO = 0.8
MATCH_ALL = 24
TOP_K = 12
VOTE_DIST = 230.0  # SIFT L2 distance of a nearest neighbour that votes
PRIOR_RADIUS_M = 1.0
PRIOR_KEYFRAMES = 6
MIN_PAIR_MATCHES = 8
MIN_INLIERS = 15
PX_THRESHOLD = 4.0
RANSAC_ITERATIONS = 2000
MIN_PX_SIGMA = 0.3
SIGMA_SCALE = 1.0  # README "Locate": calibrated against replay truth (O5)
ONE_SPOT_REL = 0.02
ONE_SPOT_M = 0.05
TURN_MIN_SHARE = 0.25
RIVAL_KEYFRAMES = 6
AMBIGUITY_POS_M = 0.10
AMBIGUITY_ROT_DEG = 5.0
AMBIGUITY_SUPPORT_RATIO = 0.6
FAR_POS_M = 0.3  # a rival lies at least this far, or is turned this much (``tolerance``)
FAR_ROT_DEG = 15.0
K_SIGMA = 3.0
DEFAULT_PRIOR_SIGMA_POS_M = 0.05
DEFAULT_PRIOR_SIGMA_ROT_DEG = 2.0
DEFAULT_HEIGHT_SIGMA_M = 0.02
EDGE_REL = 0.04
DEPTH_SIGMA_REL = 0.05  # a keyframe point's depth, relative (README "Locate")
PIXEL_SIGMA = 1.0  # a keypoint's pixel noise at 640 px
HUBER_SIGMAS = 2.0
REFINE_ITERATIONS = 8
BEHIND_SIGMAS = 1e3


# ------------------------------------------------------------------------------------------------
# features


def work_scale(width: int, height: int) -> float:
    return min(1.0, WORK_SIDE / max(width, height))


def extract(gray: NDArray[np.uint8], mask: NDArray[np.bool_] | None = None
            ) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
    """SIFT keypoints (pixels of ``gray``) and descriptors, none on ``mask`` pixels, computed on
    the image scaled to ``WORK_SIDE``."""
    import cv2

    h, w = gray.shape
    s = work_scale(w, h)
    img = gray if s >= 1.0 else cv2.resize(gray, (round(w * s), round(h * s)),
                                           interpolation=cv2.INTER_AREA)
    keep = None
    if mask is not None and mask.any():
        keep = (cv2.resize((~mask).astype(np.uint8), (img.shape[1], img.shape[0]),
                           interpolation=cv2.INTER_NEAREST) * 255)
    sift: Any = cv2.SIFT_create(nfeatures=MAX_FEATURES,  # type: ignore[attr-defined]
                                contrastThreshold=SIFT_CONTRAST)
    kps, desc = sift.detectAndCompute(img, keep)
    if desc is None or not kps:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8)
    uv = (np.array([k.pt for k in kps], np.float32) + 0.5) / s - 0.5
    return uv.astype(np.float32), np.clip(np.round(desc), 0, 255).astype(np.uint8)


def keyframe_points(uv: NDArray[Any], K: Array, depth: NDArray[Any] | None, T_map_cam: Array
                    ) -> NDArray[np.float32]:
    """Map-frame points of keypoints ``uv`` (pixels of an image of intrinsics ``K``) from a depth
    grid in metres (z along the optical axis; any size: nearest cell), off depth edges; NaN where
    there is none."""
    from oh_my_slam.core.geometry import depth_edge_mask, unproject_pixels

    out = np.full((len(uv), 3), np.nan, np.float32)
    if depth is None or not len(uv):
        return out
    d = np.asarray(depth, np.float64)
    gh, gw = d.shape
    W, H = 2 * K[0, 2], 2 * K[1, 2]
    col = np.clip(np.floor((uv[:, 0] + 0.5) * gw / W).astype(int), 0, gw - 1)
    row = np.clip(np.floor((uv[:, 1] + 0.5) * gh / H).astype(int), 0, gh - 1)
    ok_grid = np.isfinite(d) & (d > 0)
    ok_grid &= ~depth_edge_mask(np.where(ok_grid, d, 0.0), EDGE_REL)
    z = d[row, col]
    ok = ok_grid[row, col]
    cam = unproject_pixels(uv[:, 0], uv[:, 1], z, K)
    T = np.asarray(T_map_cam, np.float64)
    out[ok] = (cam[ok] @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
    return out


@dataclass
class Keyframe:
    name: str
    T: Array  # 4x4 T_map_cam
    K: Array  # 3x3, of the keypoints' pixels
    uv: NDArray[np.float32]
    desc: NDArray[np.uint8]
    xyz: NDArray[np.float32]  # map frame, NaN without depth
    sigma_pos_m: float = 0.0
    sigma_rot_deg: float = 0.0

    @property
    def centre(self) -> Array:
        return self.T[:3, 3]

    def view_point(self) -> Array:
        """Where it looks: the mean of its keypoints' points, else 1 m along its optical axis."""
        ok = np.isfinite(self.xyz).all(axis=1)
        if ok.any():
            return np.asarray(self.xyz[ok].mean(axis=0), np.float64)
        return self.centre + self.T[:3, 2]


@dataclass
class Index:
    keyframes: list[Keyframe]
    _flann: Any = None
    _owner: NDArray[np.int64] | None = None

    def votes(self, desc: NDArray[np.uint8]) -> NDArray[np.int64]:
        """Per keyframe, how many of ``desc`` have their nearest keyframe descriptor in it
        within ``VOTE_DIST``."""
        import cv2

        n = len(self.keyframes)
        if not len(desc) or not n:
            return np.zeros(n, np.int64)
        if self._owner is None:
            all_d = [k.desc for k in self.keyframes]
            self._owner = np.concatenate([np.full(len(d), i) for i, d in enumerate(all_d)])
            if len(self._owner):
                self._flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 32})
                self._flann.add([np.concatenate(all_d).astype(np.float32)])
                self._flann.train()
        if self._flann is None:
            return np.zeros(n, np.int64)
        m = self._flann.match(desc.astype(np.float32))
        idx = np.array([x.trainIdx for x in m if x.distance < VOTE_DIST], np.int64)
        return np.bincount(self._owner[idx], minlength=n) if idx.size else np.zeros(n, np.int64)


# ------------------------------------------------------------------------------------------------
# geometry


def _rot_deg(A: Array, B: Array) -> float:
    c = (float(np.trace(A.T @ B)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def pose_distance(A: Array, B: Array) -> tuple[float, float]:
    return float(np.linalg.norm(A[:3, 3] - B[:3, 3])), _rot_deg(A[:3, :3], B[:3, :3])


def yaw_deg(T: Array) -> float:
    """Heading of a camera: its optical axis in the horizontal plane, CCW from map +x."""
    z = np.asarray(T, np.float64)[:3, 2]
    return math.degrees(math.atan2(z[1], z[0]))


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _skew(w: Array) -> Array:
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def _exp(w: Array) -> Array:
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3) + _skew(w)
    k = _skew(w / th)
    return np.eye(3) + math.sin(th) * k + (1 - math.cos(th)) * k @ k


def project(T: Array, K: Array, xyz: NDArray[Any]) -> tuple[Array, Array]:
    """(pixels (N, 2), depth along the optical axis) of map points seen by camera ``T``."""
    cam = (np.asarray(xyz, np.float64) - T[:3, 3]) @ T[:3, :3]
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = cam[:, :2] / z[:, None] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    return uv, z


def inliers_of(T: Array, K: Array, uv: NDArray[Any], xyz: NDArray[Any], px: float
               ) -> NDArray[np.bool_]:
    p, z = project(T, K, xyz)
    with np.errstate(invalid="ignore"):
        return (z > 0) & (np.hypot(*(p - uv).T) <= px)


def covariance(T: Array, K: Array, uv: NDArray[Any], xyz: NDArray[Any]) -> tuple[float, float]:
    """(3-D RMS of the centre in m, of the rotation in deg) of the Gauss-Newton covariance of the
    pose ``T`` fitted to correspondences (its inliers): ``s^2 (J^T J)^-1``, ``s`` the RMS
    reprojection residual (at least ``MIN_PX_SIGMA``)."""
    n = len(uv)
    R0, C0 = T[:3, :3], T[:3, 3]

    def res(p: Array) -> Array:
        Tp = np.eye(4)
        Tp[:3, :3] = _exp(p[:3]) @ R0
        Tp[:3, 3] = C0 + p[3:]
        q, _ = project(Tp, K, xyz)
        return np.asarray((q - uv).ravel(), np.float64)

    r0 = res(np.zeros(6))
    J = np.zeros((2 * n, 6))
    for i, eps in enumerate((1e-6,) * 3 + (1e-5,) * 3):
        e = np.zeros(6)
        e[i] = eps
        J[:, i] = (res(e) - res(-e)) / (2 * eps)
    dof = max(2 * n - 6, 1)
    s2 = max(float(r0 @ r0) / dof, MIN_PX_SIGMA ** 2)
    try:
        cov = s2 * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        return math.inf, math.inf
    rot = math.degrees(math.sqrt(max(float(np.trace(cov[:3, :3])), 0.0)))
    pos = math.sqrt(max(float(np.trace(cov[3:, 3:])), 0.0))
    return pos, rot


def pnp(uv: NDArray[Any], xyz: NDArray[Any], K: Array, px: float, seed: int = 0
        ) -> tuple[Array, NDArray[np.bool_]] | None:
    """RANSAC PnP refined on its inliers: (T_map_cam, inliers), None below ``MIN_INLIERS``."""
    import cv2

    if len(uv) < max(MIN_INLIERS, 6):
        return None
    cv2.setRNGSeed(seed)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        np.ascontiguousarray(xyz, np.float64), np.ascontiguousarray(uv, np.float64), K, None,
        iterationsCount=RANSAC_ITERATIONS, reprojectionError=px, confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < MIN_INLIERS:
        return None
    idx = inl.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(np.ascontiguousarray(xyz[idx], np.float64),
                                      np.ascontiguousarray(uv[idx], np.float64), K, None,
                                      rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ tvec.ravel()
    return T, inliers_of(T, K, uv, xyz, px)


# ------------------------------------------------------------------------------------------------
# a frame


@dataclass
class Query:
    uv: NDArray[np.float32]
    desc: NDArray[np.uint8]
    K: Array
    width: int
    height: int
    prior: Array | None = None  # T_map_cam
    prior_sigma_pos_m: float = DEFAULT_PRIOR_SIGMA_POS_M
    prior_sigma_rot_deg: float = DEFAULT_PRIOR_SIGMA_ROT_DEG
    camera_z: float | None = None  # expected map z of the camera (floor + height above it)
    camera_z_sigma: float = DEFAULT_HEIGHT_SIGMA_M


@dataclass
class Hypothesis:
    T: Array
    inliers: int
    mask: NDArray[np.bool_]


@dataclass
class Outcome:
    T: Array | None = None
    sigma_pos_m: float | None = None
    sigma_rot_deg: float | None = None
    inlier_ratio: float | None = None
    inliers: int = 0
    ambiguity: dict[str, float] | None = None
    consistent_with_prior: bool | None = None
    method: str = ""
    nearest: int | None = None  # keyframe index most similar by votes (for no_overlap)
    votes: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, np.int64))

    @property
    def located(self) -> bool:
        return self.T is not None


def _matches(q: Query, kf: Keyframe) -> NDArray[np.int64]:
    """(M, 2) index pairs (query, keyframe) passing the ratio test."""
    import cv2

    if len(q.desc) < 2 or len(kf.desc) < 2:
        return np.zeros((0, 2), np.int64)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    pairs = bf.knnMatch(q.desc.astype(np.float32), kf.desc.astype(np.float32), k=2)
    out = [(m.queryIdx, m.trainIdx) for m, n in (p for p in pairs if len(p) == 2)
           if m.distance < RATIO * n.distance]
    return np.array(out, np.int64).reshape(-1, 2)


# Keyframe points from the keyframes' own geometry: each keyframe's keypoints are matched to those of
# its TRIANGULATE_NEIGHBOURS nearest keyframes whose baseline is at least TRIANGULATE_BASELINE_REL of
# its scene depth (optical axes within TRIANGULATE_MAX_ANGLE_DEG), triangulated with the stored poses,
# and kept when both reprojections are within TRIANGULATE_PX (at 640 px) and the rays meet at
# TRIANGULATE_MIN_RAY_DEG or more. Those points agree with the keyframe poses; the depth points of
# the other keypoints are scaled to them (median ratio over at least DEPTH_SCALE_MIN_POINTS).
TRIANGULATE_NEIGHBOURS = 4
TRIANGULATE_BASELINE_REL = 0.05
TRIANGULATE_MAX_ANGLE_DEG = 60.0
TRIANGULATE_PX = 2.0
TRIANGULATE_MIN_RAY_DEG = 2.0
DEPTH_SCALE_MIN_POINTS = 20
TRIANGULATE_SCALE_RANGE = (0.67, 1.5)  # a keyframe's triangulation against its depth
TRIANGULATE_DEPTH_TOL = 0.5  # a point this far (relative) from its depth is dropped
PLAUSIBLE_NEAR_M = 2.0  # a located pose at most this far from every keyframe...
PLAUSIBLE_BOUNDS_M = 1.0  # ... and inside the keyframes' bounds grown by this
PLAUSIBLE_KEYFRAMES = 2  # inliers on this many keyframes, or twice MIN_INLIERS on one


def _depth_of(kf_xyz: NDArray[Any], T: Array) -> Array:
    """Z along the optical axis of map points (NaN where none)."""
    return np.asarray(((np.asarray(kf_xyz, np.float64) - T[:3, 3]) @ T[:3, :3])[:, 2])


def triangulate(kfs: Sequence[Keyframe], pair_matches: Any = None) -> list[NDArray[np.float32]]:
    """Each keyframe's keypoint points (see above): triangulated where possible, else its depth
    points scaled to those. ``pair_matches(i, j)`` gives the ratio-test matches of two keyframes
    (default: computed); a cache may stand in for it."""
    import cv2

    n = len(kfs)
    out = [k.xyz.copy() for k in kfs]
    if n < 2:
        return out
    C = np.array([k.centre for k in kfs])
    depth = []
    for k in kfs:
        z = _depth_of(k.xyz, k.T)
        z = z[np.isfinite(z) & (z > 0)]
        depth.append(float(np.median(z)) if z.size else 2.0)
    match = pair_matches or (lambda i, j: _pair(kfs[i], kfs[j]))
    for i, a in enumerate(kfs):
        d = np.linalg.norm(C - C[i], axis=1)
        cand = [j for j in np.argsort(d) if j != i and d[j] >= TRIANGULATE_BASELINE_REL * depth[i]
                and float(a.T[:3, 2] @ kfs[j].T[:3, 2]) >= math.cos(
                    math.radians(TRIANGULATE_MAX_ANGLE_DEG))][:TRIANGULATE_NEIGHBOURS]
        pts: list[list[Array]] = [[] for _ in range(len(a.uv))]
        for j in cand:
            b = kfs[j]
            m = match(i, int(j))
            if len(m) < MIN_PAIR_MATCHES:
                continue
            Pa = a.K @ np.linalg.inv(a.T)[:3]
            Pb = b.K @ np.linalg.inv(b.T)[:3]
            X = cv2.triangulatePoints(Pa, Pb, a.uv[m[:, 0]].T.astype(np.float64),
                                      b.uv[m[:, 1]].T.astype(np.float64))
            X = (X[:3] / X[3]).T
            ok = np.isfinite(X).all(axis=1)
            px = TRIANGULATE_PX * max(2 * a.K[0, 2], 2 * a.K[1, 2]) / 640.0
            for kf, uvs in ((a, a.uv[m[:, 0]]), (b, b.uv[m[:, 1]])):
                p, z = project(kf.T, kf.K, X)
                with np.errstate(invalid="ignore"):
                    ok &= (z > 0) & (np.hypot(*(p - uvs).T) <= px)
            ra, rb = X - a.centre, X - b.centre
            with np.errstate(invalid="ignore"):
                cosang = np.sum(ra * rb, axis=1) / (np.linalg.norm(ra, axis=1)
                                                    * np.linalg.norm(rb, axis=1))
            ok &= cosang <= math.cos(math.radians(TRIANGULATE_MIN_RAY_DEG))
            for k_a, x in zip(m[ok, 0], X[ok], strict=True):
                pts[int(k_a)].append(x)
        tri = np.array([np.median(p, axis=0) if p else [np.nan] * 3 for p in pts],
                       np.float64).reshape(-1, 3)
        has = np.isfinite(tri).all(axis=1)
        if not has.any():
            continue
        z_d = _depth_of(a.xyz, a.T)
        z_t = _depth_of(tri, a.T)
        both = has & np.isfinite(z_d) & (z_d > 0)
        scale = float(np.median(z_t[both] / z_d[both])) \
            if int(both.sum()) >= DEPTH_SCALE_MIN_POINTS else 1.0
        if not TRIANGULATE_SCALE_RANGE[0] <= scale <= TRIANGULATE_SCALE_RANGE[1]:
            # the triangulation disagrees with the keyframe's own depth as a whole (poses that
            # the images did not confirm, rays too parallel): the depth points stand
            continue
        # a triangulated point far from the keyframe's own depth is a bad match or ray
        with np.errstate(invalid="ignore"):
            off = np.isfinite(z_d) & (z_d > 0) & (np.abs(z_t / (scale * z_d) - 1.0)
                                                  > TRIANGULATE_DEPTH_TOL)
        has &= ~off
        fill = a.centre + scale * (np.asarray(a.xyz, np.float64) - a.centre)
        out[i] = np.where(has[:, None], tri, fill).astype(np.float32)
    return out


def _pair(a: Keyframe, b: Keyframe) -> NDArray[np.int64]:
    """Ratio-test matches (index pairs) of two keyframes' descriptors."""
    q = Query(a.uv, a.desc, a.K, 0, 0)
    return _matches(q, b)


def candidates(q: Query, index: Index) -> tuple[list[int], NDArray[np.int64]]:
    """(keyframes to match, votes per keyframe)."""
    n = len(index.keyframes)
    votes = index.votes(q.desc)
    if n <= MATCH_ALL:
        chosen = list(range(n))
    else:
        chosen = [int(i) for i in np.argsort(-votes, kind="stable")[:TOP_K] if votes[i] > 0]
    if q.prior is not None:
        C = q.prior[:3, 3]
        radius = max(PRIOR_RADIUS_M, K_SIGMA * q.prior_sigma_pos_m + 0.5)
        near = sorted((float(np.linalg.norm(k.centre - C)), i) for i, k in enumerate(index.keyframes)
                      if float(np.linalg.norm(k.centre - C)) <= radius
                      and float(k.T[:3, 2] @ q.prior[:3, 2]) > -0.2)
        for _, i in near[:PRIOR_KEYFRAMES]:
            if i not in chosen:
                chosen.append(i)
    return chosen, votes


def consistent(q: Query, T: Array, sp: float, sr: float) -> bool | None:
    """Whether a pose lies within ``K_SIGMA`` of the prior (sigmas combined), None without one."""
    if q.prior is None:
        return None
    dp, dr = pose_distance(T, q.prior)
    return bool(dp <= K_SIGMA * math.hypot(q.prior_sigma_pos_m, sp)
                and dr <= K_SIGMA * math.hypot(q.prior_sigma_rot_deg, sr))


def _height_ok(q: Query, T: Array, sp: float) -> bool:
    if q.camera_z is None:
        return True
    return abs(float(T[2, 3]) - q.camera_z) <= K_SIGMA * math.hypot(q.camera_z_sigma, sp) + 0.05


def _first(h: Hypothesis, iq: NDArray[np.int64]) -> dict[int, int]:
    """Query keypoint -> its first inlier correspondence."""
    out: dict[int, int] = {}
    for c in np.flatnonzero(h.mask):
        out.setdefault(int(iq[c]), int(c))
    return out


def distinct(h: Hypothesis, main: Hypothesis, tol: tuple[float, float]) -> bool:
    """Whether ``h`` is another place than ``main``: farther than ``tol`` (m, deg) from it
    (``tolerance``), such as a symmetric room's twin view, or keyframes that disagree that much
    about the frame. Nearer hypotheses are the same view through keyframes whose depth or pose
    are a little off: they widen the sigmas instead (``spread``)."""
    dp, dr = pose_distance(h.T, main.T)
    return dp > tol[0] or dr > tol[1]


def tolerance(fit: tuple[float, float]) -> tuple[float, float]:
    """How far (m, deg) another hypothesis must lie to be another place: ``FAR_POS_M`` or
    ``FAR_ROT_DEG``, or ``K_SIGMA`` times the pose's own sigmas (fit and keyframes, ``fit``)
    when larger."""
    return max(FAR_POS_M, K_SIGMA * fit[0]), max(FAR_ROT_DEG, K_SIGMA * fit[1])


def _rival(best: Hypothesis, others: Sequence[Hypothesis], tol: tuple[float, float]
           ) -> Hypothesis | None:
    """The best-supported hypothesis farther than ``AMBIGUITY_POS_M`` or ``AMBIGUITY_ROT_DEG``
    from ``best`` that is another place (``distinct``)."""
    out = None
    for h in others:
        dp, dr = pose_distance(h.T, best.T)
        if dp <= AMBIGUITY_POS_M and dr <= AMBIGUITY_ROT_DEG:
            continue
        if not distinct(h, best, tol):
            continue
        if out is None or h.inliers > out.inliers:
            out = h
    return out


def spread(main: Hypothesis, others: Sequence[Hypothesis], tol: tuple[float, float]
           ) -> tuple[float, float]:
    """(m, deg): the RMS distance from ``main`` of the hypotheses of the same place, each
    weighted by its support relative to ``main``'s, squared (``main`` counts once, at 0)."""
    w_sum, dp2, dr2 = 1.0, 0.0, 0.0
    for h in others:
        if distinct(h, main, tol):
            continue
        w = (h.inliers / max(main.inliers, 1)) ** 2
        dp, dr = pose_distance(h.T, main.T)
        w_sum += w
        dp2 += w * dp * dp
        dr2 += w * dr * dr
    return math.sqrt(dp2 / w_sum), math.sqrt(dr2 / w_sum)


def locate(q: Query, index: Index) -> Outcome:
    """The pose of a frame in the index's map (see the module docstring)."""
    chosen, votes = candidates(q, index)
    out = Outcome(votes=votes)
    if votes.size and votes.max() > 0:
        out.nearest = int(np.argmax(votes))
    px = PX_THRESHOLD * max(q.width, q.height) / 640.0
    iq_l, xyz_l, kf_l, ray_q, ray_k, ray_kf = [], [], [], [], [], []
    per_kf: list[tuple[int, int]] = []
    for i in chosen:
        kf = index.keyframes[i]
        m = _matches(q, kf)
        if len(m) < MIN_PAIR_MATCHES:
            continue
        ray_q.append(m[:, 0])
        ray_k.append(kf.uv[m[:, 1]])
        ray_kf.append(np.full(len(m), i))
        pts = kf.xyz[m[:, 1]]
        ok = np.isfinite(pts).all(axis=1)
        iq_l.append(m[ok, 0])
        xyz_l.append(pts[ok])
        kf_l.append(np.full(int(ok.sum()), i))
        per_kf.append((int(ok.sum()), i))
    # every correspondence (a query keypoint may match points of several keyframes: a symmetric
    # room's twin views give each its own hypothesis); a pose's support counts query keypoints
    iq = np.concatenate(iq_l) if iq_l else np.zeros(0, np.int64)
    xyz = np.concatenate(xyz_l).astype(np.float64) if xyz_l else np.zeros((0, 3))
    kfs = np.concatenate(kf_l) if kf_l else np.zeros(0, np.int64)
    uv = np.asarray(q.uv, np.float64)[iq].reshape(-1, 2)

    def hypothesis(T: Array) -> Hypothesis:
        mask = inliers_of(T, q.K, uv, xyz, px)
        return Hypothesis(T, int(np.unique(iq[mask]).size), mask)

    pooled = pnp(uv, xyz, q.K, px)
    if pooled is not None and hypothesis(pooled[0]).inliers >= MIN_INLIERS:
        hyps = [hypothesis(pooled[0])]
        second = pnp(uv[~hyps[0].mask], xyz[~hyps[0].mask], q.K, px, seed=1)
        if second is not None:
            hyps.append(hypothesis(second[0]))
        for _, i in sorted(per_kf, reverse=True)[:RIVAL_KEYFRAMES]:
            sel = kfs == i
            alone = pnp(uv[sel], xyz[sel], q.K, px, seed=2)
            if alone is not None:
                hyps.append(hypothesis(alone[0]))
        k_best = max(range(len(hyps)), key=lambda k: (hyps[k].inliers, -k))
        main = hypothesis(_refine(hyps[k_best], q.K, uv, xyz, iq))
        origin = np.array([index.keyframes[int(i)].centre for i in kfs]).reshape(-1, 3)
        main, cov = _refined(main, hypothesis, q, uv, xyz, origin, px)
        others = [h for k, h in enumerate(hyps) if k != k_best]
        kf_sig = _keyframe_sigmas(main, kfs, index)
        tol = tolerance(_fit(main, q, uv, xyz, iq, cov, kf_sig))
        rival = _rival(main, others, tol)
        while rival is not None and _same_basin(rival, main, q, uv, xyz, origin, px, tol):
            others = [h for h in others if h is not rival]  # the same place, seen off
            rival = _rival(main, others, tol)
        if rival is not None:  # the prior, then the height, pick when only one agrees
            sp, sr = _sigmas(main, others, q, uv, xyz, iq, cov, kf_sig)
            rsp, rsr = _sigmas(rival, [main, *others], q, uv, xyz, iq,
                               keyframes=_keyframe_sigmas(rival, kfs, index))
            if (consistent(q, main.T, sp, sr) is False and consistent(q, rival.T, rsp, rsr)) or (
                    not _height_ok(q, main.T, sp) and _height_ok(q, rival.T, rsp)):
                others = [main, *[h for h in others if h is not rival]]
                main, rival = rival, main
                main, cov = _refined(main, hypothesis, q, uv, xyz, origin, px)
                kf_sig = _keyframe_sigmas(main, kfs, index)
        if not plausible(main, kfs, index):
            return out  # a degenerate solve: not located (a wrong pose is worse than none)
        sp, sr = _sigmas(main, others, q, uv, xyz, iq, cov, kf_sig)
        out.T, out.sigma_pos_m, out.sigma_rot_deg = main.T, sp, sr
        out.inliers = main.inliers
        out.inlier_ratio = round(main.inliers / int(np.unique(iq).size), 4)
        out.method = "points"
        if rival is not None and rival.inliers >= AMBIGUITY_SUPPORT_RATIO * main.inliers:
            dp, dr = pose_distance(rival.T, main.T)
            out.ambiguity = {"pos_m": round(dp, 4), "rot_deg": round(dr, 3)}
        out.consistent_with_prior = consistent(q, main.T, sp, sr)
        return out
    if ray_q:
        turned = _turn(q, index, np.concatenate(ray_q), np.concatenate(ray_k),
                       np.concatenate(ray_kf))
        if turned is not None:
            return _with_votes(turned, out)
    return out


def plausible(h: Hypothesis, kfs: NDArray[np.int64], index: Index) -> bool:
    """Whether a solve can be the frame's pose: within ``PLAUSIBLE_NEAR_M`` of a keyframe and
    inside the keyframes' bounds grown by ``PLAUSIBLE_BOUNDS_M``, its inliers on
    ``PLAUSIBLE_KEYFRAMES`` keyframes (or twice ``MIN_INLIERS`` on one)."""
    C = np.array([k.centre for k in index.keyframes])
    c = h.T[:3, 3]
    if float(np.linalg.norm(C - c, axis=1).min()) > PLAUSIBLE_NEAR_M:
        return False
    if (c < C.min(0) - PLAUSIBLE_BOUNDS_M).any() or (c > C.max(0) + PLAUSIBLE_BOUNDS_M).any():
        return False
    per = np.bincount(kfs[h.mask], minlength=len(index.keyframes))
    return int((per >= MIN_PAIR_MATCHES).sum()) >= PLAUSIBLE_KEYFRAMES \
        or int(per.max()) >= 2 * MIN_INLIERS


def _with_votes(o: Outcome, base: Outcome) -> Outcome:
    o.votes, o.nearest = base.votes, base.nearest
    return o


def _refine(h: Hypothesis, K: Array, uv: Array, xyz: Array, iq: NDArray[np.int64]) -> Array:
    """A hypothesis' pose refined (Levenberg-Marquardt) on its inliers, one per query keypoint."""
    import cv2

    sel = np.array(sorted(_first(h, iq).values()), np.int64)
    Rcw = h.T[:3, :3].T
    rvec0, _ = cv2.Rodrigues(Rcw)
    tvec0 = (-Rcw @ h.T[:3, 3]).reshape(3, 1)
    rvec, tvec = cv2.solvePnPRefineLM(np.ascontiguousarray(xyz[sel]),
                                      np.ascontiguousarray(uv[sel]), K, None, rvec0, tvec0)
    Rr = np.asarray(cv2.Rodrigues(rvec)[0], np.float64)
    T = np.eye(4)
    T[:3, :3] = Rr.T
    T[:3, 3] = -Rr.T @ np.asarray(tvec, np.float64).ravel()
    return T


def _whitened(T: Array, K: Array, uv: Array, xyz: Array, origin: Array, px: float,
              rel: float) -> Array:
    """(N, 2) residuals of correspondences in units of their sigmas: a map point is the
    keyframe's ray (from ``origin``, the keyframe's centre) at its depth, uncertain by ``rel`` of
    that distance along the ray; its projection is uncertain by ``px`` pixels and, along the image
    of that ray (the epipolar line), by the depth's share too."""
    q, z = project(T, K, xyz)
    r = q - uv
    e = xyz - origin
    dist = np.linalg.norm(e, axis=1)
    e = e / np.maximum(dist, 1e-9)[:, None]
    cam = (xyz - T[:3, 3]) @ T[:3, :3]
    dcam = e @ T[:3, :3]
    with np.errstate(divide="ignore", invalid="ignore"):
        j = np.column_stack([K[0, 0] * (dcam[:, 0] * cam[:, 2] - cam[:, 0] * dcam[:, 2]),
                             K[1, 1] * (dcam[:, 1] * cam[:, 2] - cam[:, 1] * dcam[:, 2])]) \
            / (cam[:, 2] ** 2)[:, None]
    jn = np.linalg.norm(j, axis=1)
    u = j / np.maximum(jn, 1e-12)[:, None]
    along = np.sum(r * u, axis=1)
    perp = r[:, 0] * u[:, 1] - r[:, 1] * u[:, 0]
    s_along = np.sqrt(px ** 2 + (rel * dist * jn) ** 2)
    out = np.column_stack([perp / px, along / s_along])
    out[~np.isfinite(out).all(axis=1) | (z <= 0)] = BEHIND_SIGMAS
    return out


def refine_keyframes(T0: Array, K: Array, uv: Array, xyz: Array, origin: Array, px: float,
                     rel: float = DEPTH_SIGMA_REL) -> tuple[Array, Array, float]:
    """The pose refined on every inlier correspondence of every matched keyframe (a query
    keypoint seen in several keyframes constrains it once per keyframe), each weighed by its own
    uncertainty (``_whitened``): across the epipolar line a correspondence is as precise as its
    pixels, along it as its keyframe's depth. Gauss-Newton with a Huber weight
    (``HUBER_SIGMAS``), ``REFINE_ITERATIONS`` steps. Returns (pose, 6x6 covariance of the
    rotation (rad) and centre (m), RMS whitened residual)."""
    R0, C0 = T0[:3, :3].copy(), T0[:3, 3].copy()

    def pose(p: Array) -> Array:
        T = np.eye(4)
        T[:3, :3] = _exp(p[:3]) @ R0
        T[:3, 3] = C0 + p[3:]
        return T

    p = np.zeros(6)
    eps = np.array([1e-6] * 3 + [1e-5] * 3)
    J = np.zeros((2 * len(uv), 6))
    w = np.ones(2 * len(uv))
    for _ in range(REFINE_ITERATIONS):
        r0 = _whitened(pose(p), K, uv, xyz, origin, px, rel).ravel()
        for i in range(6):
            e = np.zeros(6)
            e[i] = eps[i]
            J[:, i] = (_whitened(pose(p + e), K, uv, xyz, origin, px, rel).ravel()
                       - _whitened(pose(p - e), K, uv, xyz, origin, px, rel).ravel()) / (2 * eps[i])
        a = np.abs(r0)
        w = np.where(a <= HUBER_SIGMAS, 1.0, HUBER_SIGMAS / np.maximum(a, 1e-12))
        A = J.T @ (J * w[:, None])
        try:
            dp = -np.linalg.solve(A + 1e-9 * np.eye(6), J.T @ (w * r0))
        except np.linalg.LinAlgError:
            break
        p = p + dp
        if float(np.linalg.norm(dp[:3])) < 1e-7 and float(np.linalg.norm(dp[3:])) < 1e-6:
            break
    r = _whitened(pose(p), K, uv, xyz, origin, px, rel).ravel()
    ok = np.abs(r) <= HUBER_SIGMAS
    rms = math.sqrt(float(np.mean(r[ok] ** 2))) if ok.any() else math.inf
    try:
        cov = np.linalg.inv(J.T @ (J * w[:, None])) * max(rms, 1.0) ** 2
    except np.linalg.LinAlgError:
        cov = np.full((6, 6), math.inf)
    return pose(p), cov, rms


def _refined(h: Hypothesis, hypothesis: Any, q: Query, uv: Array, xyz: Array, origin: Array,
             px: float) -> tuple[Hypothesis, Array | None]:
    """A hypothesis refined on all its inlier correspondences (``refine_keyframes``) and that
    refinement's covariance; the hypothesis as it was (None) when the refinement explains fewer
    query keypoints."""
    m = h.mask
    T, cov, _ = refine_keyframes(h.T, q.K, uv[m], xyz[m], origin[m],
                                 PIXEL_SIGMA * max(q.width, q.height) / 640.0)
    out = hypothesis(T)
    if out.inliers < h.inliers or not np.isfinite(cov).all():
        return h, None
    return out, cov


def _same_basin(h: Hypothesis, main: Hypothesis, q: Query, uv: Array, xyz: Array,
                origin: Array, px: float, tol: tuple[float, float]) -> bool:
    """Whether ``h``, refined on its own and ``main``'s inlier correspondences
    (``refine_keyframes``), comes back to ``main``: a hypothesis the matches cannot tell from
    ``main`` (along a direction the frame barely constrains, such as its distance to far
    structure) is no other place."""
    m = h.mask | main.mask
    T, _, _ = refine_keyframes(h.T, q.K, uv[m], xyz[m], origin[m],
                               PIXEL_SIGMA * max(q.width, q.height) / 640.0)
    dp, dr = pose_distance(T, main.T)
    return dp <= tol[0] / 2 and dr <= tol[1] / 2


def _keyframe_sigmas(h: Hypothesis, kfs: NDArray[np.int64], index: Index) -> tuple[float, float]:
    """The RMS of the matched keyframes' own sigmas, weighted by the inliers on each: a pose is
    no better placed in the map than the keyframes it is measured against."""
    per = np.bincount(kfs[h.mask], minlength=len(index.keyframes)).astype(np.float64)
    if per.sum() <= 0:
        return 0.0, 0.0
    sp = np.array([k.sigma_pos_m for k in index.keyframes])
    sr = np.array([k.sigma_rot_deg for k in index.keyframes])
    return (math.sqrt(float(per @ sp ** 2 / per.sum())),
            math.sqrt(float(per @ sr ** 2 / per.sum())))


def _sigmas(h: Hypothesis, others: Sequence[Hypothesis], q: Query, uv: Array, xyz: Array,
            iq: NDArray[np.int64], cov: Array | None = None,
            keyframes: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    """``SIGMA_SCALE`` times the fit's sigmas (``_fit``) and the same-place hypotheses' spread,
    combined."""
    fit = _fit(h, q, uv, xyz, iq, cov, keyframes)
    s_pos, s_rot = spread(h, others, tolerance(fit))
    return (round(SIGMA_SCALE * math.hypot(fit[0], s_pos), 4),
            round(SIGMA_SCALE * math.hypot(fit[1], s_rot), 3))


def _fit(h: Hypothesis, q: Query, uv: Array, xyz: Array, iq: NDArray[np.int64],
         cov: Array | None, keyframes: tuple[float, float]) -> tuple[float, float]:
    """(m, deg): the fit's covariance (``cov`` of the refinement when given, else the
    reprojection one) and the keyframes' own sigmas, combined."""
    if cov is not None:
        rot = math.degrees(math.sqrt(max(float(np.trace(cov[:3, :3])), 0.0)))
        pos = math.sqrt(max(float(np.trace(cov[3:, 3:])), 0.0))
    else:
        sel = np.array(sorted(_first(h, iq).values()), np.int64)
        pos, rot = covariance(h.T, q.K, uv[sel], xyz[sel])
    return math.hypot(pos, keyframes[0]), math.hypot(rot, keyframes[1])


def _rays(uv: NDArray[Any], K: Array) -> Array:
    from oh_my_slam.core.geometry import unproject_pixels

    x = unproject_pixels(uv[:, 0], uv[:, 1], np.ones(len(uv)), K)
    return np.asarray(x / np.linalg.norm(x, axis=1, keepdims=True), np.float64)


def _turn(q: Query, index: Index, iq: NDArray[np.int64], uv_k: NDArray[Any],
          kf_of: NDArray[np.int64]) -> Outcome | None:
    """The turn fit (see the module docstring), None when the keyframes it matches were not taken
    from one spot or the rotation is not confirmed."""
    used = np.unique(kf_of)
    w = np.array([int((kf_of == i).sum()) for i in used], np.float64)
    centres = np.array([index.keyframes[i].centre for i in used])
    centre = np.median(np.repeat(centres, w.astype(int), axis=0), axis=0)
    depth = [float(np.linalg.norm(index.keyframes[i].view_point() - index.keyframes[i].centre))
             for i in used]
    spread = float(np.max(np.linalg.norm(centres - centre, axis=1)))
    if spread > max(ONE_SPOT_M, ONE_SPOT_REL * float(np.median(depth))):
        return None
    a = _rays(q.uv[iq], q.K)
    b = np.concatenate([_rays(uv_k[kf_of == i], index.keyframes[i].K) @ index.keyframes[i].T[:3, :3].T
                        for i in used])
    a = np.concatenate([a[kf_of == i] for i in used])
    R, inl = turn_ransac(a, b)
    n = int(inl.sum())
    if n < MIN_INLIERS or n < TURN_MIN_SHARE * len(a):
        return None
    R = kabsch(a[inl], b[inl])
    ang = np.degrees(np.arccos(np.clip(np.sum((a[inl] @ R.T) * b[inl], axis=1), -1.0, 1.0)))
    rot = SIGMA_SCALE * math.sqrt(float(np.mean(ang ** 2)) + MIN_PX_SIGMA_DEG ** 2) * \
        math.sqrt(2.0 / n)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = centre
    if q.camera_z is not None:
        T[2, 3] = q.camera_z
    sp = math.hypot(spread, ONE_SPOT_REL * float(np.median(depth)))
    sr = rot
    out = Outcome(T, round(sp, 4), round(sr, 3), round(n / len(a), 4), n, method="turn")
    rest = ~inl
    if rest.sum() >= MIN_INLIERS:
        R2, inl2 = turn_ransac(a[rest], b[rest])
        n2 = int(inl2.sum())
        dr = _rot_deg(R, R2)
        if dr > AMBIGUITY_ROT_DEG and n2 >= AMBIGUITY_SUPPORT_RATIO * n:
            out.ambiguity = {"pos_m": 0.0, "rot_deg": round(dr, 3)}
    out.consistent_with_prior = consistent(q, T, sp, sr)
    return out


MIN_PX_SIGMA_DEG = 0.05


def nearest_view(q: Query, index: Index, k: int) -> tuple[Keyframe, float | None]:
    """The keyframe ``k`` and the signed turn (deg, CCW) from the frame's prior heading toward
    where that keyframe looks (the mean of its keypoints' points), seen from the prior's
    position; None without a prior (the frame's heading in the map is unknown)."""
    kf = index.keyframes[k]
    if q.prior is None:
        return kf, None
    target = kf.view_point() - q.prior[:3, 3]
    heading = math.degrees(math.atan2(target[1], target[0])) if np.hypot(*target[:2]) > 1e-6 \
        else yaw_deg(kf.T)
    return kf, round(wrap_deg(heading - yaw_deg(q.prior)), 2)
