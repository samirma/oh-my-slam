"""Prior-first registration and incremental fusion (the *anchored* map; README "Prior-first
registration").

A map whose first update carries camera pose priors (``metric`` or ``drifting``) is kept here,
in the priors' frame, and never re-posed:

* **Origin.** The first frame with a prior is placed at its prior: its frame is the map frame.
* **Place first.** Every later frame is first placed by its prior — a metric prior as given, a
  drifting one through the odometry's latest correction (``Drift``) with sigmas that grow with
  the travel since the last frame the images confirmed — then refined against the stored
  keyframes nearest to that pose (by prior pose, else by retrieval descriptor):
  * SIFT matches against those keyframes, each match lifted to 3-D by the keyframe's depth;
  * PnP (RANSAC) on the lifted points and a robust Gauss-Newton refinement, the stored poses
    held fixed. PnP comes first even for a camera turning in place: the lifted points fix its
    rotation, and a wrong depth scale moves only its centre (along its distance to the
    keyframes). Without enough lifted points, a rotation-only fit at the prior's centre, when
    that is within ``ROTATION_FALLBACK_M`` of a keyframe (without a prior: at the best-matched
    keyframe's centre);
  * the image pose and the prior are combined by their sigmas (``fuse_poses``). Each frame so
    placed is a fixed reference for the later frames of the same update.
* **Unconfirmed frames.** A frame whose images do not confirm a pose but which has a prior is
  kept ``provisional`` at its prior pose with the prior's sigmas; it is rejected
  ``prior_conflict`` only when a well-supported image pose lies beyond ``K_SIGMA`` combined
  sigmas of its prior; ``no_overlap`` only without any prior.
* **Scale.** Keyframe depth is the frame's depth image when it has one (metric), else the
  monocular depth times the map's depth scale ``k``. ``k`` comes from the metric priors once
  their centres span ``PRIOR_SPAN_M``: the scale that best brings the image-only PnP positions
  (relative to their reference keyframe) onto the priors' (a similarity of the map's positions
  to the priors); before that from camera heights above the floor seen in the monocular depth;
  else 1 (the monocular model's metres). Positions never move with ``k``: only the depth of the
  monocular keyframes added later, and of the stored ones as used for matching (each keyframe
  records the ``k`` it was fused with).
* **Incremental fusion.** Only the new keyframes are lifted into the map's voxel cloud
  (``Cloud``: a running mean per ``VOXEL_M`` voxel); nothing stored is fused again.

Everything lives under the mapper folder of the map version: ``state.json`` (keyframes, scale
evidence, odometry correction) and ``kf/<id>.npz`` (features, their depth and the
retrieval descriptor), ``cloud.npz``."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.atomic import atomic_write_json
from oh_my_slam.core.log import get_logger
from oh_my_slam.core.types import Intrinsics

log = get_logger("oh_my_slam.mapping.anchored")

Array = NDArray[np.float64]

ENGINE = "anchored"
STATE_JSON = "state.json"
CLOUD_NPZ = "cloud.npz"
KF_DIR = "kf"

MAX_FEATURES = 2000
SIFT_CONTRAST = 0.02  # half OpenCV's default: indoor robot frames are dim and low in texture
ROOTSIFT_QUANT = 512.0
RATIO = 0.8  # Lowe's ratio test
MAX_CANDIDATES = 5
CANDIDATE_MAX_ANGLE_DEG = 100.0  # optical axes further apart share no view
CANDIDATE_MAX_DIST_M = 5.0
CANDIDATE_DIST_UNIT_M = 0.5  # candidate score = distance / unit + angle / unit
CANDIDATE_ANGLE_UNIT_DEG = 30.0
RECENT_CANDIDATES = 8  # without prior or descriptor: the most recent keyframes
ROTATION_FALLBACK_M = 0.20  # PnP failed: rotation-only still tried up to this distance
REPROJ_PX = 3.0  # inlier threshold at 640 px width, scaled with the image width
REF_WIDTH = 640.0
MIN_INLIERS = 20
STRONG_INLIERS = 40  # an image pose this well supported can contradict a prior
STRONG_RATIO = 0.3  # ... with this share of its matches as inliers
CONFIRM_SIGMA_POS_M = 0.10  # an image pose confirms a frame only this well determined
CONFIRM_SIGMA_ROT_DEG = 2.0
K_SIGMA = 3.0
MIN_SIGMA_POS_M = 0.01
MIN_SIGMA_ROT_DEG = 0.1
IMAGE_SIGMA_POS_REL = 0.01  # of the median scene depth: depth bias the covariance does not see
IMAGE_SIGMA_ROT_MIN_DEG = 0.2
DEFAULT_SIGMA_POS_M = 0.05  # a prior without sigmas (as service.anchor)
DEFAULT_SIGMA_ROT_DEG = 2.0
UNANCHORED_SIGMA_POS_M = 0.10  # rotation-only without a prior: the centre is the keyframe's
DRIFT_POS_REL = 0.10  # drifting prior: sigma grows by this share of the travel
DRIFT_ROT_REL = 0.05  # ... by this share of the turn
DRIFT_ROT_DEG_PER_M = 2.0  # ... and this much per metre travelled
ODO_SCALE_MIN_TRAVEL_M = 0.2
ODO_SCALE_RANGE = (0.5, 2.0)
PRIOR_SPAN_M = 0.3  # metric priors fix the scale once their centres span this
SCALE_EVIDENCE_MIN_M = 0.1  # a PnP baseline shorter than this tells little about the scale
SCALE_EVIDENCE_MIN = 3
SCALE_RANGE = (0.2, 5.0)
MAX_EVIDENCE = 500
DEPTH_SIGMA_REL = 0.01  # depth images (README "Prior-first registration")
MONOCULAR_SIGMA_REL = 0.10
MIN_SCALE_SIGMA_REL = 0.002
DEFAULT_HEIGHT_SIGMA_M = 0.02
EDGE_REL = 0.06  # a depth sample whose 3x3 neighbourhood spans more than this is a depth edge
GN_ITERATIONS = 15
HUBER_PX = 2.0
MIN_PIXEL_SIGMA = 0.5
VOXEL_M = 0.02
FUSE_STRIDE = 2
FUSE_MAX_DEPTH_M = 6.0
FLOOR_BINS = 50
FLOOR_PEAK_SHARE = 0.25
FLOOR_MIN_POINTS = 200
TOO_CLOSE_M = 0.30  # median depth (as service.checks.TOO_CLOSE_M)
MIN_CAMERA_HEIGHT = 0.05
ALIGN_MIN_POINTS = 30  # a new monocular keyframe's depth is aligned on this many lifted points
ALIGN_RANGE = (0.5, 2.0)
# the depth scale's sigma that weighs PnP centres on monocular depth while nothing has measured
# the scale: the monocular model's metres are off by up to this between scenes
MONOCULAR_REGISTRATION_SIGMA_REL = 0.30  # raw monocular units: a camera this low tells no scale


# -- small geometry -------------------------------------------------------------------------------


def rot_deg(R: Array) -> float:
    """The angle of rotation ``R``, in degrees."""
    c = (float(np.trace(R)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def exp_so3(w: Array) -> Array:
    out, _ = cv2.Rodrigues(np.asarray(w, np.float64).reshape(3, 1))
    return np.asarray(out, np.float64)


def log_so3(R: Array) -> Array:
    out, _ = cv2.Rodrigues(np.asarray(R, np.float64))
    return np.asarray(out, np.float64).ravel()


def kabsch(a: Array, b: Array) -> Array:
    """The rotation ``R`` minimising ``|R a_i - b_i|`` over rows."""
    U, _, Vt = np.linalg.svd(b.T @ a)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt))) or 1.0])
    out: Array = U @ D @ Vt
    return out


def pose(R: Array, c: Array) -> Array:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = c
    return T


def pose_residual(A: Array, B: Array) -> tuple[float, float]:
    """Distance (m) and angle (deg) between two poses."""
    return (float(np.linalg.norm(A[:3, 3] - B[:3, 3])), rot_deg(A[:3, :3].T @ B[:3, :3]))


def fuse_poses(A: Array, spa: float, sra: float, B: Array, spb: float, srb: float
               ) -> tuple[Array, float, float]:
    """The information-weighted mean of two independent pose estimates with isotropic sigmas:
    centres weighted by ``1/sigma_pos²``, rotation interpolated on the geodesic by
    ``1/sigma_rot²``; the sigmas of the mean."""
    wa, wb = 1.0 / spa ** 2, 1.0 / spb ** 2
    c = (A[:3, 3] * wa + B[:3, 3] * wb) / (wa + wb)
    va, vb = 1.0 / sra ** 2, 1.0 / srb ** 2
    R = exp_so3(log_so3(A[:3, :3] @ B[:3, :3].T) * va / (va + vb)) @ B[:3, :3]
    return pose(R, c), 1.0 / math.sqrt(wa + wb), 1.0 / math.sqrt(va + vb)


def rays(K: Intrinsics, uv: NDArray[Any]) -> Array:
    """Camera rays with unit z of pixels ``uv`` (N x 2, pixel centres at integer + 0.5 as
    OpenCV's keypoints are at integers: OpenCV's convention)."""
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    return np.column_stack([(uv[:, 0] - K.cx) / K.fx, (uv[:, 1] - K.cy) / K.fy,
                            np.ones(len(uv))])


def unit(v: Array) -> Array:
    out: Array = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return out


def lowest_layer(z: NDArray[Any], below: float) -> float | None:
    """The floor among heights ``z`` (all lower than ``below``): the lowest of ``FLOOR_BINS``
    bins between their 1st percentile and ``below`` that holds ``FLOOR_PEAK_SHARE`` of the
    fullest, as the median of it and its neighbours. None with fewer than ``FLOOR_MIN_POINTS``."""
    z = np.asarray(z, np.float64)
    z = z[np.isfinite(z) & (z < below)]
    if z.size < FLOOR_MIN_POINTS:
        return None
    lo = float(np.percentile(z, 1))
    hist, edges = np.histogram(z, bins=FLOOR_BINS, range=(lo, below))
    i = int(np.argmax(hist >= FLOOR_PEAK_SHARE * hist.max()))
    near = z[(z >= edges[max(i - 1, 0)]) & (z <= edges[min(i + 2, FLOOR_BINS)])]
    return float(np.median(near))


# -- features -------------------------------------------------------------------------------------


@dataclass
class Features:
    kps: NDArray[np.float32]  # N x 2, pixels of the canonical image
    desc: NDArray[np.uint8]  # N x 128, RootSIFT quantised


def features(rgb: NDArray[np.uint8], ignore: NDArray[np.bool_] | None = None) -> Features:
    """SIFT keypoints and RootSIFT descriptors of a canonical image, none on ``ignore``."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = None if ignore is None else (~ignore).astype(np.uint8) * 255
    sift: Any = cv2.SIFT_create(nfeatures=MAX_FEATURES,  # type: ignore[attr-defined]
                                contrastThreshold=SIFT_CONTRAST)
    kps, desc = sift.detectAndCompute(gray, mask)
    if desc is None or not len(kps):
        return Features(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8))
    d = desc.astype(np.float32)
    d = np.sqrt(d / (d.sum(axis=1, keepdims=True) + 1e-7))
    return Features(np.array([k.pt for k in kps], np.float32),
                    np.clip(np.round(d * ROOTSIFT_QUANT), 0, 255).astype(np.uint8))


def match(a: NDArray[np.uint8], b: NDArray[np.uint8]) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Ratio-test matches from ``a`` to ``b``, one per keypoint of ``b`` (the closest)."""
    if len(a) < 2 or len(b) < 2:
        return np.zeros(0, np.intp), np.zeros(0, np.intp)
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(a.astype(np.float32), b.astype(np.float32), k=2)
    best: dict[int, tuple[float, int]] = {}
    for m, n in knn:
        if m.distance < RATIO * n.distance:
            j = int(m.trainIdx)
            if j not in best or m.distance < best[j][0]:
                best[j] = (float(m.distance), int(m.queryIdx))
    ia = np.array([v[1] for v in best.values()], np.intp)
    ib = np.array(list(best.keys()), np.intp)
    return ia, ib


def edge_free(depth: NDArray[np.float32]) -> NDArray[np.bool_]:
    """Valid depth samples off depth edges (their 3x3 neighbourhood within ``EDGE_REL``)."""
    d = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0).astype(np.float32)
    k = np.ones((3, 3), np.uint8)
    hi = cv2.dilate(d, k)
    lo = cv2.erode(np.where(d > 0, d, np.float32(1e9)).astype(np.float32), k)
    out: NDArray[np.bool_] = (d > 0) & (lo > 0) & (hi <= lo * (1.0 + EDGE_REL))
    return out


def sample_depth(depth: NDArray[np.float32], K_grid: Intrinsics, K: Intrinsics,
                 uv: NDArray[Any]) -> NDArray[np.float32]:
    """The depth grid's value at image pixels ``uv`` (nearest; NaN off the grid, invalid or on
    a depth edge)."""
    ok = edge_free(depth)
    u = np.round((uv[:, 0] - K.cx) / K.fx * K_grid.fx + K_grid.cx).astype(np.intp)
    v = np.round((uv[:, 1] - K.cy) / K.fy * K_grid.fy + K_grid.cy).astype(np.intp)
    h, w = depth.shape
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    out = np.full(len(uv), np.nan, np.float32)
    uu, vv = u[inside], v[inside]
    good = ok[vv, uu]
    vals = np.where(good, depth[vv, uu], np.nan).astype(np.float32)
    out[inside] = vals
    return out


# -- robust pose estimation -----------------------------------------------------------------------


@dataclass
class Solve:
    """A refined pose: rotation, centre, 6x6 covariance (rotation vector, centre) and the
    reprojection inliers."""

    R: Array
    c: Array
    cov: Array
    inliers: NDArray[np.bool_]
    rms_px: float

    @property
    def T(self) -> Array:
        return pose(self.R, self.c)

    def sigmas(self) -> tuple[float, float]:
        """One-sigma position (m) and rotation (deg) from the covariance (largest axis)."""
        sp = math.sqrt(max(float(np.linalg.eigvalsh(self.cov[3:, 3:]).max()), 0.0))
        sr = math.degrees(math.sqrt(max(float(np.linalg.eigvalsh(self.cov[:3, :3]).max()), 0.0)))
        return sp, sr


def _project(R: Array, c: Array, X: Array, K: Intrinsics, directions: bool) -> Array:
    p = (X - (0.0 if directions else c)) @ R  # camera frame: R^T (X - c), row-wise
    z = np.where(np.abs(p[:, 2]) < 1e-9, 1e-9, p[:, 2])
    return np.column_stack([K.fx * p[:, 0] / z + K.cx, K.fy * p[:, 1] / z + K.cy])


def refine(R0: Array, c0: Array, X: Array, uv: Array, K: Intrinsics, *, directions: bool = False,
           fix_centre: bool = False, huber_px: float = HUBER_PX) -> Solve:
    """Robust Gauss-Newton (Huber IRLS) of a camera pose on reprojections of world points ``X``
    (or world directions, ``directions``: rotation only) onto pixels ``uv``. The pose is ``R = exp(w) R0``, ``c = c0 + dc``; the covariance comes from
    the normal equations with the pixel noise estimated from the inliers' residuals."""
    n_par = 3 if fix_centre else 6
    R, c = R0.copy(), c0.copy()
    sigma_px = huber_px

    def residual(R_: Array, c_: Array) -> Array:
        return (_project(R_, c_, X, K, directions) - uv).ravel()

    def step(d: Array) -> tuple[Array, Array]:
        R_ = exp_so3(d[:3]) @ R
        c_ = c if fix_centre else c + d[3:6]
        return R_, c_

    eps = 1e-6
    w = np.ones(len(uv))
    J = np.zeros((2 * len(uv), n_par))
    for _ in range(GN_ITERATIONS):
        r = residual(R, c)
        e = np.hypot(r[0::2], r[1::2])
        w = np.where(e <= huber_px, 1.0, huber_px / np.maximum(e, 1e-12))
        J = np.zeros((len(r), n_par))
        for k in range(n_par):
            d = np.zeros(6)
            d[k] = eps
            R_, c_ = step(d)
            J[:, k] = (residual(R_, c_) - r) / eps
        W = np.repeat(w, 2) / sigma_px ** 2
        H = J.T @ (W[:, None] * J)
        delta = -np.linalg.solve(H + 1e-9 * np.eye(n_par), J.T @ (W * r))
        d = np.zeros(6)
        d[:n_par] = delta
        R, c = step(d)
        if float(np.abs(delta).max()) < 1e-8:
            break
    r = residual(R, c)
    e = np.hypot(r[0::2], r[1::2])
    inl = e <= 2.0 * huber_px
    if inl.sum() >= 3:
        sigma_px = max(1.4826 * float(np.median(e[inl])) / math.sqrt(2.0) * 1.2, MIN_PIXEL_SIGMA)
    W = np.repeat(w, 2) / sigma_px ** 2
    H = J.T @ (W[:, None] * J)
    cov6 = np.zeros((6, 6))
    cov6[:n_par, :n_par] = np.linalg.inv(H + 1e-9 * np.eye(n_par))
    rms = float(np.sqrt(np.mean(e[inl] ** 2))) if inl.any() else float("inf")
    return Solve(R, c, cov6, inl, rms)


def pnp(X: Array, uv: Array, K: Intrinsics, thr_px: float, seed: int = 0
        ) -> tuple[Array, Array, NDArray[np.bool_]] | None:
    """RANSAC PnP: (R, c, inliers) of the camera seeing world points ``X`` at ``uv``."""
    if len(X) < 6:
        return None
    cv2.setRNGSeed(seed)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        X.astype(np.float64), uv.astype(np.float64), K.K(), None, iterationsCount=1000,
        reprojectionError=float(thr_px), confidence=0.999, flags=cv2.SOLVEPNP_AP3P)
    if not ok or inl is None or len(inl) < 6:
        return None
    Rcw, _ = cv2.Rodrigues(rvec)
    R = Rcw.T
    c = -R @ tvec.ravel()
    mask = np.zeros(len(X), bool)
    mask[inl.ravel()] = True
    return np.asarray(R, np.float64), np.asarray(c, np.float64), mask


def rotation_ransac(d_world: Array, b_cam: Array, thr_rad: float, seed: int = 0,
                    iterations: int = 300) -> tuple[Array, NDArray[np.bool_]] | None:
    """RANSAC rotation ``R`` with ``R b_cam ≈ d_world`` (unit rows), from pairs."""
    n = len(d_world)
    if n < 3:
        return None
    rng = np.random.default_rng(seed)
    best: tuple[int, Array] | None = None
    cos_thr = math.cos(thr_rad)
    for _ in range(iterations):
        i, j = rng.choice(n, 2, replace=False)
        if float(np.dot(b_cam[i], b_cam[j])) > math.cos(math.radians(1.0)):
            continue
        R = kabsch(b_cam[[i, j]], d_world[[i, j]])
        cnt = int(((b_cam @ R.T * d_world).sum(1) > cos_thr).sum())
        if best is None or cnt > best[0]:
            best = (cnt, R)
    if best is None:
        return None
    R = best[1]
    for _ in range(3):
        inl = (b_cam @ R.T * d_world).sum(1) > cos_thr
        if inl.sum() < 3:
            return None
        R = kabsch(b_cam[inl], d_world[inl])
    inl = (b_cam @ R.T * d_world).sum(1) > cos_thr
    return R, inl


# -- the map ---------------------------------------------------------------------------------------


@dataclass
class Keyframe:
    id: str
    name: str
    camera_id: str
    K: Intrinsics
    T: Array
    sigma_pos_m: float
    sigma_rot_deg: float
    state: str  # "fused" (confirmed by the images, or the origin) or "provisional"
    prior: dict[str, Any] | None  # {"trust", "T", "sigma_pos_m", "sigma_rot_deg"}
    depth_kind: str  # "sensor", "mono" or "none"
    k_used: float = 1.0  # the map's depth scale when a mono keyframe was fused
    median_depth: float | None = None  # metres as fused
    method: str = "prior"
    inliers: int = 0
    # loaded or set while the update runs (kf/<id>.npz)
    feats: Features | None = None
    kdepth: NDArray[np.float32] | None = None  # depth of each keypoint as fused (NaN: none)
    descriptor: NDArray[np.float32] | None = None

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "camera_id": self.camera_id,
                "K": self.K.to_dict(), "T": self.T.tolist(), "sigma_pos_m": self.sigma_pos_m,
                "sigma_rot_deg": self.sigma_rot_deg, "state": self.state, "prior": self.prior,
                "depth_kind": self.depth_kind, "k_used": self.k_used,
                "median_depth": self.median_depth, "method": self.method,
                "inliers": self.inliers}

    @staticmethod
    def from_json(d: dict[str, Any]) -> Keyframe:
        return Keyframe(d["id"], d["name"], d["camera_id"], Intrinsics.from_dict(d["K"]),
                        np.asarray(d["T"], np.float64), float(d["sigma_pos_m"]),
                        float(d["sigma_rot_deg"]), d["state"], d["prior"], d["depth_kind"],
                        float(d["k_used"]), d["median_depth"], d["method"], int(d["inliers"]))

    @property
    def centre(self) -> Array:
        out: Array = self.T[:3, 3]
        return out

    def usable(self) -> bool:
        """A reference for later frames: confirmed, or provisional at a metric prior."""
        return self.state == "fused" or (self.prior is not None
                                         and self.prior["trust"] == "metric")


@dataclass
class Cloud:
    """The fused cloud: per occupied ``VOXEL_M`` voxel the sum of its points, of their colours,
    and their count."""

    keys: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, np.int64))
    xyz: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 3)))
    rgb: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 3)))
    count: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, np.int64))

    @staticmethod
    def _keys(p: NDArray[Any]) -> NDArray[np.int64]:
        q = np.floor(p / VOXEL_M).astype(np.int64) + (1 << 20)
        out: NDArray[np.int64] = (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]
        return out

    def add(self, xyz: NDArray[Any], rgb: NDArray[Any]) -> None:
        if not len(xyz):
            return
        keys = np.concatenate([self.keys, self._keys(xyz)])
        uk, inv = np.unique(keys, return_inverse=True)
        n = len(uk)
        sx = np.zeros((n, 3))
        sc = np.zeros((n, 3))
        cnt = np.zeros(n, np.int64)
        m = len(self.keys)
        np.add.at(sx, inv[:m], self.xyz)
        np.add.at(sc, inv[:m], self.rgb)
        np.add.at(cnt, inv[:m], self.count)
        np.add.at(sx, inv[m:], xyz)
        np.add.at(sc, inv[m:], rgb)
        np.add.at(cnt, inv[m:], 1)
        self.keys, self.xyz, self.rgb, self.count = uk, sx, sc, cnt

    def points(self) -> NDArray[np.float64]:
        out: NDArray[np.float64] = self.xyz / np.maximum(self.count, 1)[:, None]
        return out

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, keys=self.keys, xyz=self.xyz, rgb=self.rgb.astype(np.float32),
                 count=self.count)
        tmp.replace(path)

    @staticmethod
    def load(path: Path) -> Cloud:
        if not path.exists():
            return Cloud()
        with np.load(path) as z:
            return Cloud(z["keys"], z["xyz"], z["rgb"].astype(np.float64), z["count"])


@dataclass
class Drift:
    """The latest correction of the drifting odometry: the last keyframe the images confirmed
    with a drifting prior (its map pose and its prior) and the odometry's scale."""

    ref_T: Array | None = None
    ref_prior: Array | None = None
    odo_scale: float = 1.0
    evidence: list[list[float]] = field(default_factory=list)  # [odometry m, map m]

    def to_json(self) -> dict[str, Any]:
        return {"ref_T": None if self.ref_T is None else self.ref_T.tolist(),
                "ref_prior": None if self.ref_prior is None else self.ref_prior.tolist(),
                "odo_scale": self.odo_scale, "evidence": self.evidence}

    @staticmethod
    def from_json(d: dict[str, Any]) -> Drift:
        return Drift(None if d["ref_T"] is None else np.asarray(d["ref_T"], np.float64),
                     None if d["ref_prior"] is None else np.asarray(d["ref_prior"], np.float64),
                     float(d["odo_scale"]), [list(e) for e in d["evidence"]])

    def predict(self, P: Array, sigma_pos: float, sigma_rot: float
                ) -> tuple[Array, float, float]:
        """The map pose and sigmas of a drifting prior ``P``."""
        if self.ref_T is None or self.ref_prior is None:
            return P.copy(), sigma_pos, sigma_rot
        D = np.linalg.inv(self.ref_prior) @ P
        D[:3, 3] *= self.odo_scale
        travel = float(np.linalg.norm(D[:3, 3]))
        turn = rot_deg(D[:3, :3])
        return (self.ref_T @ D, sigma_pos + DRIFT_POS_REL * travel,
                sigma_rot + DRIFT_ROT_REL * turn + DRIFT_ROT_DEG_PER_M * travel)

    def confirm(self, T: Array, P: Array) -> None:
        """A drifting frame the images placed at ``T``: the new reference, and odometry scale
        evidence against the previous one."""
        if self.ref_T is not None and self.ref_prior is not None:
            odo = float(np.linalg.norm(P[:3, 3] - self.ref_prior[:3, 3]))
            moved = float(np.linalg.norm(T[:3, 3] - self.ref_T[:3, 3]))
            if odo >= ODO_SCALE_MIN_TRAVEL_M:
                self.evidence = [*self.evidence, [odo, moved]][-MAX_EVIDENCE:]
                odo_m = np.array([e[0] for e in self.evidence])
                map_m = np.array([e[1] for e in self.evidence])
                s = float((odo_m * map_m).sum() / (odo_m ** 2).sum())
                self.odo_scale = float(np.clip(s, *ODO_SCALE_RANGE))
        self.ref_T, self.ref_prior = T.copy(), P.copy()


@dataclass
class ScaleState:
    """The depth scale ``k`` of monocular keyframes and its evidence."""

    k: float = 1.0
    source: str = "monocular"  # of k: "priors", "camera_height", "monocular"
    sigma_rel: float = MONOCULAR_SIGMA_REL
    # PnP baselines against metric priors: [prior baseline (m), raw image baseline, ...]
    priors: list[list[float]] = field(default_factory=list)
    heights: list[list[float]] = field(default_factory=list)  # [given m / raw height, sigma rel]
    # PnP baselines against drifting priors: [odometry baseline (m), raw image baseline]
    odometry: list[list[float]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"k": self.k, "source": self.source, "sigma_rel": self.sigma_rel,
                "priors": self.priors, "heights": self.heights, "odometry": self.odometry}

    @staticmethod
    def from_json(d: dict[str, Any]) -> ScaleState:
        return ScaleState(float(d["k"]), d["source"], float(d["sigma_rel"]),
                          [list(e) for e in d["priors"]], [list(e) for e in d["heights"]],
                          [list(e) for e in d["odometry"]])

    @staticmethod
    def _baselines(evidence: list[list[float]], span: float) -> tuple[float, float] | None:
        """The least-squares scale of prior baselines on raw image baselines and the robust
        spread of their log ratios over the root of their count, when the priors span
        ``PRIOR_SPAN_M`` and ``SCALE_EVIDENCE_MIN`` baselines are long enough."""
        ev = [e for e in evidence if e[0] >= SCALE_EVIDENCE_MIN_M and e[1] > 1e-6]
        if span < PRIOR_SPAN_M or len(ev) < SCALE_EVIDENCE_MIN:
            return None
        b_p = np.array([e[0] for e in ev])
        b_i = np.array([e[1] for e in ev])
        k = float((b_p * b_i).sum() / (b_i ** 2).sum())
        r = np.log(b_p / (k * b_i))
        mad = 1.4826 * float(np.median(np.abs(r - np.median(r))))
        return float(np.clip(k, *SCALE_RANGE)), mad / math.sqrt(len(ev))

    def estimate(self, prior_span: float, odometry_span: float = 0.0) -> None:
        """``k`` from the evidence: metric priors (once their span allows), else camera heights,
        else drifting priors (their relative motion is metric within the odometry's drift:
        ``DRIFT_POS_REL`` at best, reported as ``priors``), else held at 1."""
        got = self._baselines(self.priors, prior_span)
        if got is not None:
            self.k, self.source = got[0], "priors"
            self.sigma_rel = max(got[1], MIN_SCALE_SIGMA_REL)
            return
        if self.heights:
            r = np.array([h[0] for h in self.heights])
            rel = np.array([h[1] for h in self.heights])
            lr = np.log(r)
            mad = 1.4826 * float(np.median(np.abs(lr - np.median(lr))))
            self.k = float(np.clip(np.exp(np.median(lr)), *SCALE_RANGE))
            self.source = "camera_height"
            self.sigma_rel = max(math.sqrt(mad ** 2 + float(np.median(rel)) ** 2)
                                 / math.sqrt(len(r)), MIN_SCALE_SIGMA_REL)
            return
        got = self._baselines(self.odometry, odometry_span)
        if got is not None:
            self.k, self.source = got[0], "priors"
            self.sigma_rel = max(got[1], DRIFT_POS_REL)
            return
        self.k, self.source, self.sigma_rel = 1.0, "monocular", MONOCULAR_SIGMA_REL


@dataclass
class FrameIn:
    """A new frame as the engine gets it (canonical pinhole)."""

    name: str
    camera_id: str
    rgb: NDArray[np.uint8]
    K: Intrinsics
    prior: dict[str, Any] | None = None  # the request's Prior (trust metric or drifting)
    depth: NDArray[np.float32] | None = None  # depth image, metres (0: none)
    ignore: NDArray[np.bool_] | None = None
    height_m: float | None = None
    height_sigma_m: float | None = None
    features_count: int = 0  # the checks' count (for the reason of an unplaced frame)
    # monocular depth (grid metres, K_grid, retrieval descriptor or None, up in camera or None),
    # waited for when needed
    mono: Callable[[], tuple[NDArray[np.float32], Intrinsics, Any, Any]] | None = None
    check_close: bool = False  # reject it ``too_close`` if its monocular depth says so


@dataclass
class Placement:
    name: str
    status: str  # registered, provisional, rejected
    reason: str | None = None
    T: Array | None = None
    sigma_pos_m: float | None = None
    sigma_rot_deg: float | None = None
    inliers: int = 0
    method: str | None = None  # origin, prior, pnp, rotation
    prior_residual: tuple[float, float] | None = None
    keyframe: str | None = None  # the keyframe id when kept
    # the lifted points it was registered on and their pixels (``AnchoredMap.align_depth``)
    lifted: tuple[Array, Array] | None = None


def _lifted(corr: _Corr) -> tuple[Array, Array]:
    has = np.isfinite(corr.X).all(axis=1)
    return corr.X[has], corr.uv[has]


def depth_alignment(T: Array, X: Array, uv: Array, depth: NDArray[np.float32],
                    K_grid: Intrinsics, K: Intrinsics) -> float | None:
    """The factor bringing a new keyframe's depth onto the map: the median ratio of the depth
    the map's lifted points ``X`` have in the camera ``T`` to the keyframe's own depth at their
    pixels ``uv``. None with fewer than ``ALIGN_MIN_POINTS`` or a factor outside
    ``ALIGN_RANGE`` (the depth and the pose disagree: kept as it is)."""
    if len(X) < ALIGN_MIN_POINTS:
        return None
    z_map = ((X - T[:3, 3]) @ T[:3, :3])[:, 2]
    z_own = sample_depth(depth, K_grid, K, uv).astype(np.float64)
    ok = np.isfinite(z_own) & (z_own > 0) & (z_map > 0)
    if ok.sum() < ALIGN_MIN_POINTS:
        return None
    a = float(np.median(z_map[ok] / z_own[ok]))
    return a if ALIGN_RANGE[0] <= a <= ALIGN_RANGE[1] else None


def _prior_sigmas(p: dict[str, Any]) -> tuple[float, float]:
    return (max(float(p.get("sigma_pos_m", DEFAULT_SIGMA_POS_M)), MIN_SIGMA_POS_M),
            max(float(p.get("sigma_rot_deg", DEFAULT_SIGMA_ROT_DEG)), MIN_SIGMA_ROT_DEG))


@dataclass
class _Corr:
    uv: Array  # new frame pixels
    X: Array  # world points (NaN rows: no depth)
    d: Array  # world directions from the reference keyframe's centre
    kf: NDArray[np.intp]  # index into the candidate list


class AnchoredMap:
    """An anchored map in its mapper folder (see the module docstring)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.keyframes: list[Keyframe] = []
        self.scale = ScaleState()
        self.drift = Drift()
        self.next_id = 0
        self.cloud = Cloud()
        self.created_with: str | None = None  # the trust of the origin's prior
        self.up_check_deg: float | None = None
        self.timings: dict[str, float] = {}
        self._added: list[Keyframe] = []

    # -- persistence ---------------------------------------------------------------------------

    @staticmethod
    def load(root: Path) -> AnchoredMap:
        m = AnchoredMap(root)
        d = json.loads((root / STATE_JSON).read_text())
        m.keyframes = [Keyframe.from_json(k) for k in d["keyframes"]]
        m.scale = ScaleState.from_json(d["scale"])
        m.drift = Drift.from_json(d["drift"])
        m.next_id = int(d["next_id"])
        m.created_with = d["created_with"]
        m.cloud = Cloud.load(root / CLOUD_NPZ)
        return m

    def save(self) -> None:
        (self.root / KF_DIR).mkdir(parents=True, exist_ok=True)
        for kf in self._added:
            assert kf.feats is not None and kf.kdepth is not None
            extra = {} if kf.descriptor is None else {"descriptor": kf.descriptor}
            np.savez(self.root / KF_DIR / f"{kf.id}.npz", kps=kf.feats.kps, desc=kf.feats.desc,
                     kdepth=kf.kdepth, **extra)
        self._added = []
        self.cloud.save(self.root / CLOUD_NPZ)
        atomic_write_json(self.root / STATE_JSON, {
            "engine": ENGINE, "keyframes": [k.to_json() for k in self.keyframes],
            "scale": self.scale.to_json(), "drift": self.drift.to_json(),
            "next_id": self.next_id, "created_with": self.created_with})

    def _load_kf(self, kf: Keyframe) -> None:
        if kf.feats is not None:
            return
        with np.load(self.root / KF_DIR / f"{kf.id}.npz") as z:
            kf.feats = Features(z["kps"], z["desc"])
            kf.kdepth = z["kdepth"]
            kf.descriptor = z["descriptor"] if "descriptor" in z.files else None

    # -- summary -------------------------------------------------------------------------------

    def prior_span(self, trust: str = "metric") -> float:
        """The largest distance between two priors of that trust of the map's keyframes."""
        c = np.array([np.asarray(k.prior["T"])[:3, 3] for k in self.keyframes
                      if k.prior is not None and k.prior["trust"] == trust])
        if len(c) < 2:
            return 0.0
        return float(np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2).max())

    def scale_summary(self) -> tuple[str, float]:
        """The map's ``scale_source`` and ``scale_sigma_rel``: its metric priors once they
        span ``PRIOR_SPAN_M`` (positions are theirs), else depth images, else the depth scale's
        source (camera heights or the monocular model)."""
        span = self.prior_span()
        if span >= PRIOR_SPAN_M:
            sp = float(np.median([_prior_sigmas(k.prior)[0] for k in self.keyframes
                                  if k.prior is not None and k.prior["trust"] == "metric"]))
            return "priors", max(math.sqrt(2.0) * sp / span, MIN_SCALE_SIGMA_REL)
        if any(k.depth_kind == "sensor" for k in self.keyframes):
            return "depth", DEPTH_SIGMA_REL
        return self.scale.source, self.scale.sigma_rel

    # -- registration --------------------------------------------------------------------------

    def _predict(self, f: FrameIn) -> tuple[Array, float, float] | None:
        p = f.prior
        if p is None or p["trust"] not in ("metric", "drifting"):
            return None
        P = np.asarray(p["T_map_cam"], np.float64)
        sp, sr = _prior_sigmas(p)
        if p["trust"] == "metric":
            return P, sp, sr
        return self.drift.predict(P, sp, sr)

    def _candidates(self, f: FrameIn, T0: Array | None, descriptor: Any) -> list[Keyframe]:
        usable = [k for k in self.keyframes if k.usable()]
        if T0 is not None:
            scored = []
            for k in usable:
                d = float(np.linalg.norm(k.centre - T0[:3, 3]))
                a = math.degrees(math.acos(max(-1.0, min(1.0, float(k.T[:3, 2] @ T0[:3, 2])))))
                if d <= CANDIDATE_MAX_DIST_M and a <= CANDIDATE_MAX_ANGLE_DEG:
                    scored.append((d / CANDIDATE_DIST_UNIT_M + a / CANDIDATE_ANGLE_UNIT_DEG, k))
            scored.sort(key=lambda s: s[0])
            return [k for _, k in scored[:MAX_CANDIDATES]]
        if descriptor is not None:
            for k in usable:
                self._load_kf(k)
            with_desc = [k for k in usable if k.descriptor is not None]
            if with_desc:
                q = np.asarray(descriptor, np.float64).ravel()
                sims = [float(np.dot(q, np.asarray(k.descriptor, np.float64).ravel())
                              / (np.linalg.norm(q) * np.linalg.norm(k.descriptor) + 1e-12))
                        for k in with_desc]
                order = np.argsort(sims)[::-1][:MAX_CANDIDATES]
                return [with_desc[int(i)] for i in order]
        return usable[-RECENT_CANDIDATES:][::-1]

    def _correspondences(self, feats: Features, cands: list[Keyframe]) -> _Corr:
        uv, X, d, which = [], [], [], []
        seen = np.zeros(len(feats.kps), bool)
        for j, k in enumerate(cands):
            self._load_kf(k)
            assert k.feats is not None and k.kdepth is not None
            ia, ib = match(feats.desc, k.feats.desc)
            keep = ~seen[ia]
            ia, ib = ia[keep], ib[keep]
            if not len(ia):
                continue
            seen[ia] = True
            ray = rays(k.K, k.feats.kps[ib])
            z = k.kdepth[ib].astype(np.float64) * self._depth_factor(k)
            R, c = k.T[:3, :3], k.centre
            X.append(c + (ray * z[:, None]) @ R.T)
            d.append(unit(ray @ R.T))
            uv.append(feats.kps[ia].astype(np.float64))
            which.append(np.full(len(ia), j, np.intp))
        if not uv:
            e = np.zeros((0, 3))
            return _Corr(np.zeros((0, 2)), e, e, np.zeros(0, np.intp))
        return _Corr(np.vstack(uv), np.vstack(X), np.vstack(d), np.concatenate(which))

    def _depth_factor(self, k: Keyframe) -> float:
        """What a keyframe's stored depth is multiplied by now: the depth scale's change since
        it was fused (monocular keyframes only)."""
        return self.scale.k / k.k_used if k.depth_kind == "mono" else 1.0

    @staticmethod
    def _thr(K: Intrinsics) -> float:
        return REPROJ_PX * K.width / REF_WIDTH

    def _solve_rotation(self, f: FrameIn, corr: _Corr, cands: list[Keyframe], centre: Array
                        ) -> tuple[Solve, int, float] | None:
        """Rotation-only, the camera at ``centre``: directions to the lifted points where they
        have depth, else the keyframes' rays. Returns the solve, its inliers and the angular
        bias (rad) the centre's distance to the keyframes can leave: the parallax of the rays,
        or the depth scale's uncertainty times that parallax for lifted points."""
        has = np.isfinite(corr.X).all(axis=1)
        dirs = corr.d.copy()
        dirs[has] = unit(corr.X[has] - centre)
        dist = np.array([float(np.linalg.norm(cands[j].centre - centre)) for j in corr.kf])
        med = np.array([cands[j].median_depth or 2.0 for j in corr.kf])
        sk = np.array([self.scale.sigma_rel if cands[j].depth_kind == "mono" else DEPTH_SIGMA_REL
                       for j in corr.kf])
        bias_i = np.where(has, sk, 1.0) * dist / med
        b = unit(rays(f.K, corr.uv))
        got = rotation_ransac(dirs, b, self._thr(f.K) / f.K.fx + float(np.median(bias_i)))
        if got is None:
            return None
        R, inl = got
        if inl.sum() < MIN_INLIERS:
            return None
        s = refine(R, centre, dirs[inl], corr.uv[inl], f.K, directions=True, fix_centre=True)
        return s, int(s.inliers.sum()), float(np.median(bias_i[inl]))

    def _solve_pnp(self, f: FrameIn, corr: _Corr) -> tuple[Solve, int, float] | None:
        has = np.isfinite(corr.X).all(axis=1)
        if has.sum() < MIN_INLIERS:
            return None
        X, uv = corr.X[has], corr.uv[has]
        got = pnp(X, uv, f.K, self._thr(f.K))
        if got is None:
            return None
        R, c, inl = got
        if inl.sum() < MIN_INLIERS:
            return None
        s = refine(R, c, X[inl], uv[inl], f.K)
        n = int(s.inliers.sum())
        if n < MIN_INLIERS:
            return None
        return s, n, n / max(int(has.sum()), 1)

    def _pnp_sigmas(self, s: Solve, cands: list[Keyframe]) -> tuple[float, float]:
        """The image pose's sigmas: its covariance, floored by the depth's own bias (a share of
        the scene depth) and, for monocular depth, the depth scale's uncertainty times the
        distance to the keyframes (scaling their points about their centres moves the PnP
        centre along that distance, not its rotation)."""
        sp, sr = s.sigmas()
        med = float(np.median([k.median_depth or 2.0 for k in cands]))
        floor_p = max(MIN_SIGMA_POS_M, IMAGE_SIGMA_POS_REL * med)
        if any(k.depth_kind == "mono" for k in cands):
            base = float(np.min([np.linalg.norm(k.centre - s.c) for k in cands]))
            sk = MONOCULAR_REGISTRATION_SIGMA_REL if self.scale.source == "monocular" \
                else self.scale.sigma_rel
            floor_p = math.hypot(floor_p, sk * base)
        return max(sp, floor_p), max(sr, IMAGE_SIGMA_ROT_MIN_DEG)

    def _solve(self, f: FrameIn, corr: _Corr, cands: list[Keyframe], T0: Array | None,
               sp0: float) -> tuple[Solve, int, float, str, float, float] | None:
        """The image-only pose: PnP on the lifted points; failing that, rotation-only at the
        prior's centre (within ``ROTATION_FALLBACK_M`` of a keyframe) or, without a prior, at
        the best-matched keyframe's centre. Returns (solve, inliers, inlier ratio, method,
        sigma_pos_m, sigma_rot_deg)."""
        p = self._solve_pnp(f, corr)
        if p is not None:
            return (p[0], p[1], p[2], "pnp", *self._pnp_sigmas(p[0], cands))
        if T0 is not None:
            near = min(float(np.linalg.norm(k.centre - T0[:3, 3])) for k in cands)
            if near > ROTATION_FALLBACK_M:
                return None
            sub, centre, sp = np.ones(len(corr.uv), bool), T0[:3, 3], sp0
        else:
            best = int(np.bincount(corr.kf).argmax())
            sub, centre, sp = corr.kf == best, cands[best].centre, UNANCHORED_SIGMA_POS_M
        if sub.sum() < MIN_INLIERS:
            return None
        r = self._solve_rotation(f, _Corr(corr.uv[sub], corr.X[sub], corr.d[sub], corr.kf[sub]),
                                 cands, centre)
        if r is None:
            return None
        s, n, bias = r
        sr = max(s.sigmas()[1], math.degrees(bias), IMAGE_SIGMA_ROT_MIN_DEG)
        return s, n, n / int(sub.sum()), "rotation", sp, sr

    def place(self, f: FrameIn, feats: Features, descriptor: Any) -> Placement:
        """Place one new frame (the map has its origin); does not add it."""
        pred = self._predict(f)
        T0, sp0, sr0 = (None, 0.0, 0.0) if pred is None else pred
        cands = self._candidates(f, T0, descriptor)
        reason = "few_features" if min(f.features_count or len(feats.kps), len(feats.kps)) \
            < 150 else "no_overlap"
        corr = self._correspondences(feats, cands) if cands and len(feats.kps) else None
        sol = None
        if corr is not None and len(corr.uv) >= MIN_INLIERS:
            sol = self._solve(f, corr, cands, T0, sp0)
        if T0 is None:
            if sol is None:
                return Placement(f.name, "rejected", reason)
            s, n, _, method, spi, sri = sol
            return Placement(f.name, "registered", None, s.T, round(spi, 4), round(sri, 3), n,
                             method, lifted=_lifted(corr))
        assert f.prior is not None
        prior_T = np.asarray(f.prior["T_map_cam"], np.float64)
        ref = T0 if f.prior["trust"] == "drifting" else prior_T
        if sol is None:
            return Placement(f.name, "provisional", reason, T0, sp0, sr0, 0, "prior",
                             pose_residual(T0, ref))
        s, n, ratio, method, spi, sri = sol
        rot_only = method == "rotation"
        if rot_only:  # the centre is the prior's
            s.c = T0[:3, 3].copy()
        elif spi > CONFIRM_SIGMA_POS_M or sri > CONFIRM_SIGMA_ROT_DEG:
            return Placement(f.name, "provisional", reason, T0, sp0, sr0, n, "prior",
                             pose_residual(T0, ref))
        dp, dr = pose_residual(s.T, T0)
        if dp > K_SIGMA * math.hypot(sp0, spi) or dr > K_SIGMA * math.hypot(sr0, sri):
            if n >= STRONG_INLIERS and ratio >= STRONG_RATIO:
                return Placement(f.name, "rejected", "prior_conflict", None, None, None, n,
                                 method, (dp, dr))
            return Placement(f.name, "provisional", "no_overlap", T0, sp0, sr0, n, "prior",
                             pose_residual(T0, ref))
        T, fsp, fsr = fuse_poses(s.T, spi, sri, T0, sp0, sr0)
        if rot_only:
            T[:3, 3], fsp = T0[:3, 3], sp0
        pl = Placement(f.name, "registered", None, T, round(max(fsp, MIN_SIGMA_POS_M), 4),
                       round(max(fsr, MIN_SIGMA_ROT_DEG), 3), n, method, pose_residual(T, ref),
                       lifted=_lifted(corr))  # type: ignore[arg-type]
        if method == "pnp":
            self._scale_evidence(s, prior_T, f.prior["trust"], corr, cands)  # type: ignore[arg-type]
        return pl

    def _scale_evidence(self, s: Solve, prior_T: Array, trust: str, corr: _Corr,
                        cands: list[Keyframe]) -> None:
        """A PnP centre against its prior, both relative to the reference keyframe (the one
        with the most matches), in that keyframe's raw depth units (monocular only). A metric
        prior is compared with the reference's map centre; a drifting one with the
        reference's own drifting prior (odometry baselines)."""
        j = int(np.bincount(corr.kf).argmax())
        ref = cands[j]
        if ref.depth_kind != "mono":
            return
        b_img = float(np.linalg.norm(s.c - ref.centre)) / self.scale.k
        if trust == "metric":
            b_prior = float(np.linalg.norm(prior_T[:3, 3] - ref.centre))
            self.scale.priors = [*self.scale.priors, [b_prior, b_img]][-MAX_EVIDENCE:]
        elif ref.prior is not None and ref.prior["trust"] == "drifting":
            b_odo = float(np.linalg.norm(prior_T[:3, 3] - np.asarray(ref.prior["T"])[:3, 3]))
            self.scale.odometry = [*self.scale.odometry, [b_odo, b_img]][-MAX_EVIDENCE:]

    # -- adding keyframes ----------------------------------------------------------------------

    def _keyframe_depth(self, f: FrameIn) -> tuple[NDArray[np.float32], Intrinsics, str, Any]:
        """The depth grid (metres as fused), its intrinsics, its kind and the descriptor."""
        if f.depth is not None and np.count_nonzero(f.depth > 0) > 0:
            return f.depth.astype(np.float32), f.K, "sensor", None
        if f.mono is None:
            return np.zeros((f.K.height, f.K.width), np.float32), f.K, "none", None
        d, Kg, desc, _ = f.mono()
        return (d * self.scale.k).astype(np.float32), Kg, "mono", desc

    def add(self, f: FrameIn, feats: Features, pl: Placement, depth: NDArray[np.float32],
            K_grid: Intrinsics, kind: str, descriptor: Any) -> Keyframe:
        """Keep a placed frame as a keyframe and fuse it into the cloud."""
        assert pl.T is not None and pl.sigma_pos_m is not None and pl.sigma_rot_deg is not None
        kid = f"k{self.next_id:06d}"
        self.next_id += 1
        prior = None
        if f.prior is not None and f.prior["trust"] in ("metric", "drifting"):
            prior = {"trust": f.prior["trust"], "T": np.asarray(f.prior["T_map_cam"]).tolist(),
                     "sigma_pos_m": _prior_sigmas(f.prior)[0],
                     "sigma_rot_deg": _prior_sigmas(f.prior)[1]}
        valid = depth[np.isfinite(depth) & (depth > 0)]
        kf = Keyframe(kid, f.name, f.camera_id, f.K, pl.T.copy(), pl.sigma_pos_m,
                      pl.sigma_rot_deg, "provisional" if pl.status == "provisional" else "fused",
                      prior, kind, self.scale.k if kind == "mono" else 1.0,
                      float(np.median(valid)) if valid.size else None, pl.method or "prior",
                      pl.inliers)
        kf.feats = feats
        kf.kdepth = sample_depth(depth, K_grid, f.K, feats.kps) if len(feats.kps) \
            else np.zeros(0, np.float32)
        kf.descriptor = None if descriptor is None else np.asarray(descriptor, np.float32)
        self.keyframes.append(kf)
        self._added.append(kf)
        pl.keyframe = kid
        if kind != "none":
            self._fuse(f, kf, depth, K_grid)
        return kf

    def _fuse(self, f: FrameIn, kf: Keyframe, depth: NDArray[np.float32],
              K_grid: Intrinsics) -> None:
        from oh_my_slam.reconstruction.pointcloud import pixel_points

        ok = edge_free(depth) & (depth <= FUSE_MAX_DEPTH_M)
        lattice = np.zeros_like(ok)
        lattice[::FUSE_STRIDE, ::FUSE_STRIDE] = True
        mask = ok & lattice
        if f.ignore is not None and f.ignore.any():
            mask &= cv2.resize(f.ignore.astype(np.uint8), (depth.shape[1], depth.shape[0]),
                               interpolation=cv2.INTER_NEAREST) == 0
        if not mask.any():
            return
        pts, idx = pixel_points(depth, K_grid, mask)
        rgb = cv2.resize(f.rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)
        col = rgb.reshape(-1, 3)[idx].astype(np.float64)
        self.cloud.add(pts @ kf.T[:3, :3].T + kf.centre, col)

    def _height_evidence(self, f: FrameIn, T: Array, depth: NDArray[np.float32],
                         K_grid: Intrinsics) -> None:
        """A monocular keyframe with a known camera height: the height over its raw floor."""
        if f.height_m is None or f.height_m <= 0:
            return
        from oh_my_slam.reconstruction.pointcloud import pixel_points

        ok = edge_free(depth)
        if not ok.any():
            return
        pts, _ = pixel_points(depth / self.scale.k, K_grid, ok)
        dz = (pts @ T[:3, :3].T)[:, 2]
        floor = lowest_layer(dz, -MIN_CAMERA_HEIGHT)
        if floor is None:
            return
        sig = f.height_sigma_m if f.height_sigma_m else DEFAULT_HEIGHT_SIGMA_M
        self.scale.heights = [*self.scale.heights,
                              [f.height_m / -floor, sig / f.height_m]][-MAX_EVIDENCE:]

    # -- an update -----------------------------------------------------------------------------

    def update(self, frames: list[FrameIn], progress: Callable[[str], None] = log.info
               ) -> list[Placement]:
        """Register and fuse ``frames`` (request order) into the map; the first frame with a
        prior defines the origin of a new map."""
        t0 = time.perf_counter()
        out: dict[int, Placement] = {}
        order = list(range(len(frames)))
        if not self.keyframes:
            first = next((i for i, f in enumerate(frames) if f.prior is not None
                          and f.prior["trust"] in ("metric", "drifting")), None)
            if first is None:
                raise ValueError("an anchored map starts from a frame with a prior")
            order = [first] + [i for i in order if i != first]
        reg_s = 0.0
        fuse_s = 0.0
        for i in order:
            f = frames[i]
            t = time.perf_counter()
            mono = None
            if (f.depth is None or not np.count_nonzero(f.depth > 0)) and f.mono is not None \
                    and (f.prior is None or f.check_close):
                # the retrieval descriptor finds candidates without a prior; a suspicious
                # frame's depth may say it is too close
                mono = f.mono()
                d = mono[0][np.isfinite(mono[0]) & (mono[0] > 0)]
                if f.check_close and d.size and float(np.median(d)) * self.scale.k < TOO_CLOSE_M:
                    out[i] = Placement(f.name, "rejected", "too_close")
                    reg_s += time.perf_counter() - t
                    continue
            feats = features(f.rgb, f.ignore)
            if not self.keyframes:
                p = f.prior
                assert p is not None
                P = np.asarray(p["T_map_cam"], np.float64)
                sp, sr = _prior_sigmas(p)
                pl = Placement(f.name, "registered", None, P, sp, sr, 0, "origin", (0.0, 0.0))
                self.created_with = p["trust"]
            else:
                pl = self.place(f, feats, None if mono is None else mono[2])
            reg_s += time.perf_counter() - t
            out[i] = pl
            if pl.status == "rejected":
                continue
            t = time.perf_counter()
            depth, Kg, kind, desc = self._keyframe_depth(f)
            if kind == "mono":
                if not self.keyframes:
                    self._up_check(f, pl)
                assert pl.T is not None
                # its own scale brought onto the map's (the monocular model's metres vary from
                # frame to frame), on the points lifted at the map's current scale
                if pl.lifted is not None:
                    a = depth_alignment(pl.T, *pl.lifted, depth, Kg, f.K)
                    if a is not None:
                        depth = (depth * a).astype(np.float32)
                # its camera height, if known, refines the depth scale before it is fused
                self._height_evidence(f, pl.T, depth, Kg)
                k = self.scale.k
                self.scale.estimate(self.prior_span(), self.prior_span("drifting"))
                depth = (depth * (self.scale.k / k)).astype(np.float32)
            self.add(f, feats, pl, depth, Kg, kind, desc)
            if f.prior is not None and f.prior["trust"] == "drifting" and pl.T is not None \
                    and pl.status == "registered":
                self.drift.confirm(pl.T, np.asarray(f.prior["T_map_cam"], np.float64))
            fuse_s += time.perf_counter() - t
            progress(f"{f.name}: {pl.status} ({pl.method}, {pl.inliers} inliers)")
        self.scale.estimate(self.prior_span(), self.prior_span("drifting"))
        self.timings = {"register": round(reg_s, 3), "fuse": round(fuse_s, 3),
                        "total": round(time.perf_counter() - t0, 3)}
        return [out[i] for i in range(len(frames))]

    def _up_check(self, f: FrameIn, pl: Placement) -> None:
        """The angle between the origin prior's up axis and the monocular gravity estimate."""
        assert f.mono is not None and pl.T is not None
        up = f.mono()[3]
        if up is None:
            return
        u = pl.T[:3, :3] @ unit(np.asarray(up, np.float64))
        self.up_check_deg = round(math.degrees(math.acos(max(-1.0, min(1.0, float(u[2]))))), 3)

    def floor_and_bounds(self) -> tuple[float | None, NDArray[np.float64]]:
        """The floor height (lowest dense layer of the cloud below every camera) and the points
        bounding the map (the cloud's and the cameras')."""
        pts = self.cloud.points()
        centres = np.array([k.centre for k in self.keyframes]).reshape(-1, 3)
        below = float(centres[:, 2].min()) if len(centres) else 0.0
        floor = lowest_layer(pts[:, 2], below) if len(pts) else None
        return floor, np.vstack([pts, centres]) if len(pts) else centres
