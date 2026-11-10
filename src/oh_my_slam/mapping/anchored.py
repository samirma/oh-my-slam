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

import itertools
import json
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.atomic import atomic_write_json
from oh_my_slam.core.geometry import project, unproject_pixels, voxel_keys
from oh_my_slam.core.log import get_logger
from oh_my_slam.core.types import Intrinsics

log = get_logger("oh_my_slam.mapping.anchored")

Array = NDArray[np.float64]

ENGINE = "anchored"
STATE_JSON = "state.json"
CLOUD_NPZ = "cloud.npz"
KF_DIR = "kf"
KF_JPEG_QUALITY = 92

MAX_FEATURES = 4000
SIFT_CONTRAST = 0.01  # a quarter of OpenCV's default: simulated and indoor robot frames are low
# in texture (a few hundred features at the default)
ROOTSIFT_QUANT = 512.0
RATIO = 0.8  # Lowe's ratio test
GUIDED_RATIO = 0.9  # the ratio test among the keypoints near a match's predicted pixel
GUIDED_MIN_PX = 6.0  # the predicted pixel's search radius: at least this
GUIDED_MAX_PX = 80.0  # ... at most this
GUIDED_CHUNK = 65536  # descriptor pairs compared at once
GUIDED_ENOUGH = 300  # a keyframe with this many ratio-test matches needs no guided matching
MAX_CANDIDATES = 5
CANDIDATE_MAX_ANGLE_DEG = 100.0  # optical axes further apart share no view
CANDIDATE_MAX_DIST_M = 5.0
CANDIDATE_DIST_UNIT_M = 0.5  # candidate score = distance / unit + angle / unit
CANDIDATE_ANGLE_UNIT_DEG = 30.0
RECENT_CANDIDATES = 8  # without prior or descriptor: the most recent keyframes
RECENT_GUIDED = 3  # without prior: candidates near the last confirmed keyframe added
NO_PRIOR_MIN_INLIERS = 15  # a frame without prior is registered on this many inliers...
NO_PRIOR_MIN_RATIO = 0.15  # ... and this share of its matches
ROTATION_FALLBACK_M = 0.20  # PnP failed: rotation-only still tried up to this distance
ROTATION_FALLBACK_REL = 0.25  # ... or this share of the keyframe's median depth
REPROJ_PX = 3.0  # inlier threshold at 640 px width, scaled with the image width
REF_WIDTH = 640.0
MIN_INLIERS = 15
PIXEL_SIGMA = 1.0  # a match's pixel noise at 640 px width
DEPTH_NOISE_REL = 0.02  # a lifted point's own depth noise (the scale error is a parameter)
SENSOR_SCALE_SIGMA_REL = 0.03  # a depth image's scale, as a prior of its reference's scale
STRONG_INLIERS = 40  # an image pose this well supported can contradict a prior
STRONG_RATIO = 0.3  # ... with this share of its matches as inliers
CONFIRM_SIGMA_POS_M = 0.10  # an image pose confirms a frame only this well determined (or
# within twice the prior's sigma)
CONFIRM_SIGMA_ROT_DEG = 2.0
METRIC_IMAGE_SIGMA_POS_M = 0.05  # an image pose weighed against a metric prior: at least this
METRIC_IMAGE_SIGMA_ROT_DEG = 1.0
K_SIGMA = 3.0
MIN_SIGMA_POS_M = 0.01
MIN_SIGMA_ROT_DEG = 0.1
IMAGE_SIGMA_POS_REL = 0.01  # of the median scene depth: depth bias the covariance does not see
IMAGE_SIGMA_ROT_MIN_DEG = 0.2
REPORT_MIN_SIGMA_POS_M = 0.001  # floors of the reported sigmas
REPORT_MIN_SIGMA_ROT_DEG = 0.01
DEFAULT_SIGMA_POS_M = 0.05  # a prior without sigmas (as mapping.anchor)
DEFAULT_SIGMA_ROT_DEG = 2.0
UNANCHORED_SIGMA_POS_M = 0.10  # rotation-only without a prior: the centre is the keyframe's
DRIFT_POS_REL = 0.10  # drifting prior: sigma grows by this share of the travel
DRIFT_ROT_REL = 0.05  # ... by this share of the turn
DRIFT_ROT_DEG_PER_M = 2.0  # ... and this much per metre travelled
ODO_SCALE_MIN_TRAVEL_M = 0.2
ODO_SCALE_RANGE = (0.5, 2.0)
PRIOR_SPAN_M = 0.3  # metric priors fix the scale once their centres span this
BIAS_SPAN_M = 0.5  # metric priors' own scale bias is judged once they span this
BIAS_EVIDENCE_MIN = 5
EVIDENCE_SCALE_SIGMA = 1.0  # reference scales left free when a metric prior measures them
SCALE_EVIDENCE_MIN_M = 0.1  # a PnP baseline shorter than this tells little about the scale
SCALE_EVIDENCE_MIN = 3
SCALE_RANGE = (0.2, 5.0)
MAX_EVIDENCE = 500
DEPTH_SIGMA_REL = 0.01  # depth images (README "Prior-first registration")
MONOCULAR_SIGMA_REL = 0.10
MIN_SCALE_SIGMA_REL = 0.002
DEFAULT_HEIGHT_SIGMA_M = 0.02
MIN_DEPTH_M = 0.05
SAMPLE_SIDE = 64  # a depth agreement samples a grid of at most this side
SENSOR_DEPTH_TOL = 0.10  # relative depth agreement of depth images
MONO_DEPTH_TOL = 0.20  # ... of monocular depth (its own scale is aligned per frame)
MIN_OVERLAP = 0.10  # share of a frame's points a reference view's depth sees
MIN_AGREEMENT = 0.75  # share of those that agree
MIN_AGREEING = 40  # points
ICP_SIGMA_M = 0.01  # a depth image point's noise (plus DEPTH_NOISE_REL of its depth)
ICP_MAX_M = 0.10  # a point's match at most this far (plus the prior's reach, at first)
ICP_ITERATIONS = 12
EDGE_REL = 0.06  # a depth sample whose 3x3 neighbourhood spans more than this is a depth edge
GN_ITERATIONS = 15
BEHIND_PX = 1e6  # the pixel of a point behind the camera (a residual Huber ignores)
HUBER_PX = 2.0
HUBER_SIGMAS = 2.0
MIN_PIXEL_SIGMA = 0.5
VOXEL_M = 0.02
FUSE_STRIDE = 2
FUSE_MAX_DEPTH_M = 6.0
FLOOR_BINS = 50
FLOOR_PEAK_SHARE = 0.25
FLOOR_MIN_POINTS = 200
FLOOR_PEAK_MIN = 20.0  # points per 1 % height bin for a horizontal surface
FLOOR_BAND_REL = 0.05  # points this near a peak's height take part in its plane
FLOOR_PLANE_TOL_REL = 0.02
FLOOR_TILT_DEG = 10.0
FLOOR_RANSAC = 100
FLOOR_SAMPLE = 5000  # points a plane is fitted on
HEIGHT_VOTE_BIN = 0.03  # log scale: the frames' floor candidates vote in bins this wide
HEIGHT_VOTE_TOL = 0.15  # a frame's candidate this near the vote counts
FLOOR_CANDIDATES = 8  # the most supported height peaks tried as planes
MAP_FLOOR_SHARE = 0.5  # the map's floor: the lowest plane with this share of the best one's
TOO_CLOSE_M = 0.30  # median depth (as mapping.checks.TOO_CLOSE_M)
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
    return unproject_pixels(uv[:, 0], uv[:, 1], np.ones(len(uv)), K.K())


def unit(v: Array) -> Array:
    out: Array = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return out


def first_camera(up_cam: Any) -> Array:
    """The pose of a camera-only map's first frame (map.md §1, without priors): at the origin,
    +z up along the gravity it sees (``up_cam``, the up direction in its optical frame; level
    when unknown), +x its viewing direction on the horizontal plane."""
    up = unit(np.asarray((0.0, -1.0, 0.0) if up_cam is None else up_cam, np.float64))
    fwd = np.array([0.0, 0.0, 1.0]) - up[2] * up  # the optical axis on the horizontal plane
    x = unit(fwd)
    M = np.stack([x, np.cross(up, x), up])  # rows: the map's axes in the optical frame
    return pose(M, np.zeros(3))


def floor_candidates(p: Array, share: float = 0.25) -> list[float]:
    """The camera's heights above every strong horizontal plane below it, in its own points
    ``p`` (camera-centred, z up along gravity, unscaled), best supported first: the peaks of the
    points' log heights, each tried as a plane (``_horizontal_plane``), kept when it holds
    ``share`` of the best one's points. Which one is the floor is decided across
    frames (``ScaleState._height_votes``): fitted per frame, so a frame that sees a lower floor
    through a door or a counter offers it without moving the others."""
    out: list[float] = []
    q = p[np.isfinite(p).all(axis=1) & (p[:, 2] < -MIN_CAMERA_HEIGHT)]
    if len(q) < FLOOR_MIN_POINTS:
        return out
    z = np.log(-q[:, 2])
    hist, edges = np.histogram(z, bins=max(3, int((z.max() - z.min()) / 0.01) + 1))
    smooth = np.convolve(hist, np.ones(3) / 3.0, mode="same")
    peaks = [i for i in range(len(smooth))
             if smooth[i] >= FLOOR_PEAK_MIN and smooth[i] == smooth[max(i - 2, 0):i + 3].max()]
    centres = (edges[:-1] + edges[1:]) / 2.0
    planes = [got for i in sorted(peaks, key=lambda j: -smooth[j])[:FLOOR_CANDIDATES]
              if (got := _horizontal_plane(q, math.exp(centres[i]))) is not None]
    if planes:
        top = max(n for _, n in planes)
        out = [h for h, n in sorted(planes, key=lambda hn: -hn[1]) if n >= share * top]
    return out


def _horizontal_plane(q: Array, h0: float) -> tuple[float, int] | None:
    """The plane (RANSAC, normal within ``FLOOR_TILT_DEG`` of vertical) among the points near
    height ``-h0``: its distance to the camera and its inliers, or None."""
    near = q[np.abs(-q[:, 2] - h0) <= FLOOR_BAND_REL * h0]
    if len(near) < FLOOR_MIN_POINTS // 2:
        return None
    rng = np.random.default_rng(0)
    best: tuple[int, Array, float] | None = None
    tol = FLOOR_PLANE_TOL_REL * h0
    if len(near) > FLOOR_SAMPLE:
        near = near[rng.integers(0, len(near), FLOOR_SAMPLE)]
    for _ in range(FLOOR_RANSAC):
        a, b, c = near[rng.integers(0, len(near), 3)]
        nrm = np.cross(b - a, c - a)
        if np.linalg.norm(nrm) < 1e-12:
            continue
        nrm = nrm / np.linalg.norm(nrm)
        if abs(nrm[2]) < math.cos(math.radians(FLOOR_TILT_DEG)):
            continue
        d = -float(nrm @ a)
        cnt = int((np.abs(near @ nrm + d) < tol).sum())
        if best is None or cnt > best[0]:
            best = (cnt, nrm, d)
    if best is None or best[0] < FLOOR_MIN_POINTS // 2:
        return None
    _, nrm, d = best
    inl = near[np.abs(near @ nrm + d) < tol]
    c0 = inl.mean(axis=0)
    _, _, Vt = np.linalg.svd(inl - c0)
    nrm = Vt[2] * np.sign(Vt[2][2])
    return float(abs(nrm @ c0)), len(inl)


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
    g, _ = project(rays(K, uv), K_grid.K())
    u = np.round(g[:, 0]).astype(np.intp)
    v = np.round(g[:, 1]).astype(np.intp)
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
        # the largest singular value of a covariance block is its largest variance
        sp = math.sqrt(float(np.linalg.svd(self.cov[3:, 3:], compute_uv=False)[0]))
        sr = math.degrees(math.sqrt(float(np.linalg.svd(self.cov[:3, :3], compute_uv=False)[0])))
        return sp, sr


def _project(R: Array, c: Array, X: Array, K: Intrinsics, directions: bool) -> Array:
    """Pixels of world points (or directions) in the camera ``R``, ``c``; a point behind the
    camera lands far off the image."""
    uv, _ = project((X - (0.0 if directions else c)) @ R, K.K())  # camera frame: R^T (X - c)
    out: Array = np.nan_to_num(uv, nan=BEHIND_PX)
    return out


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


def guided_match(new: Features, ref: Features, predicted: Array, radius: Array
                 ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Matches of ``ref``'s keypoints, each looked for only among ``new``'s keypoints within
    ``radius`` px of its ``predicted`` pixel in the new frame (by the prior), with a ratio test
    among those: the low-texture frames of a robot give too few matches that pass the ratio
    test against the whole image. One match per new keypoint (the closest)."""
    from scipy.spatial import cKDTree

    ok = np.isfinite(predicted).all(axis=1)
    empty = np.zeros(0, np.intp), np.zeros(0, np.intp)
    if len(new.kps) < 2 or not ok.any():
        return empty
    jb = np.nonzero(ok)[0]
    near = cKDTree(new.kps).query_ball_point(predicted[jb], radius[jb])
    lens = np.fromiter((len(x) for x in near), np.intp, len(near))
    if not lens.sum():
        return empty
    ii = np.fromiter(itertools.chain.from_iterable(near), np.intp, int(lens.sum()))
    jj = np.repeat(jb, lens)
    a = new.desc.astype(np.float32)
    b = ref.desc.astype(np.float32)
    d = np.empty(len(ii), np.float32)
    for s in range(0, len(ii), GUIDED_CHUNK):
        e = s + GUIDED_CHUNK
        d[s:e] = np.linalg.norm(a[ii[s:e]] - b[jj[s:e]], axis=1)
    o = np.lexsort((d, jj))  # by reference keypoint, then distance
    ii, jj, d = ii[o], jj[o], d[o]
    first = np.nonzero(np.r_[True, jj[1:] != jj[:-1]])[0]
    count = np.diff(np.r_[first, len(jj)])
    d1 = d[first]
    d2 = np.where(count > 1, d[np.minimum(first + 1, len(d) - 1)], np.inf)
    keep = d1 < GUIDED_RATIO * d2
    i1, j1, d1 = ii[first][keep], jj[first][keep], d1[keep]
    if not len(i1):
        return empty
    o = np.lexsort((j1, d1, i1))  # one per new keypoint: the closest (the first on ties)
    i1, j1 = i1[o], j1[o]
    one = np.r_[True, i1[1:] != i1[:-1]]
    return i1[one].astype(np.intp), j1[one].astype(np.intp)


@dataclass
class Refs:
    """The reference keyframes of a frame's matches: centres, the sigma of each one's depth
    scale and its median depth."""

    c: Array  # n x 3
    scale_sigma: Array  # n
    median: Array  # n


def refine_pose(R0: Array, c0: Array, uv: Array, X: Array, d: Array, ref: NDArray[np.intp],
                refs: Refs, K: Intrinsics, *, prior: tuple[Array, float, float] | None = None,
                fix_centre: bool = False) -> tuple[Solve, Array]:
    """The pose of a frame (stored poses fixed) on its matches to reference keyframes: a robust
    Gauss-Newton (Huber at 2 sigmas) over the rotation, the centre (unless ``fix_centre``) and
    one log depth scale per reference, with

    * a lifted point (``X`` finite) reprojected after its reference's depth scale; its noise is
      the pixel noise and its own depth noise times the parallax its reference's distance gives;
    * a match without depth reprojected as a direction (``d``), its noise grown by the full
      parallax of its reference's distance (exact for a camera turning in place);
    * each reference's depth scale held at 1 by its sigma (a depth image's, the monocular
      scale's), so one wrong depth scale moves the centre along its distance to that reference
      only as far as the other factors let it;
    * the pose prior (``prior``: T, sigma_pos_m, sigma_rot_deg) when given.

    Returns the solve (its covariance: rotation vector, centre) and the depth scales."""
    n_ref = len(refs.c)
    n_pose = 3 if fix_centre else 6
    n = n_pose + n_ref
    has = np.isfinite(X).all(axis=1)
    Xh = np.where(has[:, None], X, 0.0)
    f = math.sqrt(K.fx * K.fy)
    px = PIXEL_SIGMA * K.width / REF_WIDTH
    cr = refs.c[ref]

    def unpack(p: Array) -> tuple[Array, Array, Array]:
        R = exp_so3(p[:3]) @ R0
        c = c0 if fix_centre else c0 + p[3:6]
        return R, c, np.exp(p[n_pose:])

    def sigmas(c: Array) -> Array:
        b = np.linalg.norm(c - cr, axis=1) / refs.median[ref]
        return np.sqrt(px ** 2 + (f * b * np.where(has, DEPTH_NOISE_REL, 1.0)) ** 2)

    def obs(p: Array, sig: Array) -> Array:
        R, c, sc = unpack(p)
        Xs = cr + sc[ref][:, None] * (Xh - cr)
        pred = np.where(has[:, None], _project(R, c, Xs, K, False),
                        _project(R, c, d, K, True))
        return ((pred - uv) / sig[:, None]).ravel()

    def extra(p: Array) -> Array:
        R, c, _ = unpack(p)
        out = [p[n_pose:] / refs.scale_sigma]
        if prior is not None:
            T, sp, sr = prior
            out.append(log_so3(R @ T[:3, :3].T) / math.radians(sr))
            out.append((c - T[:3, 3]) / sp)
        return np.concatenate(out)

    p = np.zeros(n)
    eps = 1e-6
    sig = sigmas(c0)
    w = np.ones(len(uv))
    J = np.zeros((2 * len(uv), n))
    Je = np.zeros((0, n))
    for _ in range(GN_ITERATIONS):
        sig = sigmas(unpack(p)[1])
        r = obs(p, sig)
        re = extra(p)
        e = np.hypot(r[0::2], r[1::2])
        w = np.where(e <= HUBER_SIGMAS, 1.0, HUBER_SIGMAS / np.maximum(e, 1e-12))
        J = np.zeros((len(r), n))
        Je = np.zeros((len(re), n))
        for k in range(n):
            q = p.copy()
            q[k] += eps
            J[:, k] = (obs(q, sig) - r) / eps
            Je[:, k] = (extra(q) - re) / eps
        W = np.repeat(w, 2)
        H = J.T @ (W[:, None] * J) + Je.T @ Je
        delta = -np.linalg.solve(H + 1e-9 * np.eye(n), J.T @ (W * r) + Je.T @ re)
        p = p + delta
        if float(np.abs(delta).max()) < 1e-9:
            break
    r = obs(p, sig)
    e = np.hypot(r[0::2], r[1::2])
    inl = e <= 3.0
    chi = float(np.mean(e[inl] ** 2) / 2.0) if inl.sum() > n else 1.0
    W = np.repeat(w, 2)
    H = J.T @ (W[:, None] * J) + Je.T @ Je
    cov = np.linalg.inv(H + 1e-9 * np.eye(n)) * max(chi, 1.0)
    cov6 = np.zeros((6, 6))
    cov6[:n_pose, :n_pose] = cov[:n_pose, :n_pose]
    R, c, sc = unpack(p)
    rms = float(np.sqrt(np.mean((e[inl] * px) ** 2))) if inl.any() else float("inf")
    return Solve(R, c, cov6, inl, rms), sc


# -- depth agreement ------------------------------------------------------------------------------


def backproject(depth: NDArray[np.float32], K: Array, side: int = SAMPLE_SIDE) -> Array:
    """Camera-frame points of a depth map (3x3 ``K`` of its grid), on a grid of at most
    ``side`` per axis."""
    h, w = depth.shape
    step = max(1, math.ceil(max(h, w) / side))
    vs, us = np.mgrid[step // 2:h:step, step // 2:w:step]
    z = depth[vs, us].astype(np.float64)
    ok = np.isfinite(z) & (z > MIN_DEPTH_M)
    z, us, vs = z[ok], us[ok], vs[ok]
    out: Array = unproject_pixels(us + 0.5, vs + 0.5, z, K)
    return out


@dataclass
class DepthRef:
    """A reference view of a depth agreement: a pose and a metric depth on its grid (3x3)."""

    T: Array
    depth: NDArray[np.float32]
    K: Array


@dataclass
class Agreement:
    points: int
    agree: int
    conflict: int
    median_abs_m: float = 0.0  # median depth difference of the points references see (m)
    median_ratio: float = 1.0  # median ratio of the references' depth to the points' there

    @property
    def overlap(self) -> float:
        return (self.agree + self.conflict) / self.points if self.points else 0.0

    @property
    def agreement(self) -> float:
        n = self.agree + self.conflict
        return self.agree / n if n else 0.0

    @property
    def confirmed(self) -> bool:
        return (self.agree >= MIN_AGREEING and self.overlap >= MIN_OVERLAP
                and self.agreement >= MIN_AGREEMENT)

    @property
    def contradicted(self) -> bool:
        """Enough points seen by the references, too few of them where they see a surface."""
        return (self.agree + self.conflict >= MIN_AGREEING and self.overlap >= MIN_OVERLAP
                and self.agreement < MIN_AGREEMENT)


def agreement(points_cam: Array, T: Array, refs: list[DepthRef], tol: float) -> Agreement:
    """How the camera-frame points of a frame at pose ``T`` agree with the references' depth: a
    point agrees when some reference sees it within ``tol`` (relative) of its depth, and
    conflicts when references see it but none agrees (in front of their surface, or behind it:
    the small baselines of a robot's frames leave little to occlusion)."""
    n = len(points_cam)
    if not n or not refs:
        return Agreement(n, 0, 0)
    X = points_cam @ T[:3, :3].T + T[:3, 3]
    agree = np.zeros(n, bool)
    conflict = np.zeros(n, bool)
    best = np.full(n, np.inf)
    ratio = np.full(n, np.nan)
    for r in refs:
        Xr = (X - r.T[:3, 3]) @ r.T[:3, :3]
        z = Xr[:, 2]
        front = z > MIN_DEPTH_M
        uv, _ = project(np.where(front[:, None], Xr, 1.0), r.K)
        uv = np.floor(np.nan_to_num(uv, nan=-1.0, posinf=-1.0, neginf=-1.0))
        u, v = uv[:, 0].astype(np.int64), uv[:, 1].astype(np.int64)
        h, w = r.depth.shape
        inside = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        d = np.zeros(n)
        d[inside] = r.depth[v[inside], u[inside]]
        seen = inside & (d > MIN_DEPTH_M) & np.isfinite(d)
        rel = np.where(seen, (z - d) / np.where(seen, d, 1.0), 0.0)
        agree |= seen & (np.abs(rel) <= tol)
        conflict |= seen & (np.abs(rel) > tol)
        closer = seen & (np.abs(z - d) < best)
        ratio = np.where(closer, d / np.where(seen, z, 1.0), ratio)
        best = np.where(closer, np.abs(z - d), best)
    conflict &= ~agree
    seen_any = np.isfinite(best)
    if not seen_any.any():
        return Agreement(n, int(agree.sum()), int(conflict.sum()))
    return Agreement(n, int(agree.sum()), int(conflict.sum()), float(np.median(best[seen_any])),
                     float(np.median(ratio[seen_any])))


def _normals(depth: NDArray[np.float32], K: Array, T: Array, side: int = SAMPLE_SIDE
             ) -> tuple[Array, Array]:
    """World points and unit normals of a depth grid seen from ``T``, on a grid of at most
    ``side`` per axis (central differences; points without a full neighbourhood dropped)."""
    h, w = depth.shape
    step = max(1, math.ceil(max(h, w) / side))
    vs, us = np.mgrid[step // 2:h:step, step // 2:w:step]
    z = depth[vs, us].astype(np.float64)
    P = unproject_pixels(us + 0.5, vs + 0.5, z, K)
    ok = np.isfinite(z) & (z > MIN_DEPTH_M)
    du = P[1:-1, 2:] - P[1:-1, :-2]
    dv = P[2:, 1:-1] - P[:-2, 1:-1]
    n = np.cross(du, dv)
    norm = np.linalg.norm(n, axis=-1)
    good = (ok[1:-1, 1:-1] & ok[1:-1, 2:] & ok[1:-1, :-2] & ok[2:, 1:-1] & ok[:-2, 1:-1]
            & (norm > 1e-12))
    # a depth edge is no surface: its neighbours more than EDGE_REL apart
    zc = z[1:-1, 1:-1]
    span = np.max(np.abs(np.stack([z[1:-1, 2:], z[1:-1, :-2], z[2:, 1:-1], z[:-2, 1:-1]])
                         - zc), axis=0)
    good &= span <= 3.0 * EDGE_REL * np.maximum(zc, MIN_DEPTH_M)
    pc, nc = P[1:-1, 1:-1][good], n[good] / norm[good][:, None]
    return pc @ T[:3, :3].T + T[:3, 3], nc @ T[:3, :3].T


def depth_icp(pts: Array, refs: list[DepthRef], T0: Array, sp0: float, sr0: float
              ) -> tuple[Array, float, float, int] | None:
    """A camera's pose from its camera-frame depth points ``pts`` against the references'
    depth: point-to-plane Gauss-Newton from ``T0`` with the prior (``sp0`` m, ``sr0`` deg) as
    a factor, each point's noise ``ICP_SIGMA_M`` + ``DEPTH_NOISE_REL`` of its depth, matches
    within ``K_SIGMA`` prior sigmas plus ``ICP_MAX_M``, Huber. Returns (T, sigma_pos_m,
    sigma_rot_deg, matched points), or None with fewer than ``MIN_AGREEING`` matches."""
    from scipy.spatial import cKDTree

    got = [_normals(r.depth, r.K, r.T) for r in refs]
    Q = np.vstack([g[0] for g in got])
    N = np.vstack([g[1] for g in got])
    if len(Q) < MIN_AGREEING or len(pts) < MIN_AGREEING:
        return None
    tree = cKDTree(Q)
    sig = ICP_SIGMA_M + DEPTH_NOISE_REL * pts[:, 2]
    R, c = T0[:3, :3].copy(), T0[:3, 3].copy()
    sr = math.radians(sr0)
    H = np.eye(6)
    n_in = 0
    for it in range(ICP_ITERATIONS):
        X = pts @ R.T + c
        reach = K_SIGMA * sp0 + ICP_MAX_M if it < ICP_ITERATIONS // 2 else ICP_MAX_M
        d, j = tree.query(X, distance_upper_bound=reach)
        m = np.isfinite(d)
        n_in = int(m.sum())
        if n_in < MIN_AGREEING:
            return None
        Xm, n_m, s_m = X[m], N[j[m]], sig[m]
        r = ((Xm - Q[j[m]]) * n_m).sum(1) / s_m
        w = np.where(np.abs(r) <= HUBER_SIGMAS, 1.0, HUBER_SIGMAS / np.maximum(np.abs(r), 1e-12))
        # left perturbation: X' = exp(w) (X - c) + c + dc
        J = np.hstack([np.cross(Xm - c, n_m), n_m]) / s_m[:, None]
        rp = np.concatenate([log_so3(R @ T0[:3, :3].T) / sr, (c - T0[:3, 3]) / sp0])
        Jp = np.diag(np.r_[np.full(3, 1.0 / sr), np.full(3, 1.0 / sp0)])
        H = J.T @ (w[:, None] * J) + Jp.T @ Jp
        g = J.T @ (w * r) + Jp.T @ rp
        delta = -np.linalg.solve(H + 1e-9 * np.eye(6), g)
        R = exp_so3(delta[:3]) @ R
        c = c + delta[3:]
        if float(np.abs(delta).max()) < 1e-7:
            break
    cov = np.linalg.inv(H + 1e-9 * np.eye(6))
    sp = math.sqrt(float(np.linalg.svd(cov[3:, 3:], compute_uv=False)[0]))
    srd = math.degrees(math.sqrt(float(np.linalg.svd(cov[:3, :3], compute_uv=False)[0])))
    return pose(R, c), sp, srd, n_in


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
    # set while the update runs, written with it: the depth grid as fused and the image
    grid: NDArray[np.float32] | None = None
    K_grid: Intrinsics | None = None
    rgb: NDArray[np.uint8] | None = None

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "camera_id": self.camera_id,
                "K": self.K.to_dict(), "T": self.T.tolist(), "sigma_pos_m": self.sigma_pos_m,
                "sigma_rot_deg": self.sigma_rot_deg, "state": self.state, "prior": self.prior,
                "depth_kind": self.depth_kind, "k_used": self.k_used,
                "median_depth": self.median_depth, "method": self.method,
                "inliers": self.inliers,
                "K_grid": None if self.K_grid is None else self.K_grid.to_dict()}

    @staticmethod
    def from_json(d: dict[str, Any]) -> Keyframe:
        return Keyframe(d["id"], d["name"], d["camera_id"], Intrinsics.from_dict(d["K"]),
                        np.asarray(d["T"], np.float64), float(d["sigma_pos_m"]),
                        float(d["sigma_rot_deg"]), d["state"], d["prior"], d["depth_kind"],
                        float(d["k_used"]), d["median_depth"], d["method"], int(d["inliers"]),
                        K_grid=None if d.get("K_grid") is None
                        else Intrinsics.from_dict(d["K_grid"]))

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
        q = voxel_keys(p, VOXEL_M) + (1 << 20)
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

    def drop(self, mask: NDArray[np.bool_]) -> None:
        """Remove the voxels ``mask`` selects (a struck object's place seen through)."""
        keep = ~np.asarray(mask, bool)
        self.keys, self.xyz, self.rgb, self.count = (self.keys[keep], self.xyz[keep],
                                                     self.rgb[keep], self.count[keep])

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
    with a drifting prior (its map pose and its prior), the odometry's scale, and the odometry
    travelled since that keyframe along every drifting prior seen after it (``observe``): a
    robot that drove away and came back has drifted with its path, not with the straight line
    between its two priors."""

    ref_T: Array | None = None
    ref_prior: Array | None = None
    odo_scale: float = 1.0
    evidence: list[list[float]] = field(default_factory=list)  # [odometry m, map m]
    last_prior: Array | None = None  # the last drifting prior seen
    path_m: float = 0.0  # odometry path from ``ref_prior`` to ``last_prior``
    turn_deg: float = 0.0  # ... and its turns

    def to_json(self) -> dict[str, Any]:
        return {"ref_T": None if self.ref_T is None else self.ref_T.tolist(),
                "ref_prior": None if self.ref_prior is None else self.ref_prior.tolist(),
                "odo_scale": self.odo_scale, "evidence": self.evidence,
                "last_prior": None if self.last_prior is None else self.last_prior.tolist(),
                "path_m": self.path_m, "turn_deg": self.turn_deg}

    @staticmethod
    def from_json(d: dict[str, Any]) -> Drift:
        last = d.get("last_prior")
        return Drift(None if d["ref_T"] is None else np.asarray(d["ref_T"], np.float64),
                     None if d["ref_prior"] is None else np.asarray(d["ref_prior"], np.float64),
                     float(d["odo_scale"]), [list(e) for e in d["evidence"]],
                     None if last is None else np.asarray(last, np.float64),
                     float(d.get("path_m", 0.0)), float(d.get("turn_deg", 0.0)))

    def _step(self, P: Array) -> tuple[float, float]:
        """The odometry's travel and turn from the last prior seen to ``P``."""
        if self.last_prior is None:
            return 0.0, 0.0
        D = np.linalg.inv(self.last_prior) @ P
        return float(np.linalg.norm(D[:3, 3])) * self.odo_scale, rot_deg(D[:3, :3])

    def predict(self, P: Array, sigma_pos: float, sigma_rot: float
                ) -> tuple[Array, float, float]:
        """The map pose and sigmas of a drifting prior ``P``: through the latest correction,
        the sigmas grown with the travel since it (the longer of its path and the straight
        line)."""
        if self.ref_T is None or self.ref_prior is None:
            return P.copy(), sigma_pos, sigma_rot
        D = np.linalg.inv(self.ref_prior) @ P
        D[:3, 3] *= self.odo_scale
        step, step_turn = self._step(P)
        travel = max(float(np.linalg.norm(D[:3, 3])), self.path_m + step)
        turn = max(rot_deg(D[:3, :3]), self.turn_deg + step_turn)
        return (self.ref_T @ D, sigma_pos + DRIFT_POS_REL * travel,
                sigma_rot + DRIFT_ROT_REL * turn + DRIFT_ROT_DEG_PER_M * travel)

    def observe(self, P: Array) -> None:
        """A drifting prior seen (whatever its frame became): the path grows by its step."""
        step, step_turn = self._step(P)
        self.path_m += step
        self.turn_deg += step_turn
        self.last_prior = P.copy()

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
        self.last_prior, self.path_m, self.turn_deg = P.copy(), 0.0, 0.0


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

    def _height_votes(self) -> NDArray[np.float64]:
        """Per frame, the log scale of its floor: each frame offers every strong horizontal
        plane below its camera (``heights``: [sigma_rel, scale per plane...]); the scale most
        frames agree on (the mode of all candidates, ``HEIGHT_VOTE_BIN`` wide) picks, in each
        frame, the candidate nearest it (within ``HEIGHT_VOTE_TOL``). A counter or a lower room
        seen through a door is one frame's plane, the floor every frame's."""
        cands = [np.log(np.asarray(h[1:], np.float64)) for h in self.heights]
        allc = np.concatenate(cands)
        bins = np.floor(allc / HEIGHT_VOTE_BIN)
        frames_per_bin: dict[float, int] = {}
        for c in cands:
            for b in set(np.floor(c / HEIGHT_VOTE_BIN).tolist()):
                for nb in (b - 1, b, b + 1):
                    frames_per_bin[nb] = frames_per_bin.get(nb, 0) + 1
        best = max(set(bins.tolist()), key=lambda b: (frames_per_bin[b], -b))
        mode = float(np.median(allc[np.abs(bins - best) <= 1]))
        return np.array([float(c[np.argmin(np.abs(c - mode))]) for c in cands
                         if np.min(np.abs(c - mode)) <= HEIGHT_VOTE_TOL])

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
        """``k`` from the evidence: camera heights (a scale the images measure, independent of
        the priors), else metric priors (once their span allows), else drifting priors (their
        relative motion is metric within the odometry's drift: ``DRIFT_POS_REL`` at best,
        reported as ``priors``), else held at 1."""
        if self.heights:
            lr = self._height_votes()
            rel = np.array([h[0] for h in self.heights])
            mad = 1.4826 * float(np.median(np.abs(lr - np.median(lr))))
            self.k = float(np.clip(np.exp(np.median(lr)), *SCALE_RANGE))
            self.source = "camera_height"
            self.sigma_rel = max(math.sqrt(mad ** 2 + float(np.median(rel)) ** 2)
                                 / math.sqrt(len(lr)), MIN_SCALE_SIGMA_REL)
            return
        got = self._baselines(self.priors, prior_span)
        if got is not None:
            self.k, self.source = got[0], "priors"
            self.sigma_rel = max(got[1], MIN_SCALE_SIGMA_REL)
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
    align_T: Array | None = None  # the image-only pose the lifted points agree with
    # the committed pose against the image-only pose on the stored map: how far the update puts
    # the frame from where the map says it is (the quality report's residuals and gauge shift)
    map_residual: tuple[float, float] | None = None
    image_sigmas: tuple[float, float] | None = None  # of ``align_T`` (a conflict's image pose)


def inherit_conflicts(pls: list[Placement]) -> None:
    """The frames of one update that its priors alone placed — at the prior (``prior``), by
    depth geometry that cannot see every direction (``depth``), or in position by a rotation
    fit (``rotation``) — take as their residual to the map (``map_residual``, the quality
    report's ``previous_residuals`` and gauge shift) at least the largest disagreement between
    the update's priors and the map that its prior conflicts measured: an image pose beyond the
    combined sigmas of its prior (the conflict's ``prior_residual``), or a depth that contradicts
    the map at the prior (its ``map_residual``). Priors that the images show off by that much may
    put the frames nobody confirmed as far off: such an update warps the map, and
    ``max_residual_m`` sees it. An update without a conflict is unchanged."""
    ev = [e for p in pls if p.reason == "prior_conflict"
          for e in [p.prior_residual if p.align_T is not None else p.map_residual]
          if e is not None]
    if not ev:
        return
    ep, er = max(e[0] for e in ev), max(e[1] for e in ev)
    for p in pls:
        if p.status == "rejected" or p.method not in ("prior", "depth", "rotation"):
            continue
        cur = p.map_residual or (0.0, 0.0)
        p.map_residual = (max(cur[0], ep), cur[1] if p.method == "rotation" else max(cur[1], er))


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


def _real_sigmas(s: Solve, refs: Refs) -> tuple[float, float]:
    """An image pose's sigmas as reported: its covariance (``Solve.sigmas``) and the depth bias
    the covariance does not see (``IMAGE_SIGMA_POS_REL`` of the references' median depth)."""
    sp, sr = s.sigmas()
    bias = IMAGE_SIGMA_POS_REL * float(np.median(refs.median))
    return max(math.hypot(sp, bias), REPORT_MIN_SIGMA_POS_M), max(sr, REPORT_MIN_SIGMA_ROT_DEG)


def _given_sigmas(p: dict[str, Any]) -> tuple[float, float]:
    """A prior's sigmas as the robot gave them (defaults without), for the reported sigmas."""
    return (max(float(p.get("sigma_pos_m", DEFAULT_SIGMA_POS_M)), REPORT_MIN_SIGMA_POS_M),
            max(float(p.get("sigma_rot_deg", DEFAULT_SIGMA_ROT_DEG)), REPORT_MIN_SIGMA_ROT_DEG))


def _combined(real_a: float, weight_a: float, real_b: float, weight_b: float) -> float:
    """The sigma of the weighted mean of two independent estimates whose weights are
    ``1/weight²`` and whose real sigmas are ``real_a``, ``real_b``."""
    w = weight_b ** 2 / (weight_a ** 2 + weight_b ** 2)  # the share of estimate a
    return math.hypot(w * real_a, (1.0 - w) * real_b)


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
        self.origin_c: Array | None = None  # the origin's prior centre (metric bias pivot)
        # baselines of metric priors against independently scaled image ones: [prior, image]
        self.bias: list[list[float]] = []
        self.up_check_deg: float | None = None
        self.timings: dict[str, float] = {}
        self._added: list[Keyframe] = []
        self._kf_lock = threading.Lock()

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
        m.origin_c = None if d["origin_c"] is None else np.asarray(d["origin_c"], np.float64)
        m.bias = [list(e) for e in d["bias"]]
        m.cloud = Cloud.load(root / CLOUD_NPZ)
        return m

    def save(self) -> None:
        (self.root / KF_DIR).mkdir(parents=True, exist_ok=True)
        for kf in self._added:
            assert kf.feats is not None and kf.kdepth is not None
            extra = {} if kf.descriptor is None else {"descriptor": kf.descriptor}
            np.savez(self.root / KF_DIR / f"{kf.id}.npz", kps=kf.feats.kps, desc=kf.feats.desc,
                     kdepth=kf.kdepth, **extra)
            assert kf.grid is not None and kf.rgb is not None
            np.save(self.root / KF_DIR / f"{kf.id}.depth.npy", kf.grid.astype(np.float16))
            cv2.imwrite(str(self.root / KF_DIR / f"{kf.id}.jpg"),
                        cv2.cvtColor(kf.rgb, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, KF_JPEG_QUALITY])
        self._added = []
        self.cloud.save(self.root / CLOUD_NPZ)
        atomic_write_json(self.root / STATE_JSON, {
            "engine": ENGINE, "keyframes": [k.to_json() for k in self.keyframes],
            "scale": self.scale.to_json(), "drift": self.drift.to_json(),
            "next_id": self.next_id, "created_with": self.created_with,
            "origin_c": None if self.origin_c is None else self.origin_c.tolist(),
            "bias": self.bias})

    def image_path(self, kid: str) -> Path | None:
        """A keyframe's canonical image, or None."""
        p = self.root / KF_DIR / f"{kid}.jpg"
        return p if p.exists() else None

    def depth_of(self, kid: str) -> NDArray[np.float32] | None:
        """A keyframe's depth grid in metres (0: none), at the map's depth scale now."""
        p = self.root / KF_DIR / f"{kid}.depth.npy"
        kf = next((k for k in self.keyframes if k.id == kid), None)
        if kf is None or not p.exists():
            return None
        return (np.load(p).astype(np.float32) * self._depth_factor(kf)).astype(np.float32)

    def _load_kf(self, kf: Keyframe) -> None:
        """A stored keyframe's features, keypoint depths and descriptor, once. A dry run places
        its frames on several threads: they load under a lock and the features are set last
        (a thread that sees them sees the depths too). A keyframe stored without them keeps
        none (``_correspondences`` skips it)."""
        if kf.feats is not None:
            return
        with self._kf_lock:
            if kf.feats is not None:
                return
            p = self.root / KF_DIR / f"{kf.id}.npz"
            if not p.exists():
                return
            with np.load(p) as z:
                if not {"kps", "desc", "kdepth"} <= set(z.files):
                    return
                kf.descriptor = z["descriptor"] if "descriptor" in z.files else None
                kf.kdepth = z["kdepth"]
                kf.feats = Features(z["kps"], z["desc"])

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
        if span >= PRIOR_SPAN_M and self.prior_bias() == 1.0:
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
            s_b = self.prior_bias()
            if s_b != 1.0 and self.origin_c is not None:
                P = P.copy()
                P[:3, 3] = self.origin_c + s_b * (P[:3, 3] - self.origin_c)
            return P, sp, sr
        return self.drift.predict(P, sp, sr)

    def prior_bias(self) -> float:
        """The metric priors' own scale bias, measured against image baselines whose scale the
        priors did not set (depth images, camera heights): the least-squares ratio of image to
        prior baselines once the priors span ``BIAS_SPAN_M`` with ``BIAS_EVIDENCE_MIN``
        baselines, applied only when it differs from 1 by more than twice its uncertainty and
        the image scale's (else 1: a metric prior is taken as metric)."""
        ev = [e for e in self.bias if e[0] >= SCALE_EVIDENCE_MIN_M]
        if len(ev) < BIAS_EVIDENCE_MIN or self.prior_span() < BIAS_SPAN_M:
            return 1.0
        b_p = np.array([e[0] for e in ev])
        b_i = np.array([e[1] for e in ev])
        s_b = float((b_p * b_i).sum() / (b_p ** 2).sum())
        r = np.log(np.maximum(b_i, 1e-9) / (s_b * b_p))
        sigma = 1.4826 * float(np.median(np.abs(r - np.median(r)))) / math.sqrt(len(ev))
        vis = SENSOR_SCALE_SIGMA_REL if any(k.depth_kind == "sensor" for k in self.keyframes) \
            else max(self.scale.sigma_rel, MIN_SCALE_SIGMA_REL)
        return s_b if abs(s_b - 1.0) > 2.0 * math.hypot(sigma, vis) else 1.0

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

    def _correspondences(self, feats: Features, cands: list[Keyframe],
                         pred: tuple[Array, float, float] | None = None,
                         K: Intrinsics | None = None) -> _Corr:
        """Matches of the new frame's keypoints to the candidates', each lifted by its
        keyframe's depth: ratio-test matches, and with a predicted pose (``pred``) the matches
        guided by it (``guided_match``), within its sigmas' reach in pixels."""
        uv, X, d, which = [], [], [], []
        seen = np.zeros(len(feats.kps), bool)
        for j, k in enumerate(cands):
            self._load_kf(k)
            if k.feats is None or k.kdepth is None or len(k.kdepth) != len(k.feats.kps) \
                    or not len(k.feats.kps):
                continue  # a keyframe without features or keypoint depths: no reference
            ray = rays(k.K, k.feats.kps)
            z = k.kdepth.astype(np.float64) * self._depth_factor(k)
            R, c = k.T[:3, :3], k.centre
            Xk = c + (ray * z[:, None]) @ R.T
            dk = unit(ray @ R.T)
            ia, ib = match(feats.desc, k.feats.desc)
            if pred is not None and K is not None and len(ia) < GUIDED_ENOUGH:
                T0, sp0, sr0 = pred
                Kn = K
                kf_has = np.isfinite(Xk).all(axis=1)
                P = np.where(kf_has[:, None],
                             _project(T0[:3, :3], T0[:3, 3], np.where(kf_has[:, None], Xk, 0.0),
                                      Kn, False),
                             _project(T0[:3, :3], T0[:3, 3], dk, Kn, True))
                P[np.abs(P) >= BEHIND_PX / 2] = np.nan
                zc = np.where(kf_has, ((np.where(kf_has[:, None], Xk, 0.0) - T0[:3, 3])
                                       @ T0[:3, :3])[:, 2], np.inf)
                fpx = Kn.fx
                base = float(np.linalg.norm(c - T0[:3, 3]))
                rad = (K_SIGMA * fpx * (math.radians(sr0) + np.where(
                    np.isfinite(zc) & (zc > 0.05), sp0 / np.maximum(zc, 0.05),
                    base / (k.median_depth or 2.0))) + GUIDED_MIN_PX)
                ga, gb = guided_match(feats, k.feats, P, np.clip(rad, GUIDED_MIN_PX,
                                                                 GUIDED_MAX_PX))
                known = set(ia.tolist())
                extra = np.array([i not in known for i in ga.tolist()], bool)
                ia = np.concatenate([ia, ga[extra]]) if extra.size else ia
                ib = np.concatenate([ib, gb[extra]]) if extra.size else ib
            keep = ~seen[ia]
            ia, ib = ia[keep], ib[keep]
            if not len(ia):
                continue
            seen[ia] = True
            X.append(Xk[ib])
            d.append(dk[ib])
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

    def _scale_sigma(self, k: Keyframe) -> float:
        """The sigma of a reference keyframe's depth scale: a depth image's, else the monocular
        scale's while something measured it, else ``MONOCULAR_REGISTRATION_SIGMA_REL``."""
        if k.depth_kind == "sensor":
            return SENSOR_SCALE_SIGMA_REL
        if self.scale.source == "monocular":
            return MONOCULAR_REGISTRATION_SIGMA_REL
        return max(self.scale.sigma_rel, SENSOR_SCALE_SIGMA_REL)

    def _refs(self, cands: list[Keyframe]) -> Refs:
        return Refs(np.array([k.centre for k in cands]),
                    np.array([self._scale_sigma(k) for k in cands]),
                    np.array([k.median_depth or 2.0 for k in cands]))

    @staticmethod
    def _thr(K: Intrinsics) -> float:
        return REPROJ_PX * K.width / REF_WIDTH

    def _init(self, f: FrameIn, corr: _Corr, cands: list[Keyframe], T0: Array | None
              ) -> tuple[Array, Array, NDArray[np.bool_], bool] | None:
        """A first pose and its inliers: PnP (RANSAC) on the lifted points; failing that, a
        rotation-only fit (RANSAC) at the prior's centre when that is within
        ``ROTATION_FALLBACK_M`` or ``ROTATION_FALLBACK_REL`` of the median depth of a keyframe,
        or without a prior at the best-matched keyframe's centre (a camera turning in place).
        Returns (R, c, inliers, rotation_only)."""
        has = np.isfinite(corr.X).all(axis=1)
        if has.sum() >= MIN_INLIERS:
            got = pnp(corr.X[has], corr.uv[has], f.K, 2.0 * self._thr(f.K))
            if got is not None and got[2].sum() >= MIN_INLIERS:
                inl = np.zeros(len(corr.uv), bool)
                inl[np.nonzero(has)[0][got[2]]] = True
                return got[0], got[1], inl, False
        if T0 is not None:
            near = [float(np.linalg.norm(k.centre - T0[:3, 3])) for k in cands]
            j = int(np.argmin(near))
            if near[j] > max(ROTATION_FALLBACK_M,
                             ROTATION_FALLBACK_REL * (cands[j].median_depth or 2.0)):
                return None
            centre = T0[:3, 3]
        else:
            centre = cands[int(np.bincount(corr.kf).argmax())].centre
        dirs = corr.d.copy()
        dirs[has] = unit(corr.X[has] - centre)
        dist = np.array([float(np.linalg.norm(cands[j].centre - centre)) for j in corr.kf])
        med = np.array([cands[j].median_depth or 2.0 for j in corr.kf])
        par = np.where(has, 0.1, 1.0) * dist / med
        got_r = rotation_ransac(dirs, unit(rays(f.K, corr.uv)),
                                2.0 * self._thr(f.K) / f.K.fx + float(np.median(par)))
        if got_r is None or got_r[1].sum() < MIN_INLIERS:
            return None
        return got_r[0], centre.copy(), got_r[1], True

    def place(self, f: FrameIn, feats: Features, descriptor: Any,
              predicted: tuple[Array, float, float] | None = None, evidence: bool = True
              ) -> Placement:
        """Place one new frame (the map has its origin); does not add it. ``predicted``
        replaces the prior's prediction (a dry run's, along its request); without
        ``evidence`` it leaves the map's scale evidence as it is."""
        pred = self._predict(f) if predicted is None else predicted
        T0, sp0, sr0 = (None, 0.0, 0.0) if pred is None else pred
        cands = self._candidates(f, T0, descriptor)
        if pred is None:
            # without a prior, a camera-only robot is near its last confirmed frame: the
            # keyframes near that pose join the candidates (the origin is always confirmed). Its
            # pose does not guide the matching: a guess that loose matches anything.
            last = next(k for k in reversed(self.keyframes) if k.state == "fused")
            near = self._candidates(f, last.T, None)
            cands = cands + [k for k in near if all(k is not c for c in cands)]
            cands = cands[:MAX_CANDIDATES + RECENT_GUIDED]
        reason = "few_features" if min(f.features_count or len(feats.kps), len(feats.kps)) \
            < 150 else "no_overlap"
        corr = self._correspondences(feats, cands, pred, f.K) if cands and len(feats.kps) \
            else None
        init = None
        if corr is not None and len(corr.uv) >= MIN_INLIERS:
            init = self._init(f, corr, cands, T0)
        img = None
        if init is not None:
            assert corr is not None
            R, c, inl, rot_only = init
            refs = self._refs(cands)
            sub = (corr.uv[inl], corr.X[inl], corr.d[inl], corr.kf[inl])
            s, _ = refine_pose(R, c, *sub, refs, f.K, fix_centre=rot_only)
            n = int(s.inliers.sum())
            if n >= MIN_INLIERS:
                spi, sri = s.sigmas()
                spi = max(spi, MIN_SIGMA_POS_M)
                sri = max(sri, IMAGE_SIGMA_ROT_MIN_DEG)
                if rot_only:
                    spi = sp0 if T0 is not None else UNANCHORED_SIGMA_POS_M
                img = (s, n, n / len(corr.uv), "rotation" if rot_only else "pnp", spi, sri,
                       sub, refs)
        if T0 is None:
            # without a prior, the images alone must carry it: more inliers, a fair share
            if img is None or img[1] < NO_PRIOR_MIN_INLIERS or img[2] < NO_PRIOR_MIN_RATIO:
                return Placement(f.name, "rejected", reason)
            s, n, _, method, spi, sri, _, refs = img
            rp, rr = _real_sigmas(s, refs)
            if method == "rotation":
                rp = UNANCHORED_SIGMA_POS_M
            return Placement(f.name, "registered", None, s.T, round(rp, 4), round(rr, 3), n,
                             method, lifted=_lifted(corr))  # type: ignore[arg-type]
        assert f.prior is not None
        prior_T = np.asarray(f.prior["T_map_cam"], np.float64)
        ref = T0 if f.prior["trust"] == "drifting" else prior_T
        if img is None:
            return self._depth_place(f, T0, sp0, sr0, ref) or Placement(
                f.name, "provisional", reason, T0, sp0, sr0, 0, "prior", pose_residual(T0, ref))
        s, n, ratio, method, spi, sri, sub, refs = img
        rot_only = method == "rotation"
        if sri > CONFIRM_SIGMA_ROT_DEG or (not rot_only and spi > max(CONFIRM_SIGMA_POS_M,
                                                                       2.0 * sp0)):
            return self._depth_place(f, T0, sp0, sr0, ref) or Placement(
                f.name, "provisional", reason, T0, sp0, sr0, n, "prior", pose_residual(T0, ref))
        metric = f.prior["trust"] == "metric"
        if metric:
            # a metric prior is weighed against the image pose's real uncertainty: depth
            # images misregistered to the colour, monocular depth, a few references — far more
            # than the covariance of hundreds of correlated matches says
            spi = max(spi, METRIC_IMAGE_SIGMA_POS_M)
            sri = max(sri, METRIC_IMAGE_SIGMA_ROT_DEG)
        dp, dr = pose_residual(s.T, T0)
        if (not rot_only and dp > K_SIGMA * math.hypot(sp0, spi)) \
                or dr > K_SIGMA * math.hypot(sr0, sri):
            # a conflict needs strong, independent support: enough inliers, and for a metric
            # prior inliers on at least two references
            refs_ok = int((np.bincount(sub[3][s.inliers], minlength=len(cands))
                           >= MIN_INLIERS).sum())
            if n >= STRONG_INLIERS and ratio >= STRONG_RATIO and (not metric or refs_ok >= 2):
                return Placement(f.name, "rejected", "prior_conflict", None, None, None, n,
                                 method, (dp, dr), align_T=s.T, image_sigmas=(spi, sri))
            weak = Placement(f.name, "provisional", "no_overlap", T0, sp0, sr0, n, "prior",
                             pose_residual(T0, ref))
            weak.map_residual = (dp, dr)
            return weak
        # the reported sigmas: the image pose's covariance with the depth bias it does not see,
        # combined with the prior's own sigmas as the estimate combines them
        rp, rr = _real_sigmas(s, refs)
        gp, gr = _given_sigmas(f.prior)
        if metric:
            img_T = s.T
            if rot_only:
                img_T[:3, 3] = T0[:3, 3]
            T, _, _ = fuse_poses(img_T, spi, sri, T0, sp0, sr0)
            fsp = gp if rot_only else _combined(rp, spi, gp, sp0)
            fsr = _combined(rr, sri, gr, sr0)
        else:
            # the frame refined on its matches and its prior together (stored poses fixed)
            fused, _ = refine_pose(s.R, T0[:3, 3].copy() if rot_only else s.c, *sub, refs, f.K,
                                   prior=(T0, sp0, sr0))
            T = fused.T
            fsp = sp0 if rot_only else 1.0 / math.sqrt(1.0 / rp ** 2 + 1.0 / sp0 ** 2)
            fsr = 1.0 / math.sqrt(1.0 / rr ** 2 + 1.0 / sr0 ** 2)
        pl = Placement(f.name, "registered", None, T, round(max(fsp, REPORT_MIN_SIGMA_POS_M), 4),
                       round(max(fsr, REPORT_MIN_SIGMA_ROT_DEG), 3), n, method,
                       pose_residual(T, ref),
                       lifted=_lifted(corr), align_T=s.T,  # type: ignore[arg-type]
                       map_residual=pose_residual(T, s.T) if not rot_only
                       else (0.0, pose_residual(T, s.T)[1]))
        if method == "pnp" and evidence:
            self._scale_evidence(s, prior_T, f.prior["trust"], corr, cands,  # type: ignore[arg-type]
                                 (sub, refs, f.K, _prior_sigmas(f.prior)))
        return pl

    def _depth_place(self, f: FrameIn, T0: Array, sp0: float, sr0: float, ref: Array
                     ) -> Placement | None:
        """A frame with a depth image and a prior the images do not place (a bare wall, too
        few matches): its depth aligned on the depth of the keyframes near its prior
        (``depth_icp``: point-to-plane, the prior a factor, so a plane fixes what it can and
        the prior the rest), registered when its depth then agrees with theirs
        (``Agreement.confirmed``) and it stays within ``K_SIGMA`` of the prior. None
        otherwise."""
        if f.depth is None or not np.count_nonzero(f.depth > 0):
            return None
        refs = [r for k in self._candidates(f, T0, None) for r in [self._grid(k)]
                if r is not None]
        if not refs:
            return None
        pts = backproject(f.depth.astype(np.float32), f.K.K())
        got = depth_icp(pts, refs, T0, sp0, sr0)
        if got is None:
            return None
        T, sp, sr, n = got
        dp, dr = pose_residual(T, T0)
        # prior plus depth: the depth may refine the prior, never overrule it
        if dp > K_SIGMA * sp0 or dr > K_SIGMA * sr0 \
                or not agreement(pts, T, refs, SENSOR_DEPTH_TOL).confirmed:
            return None
        return Placement(f.name, "registered", None, T, round(max(sp, REPORT_MIN_SIGMA_POS_M), 4),
                         round(max(sr, REPORT_MIN_SIGMA_ROT_DEG), 3), n, "depth",
                         pose_residual(T, ref))

    def _scale_evidence(self, s: Solve, prior_T: Array, trust: str, corr: _Corr,
                        cands: list[Keyframe], fit: tuple[Any, Refs, Intrinsics,
                                                          tuple[float, float]]) -> None:
        """An image-only centre against its prior, both relative to the reference keyframe
        (the one with the most matches). For a monocular reference, in its raw depth units: a
        metric prior is compared with the reference's map centre, a drifting one with the
        reference's own drifting prior (odometry baselines), both evidence of ``k``. For a
        reference whose depth scale is measured independently of the priors (a depth image,
        or camera heights), in metres: evidence of the metric priors' own scale bias."""
        j = int(np.bincount(corr.kf).argmax())
        ref = cands[j]
        b_img = float(np.linalg.norm(s.c - ref.centre))
        # the prior baseline between the same two frames: to the reference's own metric prior
        ref_p = np.asarray(ref.prior["T"])[:3, 3] if ref.prior is not None \
            and ref.prior["trust"] == "metric" else ref.centre
        b_prior = float(np.linalg.norm(prior_T[:3, 3] - ref_p))
        anchored = ref.depth_kind == "sensor" or (ref.depth_kind == "mono"
                                                  and self.scale.source == "camera_height")
        if trust == "metric" and anchored:
            self.bias = [*self.bias, [b_prior, b_img]][-MAX_EVIDENCE:]
        if ref.depth_kind != "mono":
            return
        b_img /= self.scale.k
        if trust == "metric":
            # the depth scale the references need for the frame to sit at its metric prior: the
            # refinement with the prior and free reference scales (stored poses fixed)
            sub, refs, K, (sp, sr) = fit
            free = Refs(refs.c, np.full(len(refs.c), EVIDENCE_SCALE_SIGMA), refs.median)
            _, sc = refine_pose(s.R, s.c, *sub, free, K, prior=(prior_T, sp, sr))
            b_img = b_prior / float(sc[j]) / self.scale.k
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
        kf.grid, kf.K_grid, kf.rgb = depth, K_grid, f.rgb
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

    def add_height_evidence(self, f: FrameIn, T: Array, depth: NDArray[np.float32],
                         K_grid: Intrinsics) -> None:
        """A monocular keyframe with a known camera height: the height over its raw floor."""
        if f.height_m is None or f.height_m <= 0:
            return
        from oh_my_slam.reconstruction.pointcloud import pixel_points

        ok = edge_free(depth)
        if not ok.any():
            return
        pts, _ = pixel_points(depth / self.scale.k, K_grid, ok)
        R = T[:3, :3]
        if f.prior is None and f.mono is not None and f.mono()[3] is not None:
            # a camera-only frame: along its own gravity, not the chain of registrations
            R = first_camera(f.mono()[3])[:3, :3]
        planes = floor_candidates(pts @ R.T)
        if not planes:
            return
        sig = f.height_sigma_m if f.height_sigma_m else DEFAULT_HEIGHT_SIGMA_M
        self.scale.heights = [*self.scale.heights,
                              [sig / f.height_m, *(f.height_m / h for h in planes)]
                              ][-MAX_EVIDENCE:]
        log.debug("%s: camera height %.3f m over planes %s (raw)", f.name, f.height_m, planes)

    # -- an update -----------------------------------------------------------------------------

    def update(self, frames: list[FrameIn], progress: Callable[[str], None] = log.info
               ) -> list[Placement]:
        """Register and fuse ``frames`` (request order) into the map; the first frame with a
        prior defines the origin of a new map."""
        t0 = time.perf_counter()
        out: dict[int, Placement] = {}
        order = list(range(len(frames)))
        if not self.keyframes:
            # the first frame with a prior, else the first frame (camera-only robots)
            first = next((i for i, f in enumerate(frames) if f.prior is not None
                          and f.prior["trust"] in ("metric", "drifting")), 0)
            order = [first] + [i for i in order if i != first]
        reg_s = 0.0
        fuse_s = 0.0
        feats_of: dict[int, Features] = {}
        pending: int | None = None  # the last strong conflict of a drifting prior

        def keep(i: int, pl: Placement) -> None:
            f = frames[i]
            depth, Kg, kind, desc = self._keyframe_depth(f)
            if not self.keyframes and self.created_with != "none" and f.mono is not None:
                self._up_check(f, pl)
            if kind == "mono":
                assert pl.T is not None
                # its own scale brought onto the map's (the monocular model's metres vary from
                # frame to frame), on the points lifted at the map's current scale
                if pl.lifted is not None:
                    a = depth_alignment(pl.T if pl.align_T is None else pl.align_T,
                                        *pl.lifted, depth, Kg, f.K)
                    if a is not None:
                        depth = (depth * a).astype(np.float32)
                # its camera height, if known, refines the depth scale before it is fused
                self.add_height_evidence(f, pl.T, depth, Kg)
                k = self.scale.k
                self.scale.estimate(self.prior_span(), self.prior_span("drifting"))
                depth = (depth * (self.scale.k / k)).astype(np.float32)
            if pl.status == "provisional" and self.keyframes:
                self._map_consistency(f, pl, depth, Kg)
                if pl.status == "rejected":
                    return
            self.add(f, feats_of[i], pl, depth, Kg, kind, desc)
            progress(f"{f.name}: {pl.status} ({pl.method}, {pl.inliers} inliers)")

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
                    self._odometry(f, out[i])
                    reg_s += time.perf_counter() - t
                    continue
            feats_of[i] = features(f.rgb, f.ignore)
            if not self.keyframes:
                p = f.prior
                if p is not None and p["trust"] in ("metric", "drifting"):
                    P = np.asarray(p["T_map_cam"], np.float64)
                    sp, sr = _prior_sigmas(p)
                    self.created_with = p["trust"]
                    self.origin_c = P[:3, 3].copy()
                    res: tuple[float, float] | None = (0.0, 0.0)
                else:
                    P = first_camera(None if f.mono is None else f.mono()[3])
                    sp, sr = MIN_SIGMA_POS_M, MIN_SIGMA_ROT_DEG
                    self.created_with = "none"
                    res = None
                pl = Placement(f.name, "registered", None, P, sp, sr, 0, "origin", res)
            else:
                pl = self.place(f, feats_of[i], None if mono is None else mono[2])
            again = []
            if pl.reason == "prior_conflict" and f.prior is not None \
                    and f.prior["trust"] == "drifting":
                if pending is not None and self._same_drift(frames[pending], out[pending], f, pl):
                    # two strong placements agree on a correction the odometry's sigmas did
                    # not allow (a slip, a long unseen drive): the odometry is re-anchored on
                    # the first, and both are placed again
                    assert out[pending].align_T is not None
                    self.drift.confirm(out[pending].align_T, np.asarray(
                        frames[pending].prior["T_map_cam"], np.float64))  # type: ignore[index]
                    again = [pending, i]
                    pending = None
                else:
                    pending = i
            elif pl.status == "registered":
                pending = None
            if not again:
                out[i] = pl
                self._odometry(f, pl)
                reg_s += time.perf_counter() - t
                if pl.status == "rejected":
                    continue
                t = time.perf_counter()
                keep(i, pl)
                fuse_s += time.perf_counter() - t
                continue
            for j in again:
                t = time.perf_counter()
                out[j] = self.place(frames[j], feats_of[j], None)
                self._odometry(frames[j], out[j])
                reg_s += time.perf_counter() - t
                if out[j].status != "rejected":
                    t = time.perf_counter()
                    keep(j, out[j])
                    fuse_s += time.perf_counter() - t
        self.scale.estimate(self.prior_span(), self.prior_span("drifting"))
        self.timings = {"register": round(reg_s, 3), "fuse": round(fuse_s, 3),
                        "total": round(time.perf_counter() - t0, 3)}
        pls = [out[i] for i in range(len(frames))]
        inherit_conflicts(pls)
        return pls

    def _grid(self, k: Keyframe) -> DepthRef | None:
        """A keyframe's depth grid at the map's depth scale now, as a depth reference."""
        if k.grid is None and k.K_grid is not None:
            p = self.root / KF_DIR / f"{k.id}.depth.npy"
            if p.exists():
                k.grid = np.load(p).astype(np.float32)
        if k.grid is None or k.K_grid is None:
            return None
        return DepthRef(k.T, (k.grid * self._depth_factor(k)).astype(np.float32), k.K_grid.K())

    def _map_consistency(self, f: FrameIn, pl: Placement, depth: NDArray[np.float32],
                         K_grid: Intrinsics) -> None:
        """A provisional frame (at its prior: the images did not confirm it) against the depth
        the map's keyframes near it see: where they see enough of its points and too few agree
        (``Agreement.contradicted``), the median depth difference is its residual to the map
        (the quality report's ``previous_residuals``), and beyond ``K_SIGMA`` of its position
        sigma and the depth's own tolerance it contradicts the map: ``prior_conflict``, not
        fused. A monocular depth is first scaled by its median ratio to the map's there."""
        assert pl.T is not None and pl.sigma_pos_m is not None
        refs = [r for k in self._candidates(f, pl.T, None) for r in [self._grid(k)]
                if r is not None]
        sensor = f.depth is not None and bool(np.count_nonzero(f.depth > 0))
        tol = SENSOR_DEPTH_TOL if sensor else MONO_DEPTH_TOL
        pts = backproject(depth, K_grid.K())
        a = agreement(pts, pl.T, refs, tol)
        if not sensor and ALIGN_RANGE[0] <= a.median_ratio <= ALIGN_RANGE[1]:
            # monocular depth has its own scale per frame: judged once brought onto the map's
            pts = pts * a.median_ratio
            a = agreement(pts, pl.T, refs, tol)
        if not a.contradicted:
            return
        pl.map_residual = (a.median_abs_m, 0.0)
        med = float(np.median(pts[:, 2])) if len(pts) else 0.0
        if a.median_abs_m > K_SIGMA * pl.sigma_pos_m + tol * med:
            pl.status, pl.reason = "rejected", "prior_conflict"

    def _odometry(self, f: FrameIn, pl: Placement) -> None:
        """A frame with a drifting prior: the new reference of the odometry when the images
        confirmed it, else one more step of its path."""
        if f.prior is None or f.prior["trust"] != "drifting":
            return
        P = np.asarray(f.prior["T_map_cam"], np.float64)
        if pl.status == "registered" and pl.T is not None:
            self.drift.confirm(pl.T, P)
        else:
            self.drift.observe(P)

    def _same_drift(self, fa: FrameIn, a: Placement, fb: FrameIn, b: Placement) -> bool:
        """Whether two drifting frames' image poses (strong conflicts with their priors) imply
        the same correction of the odometry: ``b``'s image pose predicted from ``a``'s, as a
        reference, within ``K_SIGMA`` of their combined sigmas."""
        if a.align_T is None or b.align_T is None or a.image_sigmas is None \
                or b.image_sigmas is None:
            return False
        assert fa.prior is not None and fb.prior is not None
        trial = Drift(a.align_T, np.asarray(fa.prior["T_map_cam"], np.float64),
                      self.drift.odo_scale)
        T, sp, sr = trial.predict(np.asarray(fb.prior["T_map_cam"], np.float64),
                                  *_prior_sigmas(fb.prior))
        dp, dr = pose_residual(b.align_T, T)
        return dp <= K_SIGMA * math.hypot(sp, a.image_sigmas[0], b.image_sigmas[0]) \
            and dr <= K_SIGMA * math.hypot(sr, a.image_sigmas[1], b.image_sigmas[1])

    def _up_check(self, f: FrameIn, pl: Placement) -> None:
        """The angle between the origin prior's up axis and the monocular gravity estimate."""
        assert f.mono is not None and pl.T is not None
        up = f.mono()[3]
        if up is None:
            return
        u = pl.T[:3, :3] @ unit(np.asarray(up, np.float64))
        self.up_check_deg = round(math.degrees(math.acos(max(-1.0, min(1.0, float(u[2]))))), 3)

    def floor_and_bounds(self) -> tuple[float | None, NDArray[np.float64]]:
        """The floor height (map.md §6) and the points bounding the map (the cloud's and the
        cameras'). The floor is the surface the robot stands on: among the large horizontal
        planes of the cloud below every camera (``floor_candidates``, holding at least
        ``MAP_FLOOR_SHARE`` of the best one's points), the one nearest z 0 when priors fixed the
        map frame (their origin is where the robot stands), else the lowest; without one, the
        lowest dense layer."""
        pts = self.cloud.points()
        centres = np.array([k.centre for k in self.keyframes]).reshape(-1, 3)
        below = float(centres[:, 2].min()) if len(centres) else 0.0
        floor = None
        if len(pts):
            planes = floor_candidates(pts - np.array([0.0, 0.0, below]), MAP_FLOOR_SHARE)
            if planes and self.created_with in ("metric", "drifting"):
                # priors fixed the frame: the plane the robot stands on, nearest their origin
                # height (z 0), not a lower floor seen past a worktop's edge (map.md §6)
                floor = below - min(planes, key=lambda h: abs(below - h))
            else:
                floor = below - max(planes) if planes else lowest_layer(pts[:, 2], below)
        return floor, np.vstack([pts, centres]) if len(pts) else centres
