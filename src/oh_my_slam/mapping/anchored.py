"""Prior-first registration and incremental fusion (the *anchored* map; README "Prior-first
registration").

A map whose first update carries camera pose priors (``metric`` or ``drifting``) is kept here,
in the priors' frame, and never re-posed:

* **Map frame.** The origin frame is placed at its prior: the priors' frame is the map frame.
* **Match graph** (map.md §2, ``AnchoredMap._register_job``). A job's registration does not
  depend on its frames' request order: wherever an order matters the frames go in canonical
  order (``natural_key`` of their names, then camera). Rounds place every frame not yet
  registered against the confirmed keyframes as they stand; one that registers (an edge: at
  least ``MIN_INLIERS`` image inliers and its fitted pose passing the prior checks) is a
  reference at once; a frame that fails is placed again in the next round. A component the
  confirmed map does not reach is seeded at the prior of one of its frames without an image
  pose, kept only when another frame registers through it. A frame that ends the job kept is
  never a reference; keyframes are fused into the cloud once the statuses are final.
* **Drift estimate** (map.md §1, ``_fit_drift``). Within a job every drifting prior is predicted
  from the estimate in force when the job starts; after it the estimate is fitted robustly over
  the job's drifting-prior frames with an image pose, a large change needing three frames and
  half the job, and a frame it leaves out is rejected ``prior_conflict``; a changed estimate
  registers the job again under it.
* **Place first.** Every frame is first placed by its prior — a metric prior as given, a
  drifting one through the drift estimate (``Drift``) with sigmas that grow with the travel
  since the last frame the images confirmed — then refined against the confirmed keyframes
  nearest to that pose (by prior pose, else by retrieval descriptor):
  * SIFT matches against those keyframes, each match lifted to 3-D by the keyframe's depth;
  * PnP (RANSAC) on the lifted points and a robust Gauss-Newton refinement, the stored poses
    held fixed. PnP comes first even for a camera turning in place: the lifted points fix its
    rotation, and a wrong depth scale moves only its centre (along its distance to the
    keyframes). Without enough lifted points, a rotation-only fit at the prior's centre, when
    that is within ``ROTATION_FALLBACK_M`` of a keyframe (without a prior: at the best-matched
    keyframe's centre);
  * the image pose and the prior are combined by their sigmas (``fuse_poses``). Each frame so
    placed is a fixed reference for the frames placed after it in the same update.
* **Origin.** The map frame is the priors' frame. While the map has no image-confirmed
  keyframe, the first frame with a prior in canonical order that passes the checks and is not
  too close is placed at its prior and registered (``origin``); one before it that fails them is
  kept at its prior (it still lies in the map frame, api.md §4). Without any prior, the first frame is the origin
  (``first_camera``): with a camera height, the origin is the floor below it (map.md §1).
* **Kept at prior.** A frame with a prior that its images do not confirm (no image pose, an
  image pose too loose, an image pose beyond the conflict bound without strong support, an image
  that failed the checks: ``FrameIn.unconfirmed``, or a frame too close) is ``kept_at_prior``
  (api.md §4): at its prior as placed (metric as given; drifting through the odometry's
  correction) with the prior's own sigmas (drifting: grown by the drift), prior residual 0, and
  ``Placement.detail`` saying why (``no_image_pose``, ``loose_image_pose``,
  ``conflict_unsupported``; none for a failed check). A frame without an image pose is kept
  whatever its depth says: depth never places a frame. It is stored with its features, image, depth and reason, but it is never a reference for later
  frames nor scale evidence (map.md §2): only image-confirmed frames are (``Keyframe.usable``).
  Nor is it fused into the cloud, but for the too-close exception (api.md §4,
  ``Keyframe.contributes``): kept only for ``too_close``, with a metric prior and a depth image,
  its depth inside ``DEPTH_VALID_RANGE_M`` is fused at its prior pose.
* **Re-registration.** After each round, when the job has registered new frames that overlap a
  stored
  ``kept_at_prior`` keyframe (the overlap of map.md §1 with the kept frame as the reference:
  ``REREGISTER_OVERLAP`` of its image's 8 x 8 cells holding an epipolar inlier, ``coverage``),
  that keyframe is placed again against the confirmed map (``reregister``) by its images only,
  never its depth (a frame kept for a failed check too); confirmed, its pose leaves
  ``kept_at_prior`` and it is fused (``reregistered``: name and displacement); otherwise it stays
  kept with its reason. Confirmed, it is a reference for the job's later rounds.
* **Height gate** (map.md §2, ``height_consistent``). A frame whose committed camera height
  disagrees with the one sent beyond ``K_SIGMA`` × sqrt(height_sigma² + σ_z²) stays registered
  but gives no content (``Keyframe.height_ok``; ``height_inconsistent``).
* **Static image content** (map.md §1, ``static_content``). Features fixed in the image across
  three frames of the job turned pairwise by more than ``STATIC_ROT_DEG`` are left out of
  matching.
* **prior_conflict.** Only when a strongly supported image pose lies beyond the combined bound
  ``K_SIGMA`` × sqrt(σ_prior² + σ_image²) of the prior (position and rotation separately; σ_image
  its image-only calibrated sigma, ``mapping.sigmas``; σ_prior drift-grown for a drifting prior)
  and no image hypothesis of support >= ``SUPPORT_MIN`` lies within that bound (``_near_prior``);
  a supported hypothesis near the prior registers the frame instead. A frame whose image failed
  a check is kept at its prior before any of this (api.md §4: the check wins).
* **Calibrated sigmas.** Every image pose's sigma is calibrated (``mapping.sigmas.calibrated``:
  its covariance with the image-registration noise floor of its data kind, the references'
  depth kind and the prior's trust) before it is weighed or reported.
* **too_close.** A frame whose median scene depth is below ``TOO_CLOSE_M`` and that has fewer
  than ``TOO_CLOSE_INLIERS`` image feature inliers (never depth correspondences: depth never
  overrides it) is too close (api.md §4): rejected ``too_close`` without a
  prior, kept at its prior with a prior (reason: the check its image failed, else
  ``too_close``).
* **duplicate.** One pass after the job's registration (``end_job``): a registered frame within
  ``duplicate_limits`` of a confirmed keyframe of the same camera (stored, or another frame the
  job registered, named before it in canonical order) is rejected ``duplicate``
  (``Placement.duplicate_of``; api.md §4).
* **Priors against images.** Between two registered frames of the update that overlap by at
  least ``OVERLAP_MIN``, the disagreement of the images' relative rotation with the priors' is
  measured (``prior_disagreement_deg``); beyond ``K_SIGMA`` of the priors' combined rotation
  sigmas the frame registered later (canonical order, rounds) has its rotation corrected from
  the images (map.md §1).
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

import copy
import functools
import itertools
import json
import math
import os
import re
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.atomic import atomic_write_json
from oh_my_slam.core.constants import (
    FLOOR_BAND_SIGMAS,
    FLOOR_SIGMA_DEFAULT_M,
    REGISTRATION_MIN,
)
from oh_my_slam.core.geometry import project, unproject_pixels, voxel_keys
from oh_my_slam.core.images import on_grid
from oh_my_slam.core.log import get_logger
from oh_my_slam.core.types import Intrinsics
from oh_my_slam.mapping import sigmas
from oh_my_slam.mapping.anchor import prior_sigmas

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
EPIPOLAR_MIN_MATCHES = 30  # a pair with fewer epipolar inliers has no overlap (``epipolar``)
OVERLAP_GRID = 8  # map.md §1 "Overlap": cells per image side
MIN_INLIERS = REGISTRATION_MIN  # the registration minimum (api.md §6)
PIXEL_SIGMA = 1.0  # a match's pixel noise at 640 px width
DEPTH_NOISE_REL = 0.02  # a lifted point's own depth noise (the scale error is a parameter)
SENSOR_SCALE_SIGMA_REL = 0.03  # a depth image's scale, as a prior of its reference's scale
CONFIRM_SIGMA_POS_M = 0.10  # an image pose confirms a frame only this well determined (or
# within twice the prior's sigma)
CONFIRM_SIGMA_ROT_DEG = 2.0
METRIC_IMAGE_SIGMA_POS_M = 0.05  # an image pose weighed against a metric prior: at least this
METRIC_IMAGE_SIGMA_ROT_DEG = 1.0
K_SIGMA = 3.0
SUPPORT_MIN = 0.25  # a hypothesis's inliers over the best one's: supported from this share
TOO_CLOSE_INLIERS = 300  # a frame this well registered is never rejected for distance alone
OVERLAP_MIN = 0.5  # frames overlapping this much compare their relative rotation to the priors'
# ... not matched only when neither sees any of the other's matched points, at their committed
# poses or at their priors, in its image widened by this share of its size on every side (a
# cost limit with a margin far beyond any rotation error the check exists to catch)
OVERLAP_VIEW_PAD = 1.0
MIN_SIGMA_POS_M = 0.01
MIN_SIGMA_ROT_DEG = 0.1
IMAGE_SIGMA_POS_REL = 0.01  # of the median scene depth: depth bias the covariance does not see
IMAGE_SIGMA_ROT_MIN_DEG = 0.2
REPORT_MIN_SIGMA_POS_M = 0.001  # floors of the reported sigmas
REPORT_MIN_SIGMA_ROT_DEG = 0.01
UNANCHORED_SIGMA_POS_M = 0.10  # rotation-only without a prior: the centre is the keyframe's
# a drift estimate's sigma grows by this share of the distance from its supporting frames
DRIFT_POS_REL = 0.10
DRIFT_ROT_DEG_PER_M = 2.0  # ... and this much per metre of it in rotation
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
DEPTH_SIGMA_REL = 0.01  # depth images, before their residuals (README "Prior-first registration")
DEPTH_SENSOR_SIGMA_REL = 0.005  # a depth sensor's own scale error, which no residual shows
MONOCULAR_SIGMA_REL = 0.10
MIN_SCALE_SIGMA_REL = 0.002
DEFAULT_HEIGHT_SIGMA_M = 0.02
MIN_DEPTH_M = 0.05
SAMPLE_SIDE = 64  # a depth agreement samples a grid of at most this side
SENSOR_DEPTH_TOL = 0.10  # relative depth agreement of depth images
MIN_OVERLAP = 0.10  # share of a frame's points a reference view's depth sees
MIN_AGREEMENT = 0.75  # share of those that agree
MIN_AGREEING = 40  # points
# map.md §1 "Measurable pair", §2 step 2 (P20): a relative pose of two frames carrying depth
# images is confirmed when, both ways, at least this share of one frame's depth points that the
# other's depth sees agree within ``SENSOR_DEPTH_TOL`` (and ``MIN_AGREEING`` of them). True
# poses agree at 1.00 on the tiled and the textured test rooms, a square tile lattice's false
# geometries 90 deg off at 0.59-0.78 (README "Depth confirmation")
DEPTH_CONFIRM_AGREEMENT = 0.90
EPIPOLAR_HYPOTHESES = 3  # sequential RANSAC hypotheses of a pair (``epipolar_hypotheses``)
EDGE_REL = 0.06  # a depth sample whose 3x3 neighbourhood spans more than this is a depth edge
GN_ITERATIONS = 15
BEHIND_PX = 1e6  # the pixel of a point behind the camera (a residual Huber ignores)
HUBER_PX = 2.0
HUBER_SIGMAS = 2.0
MIN_PIXEL_SIGMA = 0.5
VOXEL_M = 0.02
FUSE_STRIDE = 2
FUSE_MAX_DEPTH_M = 6.0
# the depth image's valid range (api.md §4, README "Prior-first registration"): only depth in
# [MIN_DEPTH_M, FUSE_MAX_DEPTH_M] metres is fused, a frame kept too close with a metric prior's too
DEPTH_VALID_RANGE_M = (MIN_DEPTH_M, FUSE_MAX_DEPTH_M)
FLOOR_BINS = 50
FLOOR_PEAK_SHARE = 0.25
FLOOR_MIN_POINTS = 200
FLOOR_SPREAD_M = 0.12  # floor_spread: the floor's points lie this near its level
FLOOR_PEAK_MIN = 20.0  # points per 1 % height bin for a horizontal surface
FLOOR_BAND_REL = 0.05  # points this near a peak's height take part in its plane
FLOOR_PLANE_TOL_REL = 0.02
FLOOR_TILT_DEG = 10.0
FLOOR_RANSAC = 100
FLOOR_SAMPLE = 5000  # points a plane is fitted on
HEIGHT_VOTE_BIN = 0.03  # log scale: the frames' floor candidates vote in bins this wide
HEIGHT_VOTE_TOL = 0.15  # a frame's candidate this near the vote counts
DIRECTION_SECTORS = 4  # camera heights' scale compared between this many viewing directions
FLOOR_CANDIDATES = 8  # the most supported height peaks tried as planes
MAP_FLOOR_SHARE = 0.5  # the map's floor: the lowest plane with this share of the best one's
LEVEL_MIN_DEG = 0.1  # a camera-only map's floor plane levels it from this tilt
LEVEL_BAND_REL = 0.3  # ... fitted on the points this near its height (a tilted floor spreads)
TOO_CLOSE_M = 0.30  # median scene depth: the distance limit (as mapping.checks.TOO_CLOSE_M)
# a registered frame this near a confirmed keyframe of its camera is a duplicate (api.md §4):
# defaults, raised by the environment (README "Per-input status") up to, not reaching, the caps
DUPLICATE_POS_M = 0.02
DUPLICATE_ROT_DEG = 2.0
DUPLICATE_CAP_POS_M = 0.05
DUPLICATE_CAP_ROT_DEG = 5.0
DUPLICATE_POS_ENV = "OH_MY_SLAM_DUPLICATE_M"
DUPLICATE_ROT_ENV = "OH_MY_SLAM_DUPLICATE_DEG"
DUPLICATE_REMATCH = 0.90  # a pose duplicate's features re-matched in the keyframe (api.md §4)
FAILED_CHECKS = ("plain", "blur", "few_features", "too_close")  # api.md §4, in that order
REREGISTER_OVERLAP = 0.10  # a new frame sees this share of a kept keyframe's view: placed again
# the match graph (map.md §2): an edge needs the registration minimum of inliers (MIN_INLIERS)
DRIFT_PASSES = 6  # passes before the last: a job re-tested under its drift fit's decision
DRIFT_MIN_SUPPORT = 3  # map.md §1 "Drift estimate": a first estimate or a large change needs
# min(this, n) supporting frames, n the job's frames that can support it, ...
DRIFT_SUPPORT_SHARE = 0.5  # ... and this share of those frames
DRIFT_NEVER_ALONE = 2  # ... and never one frame alone
MEASURABLE_INLIERS = 100  # map.md §1 "Measurable pair": at least this many inliers ...
MEASURABLE_OVERLAP = 0.10  # ... covering at least this share of the 8 x 8 cells
STATIC_PX = 2.0  # map.md §1 "Static image content": a feature fixed within this (px) ...
STATIC_FRAMES = 3  # ... across this many frames ...
STATIC_ROT_DEG = 5.0  # ... turned pairwise by more than this is left out of matching
STATIC_DESC = 0.25  # ... its RootSIFT descriptors this near (unit-norm L2)
STATIC_CELL_KEY = 1 << 20  # the spatial hash of keypoints (``near_pairs``): x cell × this + y
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


def duplicate_limits() -> tuple[float, float]:
    """The pose duplicate's limits (api.md §4): ``DUPLICATE_POS_M`` and ``DUPLICATE_ROT_DEG``,
    or the values of ``DUPLICATE_POS_ENV`` and ``DUPLICATE_ROT_ENV``, which must be positive
    and below ``DUPLICATE_CAP_POS_M`` / ``DUPLICATE_CAP_ROT_DEG`` (a frame 0.05 m or 5° away is
    never a duplicate); ValueError otherwise."""
    out = []
    for env, default, cap in ((DUPLICATE_POS_ENV, DUPLICATE_POS_M, DUPLICATE_CAP_POS_M),
                              (DUPLICATE_ROT_ENV, DUPLICATE_ROT_DEG, DUPLICATE_CAP_ROT_DEG)):
        raw = os.environ.get(env, "").strip()
        v = float(raw) if raw else default
        if not 0.0 < v < cap:
            raise ValueError(f"{env}={raw!r}: expected a value in (0, {cap})")
        out.append(v)
    return out[0], out[1]


def first_camera(up_cam: Any, height_m: float | None = None) -> Array:
    """The pose of a camera-only map's first frame (map.md §1, without priors): +z up along the
    gravity it sees (``up_cam``, the up direction in its optical frame; level when unknown), +x
    its viewing direction on the horizontal plane; at the origin, or with a camera height
    ``height_m`` that far above it (the origin is then the floor below the camera)."""
    up = unit(np.asarray((0.0, -1.0, 0.0) if up_cam is None else up_cam, np.float64))
    fwd = np.array([0.0, 0.0, 1.0]) - up[2] * up  # the optical axis on the horizontal plane
    x = unit(fwd)
    M = np.stack([x, np.cross(up, x), up])  # rows: the map's axes in the optical frame
    return pose(M, np.array([0.0, 0.0, height_m or 0.0]))


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
    got = horizontal_plane(q, h0)
    return None if got is None else (got[1], got[2])


def horizontal_plane(q: Array, h0: float, band: float = FLOOR_BAND_REL
                     ) -> tuple[Array, float, int] | None:
    """``_horizontal_plane`` with the plane's unit normal (z up) first, among the points within
    ``band`` of ``h0`` (relative)."""
    near = q[np.abs(-q[:, 2] - h0) <= band * h0]
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
    return nrm, float(abs(nrm @ c0)), len(inl)


def floor_spread(z: NDArray[Any], floor: float) -> float:
    """``floor_sigma_m`` of a floor at ``floor`` among point heights ``z``: the spread
    (standard deviation) of the floor's own points about its level, not the uncertainty of their
    mean (map.md §6) (``reconstruction.gravity.floor_band``, the
    floor's half thickness over ``FLOOR_BAND_SIGMAS``), so that the floor band of map.md §5-§6
    holds the floor's noise and drift; ``FLOOR_SIGMA_DEFAULT_M`` with fewer than
    ``FLOOR_MIN_POINTS`` points within ``FLOOR_SPREAD_M`` of it."""
    from oh_my_slam.reconstruction.gravity import FLOOR_BAND_SIGMAS, floor_band

    z = np.asarray(z, np.float64)
    near = z[np.isfinite(z) & (np.abs(z - floor) <= FLOOR_SPREAD_M)]
    if near.size < FLOOR_MIN_POINTS:
        return FLOOR_SIGMA_DEFAULT_M
    return round(floor_band(near, floor, margin=FLOOR_SPREAD_M)[1] / FLOOR_BAND_SIGMAS, 4)


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


def duplicate_among(camera_id: str, T: Array, feats: Features | None,
                    cands: list[tuple[str, str, Array, Callable[[], Features | None]]]
                    ) -> str | None:
    """The pose duplicate's test (api.md §4), shared by every engine and the dry run: the first
    candidate ``(name, camera_id, pose, its features)`` of camera ``camera_id`` within
    ``duplicate_limits`` of pose ``T`` whose image the frame's (``feats``) agrees with
    (``images_agree``); without ``feats``, the pose alone. None when there is none."""
    lim_m, lim_deg = duplicate_limits()
    for name, cid, U, theirs in cands:
        if cid != camera_id:
            continue
        dp, dr = pose_residual(U, T)
        if dp > lim_m or dr > lim_deg:
            continue
        if feats is not None:
            other = theirs()
            if other is None or not images_agree(feats, other):
                continue
        return name
    return None


def images_agree(a: Features, b: Features) -> bool:
    """Whether image ``a`` agrees with image ``b`` as a pose duplicate's must (api.md §4): at
    least ``DUPLICATE_REMATCH`` of its features re-matched (ratio test) in ``b``; no detection
    is run. A view whose scene changed re-matches less, so the later observation is registered
    and wins (map.md §3; README "Pose duplicates")."""
    if not len(a.kps):
        return False
    ia, _ = match(b.desc, a.desc)
    return len(ia) >= DUPLICATE_REMATCH * len(a.kps)


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


@dataclass
class Epipolar:
    """The epipolar inliers of two frames' matches (``epipolar``): their pixels and unit rays in
    each image, the essential matrix and the fit's threshold (normalised image units)."""

    ua: Array
    ub: Array
    ra: Array
    rb: Array
    E: Array
    thr: float


def epipolar(a: Features, Ka: Intrinsics, b: Features, Kb: Intrinsics,
             min_matches: int = EPIPOLAR_MIN_MATCHES) -> Epipolar | None:
    """Two frames' SIFT matches (``match``) that agree with some relative pose (essential matrix,
    RANSAC): None below ``min_matches`` (``EPIPOLAR_MIN_MATCHES``; re-registration's
    eligibility asks the registration minimum, ``MIN_INLIERS``). The largest consensus: the
    first of ``epipolar_hypotheses``."""
    return next(epipolar_hypotheses(a, Ka, b, Kb, 1, min_matches), None)


def epipolar_hypotheses(a: Features, Ka: Intrinsics, b: Features, Kb: Intrinsics,
                        k: int = EPIPOLAR_HYPOTHESES, min_matches: int = EPIPOLAR_MIN_MATCHES,
                        then: int | None = None) -> Iterator[Epipolar]:
    """Up to ``k`` sequential epipolar hypotheses of two frames' matches, lazily: the largest
    RANSAC consensus, then the largest among the matches the earlier ones left, while at least
    ``min_matches`` (after the first: ``then``, when given) remain and agree. Repetitive or
    symmetric content (a square tile lattice) can make the largest one a false geometry (map.md
    §1 "Measurable pair")."""
    ib, ia = match(b.desc, a.desc)  # one per keypoint of a
    if len(ia) < min_matches:
        return
    ua, ub = a.kps[ia].astype(np.float64), b.kps[ib].astype(np.float64)
    ra, rb = rays(Ka, ua), rays(Kb, ub)
    thr = REPROJ_PX * Ka.width / REF_WIDTH / Ka.fx
    left = np.arange(len(ia))
    for _ in range(k):
        if len(left) < min_matches:
            return
        E, mask = cv2.findEssentialMat(ra[left, :2], rb[left, :2], np.eye(3), cv2.RANSAC, 0.999,
                                       thr)
        if E is None or mask is None or int(mask.sum()) < min_matches:
            return
        inl = mask.ravel().astype(bool)
        i = left[inl]
        yield Epipolar(ua[i], ub[i], ra[i], rb[i], np.asarray(E[:3], np.float64), thr)
        left = left[~inl]
        min_matches = min_matches if then is None else then


def cell_share(uv: Array, K: Intrinsics) -> float:
    """The 8 x 8 inlier-cell coverage of an image (map.md §1 "Overlap"): the share of its
    ``OVERLAP_GRID`` x ``OVERLAP_GRID`` cells holding at least one of the pixels ``uv``."""
    cx = np.clip((uv[:, 0] / K.width * OVERLAP_GRID).astype(int), 0, OVERLAP_GRID - 1)
    cy = np.clip((uv[:, 1] / K.height * OVERLAP_GRID).astype(int), 0, OVERLAP_GRID - 1)
    return len(set(zip(cx.tolist(), cy.tolist(), strict=True))) / float(OVERLAP_GRID ** 2)


def coverage(a: Features, Ka: Intrinsics, b: Features, Kb: Intrinsics) -> tuple[float, float]:
    """The one overlap measure (map.md §1): the inlier-cell coverage (``cell_share``) of image
    ``a`` and of image ``b`` by their epipolar inliers (``epipolar``); (0, 0) without a fit. A
    check that names a reference takes its image's; a pair without one, the smaller."""
    e = epipolar(a, Ka, b, Kb)
    if e is None:
        return 0.0, 0.0
    return cell_share(e.ua, Ka), cell_share(e.ub, Kb)


def edge_free(depth: NDArray[np.float32]) -> NDArray[np.bool_]:
    """Valid depth samples off depth edges (their 3x3 neighbourhood within ``EDGE_REL``)."""
    d = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0).astype(np.float32)
    k = np.ones((3, 3), np.uint8)
    hi = cv2.dilate(d, k)
    lo = cv2.erode(np.where(d > 0, d, np.float32(1e9)).astype(np.float32), k)
    out: NDArray[np.bool_] = (d > 0) & (lo > 0) & (hi <= lo * (1.0 + EDGE_REL))
    return out


def valid_depth(depth: NDArray[np.float32]) -> NDArray[np.bool_]:
    """The samples of a depth grid inside ``DEPTH_VALID_RANGE_M`` (finite)."""
    lo, hi = DEPTH_VALID_RANGE_M
    out: NDArray[np.bool_] = np.isfinite(depth) & (depth >= lo) & (depth <= hi)
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
        """Position (m) and rotation (deg) sigmas from the covariance, 3-D RMS (api.md §6
        "Sigma convention": σ² the trace of the block)."""
        sp = math.sqrt(max(float(np.trace(self.cov[3:, 3:])), 0.0))
        sr = math.degrees(math.sqrt(max(float(np.trace(self.cov[:3, :3])), 0.0)))
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


def seen_through(p: DepthRef, q: DepthRef, tol: float) -> tuple[int, int]:
    """View ``p``'s depth points (``backproject``) at its pose, seen by view ``q``: how many
    agree with ``q``'s depth within ``tol`` (relative) and how many lie in front of ``q``'s
    surface by more than that — ``q`` would see through them, a contradiction. A point behind
    ``q``'s surface is occluded there and tells nothing (the wide baselines of frames around an
    object)."""
    X = backproject(p.depth, p.K) @ p.T[:3, :3].T + p.T[:3, 3]
    Xq = (X - q.T[:3, 3]) @ q.T[:3, :3]
    z = Xq[:, 2]
    front = z > MIN_DEPTH_M
    uv, _ = project(np.where(front[:, None], Xq, 1.0), q.K)
    uv = np.floor(np.nan_to_num(uv, nan=-1.0, posinf=-1.0, neginf=-1.0))
    u, v = uv[:, 0].astype(np.int64), uv[:, 1].astype(np.int64)
    h, w = q.depth.shape
    inside = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    d = np.zeros(len(z))
    d[inside] = q.depth[v[inside], u[inside]]
    seen = inside & np.isfinite(d) & (d > MIN_DEPTH_M)
    agree = seen & (np.abs(z - d) <= tol * d)
    through = seen & (z < d * (1.0 - tol))
    return int(agree.sum()), int(through.sum())


def depth_confirmed(a: DepthRef, b: DepthRef, tol: float = SENSOR_DEPTH_TOL) -> bool:
    """The depth confirmation of two views' relative pose (map.md §1 "Measurable pair", §2 step
    2): each view's depth points at its pose, seen by the other (``seen_through``), agree within
    ``tol`` for at least ``DEPTH_CONFIRM_AGREEMENT`` of those that agree or that the other would
    see through (at least ``MIN_AGREEING`` agreeing), both ways. Every engine's candidate poses
    and the pairs' two-view fits take this one test."""
    for p, q in ((a, b), (b, a)):
        agree, through = seen_through(p, q, tol)
        if agree < MIN_AGREEING or agree < DEPTH_CONFIRM_AGREEMENT * (agree + through):
            return False
    return True


def depth_image(f: FrameIn) -> NDArray[np.float32] | None:
    """A frame's depth image (on its canonical pixels) with its ignored pixels holding none
    (map.md §2), or None without one."""
    if f.depth is None or not np.count_nonzero(f.depth > 0):
        return None
    m = on_grid(f.ignore, f.depth.shape)
    return f.depth.astype(np.float32) if m is None else np.where(m, 0.0, f.depth).astype(
        np.float32)


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
    state: str  # "fused" (confirmed by the images, or the origin) or "provisional" (kept at
    # its prior: the images did not confirm it)
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
    # its ignored pixels on the grid (map.md §2 "Ignored pixels"; kf/<id>.ignore.png), None: none
    ignore: NDArray[np.bool_] | None = None
    # kept at its prior: the reason (the check its image failed, too_close, no_overlap ...)
    reason: str | None = None
    # its committed camera height agrees with the one sent (map.md §2 height gate): False
    # keeps it a reference but out of occupancy, heights and objects
    height_ok: bool = True
    # the camera height and its sigma sent with it (the height gate, again on re-registration)
    height_m: float | None = None
    height_sigma_m: float | None = None
    # in a seeded component (map.md §2 match graph, step 2): the seed keyframe's id and its
    # prior's sigmas, the floor of every placement that depends on the component until it
    # links to a confirmed keyframe outside it (``AnchoredMap._seed_floor``)
    seed: str | None = None
    seed_sigma: tuple[float, float] | None = None

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "camera_id": self.camera_id,
                "K": self.K.to_dict(), "T": self.T.tolist(), "sigma_pos_m": self.sigma_pos_m,
                "sigma_rot_deg": self.sigma_rot_deg, "state": self.state, "prior": self.prior,
                "depth_kind": self.depth_kind, "k_used": self.k_used,
                "median_depth": self.median_depth, "method": self.method,
                "inliers": self.inliers, "reason": self.reason, "height_ok": self.height_ok,
                "height_m": self.height_m, "height_sigma_m": self.height_sigma_m,
                "seed": self.seed,
                "seed_sigma": None if self.seed_sigma is None else list(self.seed_sigma),
                "K_grid": None if self.K_grid is None else self.K_grid.to_dict()}

    @staticmethod
    def from_json(d: dict[str, Any]) -> Keyframe:
        return Keyframe(d["id"], d["name"], d["camera_id"], Intrinsics.from_dict(d["K"]),
                        np.asarray(d["T"], np.float64), float(d["sigma_pos_m"]),
                        float(d["sigma_rot_deg"]), d["state"], d["prior"], d["depth_kind"],
                        float(d["k_used"]), d["median_depth"], d["method"], int(d["inliers"]),
                        K_grid=None if d.get("K_grid") is None
                        else Intrinsics.from_dict(d["K_grid"]), reason=d.get("reason"),
                        height_ok=bool(d.get("height_ok", True)), height_m=d.get("height_m"),
                        height_sigma_m=d.get("height_sigma_m"), seed=d.get("seed"),
                        seed_sigma=None if d.get("seed_sigma") is None
                        else (float(d["seed_sigma"][0]), float(d["seed_sigma"][1])))

    @property
    def centre(self) -> Array:
        out: Array = self.T[:3, 3]
        return out

    def usable(self) -> bool:
        """A reference for later frames, and scale evidence: confirmed by the images. A
        keyframe kept at its prior is neither (map.md §2)."""
        return self.state == "fused"

    def contributes(self) -> bool:
        """Depth content for the cloud, occupancy and heights (never scale evidence nor a
        reference): confirmed by the images, or the too-close exception (api.md §4) with a depth
        image (its depth inside ``DEPTH_VALID_RANGE_M``, at the prior pose: depth points come
        only from a depth image); never a frame the height gate failed (``height_ok``, map.md
        §2)."""
        return self.detects() and (self.usable() or self.depth_kind == "sensor")

    def too_close_exception(self) -> bool:
        """Kept at its prior only because it is ``too_close``, with a ``metric`` prior (api.md
        §4 "Too-close exception")."""
        return self.state != "fused" and self.reason == "too_close" \
            and self.prior is not None and self.prior["trust"] == "metric"

    def detects(self) -> bool:
        """Content for objects and ``detection_votes``: confirmed by the images, or the too-close
        exception whatever its depth (without a depth image its detections are placed on its
        monocular depth at the prior pose, as any monocular detection: api.md §4); never a frame
        the height gate failed."""
        return self.height_ok and (self.usable() or (
            self.too_close_exception() and self.depth_kind != "none"))


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

    def rotated(self, R: Array, t: Array | None = None) -> Cloud:
        """The cloud turned by ``R`` about the origin, then moved by ``t``, its voxels binned
        again."""
        out = Cloud()
        if not len(self.keys):
            return out
        sx = self.xyz @ R.T
        if t is not None:
            sx = sx + self.count[:, None] * np.asarray(t, np.float64)
        uk, inv = np.unique(self._keys(sx / np.maximum(self.count, 1)[:, None]),
                            return_inverse=True)
        n = len(uk)
        out.keys, out.xyz = uk, np.zeros((n, 3))
        out.rgb, out.count = np.zeros((n, 3)), np.zeros(n, np.int64)
        np.add.at(out.xyz, inv, sx)
        np.add.at(out.rgb, inv, self.rgb)
        np.add.at(out.count, inv, self.count)
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
    with a drifting prior (its map pose and its prior), the odometry's scale, and the prior
    centres of the frames that supported the estimate (``support``): its sigma grows with the
    distance from them (map.md §1 "Drift-grown sigma", P16). Travel since the client's last fix
    is the client's part: drifting priors carry sigmas grown with it."""

    ref_T: Array | None = None
    ref_prior: Array | None = None
    odo_scale: float = 1.0
    evidence: list[list[float]] = field(default_factory=list)  # [odometry m, map m]
    # the fitted estimate's own sigma at its supporting frames (position m, rotation deg;
    # ``_drift_sigma``): the base its sigma grows from away from them (``predict``)
    fit_sigma: tuple[float, float] = (0.0, 0.0)
    # the prior centres (odometry frame) of the frames that supported it: the fit's inliers and
    # every drifting frame the images confirmed since (``confirm``)
    support: list[list[float]] = field(default_factory=list)
    # the supporting frames the running job confirmed (``confirm``): they join the support only
    # from the commit of the version that registered them, never within their own job (map.md
    # §1 "Supporting frames", P19), so ``predict`` leaves them out until the next job starts
    # (``AnchoredMap.begin_job``); never written (``to_json``: the committed support has them)
    fresh: list[list[float]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"ref_T": None if self.ref_T is None else self.ref_T.tolist(),
                "ref_prior": None if self.ref_prior is None else self.ref_prior.tolist(),
                "odo_scale": self.odo_scale, "evidence": self.evidence,
                "fit_sigma": list(self.fit_sigma), "support": self.support}

    @staticmethod
    def from_json(d: dict[str, Any]) -> Drift:
        return Drift(None if d["ref_T"] is None else np.asarray(d["ref_T"], np.float64),
                     None if d["ref_prior"] is None else np.asarray(d["ref_prior"], np.float64),
                     float(d["odo_scale"]), [list(e) for e in d["evidence"]],
                     tuple(d.get("fit_sigma", (0.0, 0.0))),  # type: ignore[arg-type]
                     [list(c) for c in d.get("support", [])])

    def predict(self, P: Array, sigma_pos: float, sigma_rot: float
                ) -> tuple[Array, float, float]:
        """The map pose and drift-grown sigmas of a drifting prior ``P`` (map.md §1, P16):
        through the latest correction; the sigmas ``drift_grown`` from its supporting frames
        (a state written before they were kept: its reference). Without an estimate (the
        identity), the prior as sent with its sigmas."""
        if self.ref_T is None or self.ref_prior is None:
            return P.copy(), sigma_pos, sigma_rot
        D = np.linalg.inv(self.ref_prior) @ P
        D[:3, 3] *= self.odo_scale
        support = [c for c in self.support if c not in self.fresh]
        sp, sr = drift_grown(sigma_pos, sigma_rot, P, self.fit_sigma,
                             support or [self.ref_prior[:3, 3].tolist()])
        return self.ref_T @ D, sp, sr

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
        c = P[:3, 3].tolist()
        if c not in self.support:  # no count bound (SI-19): each supporting frame once
            self.support = [*self.support, c]
            self.fresh = [*self.fresh, c]  # ... from the commit on (P19)


def drift_grown(sigma_pos: float, sigma_rot: float, P: Array, fit_sigma: tuple[float, float],
                support: list[list[float]]) -> tuple[float, float]:
    """A drifting prior ``P``'s drift-grown sigmas (map.md §1 "Drift-grown sigma", P16; every
    engine): the root sum of squares of the prior's (``sigma_pos``, ``sigma_rot``) and the drift
    estimate's own at that pose, which is its fit's sigma (``fit_sigma``) and its extrapolation
    away from the frames that supported it (``support``, their prior centres): ``DRIFT_POS_REL``
    × the distance from ``P``'s prior position to the nearest in position, ``DRIFT_ROT_DEG_PER_M``
    per metre of it in rotation."""
    c = P[:3, 3]
    d = min((float(np.linalg.norm(c - np.asarray(q, np.float64))) for q in support),
            default=0.0)
    return (math.hypot(sigma_pos, fit_sigma[0], DRIFT_POS_REL * d),
            math.hypot(sigma_rot, fit_sigma[1], DRIFT_ROT_DEG_PER_M * d))


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
    # the map-frame viewing yaw (deg) of each ``heights`` entry's camera (``direction_spread``)
    height_yaws: list[float] = field(default_factory=list)
    # per registered keyframe, the log of the factor bringing its own depth onto the map's
    # lifted points (``depth_alignment``): the residuals of its depth source, by depth kind
    depth_residuals: dict[str, list[float]] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"k": self.k, "source": self.source, "sigma_rel": self.sigma_rel,
                "priors": self.priors, "heights": self.heights, "odometry": self.odometry,
                "height_yaws": self.height_yaws, "depth_residuals": self.depth_residuals}

    @staticmethod
    def from_json(d: dict[str, Any]) -> ScaleState:
        return ScaleState(float(d["k"]), d["source"], float(d["sigma_rel"]),
                          [list(e) for e in d["priors"]], [list(e) for e in d["heights"]],
                          [list(e) for e in d["odometry"]],
                          [float(y) for y in d.get("height_yaws", [])],
                          {k: [float(x) for x in v]
                           for k, v in d.get("depth_residuals", {}).items()})

    def residual_spread(self, kind: str) -> float | None:
        """The robust spread (1.4826 MAD) of a depth kind's residuals (``depth_residuals``),
        None with fewer than ``SCALE_EVIDENCE_MIN``."""
        r = np.asarray(self.depth_residuals.get(kind, []), np.float64)
        if len(r) < SCALE_EVIDENCE_MIN:
            return None
        return 1.4826 * float(np.median(np.abs(r - np.median(r))))

    def direction_spread(self) -> float:
        """How the camera heights' scale differs between viewing directions (map.md §1: the
        scale holds in every horizontal direction): each frame's log scale (``_height_votes``)
        binned by its viewing yaw in ``DIRECTION_SECTORS`` sectors, the standard deviation of
        the sectors' medians; 0 with fewer than two sectors (or yaws not recorded)."""
        if len(self.height_yaws) != len(self.heights):
            return 0.0
        lr, idx = self._height_votes(True)
        sectors: dict[int, list[float]] = {}
        for v, i in zip(lr.tolist(), idx, strict=True):
            sec = int(((self.height_yaws[i] + 180.0 / DIRECTION_SECTORS) % 360.0)
                      // (360.0 / DIRECTION_SECTORS))
            sectors.setdefault(sec, []).append(v)
        if len(sectors) < 2:
            return 0.0
        return float(np.std([np.median(v) for v in sectors.values()]))

    def _height_votes(self, with_index: bool = False) -> Any:
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
        idx = [i for i, c in enumerate(cands) if np.min(np.abs(c - mode)) <= HEIGHT_VOTE_TOL]
        out = np.array([float(cands[i][np.argmin(np.abs(cands[i] - mode))]) for i in idx])
        return (out, idx) if with_index else out

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
            # one sigma that holds in every horizontal direction: the spread between viewing
            # directions is not averaged away (README "Prior-first registration")
            self.sigma_rel = max(math.hypot(math.sqrt(mad ** 2 + float(np.median(rel)) ** 2)
                                            / math.sqrt(len(lr)), self.direction_spread()),
                                 MIN_SCALE_SIGMA_REL)
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
        # the monocular model's metres: its stated error, with its keyframes' residual spread
        spread = self.residual_spread("mono") or 0.0
        self.k, self.source = 1.0, "monocular"
        self.sigma_rel = math.hypot(MONOCULAR_SIGMA_REL, spread)


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
    check_close: bool = False  # measure its scene distance on its monocular depth (too_close)
    # the check that its image failed (plain, blur, few_features): a frame with a prior is then
    # kept at it unless its depth image places it (api.md §4)
    unconfirmed: str | None = None
    # its features when measured already (``job_features``: static content left out over the
    # whole job); else measured by ``AnchoredMap._prepare``
    feats: Features | None = None


@dataclass
class Placement:
    name: str
    status: str  # registered, kept_at_prior, rejected
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
    # the frame from where the map says it is (the new frames' displacement the gate checks)
    map_residual: tuple[float, float] | None = None
    image_sigmas: tuple[float, float] | None = None  # of ``align_T`` (a conflict's image pose)
    scene_m: float | None = None  # its median scene depth, with reason too_close
    prior_ref: Array | None = None  # the prior as placed, which ``prior_residual`` is against
    duplicate_of: str | None = None  # with reason duplicate: the frame it duplicates
    # its image-only pose is unconstrained in translation: two-view parallax below
    # ``standpoint.PARALLAX_MAX_PX`` against every reference it registers to (map.md §1)
    unconstrained: bool = False
    # with reason no_overlap, why its images did not confirm it (api.md §4 ``detail``):
    # no_image_pose, loose_image_pose or conflict_unsupported; None otherwise
    detail: str | None = None
    has_prior: bool = False  # its displacement is against the prior (map.md §1)
    # the scale evidence of its image pose, recorded only once the job keeps it registered
    evidence: Callable[[], None] | None = None
    # its image-only pose's best 8 x 8 cell coverage of its image by the inliers on one
    # reference (``ref_share``): with ``inliers`` (over all its references), whether that pose
    # can support a drift estimate (``_can_support``); 0 without an image-only pose
    inlier_share: float = 0.0
    # the keyframes its pose depends on (its image pose's references, or the other engine's
    # anchors): a seeded component's sigma floor (``AnchoredMap._seed_floor``)
    refs: tuple[str, ...] = ()
    # its image-only pose meets the measurable-pair threshold with one reference
    # (``measurable_pose``): when it fails the prior checks, the images confirm the conflict
    # (rejected ``prior_conflict``), else it is kept ``conflict_unsupported`` (api.md §4, P16)
    measurable: bool = False


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
    sp, sr = prior_sigmas(p)
    return max(sp, REPORT_MIN_SIGMA_POS_M), max(sr, REPORT_MIN_SIGMA_ROT_DEG)


def _combined(real_a: float, weight_a: float, real_b: float, weight_b: float) -> float:
    """The sigma of the weighted mean of two independent estimates whose weights are
    ``1/weight²`` and whose real sigmas are ``real_a``, ``real_b``."""
    w = weight_b ** 2 / (weight_a ** 2 + weight_b ** 2)  # the share of estimate a
    return math.hypot(w * real_a, (1.0 - w) * real_b)


def _prior_sigmas(p: dict[str, Any]) -> tuple[float, float]:
    sp, sr = prior_sigmas(p)
    return max(sp, MIN_SIGMA_POS_M), max(sr, MIN_SIGMA_ROT_DEG)


def _median(depth: NDArray[np.float32] | None) -> float | None:
    """The median of a depth image's valid (positive, finite) values, or None."""
    if depth is None:
        return None
    v = depth[np.isfinite(depth) & (depth > 0)]
    return float(np.median(v)) if v.size else None


def _kept(f: FrameIn, T0: Array, sp0: float, sr0: float, reason: str, n: int,
          scene_m: float | None = None, detail: str | None = None) -> Placement:
    """A frame kept at its prior as placed (api.md §4): ``T0`` with the prior's sigmas (as
    given, a drifting prior's grown by the drift: ``AnchoredMap.kept_at``), residual 0."""
    return Placement(f.name, "kept_at_prior", reason, T0.copy(), round(sp0, 4), round(sr0, 3), n,
                     "prior", (0.0, 0.0), scene_m=scene_m, detail=detail, has_prior=True)


def _displacement(T: Array, image_T: Array, rot_only: bool) -> tuple[float, float]:
    """The committed pose against the image-only pose (map.md §1): a rotation-only image pose
    has no translation of its own (its translation counts as 0)."""
    dp, dr = pose_residual(T, image_T)
    return (0.0 if rot_only else dp), dr


def view_overlap(X: Array, T: Array, K: Intrinsics, pad: float = 0.0) -> float:
    """The share of the world points ``X`` (those one frame saw) in front of the camera ``T``
    and inside its image (widened by ``pad`` of its width and height on every side): how much
    of the first frame's view the second one shares."""
    if not len(X):
        return 0.0
    pc = (X - T[:3, 3]) @ T[:3, :3]
    front = pc[:, 2] > MIN_DEPTH_M
    if not front.any():
        return 0.0
    uv, _ = project(pc[front], K.K())
    pw, ph = pad * K.width, pad * K.height
    inside = (uv[:, 0] >= -pw) & (uv[:, 0] < K.width + pw) & (uv[:, 1] >= -ph) \
        & (uv[:, 1] < K.height + ph)
    return float(inside.sum()) / len(X)


def _may_overlap(Xa: Array, Ta: Array, Pa: Array, Ka: Intrinsics,
                 Xb: Array, Tb: Array, Pb: Array, Kb: Intrinsics) -> bool:
    """Whether two frames may share a view (``_against_priors``' cost limit): some point either
    matched (``Xa``, ``Xb``, at the committed poses ``Ta``, ``Tb``) is seen by the other, in its
    image widened by ``OVERLAP_VIEW_PAD``, at the committed poses or at the priors ``Pa``,
    ``Pb`` (the priors' rotations, the matched points moved with them)."""
    for A, B in ((Ta, Tb), (Pa, Pb)):
        xa = Xa if A is Ta else (Xa - Ta[:3, 3]) @ Ta[:3, :3] @ A[:3, :3].T + A[:3, 3]
        xb = Xb if B is Tb else (Xb - Tb[:3, 3]) @ Tb[:3, :3] @ B[:3, :3].T + B[:3, 3]
        if view_overlap(xa, B, Kb, OVERLAP_VIEW_PAD) > 0 or view_overlap(xb, A, Ka,
                                                                         OVERLAP_VIEW_PAD) > 0:
            return True
    return False


@dataclass
class _Corr:
    uv: Array  # new frame pixels
    X: Array  # world points (NaN rows: no depth)
    d: Array  # world directions from the reference keyframe's centre
    kf: NDArray[np.intp]  # index into the candidate list


@dataclass
class _Prep:
    """A job frame as registration needs it, measured once (``AnchoredMap._prepare``)."""

    feats: Features
    scene: float | None  # its median scene depth (too_close), or None
    close: bool
    descriptor: Any  # the retrieval descriptor (monocular inference), or None


def _order(frames: list[FrameIn], i: int) -> tuple[Any, ...]:
    """The canonical order of a job's frames (map.md §2: the outcome does not depend on the
    request order): by name in natural order (its digit runs as numbers: ``f2`` before
    ``f10``, so a capture's numbered frames go in capture order), then camera."""
    return natural_key(frames[i].name), frames[i].camera_id, i


def natural_key(name: str) -> tuple[tuple[int, int, str], ...]:
    """A name's natural order key: its digit runs as numbers (``f2`` before ``f10``)."""
    return tuple((0, int(t), "") if t.isdigit() else (1, 0, t)
                 for t in re.split(r"(\d+)", name) if t)


@dataclass
class _DriftObs:
    """A drifting-prior frame's own image pose, for the drift fit (``AnchoredMap._fit_drift``)."""

    index: int
    name: str
    T: Array  # its image-only pose (an origin's or a seed's: its committed one)
    P: Array  # its prior as sent (odometry frame)
    sigma_prior: tuple[float, float]
    sigma_image: tuple[float, float]
    rotation_only: bool
    pl: Placement


def _drift_inliers(d: Drift, obs: list[_DriftObs]) -> set[int]:
    """The observations a drift estimate predicts within the combined bound (api.md §4, the
    drift-grown prior sigma and the image-only sigma; a rotation-only pose, the rotation)."""
    out = set()
    for o in obs:
        T, sp, sr = d.predict(o.P, *o.sigma_prior)
        dp, dr = pose_residual(o.T, T)
        if not sigmas.conflicts(dp, dr, (sp, sr), o.sigma_image, position=not o.rotation_only):
            out.add(o.index)
    return out


def _drift_outliers(d: Drift, obs: list[_DriftObs], inliers: set[int]) -> dict[int, Placement]:
    """The observations a drift estimate leaves out (map.md §1 "Drift estimate"), its residual
    against its prior through that estimate: rejected ``prior_conflict`` when its image-only
    pose meets the measurable-pair threshold (``Placement.measurable``), else kept at its prior
    through that estimate, ``conflict_unsupported`` (api.md §4, P16). Each keeps its image-only
    pose and its overlap (``inlier_share``): it still counts, and can support, in the next fit
    (``_can_support``)."""
    out = {}
    for o in obs:
        if o.index in inliers:
            continue
        T, sp, sr = d.predict(o.P, *o.sigma_prior)
        if o.pl.measurable:
            out[o.index] = Placement(o.name, "rejected", "prior_conflict", None, None, None,
                                     o.pl.inliers, o.pl.method, pose_residual(o.T, T),
                                     align_T=o.T.copy(), image_sigmas=o.sigma_image,
                                     has_prior=True, prior_ref=T,
                                     inlier_share=o.pl.inlier_share, measurable=True)
        else:
            out[o.index] = Placement(o.name, "kept_at_prior", "no_overlap", T.copy(),
                                     round(sp, 4), round(sr, 3), o.pl.inliers, "prior",
                                     (0.0, 0.0), align_T=o.T.copy(),
                                     image_sigmas=o.sigma_image, has_prior=True,
                                     detail="conflict_unsupported",
                                     inlier_share=o.pl.inlier_share)
    return out


def _refined_drift(trial: Drift, obs: list[_DriftObs]) -> Drift:
    """A trial drift estimate refined over its inliers ``obs``: its reference pose as the mean of
    the reference poses each inlier implies (its image pose back through its odometry from the
    reference prior; rotations by their chordal mean)."""
    assert trial.ref_prior is not None
    Rs, cs = [], []
    for o in obs:
        D = np.linalg.inv(trial.ref_prior) @ o.P
        D[:3, 3] *= trial.odo_scale
        R = o.T[:3, :3] @ D[:3, :3].T
        Rs.append(R)
        cs.append(o.T[:3, 3] - R @ D[:3, 3])
    U, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    R = U @ np.diag([1.0, 1.0, float(np.linalg.det(U @ Vt))]) @ Vt
    return Drift(pose(R, np.mean(cs, axis=0)), trial.ref_prior.copy(), trial.odo_scale)


def _drift_sigma(obs: list[_DriftObs]) -> tuple[float, float]:
    """A fitted drift estimate's own sigma (position, rotation): the RMS of its inliers' combined
    prior and image sigmas over the root of their count."""
    sp = [math.hypot(o.sigma_prior[0], o.sigma_image[0]) for o in obs]
    sr = [math.hypot(o.sigma_prior[1], o.sigma_image[1]) for o in obs]
    n = math.sqrt(len(obs))
    return (math.sqrt(float(np.mean(np.square(sp)))) / n,
            math.sqrt(float(np.mean(np.square(sr)))) / n)


def _can_support(pl: Placement) -> bool:
    """Whether a frame can support a drift estimate (map.md §1, P13): an image-only pose (an
    origin or a seed has none: ``inlier_share`` 0) with at least ``MEASURABLE_INLIERS`` inliers
    against confirmed keyframes, counted across all its references, and an 8 x 8 overlap of at
    least ``MEASURABLE_OVERLAP`` with at least one of them (``ref_share``), even when it
    conflicts under the current estimate, whatever its final status."""
    return pl.inliers >= MEASURABLE_INLIERS and pl.inlier_share >= MEASURABLE_OVERLAP


def measurable_pose(uv: Array, ref_of: NDArray[np.intp], K: Intrinsics) -> bool:
    """Whether an image-only pose meets the measurable-pair threshold (api.md §4, map.md §2 match
    graph step 3, P16): on at least one confirmed or registered reference, at least
    ``MEASURABLE_INLIERS`` of its inliers ``uv`` (``ref_of``: each inlier's reference) covering
    at least ``MEASURABLE_OVERLAP`` of its 8 x 8 cells."""
    return any(int((ref_of == j).sum()) >= MEASURABLE_INLIERS
               and cell_share(uv[ref_of == j], K) >= MEASURABLE_OVERLAP
               for j in np.unique(ref_of))


def kf_detail(pl: Placement | None) -> str | None:
    """A kept placement's ``detail`` (``AnchoredMap.retry_kept``: a stored kept job frame's is
    ``no_image_pose``)."""
    return "no_image_pose" if pl is None else pl.detail


def _ref_ids(cands: list[Keyframe], ref_of: NDArray[np.intp]) -> tuple[str, ...]:
    """The ids of the candidates an image pose's inliers lie on (``ref_of``: each inlier's)."""
    return tuple(cands[int(j)].id for j in np.unique(ref_of))


def ref_share(uv: Array, ref_of: NDArray[np.intp], K: Intrinsics) -> float:
    """The best 8 x 8 cell coverage (``cell_share``) of an image by its inliers ``uv`` on one
    reference (``ref_of``: each inlier's reference): its overlap with that reference."""
    if not len(uv):
        return 0.0
    return max(cell_share(uv[ref_of == j], K) for j in np.unique(ref_of))


def _pair_rotation(fa: FrameIn, a: Features, fb: FrameIn, b: Features) -> float | None:
    """Two job frames' relative rotation (deg): their priors' when both have one, else the
    rotation-only fit of their matched rays; None without enough matches."""
    if fa.prior is not None and fb.prior is not None:
        Ra = np.asarray(fa.prior["T_map_cam"], np.float64)[:3, :3]
        Rb = np.asarray(fb.prior["T_map_cam"], np.float64)[:3, :3]
        return rot_deg(Ra.T @ Rb)
    ib, ia = match(b.desc, a.desc)
    if len(ia) < MIN_INLIERS:
        return None
    got = rotation_ransac(unit(rays(fb.K, b.kps[ib].astype(np.float64))),
                          unit(rays(fa.K, a.kps[ia].astype(np.float64))),
                          2.0 * REPROJ_PX * fa.K.width / REF_WIDTH / fa.K.fx)
    return None if got is None else rot_deg(got[0])


def near_pairs(a: NDArray[Any], b: NDArray[Any], r: float
               ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Every pair (index in ``a``, index in ``b``) of image positions within ``r`` px, by a
    spatial hash of ``b`` on an ``r`` px grid (the 3 x 3 cells around each point of ``a``): the
    cost grows with the points and their near pairs, not with the product of their counts."""
    if not len(a) or not len(b):
        return np.zeros(0, np.intp), np.zeros(0, np.intp)
    cb = np.floor(np.asarray(b, np.float64) / r).astype(np.int64)
    kb = cb[:, 0] * STATIC_CELL_KEY + cb[:, 1]
    order = np.argsort(kb, kind="stable")
    sk = kb[order]
    ca = np.floor(np.asarray(a, np.float64) / r).astype(np.int64)
    got_a, got_b = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            q = (ca[:, 0] + dx) * STATIC_CELL_KEY + ca[:, 1] + dy
            lo, hi = np.searchsorted(sk, q, "left"), np.searchsorted(sk, q, "right")
            cnt = hi - lo
            total = int(cnt.sum())
            if not total:
                continue
            ia = np.repeat(np.arange(len(a)), cnt)
            offs = np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            got_a.append(ia)
            got_b.append(order[np.repeat(lo, cnt) + offs])
    if not got_a:
        return np.zeros(0, np.intp), np.zeros(0, np.intp)
    ia, ib = np.concatenate(got_a), np.concatenate(got_b)
    d = np.linalg.norm(np.asarray(a, np.float64)[ia] - np.asarray(b, np.float64)[ib], axis=1)
    keep = d <= r
    return ia[keep].astype(np.intp), ib[keep].astype(np.intp)


def static_content(frames: list[FrameIn], prep: list[_Prep]) -> None:
    """Static image content out of matching (map.md §1 "Static image content"): a feature whose
    image position stays within ``STATIC_PX`` (and its descriptor within ``STATIC_DESC``) across
    at least ``STATIC_FRAMES`` frames of the job whose relative rotations exceed
    ``STATIC_ROT_DEG`` (pairwise) is removed from each of those frames' features — the robot's
    own shadow, a mark on the lens, a frame overlay. A job without such rotation keeps all its
    features."""
    n = len(frames)
    if n < STATIC_FRAMES:
        return
    rot: dict[tuple[int, int], float | None] = {}

    def turned(i: int, j: int) -> bool:
        key = (min(i, j), max(i, j))
        if key not in rot:
            rot[key] = _pair_rotation(frames[key[0]], prep[key[0]].feats, frames[key[1]],
                                      prep[key[1]].feats)
        r = rot[key]
        return r is not None and r > STATIC_ROT_DEG

    drop: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        a = prep[i].feats
        if not len(a.kps):
            continue
        # per keypoint of i: the frames holding a like keypoint within STATIC_PX
        hits: list[list[tuple[int, int]]] = [[] for _ in range(len(a.kps))]
        for j in range(n):
            b = prep[j].feats
            if j == i or not len(b.kps) or frames[j].K.width != frames[i].K.width:
                continue
            ka, kb = near_pairs(a.kps, b.kps, STATIC_PX)
            if not len(ka):
                continue
            dd = np.linalg.norm(a.desc[ka].astype(np.float32) - b.desc[kb].astype(np.float32),
                                axis=1) / ROOTSIFT_QUANT
            for p, q in zip(ka[dd <= STATIC_DESC].tolist(), kb[dd <= STATIC_DESC].tolist(),
                            strict=True):
                hits[p].append((j, q))
        for p, h in enumerate(hits):
            if len(h) + 1 < STATIC_FRAMES:
                continue
            # frames pairwise turned beyond STATIC_ROT_DEG (greedy, canonical order)
            group = [i]
            for j, _ in sorted(h, key=lambda t: _order(frames, t[0])):
                if all(turned(j, g) for g in group):
                    group.append(j)
            if len(group) >= STATIC_FRAMES:
                drop[i].add(p)
                for j, q in h:
                    if j in group:
                        drop[j].add(q)
    for i, d in enumerate(drop):
        if d:
            keep = np.array([k not in d for k in range(len(prep[i].feats.kps))], bool)
            f = prep[i].feats
            prep[i].feats = Features(f.kps[keep], f.desc[keep])
            log.info("%s: %d static features left out of matching", frames[i].name, len(d))


def job_features(frames: list[FrameIn]) -> list[Features]:
    """A whole job's features with its static image content left out (``static_content``, over
    every frame of the job, whichever engine registers it), each set on its ``FrameIn``."""
    prep = [_Prep(features(f.rgb, f.ignore), None, False, None) for f in frames]
    static_content(frames, prep)
    for f, p in zip(frames, prep, strict=True):
        f.feats = p.feats
    return [p.feats for p in prep]


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
        # the last update's largest disagreement of the images' relative rotation with the
        # priors' between overlapping frames (``_against_priors``)
        self.prior_disagreement_deg = 0.0
        # a camera-only map whose first frame had a camera height: that height (the origin is
        # the floor below that frame, map.md §1); None otherwise
        self.origin_height_m: float | None = None
        # the last update's stored kept_at_prior keyframes its images confirmed (``reregister``):
        # {"name", "keyframe", "displacement_m", "displacement_deg"}
        self.reregistered: list[dict[str, Any]] = []
        self.timings: dict[str, float] = {}
        self._added: list[Keyframe] = []
        # a frame without a prior, registered: its placement as one standpoint with a confirmed
        # keyframe when it shares one (map.md §1; set by the update job: ``mapping.standpoint``)
        self.standpoint: Callable[[FrameIn, Features, Placement], Placement] | None = None
        # the last update's frames the height gate kept out of content (map.md §2)
        self.height_inconsistent: list[str] = []
        # while an update runs: the keyframes whose fusion waits for its end (``_fuse_later``)
        self._defer: list[tuple[Keyframe, NDArray[np.uint8], NDArray[np.bool_] | None,
                                NDArray[np.float32], Intrinsics]] | None = None
        self._keep_s = 0.0  # the running update's seconds storing keyframes (its fusion)
        self._kf_lock = threading.Lock()
        # the keyframes stored before the running job (``begin_job``): the others are its own
        self._stored_ids: set[str] = set()
        # the running job's epipolar fits by pair of feature sets (``_epipolar``): a job's
        # features never change, so its rounds and drift passes fit each pair once; and its
        # placements (``_placed``) by frame, prediction and candidates
        self._epi: dict[tuple[int, int, int], tuple[Features, Features, Epipolar | None]] = {}
        self._placed: dict[tuple[Any, ...], Placement] = {}
        # the running job (map.md §2 match graph, step 6): the frames that gave scale evidence,
        # the new keyframes each re-registration rested on (by re-registered id) and its
        # keyframe's state before it (``_rereg_state``, to undo it), its frames with a prior
        # registered so far (``_mine``, ``_against_priors``) and the drift estimate it
        # registered under (``_job_used``); what a second registration takes (``plan_rerun``,
        # ``plan_drift_rerun``): frames forced to a placement by name, stored kept keyframes
        # not to re-register, the drift estimate to start from; whether it ran, and whether a
        # cross-engine retry gave a drifting frame an image pose (``retry_kept``)
        self._evidence_from: set[str] = set()
        self._rereg_by: dict[str, set[str]] = {}
        self._rereg_state: dict[str, tuple[Any, ...]] = {}
        self._mine: list[tuple[FrameIn, Placement, Features]] = []
        self._job_used: Drift | None = None
        self._exclude: dict[str, Placement] = {}
        self._no_rereg: set[str] = set()
        self._start_drift: Drift | None = None
        # the estimate a held job's first registration closed from (``close_job``)
        self._close_base: Drift | None = None
        self.reran: set[str] = set()
        self._retried_drifting = False

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
        m.origin_height_m = d.get("origin_height_m")
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
            if kf.ignore is not None:
                cv2.imwrite(str(self.root / KF_DIR / f"{kf.id}.ignore.png"),
                            kf.ignore.astype(np.uint8) * 255)
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
            "bias": self.bias, "origin_height_m": self.origin_height_m})

    def image_path(self, kid: str) -> Path | None:
        """A keyframe's canonical image, or None."""
        p = self.root / KF_DIR / f"{kid}.jpg"
        return p if p.exists() else None

    def ignore_of(self, kid: str) -> NDArray[np.bool_] | None:
        """A keyframe's ignored pixels on its depth grid (map.md §2), None when it has none."""
        kf = self.keyframe(kid)
        if kf is not None and kf.ignore is not None:
            return kf.ignore
        p = self.root / KF_DIR / f"{kid}.ignore.png"
        got = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.exists() else None
        return None if got is None else got > 0

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

    def confirmed(self) -> list[Keyframe]:
        """The image-confirmed keyframes: the only scale evidence and references (map.md §2)."""
        return [k for k in self.keyframes if k.usable()]

    def prior_span(self, trust: str = "metric") -> float:
        """The largest distance between two priors of that trust of the map's image-confirmed
        keyframes."""
        c = np.array([np.asarray(k.prior["T"])[:3, 3] for k in self.confirmed()
                      if k.prior is not None and k.prior["trust"] == trust])
        if len(c) < 2:
            return 0.0
        return float(np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2).max())

    def scale_summary(self) -> tuple[str, float]:
        """The map's ``scale_source`` and ``scale_sigma_rel``, from its image-confirmed
        keyframes only (map.md §1): their metric priors once they span ``PRIOR_SPAN_M``
        (positions are theirs), else depth images, else the depth scale's source (camera
        heights or the monocular model)."""
        span = self.prior_span()
        if span >= PRIOR_SPAN_M and self.prior_bias() == 1.0:
            # the metric priors' residuals against the committed positions of the confirmed
            # keyframes (their RMS), at least their stated sigma over the root of their count
            ms = [k for k in self.confirmed() if k.prior is not None
                  and k.prior["trust"] == "metric"]
            sp = float(np.median([_prior_sigmas(k.prior)[0] for k in ms]))
            res = np.array([float(np.linalg.norm(k.centre - np.asarray(k.prior["T"])[:3, 3]))  # type: ignore[index]
                            for k in ms])
            rms = math.sqrt(float(np.mean(res ** 2)))
            return "priors", max(math.sqrt(2.0) * max(rms, sp / math.sqrt(len(ms))) / span,
                                 MIN_SCALE_SIGMA_REL)
        if any(k.depth_kind == "sensor" for k in self.confirmed()):
            # the depth images' residuals against the map (``ScaleState.depth_residuals``),
            # with the sensor's own scale error; the documented constant before enough of them
            spread = self.scale.residual_spread("sensor")
            return "depth", DEPTH_SIGMA_REL if spread is None else max(
                math.hypot(spread, DEPTH_SENSOR_SIGMA_REL), MIN_SCALE_SIGMA_REL)
        return self.scale.source, self.scale.sigma_rel

    def begin_job(self) -> None:
        """A job starts: the keyframes stored now are the earlier versions' (``_job_duplicates``);
        the job's caches start empty."""
        self._stored_ids = {k.id for k in self.keyframes}
        self._epi, self._placed = {}, {}
        self.drift.fresh = []  # an earlier job's confirmations are committed: they support now

    def _epipolar(self, a: Features, Ka: Intrinsics, b: Features, Kb: Intrinsics,
                  min_matches: int = EPIPOLAR_MIN_MATCHES) -> Epipolar | None:
        """``epipolar``, each pair of feature sets fitted once per job (the sets are kept with
        the fit, so their ids stay theirs)."""
        key = (id(a), id(b), min_matches)
        got = self._epi.get(key)
        if got is None:
            got = (a, b, epipolar(a, Ka, b, Kb, min_matches))
            self._epi[key] = got
        return got[2]

    def _coverage(self, a: Features, Ka: Intrinsics, b: Features, Kb: Intrinsics
                  ) -> tuple[float, float]:
        """``coverage`` through the job's fits (``_epipolar``)."""
        e = self._epipolar(a, Ka, b, Kb)
        return (0.0, 0.0) if e is None else (cell_share(e.ua, Ka), cell_share(e.ub, Kb))

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
        vis = SENSOR_SCALE_SIGMA_REL if any(k.depth_kind == "sensor" for k in self.confirmed()) \
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

    @staticmethod
    def _on_sensor(cands: list[Keyframe], which: NDArray[np.intp]) -> bool:
        """Whether most of the correspondences ``which`` (indices into ``cands``) lie on
        keyframes with depth images: the calibration's data kind (``mapping.sigmas``)."""
        if not len(which):
            return False
        sensor = np.array([k.depth_kind == "sensor" for k in cands], bool)
        return bool(sensor[which].mean() >= 0.5)

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

    def kept_at(self, f: FrameIn, reason: str, n: int, scene_m: float | None = None,
                detail: str | None = None, pred: tuple[Array, float, float] | None = None
                ) -> Placement:
        """A frame with a prior kept at it (api.md §4 "Kept at prior"): at its prior as placed
        (``pred``, else its prediction: a metric prior as given, a drifting one through the
        drift estimate), with the prior's own sigmas (the trust defaults without; a drifting
        prior's grown by the drift estimate's uncertainty)."""
        assert f.prior is not None
        T0 = (pred or self._predict(f))[0]  # type: ignore[index]
        if reason in FAILED_CHECKS:  # kept for a failed check: no detail (api.md §4)
            detail = None
        gp, gr = _given_sigmas(f.prior)
        if f.prior["trust"] == "drifting":
            _, gp, gr = self.drift.predict(np.asarray(f.prior["T_map_cam"], np.float64), gp, gr)
        return _kept(f, T0, gp, gr, reason, n, scene_m, detail)

    def candidates_for(self, f: FrameIn, descriptor: Any,
                       pred: tuple[Array, float, float] | None) -> list[Keyframe]:
        """The confirmed keyframes a frame is matched against (``place``): nearest its prior
        as placed, else by retrieval descriptor; without a prior, also those near the last
        confirmed keyframe."""
        cands = self._candidates(f, None if pred is None else pred[0], descriptor)
        if pred is None:
            # without a prior, a camera-only robot is near its last confirmed frame: the
            # keyframes near that pose join the candidates. Its
            # pose does not guide the matching: a guess that loose matches anything.
            last = self.confirmed()[-1]  # placed only once the map has a confirmed keyframe
            near = self._candidates(f, last.T, None)
            cands = cands + [k for k in near if all(k is not c for c in cands)]
            cands = cands[:MAX_CANDIDATES + RECENT_GUIDED]
        return cands

    def place(self, f: FrameIn, feats: Features, descriptor: Any,
              predicted: tuple[Array, float, float] | None = None,
              cands: list[Keyframe] | None = None) -> Placement:
        """Place one new frame (the map has its origin) against the confirmed keyframes
        ``cands`` (``candidates_for`` without); does not add it. ``predicted`` replaces the
        prior's prediction (a stored kept keyframe placed again: its pose as kept). Its scale
        evidence is left in ``Placement.evidence``, recorded only when the job keeps it."""
        pred = self._predict(f) if predicted is None else predicted
        T0, sp0, sr0 = (None, 0.0, 0.0) if pred is None else pred
        if cands is None:
            cands = self.candidates_for(f, descriptor, pred)
        # an unplaced frame is no_overlap: in updates few_features is only the checks' verdict
        # (api.md §4, held in ``FrameIn.unconfirmed``)
        reason = "no_overlap"
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
            # a candidate pose the depth images do not confirm is no placement (map.md §2 step
            # 2): a lattice pose is never registered
            if n >= MIN_INLIERS and self.depth_confirms(f, s.T, sub[3][s.inliers], cands):
                spi, sri = s.sigmas()
                spi = max(spi, MIN_SIGMA_POS_M)
                sri = max(sri, IMAGE_SIGMA_ROT_MIN_DEG)
                if rot_only:
                    spi = sp0 if T0 is not None else UNANCHORED_SIGMA_POS_M
                img = (s, n, n / len(corr.uv), "rotation" if rot_only else "pnp", spi, sri,
                       sub, refs)
        trust = None if f.prior is None else f.prior["trust"]
        if T0 is None:
            # without a prior, the images alone must carry it: more inliers, a fair share
            if img is None or img[1] < NO_PRIOR_MIN_INLIERS or img[2] < NO_PRIOR_MIN_RATIO:
                return Placement(f.name, "rejected", reason)
            s, n, _, method, spi, sri, sub, refs = img
            rp, rr = sigmas.calibrated(*_real_sigmas(s, refs), self._on_sensor(cands, sub[3]),
                                       None)
            if method == "rotation":
                rp = max(rp, UNANCHORED_SIGMA_POS_M)
            # its committed pose is its image-only one (displacement 0) until a standpoint
            # turns it (``standpoint``); its displacement counts the rotation only when that
            # pose is unconstrained in translation (map.md §1)
            return Placement(f.name, "registered", None, s.T, round(rp, 4), round(rr, 3), n,
                             method, lifted=_lifted(corr),  # type: ignore[arg-type]
                             align_T=s.T, map_residual=(0.0, 0.0),
                             refs=_ref_ids(cands, sub[3][s.inliers]),
                             inlier_share=ref_share(sub[0][s.inliers], sub[3][s.inliers], f.K),
                             unconstrained=method != "rotation" and self._unconstrained(
                                 f, feats, s, sub[3], cands))
        assert f.prior is not None
        prior_T = np.asarray(f.prior["T_map_cam"], np.float64)
        ref = T0 if f.prior["trust"] == "drifting" else prior_T

        if f.unconfirmed is not None:
            # its image failed a check: the prior holds it, and depth agreement alone never
            # confirms it (api.md §4)
            return self.kept_at(f, f.unconfirmed, 0, pred=pred)
        if img is None:
            # no edge (map.md §2 match graph): the images give no pose
            return self.kept_at(f, reason, 0, detail="no_image_pose", pred=pred)
        s, n, _, method, spi, sri, sub, refs = img
        rot_only = method == "rotation"
        if sri > CONFIRM_SIGMA_ROT_DEG or (not rot_only and spi > max(CONFIRM_SIGMA_POS_M,
                                                                       2.0 * sp0)):
            return self.kept_at(f, reason, n, detail="loose_image_pose", pred=pred)
        dp, dr = pose_residual(s.T, T0)
        # the image-only calibrated sigmas (api.md §4: σ_hypothesis); a rotation-only pose takes
        # the standpoint's position sigma (its centre is the prior's: dp is 0)
        ci = sigmas.calibrated(*_real_sigmas(s, refs), self._on_sensor(cands, sub[3]), trust)
        if rot_only:
            ci = (spi, ci[1])
        share = ref_share(sub[0][s.inliers], sub[3][s.inliers], f.K)
        measurable = measurable_pose(sub[0][s.inliers], sub[3][s.inliers], f.K)
        if sigmas.conflicts(dp, dr, (sp0, sr0), ci, position=not rot_only):
            # beyond the combined bound (api.md §4, P16): the images confirm the conflict only
            # when its image-only pose meets the measurable-pair threshold, and no supported
            # hypothesis lies near the prior; else it is kept, its image-only pose (which may
            # still support a drift estimate: ``_can_support``) with it
            if not measurable:
                kept = self.kept_at(f, "no_overlap", n, detail="conflict_unsupported",
                                    pred=pred)
                kept.align_T, kept.image_sigmas, kept.inlier_share = s.T, ci, share
                return kept
            assert corr is not None
            hyp = None if rot_only else self._near_prior(f, corr, cands, T0, sp0, sr0, n)
            if hyp is None:
                return Placement(f.name, "rejected", "prior_conflict", None, None, None, n,
                                 method, (dp, dr), align_T=s.T, image_sigmas=ci, has_prior=True,
                                 prior_ref=ref, inlier_share=share, measurable=True)
            s, n, sub, refs = hyp
            spi, sri = s.sigmas()
            spi, sri = max(spi, MIN_SIGMA_POS_M), max(sri, IMAGE_SIGMA_ROT_MIN_DEG)
        assert corr is not None
        pl = self._register(f, s, n, method, spi, sri, sub, refs, corr, cands,
                            (T0, sp0, sr0), ref)
        # (a frame with a prior: its displacement is against the prior, never unconstrained)
        pl.inlier_share = ref_share(sub[0][s.inliers], sub[3][s.inliers], f.K)
        pl.measurable = measurable_pose(sub[0][s.inliers], sub[3][s.inliers], f.K)
        pl.refs = _ref_ids(cands, sub[3][s.inliers])
        return pl

    def _unconstrained(self, f: FrameIn, feats: Features, s: Solve, ref_of: NDArray[np.intp],
                       cands: list[Keyframe]) -> bool:
        """Whether an image-only pose ``s`` is unconstrained in translation (map.md §1): its
        two-view parallax (``standpoint.two_view``) below ``standpoint.PARALLAX_MAX_PX`` against
        every reference it registers to (``MIN_INLIERS`` of its inliers on it). A reference whose
        pair cannot be fitted constrains it."""
        from oh_my_slam.mapping import standpoint

        per = np.bincount(ref_of[s.inliers], minlength=len(cands))
        used = [cands[i] for i in np.flatnonzero(per >= MIN_INLIERS)]
        new = standpoint.View(f.name, f.K, feats, s.R, s.c)
        for k in used:
            self._load_kf(k)
            tv = None if k.feats is None else standpoint.two_view(
                new, standpoint.View(k.name, k.K, k.feats, k.T[:3, :3], k.centre, True))
            if tv is None or not tv.unconstrained:
                return False
        return bool(used)

    def _near_prior(self, f: FrameIn, corr: _Corr, cands: list[Keyframe], T0: Array,
                    sp0: float, sr0: float, best: int
                    ) -> tuple[Solve, int, tuple[Array, Array, Array, NDArray[np.intp]], Refs] | None:
        """An image hypothesis near the prior (api.md §4 "Choosing a status"): the lifted
        matches that reproject at the prior within ``K_SIGMA`` of its sigmas' reach (as the
        guided matching), refined on their own; it counts when its inliers are at least
        ``SUPPORT_MIN`` of the best hypothesis's ``best`` (and ``MIN_INLIERS``) and its pose
        lies within the combined conflict bound (``sigmas.conflicts``, its own calibrated
        sigmas). None otherwise."""
        has = np.isfinite(corr.X).all(axis=1)
        Xh = np.where(has[:, None], corr.X, 0.0)
        z = ((Xh - T0[:3, 3]) @ T0[:3, :3])[:, 2]
        P = _project(T0[:3, :3], T0[:3, 3], Xh, f.K, False)
        reach = K_SIGMA * f.K.fx * (math.radians(sr0) + sp0 / np.maximum(z, MIN_DEPTH_M)) \
            + 2.0 * self._thr(f.K)
        near = has & (z > MIN_DEPTH_M) & (np.linalg.norm(P - corr.uv, axis=1) <= reach)
        need = max(MIN_INLIERS, SUPPORT_MIN * best)
        if near.sum() < need:
            return None
        refs = self._refs(cands)
        sub = (corr.uv[near], corr.X[near], corr.d[near], corr.kf[near])
        s, _ = refine_pose(T0[:3, :3].copy(), T0[:3, 3].copy(), *sub, refs, f.K)
        n = int(s.inliers.sum())
        dp, dr = pose_residual(s.T, T0)
        trust = None if f.prior is None else f.prior["trust"]
        ci = sigmas.calibrated(*_real_sigmas(s, refs), self._on_sensor(cands, sub[3]), trust)
        if n < need or sigmas.conflicts(dp, dr, (sp0, sr0), ci) \
                or not self.depth_confirms(f, s.T, sub[3][s.inliers], cands):
            return None
        return s, n, sub, refs

    def depth_confirms(self, f: FrameIn, T: Array, ref_of: NDArray[np.intp],
                       cands: list[Keyframe]) -> bool:
        """Whether a candidate pose ``T`` of a frame passes the depth confirmation (map.md §1
        "Measurable pair", §2 step 2, ``depth_confirmed``) against the reference carrying a
        depth image that holds most of its inliers (``ref_of``, at least ``MIN_INLIERS``; ties
        to the lowest keyframe id). True when the frame or every such reference carries none."""
        own = depth_image(f)
        if own is None:
            return True
        per = np.bincount(ref_of, minlength=len(cands))
        best = None
        for i in np.flatnonzero(per >= MIN_INLIERS):
            k = cands[int(i)]
            ref = None if k.depth_kind != "sensor" else self._grid(k)
            if ref is not None and (best is None or (-per[i], k.id) < (-best[0], best[1])):
                best = (int(per[i]), k.id, ref)
        if best is None:
            return True
        ok = depth_confirmed(DepthRef(T, own, f.K.K()), best[2])
        if not ok:
            log.info("%s: its image pose against %s fails the depth confirmation", f.name,
                     best[1])
        return ok

    def _register(self, f: FrameIn, s: Solve, n: int, method: str, spi: float, sri: float,
                  sub: tuple[Array, Array, Array, NDArray[np.intp]], refs: Refs, corr: _Corr,
                  cands: list[Keyframe], pred: tuple[Array, float, float], ref: Array
                  ) -> Placement:
        """A frame with a prior that its image pose ``s`` confirms: the two combined."""
        assert f.prior is not None
        T0, sp0, sr0 = pred
        rot_only = method == "rotation"
        metric = f.prior["trust"] == "metric"
        # the image pose's calibrated sigmas (``mapping.sigmas``): its covariance with the depth
        # bias it does not see and the image-registration noise of its data kind
        rp, rr = sigmas.calibrated(*_real_sigmas(s, refs), self._on_sensor(cands, sub[3]),
                                   f.prior["trust"])
        if metric:
            # a metric prior is weighed against the image pose's real uncertainty: depth
            # images misregistered to the colour, monocular depth, a few references — far more
            # than the covariance of hundreds of correlated matches says
            spi = max(spi, METRIC_IMAGE_SIGMA_POS_M, rp)
            sri = max(sri, METRIC_IMAGE_SIGMA_ROT_DEG, rr)
        # the reported sigmas: the calibrated image sigmas combined with the prior's own as the
        # estimate combines them
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
                       map_residual=_displacement(T, s.T, rot_only), prior_ref=ref,
                       image_sigmas=(rp, rr), has_prior=True)
        if method == "pnp":
            ev = functools.partial(self._scale_evidence, s,
                                   np.asarray(f.prior["T_map_cam"], np.float64),
                                   f.prior["trust"], corr, cands,
                                   (sub, refs, f.K, _prior_sigmas(f.prior)), self.scale.k)
            pl.evidence = lambda: self._record_evidence(ev())
        return pl

    def _scale_evidence(self, s: Solve, prior_T: Array, trust: str, corr: _Corr,
                        cands: list[Keyframe], fit: tuple[Any, Refs, Intrinsics,
                                                          tuple[float, float]],
                        k: float) -> list[tuple[str, list[float]]]:
        """An image-only centre against its prior, both relative to the reference keyframe
        (the one with the most matches). For a monocular reference, in its raw depth units: a
        metric prior is compared with the reference's map centre, a drifting one with the
        reference's own drifting prior (odometry baselines), both evidence of ``k``. For a
        reference whose depth scale is measured independently of the priors (a depth image,
        or camera heights), in metres: evidence of the metric priors' own scale bias. ``k`` is
        the map's depth scale at placement; the evidence is recorded only when the job keeps
        the frame (``_record_evidence``): a list of (``bias`` | ``priors`` | ``odometry``,
        entry)."""
        out: list[tuple[str, list[float]]] = []
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
            out.append(("bias", [b_prior, b_img]))
        if ref.depth_kind != "mono":
            return out
        b_img /= k
        if trust == "metric":
            # the depth scale the references need for the frame to sit at its metric prior: the
            # refinement with the prior and free reference scales (stored poses fixed)
            sub, refs, K, (sp, sr) = fit
            free = Refs(refs.c, np.full(len(refs.c), EVIDENCE_SCALE_SIGMA), refs.median)
            _, sc = refine_pose(s.R, s.c, *sub, free, K, prior=(prior_T, sp, sr))
            b_img = b_prior / float(sc[j]) / k
            out.append(("priors", [b_prior, b_img]))
        elif ref.prior is not None and ref.prior["trust"] == "drifting":
            b_odo = float(np.linalg.norm(prior_T[:3, 3] - np.asarray(ref.prior["T"])[:3, 3]))
            out.append(("odometry", [b_odo, b_img]))
        return out

    def _record_evidence(self, entries: list[tuple[str, list[float]]]) -> None:
        """Scale evidence measured at placement (``_scale_evidence``), recorded."""
        for kind, e in entries:
            if kind == "bias":
                self.bias = [*self.bias, e][-MAX_EVIDENCE:]
            elif kind == "priors":
                self.scale.priors = [*self.scale.priors, e][-MAX_EVIDENCE:]
            else:
                self.scale.odometry = [*self.scale.odometry, e][-MAX_EVIDENCE:]

    def _evidence(self, f: FrameIn, pl: Placement) -> None:
        """A registered frame's scale evidence measured at placement (``Placement.evidence``),
        recorded; the frame noted as scale evidence (``_evidence_from``: a pose duplicate that
        gave some is undone by a second registration, map.md §2 match graph step 6)."""
        if pl.evidence is not None:
            pl.evidence()
            self._evidence_from.add(f.name)

    # -- adding keyframes ----------------------------------------------------------------------

    def _keyframe_depth(self, f: FrameIn) -> tuple[NDArray[np.float32], Intrinsics, str, Any]:
        """The depth grid (metres as fused), its intrinsics, its kind and the descriptor."""
        if f.depth is not None and np.count_nonzero(f.depth > 0) > 0:
            return f.depth.astype(np.float32), f.K, "sensor", None
        if f.mono is None:
            return np.zeros((f.K.height, f.K.width), np.float32), f.K, "none", None
        d, Kg, desc, _ = f.mono()
        return (d * self.scale.k).astype(np.float32), Kg, "mono", desc

    def height_consistent(self, f: FrameIn, pl: Placement) -> bool:
        """The height gate (map.md §2) of a placed frame (``height_gate``)."""
        return self.height_gate(f.height_m, f.height_sigma_m, pl.T, pl.sigma_pos_m)

    def height_gate(self, height_m: float | None, height_sigma_m: float | None,
                    T: Array | None, sigma_pos_m: float | None) -> bool:
        """The height gate (map.md §2): with ``height_above_floor_m`` and ``height_sigma_m``
        sent, whether the frame's committed camera height (its centre's z: the map's floor is
        z 0 when priors or a camera height fixed the frame, map.md §1) agrees with the one sent
        within ``K_SIGMA`` × sqrt(height_sigma² + σ_z²), σ_z = ``sigma_pos_m`` / √3 (its
        committed position sigma being 3-D RMS). True when nothing is sent or the map has no
        floor convention (a camera-only map without a camera height)."""
        if height_m is None or height_sigma_m is None or T is None or sigma_pos_m is None:
            return True
        if self.created_with not in ("metric", "drifting") and self.origin_height_m is None:
            return True
        sz = sigma_pos_m / math.sqrt(3.0)
        return abs(float(T[2, 3]) - height_m) <= K_SIGMA * math.hypot(height_sigma_m, sz)

    def add(self, f: FrameIn, feats: Features, pl: Placement, depth: NDArray[np.float32],
            K_grid: Intrinsics, kind: str, descriptor: Any) -> Keyframe:
        """Keep a placed frame as a keyframe and, when the images confirmed it, fuse it into the
        cloud (a frame kept at its prior is stored, not fused: map.md §2). A frame the height
        gate fails (``height_consistent``) stays a reference but gives no content, and is
        listed in ``height_inconsistent``. While an update runs, the fusion waits for its end
        (``update``: a frame registered there may still be dropped)."""
        assert pl.T is not None and pl.sigma_pos_m is not None and pl.sigma_rot_deg is not None
        seed = self._seed_floor(pl) if pl.status == "registered" else None
        kid = f"k{self.next_id:06d}"
        self.next_id += 1
        prior = None
        if f.prior is not None and f.prior["trust"] in ("metric", "drifting"):
            prior = {"trust": f.prior["trust"], "T": np.asarray(f.prior["T_map_cam"]).tolist(),
                     "sigma_pos_m": _prior_sigmas(f.prior)[0],
                     "sigma_rot_deg": _prior_sigmas(f.prior)[1]}
        ignore = on_grid(f.ignore, depth.shape)
        if ignore is not None:  # no depth point there: no cloud, occupancy, free space or floor
            depth = np.where(ignore, 0.0, depth).astype(np.float32)
        valid = depth[np.isfinite(depth) & (depth > 0)]
        kf = Keyframe(kid, f.name, f.camera_id, f.K, pl.T.copy(), pl.sigma_pos_m,
                      pl.sigma_rot_deg, "provisional" if pl.status == "kept_at_prior" else "fused",
                      prior, kind, self.scale.k if kind == "mono" else 1.0,
                      float(np.median(valid)) if valid.size else None, pl.method or "prior",
                      pl.inliers, reason=pl.reason if pl.status == "kept_at_prior" else None)
        kf.feats = feats
        kf.kdepth = sample_depth(depth, K_grid, f.K, feats.kps) if len(feats.kps) \
            else np.zeros(0, np.float32)
        kf.descriptor = None if descriptor is None else np.asarray(descriptor, np.float32)
        kf.grid, kf.K_grid, kf.rgb, kf.ignore = depth, K_grid, f.rgb, ignore
        kf.height_m, kf.height_sigma_m = f.height_m, f.height_sigma_m
        if seed is not None:
            kf.seed, kf.seed_sigma = seed
        kf.height_ok = self.height_consistent(f, pl)
        if not kf.height_ok:
            self.height_inconsistent.append(f.name)
        self.keyframes.append(kf)
        self._added.append(kf)
        pl.keyframe = kid
        if kind != "none":
            self._fuse_later(kf, f.rgb, f.ignore, depth, K_grid)
        return kf

    def _fuse_later(self, kf: Keyframe, image: NDArray[np.uint8],
                    ignore: NDArray[np.bool_] | None, depth: NDArray[np.float32],
                    K_grid: Intrinsics) -> None:
        """A keyframe's depth into the cloud when it carries content (``Keyframe.contributes``):
        now, or at the end of the running update (``_defer``)."""
        if self._defer is not None:
            self._defer.append((kf, image, ignore, depth, K_grid))
        elif kf.contributes():
            self._fuse(image, ignore, kf, depth, K_grid)

    def _fuse(self, image: NDArray[np.uint8], ignore: NDArray[np.bool_] | None, kf: Keyframe,
              depth: NDArray[np.float32], K_grid: Intrinsics) -> None:
        from oh_my_slam.reconstruction.pointcloud import pixel_points

        ok = edge_free(depth) & valid_depth(depth)
        lattice = np.zeros_like(ok)
        lattice[::FUSE_STRIDE, ::FUSE_STRIDE] = True
        mask = ok & lattice
        if ignore is not None and ignore.any():
            mask &= cv2.resize(ignore.astype(np.uint8), (depth.shape[1], depth.shape[0]),
                               interpolation=cv2.INTER_NEAREST) == 0
        if not mask.any():
            return
        pts, idx = pixel_points(depth, K_grid, mask)
        rgb = cv2.resize(image, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)
        col = rgb.reshape(-1, 3)[idx].astype(np.float64)
        self.cloud.add(pts @ kf.T[:3, :3].T + kf.centre, col)

    def depth_residual(self, f: FrameIn, T: Array, depth: NDArray[np.float32], K_grid: Intrinsics
                       ) -> None:
        """A depth image's scale against the map, recorded as its source's residual
        (``ScaleState.depth_residuals['sensor']``, map.md §2 ``scale_sigma_rel``) for a frame
        registered without lifted points (MapAnything): its depth at its committed pose ``T``
        against the confirmed keyframes' depth near it (``agreement``: the median ratio of their
        depth to its own where they see it), when at least ``ALIGN_MIN_POINTS`` are seen and the
        ratio lies in ``ALIGN_RANGE``."""
        refs = [r for k in self._candidates(f, T, None) for r in [self._grid(k)] if r is not None]
        a = agreement(backproject(depth, K_grid.K()), T, refs, SENSOR_DEPTH_TOL)
        if a.agree + a.conflict < ALIGN_MIN_POINTS \
                or not ALIGN_RANGE[0] <= a.median_ratio <= ALIGN_RANGE[1]:
            return
        self.scale.depth_residuals.setdefault("sensor", []).append(math.log(a.median_ratio))
        del self.scale.depth_residuals["sensor"][:-MAX_EVIDENCE]
        self._evidence_from.add(f.name)

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
        fwd = T[:3, 2]
        yaw = math.degrees(math.atan2(float(fwd[1]), float(fwd[0])))
        aligned = len(self.scale.height_yaws) == len(self.scale.heights)
        self.scale.heights = [*self.scale.heights,
                              [sig / f.height_m, *(f.height_m / h for h in planes)]
                              ][-MAX_EVIDENCE:]
        if aligned:
            self.scale.height_yaws = [*self.scale.height_yaws, yaw][-MAX_EVIDENCE:]
        self._evidence_from.add(f.name)
        log.debug("%s: camera height %.3f m over planes %s (raw)", f.name, f.height_m, planes)

    # -- an update -----------------------------------------------------------------------------

    def update(self, frames: list[FrameIn], progress: Callable[[str], None] = log.info,
               static: bool = True, hold: bool = False) -> list[Placement]:
        """Register and fuse ``frames`` into the map over the job's match graph (map.md §2,
        order-independent): ``_register_job`` under the drift estimate as it stands, then the
        robust drift fit (``_fit_drift``); while it changes the estimate or leaves out other
        frames, the job is registered again under its decision (its outliers rejected, every
        other frame re-tested) until a fixed point; after ``DRIFT_PASSES`` fits a last pass
        applies the last fit's outliers without fitting again, so no outlier stays registered. The
        drift estimate takes this job's confirmations (``_close_drift``); then the job ends
        (``end_job``: the pose-duplicate pass, the fusion of the keyframes once their statuses
        are final). When a pose duplicate served the job (``plan_rerun``), it registers once
        more without the duplicates (map.md §2 match graph, step 6). With ``hold`` the job
        stays open for another engine's frames (MapAnything, ``retry_kept``): its caller ends
        it with ``end_job`` (and plans a second registration itself). A second registration
        starts from ``_start_drift`` when one is planned, its ``_exclude`` frames forced to
        their placement."""
        t0 = time.perf_counter()
        new = not self.keyframes
        self.begin_job()
        prep = [self._prepare(f) for f in frames]
        if static:  # else the job's features left static content out already (``job_features``)
            static_content(frames, prep)
        snap = self._snapshot()
        self._keep_s = 0.0
        while True:
            out, used = self._register_drift(frames, prep, snap, progress)
            self._close_drift(frames, out, used)
            pls = [out[i] for i in range(len(frames))]
            if hold or not self.plan_rerun(pls):
                break
            self._restore(snap)
        if hold and self._close_base is None:
            self._close_base = copy.deepcopy(used)
        fuse_s = self._keep_s
        if not hold:
            dups, end_s = self.end_job(pls, new)
            pls = [dups.get(pl.name, pl) for pl in pls]
            fuse_s += end_s
        total = time.perf_counter() - t0
        self.timings = {"register": round(total - fuse_s, 3), "fuse": round(fuse_s, 3),
                        "total": round(total, 3)}
        return pls

    def _register_drift(self, frames: list[FrameIn], prep: list[_Prep], snap: dict[str, Any],
                        progress: Callable[[str], None]) -> tuple[dict[int, Placement], Drift]:
        """The job's registration (``_register_job``) under the drift estimate as it stands
        (``_start_drift`` when a second registration starts from one), then the robust drift
        fit (``_fit_drift``) until a fixed point (``update``). Returns the placements by frame
        index and the estimate they were registered under (``_job_used``)."""
        used = copy.deepcopy(self._start_drift or self.drift)
        excluded = {i: self._exclude[f.name] for i, f in enumerate(frames)
                    if f.name in self._exclude}
        forced: dict[int, Placement] = {}
        passes, final = 0, False
        while True:
            if passes:
                self._restore(snap)
            passes += 1
            self.drift = copy.deepcopy(used)
            self.prior_disagreement_deg, self.reregistered, self.height_inconsistent = 0.0, [], []
            self._defer = list(snap["defer"] or [])
            out = self._register_job(frames, prep, {**forced, **excluded}, progress)
            if final:  # the last pass applies the last fit's outliers
                break
            got, outliers = self._fit_drift(frames, out, used)
            outliers = {i: pl for i, pl in outliers.items() if i not in excluded}
            if got is None and outliers.keys() == forced.keys():
                break  # a fixed point: the estimate stays and leaves out the same frames
            # re-test every frame under the accepted estimate (map.md §1): the left-out ones are
            # the latest fit's, a frame left out before may come back
            used = got or used
            forced = outliers
            final = passes >= DRIFT_PASSES
        self._job_used = copy.deepcopy(used)
        return out, used

    def end_job(self, pls: list[Placement], new: bool) -> tuple[dict[str, Placement], float]:
        """The end of a job's registration (``update``; after the other engine's frames when it
        held the job open): the pose-duplicate pass (``_job_duplicates``), then the fusion of
        the keyframes left, in the order the job kept them, the depth scale, and a new
        camera-only map levelled by its floor (``_level_by_floor``, ``pls`` turned with it);
        the job's caches and its second-registration plan are dropped.
        Returns the duplicates' placements by frame name and the seconds of the fusion."""
        t = time.perf_counter()
        dups, gone = self._drop_duplicates(self._job_duplicates())
        pending, self._defer = self._defer or [], None
        for kf, image, ignore, depth, K_grid in pending:
            if kf.id not in gone and kf.contributes():
                self._fuse(image, ignore, kf, depth, K_grid)
        self.scale.estimate(self.prior_span(), self.prior_span("drifting"))
        if new and self.created_with == "none" and self.keyframes:
            self._level_by_floor([pl for pl in pls if pl.name not in dups])
        for pl in dups.values():
            pl.keyframe = None
        self._epi, self._placed = {}, {}  # the job's caches end with it (memory)
        self._exclude, self._no_rereg, self._start_drift = {}, set(), None
        self._close_base = None
        self._evidence_from, self._rereg_by, self._rereg_state = set(), {}, {}
        self._mine, self._job_used, self.reran, self._retried_drifting = [], None, set(), False
        return dups, time.perf_counter() - t

    def _rereg_ids(self) -> set[str]:
        """The stored kept keyframes this job re-registered."""
        return {e["keyframe"] for e in self.reregistered}

    def _job_duplicates(self) -> dict[str, Placement]:
        """The pose-duplicate pass (map.md §2 match graph, step 6; api.md §4): after the
        registration fixed point, every frame the job registered, and every stored kept
        keyframe it re-registered (a frame of this job for the pass, P16), in canonical name
        order, against the stored confirmed keyframes and the job's frames named before it that
        are not duplicates themselves (``duplicate_among``: the registered pose within
        ``duplicate_limits``, its image agreeing). Of a pair, the later-named frame is the
        duplicate: rejected ``duplicate`` naming the other (``keyframe``: its id). Nothing is
        changed (``_drop_duplicates``)."""
        def loaded(k: Keyframe) -> Features | None:
            self._load_kf(k)
            return k.feats

        def entry(k: Keyframe) -> tuple[str, str, Array, Callable[[], Features | None]]:
            return k.name, k.camera_id, k.T, functools.partial(loaded, k)

        again = self._rereg_ids()
        cands = [entry(k) for k in self.confirmed() if k.id in self._stored_ids
                 and k.id not in again]
        out: dict[str, Placement] = {}
        for k in sorted((k for k in self.confirmed() if k.id not in self._stored_ids
                         or k.id in again), key=lambda k: (natural_key(k.name), k.camera_id)):
            dup = duplicate_among(k.camera_id, k.T, loaded(k), cands)
            if dup is None:
                cands.append(entry(k))
                continue
            out[k.name] = Placement(k.name, "rejected", "duplicate", inliers=k.inliers,
                                    duplicate_of=dup, keyframe=k.id)
            log.info("%s: a pose duplicate of %s", k.name, dup)
        return out

    def _drop_duplicates(self, dups: dict[str, Placement]
                         ) -> tuple[dict[str, Placement], set[str]]:
        """The pass's duplicates applied: a job frame's keyframe dropped (no content, no
        reference); a re-registered stored keyframe's re-registration undone (it stays kept, as
        stored). Returns the job frames' placements by name and the keyframe ids whose fusion
        is dropped."""
        gone = {pl.keyframe for pl in dups.values() if pl.keyframe is not None}
        again = gone & self._rereg_ids()
        for kid in again:
            kf = self.keyframe(kid)
            assert kf is not None
            (kf.T, kf.sigma_pos_m, kf.sigma_rot_deg, kf.state, kf.method, kf.inliers, kf.rgb,
             kf.reason, kf.height_ok, kf.seed, kf.seed_sigma) = self._rereg_state[kid]
        self.reregistered = [e for e in self.reregistered if e["keyframe"] not in again]
        drop = gone - again
        self.keyframes = [k for k in self.keyframes if k.id not in drop]
        self._added = [k for k in self._added if k.id not in drop]
        self.height_inconsistent = [n for n in self.height_inconsistent if n not in dups]
        return {n: pl for n, pl in dups.items() if pl.keyframe not in again}, gone

    def _dup_effects(self, dups: dict[str, Placement], pls: list[Placement]) -> bool:
        """Whether a duplicate of the pass served the job (map.md §2 match graph, step 6): as a
        reference of another placement (``pls``: the job's, ``Placement.refs``; a seeded
        component's), as the drift reference or support (a job frame registered with a
        drifting prior: ``_close_drift`` confirmed it), as scale evidence (``_evidence_from``),
        or for a re-registration (``_rereg_by``)."""
        for name, d in dups.items():
            kid = d.keyframe
            kf = self.keyframe(kid or "")
            assert kid is not None and kf is not None
            if any(kid in p.refs for p in pls if p.name != name) \
                    or any(k.seed == kid and k.id != kid for k in self.keyframes) \
                    or any(kid in by for by in self._rereg_by.values()) \
                    or name in self._evidence_from:
                return True
            if kid not in self._stored_ids and kf.prior is not None \
                    and kf.prior["trust"] == "drifting":
                return True
        return False

    def job_start(self) -> dict[str, Any]:
        """A held job as it starts (``service.prior_first``): its fusion deferred to its end
        (``_fuse_later``), whichever engine adds a keyframe first, and the state a second
        registration starts again from (``job_restart``)."""
        self._defer = [] if self._defer is None else self._defer
        return self._snapshot()

    def job_restart(self, start: dict[str, Any]) -> None:
        """A held job back at ``start`` (``job_start``) for its second registration; the plan
        (``excluded``, ``reran``) stays."""
        self._restore(start)
        self._close_base = None

    def close_job(self, frames: list[FrameIn], pls: list[Placement]) -> None:
        """A held job's drift after both engines (before ``end_job``): ``_close_drift`` again
        over its final placements (``pls``, of ``frames``) from the estimate its registration
        closed from, so a drifting frame only the cross-engine retries confirmed
        (``retry_kept``) becomes a supporting frame, the reference and odometry-scale evidence
        like any other, all in canonical order (map.md §1, §2; none confirmed twice)."""
        if self._close_base is not None:
            self._close_drift(frames, dict(enumerate(pls)), self._close_base)

    def excluded(self) -> dict[str, Placement]:
        """The frames a planned second registration forces to a placement, by name."""
        return self._exclude

    def plan_rerun(self, pls: list[Placement]) -> bool:
        """Whether the job registers once more (map.md §2 match graph, step 6, P16): when a
        frame the pose-duplicate pass rejects served the job (``_dup_effects``), the second
        registration runs without the pass's duplicates (``_exclude``: their rejection; a
        re-registered stored keyframe: ``_no_rereg``). Once per job (``reran``: its ``duplicates``)."""
        if "duplicates" in self.reran:
            return False
        dups = self._job_duplicates()
        if not dups or not self._dup_effects(dups, pls):
            return False
        again = self._rereg_ids()
        for name, d in dups.items():
            assert d.keyframe is not None
            if d.keyframe in again:
                self._no_rereg.add(d.keyframe)
            else:
                self._exclude[name] = Placement(name, "rejected", "duplicate",
                                                inliers=d.inliers, duplicate_of=d.duplicate_of)
        self.reran.add("duplicates")
        log.info("pose duplicates %s served the job: it registers once more without them",
                 sorted(dups))
        return True

    def _prepare(self, f: FrameIn) -> _Prep:
        """A frame's features, its scene distance (depth image, else, for a suspicious frame,
        its monocular depth: too_close) and retrieval descriptor (frames without a prior)."""
        scene = _median(f.depth)
        mono = None
        if scene is None and f.mono is not None and (f.prior is None or f.check_close):
            mono = f.mono()
            if f.check_close:
                d = _median(mono[0])
                scene = None if d is None else d * self.scale.k
        feats = f.feats if f.feats is not None else features(f.rgb, f.ignore)
        return _Prep(feats, scene, scene is not None and scene < TOO_CLOSE_M,
                     None if mono is None else mono[2])

    def _snapshot(self) -> dict[str, Any]:
        """What registering a job changes in the map (``_restore``): the keyframes and the
        state of the stored ones, the scale evidence, the drift and the origin's records."""
        return {"keyframes": list(self.keyframes), "added": list(self._added),
                "next_id": self.next_id, "scale": copy.deepcopy(self.scale),
                "drift": copy.deepcopy(self.drift), "bias": list(self.bias),
                "origin": (self.created_with, self.origin_c, self.origin_height_m,
                           self.up_check_deg),
                "report": (self.prior_disagreement_deg, list(self.reregistered),
                           list(self.height_inconsistent)),
                "defer": None if self._defer is None else list(self._defer),
                "job": (set(self._evidence_from), copy.deepcopy(self._rereg_by),
                        dict(self._rereg_state)),
                "stored": [(k, k.T.copy(), k.sigma_pos_m, k.sigma_rot_deg, k.state, k.method,
                            k.inliers, k.rgb, k.reason, k.height_ok, k.seed, k.seed_sigma)
                           for k in self.keyframes]}

    def _restore(self, snap: dict[str, Any]) -> None:
        self.keyframes, self._added = list(snap["keyframes"]), list(snap["added"])
        self.next_id, self.bias = snap["next_id"], list(snap["bias"])
        self.scale, self.drift = copy.deepcopy(snap["scale"]), copy.deepcopy(snap["drift"])
        self.created_with, self.origin_c, self.origin_height_m, self.up_check_deg = snap["origin"]
        d, r, h = snap["report"]
        self.prior_disagreement_deg, self.reregistered, self.height_inconsistent = \
            d, list(r), list(h)
        self._defer = None if snap["defer"] is None else list(snap["defer"])
        ev, by, st = snap["job"]
        self._evidence_from, self._rereg_by, self._rereg_state = set(ev), copy.deepcopy(by), \
            dict(st)
        for k, T, sp, sr, state, method, n, rgb, reason, h_ok, seed, seed_s in snap["stored"]:
            k.T, k.sigma_pos_m, k.sigma_rot_deg, k.state, k.method = T.copy(), sp, sr, state, method
            k.inliers, k.rgb, k.reason, k.height_ok = n, rgb, reason, h_ok
            k.seed, k.seed_sigma = seed, seed_s

    def _keep(self, f: FrameIn, pr: _Prep, pl: Placement,
              progress: Callable[[str], None]) -> Keyframe:
        """Store a placed frame as a keyframe (``add``): its depth (a depth image; else its
        monocular depth brought onto the map's scale, its camera height refining that scale)
        and, registered with a depth image, its depth's residual against the map."""
        t = time.perf_counter()
        depth, Kg, kind, desc = self._keyframe_depth(f)
        if not self.keyframes and self.created_with != "none" and f.mono is not None:
            self._up_check(f, pl)
        if kind == "sensor" and pl.status == "registered" and pl.lifted is not None:
            assert pl.T is not None
            # the depth image against the map's lifted points: its source's residual
            a = depth_alignment(pl.T if pl.align_T is None else pl.align_T, *pl.lifted,
                                depth, Kg, f.K)
            if a is not None:
                self.scale.depth_residuals.setdefault("sensor", []).append(math.log(a))
                del self.scale.depth_residuals["sensor"][:-MAX_EVIDENCE]
                self._evidence_from.add(f.name)
        if kind == "mono" and pl.status == "registered":
            assert pl.T is not None
            # its own scale brought onto the map's (the monocular model's metres vary from
            # frame to frame), on the points lifted at the map's current scale
            if pl.lifted is not None:
                a = depth_alignment(pl.T if pl.align_T is None else pl.align_T,
                                    *pl.lifted, depth, Kg, f.K)
                if a is not None:
                    depth = (depth * a).astype(np.float32)
                    self.scale.depth_residuals.setdefault("mono", []).append(math.log(a))
                    del self.scale.depth_residuals["mono"][:-MAX_EVIDENCE]
                    self._evidence_from.add(f.name)
            # its camera height, if known, refines the depth scale before it is fused (a
            # frame kept at its prior is no scale evidence: map.md §2)
            self.add_height_evidence(f, pl.T, depth, Kg)
            k = self.scale.k
            self.scale.estimate(self.prior_span(), self.prior_span("drifting"))
            depth = (depth * (self.scale.k / k)).astype(np.float32)
        kf = self.add(f, pr.feats, pl, depth, Kg, kind, desc)
        progress(f"{f.name}: {pl.status} ({pl.method}, {pl.inliers} inliers)")
        self._keep_s += time.perf_counter() - t
        return kf

    @staticmethod
    def _trusted(f: FrameIn) -> bool:
        return f.prior is not None and f.prior["trust"] in ("metric", "drifting")

    def _register_job(self, frames: list[FrameIn], prep: list[_Prep],
                      forced: dict[int, Placement], progress: Callable[[str], None]
                      ) -> dict[int, Placement]:
        """One registration of the job over its match graph (map.md §2), the ``forced``
        frames (left out of the drift fit) rejected as given:

        1. without a confirmed keyframe, the origin (``_origin``): the first frame with a prior
           in canonical order (name, camera) that passes the checks, the ones before it kept at
           their priors; without any prior, the first frame in canonical order (map.md §1);
        2. rounds (``_rounds``): every frame not yet registered is placed against the confirmed
           keyframes as they stand, and those that register become references together;
        3. a component the confirmed map does not reach is placed by its own priors: a frame
           with a prior and no image pose seeds it at its prior, and stays registered when
           another frame registers through it (else it is undone);
        4. what never registered keeps its last placement: kept at its prior (stored, never a
           reference) or rejected.

        The result depends on the frames, not their request order (canonical order wherever an
        order matters). The pose duplicates are decided after it, once (``end_job``)."""
        out = {i: copy.copy(pl) for i, pl in forced.items()}
        pending = [i for i in range(len(frames)) if i not in forced]
        mine: list[tuple[FrameIn, Placement, Features]] = []
        self._mine = mine  # the cross-engine retries check against them (``retry_kept``)
        kept_before = [k for k in self.keyframes if not k.usable()]
        if not self.confirmed():
            self._origin_of(frames, prep, pending, out, progress)
        results: dict[int, Placement] = {}
        seen: dict[int, tuple[str, ...]] = {}
        self._rounds(frames, prep, pending, out, results, seen, mine, kept_before, progress)
        tried: set[int] = set()
        while True:
            seeds = [i for i in sorted(pending, key=lambda j: _order(frames, j))
                     if i not in tried and self._trusted(frames[i])
                     and results[i].detail == "no_image_pose"]
            if not seeds:
                break
            s = seeds[0]
            tried.add(s)
            snap = self._snapshot()
            before = (dict(results), dict(seen), list(mine), dict(out))
            pl = self._seed(frames[s])
            kf = self._keep(frames[s], prep[s], pl, lambda _: None)
            assert pl.sigma_pos_m is not None and pl.sigma_rot_deg is not None
            # what registers through it is floored by its prior's sigmas (``_seed_floor``)
            kf.seed, kf.seed_sigma = kf.id, (pl.sigma_pos_m, pl.sigma_rot_deg)
            rest = [i for i in pending if i != s]
            if self._rounds(frames, prep, rest, out, results, seen, mine, kept_before, progress):
                # the seed placed again against what registered since: an image pose when the
                # images now give it one (its prior was only the component's start)
                out[s] = results[s] = self._unseed(frames[s], prep[s], pl, kf, mine)
                pending = rest
                progress(f"{frames[s].name}: registered ({out[s].method})")
                continue
            self._restore(snap)
            results, seen, mine, out = before[0], before[1], before[2], before[3]
        for i in sorted(pending, key=lambda j: _order(frames, j)):
            out[i] = results.get(i) or Placement(frames[i].name, "rejected", "no_overlap")
            if out[i].status == "kept_at_prior":
                self._keep(frames[i], prep[i], out[i], progress)
        # a frame the drift fit left out without a measurable pose: kept at its prior
        # (``_drift_outliers``), stored as every kept frame
        for i in sorted(forced, key=lambda j: _order(frames, j)):
            if out[i].status == "kept_at_prior":
                self._keep(frames[i], prep[i], out[i], progress)
        return out

    def _origin_of(self, frames: list[FrameIn], prep: list[_Prep], pending: list[int],
                   out: dict[int, Placement], progress: Callable[[str], None]) -> None:
        """The origin of a map without a confirmed keyframe (``_register_job`` step 1), taken
        out of ``pending``; when no frame with a prior can be it (a map holding only kept
        frames), every frame left is rejected ``no_overlap``."""
        trusted = [i for i in pending if self._trusted(frames[i])]
        if trusted or self.keyframes:
            for i in sorted(trusted, key=lambda j: _order(frames, j)):
                pl = self._origin(frames[i], prep[i].close, prep[i].scene)
                pending.remove(i)
                out[i] = pl
                self._keep(frames[i], prep[i], pl, progress)
                if pl.status == "registered":
                    return
            for j in pending:  # nothing to register on
                out[j] = Placement(frames[j].name, "rejected", "no_overlap")
            pending.clear()
            return
        # the first camera, in canonical name order (map.md §1, §2)
        for i in sorted(pending, key=lambda j: _order(frames, j)):
            pl = self._origin(frames[i], prep[i].close, prep[i].scene)
            pending.remove(i)
            out[i] = pl
            if pl.status != "rejected":
                self._keep(frames[i], prep[i], pl, progress)
                return

    def _seed_floor(self, pl: Placement) -> tuple[str, tuple[float, float]] | None:
        """A seeded component's sigma floor (map.md §2 match graph, step 2; P13): a placement
        whose references (``Placement.refs``) are all in seeded components reports the seed
        prior's sigma combined with its own (root sum of squares: never tighter than the seed
        prior's), whatever the engine or pass; returns the seed it joins, for its keyframe. A
        placement that also rests on a confirmed keyframe outside them links those components
        to the map: their floor ends (``Keyframe.seed`` cleared), and it is not floored."""
        ks = [k for r in pl.refs for k in [self.keyframe(r)] if k is not None and k.usable()]
        seeded = [k for k in ks if k.seed is not None and k.seed_sigma is not None]
        if not seeded:
            return None
        if len(seeded) < len(ks):
            gone = {k.seed for k in seeded}
            for k in self.keyframes:
                if k.seed in gone:
                    k.seed = k.seed_sigma = None
            return None
        k0 = max(seeded, key=lambda k: (k.seed_sigma or (0.0, 0.0), k.id))
        assert k0.seed is not None and k0.seed_sigma is not None
        assert pl.sigma_pos_m is not None and pl.sigma_rot_deg is not None
        sp, sr = k0.seed_sigma
        pl.sigma_pos_m = round(math.hypot(sp, pl.sigma_pos_m), 4)
        pl.sigma_rot_deg = round(math.hypot(sr, pl.sigma_rot_deg), 3)
        return k0.seed, (sp, sr)

    def _unseed(self, f: FrameIn, pr: _Prep, seed: Placement, kf: Keyframe,
                mine: list[tuple[FrameIn, Placement, Features]]) -> Placement:
        """A component's seed once frames registered through it (``_register_job`` step 3):
        placed again against the confirmed keyframes but its own (``place``). When its images
        now give it a pose (the distance limit as in ``_rounds``), its keyframe takes that
        image pose and its sigmas (the seed floor, ``_seed_floor``: a pose resting on a
        confirmed keyframe outside the component links it to the map) and that placement is
        its own, checked against the job's other frames with a prior (``_checked``) and its
        scale evidence recorded as any registered frame's; else it stays at its prior, the
        seed."""
        cands = [k for k in self.candidates_for(f, pr.descriptor, self._predict(f))
                 if k is not kf]
        if not cands:
            return seed
        pl = self.place(f, pr.feats, pr.descriptor, cands=cands)
        if pl.status != "registered" or (pr.close and pl.inliers < TOO_CLOSE_INLIERS):
            return seed
        pl = self._checked(f, pl, mine, pr.feats)
        if not pr.close:
            self._evidence(f, pl)
        assert pl.T is not None and pl.sigma_pos_m is not None and pl.sigma_rot_deg is not None
        floor = self._seed_floor(pl)
        kf.T, kf.sigma_pos_m, kf.sigma_rot_deg = pl.T.copy(), pl.sigma_pos_m, pl.sigma_rot_deg
        kf.method, kf.inliers = pl.method or "pnp", pl.inliers
        if floor is None:  # its pose rests on confirmed keyframes outside: the component links
            for k in self.keyframes:
                if k.seed is not None and k.seed == kf.id:
                    k.seed = k.seed_sigma = None
        kf.height_ok = self.height_gate(kf.height_m, kf.height_sigma_m, kf.T, kf.sigma_pos_m)
        self.height_inconsistent = [n for n in self.height_inconsistent if n != kf.name] + (
            [] if kf.height_ok else [kf.name])
        pl.keyframe = kf.id
        log.info("%s: a component's seed, placed by its images now (%s)", f.name, pl.method)
        return pl

    def _seed(self, f: FrameIn) -> Placement:
        """A component's seed (``_register_job`` step 3): registered at its prior as placed,
        with the prior's sigmas, prior residual 0."""
        kept = self.kept_at(f, "no_overlap", 0)
        assert kept.T is not None
        return Placement(f.name, "registered", None, kept.T, kept.sigma_pos_m,
                         kept.sigma_rot_deg, 0, "prior", (0.0, 0.0), has_prior=True)

    def _rounds(self, frames: list[FrameIn], prep: list[_Prep], pending: list[int],
                out: dict[int, Placement], results: dict[int, Placement],
                seen: dict[int, tuple[str, ...]], mine: list[tuple[FrameIn, Placement, Features]],
                kept_before: list[Keyframe], progress: Callable[[str], None]) -> bool:
        """Rounds over the match graph (``_register_job`` step 2) until one registers nothing:
        in canonical order, each frame of ``pending`` is placed against the confirmed keyframes
        as they stand (again only when its candidates changed since its last placement), a
        frame without a prior joining a standpoint, the distance limit applied; one that
        registers becomes a reference at once (``_checked``, its scale evidence,
        ``_keep``) and leaves ``pending``. After each round the stored kept keyframes the new
        ones overlap are placed again (``reregister``), confirmed ones becoming references too.
        A frame that fails in one round is placed again in the next against what registered
        since: the fixed point does not depend on the request order. Returns whether any frame
        registered."""
        grown = False
        while pending:
            added = []
            for i in sorted(pending, key=lambda j: _order(frames, j)):
                f, pr = frames[i], prep[i]
                pred = self._predict(f)
                cands = self.candidates_for(f, pr.descriptor, pred)
                ids = tuple(k.id for k in cands)
                if i in results and seen.get(i) == ids:
                    continue  # nothing new to match: as it was
                seen[i] = ids
                pl = self._place_cached(i, f, pr, pred, cands)
                if f.prior is None and pl.status == "registered" and self.standpoint is not None:
                    pl = self.standpoint(f, pr.feats, pl)
                # the 300-inlier exemption counts image feature inliers only (api.md §4)
                if pr.close and pl.inliers < TOO_CLOSE_INLIERS:
                    pl = self._too_close(f, pl, pr.scene)
                results[i] = pl
                if pl.status != "registered":
                    continue
                pending.remove(i)
                pl = self._checked(f, pl, mine, pr.feats)
                if not pr.close:
                    self._evidence(f, pl)
                out[i] = results[i] = pl
                added.append(self._keep(f, pr, pl, progress))
            if not added:
                break
            grown = True
            self.reregistered += self.reregister([k for k in kept_before if not k.usable()],
                                                 added)
        return grown

    def _place_cached(self, i: int, f: FrameIn, pr: _Prep,
                      pred: tuple[Array, float, float] | None, cands: list[Keyframe]
                      ) -> Placement:
        """``place`` of job frame ``i``, once per prediction, candidate set (their ids, names
        and poses) and map depth scale (its factor, source and sigma: the lifted points, sigmas
        and scale evidence depend on it): a drift pass that predicts it as before, on the same
        scale, reuses it. A copy each time (its fields are reassigned, never changed in place,
        by the steps after)."""
        key = (i, None if pred is None else (pred[0].tobytes(), pred[1], pred[2]),
               tuple((k.id, k.name, k.T.tobytes()) for k in cands),
               (self.scale.k, self.scale.source, self.scale.sigma_rel))
        if key not in self._placed:
            self._placed[key] = self.place(f, pr.feats, pr.descriptor, cands=cands)
        return copy.copy(self._placed[key])

    def _drift_evidence(self, frames: list[FrameIn], out: dict[int, Placement]
                        ) -> list[_DriftObs]:
        """The job's frames with a drifting prior whose images give a pose of their own
        (registered, rejected ``prior_conflict``, or kept ``conflict_unsupported`` when it can
        support an estimate: ``_can_support``, map.md §1 "even if it conflicts"): the drift
        fit's observations."""
        obs = []
        for i, f in enumerate(frames):
            pl = out.get(i)
            if pl is None or f.prior is None or f.prior["trust"] != "drifting":
                continue
            if pl.status == "registered" and pl.method in ("origin", "prior"):
                assert pl.T is not None
                T, sh = pl.T, (MIN_SIGMA_POS_M, MIN_SIGMA_ROT_DEG)
            elif (pl.status == "registered" or pl.reason == "prior_conflict"
                  or (pl.detail == "conflict_unsupported" and _can_support(pl))) \
                    and pl.align_T is not None and pl.image_sigmas is not None:
                T, sh = pl.align_T, pl.image_sigmas
            else:
                continue
            obs.append(_DriftObs(i, f.name, T, np.asarray(f.prior["T_map_cam"], np.float64),
                                 _prior_sigmas(f.prior), sh, pl.method == "rotation", pl))
        return obs

    def _fit_drift(self, frames: list[FrameIn], out: dict[int, Placement], used: Drift
                   ) -> tuple[Drift | None, dict[int, Placement]]:
        """The robust drift estimate (map.md §1 "Drift estimate") over the job's image-confirmed
        drifting-prior frames (``_drift_evidence``): the estimate in force (``used``) and one
        trial per observation (that frame as the reference) each count the observations within
        ``K_SIGMA`` × sqrt(σ_prior² + σ_hypothesis²) of their prediction (position and
        rotation); the trial with the most, refined over them (``_refined_drift``), replaces
        the estimate in force only when it has more. A change larger than ``K_SIGMA`` × its own
        sigma (``_drift_sigma``), and a first estimate (a map whose frame no drifting prior fixed,
        before its first), need min(``DRIFT_MIN_SUPPORT``, n) supporting inliers, n the job's
        drifting-prior frames that can support it, whatever their status (``_can_support``),
        ``DRIFT_SUPPORT_SHARE`` of those n, and never one alone (``DRIFT_NEVER_ALONE``). A
        refused first estimate leaves the drift at the identity (priors as sent, with their
        sigmas): only the frames that conflict with their priors under it are rejected. Returns
        the new estimate (None: the one in force stays) and the observations it leaves out,
        each a ``prior_conflict`` rejection."""
        obs = self._drift_evidence(frames, out)
        able = {i for i, pl in out.items() for p in [frames[i].prior]
                if p is not None and p["trust"] == "drifting" and _can_support(pl)}
        if not obs:
            return None, {}
        base = _drift_inliers(used, obs)
        best, best_in = None, base
        for o in sorted(obs, key=lambda o: o.name):
            trial = Drift(o.T, o.P, used.odo_scale)
            got = _drift_inliers(trial, obs)
            if len(got) > len(best_in):
                best, best_in = trial, got
        if best is None:
            return None, _drift_outliers(used, obs, base)
        fitted = _refined_drift(best, [o for o in obs if o.index in best_in])
        previous = used.ref_T is not None or self.created_with == "drifting"
        ref = best.ref_prior
        assert ref is not None
        dp, dr = pose_residual(fitted.predict(ref, 0.0, 0.0)[0], used.predict(ref, 0.0, 0.0)[0])
        sp, sr = _drift_sigma([o for o in obs if o.index in best_in])
        large = dp > K_SIGMA * sp or dr > K_SIGMA * sr
        # robust support (map.md §1): the inliers that can support it, at least min(3, n) of the
        # n that can, at least half of them, never one alone
        n, got_n = len(able), len(best_in & able)
        supported = got_n >= max(min(DRIFT_MIN_SUPPORT, n), DRIFT_NEVER_ALONE) \
            and got_n >= DRIFT_SUPPORT_SHARE * n
        if (large or not previous) and not supported:
            log.info("drift change of %.3f m / %.2f deg refused: %d of %d frames support it",
                     dp, dr, got_n, n)
            return None, _drift_outliers(used, obs, base)
        fitted.fit_sigma = (sp, sr)
        fitted.support = [o.P[:3, 3].tolist() for o in obs if o.index in best_in]
        return fitted, _drift_outliers(fitted, obs, best_in)

    def _close_drift(self, frames: list[FrameIn], out: dict[int, Placement], used: Drift
                     ) -> None:
        """The drift estimate after the job: the one the job was registered under, then each
        registered drifting frame in canonical order (map.md §2: the request order never
        matters) as the new reference and a supporting frame (``Drift.confirm``: its
        odometry-scale evidence)."""
        self.drift = copy.deepcopy(used)
        for i in sorted(range(len(frames)), key=lambda j: _order(frames, j)):
            f = frames[i]
            if f.prior is None or f.prior["trust"] != "drifting":
                continue
            P = np.asarray(f.prior["T_map_cam"], np.float64)
            pl = out[i]
            if pl.status == "registered" and pl.T is not None:
                self.drift.confirm(pl.T, P)

    def keyframe(self, kid: str) -> Keyframe | None:
        """The keyframe of id ``kid``, or None."""
        return next((k for k in self.keyframes if k.id == kid), None)

    def _origin(self, f: FrameIn, close: bool, scene: float | None) -> Placement:
        """A frame placed while the map has no image-confirmed keyframe (map.md §1): with a
        prior, at it (the priors' frame is the map frame): registered as the origin, or kept at
        it when its image failed a check or it is too close (api.md §4, the first failing check
        in the order of ``FrameIn.unconfirmed``, then ``too_close``); without one, the first
        camera (``first_camera``), rejected ``too_close`` when too close."""
        p = f.prior
        if p is not None and p["trust"] in ("metric", "drifting"):
            P = np.asarray(p["T_map_cam"], np.float64)
            sp, sr = _prior_sigmas(p)
            if self.created_with is None:
                self.created_with = p["trust"]
                self.origin_c = P[:3, 3].copy()
            if f.unconfirmed is not None or close:
                return self.kept_at(f, f.unconfirmed or "too_close", 0,
                                    scene if f.unconfirmed is None else None)
            return Placement(f.name, "registered", None, P, sp, sr, 0, "origin", (0.0, 0.0),
                             has_prior=True)
        if close:  # an origin has no inliers: too close, it cannot be registered
            return Placement(f.name, "rejected", "too_close", scene_m=scene)
        h = f.height_m if f.height_m is not None and f.height_m > 0 else None
        self.created_with, self.origin_height_m = "none", h
        P = first_camera(None if f.mono is None else f.mono()[3], h)
        return Placement(f.name, "registered", None, P, MIN_SIGMA_POS_M, MIN_SIGMA_ROT_DEG, 0,
                         "origin", None)

    def _too_close(self, f: FrameIn, pl: Placement, scene: float | None) -> Placement:
        """A placed frame within the distance limit with fewer than ``TOO_CLOSE_INLIERS``
        inliers (api.md §4): kept at its prior when it has one, its reason the check its image
        failed, else ``too_close``; rejected ``too_close`` without one."""
        pred = self._predict(f)
        if pred is None:
            return Placement(f.name, "rejected", "too_close", inliers=pl.inliers, scene_m=scene)
        if f.unconfirmed is not None:
            return self.kept_at(f, f.unconfirmed, pl.inliers, pred=pred)
        return self.kept_at(f, "too_close", pl.inliers, scene, pred=pred)

    def overlaps(self, kept: Keyframe, other: Keyframe) -> bool:
        """Whether a new keyframe ``other`` makes a stored kept one eligible for re-registration
        (map.md §2): their matches (``epipolar``) hold at least the registration minimum
        ``MIN_INLIERS`` of inliers and cover at least ``REREGISTER_OVERLAP`` of the kept frame's
        image (the overlap of map.md §1 with the kept frame as the reference, ``cell_share``).
        Every new frame is matched: the kept frame's pose is a prior, which re-registration
        exists to correct, so no pose test chooses the pairs."""
        self._load_kf(kept)
        if kept.feats is None or other.feats is None:
            return False
        e = self._epipolar(kept.feats, kept.K, other.feats, other.K, MIN_INLIERS)
        return e is not None and cell_share(e.ua, kept.K) >= REREGISTER_OVERLAP

    def reregister(self, kept: list[Keyframe], new: list[Keyframe]) -> list[dict[str, Any]]:
        """The stored ``kept_at_prior`` keyframes ``kept`` that a new image-confirmed keyframe
        of ``new`` overlaps (``overlaps``), each placed again against the confirmed map
        (``place``, from its pose as kept, no scale evidence: map.md §2). One whose images now
        confirm it takes its registered pose and sigmas, becomes confirmed and is fused (its
        depth as stored, at the map's scale now). One a second registration of the job leaves
        out (``_no_rereg``) stays kept. Returns one ``{name, keyframe,
        displacement_m, displacement_deg}`` per keyframe confirmed: how far confirmation moved
        it from its prior-placed pose."""
        out: list[dict[str, Any]] = []
        for kf in kept:
            if kf.usable() or kf.prior is None or kf.id in self._no_rereg:
                continue
            by = next((n.id for n in new if self.overlaps(kf, n)), None)
            if by is None:
                continue
            got = self._place_again(kf)
            if got is None:
                continue
            pl, rgb = got
            # what it rests on, and its state before (a pose duplicate undoes it: map.md §2)
            self._rereg_by[kf.id] = {by, *pl.refs}
            self._rereg_state[kf.id] = (kf.T.copy(), kf.sigma_pos_m, kf.sigma_rot_deg,
                                        kf.state, kf.method, kf.inliers, kf.rgb, kf.reason,
                                        kf.height_ok, kf.seed, kf.seed_sigma)
            assert pl.T is not None and pl.sigma_pos_m is not None \
                and pl.sigma_rot_deg is not None
            seed = self._seed_floor(pl)  # through a seeded component: its floor
            # the too-close exception: its depth is in the cloud already, its detections in the
            # objects
            had, had_objects = kf.contributes(), kf.detects()
            dp, dr = pose_residual(kf.T, pl.T)
            kf.T, kf.sigma_pos_m, kf.sigma_rot_deg = pl.T.copy(), pl.sigma_pos_m, pl.sigma_rot_deg
            kf.state, kf.method, kf.inliers = "fused", pl.method or "pnp", pl.inliers
            kf.rgb, kf.reason = rgb, None
            kf.seed, kf.seed_sigma = seed or (None, None)
            # the height gate at its confirmed pose (map.md §2)
            kf.height_ok = self.height_gate(kf.height_m, kf.height_sigma_m, kf.T, kf.sigma_pos_m)
            if not kf.height_ok and kf.name not in self.height_inconsistent:
                self.height_inconsistent.append(kf.name)
            ref = self._grid(kf)
            if not had and ref is not None and kf.depth_kind != "none" and kf.K_grid is not None:
                self._fuse_later(kf, rgb, None, ref.depth, kf.K_grid)
            out.append({"name": kf.name, "keyframe": kf.id, "displacement_m": round(dp, 4),
                        "displacement_deg": round(dr, 3), "content": had,
                        "objects": had_objects})
            log.info("%s: re-registered (%.3f m, %.2f deg from its prior)", kf.name, dp, dr)
        return out

    def retry_kept(self, kept: list[tuple[FrameIn, Keyframe]], new: list[Keyframe]
                   ) -> dict[str, Placement]:
        """The running job's own frames with a prior that ended ``kept_at_prior`` for want of an
        image pose (``no_overlap``), placed again once other job frames registered after them
        (``new``: those of another engine, MapAnything): one more round of the match graph
        (map.md §2, a job frame is an edge once it is registered), in canonical order, repeated
        until a pass confirms nothing new (a fixed point: a frame linked to the map only through
        one confirmed later in the same pass is tried again). A frame is placed again only when
        one of ``new``, or a frame confirmed here, is among its candidates. One that registers
        goes through the prior checks against the job's other frames (``_checked``:
        ``_against_priors``), records its scale evidence, takes its pose and sigmas and is
        fused (``Keyframe.contributes``; the job's pose-duplicate pass follows, ``end_job``).
        One whose images now give a pose that fails the prior checks takes that verdict
        (api.md §4, P16): rejected ``prior_conflict`` (its keyframe dropped) or kept with
        ``detail`` ``conflict_unsupported`` / ``loose_image_pose``. Returns the new placements
        by frame name. Frames kept for a failed check, or too close, stay kept. The drift fit
        over the job with these frames follows (``plan_drift_rerun``)."""
        ids = {k.id for k in new}
        out: dict[str, Placement] = {}
        tried: dict[str, frozenset[str]] = {}
        grown = True
        while grown:
            grown = False
            for f, kf in sorted(kept, key=lambda t: (natural_key(t[0].name), t[0].camera_id)):
                if (f.name in out and out[f.name].status != "kept_at_prior") or kf.usable() \
                        or f.unconfirmed is not None or kf.reason != "no_overlap" \
                        or tried.get(f.name) == frozenset(ids):
                    continue
                tried[f.name] = frozenset(ids)
                grown |= self._retry_one(f, kf, ids, out)
        return out

    def _retry_one(self, f: FrameIn, kf: Keyframe, ids: set[str], out: dict[str, Placement]
                   ) -> bool:
        """One kept job frame placed again (``retry_kept``): whether it registered (its
        keyframe id joins ``ids``, its placement into ``out``; a new verdict without a
        registration too)."""
        self._load_kf(kf)
        assert kf.feats is not None  # a job keyframe keeps its features
        cands = self.candidates_for(f, kf.descriptor, self._predict(f))
        if not any(c.id in ids for c in cands):
            return False
        pl = self.place(f, kf.feats, kf.descriptor, cands=cands)
        drifting = f.prior is not None and f.prior["trust"] == "drifting"
        if pl.align_T is not None and drifting:
            self._retried_drifting = True  # an image pose for the drift fit
        if pl.status == "rejected":  # the images confirm a conflict (api.md §4)
            self.keyframes = [k for k in self.keyframes if k is not kf]
            self._added = [k for k in self._added if k is not kf]
            out[f.name] = pl
            return False
        if pl.status != "registered":
            if pl.detail != kf_detail(out.get(f.name)) and pl.detail != "no_image_pose":
                pl.keyframe = kf.id
                out[f.name] = pl
            return False
        pl = self._checked(f, pl, self._mine, kf.feats)
        self._evidence(f, pl)
        assert pl.T is not None and pl.sigma_pos_m is not None \
            and pl.sigma_rot_deg is not None
        seed = self._seed_floor(pl)  # through a seeded component: its floor
        kf.T, kf.sigma_pos_m, kf.sigma_rot_deg = pl.T.copy(), pl.sigma_pos_m, pl.sigma_rot_deg
        kf.state, kf.method, kf.inliers = "fused", pl.method or "pnp", pl.inliers
        kf.reason = None
        kf.seed, kf.seed_sigma = seed or (None, None)
        kf.height_ok = self.height_gate(kf.height_m, kf.height_sigma_m, kf.T, kf.sigma_pos_m)
        if not kf.height_ok and kf.name not in self.height_inconsistent:
            self.height_inconsistent.append(kf.name)
        ref = self._grid(kf)
        if ref is not None and kf.K_grid is not None and kf.depth_kind != "none":
            self._fuse_later(kf, f.rgb, f.ignore, ref.depth, kf.K_grid)
        pl.keyframe = kf.id
        out[f.name] = pl
        ids.add(kf.id)
        return True

    def plan_drift_rerun(self, frames: list[FrameIn], pls: list[Placement]) -> bool:
        """After the cross-engine fixed point (``retry_kept``), the robust drift fit over the
        job's frames with a prior as they ended (``pls``, of ``frames``), when a retry gave a
        drifting frame an image pose (map.md §1: the estimate is fitted over every
        image-confirmed drifting frame, whichever engine's frames confirmed it). Under the
        estimate the job registered under (``_job_used``), when the fit changes it or leaves
        out a frame not already left out, the job registers once more from the fitted
        estimate with those frames forced to the fit's verdict (``_start_drift``, ``_exclude``).
        Once per job (``reran``: its ``drift``); whether it does."""
        if not self._retried_drifting or "drift" in self.reran or self._job_used is None:
            return False
        out = dict(enumerate(pls))
        got, outliers = self._fit_drift(frames, out, self._job_used)
        new = {i for i, pl in outliers.items()
               if (out[i].status, out[i].reason, out[i].detail) != (pl.status, pl.reason,
                                                                      pl.detail)}
        if got is None and not new:
            return False
        self._start_drift = got or self._job_used
        self._exclude.update({frames[i].name: pl for i, pl in outliers.items()})
        self.reran.add("drift")
        log.info("the cross-engine drift fit changed the job's: it registers once more")
        return True

    def _place_again(self, kf: Keyframe) -> tuple[Placement, NDArray[np.uint8]] | None:
        """A stored kept keyframe placed again (``reregister``): its placement when its images
        register it, with its stored image; None otherwise (its features or image missing, or
        still unconfirmed). Only images confirm it, never depth (api.md §4, map.md §2: a frame
        kept for a failed check is eligible too): it is placed without its depth image, and a
        placement by depth is no confirmation. Unconfirmed, it stays kept with its reason."""
        self._load_kf(kf)
        path = self.image_path(kf.id)
        if kf.feats is None or path is None or kf.prior is None:
            return None
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        rgb = np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), np.uint8)
        prior = {"trust": kf.prior["trust"], "T_map_cam": kf.prior["T"],
                 "sigma_pos_m": kf.prior["sigma_pos_m"], "sigma_rot_deg": kf.prior["sigma_rot_deg"]}
        f = FrameIn(kf.name, kf.camera_id, rgb, kf.K, prior, None,
                    features_count=len(kf.feats.kps))
        pl = self.place(f, kf.feats, kf.descriptor, (kf.T, kf.sigma_pos_m, kf.sigma_rot_deg))
        # its registration re-run whole, the distance limit included (as ``_rounds``: only
        # image feature inliers count toward ``TOO_CLOSE_INLIERS``)
        scene = None if kf.median_depth is None else kf.median_depth * self._depth_factor(kf)
        if scene is not None and scene < TOO_CLOSE_M and pl.inliers < TOO_CLOSE_INLIERS:
            return None
        return (pl, rgb) if pl.status == "registered" else None

    def _level_by_floor(self, pls: list[Placement]) -> None:
        """A new camera-only map's up from the floor plane too (map.md §1): the first-camera
        rule levelled it by the origin frame's gravity; the large horizontal plane below the
        cameras that holds the most points (``floor_candidates``, its normal within
        ``FLOOR_TILT_DEG`` of that up) refines it. The whole map turns about the origin
        camera's centre (it stays straight above the origin) by the rotation bringing that
        normal onto +z, then about +z so its viewing direction stays +x on the horizontal
        plane. Nothing turns without such a plane or when it tilts by less than
        ``LEVEL_MIN_DEG``."""
        pts = self.cloud.points()
        centres = np.array([k.centre for k in self.keyframes])
        below = float(centres[:, 2].min())
        q = pts - np.array([0.0, 0.0, below])
        planes = floor_candidates(q, MAP_FLOOR_SHARE)
        got = None if not planes else horizontal_plane(
            q[np.isfinite(q).all(axis=1) & (q[:, 2] < -MIN_CAMERA_HEIGHT)], planes[0],
            LEVEL_BAND_REL)
        if got is None:
            return
        n = got[0]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(n[2])))))
        if tilt < LEVEL_MIN_DEG:
            return
        axis = unit(np.cross(n, [0.0, 0.0, 1.0]))
        R = exp_so3(axis * math.radians(tilt))
        fwd = R @ self.keyframes[0].T[:3, 2]
        yaw = math.atan2(float(fwd[1]), float(fwd[0]))
        R = exp_so3(np.array([0.0, 0.0, -yaw])) @ R
        c0 = self.keyframes[0].centre.copy()
        M = pose(R, c0 - R @ c0)
        for k in self.keyframes:
            k.T = M @ k.T
        for pl in pls:
            pl.T = None if pl.T is None else M @ pl.T
            pl.align_T = None if pl.align_T is None else M @ pl.align_T
        self.cloud = self.cloud.rotated(R, M[:3, 3])
        log.info("levelled the camera-only map by its floor plane (%.2f deg)", tilt)

    def _checked(self, f: FrameIn, pl: Placement,
                 mine: list[tuple[FrameIn, Placement, Features]], feats: Features) -> Placement:
        """``pl``, a registered frame with a prior checked against the update's others
        (``_against_priors``)."""
        if pl.status == "registered" and f.prior is not None:
            self._against_priors(f, pl, mine, feats)
        return pl

    def _against_priors(self, f: FrameIn, pl: Placement,
                        mine: list[tuple[FrameIn, Placement, Features]], feats: Features) -> None:
        """A registered frame with a prior against the earlier ones of the update (map.md §1
        "Priors against images"): for each that overlaps it by at least ``OVERLAP_MIN`` (the
        overlap of map.md §1, no named reference: the smaller of the two ``coverage``s; a pair
        is not matched only when neither frame sees any point the other matched, at the
        committed poses or at the priors, in an image widened by ``OVERLAP_VIEW_PAD`` on every
        side (``_may_overlap``): a cost limit that cannot skip a pair the check must test, as
        the committed poses carry the very rotation error it corrects), the disagreement between the images'
        relative rotation and the priors' raises ``prior_disagreement_deg``; beyond ``K_SIGMA``
        of the two priors' combined rotation sigmas, this frame's rotation is corrected from the
        images (relative to the earlier frame's committed rotation). Then it joins ``mine``."""
        # registered with a prior in a round: placed by its image pose (``_register``)
        assert f.prior is not None and pl.T is not None and pl.align_T is not None \
            and pl.lifted is not None
        P = np.asarray(f.prior["T_map_cam"], np.float64)
        sr = _prior_sigmas(f.prior)[1]
        for g, q, gf in mine:
            assert g.prior is not None and q.T is not None and q.align_T is not None \
                and q.lifted is not None
            Q = np.asarray(g.prior["T_map_cam"], np.float64)
            if not _may_overlap(pl.lifted[0], pl.T, P, f.K, q.lifted[0], q.T, Q, g.K) \
                    or min(self._coverage(feats, f.K, gf, g.K)) < OVERLAP_MIN:
                continue
            img_rel = q.align_T[:3, :3].T @ pl.align_T[:3, :3]
            dis = rot_deg(img_rel.T @ (Q[:3, :3].T @ P[:3, :3]))
            self.prior_disagreement_deg = max(self.prior_disagreement_deg, round(dis, 3))
            if dis > K_SIGMA * math.hypot(sr, _prior_sigmas(g.prior)[1]):
                T = pl.T.copy()
                T[:3, :3] = q.T[:3, :3] @ img_rel
                assert pl.prior_ref is not None  # a frame its image pose and prior placed
                pl.prior_residual = pose_residual(T, pl.prior_ref)
                pl.map_residual = _displacement(T, pl.align_T, pl.method == "rotation")
                pl.T = T
        mine.append((f, pl, feats))

    def sensor_depth(self, k: Keyframe) -> tuple[NDArray[np.float32], Intrinsics] | None:
        """A keyframe's depth image (``depth_kind`` sensor) at the map's depth scale now, on its
        grid, with the grid's intrinsics; None for a monocular or depth-less keyframe."""
        if k.depth_kind != "sensor" or k.K_grid is None:
            return None
        ref = self._grid(k)
        return None if ref is None else (ref.depth, k.K_grid)

    def _grid(self, k: Keyframe) -> DepthRef | None:
        """A keyframe's depth grid at the map's depth scale now, as a depth reference."""
        if k.grid is None and k.K_grid is not None:
            p = self.root / KF_DIR / f"{k.id}.depth.npy"
            if p.exists():
                k.grid = np.load(p).astype(np.float32)
        if k.grid is None or k.K_grid is None:
            return None
        return DepthRef(k.T, (k.grid * self._depth_factor(k)).astype(np.float32), k.K_grid.K())

    def _up_check(self, f: FrameIn, pl: Placement) -> None:
        """The angle between the origin prior's up axis and the monocular gravity estimate."""
        assert f.mono is not None and pl.T is not None
        up = f.mono()[3]
        if up is None:
            return
        u = pl.T[:3, :3] @ unit(np.asarray(up, np.float64))
        self.up_check_deg = round(math.degrees(math.acos(max(-1.0, min(1.0, float(u[2]))))), 3)

    def floor(self) -> tuple[float | None, float]:
        """The version's one floor estimate (map.md §5-§6): its height and the spread of the
        floor's points about it (``floor_spread``), which the objects and the grid both use.
        The floor is the surface the robot stands on, among the large horizontal planes of the
        cloud below every image-confirmed camera (``floor_candidates``, holding at least
        ``MAP_FLOOR_SHARE`` of the best one's points):

        * priors fixed the map frame: the plane nearest z 0 (their origin is where the robot
          stands), not a lower floor seen past a worktop's edge;
        * a camera-only map whose first frame had a camera height: z 0 (the floor that height
          defines, map.md §1), refined to the observed plane nearest it when that lies within
          ``FLOOR_BAND_SIGMAS`` of its own points' spread (``floor_sigma_m``);
        * otherwise the lowest; without one, the lowest dense layer."""
        pts = self.cloud.points()
        centres = np.array([k.centre for k in self.confirmed()]).reshape(-1, 3)
        below = float(centres[:, 2].min()) if len(centres) else 0.0
        floor = None
        if self.created_with == "none" and self.origin_height_m is not None:
            floor = 0.0
            if len(pts):
                # the observed plane nearest the height's floor, accepted within
                # FLOOR_BAND_SIGMAS of its own points' spread (map.md §6)
                planes = [below - h for h in floor_candidates(
                    pts - np.array([0.0, 0.0, below]), MAP_FLOOR_SHARE)]
                if planes:
                    z = min(planes, key=abs)
                    if abs(z) <= FLOOR_BAND_SIGMAS * floor_spread(pts[:, 2], z):
                        floor = z
        elif len(pts):
            planes = floor_candidates(pts - np.array([0.0, 0.0, below]), MAP_FLOOR_SHARE)
            if planes and self.created_with in ("metric", "drifting"):
                # priors fixed the frame: the plane the robot stands on, nearest their origin
                # height (z 0), not a lower floor seen past a worktop's edge (map.md §6)
                floor = below - min(planes, key=lambda h: abs(below - h))
            else:
                floor = below - max(planes) if planes else lowest_layer(pts[:, 2], below)
        if floor is None:
            return None, FLOOR_SIGMA_DEFAULT_M
        return floor, floor_spread(pts[:, 2], floor)

    def bounds(self) -> NDArray[np.float64]:
        """The points bounding the map: the cloud's and the cameras'."""
        pts = self.cloud.points()
        centres = np.array([k.centre for k in self.keyframes]).reshape(-1, 3)
        return np.vstack([pts, centres]) if len(pts) else centres

    def floor_and_bounds(self) -> tuple[float | None, NDArray[np.float64]]:
        """The floor height (``floor``) and the points bounding the map (``bounds``)."""
        return self.floor()[0], self.bounds()
