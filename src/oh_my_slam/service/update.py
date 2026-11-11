"""An update job (O1 anchored frame and priors, O4 per-input status; map.md §1, api.md §4).

1. **Checks.** A frame whose image does not decode or does not have its camera's size is
   ``unusable_image``; a frame whose name the map or the request already holds is a
   ``duplicate`` of that name (the pose duplicate is decided after registration). The others
   are measured (``checks``): one that is ``plain``, ``blur`` or has ``few_features`` is
   rejected before it reaches the mapper, or with a prior kept at it (``kept_at_prior``, api.md
   §4); the median scene depth is kept for the distance limit (``too_close``), decided after
   registration (``too_close``: unless the frame has ``TOO_CLOSE_INLIERS`` inliers; a frame
   with a prior is then kept at it).
2. **Registration.** The other frames, rectified to their camera's canonical pinhole
   (``cameras``) with its intrinsics given to the mapper in place of EXIF and focal estimation,
   are mapped into a clone of the map's mapper map (``backend``): COLMAP SIFT, global or
   incremental SfM, the multi-view fallback, depth alignment and fusion (``mapping.api.update``).
   A frame the mapper does not place is ``few_features`` or ``no_overlap``.
3. **Anchoring.** The similarity ``S`` from the mapper's internal frame to the map frame is fitted
   to the anchors (``anchor``): on a map's first update the metric priors (else the drifting
   ones; else the first-camera rule), afterwards the previous version's poses of the frames both
   versions hold together with every metric prior. The map frame therefore never moves: a
   rebuild that re-estimates stored poses is aligned back onto them. The scale comes from the
   metric priors, else depth PNGs, else camera heights, else the monocular depth.
4. **Post-checks.** A registered frame whose pose disagrees with its prior by more than
   ``anchor.K_SIGMA`` of the prior's own sigmas is ``prior_conflict`` (the mapper gives one
   hypothesis: none other lies near the prior); one whose scene depth is below
   ``checks.TOO_CLOSE_M`` and that has fewer than ``TOO_CLOSE_INLIERS`` inliers is
   ``too_close`` (kept at its prior when it has one); one within ``duplicate_limits`` of a
   stored fused frame or an earlier registered frame of the request, of its camera, is a
   ``duplicate``. If any, registration runs once more without them. A frame with a prior that
   the mapper does not register is ``kept_at_prior``: at its metric prior, or at its drifting
   prior mapped by the similarity fitted on the update's registered drifting frames.
5. **Quality and gate.** The quality report (map.md §4) and ``accept_if``; nothing registered is
   a hard limit. A passing job commits one version.
"""

from __future__ import annotations

import functools
import hashlib
import math
import shutil
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.constants import FLOOR_BAND_SIGMAS, FLOOR_SIGMA_DEFAULT_M
from oh_my_slam.core.images import png_bytes, save_jpeg
from oh_my_slam.core.log import get_logger
from oh_my_slam.core.types import Intrinsics
from oh_my_slam.mapping import anchor, checks, sigmas, standpoint
from oh_my_slam.mapping.anchor import Anchor, Sim3
from oh_my_slam.mapping.anchored import (
    DRIFT_MIN_SUPPORT,
    DRIFT_NEVER_ALONE,
    DRIFT_SUPPORT_SHARE,
    MEASURABLE_INLIERS,
    OVERLAP_MIN,
    Features,
    coverage,
    duplicate_among,
    features,
    floor_spread,
    lowest_layer,
    natural_key,
)
from oh_my_slam.mapping.anchored import (
    drift_grown as anchored_drift_grown,
)
from oh_my_slam.mapping.quality import displacement_quality, gate, residual_stats
from oh_my_slam.service.backend import Backend, InternalFrame, InternalMap, NothingRegisteredError
from oh_my_slam.service.cameras import (
    Camera,
    decode_mask,
    decode_png16,
    decode_rgb,
    mapper_depth,
    mapper_image,
)
from oh_my_slam.service.errors import ApiError
from oh_my_slam.service.jobs import Job
from oh_my_slam.service.store import POSE_STATES, Store, clone_tree
from oh_my_slam.service.timings import FUSION_STAGES, Timings

log = get_logger("oh_my_slam.service.update")

MONOCULAR_SIGMA_REL = 0.10  # MoGe-2 metric depth: about 10 % scale error between scenes
MIN_SCALE_SIGMA_REL = 0.005
DEFAULT_HEIGHT_SIGMA_M = 0.02
MIN_CAMERA_HEIGHT_M = 0.05  # internal camera heights below this tell no scale
LOW_CONFIDENCE_SIGMA_M = 0.10  # registered frames above it count in low_confidence_share
TOO_CLOSE_INLIERS = 300  # a frame this well registered is never too close (as mapping.anchored)
SIGMA_POS_FLOOR_M = 0.01  # pose sigma proxy floors (README "Per-input status")
SIGMA_ROT_FLOOR_DEG = 0.1
DEFAULT_SCENE_DEPTH_M = 2.0
LOW_CONFIDENCE_FACTOR = 3.0
JPEG_QUALITY = 95
CHECK_THREADS = 4  # frames measured at once (OpenCV releases the GIL)


@dataclass
class FrameWork:
    """One frame of a request, as the job runs it."""

    index: int
    name: str
    camera_id: str
    image: bytes
    depth: bytes | None = None
    prior: dict[str, Any] | None = None
    height_m: float | None = None
    height_sigma_m: float | None = None
    mask: bytes | None = None  # the frame's own ignore mask (PNG), combined with the camera's
    # rejected before any check (api.md §4): ``duplicate`` (a name already in the request) or
    # ``unusable_image`` (an image that does not decode or is not its camera's size)
    reject: str | None = None

    @property
    def trust(self) -> str:
        return "none" if self.prior is None else str(self.prior["trust"])


@dataclass
class UpdateWork:
    map: str
    frames: list[FrameWork]
    accept_if: dict[str, float] | None = None
    # the cameras the request adds to the map and their ignore masks (``camera_id -> (digest,
    # PNG bytes)``): recorded with the version the job commits, never before
    cameras: dict[str, dict[str, Any]] = field(default_factory=dict)
    masks: dict[str, tuple[str, bytes]] = field(default_factory=dict)


@dataclass
class _Frame:
    """A frame's state through the job."""

    work: FrameWork
    cam: Camera
    sha: str
    measures: checks.Measures | None = None
    depth_m: NDArray[np.float32] | None = None  # on the canonical pinhole
    path: Path | None = None  # the image the mapper gets
    mask_path: Path | None = None  # its ignore mask (PNG, the mapper's pixels), when it has one
    rgb: NDArray[np.uint8] | None = None  # the image the mapper gets (canonical pinhole)
    ignore: NDArray[np.bool_] | None = None  # its ignore mask on the canonical pinhole
    deferred: bool = False  # suspicious, its median depth not measured yet (``_check``)
    reason: str | None = None
    scene_m: float | None = None  # its median scene depth (the distance limit), when measured
    unconfirmed: str | None = None  # the check a frame with a prior failed: kept at its prior
    duplicate_of: str | None = None  # with reason duplicate: the frame it duplicates
    detail: str | None = None  # kept with no_overlap: why its images did not confirm it
    height_off: bool = False  # registered, its camera height disagrees with the one sent
    internal: InternalFrame | None = None
    pose: NDArray[np.float64] | None = None  # T_map_cam
    sigma_pos_m: float | None = None
    sigma_rot_deg: float | None = None
    prior_residual: tuple[float, float] | None = None


@dataclass
class _Anchored:
    sim: Sim3
    scale_source: str
    scale_sigma_rel: float
    align_rms_m: float
    prior_residuals: list[tuple[float, float]] = field(default_factory=list)
    previous_residuals: list[tuple[float, float]] = field(default_factory=list)
    gauge_shift: dict[str, float] | None = None
    prior_gravity_deg: float | None = None
    scale_from_priors: bool = False
    scale_ratio_vs_prev: float | None = None
    # the similarity from the map frame to the update's drifting priors' frame, fitted on its
    # registered drifting frames (``_drifting_residuals``); None without any
    drift_sim: Sim3 | None = None
    # that estimate's own sigma at its supporting frames (position m, rotation deg), and those
    # frames' prior centres (the priors' frame): kept drifting frames grow by it, more away
    # from them (``drift_grown``)
    drift_sigma: tuple[float, float] | None = None
    drift_support: list[list[float]] | None = None
    prior_disagreement_deg: float = 0.0
    # a map without priors whose first registered frame has a camera height: that height (the
    # origin is the floor below that frame, map.md §1); carried by every later version
    origin_height_m: float | None = None


def _mat(T: Any) -> NDArray[np.float64]:
    return np.asarray(T, np.float64)


def _prior_sigmas(p: dict[str, Any]) -> tuple[float, float]:
    return anchor.prior_sigmas(p)


def pose_sigmas(f: InternalFrame, scene_depth_m: float | None, align_rms_m: float,
                depth: bool = False, trust: str | None = None, calibrate: bool = True
                ) -> tuple[float, float]:
    """The pose sigma proxy (README "Per-input status"): the registration's angular noise per
    observation (its mean reprojection error over the focal length, or the multi-view
    refinement's median match residual) over the square root of its observations, times the
    scene depth for the position, combined with the anchoring's RMS residual; three times that
    for a low-confidence keyframe; reported as 3-D RMS (sqrt(3) times the per-axis proxy), then
    calibrated (``mapping.sigmas.calibrated``: the image-registration noise of the data kind, a
    depth image or not and the prior's trust), floored. Without ``calibrate``, the per-axis
    proxy alone: the weight of a stored pose as an anchor of the next version's fit
    (``_anchor``), which the noise floors would leave without scale. The weights stay per axis
    as the metric priors' sigmas they are weighed against (the 3-D RMS convention of api.md §6
    is the reported sigmas'; scaling the anchors by sqrt(3) weighs the priors more, a likely
    cause of the round-5 replay's scale.error_metric rise, unverified on the replay)."""
    st = f.stats
    if "pose_matches" in st:
        n = float(st.get("pose_matches") or 0)
        ang = math.radians(float(st.get("pose_residual_deg") or 0.5))
    else:
        n = float(st.get("observations") or 0)
        ang = float(st.get("reproj_error") or 1.0) / max(f.fx, 1.0)
    n = max(n, 1.0)
    d = scene_depth_m or DEFAULT_SCENE_DEPTH_M
    # reported: per axis times sqrt(3), 3-D RMS (api.md §6 "Sigma convention", every engine)
    rms = math.sqrt(3.0) if calibrate else 1.0
    rot = rms * ang * math.sqrt(2.0 / n)
    pos = rms * d * ang * math.sqrt(8.0 / n)
    k = LOW_CONFIDENCE_FACTOR if f.low_confidence else 1.0
    pos, rot = math.hypot(k * pos, align_rms_m), math.degrees(k * rot)
    if calibrate:
        pos, rot = sigmas.calibrated(pos, rot, depth, trust)
    return round(max(pos, SIGMA_POS_FLOOR_M), 4), round(max(rot, SIGMA_ROT_FLOOR_DEG), 3)


def map_floor(internal: InternalMap) -> float | None:
    """The internal map's floor height (a robot looking ahead may never see a floor surface the
    mapper detects): the mapper's, else the lowest dense layer of its cloud below every camera
    (``mapping.anchored.lowest_layer``)."""
    if internal.floor_z is not None:
        return internal.floor_z
    if not internal.frames:
        return None
    if internal.points_z is None:
        return None
    return lowest_layer(internal.points_z, min(float(f.T[2, 3]) for f in internal.frames))


def _height(fr: _Frame) -> float | None:
    """A frame's camera height above the floor: its own, else its camera's; None without."""
    h = fr.work.height_m if fr.work.height_m is not None else fr.cam.height_above_floor_m
    return float(h) if h is not None and h > 0 else None


def _has_depth(fr: _Frame) -> bool:
    return fr.depth_m is not None and bool(np.count_nonzero(fr.depth_m > 0))


def drift_grown(sigma_pos: float, sigma_rot: float, P: NDArray[np.float64],
                sigma: tuple[float, float] | None, support: list[list[float]] | None
                ) -> tuple[float, float]:
    """A drifting prior ``P``'s drift-grown sigmas on the older mapper (map.md §1, P16; every
    engine the same rule, ``anchored.drift_grown``): its fit's sigma (``sigma``; a record
    before the drift sigma: none) and the distance from ``P``'s prior position to the nearest
    supporting frame's (``support``, their prior centres; none: no growth)."""
    return anchored_drift_grown(sigma_pos, sigma_rot, P, sigma or (0.0, 0.0), support or [])


def _too_close_content(fr: _Frame) -> bool:
    """The too-close exception on the older mapper (api.md §4, every engine): a frame kept at
    its prior only because it is too close, with a metric prior and a depth image, gives its
    in-range depth at its prior pose (``KEPT_DEPTH``)."""
    return fr.unconfirmed == "too_close" and fr.work.trust == "metric" and _has_depth(fr)


def kept_depth(frames: list[_Frame]) -> dict[str, Any]:
    """The depth images of the version's too-close kept frames (``_too_close_content``), by
    name: 16-bit PNG millimetres, base64 (``KEPT_DEPTH``, read by ``derived``)."""
    import base64

    import cv2

    out = {}
    for fr in frames:
        if _too_close_content(fr):
            assert fr.depth_m is not None
            mm = np.clip(np.round(fr.depth_m * 1000.0), 0, 65535).astype(np.uint16)
            out[fr.work.name] = base64.b64encode(cv2.imencode(".png", mm)[1].tobytes()).decode()
    return out


def _inliers(f: InternalFrame) -> int:
    return int(f.stats.get("observations") or f.stats.get("pose_matches") or 0)


KEPT_DEPTH = "kept_depth.json"  # the version file of its too-close kept frames' depth


class Updater:
    """Runs update jobs (``jobs.Runner``)."""

    def __init__(self, store: Store, backend: Backend, timings: Timings | None = None,
                 prior_first: bool = True) -> None:
        self.store = store
        self.backend = backend
        self.timings = timings or Timings()
        # maps started with priors take the prior-first registration (``prior_first``); False
        # keeps every map on the mapper (the earlier path)
        self.prior_first = prior_first

    def __call__(self, job: Job, changed: Callable[[], None]) -> None:
        w: UpdateWork = job.work
        t0 = time.perf_counter()
        job.state = "registering"
        job.base_version = self.store.latest(w.map)
        changed()
        workdir = self.store.work / job.id
        try:
            self.backend.require_inference()
            self._run(job, w, workdir, changed)
        except ApiError as exc:
            job.state, job.error = "failed", exc.body()
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
            tm = job.timings_s
            tm["total"] = round(time.perf_counter() - t0, 3)
        if job.state != "failed" and w.frames:
            # the phases a prediction adds up (``Timings.predict_phases``)
            self.timings.add("checks_per_frame", tm.get("checks", 0.0) / len(w.frames),
                             len(w.frames))
            self.timings.add("job_overhead", max(tm["total"] - tm.get("checks", 0.0)
                                                 - tm.get("register", 0.0)
                                                 - tm.get("fusion", 0.0)
                                                 - tm.get("commit", 0.0), 0.0))
            if "commit" in tm and job.result_version is not None:
                # the commit and the version's derivation grow with the map
                n = len(self.store.version(w.map, job.result_version)["frames"])
                self.timings.add("commit_per_keyframe", tm["commit"] / max(n, 1))

    # -- the job -------------------------------------------------------------------------------

    def _frames(self, w: UpdateWork) -> tuple[list[_Frame], dict[str, NDArray[np.bool_] | None]]:
        """The request's frames and the cameras' ignore masks: the map's committed cameras and
        the request's own (``w.cameras``, ``w.masks``). A request camera that a version committed
        since the request was accepted defines otherwise is ``camera_conflict``; a frame whose
        camera came from an earlier job of the map that did not commit is ``invalid_request``."""
        from oh_my_slam.service.cameras import same_camera

        known = self.store.cameras(w.map)
        for cid, cam in w.cameras.items():
            digest = w.masks[cid][0] if cid in w.masks else None
            if cid in known and (not same_camera(known[cid], cam) or (
                    digest is not None and digest != self.store.mask_digest(w.map, cid))):
                raise ApiError("camera_conflict", f"camera {cid!r} of map {w.map!r} was defined "
                               "with other values by an update committed meanwhile: omit it, or "
                               "use a new camera id",
                               {"field": f"cameras.{cid}", "camera_id": cid,
                                "known": known[cid]})
        cams = {**known, **w.cameras}
        for i, f in enumerate(w.frames):
            if f.camera_id not in cams:
                raise ApiError("invalid_request", f"frames[{i}].camera_id {f.camera_id!r}: the "
                               "earlier update that defined this camera was not committed — "
                               "send the camera with the frames",
                               {"field": f"frames[{i}].camera_id"})
        masks: dict[str, NDArray[np.bool_] | None] = {}
        for cid in {f.camera_id for f in w.frames}:
            if cid in w.masks:
                masks[cid] = decode_mask(w.masks[cid][1], f"cameras.{cid}.ignore_mask")
                continue
            p = self.store.mask_path(w.map, cid)
            masks[cid] = decode_mask(p.read_bytes(), f"cameras.{cid}.ignore_mask") \
                if self.store.mask_digest(w.map, cid) else None
        frames = [_Frame(f, Camera.from_json(cams[f.camera_id]),
                         hashlib.sha256(f.image).hexdigest()) for f in w.frames]
        return frames, masks

    def _check(self, frames: list[_Frame], masks: dict[str, NDArray[np.bool_] | None],
               prev: dict[str, Any] | None, indir: Path) -> None:
        """The per-frame checks: duplicates by name in request order (api.md §4: the frame holding
        that name; the pose duplicate is the registration's), then the measurements of the
        others, ``CHECK_THREADS`` frames at a time."""
        names = {r["name"] for r in prev["frames"]} if prev else set()
        todo = []
        for fr in frames:
            if fr.work.reject is not None:
                fr.reason = fr.work.reject
                fr.duplicate_of = fr.work.name if fr.reason == "duplicate" else None
                continue
            if fr.work.name in names:
                fr.reason, fr.duplicate_of = "duplicate", fr.work.name
                continue
            names.add(fr.work.name)
            todo.append(fr)
        if len(todo) < 2:
            for fr in todo:
                self._check_one(fr, masks, indir)
            return
        with ThreadPoolExecutor(CHECK_THREADS) as pool:
            list(pool.map(lambda fr: self._check_one(fr, masks, indir), todo))

    @staticmethod
    def _check_one(fr: _Frame, masks: dict[str, NDArray[np.bool_] | None], indir: Path) -> None:
        f, mask = fr.work, masks[fr.work.camera_id]
        if f.mask is not None:
            own = decode_mask(f.mask, f"frames[{f.index}].ignore_mask")
            mask = own if mask is None else (mask | own)
        rgb = decode_rgb(f.image, f"frames[{f.index}].image")
        fr.measures = checks.measure(rgb, mask)
        med = None
        if f.depth is not None:
            fr.depth_m = mapper_depth(decode_png16(f.depth, f"frames[{f.index}].depth"),
                                      fr.cam, mask)
            med = checks.median_depth(fr.depth_m)
        fr.path = indir / f"{f.index:04d}.jpg"
        if fr.cam.is_canonical and (mask is None or not mask.any()) \
                and f.image[:3] == b"\xff\xd8\xff":
            fr.path.write_bytes(f.image)
            fr.rgb = rgb
        else:
            fr.rgb = mapper_image(rgb, fr.cam, mask)
            save_jpeg(fr.rgb, fr.path, quality=JPEG_QUALITY)
        if mask is not None and mask.any():
            fr.ignore = fr.cam.rectify(mask.astype(np.uint8), nearest=True) > 0
            fr.mask_path = indir / f"{f.index:04d}.mask.png"
            fr.mask_path.write_bytes(png_bytes(fr.ignore.astype(np.uint8) * 255))
        # the monocular depth the distance limit needs is measured once the path is known:
        # the prior-first engine gets it with the frame's depth, the mapper's path and the dry
        # run ask it (``_resolve_deferred``)
        fr.deferred = med is None and fr.measures.suspicious
        fr.scene_m = med
        Updater._verdict(fr)

    @staticmethod
    def _verdict(fr: _Frame) -> None:
        """A measured frame's verdict before registration: a frame that fails a quality check
        is kept at its prior when it has one, its reason the first failing check (api.md §4:
        before ``too_close``); without one it is rejected with it, or ``too_close`` when it is
        known to be too close (it cannot reach ``TOO_CLOSE_INLIERS``)."""
        assert fr.measures is not None
        r = checks.reason(fr.measures, None)
        fr.reason = fr.unconfirmed = None
        if r is not None and fr.work.trust in ("metric", "drifting"):
            fr.unconfirmed = r
        elif r is not None and fr.scene_m is not None and fr.scene_m < checks.TOO_CLOSE_M:
            fr.reason = "too_close"
        else:
            fr.reason = r

    def _resolve_deferred(self, frames: list[_Frame]) -> None:
        """The median monocular depth of the suspicious frames, ``CHECK_THREADS`` asked at once
        (``too_close`` takes precedence over the other reasons): the mapper's path and the dry
        run."""
        todo = [fr for fr in frames if fr.deferred and fr.reason in (None, *checks.QUALITY)]

        def one(fr: _Frame) -> None:
            assert fr.path is not None
            fr.scene_m = self.backend.median_depth(fr.path, fr.cam.canonical())
            Updater._verdict(fr)

        with ThreadPoolExecutor(CHECK_THREADS) as pool:
            list(pool.map(one, todo))

    def _run(self, job: Job, w: UpdateWork, workdir: Path, changed: Callable[[], None]) -> None:
        tm = job.timings_s
        t = time.perf_counter()
        prev = self.store.version(w.map) if job.base_version else None
        indir = workdir / "in"
        indir.mkdir(parents=True, exist_ok=True)
        frames, masks = self._frames(w)
        self._check(frames, masks, prev, indir)
        tm["checks"] = round(time.perf_counter() - t, 3)
        job.progress = 0.1
        changed()
        if self._is_anchored(prev, frames):
            from oh_my_slam.service.prior_first import run_anchored

            run_anchored(self, job, w, frames, prev, workdir, changed)
            return
        t = time.perf_counter()
        self._resolve_deferred(frames)
        tm["checks"] = round(tm["checks"] + time.perf_counter() - t, 3)
        # frames rejected, or kept at their prior, by the checks never reach the mapper
        pre = {fr.work.index: fr.reason or fr.unconfirmed for fr in frames}
        exclude: set[int] = set()
        result: tuple[Path, InternalMap, _Anchored, int] | None = None
        attempt = 0
        while True:
            todo = [fr for fr in frames if fr.path is not None and pre[fr.work.index] is None
                    and fr.work.index not in exclude]
            if not todo:
                break
            for fr in todo:  # a second attempt starts again from the checks' verdicts
                fr.reason, fr.internal, fr.pose, fr.prior_residual = None, None, None, None
                fr.unconfirmed = None
            t = time.perf_counter()
            mapper = workdir / f"mapper{attempt}"
            if self.store.mapper_dir(w.map).exists():
                clone_tree(self.store.mapper_dir(w.map), mapper)
            paths = [fr.path for fr in todo if fr.path is not None]
            intrinsics = {fr.path: fr.cam.canonical() for fr in todo if fr.path is not None}
            try:
                reg = self.backend.register(
                    mapper, paths, intrinsics, workdir / f"mapper{attempt}.log",
                    {fr.path: fr.mask_path for fr in todo if fr.path and fr.mask_path})
            except NothingRegisteredError:
                for fr in todo:
                    self._unregistered(fr)
                result = None
                tm["register"] = round(tm.get("register", 0.0) + time.perf_counter() - t, 3)
                break
            internal = self.backend.read(mapper)
            wall = time.perf_counter() - t
            tm["register"] = round(tm.get("register", 0.0) + wall, 3)
            fusion = sum(v for k, v in reg.stages.items() if k in FUSION_STAGES)
            self.timings.add("registration_per_frame", max(wall - fusion, 0.0) / len(todo),
                             len(todo))
            self.timings.add("fusion_per_frame", fusion / len(todo), len(todo))
            job.progress = 0.8
            changed()
            t = time.perf_counter()
            by_source = {f.source: f for f in internal.frames}
            for fr in todo:
                fr.internal = by_source.get(str(fr.path))
                if fr.internal is None:
                    self._unregistered(fr)
            placed = [fr for fr in todo if fr.internal is not None]
            if not placed:
                result = None
                break
            anchored = self._anchor(placed, prev, internal)
            post = self._post_check(placed, anchored, internal, prev)
            tm["anchor"] = round(tm.get("anchor", 0.0) + time.perf_counter() - t, 3)
            result = (mapper, internal, anchored, reg.objects)
            if not post or attempt:
                break
            exclude |= {fr.work.index for fr in post}
            attempt += 1
        self._finish(job, w, frames, prev, result)

    @staticmethod
    def _unregistered(fr: _Frame) -> None:
        """A frame the mapper did not register: kept at its prior when it has one (its images
        merely did not confirm it; ``too_close`` when its scene is within the distance limit),
        else rejected ``too_close`` or ``no_overlap`` (in updates ``few_features`` is only the
        checks' verdict, api.md §4)."""
        why = "no_overlap"
        if fr.scene_m is not None and fr.scene_m < checks.TOO_CLOSE_M:
            why = "too_close"
        if fr.work.trust in ("metric", "drifting"):
            fr.unconfirmed = why
            # the mapper gave it no pose (api.md §4 ``detail``)
            fr.detail = "no_image_pose" if why == "no_overlap" else None
        else:
            fr.reason = why

    def dry_run(self, w: UpdateWork) -> dict[str, Any]:
        """The ``DryRun`` of an update request (``service.dry_run``): nothing is written."""
        from oh_my_slam.service.dry_run import dry_run

        return dry_run(self, w)

    def _is_anchored(self, prev: dict[str, Any] | None, frames: list[_Frame]) -> bool:
        """Whether the job runs on the prior-first registration: the map has been anchored by
        it, or it is new (with or without priors) and ``prior_first`` holds."""
        if prev is not None:
            return prev.get("engine") == "anchored"
        return self.prior_first

    # -- anchoring -------------------------------------------------------------------------------

    def _fallback_scale(self, placed: list[_Frame], internal: InternalMap
                        ) -> tuple[float, str, float]:
        """Scale, source and relative sigma without usable metric priors: depth PNGs, else
        camera heights, else the monocular depth the mapper aligned to."""
        ratios = []
        for fr in placed:
            assert fr.internal is not None
            if fr.depth_m is None:
                continue
            d_int = internal.depth(fr.internal.name)
            if d_int is None:
                continue
            import cv2

            user = cv2.resize(fr.depth_m, (d_int.shape[1], d_int.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
            ok = (user > 0) & (d_int > 0) & np.isfinite(user) & np.isfinite(d_int)
            if ok.sum() >= 100:
                ratios.append(float(np.median(user[ok] / d_int[ok])))
        if ratios:
            r = np.array(ratios)
            mad = 1.4826 * float(np.median(np.abs(np.log(r) - np.median(np.log(r)))))
            return (float(np.median(r)), "depth",
                    max(mad / math.sqrt(len(r)), MIN_SCALE_SIGMA_REL))
        heights = []
        floor_z = map_floor(internal)
        if floor_z is not None:
            for fr in placed:
                assert fr.internal is not None
                h = fr.work.height_m if fr.work.height_m is not None else \
                    fr.cam.height_above_floor_m
                if h is None:
                    continue
                sig = fr.work.height_sigma_m if fr.work.height_m is not None else \
                    fr.cam.height_sigma_m
                h_int = float(fr.internal.T[2, 3]) - floor_z
                if h_int > MIN_CAMERA_HEIGHT_M and h > 0:
                    heights.append((h / h_int, (sig or DEFAULT_HEIGHT_SIGMA_M) / h))
        if heights:
            r = np.array([h[0] for h in heights])
            rel = np.array([h[1] for h in heights])
            mad = 1.4826 * float(np.median(np.abs(np.log(r) - np.median(np.log(r)))))
            sigma = math.sqrt(mad ** 2 + float(np.median(rel)) ** 2) / math.sqrt(len(r))
            return float(np.median(r)), "camera_height", max(sigma, MIN_SCALE_SIGMA_REL)
        return 1.0, "monocular", MONOCULAR_SIGMA_REL

    def _anchor(self, placed: list[_Frame], prev: dict[str, Any] | None, internal: InternalMap
                ) -> _Anchored:
        by_name = {f.name: f for f in internal.frames}
        metric: list[Anchor] = []
        previous: list[Anchor] = []
        if prev:
            for r in prev["frames"]:
                f = by_name.get(r["internal"])
                if f is None:
                    continue
                # its uncalibrated sigmas when recorded (``pose_sigmas``), else the reported
                sp, sr = r.get("anchor_sigmas") or (r["sigma_pos_m"], r["sigma_rot_deg"])
                previous.append(Anchor(r["name"], "previous", f.T, _mat(r["T_map_cam"]),
                                       float(sp), float(sr)))
                p = r.get("prior")
                if p and p["trust"] == "metric":
                    metric.append(Anchor(r["name"], "prior", f.T, _mat(p["T_map_cam"]),
                                         *_prior_sigmas(p)))
        new_metric = {}
        drifting: list[Anchor] = []
        for fr in placed:
            assert fr.internal is not None
            p = fr.work.prior
            if p is None or p["trust"] == "none":
                continue
            a = Anchor(fr.work.name, "prior" if p["trust"] == "metric" else "drifting",
                       fr.internal.T, _mat(p["T_map_cam"]), *_prior_sigmas(p))
            if p["trust"] == "metric":
                metric.append(a)
                new_metric[fr.work.name] = a
            else:
                drifting.append(a)
        out = self._fit(prev, placed, internal, metric, previous, drifting)
        # per-frame poses and prior residuals
        for fr in placed:
            assert fr.internal is not None
            fr.pose = out.sim.pose(fr.internal.T)
            if fr.work.name in new_metric:
                fr.prior_residual = anchor.residual(out.sim, new_metric[fr.work.name])
        out.drift_sim, out.drift_sigma, out.drift_support = self._drifting_residuals(
            placed, out.sim, prev)
        out.prior_disagreement_deg = self._prior_disagreement(placed, internal, out.sim)
        out.prior_residuals = [anchor.residual(out.sim, a) for a in metric]
        out.previous_residuals = [anchor.residual(out.sim, a) for a in previous]
        return out

    def _fit(self, prev: dict[str, Any] | None, placed: list[_Frame], internal: InternalMap,
             metric: list[Anchor], previous: list[Anchor], drifting: list[Anchor]) -> _Anchored:
        if prev is None:
            group = metric or drifting
            if group:
                fit = anchor.fit(group)
                source, sigma = "priors", fit.scale_sigma_rel
                if not fit.scale_observable:
                    s, source, sigma = self._fallback_scale(placed, internal)
                    fit = anchor.fit(group, scale=s)
                rms = self._rms(fit.sim, [a for a, u in zip(group, fit.inliers, strict=True) if u])
                up = fit.sim.R @ np.array([0.0, 0.0, 1.0])
                gravity = math.degrees(math.acos(max(-1.0, min(1.0, float(up[2])))))
                assert sigma is not None
                return _Anchored(fit.sim, source, sigma, rms, prior_gravity_deg=round(gravity, 3),
                                 scale_from_priors=source == "priors")
            s, source, sigma = self._fallback_scale(placed, internal)
            first = min(placed, key=lambda fr: fr.work.index)
            assert first.internal is not None
            out = _Anchored(anchor.yaw_frame(first.internal.T, s, _height(first)), source, sigma,
                            0.0)
            out.origin_height_m = _height(first)
            return out
        prev_sim = Sim3.from_dict(prev["sim3"])
        group = previous + metric
        if not group:  # the mapper kept none of the stored frames: its frame is unchanged
            return _Anchored(prev_sim, prev["scale_source"], prev["scale_sigma_rel"], 0.0,
                             gauge_shift={"translation_m": 0.0, "rotation_deg": 0.0,
                                          "scale_ratio": 1.0}, scale_ratio_vs_prev=1.0,
                             origin_height_m=prev.get("origin_height_m"))
        fit = anchor.fit(group)
        if not fit.scale_observable:
            fit = anchor.fit(group, scale=prev_sim.s)
        used = [a for a, u in zip(group, fit.inliers, strict=True) if u]
        source, sigma = prev["scale_source"], float(prev["scale_sigma_rel"])
        metric_in = [a for a in used if a.kind == "prior"]
        if metric_in and fit.scale_observable and len(metric_in) >= 2:
            source = "priors"
            sigma = min(sigma, fit.scale_sigma_rel or sigma) if prev["scale_source"] == "priors" \
                else (fit.scale_sigma_rel or sigma)
        sim = fit.sim
        if prev["scale_source"] == "monocular" and source == "monocular":
            # a map anchored on monocular depth takes the first metric anchor that comes (depth
            # images, camera heights): every stored position scales about the origin, which
            # scale_ratio_vs_prev and the previous residuals report
            s_new, src, sig = self._fallback_scale(placed, internal)
            if src != "monocular":
                sim = Sim3(s_new / sim.s).compose(sim)
                source, sigma = src, sig
        fit.sim = sim
        A = fit.sim.compose(prev_sim.inverse())
        shifts = [float(np.linalg.norm(A.point(x) - x))
                  for x in (prev_sim.point(a.src[:3, 3]) for a in previous)]
        gauge = {"translation_m": round(float(np.median(shifts)) if shifts else 0.0, 4),
                 "rotation_deg": round(anchor.rotation_deg(A.R), 3),
                 "scale_ratio": round(A.s, 5)}
        out = _Anchored(fit.sim, source, sigma, self._rms(fit.sim, used), gauge_shift=gauge,
                        scale_from_priors=source == "priors",
                        origin_height_m=prev.get("origin_height_m"))
        out.scale_ratio_vs_prev = self._spread_ratio(fit.sim, previous)
        return out

    @staticmethod
    def _spread_ratio(sim: Sim3, previous: list[Anchor]) -> float:
        """The map's scale change on the stored frames: the spread of their new map-frame
        centres over the spread of their previous ones (1 with fewer than two)."""
        if len(previous) < 2:
            return 1.0
        new = np.array([sim.point(a.src[:3, 3]) for a in previous])
        old = np.array([a.dst[:3, 3] for a in previous])
        s_old = float(np.sqrt(((old - old.mean(0)) ** 2).sum(1).mean()))
        s_new = float(np.sqrt(((new - new.mean(0)) ** 2).sum(1).mean()))
        return round(s_new / s_old, 5) if s_old > 1e-9 else 1.0

    @staticmethod
    def _rms(sim: Sim3, used: list[Anchor]) -> float:
        return float(np.sqrt(np.mean([anchor.residual(sim, a)[0] ** 2 for a in used])))

    @staticmethod
    def _prior_disagreement(placed: list[_Frame], internal: InternalMap, sim: Sim3) -> float:
        """``prior_disagreement_deg`` (map.md §1): over the pairs of registered frames with a
        metric or drifting prior that overlap by at least ``OVERLAP_MIN`` (the overlap of map.md
        §1, no named reference: the smaller of the two ``coverage``s of their request images),
        the largest angle between the images' relative rotation and the priors'. The mapper's
        rotations are the images' own: the correction is inherent."""
        views = []
        for fr in placed:
            assert fr.internal is not None and fr.pose is not None
            if fr.work.trust not in ("metric", "drifting") or fr.rgb is None:
                continue
            assert fr.work.prior is not None
            views.append((fr.pose, fr.cam.canonical(), features(fr.rgb, fr.ignore),
                          _mat(fr.work.prior["T_map_cam"])))
        worst = 0.0
        for i, (Ta, Ka, Fa, Pa) in enumerate(views):
            for Tb, Kb, Fb, Pb in views[i + 1:]:
                if min(coverage(Fa, Ka, Fb, Kb)) < OVERLAP_MIN:
                    continue
                img = Ta[:3, :3].T @ Tb[:3, :3]
                worst = max(worst, anchor.rotation_deg(img.T @ (Pa[:3, :3].T @ Pb[:3, :3])))
        return round(worst, 3)

    @staticmethod
    def _drifting_residuals(placed: list[_Frame], sim: Sim3, prev: dict[str, Any] | None = None
                            ) -> tuple[Sim3 | None, tuple[float, float] | None,
                                       list[list[float]] | None]:
        """The robust drift estimate across versions (map.md §1 "Drift estimate", every engine):
        the similarity that best brings the request's aligned poses onto their drifting priors
        (``anchor.fit``, robust: its inliers), against the previous version's (``drift_sim`` of
        its record; none yet: the identity). A first estimate, or a change larger than
        ``K_SIGMA`` × the fit's own sigma (the RMS of its inliers' prior sigmas over the root of
        their count), needs min(``DRIFT_MIN_SUPPORT``, n) supporting inliers, n the frames that
        can support it (the mapper's ``MEASURABLE_INLIERS`` inliers: the measurable-pair
        threshold), ``DRIFT_SUPPORT_SHARE`` of the n and never one alone; without it a change
        keeps the previous estimate, and a refused first estimate leaves the drift at the
        identity (None: priors as sent, with their sigmas). Each drifting frame's residual is
        against the estimate kept, so ``_post_check`` rejects those beyond the bound
        (``prior_conflict``) and the mapper runs again without them: every frame re-tested
        under it. Returns the estimate (map frame to the priors' frame), its sigma and its
        supporting frames' prior centres; the previous version's without drifting frames."""
        before, before_sigma, before_support = Updater._previous_drift(prev)
        drift = [fr for fr in placed if fr.work.trust == "drifting"]
        if not drift:
            return before, before_sigma, before_support
        anchors = []
        for fr in drift:
            assert fr.internal is not None and fr.work.prior is not None
            anchors.append(Anchor(fr.work.name, "drifting", sim.pose(fr.internal.T),
                                  _mat(fr.work.prior["T_map_cam"]),
                                  *_prior_sigmas(fr.work.prior)))
        fit = anchor.fit(anchors)
        inl = [i for i, u in enumerate(fit.inliers) if u]
        sig = (math.sqrt(float(np.mean([anchors[i].sigma_pos_m ** 2 for i in inl])) / len(inl)),
               math.sqrt(float(np.mean([anchors[i].sigma_rot_deg ** 2 for i in inl])) / len(inl)))
        used = before or Sim3()
        moved = [anchor.residual(used, Anchor(a.name, "drifting", a.src, fit.sim.pose(a.src)))
                 for a in (anchors[i] for i in inl)]
        large = float(np.median([m for m, _ in moved])) > anchor.K_SIGMA * sig[0] \
            or float(np.median([d for _, d in moved])) > anchor.K_SIGMA * sig[1]
        able = {i for i, fr in enumerate(drift)
                if fr.internal is not None and _inliers(fr.internal) >= MEASURABLE_INLIERS}
        n, got = len(able), len(able & set(inl))
        supported = got >= max(min(DRIFT_MIN_SUPPORT, n), DRIFT_NEVER_ALONE) \
            and got >= DRIFT_SUPPORT_SHARE * n
        out: Sim3 | None = fit.sim
        out_sigma: tuple[float, float] | None = sig
        support: list[list[float]] | None = [
            [round(float(v), 4) for v in anchors[i].dst[:3, 3]] for i in inl]
        if (large or before is None) and not supported:
            out, out_sigma, support = before, before_sigma, before_support
        for fr, a in zip(drift, anchors, strict=True):
            fr.prior_residual = anchor.residual(out or Sim3(), a)
        return out, out_sigma, support

    def _post_check(self, placed: list[_Frame], anchored: _Anchored, internal: InternalMap,
                    prev: dict[str, Any] | None = None) -> list[_Frame]:
        """The registered frames to leave out: ``too_close`` (kept at its prior when it has one:
        a failed check wins over ``prior_conflict``, api.md §4), ``prior_conflict`` (kept at its
        prior, ``conflict_unsupported``, below the measurable-pair threshold) and pose
        ``duplicate`` (``_duplicate_of``)."""
        out = []
        earlier: list[_Frame] = []
        # in canonical name order: a duplicate's earlier frame (api.md §4, map.md §2)
        for fr in sorted(placed, key=lambda fr: (natural_key(fr.work.name), fr.work.camera_id,
                                                 fr.work.index)):
            assert fr.internal is not None
            d = internal.depth(fr.internal.name)
            med = None if d is None else checks.median_depth(d)
            if med is not None:
                fr.scene_m = med * anchored.sim.s
            if fr.scene_m is not None and fr.scene_m < checks.TOO_CLOSE_M \
                    and _inliers(fr.internal) < TOO_CLOSE_INLIERS:
                if fr.work.trust in ("metric", "drifting"):  # kept at its prior (api.md §4)
                    fr.unconfirmed, fr.internal = "too_close", None
                else:
                    fr.reason = "too_close"
                out.append(fr)
                continue
            if fr.prior_residual is not None and fr.work.prior is not None:
                # beyond the combined bound 3 x sqrt(sigma_prior^2 + sigma_image^2) (api.md §4:
                # the prior's sigmas floored as the anchors' noise; the image pose's calibrated
                # proxy before the anchoring), and the mapper's one hypothesis is the only one:
                # none lies near the prior
                sp, sr = _prior_sigmas(fr.work.prior)
                if fr.work.trust == "drifting" and anchored.drift_sim is not None:
                    # a drifting prior's sigma is the drift-grown one (api.md §4)
                    sp, sr = drift_grown(sp, sr, _mat(fr.work.prior["T_map_cam"]),
                                         anchored.drift_sigma, anchored.drift_support)
                hyp = pose_sigmas(fr.internal, None if med is None else med * anchored.sim.s,
                                  0.0, _has_depth(fr), fr.work.trust)
                dp, dr = fr.prior_residual
                if sigmas.conflicts(dp, dr, (max(sp, anchor.MIN_SIGMA_POS_M),
                                             max(sr, anchor.MIN_SIGMA_ROT_DEG)), hyp):
                    # the images confirm the conflict only when its pose meets the
                    # measurable-pair threshold (api.md §4, P16; on this engine its mapper
                    # inliers, as its drift support); else it is kept at its prior,
                    # ``conflict_unsupported``, and the mapper runs again without it
                    if _inliers(fr.internal) >= MEASURABLE_INLIERS:
                        fr.reason = "prior_conflict"
                    else:
                        fr.unconfirmed, fr.detail = "no_overlap", "conflict_unsupported"
                        fr.internal = None
                    out.append(fr)
                    continue
            dup = self._duplicate_of(fr, prev, earlier, internal)
            if dup is not None:
                fr.reason, fr.duplicate_of = "duplicate", dup
                out.append(fr)
                continue
            earlier.append(fr)
        return out

    @staticmethod
    def _duplicate_of(fr: _Frame, prev: dict[str, Any] | None, earlier: list[_Frame],
                      internal: InternalMap | None = None) -> str | None:
        """The frame a registered frame duplicates (api.md §4, ``anchored.duplicate_among``): a
        stored fused frame of the previous version, or an earlier registered frame of the
        request, of its camera, within ``duplicate_limits`` of its pose, whose image its own
        agrees with (their images: the mapper map's, the request's); None."""
        assert fr.pose is not None

        def of_rgb(rgb: Any) -> Features | None:
            return features(rgb, None)  # a registered frame of the request has its image

        def of_image(name: str | None) -> Features | None:
            p = None if internal is None or not name else internal.image(name)
            if p is None or not p.exists():
                return None
            import cv2

            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            return None if bgr is None else features(
                np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), np.uint8))

        cands: list[tuple[str, str, Any, Callable[[], Features | None]]] = [
            (r["name"], r.get("camera_id", ""), _mat(r["T_map_cam"]),
             functools.partial(of_image, r.get("internal")))
            for r in (prev["frames"] if prev else [])
            if r.get("pose_state", "fused") != "kept_at_prior"]
        cands += [(e.work.name, e.work.camera_id, e.pose, functools.partial(of_rgb, e.rgb))
                  for e in earlier if e.pose is not None]
        assert fr.rgb is not None
        return duplicate_among(fr.work.camera_id, fr.pose, of_rgb(fr.rgb), cands)

    # -- result ----------------------------------------------------------------------------------

    def _finish(self, job: Job, w: UpdateWork, frames: list[_Frame], prev: dict[str, Any] | None,
                result: tuple[Path, InternalMap, _Anchored, int] | None) -> None:
        t = time.perf_counter()
        registered = [fr for fr in frames if fr.reason is None and fr.internal is not None]
        record: dict[str, Any] | None = None
        anchored = None if result is None else result[2]
        kept = [fr for fr in frames if fr.reason is None and fr.unconfirmed is not None]
        drift = (anchored.drift_sim, anchored.drift_sigma, anchored.drift_support) \
            if anchored is not None else self._previous_drift(prev)
        for fr in kept:
            self._keep_at_prior(fr, *drift)
        if result is not None:
            _, internal, anchored, objects = result
            record = self._record(job, frames, registered + kept, prev, internal, anchored,
                                  objects)
        elif prev is not None and kept:
            # every frame kept at its prior (map.md §4, every engine): accepted and stored, the
            # mapper's map as it was
            record = self._kept_only(job, kept, prev, drift)
        job.inputs = [self._input(fr) for fr in frames]
        # the mapper's engine ties kept frames to the map frame only through registered ones:
        # a first update without a registered frame with a prior cannot fix the priors' frame
        no_prior = prev is None and any(fr.work.trust in ("metric", "drifting") for fr in frames) \
            and not any(fr.work.trust in ("metric", "drifting") for fr in registered)
        # rotation consistency applies to this engine too (map.md §1 "Every engine"; updates
        # without priors, registered frames only)
        bad, rotation_pairs = self._rotations_bad(registered, record, result[1]) \
            if result is not None and record is not None and registered and not any(
                fr.work.trust in ("metric", "drifting") for fr in frames) else (False, [])
        # a first update stores kept frames only with a registration (the engine exemption of
        # map.md §1): a kept-only first update fails the hard limit
        quality = self._quality(frames, registered, kept if record is not None else [], prev,
                                anchored, w.accept_if, result is not None and result[1].rebuilt,
                                no_prior, bool(bad), rotation_pairs)
        job.quality = quality
        if anchored is not None:
            job.scale_source, job.scale_sigma_rel = anchored.scale_source, round(
                anchored.scale_sigma_rel, 5)
        elif record is not None:
            job.scale_source, job.scale_sigma_rel = record["scale_source"], record[
                "scale_sigma_rel"]
        if not quality["accepted"]:
            job.state = "rejected"
            job.progress = 1.0
            return
        assert record is not None
        depths = kept_depth(kept)
        job.result_version = self.store.commit(w.map, record,
                                               None if result is None else result[0],
                                               {KEPT_DEPTH: depths} if depths else None,
                                               w.cameras, w.masks)
        job.state = "committed"
        job.progress = 1.0
        job.timings_s["commit"] = round(time.perf_counter() - t, 3)

    @staticmethod
    def _rotations_bad(registered: list[_Frame], record: dict[str, Any], internal: InternalMap
                       ) -> tuple[bool, list[dict[str, Any]]]:
        """``rotation_inconsistent`` of the older mapper (``standpoint.views_inconsistent``):
        the registered frames at their committed poses, with their request images, against the
        version's other fused frames at the poses it commits, with their stored images."""
        import cv2

        mine = {fr.work.name for fr in registered}
        fx = {g.name: g.fx for g in internal.frames}
        new = [standpoint.View(fr.work.name, fr.cam.canonical(), features(fr.rgb, fr.ignore),
                               fr.pose[:3, :3].copy(), fr.pose[:3, 3].copy(), False, fr.work.index)
               for fr in registered if fr.pose is not None and fr.rgb is not None]
        stored = []
        for r in record["frames"]:
            if r["name"] in mine or r.get("pose_state") != "fused" or r.get("internal") not in fx:
                continue
            img = internal.image(r["internal"])
            bgr = None if img is None else cv2.imread(str(img))
            if bgr is None:
                continue
            rgb = np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), np.uint8)
            h, w = rgb.shape[:2]
            f, T = fx[r["internal"]], _mat(r["T_map_cam"])
            stored.append(standpoint.View(r["name"], Intrinsics(f, f, w / 2.0, h / 2.0, w, h),
                                          features(rgb), T[:3, :3].copy(), T[:3, 3].copy(), True))
        return standpoint.views_inconsistent(new, stored)

    @staticmethod
    def _height_off(fr: _Frame, floor: float | None) -> bool:
        """The height gate (map.md §2) on a registered frame: with a camera height and its sigma
        sent, its committed camera height over the version's floor (z 0 without one) disagrees
        by more than ``K_SIGMA`` × sqrt(height_sigma² + σ_z²) (σ_z: its position sigma over
        √3)."""
        h = _height(fr)
        hs = fr.work.height_sigma_m if fr.work.height_m is not None else fr.cam.height_sigma_m
        if h is None or hs is None or fr.pose is None or fr.sigma_pos_m is None:
            return False
        dz = abs(float(fr.pose[2, 3]) - (floor or 0.0) - h)
        return dz > anchor.K_SIGMA * math.hypot(hs, fr.sigma_pos_m / math.sqrt(3.0))

    @staticmethod
    def _previous_drift(prev: dict[str, Any] | None
                        ) -> tuple[Sim3 | None, tuple[float, float] | None,
                                   list[list[float]] | None]:
        """The previous version's drift estimate, its sigma and its supporting frames' prior
        centres (``_drifting_residuals``)."""
        if prev is None or not prev.get("drift_sim"):
            return None, None, None
        sig = prev.get("drift_sigma")
        return (Sim3.from_dict(prev["drift_sim"]),
                None if not sig else (float(sig[0]), float(sig[1])),
                prev.get("drift_support"))

    @staticmethod
    def _keep_at_prior(fr: _Frame, drift_sim: Sim3 | None = None,
                       drift_sigma: tuple[float, float] | None = None,
                       drift_support: list[list[float]] | None = None) -> None:
        """A frame kept at its prior (api.md §4): a metric prior as given, a drifting one mapped
        into the map frame by the drift estimate (``drift_sim``, else as given), with the
        prior's sigmas, a drifting prior's drift-grown (``drift_grown``, map.md §1 "Every
        engine"); prior residual 0."""
        p = fr.work.prior
        assert p is not None
        P = _mat(p["T_map_cam"])
        fr.sigma_pos_m, fr.sigma_rot_deg = _prior_sigmas(p)
        if p["trust"] == "drifting" and drift_sim is not None:
            sp, sr = drift_grown(fr.sigma_pos_m, fr.sigma_rot_deg, P, drift_sigma, drift_support)
            P = drift_sim.inverse().pose(P)
            fr.sigma_pos_m, fr.sigma_rot_deg = round(sp, 4), round(sr, 3)
        fr.pose = P
        fr.prior_residual = (0.0, 0.0)

    @staticmethod
    def _kept_row(fr: _Frame, version: int) -> dict[str, Any]:
        """A kept frame's row of the version record (``_keep_at_prior`` placed it)."""
        assert fr.pose is not None
        p = fr.work.prior
        return {"name": fr.work.name, "camera_id": fr.work.camera_id, "internal": None,
                "T_map_cam": fr.pose.tolist(), "sigma_pos_m": fr.sigma_pos_m,
                "sigma_rot_deg": fr.sigma_rot_deg, "pose_state": "kept_at_prior",
                "added_version": version, "sha256": fr.sha, "inliers": 0,
                # the too-close exception (api.md §4): its in-range depth feeds occupancy and
                # heights at its prior (``derived``)
                "content": _too_close_content(fr),
                "prior": p if p is not None and p["trust"] == "metric" else None}

    def _kept_only(self, job: Job, kept: list[_Frame], prev: dict[str, Any],
                   drift: tuple[Sim3 | None, tuple[float, float] | None,
                                list[list[float]] | None]) -> dict[str, Any]:
        """The record of an update whose frames are all kept at their priors (map.md §4): the
        previous version's, its mapper map unchanged, with the kept frames' rows added."""
        version = int(prev["version"]) + 1
        rows = [*prev["frames"], *(self._kept_row(fr, version)
                                   for fr in sorted(kept, key=lambda fr: fr.work.index))]
        centres = np.array([np.asarray(r["T_map_cam"])[:3, 3] for r in rows])
        b = prev.get("bounds_m") or {"min": centres.min(0).tolist(),
                                     "max": centres.max(0).tolist()}
        lo = np.minimum(np.asarray(b["min"]), centres.min(0))
        hi = np.maximum(np.asarray(b["max"]), centres.max(0))
        rec = {k: v for k, v in prev.items() if k not in ("version", "created_at", "cameras")}
        rec.update({"kind": "update", "job_id": job.id, "frames": rows,
                    "drift_sim": None if drift[0] is None else drift[0].to_dict(),
                    "drift_sigma": None if drift[1] is None else list(drift[1]),
                    "drift_support": drift[2],
                    "bounds_m": {"min": lo.round(4).tolist(), "max": hi.round(4).tolist()}})
        return rec

    def _record(self, job: Job, frames: list[_Frame], registered: list[_Frame],
                prev: dict[str, Any] | None, internal: InternalMap, anchored: _Anchored,
                objects: int) -> dict[str, Any]:
        sim = anchored.sim
        by_name = {f.name: f for f in internal.frames}

        def depth_m(f: InternalFrame) -> float | None:
            d = internal.depth(f.name)
            med = None if d is None else checks.median_depth(d)
            return None if med is None else med * sim.s

        out = []
        version = (prev["version"] + 1) if prev else 1
        for r in (prev["frames"] if prev else []):
            f = by_name.get(r["internal"])
            if f is None:  # kept as it was
                out.append(r)
                continue
            p = r.get("prior")
            sp, sr = pose_sigmas(f, depth_m(f), anchored.align_rms_m, False,
                                 None if p is None else p.get("trust"))
            out.append({**r, "T_map_cam": sim.pose(f.T).tolist(), "sigma_pos_m": sp,
                        "sigma_rot_deg": sr, "anchor_sigmas": list(pose_sigmas(
                            f, depth_m(f), anchored.align_rms_m, calibrate=False))})
        for fr in sorted(registered, key=lambda fr: fr.work.index):
            assert fr.pose is not None
            raw = None
            if fr.internal is not None:
                fr.sigma_pos_m, fr.sigma_rot_deg = pose_sigmas(
                    fr.internal, depth_m(fr.internal), anchored.align_rms_m, _has_depth(fr),
                    fr.work.trust)
                raw = list(pose_sigmas(fr.internal, depth_m(fr.internal),
                                       anchored.align_rms_m, calibrate=False))
            p = fr.work.prior
            out.append({"name": fr.work.name, "camera_id": fr.work.camera_id,
                        "internal": None if fr.internal is None else fr.internal.name,
                        "T_map_cam": fr.pose.tolist(),
                        "sigma_pos_m": fr.sigma_pos_m, "sigma_rot_deg": fr.sigma_rot_deg,
                        "pose_state": "fused" if fr.internal is not None else "kept_at_prior",
                        "added_version": version, "sha256": fr.sha,
                        "inliers": 0 if fr.internal is None else _inliers(fr.internal),
                        # the too-close exception (api.md §4): a kept frame's in-range depth
                        # feeds occupancy and heights at its prior (``derived``)
                        "content": fr.internal is None and _too_close_content(fr),
                        **({} if raw is None else {"anchor_sigmas": raw}),
                        # its camera height sigma (locate's height_sigma, api.md §6)
                        "height_sigma_m": fr.work.height_sigma_m if fr.work.height_m is not None
                        else fr.cam.height_sigma_m,
                        "prior": p if p is not None and p["trust"] == "metric" else None})
        centres = np.array([np.asarray(r["T_map_cam"])[:3, 3] for r in out])
        if internal.bounds is not None:
            lo, hi = np.array(internal.bounds[0]), np.array(internal.bounds[1])
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                                for z in (lo[2], hi[2])])
            pts = np.vstack([sim.point(corners), centres])
        else:
            pts = centres
        floor = None
        floor_sigma = FLOOR_SIGMA_DEFAULT_M  # the version's one floor estimate (map.md §5)
        floor_z = map_floor(internal)
        if floor_z is not None:
            c = np.array([*np.mean([f.T[:2, 3] for f in internal.frames], axis=0), floor_z])
            floor = round(float(sim.point(c)[2]), 4)
            if internal.points_z is not None:
                floor_sigma = floor_spread(internal.points_z, floor_z) * sim.s
        if anchored.origin_height_m is not None:
            # the camera height's floor, z 0 (map.md §6), refined by the mapper's floor when it
            # lies within FLOOR_BAND_SIGMAS of the floor points' spread of it
            floor = floor if floor is not None and abs(floor) <= FLOOR_BAND_SIGMAS * floor_sigma \
                else 0.0
        rows = {r["name"]: r for r in out}
        for fr in registered:
            if fr.internal is not None and self._height_off(fr, floor):
                fr.height_off = True
                rows[fr.work.name]["height_inconsistent"] = True
        return {"kind": "update", "job_id": job.id, "sim3": sim.to_dict(),
                "origin_height_m": anchored.origin_height_m,
                # the drift estimate across versions (``_drifting_residuals``)
                "drift_sim": None if anchored.drift_sim is None else anchored.drift_sim.to_dict(),
                "drift_sigma": None if anchored.drift_sigma is None
                else [round(v, 5) for v in anchored.drift_sigma],
                "drift_support": anchored.drift_support,
                "scale_source": anchored.scale_source,
                "scale_sigma_rel": round(anchored.scale_sigma_rel, 5), "frames": out,
                "objects": objects, "floor_z_m": floor, "floor_sigma_m": round(floor_sigma, 4),
                "pose_states": POSE_STATES,
                "bounds_m": {"min": pts.min(0).round(4).tolist(),
                             "max": pts.max(0).round(4).tolist()}}

    @staticmethod
    def _input(fr: _Frame) -> dict[str, Any]:
        from oh_my_slam.service.prior_first import input_row

        ok = fr.reason is None and fr.internal is not None
        status = "registered" if ok else "kept_at_prior" if fr.reason is None \
            and fr.unconfirmed is not None else "rejected"
        return input_row(fr.work.name, status,
                         fr.unconfirmed if status == "kept_at_prior" else fr.reason,
                         _inliers(fr.internal) if fr.internal is not None else 0, fr.pose,
                         fr.sigma_pos_m, fr.sigma_rot_deg, fr.prior_residual, fr.scene_m,
                         fr.duplicate_of, fr.detail)

    @staticmethod
    def _quality(frames: list[_Frame], registered: list[_Frame], kept: list[_Frame],
                 prev: dict[str, Any] | None, anchored: _Anchored | None,
                 accept_if: dict[str, float] | None, rebuilt: bool, no_prior: bool,
                 rotations_bad: bool = False,
                 rotation_pairs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        rejected = len(frames) - len(registered) - len(kept)
        low = sum((fr.sigma_pos_m or 0.0) > LOW_CONFIDENCE_SIGMA_M
                  or (fr.internal is not None and fr.internal.low_confidence) for fr in registered)
        prior_res = residual_stats(anchored.prior_residuals) if anchored else None
        prev_res = residual_stats(anchored.previous_residuals) if anchored else None
        gauge = anchored.gauge_shift if anchored else None
        q: dict[str, Any] = {
            "scale_ratio_vs_prev": None if prev is None or anchored is None
            else anchored.scale_ratio_vs_prev,
            "gauge_shift": gauge,
            # map.md §1 "New-frame displacement": a frame with a prior against that prior (its
            # prior residual, through the drift estimate for a drifting one); one without, its
            # committed pose is its aligned image pose: 0
            **displacement_quality([(0.0, 0.0) if fr.prior_residual is None
                                    else fr.prior_residual for fr in registered]),
            "prior_disagreement_deg": anchored.prior_disagreement_deg if anchored else 0.0,
            # the mapper ties kept frames to no image: nothing re-registers them (map.md §2)
            "reregistered": [],
            "height_inconsistent": [fr.work.name for fr in registered if fr.height_off],
            "rotation_pairs": list(rotation_pairs or []),
            "prior_residuals": prior_res, "previous_residuals": prev_res,
            "rejected_share": round(rejected / len(frames), 4),
            "low_confidence_share": round(low / len(registered), 4) if registered else 0.0,
            "rebuilt": rebuilt,
            "prior_gravity_deg": None if prev is not None or anchored is None
            else anchored.prior_gravity_deg,
        }
        # the hard limit on content: nothing registered and nothing kept (map.md §4)
        return gate(q, bool(registered or kept), accept_if, no_prior, rotations_bad)


