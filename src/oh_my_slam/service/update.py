"""An update job (O1 anchored frame and priors, O4 per-input status; map.md §1, api.md §4).

1. **Checks.** Each frame is measured (``checks``): a frame already in the map (same name or
   same image bytes) or earlier in the request is a ``duplicate``; one that is ``too_close``,
   ``plain``, ``blur`` or has ``few_features`` is rejected before it reaches the mapper.
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
   ``anchor.K_SIGMA`` sigmas is ``prior_conflict``; one whose aligned depth has a median below
   ``checks.TOO_CLOSE_M`` is ``too_close``. If any, registration runs once more without them.
5. **Quality and gate.** The quality report (map.md §4) and ``accept_if``; nothing registered is
   a hard limit. A passing job commits one version.
"""

from __future__ import annotations

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

from oh_my_slam.core.images import png_bytes, save_jpeg
from oh_my_slam.core.log import get_logger
from oh_my_slam.mapping import anchor, checks
from oh_my_slam.mapping.anchor import Anchor, Sim3
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
from oh_my_slam.service.store import Store, clone_tree
from oh_my_slam.service.timings import FUSION_STAGES, Timings

log = get_logger("oh_my_slam.service.update")

MONOCULAR_SIGMA_REL = 0.10  # MoGe-2 metric depth: about 10 % scale error between scenes
MIN_SCALE_SIGMA_REL = 0.005
DEFAULT_HEIGHT_SIGMA_M = 0.02
MIN_CAMERA_HEIGHT_M = 0.05  # internal camera heights below this tell no scale
LOW_CONFIDENCE_SIGMA_M = 0.10  # registered frames above it count in low_confidence_share
SIGMA_POS_FLOOR_M = 0.01  # pose sigma proxy floors (README "Per-input status")
SIGMA_ROT_FLOOR_DEG = 0.1
DEFAULT_SCENE_DEPTH_M = 2.0
LOW_CONFIDENCE_FACTOR = 3.0
FLOOR_BINS = 50  # estimate_floor
FLOOR_PEAK_SHARE = 0.25
FLOOR_MIN_POINTS = 200
JPEG_QUALITY = 95
DEPTH_PLACED = ("plain", "few_features")  # reasons a frame with depth and a prior escapes
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


def _mat(T: Any) -> NDArray[np.float64]:
    return np.asarray(T, np.float64)


def _prior_sigmas(p: dict[str, Any]) -> tuple[float, float]:
    return (float(p.get("sigma_pos_m", anchor.DEFAULT_SIGMA_POS_M)),
            float(p.get("sigma_rot_deg", anchor.DEFAULT_SIGMA_ROT_DEG)))


def residual_stats(res: list[tuple[float, float]]) -> dict[str, Any] | None:
    """The ``Residuals`` object of (m, deg) residuals, or None without any."""
    if not res:
        return None
    m = np.array([r[0] for r in res])
    d = np.array([r[1] for r in res])
    return {"median_m": round(float(np.median(m)), 4), "max_m": round(float(m.max()), 4),
            "median_deg": round(float(np.median(d)), 3), "max_deg": round(float(d.max()), 3),
            "count": len(res)}


def pose_sigmas(f: InternalFrame, scene_depth_m: float | None, align_rms_m: float
                ) -> tuple[float, float]:
    """The pose sigma proxy (README "Per-input status"): the registration's angular noise per
    observation (its mean reprojection error over the focal length, or the multi-view
    refinement's median match residual) over the square root of its observations, times the
    scene depth for the position, combined with the anchoring's RMS residual; three times that
    for a low-confidence keyframe, floored."""
    st = f.stats
    if "pose_matches" in st:
        n = float(st.get("pose_matches") or 0)
        ang = math.radians(float(st.get("pose_residual_deg") or 0.5))
    else:
        n = float(st.get("observations") or 0)
        ang = float(st.get("reproj_error") or 1.0) / max(f.fx, 1.0)
    n = max(n, 1.0)
    d = scene_depth_m or DEFAULT_SCENE_DEPTH_M
    rot = ang * math.sqrt(2.0 / n)
    pos = d * ang * math.sqrt(8.0 / n)
    k = LOW_CONFIDENCE_FACTOR if f.low_confidence else 1.0
    pos = math.hypot(k * pos, align_rms_m)
    return (round(max(pos, SIGMA_POS_FLOOR_M), 4),
            round(max(math.degrees(k * rot), SIGMA_ROT_FLOOR_DEG), 3))


def estimate_floor(points_z: NDArray[np.float64] | None, below: float) -> float | None:
    """The floor's height from a cloud's point heights when the mapper has none (it takes its
    floor from a detected floor surface, which a robot looking ahead may never see): among the
    points lower than every camera (``below``), the lowest dense layer — the lowest of
    ``FLOOR_BINS`` height bins between their 1st percentile and ``below`` that holds at least
    ``FLOOR_PEAK_SHARE`` of the fullest bin — as the median height of that bin and its two
    neighbours. None with fewer than ``FLOOR_MIN_POINTS`` such points."""
    if points_z is None:
        return None
    z = points_z[np.isfinite(points_z) & (points_z < below)]
    if z.size < FLOOR_MIN_POINTS:
        return None
    lo = float(np.percentile(z, 1))
    hist, edges = np.histogram(z, bins=FLOOR_BINS, range=(lo, below))
    i = int(np.argmax(hist >= FLOOR_PEAK_SHARE * hist.max()))
    near = z[(z >= edges[max(i - 1, 0)]) & (z <= edges[min(i + 2, FLOOR_BINS)])]
    return float(np.median(near))


def map_floor(internal: InternalMap) -> float | None:
    """The internal map's floor height: the mapper's, else ``estimate_floor`` of its cloud."""
    if internal.floor_z is not None:
        return internal.floor_z
    if not internal.frames:
        return None
    return estimate_floor(internal.points_z, min(float(f.T[2, 3]) for f in internal.frames))


def _inliers(f: InternalFrame) -> int:
    return int(f.stats.get("observations") or f.stats.get("pose_matches") or 0)


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
                                                 - tm.get("fusion", 0.0), 0.0))

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
        """The per-frame checks: duplicates in request order, then the measurements of the
        others, ``CHECK_THREADS`` frames at a time."""
        names = {r["name"] for r in prev["frames"]} if prev else set()
        shas = {r.get("sha256") for r in prev["frames"]} if prev else set()
        todo = []
        for fr in frames:
            if fr.work.name in names or fr.sha in shas:
                fr.reason = "duplicate"
                continue
            names.add(fr.work.name)
            shas.add(fr.sha)
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
        # the monocular depth ``too_close`` needs is measured once the path is known: the
        # prior-first path gets it with the frame's depth (``_resolve_deferred``)
        fr.deferred = med is None and fr.measures.suspicious
        fr.reason = checks.reason(fr.measures, med)
        if fr.reason in DEPTH_PLACED and med is not None and f.trust in ("metric", "drifting"):
            # a bare wall with a depth image and a prior is placed by both (the prior-first
            # engine's depth registration), not rejected for its texture
            fr.reason = None

    def _resolve_deferred(self, frames: list[_Frame]) -> None:
        """The median monocular depth of the suspicious frames, ``CHECK_THREADS`` asked at once
        (``too_close`` takes precedence over the other reasons): the mapper's path and the dry
        run."""
        todo = [fr for fr in frames if fr.deferred]

        def one(fr: _Frame) -> None:
            assert fr.path is not None and fr.measures is not None
            fr.reason = checks.reason(fr.measures, self.backend.median_depth(
                fr.path, fr.cam.canonical()))

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
        pre = {fr.work.index: fr.reason for fr in frames}
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
                    assert fr.measures is not None
                    fr.reason = checks.unregistered_reason(fr.measures)
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
                    assert fr.measures is not None
                    fr.reason = checks.unregistered_reason(fr.measures)
            placed = [fr for fr in todo if fr.internal is not None]
            if not placed:
                result = None
                break
            anchored = self._anchor(placed, prev, internal)
            post = self._post_check(placed, anchored, internal)
            tm["anchor"] = round(tm.get("anchor", 0.0) + time.perf_counter() - t, 3)
            result = (mapper, internal, anchored, reg.objects)
            if not post or attempt:
                break
            exclude |= {fr.work.index for fr in post}
            attempt += 1
        self._finish(job, w, frames, prev, result)

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
                previous.append(Anchor(r["name"], "previous", f.T, _mat(r["T_map_cam"]),
                                       r["sigma_pos_m"], r["sigma_rot_deg"]))
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
        self._drifting_residuals(placed, out.sim)
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
            return _Anchored(anchor.yaw_frame(first.internal.T, s), source, sigma, 0.0)
        prev_sim = Sim3.from_dict(prev["sim3"])
        group = previous + metric
        if not group:  # the mapper kept none of the stored frames: its frame is unchanged
            return _Anchored(prev_sim, prev["scale_source"], prev["scale_sigma_rel"], 0.0,
                             gauge_shift={"translation_m": 0.0, "rotation_deg": 0.0,
                                          "scale_ratio": 1.0}, scale_ratio_vs_prev=1.0)
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
                        scale_from_priors=source == "priors")
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
    def _drifting_residuals(placed: list[_Frame], sim: Sim3) -> None:
        """Residuals of drifting priors: after the similarity that best brings the request's
        aligned poses onto them (their frame drifts; their relative motion is reliable)."""
        drift = [fr for fr in placed if fr.work.trust == "drifting"]
        if not drift:
            return
        anchors = []
        for fr in drift:
            assert fr.internal is not None and fr.work.prior is not None
            anchors.append(Anchor(fr.work.name, "drifting", sim.pose(fr.internal.T),
                                  _mat(fr.work.prior["T_map_cam"]),
                                  *_prior_sigmas(fr.work.prior)))
        fit = anchor.fit(anchors)
        for fr, a in zip(drift, anchors, strict=True):
            fr.prior_residual = anchor.residual(fit.sim, a)

    def _post_check(self, placed: list[_Frame], anchored: _Anchored, internal: InternalMap
                    ) -> list[_Frame]:
        out = []
        for fr in placed:
            assert fr.internal is not None
            if fr.prior_residual is not None and fr.work.prior is not None:
                sp, sr = _prior_sigmas(fr.work.prior)
                dp, dr = fr.prior_residual
                if (dp > anchor.K_SIGMA * max(sp, anchor.MIN_SIGMA_POS_M)
                        or dr > anchor.K_SIGMA * max(sr, anchor.MIN_SIGMA_ROT_DEG)):
                    fr.reason = "prior_conflict"
                    out.append(fr)
                    continue
            d = internal.depth(fr.internal.name)
            med = None if d is None else checks.median_depth(d)
            if med is not None and med * anchored.sim.s < checks.TOO_CLOSE_M:
                fr.reason = "too_close"
                out.append(fr)
        return out

    # -- result ----------------------------------------------------------------------------------

    def _finish(self, job: Job, w: UpdateWork, frames: list[_Frame], prev: dict[str, Any] | None,
                result: tuple[Path, InternalMap, _Anchored, int] | None) -> None:
        t = time.perf_counter()
        registered = [fr for fr in frames if fr.reason is None and fr.internal is not None]
        record: dict[str, Any] | None = None
        anchored = None
        if result is not None:
            _, internal, anchored, objects = result
            record = self._record(job, frames, registered, prev, internal, anchored, objects)
        job.inputs = [self._input(fr) for fr in frames]
        quality = self._quality(frames, registered, prev, anchored, w.accept_if,
                                result is not None and result[1].rebuilt)
        job.quality = quality
        if anchored is not None:
            job.scale_source, job.scale_sigma_rel = anchored.scale_source, round(
                anchored.scale_sigma_rel, 5)
        if not quality["accepted"]:
            job.state = "rejected"
            job.progress = 1.0
            return
        assert result is not None and record is not None
        job.result_version = self.store.commit(w.map, record, result[0], None, w.cameras,
                                               w.masks)
        job.state = "committed"
        job.progress = 1.0
        job.timings_s["commit"] = round(time.perf_counter() - t, 3)

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
            sp, sr = pose_sigmas(f, depth_m(f), anchored.align_rms_m)
            out.append({**r, "T_map_cam": sim.pose(f.T).tolist(), "sigma_pos_m": sp,
                        "sigma_rot_deg": sr})
        for fr in registered:
            assert fr.internal is not None and fr.pose is not None
            fr.sigma_pos_m, fr.sigma_rot_deg = pose_sigmas(fr.internal, depth_m(fr.internal),
                                                           anchored.align_rms_m)
            p = fr.work.prior
            out.append({"name": fr.work.name, "camera_id": fr.work.camera_id,
                        "internal": fr.internal.name, "T_map_cam": fr.pose.tolist(),
                        "sigma_pos_m": fr.sigma_pos_m, "sigma_rot_deg": fr.sigma_rot_deg,
                        "pose_state": "fused", "added_version": version, "sha256": fr.sha,
                        "inliers": _inliers(fr.internal),
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
        floor_z = map_floor(internal)
        if floor_z is not None:
            c = np.array([*np.mean([f.T[:2, 3] for f in internal.frames], axis=0), floor_z])
            floor = round(float(sim.point(c)[2]), 4)
        return {"kind": "update", "job_id": job.id, "sim3": sim.to_dict(),
                "scale_source": anchored.scale_source,
                "scale_sigma_rel": round(anchored.scale_sigma_rel, 5), "frames": out,
                "objects": objects, "floor_z_m": floor,
                "bounds_m": {"min": pts.min(0).round(4).tolist(),
                             "max": pts.max(0).round(4).tolist()}}

    @staticmethod
    def _input(fr: _Frame) -> dict[str, Any]:
        ok = fr.reason is None and fr.internal is not None
        res = fr.prior_residual
        return {"name": fr.work.name, "status": "registered" if ok else "rejected",
                "reason": None if ok else fr.reason,
                "inliers": _inliers(fr.internal) if fr.internal is not None else 0,
                "T_map_cam": fr.pose.round(6).tolist() if ok and fr.pose is not None else None,
                "sigma_pos_m": fr.sigma_pos_m if ok else None,
                "sigma_rot_deg": fr.sigma_rot_deg if ok else None,
                "prior_residual_m": None if res is None else round(res[0], 4),
                "prior_residual_deg": None if res is None else round(res[1], 3)}

    @staticmethod
    def _quality(frames: list[_Frame], registered: list[_Frame], prev: dict[str, Any] | None,
                 anchored: _Anchored | None, accept_if: dict[str, float] | None,
                 rebuilt: bool) -> dict[str, Any]:
        rejected = sum(fr.reason is not None or fr.internal is None for fr in frames)
        low = sum((fr.sigma_pos_m or 0.0) > LOW_CONFIDENCE_SIGMA_M
                  or (fr.internal is not None and fr.internal.low_confidence) for fr in registered)
        prior_res = residual_stats(anchored.prior_residuals) if anchored else None
        prev_res = residual_stats(anchored.previous_residuals) if anchored else None
        gauge = anchored.gauge_shift if anchored else None
        q: dict[str, Any] = {
            "scale_ratio_vs_prev": None if prev is None or anchored is None
            else anchored.scale_ratio_vs_prev,
            "gauge_shift": gauge, "prior_residuals": prior_res, "previous_residuals": prev_res,
            "rejected_share": round(rejected / len(frames), 4),
            "low_confidence_share": round(low / len(registered), 4) if registered else 0.0,
            "rebuilt": rebuilt,
            "prior_gravity_deg": None if prev is not None or anchored is None
            else anchored.prior_gravity_deg,
        }
        return gate(q, bool(registered), accept_if)


def gate(q: dict[str, Any], kept_any: bool, accept_if: dict[str, float] | None
         ) -> dict[str, Any]:
    """``accepted`` and ``failed_checks`` of a quality report (map.md §4): nothing kept is the
    hard limit; ``accept_if`` checks the residual maxima, the gauge shift and the rejected
    share."""
    prior_res, prev_res, gauge = q["prior_residuals"], q["previous_residuals"], q["gauge_shift"]
    failed = []
    if not kept_any:
        failed.append("hard_limit")
    a = accept_if or {}
    if "max_residual_m" in a:
        worst = max([r["max_m"] for r in (prior_res, prev_res) if r] or [0.0])
        if worst > a["max_residual_m"]:
            failed.append("max_residual_m")
    if "max_gauge_shift_m" in a and gauge and gauge["translation_m"] > a["max_gauge_shift_m"]:
        failed.append("max_gauge_shift_m")
    if "max_rejected_share" in a and q["rejected_share"] > a["max_rejected_share"]:
        failed.append("max_rejected_share")
    q["accepted"] = not failed
    q["failed_checks"] = failed
    return q
