"""The update job of an anchored map: prior-first registration (``mapping.anchored``; README
"Prior-first registration").

A map whose first update carries a metric or drifting prior on a frame the checks keep is
anchored: its frames are placed by their priors and refined against the stored keyframes, and
only the new keyframes are fused. Its versions record ``engine: anchored`` and the identity
similarity: the mapper's frame is the map frame, and stored poses never move.

1. The checks (``update.Updater._check``) have run: frames they reject never reach the engine,
   so a job whose frames are all rejected ends in the time of the checks.
2. The monocular depth of every kept frame without a depth image is asked of the inference
   server, two frames in flight, while the engine registers the frames in request order.
3. The engine (``AnchoredMap.update``) places, keeps and fuses the frames in a clone of the
   map's mapper folder; the detections of every frame, asked after the depth, are fused into the
   map's objects for the new keyframes only (``fuse_objects``: ``mapping.anchored_objects``);
   the job reports one ``InputStatus`` per frame and the quality report; if the gate passes, the
   clone is committed with the new version and its object records (``versions/<n>.objects.json``,
   read by ``service.objects``)."""

from __future__ import annotations

import functools
import os
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.images import size_at_max_side
from oh_my_slam.core.types import Intrinsics
from oh_my_slam.mapping import register_ma
from oh_my_slam.mapping.anchor import Sim3
from oh_my_slam.mapping.anchored import AnchoredMap, FrameIn, Placement, features, pose_residual
from oh_my_slam.mapping.anchored_objects import (
    MONO_DEPTH_REL,
    SENSOR_DEPTH_REL,
    ObjectMap,
    View,
)
from oh_my_slam.segmentation.api import (
    Detection,
    Region,
    raised_regions,
)
from oh_my_slam.service.backend import MONO_GRID_SIDE, Mono
from oh_my_slam.service.store import clone_tree

if TYPE_CHECKING:
    from pathlib import Path

    from oh_my_slam.service.jobs import Job
    from oh_my_slam.service.update import Updater, UpdateWork, _Frame

    # the closer look at a keyframe looking down at its support (image, raised areas,
    # detections, grid): its detections, more and with their other labels
    Closer = Callable[[Path, list[Region], list[Detection], tuple[int, int]], list[Detection]]

INFERENCE_IN_FLIGHT = 2
LOW_CONFIDENCE_SIGMA_M = 0.10  # as update.LOW_CONFIDENCE_SIGMA_M
NONE_ENGINE_ENV = "OH_MY_SLAM_NONE_ENGINE"
DEFAULT_NONE_ENGINE = "mapanything"


def none_engine() -> str:
    """The engine of the frames without a prior (``register_ma``): ``OH_MY_SLAM_NONE_ENGINE``,
    else ``DEFAULT_NONE_ENGINE``."""
    v = os.environ.get(NONE_ENGINE_ENV, DEFAULT_NONE_ENGINE).strip().lower()
    if v not in register_ma.ENGINES:
        raise ValueError(f"{NONE_ENGINE_ENV}={v!r}: expected one of {register_ma.ENGINES}")
    return v


def camera_only(w: UpdateWork) -> int:
    """The frames of a request MapAnything registers (``none_engine``): those without a metric
    or drifting prior."""
    if none_engine() != "mapanything":
        return 0
    return sum(f.trust not in ("metric", "drifting") for f in w.frames)


def _trusted(fr: _Frame) -> bool:
    return fr.work.trust in ("metric", "drifting")


def _register_none(up: Updater, am: AnchoredMap, frames: list[_Frame], workdir: Path,
                   progress: Callable[[str], None]) -> tuple[list[Placement], float, float]:
    """The camera-only frames by MapAnything (``register_ma``) against the map's confirmed
    keyframes, each registered one added as a keyframe; its depth scale is the map's, and its
    camera height feeds the map's depth scale (``ScaleState``) as a monocular keyframe's does.
    Returns the placements (frames' order) and the registration and fusion seconds."""
    t = time.perf_counter()
    am.save()  # this update's keyframes get their image and depth on disk: anchors
    stored = [register_ma.Stored(k.id, img, k.K, k.T, i, functools.partial(am.depth_of, k.id))
              for i, k in enumerate(am.keyframes)
              if k.state == "fused" for img in [am.image_path(k.id)] if img is not None]
    if not stored:  # nothing confirmed to register against (a camera-only map without origin)
        return [Placement(fr.work.name, "rejected", "no_overlap") for fr in frames], \
            time.perf_counter() - t, 0.0
    new = []
    for fr in frames:
        assert fr.path is not None and fr.measures is not None
        new.append(register_ma.New(fr.work.index, fr.path, fr.cam.canonical(),
                                   fr.depth_m if _has_depth(fr) else None,
                                   fr.measures.features))
    calls = [0]

    def multiview(images: list[Path], Ks: list[Intrinsics], poses: list[Any]) -> Any:
        calls[0] += 1
        return up.backend.geometry_views(images, Ks, poses, workdir / f"mv{calls[0]}")

    res = register_ma.Registration(multiview, new, stored).run()
    reg_s = time.perf_counter() - t
    t = time.perf_counter()
    out = []
    for fr in frames:
        p = res.placements[fr.work.index]
        if p.status == "rejected":
            out.append(Placement(fr.work.name, "rejected", p.reason))
            progress(f"{fr.work.name}: rejected ({p.reason})")
            continue
        assert p.T is not None and p.depth is not None and p.K_depth is not None
        pl = Placement(fr.work.name, "registered", None, p.T, p.sigma_pos_m, p.sigma_rot_deg,
                       p.inliers, "mapanything")
        f = _frame_in(fr, None)
        h, w = p.depth.shape
        Kg = Intrinsics(float(p.K_depth[0, 0]), float(p.K_depth[1, 1]), float(p.K_depth[0, 2]),
                        float(p.K_depth[1, 2]), w, h)
        depth = p.depth
        kind = "sensor" if _has_depth(fr) else "mono"
        if kind == "mono":
            am.add_height_evidence(f, p.T, depth, Kg)
            k = am.scale.k
            am.scale.estimate(am.prior_span(), am.prior_span("drifting"))
            depth = (depth * (am.scale.k / k)).astype(np.float32)
        assert fr.rgb is not None
        am.add(f, features(fr.rgb, fr.ignore), pl, depth, Kg, kind, None)
        out.append(pl)
        progress(f"{fr.work.name}: registered (mapanything, {p.inliers} points)")
    return out, reg_s, time.perf_counter() - t
def detection_grid(K: Intrinsics) -> tuple[int, int]:
    """The grid (width, height) a frame's detections are asked on: its monocular depth grid."""
    return size_at_max_side(K.width, K.height, MONO_GRID_SIDE)


def _on_grid(a: NDArray[Any], shape: tuple[int, int], interp: int) -> NDArray[Any]:
    if a.shape[:2] == shape:
        return a
    return cv2.resize(a, (shape[1], shape[0]), interpolation=interp)


def fuse_objects(am: AnchoredMap, frames: list[tuple[_Frame, Placement, Future[list[Detection]]]],
                 version: int, closer: Closer) -> tuple[ObjectMap, int]:
    """The objects of the new keyframes (``mapping.anchored_objects``), fused in request order
    into the map's: struck objects carved from the cloud, then the surfaces and relations.
    ``closer`` looks closer at a keyframe looking down at its support
    (``segmentation.api.raised_regions``; the backend's ``look_closer``). Returns the objects and how many are listed."""
    om = ObjectMap.load(am.root)
    by_id = {k.id: k for k in am.keyframes}
    views = []
    for fr, pl, fut in frames:
        kf = by_id.get(pl.keyframe or "")
        if pl.status == "rejected" or kf is None or kf.grid is None or kf.K_grid is None:
            continue
        assert kf.rgb is not None
        shape = kf.grid.shape
        dets = []
        for d in fut.result():
            m = _on_grid(d.mask.astype(np.uint8), shape, cv2.INTER_NEAREST) > 0
            sx, sy = shape[1] / d.mask.shape[1], shape[0] / d.mask.shape[0]
            b = d.box
            dets.append(Detection(d.label, d.score, d.source, m,
                                  (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy)))
        ignore = None if fr.ignore is None else _on_grid(
            fr.ignore.astype(np.uint8), shape, cv2.INTER_NEAREST) > 0
        rel = SENSOR_DEPTH_REL if kf.depth_kind == "sensor" else MONO_DEPTH_REL
        regions = raised_regions(kf.grid, kf.K_grid, kf.T, rel,
                                 None if ignore is None else ~ignore)
        if regions:  # a view looking down at its support
            assert fr.path is not None  # a frame kept has its image
            dets = closer(fr.path, regions, dets, (shape[1], shape[0]))
        views.append(View(kf.id, kf.T, kf.sigma_pos_m, kf.K_grid, kf.grid,
                          _on_grid(kf.rgb, shape, cv2.INTER_AREA), dets, ignore, rel))
    floor, _ = am.floor_and_bounds()
    struck = om.update(views, version, floor)
    if struck:
        am.cloud.drop(om.carve(struck, am.cloud.points()))
    top = max((float(k.centre[2]) for k in am.keyframes), default=None)
    om.finish(am.cloud.points(), top, version,
              np.array([k.centre for k in am.keyframes]).reshape(-1, 3), floor)
    return om, len(om.records()["objects"])


def _waiter(fut: Future[Mono]) -> Callable[[], tuple[Any, Any, Any, Any]]:
    return lambda: fut.result().as_tuple()


def _frame_in(fr: _Frame, mono: Future[Mono] | None) -> FrameIn:
    f = fr.work
    assert fr.rgb is not None and fr.measures is not None
    prior = f.prior if f.trust in ("metric", "drifting") else None
    height, sigma = (f.height_m, f.height_sigma_m) if f.height_m is not None else (
        fr.cam.height_above_floor_m, fr.cam.height_sigma_m)
    return FrameIn(f.name, f.camera_id, fr.rgb, fr.cam.canonical(), prior, fr.depth_m, fr.ignore,
                   height, sigma, fr.measures.features, None if mono is None else _waiter(mono),
                   check_close=fr.deferred)


def _has_depth(fr: _Frame) -> bool:
    return fr.depth_m is not None and bool(np.count_nonzero(fr.depth_m > 0))


def _input(fr: _Frame, pl: Placement | None) -> dict[str, Any]:
    if pl is None:  # rejected by the checks
        return {"name": fr.work.name, "status": "rejected", "reason": fr.reason, "inliers": 0,
                "T_map_cam": None, "sigma_pos_m": None, "sigma_rot_deg": None,
                "prior_residual_m": None, "prior_residual_deg": None}
    kept = pl.status != "rejected"
    res = pl.prior_residual if fr.work.trust in ("metric", "drifting") else None
    return {"name": fr.work.name, "status": pl.status, "reason": pl.reason,
            "inliers": pl.inliers,
            "T_map_cam": pl.T.round(6).tolist() if kept and pl.T is not None else None,
            "sigma_pos_m": pl.sigma_pos_m if kept else None,
            "sigma_rot_deg": pl.sigma_rot_deg if kept else None,
            "prior_residual_m": None if res is None else round(res[0], 4),
            "prior_residual_deg": None if res is None else round(res[1], 3)}


def run_anchored(up: Updater, job: Job, w: UpdateWork, frames: list[_Frame],
                 prev: dict[str, Any] | None, workdir: Path,
                 changed: Callable[[], None] = lambda: None) -> None:
    from oh_my_slam.service.update import gate, residual_stats

    tm = job.timings_s
    todo = [fr for fr in frames if fr.reason is None and fr.path is not None]
    placements: dict[int, Placement] = {}
    am: AnchoredMap | None = None
    k_prev: float | None = None
    mapper = workdir / "anchored"
    om: ObjectMap | None = None
    n_objects = 0
    if todo:
        t = time.perf_counter()
        if prev is not None:
            clone_tree(up.store.mapper_dir(w.map), mapper)
            am = AnchoredMap.load(mapper)
            k_prev = am.scale.k
        else:
            mapper.mkdir(parents=True, exist_ok=True)
            am = AnchoredMap(mapper)
        tm["clone"] = round(time.perf_counter() - t, 3)
        pool = ThreadPoolExecutor(INFERENCE_IN_FLIGHT)
        # a new map's origin frame (its first with a prior, else its first) also gives
        # gravity: the angle to its prior's up axis, or the camera-only map's up
        origin = None
        if prev is None:
            origin = next((fr for fr in todo if _trusted(fr)), todo[0])
        # frames without a prior go to MapAnything after the others, but a new camera-only
        # map's origin is the prior-first engine's
        late = [fr for fr in todo if not _trusted(fr) and fr is not origin] \
            if none_engine() == "mapanything" else []
        geo = [fr for fr in todo if all(fr is not x for x in late)]
        ins = []
        for fr in geo:
            fut = None
            if not _has_depth(fr) or fr is origin:
                assert fr.path is not None
                fut = pool.submit(up.backend.infer, fr.path, fr.cam.canonical(), fr is origin)
            ins.append(_frame_in(fr, fut))
        # the detections queue behind the monocular depth registration waits for
        dets = {fr.work.index: pool.submit(up.backend.detect, fr.path,
                                           detection_grid(fr.cam.canonical()))
                for fr in todo if fr.path is not None}
        done = [0]

        def progress(msg: str) -> None:
            done[0] += 1
            job.progress = round(0.1 + 0.8 * done[0] / len(todo), 3)
            changed()

        try:
            pls = am.update(ins, progress) if ins else []
            reg_s, fuse_s = am.timings.get("register", 0.0), am.timings.get("fuse", 0.0)
            for fr, pl in zip(geo, pls, strict=True):
                placements[fr.work.index] = pl
            if geo:
                up.timings.add("registration_per_frame", reg_s / len(geo), len(geo))
            if late:
                lp, r, fu = _register_none(up, am, late, workdir, progress)
                up.timings.add("registration_mapanything_per_frame", r / len(late), len(late))
                reg_s, fuse_s = reg_s + r, fuse_s + fu
                for fr, pl in zip(late, lp, strict=True):
                    placements[fr.work.index] = pl
            t = time.perf_counter()
            version = (prev["version"] + 1) if prev else 1
            om, n_objects = fuse_objects(am, [(fr, placements[fr.work.index], dets[fr.work.index])
                                              for fr in todo], version, up.backend.look_closer)
            tm["objects"] = round(time.perf_counter() - t, 3)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        t = time.perf_counter()
        am.save()
        om.save()
        tm["register"] = round(reg_s, 3)
        tm["fusion"] = round(fuse_s + tm["objects"] + time.perf_counter() - t, 3)
        n_kept = sum(pl.status != "rejected" for pl in placements.values())
        if n_kept:
            up.timings.add("fusion_per_frame", tm["fusion"] / n_kept, n_kept)
    job.progress = 0.95
    changed()
    t = time.perf_counter()
    job.inputs = [_input(fr, placements.get(fr.work.index)) for fr in frames]
    kept = [(fr, placements[fr.work.index]) for fr in todo
            if placements[fr.work.index].status != "rejected"]
    version = (prev["version"] + 1) if prev else 1
    rows = list(prev["frames"]) if prev else []
    for fr, pl in kept:
        assert pl.T is not None
        p = fr.work.prior
        rows.append({"name": fr.work.name, "camera_id": fr.work.camera_id,
                     "internal": pl.keyframe, "T_map_cam": pl.T.tolist(),
                     "sigma_pos_m": pl.sigma_pos_m, "sigma_rot_deg": pl.sigma_rot_deg,
                     "pose_state": "fused" if pl.status == "registered" else "provisional",
                     "added_version": version, "sha256": fr.sha, "inliers": pl.inliers,
                     "method": pl.method,
                     "prior": p if p is not None and p["trust"] == "metric" else None})
    prior_res = [pose_residual(np.asarray(r["T_map_cam"]), np.asarray(r["prior"]["T_map_cam"]))
                 for r in rows if r.get("prior")]
    source, sigma = am.scale_summary() if am is not None else ("monocular", 0.1)
    ratio = None
    if prev is not None and am is not None:
        # the monocular keyframes' depth follows k; positions and depth images do not move
        mono = any(k.depth_kind == "mono" for k in am.keyframes)
        ratio = round(am.scale.k / k_prev, 5) if mono and k_prev else 1.0
    low = sum(pl.status == "provisional" or (pl.sigma_pos_m or 0.0) > LOW_CONFIDENCE_SIGMA_M
              for _, pl in kept)
    # the new frames against the stored map: how far the update puts each from where the
    # images on the previous version place it (stored frames never move)
    moved = [pl.map_residual for _, pl in kept if pl.map_residual is not None]
    gauge = None
    if prev is not None:
        gauge = {"translation_m": round(float(np.median([m for m, _ in moved])), 4)
                 if moved else 0.0,
                 "rotation_deg": round(float(np.median([d for _, d in moved])), 3)
                 if moved else 0.0, "scale_ratio": 1.0}
    q: dict[str, Any] = {
        "scale_ratio_vs_prev": ratio,
        "gauge_shift": gauge,
        "prior_residuals": residual_stats(prior_res),
        "previous_residuals": residual_stats(moved) if prev else None,
        "rejected_share": round(1.0 - len(kept) / len(frames), 4),
        "low_confidence_share": round(low / len(kept), 4) if kept else 0.0,
        "rebuilt": False,
        "prior_gravity_deg": None if prev is not None or am is None else am.up_check_deg,
    }
    job.quality = gate(q, bool(kept), w.accept_if)
    if am is not None:
        job.scale_source, job.scale_sigma_rel = source, round(sigma, 5)
    if not job.quality["accepted"]:
        job.state, job.progress = "rejected", 1.0
        return
    assert am is not None and om is not None
    floor, pts = am.floor_and_bounds()
    record = {"kind": "update", "engine": "anchored", "job_id": job.id,
              "sim3": Sim3().to_dict(), "scale_source": source,
              "scale_sigma_rel": round(sigma, 5), "frames": rows, "objects": n_objects,
              "floor_z_m": None if floor is None else round(floor, 4),
              "bounds_m": {"min": pts.min(0).round(4).tolist(),
                           "max": pts.max(0).round(4).tolist()},
              "points": len(am.cloud.keys)}
    job.result_version = up.store.commit(w.map, record, mapper,
                                         {"objects.json": om.records()}, w.cameras, w.masks)
    job.state, job.progress = "committed", 1.0
    tm["commit"] = round(time.perf_counter() - t, 3)
