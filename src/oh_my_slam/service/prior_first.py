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
   map's mapper folder; the job reports one ``InputStatus`` per frame and the quality report;
   if the gate passes, the clone is committed with the new version."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np

from oh_my_slam.mapping.anchored import AnchoredMap, FrameIn, Placement, pose_residual
from oh_my_slam.service.anchor import Sim3
from oh_my_slam.service.backend import Mono
from oh_my_slam.service.store import clone_tree

if TYPE_CHECKING:
    from pathlib import Path

    from oh_my_slam.service.jobs import Job
    from oh_my_slam.service.update import Updater, UpdateWork, _Frame

INFERENCE_IN_FLIGHT = 2
LOW_CONFIDENCE_SIGMA_M = 0.10  # as update.LOW_CONFIDENCE_SIGMA_M
DEPTH_SCALED_SOURCES = ("camera_height", "monocular")


def _waiter(fut: Future[Mono]) -> Callable[[], tuple[Any, Any, Any, Any]]:
    return lambda: fut.result().as_tuple()


def _frame_in(fr: _Frame, mono: Future[Mono] | None) -> FrameIn:
    f = fr.work
    assert fr.rgb is not None and fr.measures is not None
    prior = f.prior if f.trust in ("metric", "drifting") else None
    height, sigma = (f.height_m, f.height_sigma_m) if f.height_m is not None else (
        fr.cam.height_above_floor_m, fr.cam.height_sigma_m)
    return FrameIn(f.name, f.camera_id, fr.rgb, fr.cam.canonical(), prior, fr.depth_m, fr.ignore,
                   height, sigma, fr.measures.features, None if mono is None else _waiter(mono))


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
        gravity = prev is None
        ins = []
        for fr in todo:
            fut = None
            if not _has_depth(fr):
                assert fr.path is not None
                fut = pool.submit(up.backend.infer, fr.path, fr.cam.canonical(),
                                  gravity and fr.work.trust in ("metric", "drifting"))
                gravity = gravity and fr.work.trust not in ("metric", "drifting")
            ins.append(_frame_in(fr, fut))
        done = [0]

        def progress(msg: str) -> None:
            done[0] += 1
            job.progress = round(0.1 + 0.8 * done[0] / len(todo), 3)
            changed()

        try:
            pls = am.update(ins, progress)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        for fr, pl in zip(todo, pls, strict=True):
            placements[fr.work.index] = pl
        t = time.perf_counter()
        am.save()
        tm["register"] = am.timings["register"]
        tm["fusion"] = round(am.timings["fuse"] + time.perf_counter() - t, 3)
        n = len(todo)
        up.timings.add("registration_per_frame", tm["register"] / n, n)
        up.timings.add("fusion_per_frame", tm["fusion"] / n, n)
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
        ratio = round(am.scale.k / k_prev, 5) if source in DEPTH_SCALED_SOURCES \
            and k_prev else 1.0
    low = sum(pl.status == "provisional" or (pl.sigma_pos_m or 0.0) > LOW_CONFIDENCE_SIGMA_M
              for _, pl in kept)
    q: dict[str, Any] = {
        "scale_ratio_vs_prev": ratio,
        "gauge_shift": None if prev is None else {"translation_m": 0.0, "rotation_deg": 0.0,
                                                  "scale_ratio": 1.0},
        "prior_residuals": residual_stats(prior_res),
        "previous_residuals": residual_stats([(0.0, 0.0)] * len(prev["frames"])) if prev
        else None,
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
    assert am is not None
    floor, pts = am.floor_and_bounds()
    record = {"kind": "update", "engine": "anchored", "job_id": job.id,
              "sim3": Sim3().to_dict(), "scale_source": source,
              "scale_sigma_rel": round(sigma, 5), "frames": rows, "objects": 0,
              "floor_z_m": None if floor is None else round(floor, 4),
              "bounds_m": {"min": pts.min(0).round(4).tolist(),
                           "max": pts.max(0).round(4).tolist()},
              "points": len(am.cloud.keys)}
    job.result_version = up.store.commit(w.map, record, mapper)
    job.state, job.progress = "committed", 1.0
    tm["commit"] = round(time.perf_counter() - t, 3)
