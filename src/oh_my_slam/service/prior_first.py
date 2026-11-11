"""The update job of an anchored map: prior-first registration (``mapping.anchored``; README
"Prior-first registration").

A map whose first update carries a metric or drifting prior on a frame the checks keep is
anchored: its frames are placed by their priors and refined against the stored keyframes, and
only the new keyframes are fused. Its versions record ``engine: anchored`` and the identity
similarity: the mapper's frame is the map frame, and stored poses never move.

1. The checks (``update.Updater._check``) have run: frames they reject never reach the engine,
   so a job whose frames are all rejected ends in the time of the checks.
2. The monocular depth of every kept frame without a depth image is asked of the inference
   server, two frames in flight, while the engine registers the frames over the job's match
   graph (map.md §2: the outcome does not depend on the request order).
3. The engine (``AnchoredMap.update``) places, keeps and fuses the frames in a clone of the
   map's mapper folder; the detections of every frame, asked after the depth, are fused into the
   map's objects for the new keyframes only (``fuse_objects``: ``mapping.anchored_objects``);
   the job reports one ``InputStatus`` per frame and the quality report; if the gate passes, the
   clone is committed with the new version and its object records (``versions/<n>.objects.json``,
   read by ``service.objects``)."""

from __future__ import annotations

import functools
import itertools
import os
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np

from oh_my_slam.core.images import size_at_max_side
from oh_my_slam.core.types import Intrinsics
from oh_my_slam.mapping import register_ma, sigmas, standpoint
from oh_my_slam.mapping.anchor import Sim3
from oh_my_slam.mapping.anchored import (
    TOO_CLOSE_M,
    AnchoredMap,
    FrameIn,
    Keyframe,
    Placement,
    features,
    natural_key,
    pose_residual,
    rot_deg,
)
from oh_my_slam.mapping.anchored_objects import ObjectMap, fuse_objects
from oh_my_slam.mapping.quality import (
    displacement_quality,
    displacements,
    gate,
    reregistered,
    residual_stats,
    too_close,
)
from oh_my_slam.service.backend import MONO_GRID_SIDE, Mono
from oh_my_slam.service.store import POSE_STATES, clone_tree
from oh_my_slam.service.timings import fusion_class

if TYPE_CHECKING:
    from pathlib import Path

    from oh_my_slam.service.jobs import Job
    from oh_my_slam.service.update import Updater, UpdateWork, _Frame

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
                   progress: Callable[[str], None], pairs: standpoint.Pairs | None = None
                   ) -> tuple[list[Placement], float, float]:
    """The camera-only frames by MapAnything (``register_ma``) against the map's confirmed
    keyframes, each registered one added as a keyframe; its depth scale is the map's, and its
    camera height feeds the map's depth scale (``ScaleState``) as a monocular keyframe's does.
    Order-independent (map.md §2): the frames go in canonical order (name, camera), and the ones
    no reference confirmed (``no_overlap``) are tried again against what the others added, until
    a pass registers nothing. Returns the placements (frames' order) and the registration and
    fusion seconds."""
    got: dict[int, Placement] = {}
    reg_s = fuse_s = 0.0
    pairs = pairs or standpoint.Pairs()
    todo = sorted(frames, key=lambda fr: (natural_key(fr.work.name), fr.work.index))
    passes = itertools.count()
    while True:
        r, f_s = _register_none_pass(up, am, todo, workdir / f"ma{next(passes)}", pairs, got)
        reg_s, fuse_s = reg_s + r, fuse_s + f_s
        again = [fr for fr in todo if got[fr.work.index].reason == "no_overlap"]
        if len(again) == len(todo) or not again:
            break
        todo = again
    for fr in frames:
        pl = got[fr.work.index]
        progress(f"{fr.work.name}: {pl.status} ({pl.reason or pl.method})")
    return [got[fr.work.index] for fr in frames], reg_s, fuse_s


def _register_none_pass(up: Updater, am: AnchoredMap, frames: list[_Frame], workdir: Path,
                        pairs: standpoint.Pairs, got: dict[int, Placement]) -> tuple[float, float]:
    """One pass of ``_register_none`` over ``frames``, their placements into ``got``."""
    t = time.perf_counter()
    am.save()  # this update's keyframes get their image and depth on disk: anchors
    stored = [register_ma.Stored(k.id, img, k.K, k.T, i, functools.partial(am.depth_of, k.id))
              for i, k in enumerate(am.keyframes)
              if k.state == "fused" for img in [am.image_path(k.id)] if img is not None]
    if not stored:  # nothing confirmed to register against (a camera-only map without origin)
        for fr in frames:
            got[fr.work.index] = Placement(fr.work.name, "rejected", "no_overlap")
        return time.perf_counter() - t, 0.0
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
    # frames of one standpoint: a pure rotation about its centre (map.md §1)
    feats = {}
    for fr in frames:
        assert fr.rgb is not None
        feats[fr.work.index] = features(fr.rgb, fr.ignore)
    turned = standpoint.standpoints(
        am, [standpoint.NewFrame(fr.work.name, fr.cam.canonical(), fr.work.index, _has_depth(fr),
                                 p.T, feats[fr.work.index]) for fr in frames
             for p in [res.placements[fr.work.index]]
             if p.status == "registered" and p.T is not None], pairs)
    reg_s = time.perf_counter() - t
    t = time.perf_counter()
    for fr in frames:
        p = res.placements[fr.work.index]
        scene = fr.scene_m if fr.scene_m is not None else p.scene_m
        if too_close(scene, p.inliers):
            got[fr.work.index] = Placement(fr.work.name, "rejected", "too_close",
                                           inliers=p.inliers, scene_m=scene)
            continue
        if p.status == "rejected":
            got[fr.work.index] = Placement(fr.work.name, "rejected", p.reason)
            continue
        assert p.T is not None and p.depth is not None and p.K_depth is not None
        T, sp, sr = turned.get(fr.work.index) or (p.T, *sigmas.calibrated(
            p.sigma_pos_m or 0.0, p.sigma_rot_deg or 0.0, _has_depth(fr), None))
        dup = am.duplicate_of(fr.work.camera_id, T, feats[fr.work.index])
        if dup is not None:  # within the duplicate limits of a confirmed keyframe (api.md §4)
            got[fr.work.index] = Placement(fr.work.name, "rejected", "duplicate",
                                           inliers=p.inliers, duplicate_of=dup)
            continue
        # its sigmas calibrated as every image pose's (``mapping.sigmas``; no prior); its
        # displacement (map.md §1): the committed pose against MapAnything's image-only one, 0
        # unless a standpoint turned it
        pl = Placement(fr.work.name, "registered", None, T, round(sp, 4), round(sr, 3),
                       p.inliers, "rotation" if fr.work.index in turned else "mapanything",
                       align_T=p.T, map_residual=(0.0, round(rot_deg(T[:3, :3].T @ p.T[:3, :3]),
                                                             3)))
        f = _frame_in(fr, None)
        h, w = p.depth.shape
        Kg = Intrinsics(float(p.K_depth[0, 0]), float(p.K_depth[1, 1]), float(p.K_depth[0, 2]),
                        float(p.K_depth[1, 2]), w, h)
        depth = p.depth
        kind = "sensor" if _has_depth(fr) else "mono"
        if kind == "sensor":  # its depth image's residual against the map (scale_sigma_rel)
            am.depth_residual(f, T, depth, Kg)
        if kind == "mono":
            am.add_height_evidence(f, T, depth, Kg)
            k = am.scale.k
            am.scale.estimate(am.prior_span(), am.prior_span("drifting"))
            depth = (depth * (am.scale.k / k)).astype(np.float32)
        assert fr.rgb is not None
        am.add(f, feats[fr.work.index], pl, depth, Kg, kind, None)
        got[fr.work.index] = pl
    return reg_s, time.perf_counter() - t


def frame_class(fr: _Frame) -> str:
    """A frame's fusion class (``timings.fusion_class``): its prior's trust, its depth kind."""
    return fusion_class(_trusted(fr), _has_depth(fr))


def _fusion_by_class(up: Updater, geo: list[_Frame], late: list[_Frame],
                     placements: dict[int, Placement], fused: dict[str, float], shared: float
                     ) -> None:
    """The fusion seconds per kept frame by fusion class (``Timings``): each engine's fusion
    over its kept frames, plus the objects and the save (``shared``) over every kept frame."""
    kept = {g: [fr for fr in frs if placements[fr.work.index].status != "rejected"]
            for g, frs in (("geo", geo), ("late", late))}
    total = sum(len(v) for v in kept.values())
    per: dict[str, list[float]] = {}
    for g, frs in kept.items():
        for fr in frs:
            per.setdefault(frame_class(fr), []).append(fused[g] / len(frs) + shared / total)
    for cls, v in per.items():
        up.timings.add(f"fusion_per_frame_{cls}", float(np.mean(v)), len(v))


def classes(w: UpdateWork) -> dict[str, int]:
    """The request's frames by fusion class (the job's ETA, ``Timings.predict_phases``)."""
    out: dict[str, int] = {}
    for f in w.frames:
        c = fusion_class(f.trust in ("metric", "drifting"), f.depth is not None)
        out[c] = out.get(c, 0) + 1
    return out


def detection_grid(K: Intrinsics) -> tuple[int, int]:
    """The grid (width, height) a frame's detections are asked on: its monocular depth grid."""
    return size_at_max_side(K.width, K.height, MONO_GRID_SIDE)


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
                   check_close=fr.deferred, unconfirmed=fr.unconfirmed)


def _depth_kind(am: AnchoredMap | None, pl: Placement) -> str:
    """The depth kind (``sensor``, ``mono``, ``none``) of a placed frame's keyframe."""
    k = None if am is None or pl.keyframe is None else am.keyframe(pl.keyframe)
    return "none" if k is None else k.depth_kind


def _contributes(am: AnchoredMap | None, pl: Placement) -> bool:
    """Whether a placed frame's keyframe carries content (``Keyframe.contributes``)."""
    k = None if am is None or pl.keyframe is None else am.keyframe(pl.keyframe)
    return k is not None and k.contributes()


def _height_off(am: AnchoredMap | None, pl: Placement) -> bool:
    """Whether the height gate left a placed frame's keyframe out of content (map.md §2)."""
    k = None if am is None or pl.keyframe is None else am.keyframe(pl.keyframe)
    return k is not None and not k.height_ok


def _has_depth(fr: _Frame) -> bool:
    return fr.depth_m is not None and bool(np.count_nonzero(fr.depth_m > 0))


def _input(fr: _Frame, pl: Placement | None) -> dict[str, Any]:
    """The ``InputStatus`` of a frame: rejected by the checks (``pl`` None) or as placed."""
    if pl is None:  # rejected by the checks
        return input_row(fr.work.name, "rejected", fr.reason, 0, None, None, None, None,
                         fr.scene_m, fr.duplicate_of)
    res = pl.prior_residual if fr.work.trust in ("metric", "drifting") else None
    return input_row(fr.work.name, pl.status, pl.reason, pl.inliers, pl.T, pl.sigma_pos_m,
                     pl.sigma_rot_deg, res, pl.scene_m, pl.duplicate_of, pl.detail)


def input_row(name: str, status: str, reason: str | None, inliers: int, T: Any,
              sigma_pos_m: float | None, sigma_rot_deg: float | None,
              residual: tuple[float, float] | None, scene_m: float | None,
              duplicate_of: str | None = None, detail: str | None = None) -> dict[str, Any]:
    """An ``InputStatus`` (api.md §4): pose and sigmas null when rejected, the prior residuals
    null without a prior, with reason ``too_close`` the distance limit and the frame's median
    scene depth, with reason ``duplicate`` the frame it duplicates, and ``detail`` (why the
    images did not confirm a kept frame) only on a ``kept_at_prior`` frame not kept for a
    failed check."""
    kept = status != "rejected"
    row = {"name": name, "status": status, "reason": reason, "inliers": inliers,
           "T_map_cam": np.asarray(T).round(6).tolist() if kept and T is not None else None,
           "sigma_pos_m": sigma_pos_m if kept else None,
           "sigma_rot_deg": sigma_rot_deg if kept else None,
           "prior_residual_m": None if residual is None else round(residual[0], 4),
           "prior_residual_deg": None if residual is None else round(residual[1], 3),
           "detail": detail if status == "kept_at_prior" else None}
    if reason == "too_close":
        row["min_distance_m"] = TOO_CLOSE_M
        row["scene_distance_m"] = round(scene_m if scene_m is not None else 0.0, 3)
    if reason == "duplicate":
        row["duplicate_of"] = duplicate_of or name
    return row


def prior_frames_rejected(frames: list[_Frame], status: dict[int, str]) -> bool:
    """Whether every prior-carrying frame of a map's first update is rejected (map.md §1):
    ``status`` holds each frame's status by index (absent: rejected by the checks)."""
    trusted = [fr for fr in frames if fr.work.trust in ("metric", "drifting")]
    return bool(trusted) and all(status.get(fr.work.index, "rejected") == "rejected"
                                 for fr in trusted)


def run_anchored(up: Updater, job: Job, w: UpdateWork, frames: list[_Frame],
                 prev: dict[str, Any] | None, workdir: Path,
                 changed: Callable[[], None] = lambda: None) -> None:
    tm = job.timings_s
    todo = [fr for fr in frames if fr.reason is None and fr.path is not None]
    placements: dict[int, Placement] = {}
    am: AnchoredMap | None = None
    stored: list[Keyframe] = []
    k_prev: float | None = None
    mapper = workdir / "anchored"
    om: ObjectMap | None = None
    n_objects = 0
    pairs = standpoint.Pairs()  # the two-view fits of this update (standpoints, the hard limit)
    rotations_bad = False
    if todo:
        t = time.perf_counter()
        if prev is not None:
            clone_tree(up.store.mapper_dir(w.map), mapper)
            am = AnchoredMap.load(mapper)
            k_prev = am.scale.k
            stored = list(am.keyframes)
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

        # a frame without a prior registered by the prior-first engine joins a standpoint
        am.standpoint = functools.partial(standpoint.join_standpoint, am, pairs)
        try:
            pls = am.update(ins, progress) if ins else []
            reg_s, fuse_s = am.timings.get("register", 0.0), am.timings.get("fuse", 0.0)
            for fr, pl in zip(geo, pls, strict=True):
                placements[fr.work.index] = pl
            if geo:
                up.timings.add("registration_per_frame", reg_s / len(geo), len(geo))
            fused = {"geo": fuse_s, "late": 0.0}  # each engine's fusion seconds
            if late:  # their scene distance is MapAnything's depth (``register_ma``)
                lp, r, fu = _register_none(up, am, late, workdir, progress, pairs)
                up.timings.add("registration_mapanything_per_frame", r / len(late), len(late))
                reg_s, fuse_s = reg_s + r, fuse_s + fu
                fused["late"] = fu
                for fr, pl in zip(late, lp, strict=True):
                    placements[fr.work.index] = pl
                # the stored kept keyframes these frames overlap too (map.md §2)
                news = [am.keyframe(pl.keyframe) for pl in lp
                        if pl.status == "registered" and pl.keyframe is not None]
                am.reregistered += am.reregister([k for k in stored if not k.usable()],
                                                 [k for k in news if k is not None])
            if not any(_trusted(fr) for fr in frames):
                # an update without priors: committed relative rotations against the images'
                # (map.md §4 rotation_inconsistent; registered frames only)
                t = time.perf_counter()
                rotations_bad = standpoint.rotation_check(am, list(placements.values()), pairs)
                reg_s += time.perf_counter() - t
            t = time.perf_counter()
            version = (prev["version"] + 1) if prev else 1
            again = []  # the re-registered keyframes' objects: their stored image's detections
            for e in am.reregistered:  # confirmed from their stored image: it exists
                if e.get("content"):  # the too-close exception: its detections are in already
                    continue
                kf, path = am.keyframe(e["keyframe"]), am.image_path(e["keyframe"])
                assert kf is not None and path is not None
                again.append((kf, pool.submit(up.backend.detect, path, detection_grid(kf.K))))
            if am.keyframes:  # else no origin: every frame with a prior was rejected
                om, n_objects = fuse_objects(am, [(placements[fr.work.index], fr.ignore,
                                                   fr.path, dets[fr.work.index]) for fr in todo],
                                             version, up.backend.look_closer, again)
            tm["objects"] = round(time.perf_counter() - t, 3)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        t = time.perf_counter()
        am.save()
        if om is not None:
            om.save()
        tm["register"] = round(reg_s, 3)
        tm["fusion"] = round(fuse_s + tm["objects"] + time.perf_counter() - t, 3)
        n_kept = sum(pl.status != "rejected" for pl in placements.values())
        if n_kept:
            up.timings.add("fusion_per_frame", tm["fusion"] / n_kept, n_kept)
            _fusion_by_class(up, geo, late, placements, fused, tm["fusion"] - fuse_s)
    job.progress = 0.95
    changed()
    t = time.perf_counter()
    job.inputs = [_input(fr, placements.get(fr.work.index)) for fr in frames]
    kept = [(fr, placements[fr.work.index]) for fr in todo
            if placements[fr.work.index].status != "rejected"]
    version = (prev["version"] + 1) if prev else 1
    # the stored frames at the poses this version would commit (their keyframes'), and how far
    # each moved from the previous version: the gauge shift and ``previous_residuals``
    by_id = {k.id: k for k in am.keyframes} if am is not None else {}
    again_ids = {e["keyframe"] for e in am.reregistered} if am is not None else set()
    rows, shifts = [], []
    for r in prev["frames"] if prev else []:
        k = by_id.get(r.get("internal") or "")
        if k is not None and k.id in again_ids:
            # confirmed by this job's images (map.md §2): registered now; the movement of its
            # confirmation is in ``reregistered`` only, not in the gauge shift or the residuals
            rows.append({**r, "T_map_cam": k.T.tolist(), "sigma_pos_m": k.sigma_pos_m,
                         "sigma_rot_deg": k.sigma_rot_deg, "pose_state": "fused",
                         "inliers": k.inliers, "method": k.method})
            continue
        row = r if k is None else {**r, "T_map_cam": k.T.tolist()}
        shifts.append(pose_residual(np.asarray(row["T_map_cam"]), np.asarray(r["T_map_cam"])))
        rows.append(row)
    for fr, pl in kept:
        assert pl.T is not None
        p = fr.work.prior
        rows.append({"name": fr.work.name, "camera_id": fr.work.camera_id,
                     "internal": pl.keyframe, "T_map_cam": pl.T.tolist(),
                     "sigma_pos_m": pl.sigma_pos_m, "sigma_rot_deg": pl.sigma_rot_deg,
                     "pose_state": "fused" if pl.status == "registered" else "kept_at_prior",
                     "added_version": version, "sha256": fr.sha, "inliers": pl.inliers,
                     "method": pl.method, "depth_kind": _depth_kind(am, pl),
                     "prior": p if p is not None and p["trust"] == "metric" else None,
                     # the too-close exception (api.md §4): a kept frame whose in-range depth
                     # feeds occupancy and heights (``derived``)
                     "content": _contributes(am, pl),
                     # the height gate (map.md §2): a reference, but no content
                     "height_inconsistent": _height_off(am, pl)})
    prior_res = [pose_residual(np.asarray(r["T_map_cam"]), np.asarray(r["prior"]["T_map_cam"]))
                 for r in rows if r.get("prior")]
    source, sigma = am.scale_summary() if am is not None else ("monocular", 0.1)
    ratio = None
    if prev is not None and am is not None:
        # the monocular keyframes' depth follows k; positions and depth images do not move
        mono = any(k.depth_kind == "mono" for k in am.keyframes)
        ratio = round(am.scale.k / k_prev, 5) if mono and k_prev else 1.0
    # registered inputs whose pose sigma exceeds the threshold (map.md §4)
    registered = [pl for _, pl in kept if pl.status == "registered"]
    low = sum((pl.sigma_pos_m or 0.0) > LOW_CONFIDENCE_SIGMA_M for pl in registered)
    gauge = None
    if prev is not None:
        gauge = {"translation_m": round(float(np.median([m for m, _ in shifts])), 4)
                 if shifts else 0.0,
                 "rotation_deg": round(float(np.median([d for _, d in shifts])), 3)
                 if shifts else 0.0, "scale_ratio": 1.0}
    # the new frames against their image-only poses (map.md §1, ``max_displacement_*``)
    moved = displacements([pl for _, pl in kept])
    q: dict[str, Any] = {
        "scale_ratio_vs_prev": ratio,
        "gauge_shift": gauge,
        **displacement_quality(moved),
        "prior_disagreement_deg": 0.0 if am is None else am.prior_disagreement_deg,
        "reregistered": [] if am is None else reregistered(am.reregistered),
        "height_inconsistent": [] if am is None else list(am.height_inconsistent),
        "prior_residuals": residual_stats(prior_res),
        "previous_residuals": residual_stats(shifts) if prev else None,
        "rejected_share": round(1.0 - len(kept) / len(frames), 4),
        "low_confidence_share": round(low / len(registered), 4) if registered else 0.0,
        "rebuilt": False,
        "prior_gravity_deg": None if prev is not None or am is None else am.up_check_deg,
    }
    status = {i: pl.status for i, pl in placements.items()}
    job.quality = gate(q, bool(kept), w.accept_if,
                       prev is None and prior_frames_rejected(frames, status), rotations_bad)
    if am is not None:
        job.scale_source, job.scale_sigma_rel = source, round(sigma, 5)
    if not job.quality["accepted"]:
        job.state, job.progress = "rejected", 1.0
        return
    assert am is not None and om is not None
    floor, floor_sigma, pts = om.floor_z, om.floor_sigma, am.bounds()
    record = {"kind": "update", "engine": "anchored", "job_id": job.id,
              "sim3": Sim3().to_dict(), "scale_source": source,
              "scale_sigma_rel": round(sigma, 5), "frames": rows, "objects": n_objects,
              "floor_z_m": None if floor is None else round(floor, 4),
              "floor_sigma_m": round(floor_sigma, 4), "pose_states": POSE_STATES,
              "bounds_m": {"min": pts.min(0).round(4).tolist(),
                           "max": pts.max(0).round(4).tolist()},
              "points": len(am.cloud.keys)}
    job.result_version = up.store.commit(w.map, record, mapper,
                                         {"objects.json": om.records()}, w.cameras, w.masks)
    job.state, job.progress = "committed", 1.0
    tm["commit"] = round(time.perf_counter() - t, 3)
