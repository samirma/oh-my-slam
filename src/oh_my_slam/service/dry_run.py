"""A dry run of an update (O10, api.md §4): ``POST /api/maps/{name}/updates?dry_run=1``.

It runs synchronously and writes nothing (no camera, version, job or mapper file):

* the per-frame checks of an update (``Updater._check``), with the monocular median depth of the
  first ``DRY_DEPTH_FRAMES`` suspicious frames (``too_close``);
* per kept frame, its link to the map: on a map the prior-first engine registers (an anchored
  map, or a new map while ``Updater.prior_first`` holds), the job's own registration
  (``AnchoredMap.update``: the match graph, the drift fit, the distance limit, pose duplicates,
  re-registration and the height gate) on the snapshot of the current version, its monocular
  depth asked only when a frame needs it, so the predicted statuses are the update's (map.md
  §2); the snapshot is discarded. ``map_inliers`` are its inliers. Frames without a prior that
  the update would give MapAnything (``prior_first.none_engine``) are predicted by the same
  geometric registration. On a map of the mapper, null;
* its links to its neighbours in the request (the previous and the next frame): the inliers of a
  fundamental matrix (RANSAC) on their ratio-test SIFT matches;
* the predicted status (``registered``, ``kept_at_prior`` or ``rejected``) and reason, with
  ``too_close`` its distances and with ``duplicate`` the frame it duplicates (a predicted pose
  within the duplicate limits of a stored confirmed keyframe or of an earlier predicted
  registered frame of its camera), the predicted quality report with its gate (``accept_if``)
  and the stored kept frames the update would re-register (``reregistered``: the update's test,
  ``AnchoredMap.reregister``, on the read-only copy with the predicted frames added at their
  predicted poses, with their depth images when they have one), the predicted scale and the
  predicted durations from the measured per-frame timings.

Its inference requests (monocular depth, gravity) are sent as interactive
(``Backend.interactive``): the inference server serves them before a job's fusion requests
(inference.md, single worker)."""

from __future__ import annotations

import contextlib
import copy
import functools
import math
import shutil
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from oh_my_slam.mapping import checks, standpoint
from oh_my_slam.mapping.anchored import DEFAULT_HEIGHT_SIGMA_M as HEIGHT_SIGMA_M
from oh_my_slam.mapping.anchored import (
    DEPTH_SIGMA_REL,
    MIN_INLIERS,
    MIN_SCALE_SIGMA_REL,
    MONOCULAR_SIGMA_REL,
    PRIOR_SPAN_M,
    TOO_CLOSE_M,
    AnchoredMap,
    Features,
    Placement,
    _prior_sigmas,
    duplicate_among,
    features,
    match,
    natural_key,
    pose_residual,
)
from oh_my_slam.mapping.quality import (
    displacement_quality,
    displacements,
    gate,
    reregistered,
    residual_stats,
    too_close,
)
from oh_my_slam.segmentation.detect import vocabulary
from oh_my_slam.service.prior_first import (
    LOW_CONFIDENCE_SIGMA_M,
    _canonical,
    _frame_in,
    _has_depth,
    frame_class,
    job_drift_correction,
    none_engine,
)

if TYPE_CHECKING:
    from pathlib import Path

    from oh_my_slam.service.update import Updater, UpdateWork, _Frame

NEIGHBOUR_PX = 3.0  # the fundamental matrix's RANSAC threshold
DRY_THREADS = 6
DRY_DEPTH_FRAMES = 4


def link(a: Features, b: Features) -> int:
    """Geometrically verified matches between two frames' features."""
    ia, ib = match(a.desc, b.desc)
    if len(ia) < 8:
        return 0
    _, mask = cv2.findFundamentalMat(a.kps[ia].astype(np.float64), b.kps[ib].astype(np.float64),
                                     cv2.FM_RANSAC, NEIGHBOUR_PX, 0.999)
    return 0 if mask is None else int(mask.sum())


def dry_run(up: Updater, w: UpdateWork) -> dict[str, Any]:
    """The ``DryRun`` body of an update request."""
    up = copy.copy(up)
    up.backend = up.backend.interactive()  # served ahead of fusion (inference.md)
    store = up.store
    workdir = store.work / f"dry-{uuid.uuid4().hex}"
    indir = workdir / "in"
    indir.mkdir(parents=True, exist_ok=True)
    holds = contextlib.ExitStack()  # the version it answers from (map.md §2 Holds)
    try:
        # the latest committed version and its mapper map, as one: a job may commit meanwhile
        prev = store.snapshot(w.map, workdir / "mapper", holds)
        frames, masks = up._frames(w)
        up._check(frames, masks, prev, indir)
        engine = up._is_anchored(prev, frames)
        if not engine:
            # the monocular median depth of at most DRY_DEPTH_FRAMES suspicious frames
            # (too_close): a dry run of the mapper stays a few seconds; the others are predicted
            # by their other checks. The engine measures them itself, as the job's does
            up._resolve_deferred([fr for fr in frames if fr.deferred][:DRY_DEPTH_FRAMES])
        with ThreadPoolExecutor(DRY_THREADS) as pool:
            got = list(pool.map(lambda fr: None if fr.rgb is None else features(fr.rgb, fr.ignore),
                                frames))
            feats: dict[int, Features] = {fr.work.index: g for fr, g in zip(frames, got,
                                                                           strict=True)
                                          if g is not None}
            links = list(pool.map(lambda k: _neighbours(k, frames, feats), range(len(frames))))
        am, placed, pairs = _engine(up, frames, prev, workdir) if engine else (None, {}, None)
        out, kept, prior_res = [], [], []
        first_kept = True
        rows = []
        for k, fr in enumerate(frames):
            row, pl = _frame(fr, links[k], placed.get(fr.work.index), engine,
                             prev is None, first_kept)
            first_kept = first_kept and row["predicted_status"] == "rejected"
            rows.append((row, pl, _predicted_pose(fr, pl, engine, prev is None)))
        # without the engine, one pose-duplicate pass over the predicted registered frames in
        # canonical name order (api.md §4; the engine's own pass is ``AnchoredMap.end_job``):
        # name, camera, pose, features of the duplicate's candidates
        earlier: list[tuple[str, str, Any, Callable[[], Features | None]]] = []
        for k in sorted(range(len(frames)) if not engine else [],
                        key=lambda j: (natural_key(frames[j].work.name),
                                       frames[j].work.camera_id, j)):
            fr, (row, pl, T) = frames[k], rows[k]
            if row["predicted_status"] != "registered" or T is None:
                continue
            dup = duplicate_among(fr.work.camera_id, T, feats.get(fr.work.index), earlier)
            if dup is not None:
                row.update(predicted_status="rejected", predicted_reason="duplicate",
                           duplicate_of=dup)
            else:
                earlier.append((fr.work.name, fr.work.camera_id, T,
                                functools.partial(feats.get, fr.work.index)))
        for fr, (row, pl, _) in zip(frames, rows, strict=True):
            if row["predicted_status"] != "rejected":
                kept.append((row["predicted_status"], pl))
                if pl is not None and pl.T is not None and fr.work.trust == "metric":
                    assert fr.work.prior is not None
                    prior_res.append(pose_residual(pl.T, np.asarray(fr.work.prior["T_map_cam"])))
            out.append(row)
        # a new anchored map's: the origin's gravity, as the job measures it (null without a
        # frame with a prior: a camera-only map is levelled by that gravity)
        gravity = am.up_check_deg if prev is None and am is not None \
            and am.created_with in ("metric", "drifting") else None
        disagreement = 0.0 if am is None else am.prior_disagreement_deg
        heights = [] if am is None else list(am.height_inconsistent)
        floors = [] if am is None else list(am.floor_inconsistent)
        source, sigma = _scale(prev, am, frames)
        # an update without priors: its predicted poses against the images' relative rotations
        # (map.md §4 rotation_inconsistent), as the job measures them
        rotations_bad, rotation_pairs = standpoint.rotation_check(
            am, list(placed.values()), pairs) if am is not None and pairs is not None \
            and not any(fr.work.trust in ("metric", "drifting") for fr in frames) else (False, [])
        again = [] if am is None else reregistered(am.reregistered)
        # the estimate the job would place its drifting priors with (map.md §4, oms/2.5)
        drift = job_drift_correction(am, frames) if engine else _mapper_drift(up, frames, prev)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        holds.close()  # what follows reads no stored file
    registered = [pl for status, pl in kept if status == "registered"]
    low = sum(pl is not None and (pl.sigma_pos_m or 0.0) > LOW_CONFIDENCE_SIGMA_M
              for pl in registered)
    # stored frames do not move (no gauge shift, no previous residual); the registered frames
    # against their image-only poses, as the update measures them (``prior_first``)
    moved = displacements([pl for pl in registered if pl is not None])
    q: dict[str, Any] = {
        "scale_ratio_vs_prev": None if prev is None else 1.0,
        "gauge_shift": None if prev is None else {"translation_m": 0.0, "rotation_deg": 0.0,
                                                  "scale_ratio": 1.0},
        **displacement_quality(moved),
        "prior_disagreement_deg": disagreement,
        "reregistered": again,
        "height_inconsistent": heights,
        "floor_inconsistent": floors,
        "rotation_pairs": rotation_pairs,
        "drift_correction": drift,
        "prior_residuals": residual_stats(prior_res),
        "previous_residuals": residual_stats([(0.0, 0.0)] * len(prev["frames"])) if prev
        else None,
        "rejected_share": round(1.0 - len(kept) / len(frames), 4),
        "low_confidence_share": round(low / len(registered), 4) if registered else 0.0,
        "rebuilt": False, "prior_gravity_deg": gravity,
    }
    reached = [fr for fr in frames if fr.reason is None]
    # the frames a job hands MapAnything (map.md §2 step 8): monocular ones without a prior
    ma = len([fr for fr in reached if fr.work.trust not in ("metric", "drifting")
              and not _has_depth(fr)]) \
        if none_engine() == "mapanything" and (prev is None or am is not None) else 0
    reg, fus = up.timings.predict_phases(
        len(frames), len(reached) - ma, ma, len(kept),
        _classes([fr for fr, r in zip(frames, out, strict=True)
                  if r["predicted_status"] != "rejected"]),
        0 if prev is None else len(prev["frames"]))
    trusted = [r for fr, r in zip(frames, out, strict=True)
               if fr.work.trust in ("metric", "drifting")]
    return {"map": w.map,
            "base_version": None if prev is None else prev["version"], "frames": out,
            "quality": gate(q, bool(kept), w.accept_if, prev is None and bool(trusted) and all(
                r["predicted_status"] == "rejected" for r in trusted), rotations_bad),
            "scale_source": source, "scale_sigma_rel": round(sigma, 5),
            "predicted_s": {"registration": round(reg, 3), "fusion": round(fus, 3)},
            # the vocabulary detection would use (map.md §5): the defaults and the map's with
            # the request's labels
            "labels": vocabulary(_map_labels(prev, w.labels))}


def _predicted_pose(fr: _Frame, pl: Placement | None, anchored: bool, new_map: bool
                    ) -> Any:
    """A frame's predicted registered pose, for the pose-duplicate test (api.md §4): its
    placement on an anchored map; on a map's first update, the prior of a frame with a metric or
    drifting prior (the priors' frame is the map frame, and the origin's registration keeps the
    prior); None otherwise (a camera-only first update, a map of the older mapper)."""
    if anchored:
        return None if pl is None else pl.T
    if new_map and fr.work.trust in ("metric", "drifting") and fr.work.prior is not None:
        return np.asarray(fr.work.prior["T_map_cam"], np.float64)
    return None


def _classes(frames: list[_Frame]) -> dict[str, int]:
    """The predicted kept frames by fusion class (``prior_first.frame_class``)."""
    out: dict[str, int] = {}
    for fr in frames:
        c = frame_class(fr)
        out[c] = out.get(c, 0) + 1
    return out


def _scale(prev: dict[str, Any] | None, am: AnchoredMap | None, frames: list[_Frame]
           ) -> tuple[str, float]:
    """The predicted ``scale_source`` and ``scale_sigma_rel`` (map.md §1 order): an anchored
    map's own (``AnchoredMap.scale_summary``), a mapper map's version's; on a new map, metric
    priors spanning ``PRIOR_SPAN_M`` (as the engine judges them), else depth images, else
    camera heights (their relative sigma), else the monocular model."""
    if am is not None:
        return am.scale_summary()
    if prev is not None:
        return str(prev["scale_source"]), float(prev["scale_sigma_rel"])
    kept = [fr for fr in frames if fr.reason is None]
    metric = [fr.work.prior for fr in kept if fr.work.trust == "metric"]
    if len(metric) > 1:
        c = np.array([np.asarray(p["T_map_cam"])[:3, 3] for p in metric if p is not None])
        span = float(np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2).max())
        if span >= PRIOR_SPAN_M:
            sp = float(np.median([_prior_sigmas(p)[0] for p in metric if p is not None]))
            return "priors", max(math.sqrt(2.0) * sp / span, MIN_SCALE_SIGMA_REL)
    if any(fr.depth_m is not None and bool(np.count_nonzero(fr.depth_m > 0)) for fr in kept):
        return "depth", DEPTH_SIGMA_REL
    rel = []
    for fr in kept:
        h, sig = (fr.work.height_m, fr.work.height_sigma_m) if fr.work.height_m is not None \
            else (fr.cam.height_above_floor_m, fr.cam.height_sigma_m)
        if h:
            rel.append((sig or HEIGHT_SIGMA_M) / h)
    if rel:
        return "camera_height", max(float(np.median(rel)) / math.sqrt(len(rel)),
                                    MIN_SCALE_SIGMA_REL)
    return "monocular", MONOCULAR_SIGMA_REL


def _mapper_drift(up: Updater, frames: list[_Frame], prev: dict[str, Any] | None
                  ) -> dict[str, Any] | None:
    """``quality.drift_correction`` the older mapper would report (``mapper_drift_correction``,
    the previous version's drift estimate)."""
    from oh_my_slam.service.update import mapper_drift_correction

    return mapper_drift_correction(frames, up._previous_drift(prev)[0])


def _engine(up: Updater, frames: list[_Frame], prev: dict[str, Any] | None, workdir: Path
            ) -> tuple[AnchoredMap, dict[int, Placement], standpoint.Pairs]:
    """The job's registration on the snapshot (``AnchoredMap.update``, every frame the checks
    keep, in the geometric engine): the map as the job would leave it (discarded), each frame's
    placement by request index, and the update's two-view fits (the rotation hard limit)."""
    root = workdir / "mapper"
    am = AnchoredMap.load(root) if prev is not None else AnchoredMap(workdir / "new-map")
    am.root.mkdir(parents=True, exist_ok=True)
    pairs = standpoint.Pairs(dict(am.pair_cache))  # the map's stored pairs, never written
    am.standpoint = functools.partial(standpoint.join_standpoint, am, pairs)
    todo = [fr for fr in frames if fr.reason is None and fr.path is not None]
    # a new map's origin as the job takes it (``prior_first.run_anchored``): the first frame
    # with a prior in canonical order, else the first frame (map.md §1, §2)
    origin = min([fr for fr in todo if fr.work.trust in ("metric", "drifting")] or todo,
                 key=_canonical) if prev is None and todo else None
    ins = [_frame_in_lazy(up, fr, fr is origin) for fr in todo]
    pls = am.update(ins, lambda _: None) if ins else []
    return am, {fr.work.index: pl for fr, pl in zip(todo, pls, strict=True)}, pairs


def _frame_in_lazy(up: Updater, fr: _Frame, gravity: bool) -> Any:
    """A frame as the engine takes it, its monocular depth asked only when the engine needs it
    (a frame without a depth image, or a new map's origin for its gravity)."""
    f = _frame_in(fr, None)
    if not _has_depth(fr) or gravity:
        assert fr.path is not None
        path, K = fr.path, fr.cam.canonical()
        f.mono = functools.cache(lambda: up.backend.infer(path, K, gravity).as_tuple())
    return f


def _neighbours(k: int, frames: list[_Frame], feats: dict[int, Features]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    own = feats.get(frames[k].work.index)
    for j in (k - 1, k + 1):
        if 0 <= j < len(frames) and own is not None and frames[j].work.index in feats:
            out.append({"name": frames[j].work.name,
                        "inliers": link(own, feats[frames[j].work.index])})
    return out


def _frame(fr: _Frame, neighbours: list[dict[str, Any]], pl: Placement | None, anchored: bool,
           new_map: bool, first: bool) -> tuple[dict[str, Any], Placement | None]:
    m = fr.measures
    status, reason, map_inliers = "registered", fr.reason, None
    if reason is not None:
        status = "rejected"
    elif anchored:  # the job's own registration (``_engine``): it applied the distance limit
        assert pl is not None
        status, reason = pl.status, pl.reason
        row = _row(fr, m, None if new_map else pl.inliers, neighbours, status, reason)
        row["too_close"] = row["too_close"] or (pl.scene_m is not None
                                                and pl.scene_m < TOO_CLOSE_M)
        if reason == "duplicate" and status == "rejected":
            row["duplicate_of"] = pl.duplicate_of or fr.work.name
        if reason == "too_close" and status != "registered":
            row["min_distance_m"] = TOO_CLOSE_M
            row["scene_distance_m"] = round(pl.scene_m or fr.scene_m or 0.0, 3)
        return row, pl
    elif fr.unconfirmed is not None:  # its image failed a check: kept at its prior
        status, reason = "kept_at_prior", fr.unconfirmed
    elif new_map and not first and fr.work.trust not in ("metric", "drifting") \
            and max([int(n["inliers"]) for n in neighbours] or [0]) < MIN_INLIERS:
        # a new map's frame without prior holds only by its links to the request's frames
        status, reason = "rejected", "no_overlap"
    close = too_close(fr.scene_m, 0)  # no registration predicted: no inliers
    if close and fr.reason is None and fr.work.trust in ("metric", "drifting"):
        # the distance limit (api.md §4): a frame with a prior is kept at it, its reason the
        # check its image failed, else too_close
        status, reason = "kept_at_prior", fr.unconfirmed or "too_close"
    elif close and status != "rejected":
        status, reason = "rejected", "too_close"
    row = _row(fr, m, map_inliers, neighbours, status, reason)
    if reason == "too_close" and status != "registered":
        row["min_distance_m"] = TOO_CLOSE_M
        row["scene_distance_m"] = round(fr.scene_m or 0.0, 3)
    if reason == "duplicate" and status == "rejected":
        row["duplicate_of"] = fr.duplicate_of or fr.work.name
    return row, pl


def _row(fr: _Frame, m: Any, map_inliers: int | None, neighbours: list[dict[str, Any]],
         status: str, reason: str | None) -> dict[str, Any]:
    return {"name": fr.work.name, "features": 0 if m is None else m.features,
            "map_inliers": map_inliers, "neighbour_inliers": neighbours,
            "plain": m is not None and m.texture < checks.PLAIN_TEXTURE,
            "too_close": fr.scene_m is not None and fr.scene_m < TOO_CLOSE_M,
            "predicted_status": status, "predicted_reason": None if status == "registered"
            else reason}


def _map_labels(prev: dict[str, Any] | None, labels: list[str]) -> list[str]:
    from oh_my_slam.service.update import map_labels

    return map_labels(prev, labels)
