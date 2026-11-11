"""A dry run of an update (O10, api.md §4): ``POST /api/maps/{name}/updates?dry_run=1``.

It runs synchronously and writes nothing (no camera, version, job or mapper file):

* the per-frame checks of an update (``Updater._check``), with the monocular median depth of the
  first ``DRY_DEPTH_FRAMES`` suspicious frames (``too_close``);
* per kept frame, its link to the map: on an anchored map the prior-first placement against the
  current version (``AnchoredMap.place`` on a copy loaded read-only; nothing is added, no scale
  evidence), whose inliers are ``map_inliers``; on a new map or a map of the mapper, null. The
  frames are placed ``DRY_THREADS`` at a time, each against the stored map only; a drifting
  prior's prediction follows the request (the odometry's path grows frame by frame, as in the
  update), and two strong conflicts that agree on one correction of the odometry re-anchor it,
  as the update does (``AnchoredMap._same_drift``): the conflicts that correction explains are
  predicted registered;
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

import copy
import functools
import itertools
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
    pose_residual,
    unit,
)
from oh_my_slam.mapping.quality import (
    displacement_quality,
    displacements,
    gate,
    reregistered,
    residual_stats,
    too_close,
)
from oh_my_slam.service.prior_first import (
    LOW_CONFIDENCE_SIGMA_M,
    _frame_in,
    frame_class,
    none_engine,
)

if TYPE_CHECKING:
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
    try:
        # the latest committed version and its mapper map, as one: a job may commit meanwhile
        prev = store.snapshot(w.map, workdir / "mapper")
        frames, masks = up._frames(w)
        up._check(frames, masks, prev, indir)
        # the monocular depth of at most DRY_DEPTH_FRAMES suspicious frames (too_close): a
        # dry run stays a few seconds; the others are predicted by their other checks
        up._resolve_deferred([fr for fr in frames if fr.deferred][:DRY_DEPTH_FRAMES])
        with ThreadPoolExecutor(DRY_THREADS) as pool:
            got = list(pool.map(lambda fr: None if fr.rgb is None else features(fr.rgb, fr.ignore),
                                frames))
            feats: dict[int, Features] = {fr.work.index: g for fr, g in zip(frames, got,
                                                                           strict=True)
                                          if g is not None}
            am = AnchoredMap.load(workdir / "mapper") \
                if prev is not None and prev.get("engine") == "anchored" else None
            placed = _place_all(pool, am, frames, feats)
            links = list(pool.map(lambda k: _neighbours(k, frames, feats), range(len(frames))))
        out, kept, prior_res = [], [], []
        first_kept = True
        # predicted registered: name, camera, pose, features (the pose duplicate's candidates)
        earlier: list[tuple[str, str, Any, Callable[[], Features | None]]] = []
        for k, fr in enumerate(frames):
            row, pl = _frame(fr, links[k], placed.get(fr.work.index), am is not None,
                             prev is None, first_kept)
            T = _predicted_pose(fr, pl, am is not None, prev is None)
            if row["predicted_status"] == "registered" and T is not None:
                own = feats.get(fr.work.index)
                dup = (None if am is None else am.duplicate_of(fr.work.camera_id, T, own)) \
                    or duplicate_among(fr.work.camera_id, T, own, earlier)
                if dup is not None:
                    row.update(predicted_status="rejected", predicted_reason="duplicate",
                               duplicate_of=dup)
                else:
                    earlier.append((fr.work.name, fr.work.camera_id, T,
                                    functools.partial(feats.get, fr.work.index)))
            if row["predicted_status"] != "rejected":
                first_kept = False
                kept.append((row["predicted_status"], pl))
                if pl is not None and pl.T is not None and fr.work.trust == "metric":
                    assert fr.work.prior is not None
                    prior_res.append(pose_residual(pl.T, np.asarray(fr.work.prior["T_map_cam"])))
            out.append(row)
        gravity = _prior_gravity(up, frames) if prev is None and up.prior_first else None
        disagreement = 0.0
        if am is not None:  # the registered frames with a prior against each other
            mine: list[Any] = []
            ins = {fr.work.index: fr for fr in frames}
            for idx, pl in placed.items():
                if pl.status == "registered":
                    am._checked(_frame_in(ins[idx], None), pl, mine, feats[idx])
            disagreement = am.prior_disagreement_deg
        source, sigma = _scale(prev, am, frames)
        # an update without priors: its predicted poses against the images' relative rotations
        # (map.md §4 rotation_inconsistent), as the job measures them
        rotations_bad = am is not None and not any(
            fr.work.trust in ("metric", "drifting") for fr in frames) and _rotations_bad(
            am, frames, feats, placed)
        again = [] if am is None else _reregistered(am, frames, feats, placed, out)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
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
        "prior_residuals": residual_stats(prior_res),
        "previous_residuals": residual_stats([(0.0, 0.0)] * len(prev["frames"])) if prev
        else None,
        "rejected_share": round(1.0 - len(kept) / len(frames), 4),
        "low_confidence_share": round(low / len(registered), 4) if registered else 0.0,
        "rebuilt": False, "prior_gravity_deg": gravity,
    }
    reached = [fr for fr in frames if fr.reason is None]
    ma = len([fr for fr in reached if fr.work.trust not in ("metric", "drifting")]) \
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
            "predicted_s": {"registration": round(reg, 3), "fusion": round(fus, 3)}}


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


def _rotations_bad(am: AnchoredMap, frames: list[_Frame], feats: dict[int, Features],
                   placed: dict[int, Placement]) -> bool:
    """The predicted ``rotation_inconsistent`` (``standpoint.consistency``) of the frames
    predicted registered, at their predicted poses, with the stored keyframes they neighbour."""
    new = [(fr, pl) for fr in frames for pl in [placed.get(fr.work.index)]
           if pl is not None and pl.status == "registered" and pl.T is not None
           and fr.work.index in feats]
    views = standpoint.stored_views(am, [(pl.T[:3, :3], pl.T[:3, 3]) for _, pl in new  # type: ignore[index]
                              ])
    views += [standpoint.View(fr.work.name, fr.cam.canonical(), feats[fr.work.index],
                              pl.T[:3, :3].copy(), pl.T[:3, 3].copy(),  # type: ignore[index]
                              False, fr.work.index) for fr, pl in new]
    return standpoint.inconsistent(*standpoint.consistency(views, standpoint.Pairs()))


def _classes(frames: list[_Frame]) -> dict[str, int]:
    """The predicted kept frames by fusion class (``prior_first.frame_class``)."""
    out: dict[str, int] = {}
    for fr in frames:
        c = frame_class(fr)
        out[c] = out.get(c, 0) + 1
    return out


def _reregistered(am: AnchoredMap, frames: list[_Frame], feats: dict[int, Features],
                  placed: dict[int, Placement], rows: list[dict[str, Any]]
                  ) -> list[dict[str, Any]]:
    """The predicted ``quality.reregistered`` (map.md §2): the frames predicted registered are
    added to the read-only copy at their predicted poses (their depth image, else no depth),
    then the stored kept keyframes they overlap are placed again as the update does
    (``AnchoredMap.reregister``). Nothing is written."""
    kept = [k for k in am.keyframes if not k.usable()]
    if not kept:
        return []
    added = []
    for fr, row in zip(frames, rows, strict=True):
        pl = placed.get(fr.work.index)
        if row["predicted_status"] != "registered" or pl is None or pl.T is None \
                or pl.sigma_pos_m is None or fr.work.index not in feats:
            continue
        f = _frame_in(fr, None)
        K = f.K
        depth, kind = (f.depth, "sensor") if f.depth is not None \
            and bool(np.count_nonzero(f.depth > 0)) else (
            np.zeros((K.height, K.width), np.float32), "none")
        added.append(am.add(f, feats[fr.work.index], copy.copy(pl), depth, K, kind, None))
    return reregistered(am.reregister(kept, added))


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


def _prior_gravity(up: Updater, frames: list[_Frame]) -> float | None:
    """A new anchored map's ``prior_gravity_deg``, as its first update reports it
    (``AnchoredMap._up_check``): the angle between the up axis of the origin frame's prior (the
    first kept frame with a metric or drifting prior) and the gravity the monocular model sees in
    it. None without such a frame (a camera-only map is levelled by that gravity) or estimate."""
    origin = next((fr for fr in frames if fr.reason is None and fr.path is not None
                   and fr.work.trust in ("metric", "drifting")), None)
    if origin is None:
        return None
    assert origin.path is not None and origin.work.prior is not None
    up_cam = up.backend.infer(origin.path, origin.cam.canonical(), True).up_cam
    if up_cam is None:
        return None
    R = np.asarray(origin.work.prior["T_map_cam"], np.float64)[:3, :3]
    u = R @ unit(np.asarray(up_cam, np.float64))
    return round(math.degrees(math.acos(max(-1.0, min(1.0, float(u[2]))))), 3)


def _place_all(pool: ThreadPoolExecutor, am: AnchoredMap | None, frames: list[_Frame],
               feats: dict[int, Features]) -> dict[int, Placement]:
    """Every kept frame placed against the stored map (``AnchoredMap.place``): the drifting
    priors predicted along the request on a copy of the map's odometry, then the strong
    conflicts a re-anchored odometry explains predicted registered."""
    if am is None:
        return {}
    todo = [fr for fr in frames if fr.reason is None and fr.work.index in feats]
    ins = {fr.work.index: _frame_in(fr, None) for fr in todo}
    drift = copy.deepcopy(am.drift)
    preds: dict[int, tuple[Any, float, float] | None] = {}
    for fr in todo:
        f = ins[fr.work.index]
        if f.prior is not None and f.prior["trust"] == "drifting":
            P = np.asarray(f.prior["T_map_cam"], np.float64)
            preds[fr.work.index] = drift.predict(P, *_prior_sigmas(f.prior))
            drift.observe(P)
    pls = list(pool.map(lambda fr: am.place(ins[fr.work.index], feats[fr.work.index], None,
                                            preds.get(fr.work.index), evidence=False), todo))
    out = {fr.work.index: pl for fr, pl in zip(todo, pls, strict=True)}
    conflicts = [fr.work.index for fr, pl in zip(todo, pls, strict=True)
                 if pl.reason == "prior_conflict" and fr.work.trust == "drifting"]
    for a, b in itertools.pairwise(conflicts):
        if not am._same_drift(ins[a], out[a], ins[b], out[b]):
            continue
        for c in conflicts:  # the conflicts this correction explains
            if c == a or am._same_drift(ins[a], out[a], ins[c], out[c]):
                pl = out[c]
                pl.status, pl.reason, pl.T = "registered", None, pl.align_T
                pl.sigma_pos_m, pl.sigma_rot_deg = pl.image_sigmas or (None, None)
        break
    return out


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
    elif anchored:
        assert pl is not None
        map_inliers = pl.inliers
        status, reason = pl.status, pl.reason
    elif fr.unconfirmed is not None:  # its image failed a check: kept at its prior
        status, reason = "kept_at_prior", fr.unconfirmed
    elif new_map and not first and fr.work.trust not in ("metric", "drifting") \
            and max([int(n["inliers"]) for n in neighbours] or [0]) < MIN_INLIERS:
        # a new map's frame without prior holds only by its links to the request's frames
        status, reason = "rejected", "no_overlap"
    # the 300-inlier exemption counts image feature inliers only, never depth (api.md §4)
    close = too_close(fr.scene_m, 0 if map_inliers is None or pl is None
                      or pl.method == "depth" else map_inliers)
    if close and fr.reason is None and fr.work.trust in ("metric", "drifting"):
        # the distance limit (api.md §4): a frame with a prior is kept at it, its reason the
        # check its image failed, else too_close
        status, reason = "kept_at_prior", fr.unconfirmed or "too_close"
    elif close and status != "rejected":
        status, reason = "rejected", "too_close"
    row = {"name": fr.work.name, "features": 0 if m is None else m.features,
           "map_inliers": map_inliers, "neighbour_inliers": neighbours,
           "plain": m is not None and m.texture < checks.PLAIN_TEXTURE,
           "too_close": fr.scene_m is not None and fr.scene_m < TOO_CLOSE_M,
           "predicted_status": status, "predicted_reason": None if status == "registered"
           else reason}
    if reason == "too_close" and status != "registered":
        row["min_distance_m"] = TOO_CLOSE_M
        row["scene_distance_m"] = round(fr.scene_m or 0.0, 3)
    if reason == "duplicate" and status == "rejected":
        row["duplicate_of"] = fr.duplicate_of or fr.work.name
    return row, pl
