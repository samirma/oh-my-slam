"""A dry run of an update (O10, api.md §4): ``POST /api/maps/{name}/updates?dry_run=1``.

It runs synchronously and writes nothing (no camera, version, job or mapper file):

* the per-frame checks of an update (``Updater._check``), with the monocular median depth of the
  suspicious frames (``too_close``);
* per kept frame, its link to the map: on an anchored map the prior-first placement against the
  current version (``AnchoredMap.place`` on a copy loaded read-only; nothing is added), whose
  inliers are ``map_inliers``; on a new map or a map of the mapper, null;
* its links to its neighbours in the request (the previous and the next frame): the inliers of a
  fundamental matrix (RANSAC) on their ratio-test SIFT matches;
* the predicted status and reason, the predicted quality report with its gate (``accept_if``),
  and the predicted durations from the measured per-frame timings."""

from __future__ import annotations

import shutil
import uuid
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from oh_my_slam.mapping.anchored import (
    MIN_INLIERS,
    AnchoredMap,
    Features,
    Placement,
    features,
    match,
    pose_residual,
)
from oh_my_slam.service import checks
from oh_my_slam.service.prior_first import LOW_CONFIDENCE_SIGMA_M, _frame_in

if TYPE_CHECKING:
    from oh_my_slam.service.update import Updater, UpdateWork, _Frame

NEIGHBOUR_PX = 3.0  # the fundamental matrix's RANSAC threshold


def link(a: Features, b: Features) -> int:
    """Geometrically verified matches between two frames' features."""
    ia, ib = match(a.desc, b.desc)
    if len(ia) < 8:
        return 0
    _, mask = cv2.findFundamentalMat(a.kps[ia].astype(np.float64), b.kps[ib].astype(np.float64),
                                     cv2.FM_RANSAC, NEIGHBOUR_PX, 0.999)
    return 0 if mask is None else int(mask.sum())


def dry_run(up: Updater, w: UpdateWork, new_cameras: dict[str, dict[str, Any]],
            new_masks: dict[str, tuple[str, bytes]]) -> dict[str, Any]:
    """The ``DryRun`` body of an update request."""
    from oh_my_slam.service.update import gate, residual_stats

    store = up.store
    prev = store.version(w.map) if store.latest(w.map) else None
    workdir = store.work / f"dry-{uuid.uuid4().hex}"
    indir = workdir / "in"
    indir.mkdir(parents=True, exist_ok=True)
    try:
        frames, masks = up._frames(w, new_cameras, new_masks)
        up._check(frames, masks, prev, indir)
        up._resolve_deferred(frames)
        feats: dict[int, Features] = {fr.work.index: features(fr.rgb, fr.ignore)
                                      for fr in frames if fr.rgb is not None}
        am = AnchoredMap.load(store.mapper_dir(w.map)) \
            if prev is not None and prev.get("engine") == "anchored" else None
        out, kept, prior_res = [], [], []
        first_kept = True
        for k, fr in enumerate(frames):
            row, pl = _frame(fr, k, frames, feats, am, prev is None, first_kept)
            if row["predicted_status"] == "registered":
                first_kept = False
                kept.append(pl)
                if pl is not None and pl.T is not None and fr.work.trust == "metric":
                    assert fr.work.prior is not None
                    prior_res.append(pose_residual(pl.T, np.asarray(fr.work.prior["T_map_cam"])))
            out.append(row)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    low = sum(pl is not None and (pl.status == "provisional"
                                  or (pl.sigma_pos_m or 0.0) > LOW_CONFIDENCE_SIGMA_M)
              for pl in kept)
    q: dict[str, Any] = {
        "scale_ratio_vs_prev": None if prev is None else 1.0,
        "gauge_shift": None if prev is None else {"translation_m": 0.0, "rotation_deg": 0.0,
                                                  "scale_ratio": 1.0},
        "prior_residuals": residual_stats(prior_res),
        "previous_residuals": residual_stats([(0.0, 0.0)] * len(prev["frames"])) if prev
        else None,
        "rejected_share": round(1.0 - len(kept) / len(frames), 4),
        "low_confidence_share": round(low / len(kept), 4) if kept else 0.0,
        "rebuilt": False, "prior_gravity_deg": None,
    }
    n = len(kept)
    return {"map": w.map,
            "base_version": None if prev is None else prev["version"], "frames": out,
            "quality": gate(q, bool(kept), w.accept_if),
            "predicted_s": {
                "registration": round(n * up.timings.median("registration_per_frame"), 3),
                "fusion": round(n * up.timings.median("fusion_per_frame"), 3)}}


def _frame(fr: _Frame, k: int, frames: list[_Frame], feats: dict[int, Features],
           am: AnchoredMap | None, new_map: bool, first: bool
           ) -> tuple[dict[str, Any], Placement | None]:
    m = fr.measures
    neighbours: list[dict[str, Any]] = []
    own = feats.get(fr.work.index)
    for j in (k - 1, k + 1):
        if 0 <= j < len(frames) and own is not None \
                and frames[j].work.index in feats:
            neighbours.append({"name": frames[j].work.name,
                               "inliers": link(own, feats[frames[j].work.index])})
    status, reason, map_inliers, pl = "registered", fr.reason, None, None
    if reason is not None:
        status = "rejected"
    elif am is not None:
        assert own is not None
        f = _frame_in(fr, None)
        pl = am.place(f, own, None)
        map_inliers = pl.inliers
        if pl.status == "rejected":
            status, reason = "rejected", pl.reason
    elif new_map and not first and fr.work.trust not in ("metric", "drifting") \
            and max([int(n["inliers"]) for n in neighbours] or [0]) < MIN_INLIERS:
        # a new map's frame without prior holds only by its links to the request's frames
        status, reason = "rejected", "no_overlap"
    return {"name": fr.work.name, "features": 0 if m is None else m.features,
            "map_inliers": map_inliers, "neighbour_inliers": neighbours,
            "plain": m is not None and m.texture < checks.PLAIN_TEXTURE,
            "too_close": fr.reason == "too_close",
            "predicted_status": status, "predicted_reason": None if status == "registered"
            else reason}, pl
