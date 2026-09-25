"""Map quality: point-cloud consistency across same-heading keyframes, and the stability of the
objects when the same sequence is mapped in one update versus split across several.

Consistency method: keyframe ``i``'s stored depth is back-projected into keyframe ``j`` with the
map's poses; where it lands on a valid pixel of ``j`` that shows the same surface (relative
difference below ``SAME_SURFACE``), ``|z_i→j / z_j - 1|`` is the disagreement. Stacked copies of a
surface (frames merged without agreeing) show up here.

Stability method: the split map is brought into the one-update map's frame by the rigid transform
that best maps the camera poses of the captures registered in both (rotation average + mean
translation, robust for a camera turning in place). Objects are then paired one-to-one (Hungarian)
on cost ``(1 - IoU) + centre distance``, plus ``LABEL_PENALTY`` for incompatible labels; a pair is
admissible when its boxes overlap (IoU ≥ ``MATCH_IOU``) or its centres lie within
``MATCH_CENTRE_M``. Labels only break geometric ambiguity — a desk standing on a carpet overlaps
it, and pairing the desk of one map with the carpet of the other would count a correct map as
unstable twice (label and id) — and never gate a pair: an object whose label changed is still
paired with its nearest box, so label agreement stays an independent measurement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from oh_my_slam.core.geometry import project
from oh_my_slam.core.types import Pose
from oh_my_slam.mapping.frame import align_by_poses
from oh_my_slam.mapping.store import FrameRecord, MapReader
from oh_my_slam.segmentation.detect import compatible
from oh_my_slam.segmentation.obb import OBB, obb_iou_upright
from oh_my_slam.tools.evaluate.metrics import Metrics
from oh_my_slam.tools.evaluate.names import Capture, same_heading_pairs
from oh_my_slam.tools.evaluate.scene import DocObject

MATCH_IOU = 0.05
MATCH_CENTRE_M = 0.25
LABEL_PENALTY = 1.0
MIN_OVERLAP_PX = 500  # fewer shared pixels: the pair is not compared
SAME_SURFACE = 0.3  # larger relative differences are occlusions, not the same surface
PIXEL_STEP = 7  # every 7th valid pixel of the source keyframe is back-projected
AGREEMENT_METRICS = ("frame_agreement_median_pct", "frame_agreement_p90_pct")
STABILITY_METRICS = ("matched_fraction", "label_agreement", "id_agreement",
                     "centre_delta_median_m", "extent_delta_median_rel", "obb_iou_median")


def pair_agreement(reader: MapReader, ri: FrameRecord, rj: FrameRecord
                   ) -> tuple[float, float] | None:
    """Median and p90 of |z_i→j / z_j - 1| with keyframe ``ri``'s depth back-projected into
    ``rj`` (same surfaces only); None when they share fewer than ``MIN_OVERLAP_PX`` pixels."""
    di, vi = reader.depth(ri), reader.valid(ri)
    dj, vj = reader.depth(rj), reader.valid(rj)
    v, u = np.nonzero(vi & (di > 0))
    u, v = u[::PIXEL_STEP], v[::PIXEL_STEP]
    Ki = ri.K_grid.K()
    z = di[v, u].astype(np.float64)
    pc = np.stack([(u - Ki[0, 2]) / Ki[0, 0] * z, (v - Ki[1, 2]) / Ki[1, 1] * z, z], 1)
    T = rj.T_map_cam.inverse().compose(ri.T_map_cam)
    uv, zq = project(pc @ T.R.T + T.t, rj.K_grid.K())
    h, w = dj.shape
    with np.errstate(invalid="ignore"):
        uu, vv = np.floor(uv[:, 0] + 0.5), np.floor(uv[:, 1] + 0.5)
        ok = (zq > 0.1) & (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
    uu, vv, zq = uu[ok].astype(int), vv[ok].astype(int), zq[ok]
    keep = vj[vv, uu] & (dj[vv, uu] > 0)
    if keep.sum() < MIN_OVERLAP_PX:
        return None
    r = np.abs(zq[keep] / dj[vv[keep], uu[keep]] - 1)
    r = r[r < SAME_SURFACE]
    return float(np.median(r)), float(np.percentile(r, 90))


def agreement_metrics(m: Metrics, prefix: str, map_dir: Path, captures: list[Capture]) -> None:
    """``<prefix>.frame_agreement_*``: depth disagreement of the spec's same-heading keyframe
    pairs (worst pair's median and p90, in %)."""
    ids = [f"{prefix}.{k}" for k in AGREEMENT_METRICS]
    reader = MapReader(map_dir)
    by_capture = {Path(r.source).name: r for r in reader.frames}
    pairs: dict[str, Any] = {}
    for a, b in same_heading_pairs(captures):
        key = f"{a.name}~{b.name}"
        if a.name not in by_capture or b.name not in by_capture:
            pairs[key] = "not registered"
            continue
        res = pair_agreement(reader, by_capture[a.name], by_capture[b.name])
        pairs[key] = "no overlap" if res is None else {
            "median_pct": round(res[0] * 100, 2), "p90_pct": round(res[1] * 100, 2)}
    done = [v for v in pairs.values() if isinstance(v, dict)]
    if not done:
        m.fail(ids, f"no same-heading pair could be compared: {pairs}")
        return
    m.add(ids[0], max(v["median_pct"] for v in done), pairs)
    m.add(ids[1], max(v["p90_pct"] for v in done), pairs)


def split_alignment(single: dict[str, Pose], split: dict[str, Pose]) -> Pose:
    """``T_single_split`` from the captures registered in both maps."""
    shared = sorted(single.keys() & split.keys())
    if not shared:
        raise ValueError("the two maps share no registered capture")
    return align_by_poses([split[n] for n in shared], [single[n] for n in shared])


def match_objects(a: list[tuple[DocObject, OBB]], b: list[tuple[DocObject, OBB]]
                  ) -> list[tuple[int, int, float, float]]:
    """One-to-one pairs ``(i, j, iou, centre distance)`` of boxes ``a[i]`` and ``b[j]``."""
    if not a or not b:
        return []
    big = 1e6
    cost = np.full((len(a), len(b)), big)
    iou = np.zeros_like(cost)
    dist = np.linalg.norm(np.array([x.center for _, x in a])[:, None, :]
                          - np.array([y.center for _, y in b])[None, :, :], axis=2)
    for i, (_, ba) in enumerate(a):
        for j, (_, bb) in enumerate(b):
            reach = (np.linalg.norm(ba.size) + np.linalg.norm(bb.size)) / 2
            if dist[i, j] <= reach:  # otherwise the boxes cannot intersect
                iou[i, j] = obb_iou_upright(ba, bb, samples=4000)
            if iou[i, j] >= MATCH_IOU or dist[i, j] <= MATCH_CENTRE_M:
                cost[i, j] = (1.0 - iou[i, j]) + dist[i, j]
                if not compatible(a[i][0].label, b[j][0].label):
                    cost[i, j] += LABEL_PENALTY
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j), float(iou[i, j]), float(dist[i, j]))
            for i, j in zip(rows, cols, strict=True) if cost[i, j] < big]


def stability_metrics(m: Metrics, prefix: str, single: list[DocObject], split: list[DocObject],
                      T_single_split: Pose) -> list[dict[str, Any]]:
    """``<prefix>.*`` (see ``STABILITY_METRICS``); returns the matched pairs for the report."""
    ids = {k: f"{prefix}.{k}" for k in STABILITY_METRICS}
    a = [(o, box) for o in single if (box := o.obb()) is not None]
    b = [(o, box.transformed(T_single_split)) for o in split if (box := o.obb()) is not None]
    pairs = match_objects(a, b)
    n = max(len(a), len(b))
    m.add(ids["matched_fraction"], len(pairs) / n if n else None,
          {"single": len(a), "split": len(b), "matched": len(pairs)},
          error=None if n else "neither map has objects")
    rows: list[dict[str, Any]] = []
    for i, j, iou, d in pairs:
        (oa, ba), (ob, bb) = a[i], b[j]
        rel = np.abs(ba.size - bb.size) / np.maximum(np.maximum(ba.size, bb.size), 1e-6)
        rows.append({"single_id": oa.id, "split_id": ob.id, "single_label": oa.label,
                     "split_label": ob.label, "iou": round(iou, 3), "centre_delta_m": round(d, 3),
                     "extent_delta_rel": round(float(rel.mean()), 3)})
    none = "no object pairs"
    k = len(rows)
    m.add(ids["label_agreement"],
          sum(compatible(r["single_label"], r["split_label"]) for r in rows) / k if k else None,
          error=None if k else none)
    m.add(ids["id_agreement"], sum(r["single_id"] == r["split_id"] for r in rows) / k
          if k else None, error=None if k else none)
    for key, col in (("centre_delta_median_m", "centre_delta_m"),
                     ("extent_delta_median_rel", "extent_delta_rel"), ("obb_iou_median", "iou")):
        m.add(ids[key], float(np.median([r[col] for r in rows])) if k else None,
              error=None if k else none)
    return rows
