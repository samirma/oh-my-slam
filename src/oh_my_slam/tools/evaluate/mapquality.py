"""Map quality: point-cloud consistency across same-heading keyframes, duplicated objects, and the
stability of the objects when the same sequence is mapped in one update versus split across
several.

Consistency method: keyframe ``i``'s stored depth is back-projected into keyframe ``j`` with the
map's poses; where it lands on a valid pixel of ``j`` that shows the same surface (relative
difference below ``SAME_SURFACE``), ``|z_i→j / z_j - 1|`` is the disagreement. Stacked copies of a
surface (frames merged without agreeing) show up here.

Duplicates method (``near_duplicates``): pairs of exported objects with compatible labels that no
keyframe observed together (their ``frame_intervals`` are disjoint: a keyframe that detected both
saw two things) and whose boxes overlap or lie within ``NEAR_DUPLICATE_GAP_M`` of each other
(``box_gap``). An object mapped twice — typically by keyframes whose monocular depth disagrees,
which places the copies along the same viewing rays at different depths — is such a pair.

Stability method: the split map is brought into the one-update map's frame by the rigid transform
that best maps the camera poses of the captures registered in both (rotation average + mean
translation, robust for a camera turning in place). Objects are then paired one-to-one (Hungarian)
on cost ``(1 - IoU) + centre distance`` (metres); a pair is admissible when its boxes overlap
(IoU ≥ ``MATCH_IOU``) or its centres lie within ``MATCH_CENTRE_M``.

* Ids and boxes (``id_agreement``, centre, extent, IoU) use a label-aware pairing: incompatible
  labels add ``LABEL_PENALTY`` = 1.0, the whole range of the IoU term, so the pairing prefers
  candidates of compatible labels over any difference in overlap — a desk standing on a carpet
  overlaps it, and pairing the desk of one map with the carpet of the other would count a
  correct map as unstable twice. Labels never gate a pair: an object whose label changed still
  pairs with its nearest admissible box.
* ``label_agreement`` uses a label-blind pairing (no penalty), so labels play no part in choosing
  the pairs whose labels it compares: it is the share of geometric pairs with compatible labels,
  an independent measurement (over the label-aware pairs, agreement would be nearly true by
  construction). Its detail also reports the agreement over the label-aware pairs."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
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
LABEL_PENALTY = 1.0  # label-aware pairing (see the module docstring)
MIN_OVERLAP_PX = 500  # fewer shared pixels: the pair is not compared
SAME_SURFACE = 0.3  # larger relative differences are occlusions, not the same surface
PIXEL_STEP = 7  # every 7th valid pixel of the source keyframe is back-projected
AGREEMENT_METRICS = ("frame_agreement_median_pct", "frame_agreement_p90_pct")
STABILITY_METRICS = ("matched_fraction", "label_agreement", "id_agreement",
                     "centre_delta_median_m", "extent_delta_median_rel", "obb_iou_median")
DUPLICATE_METRIC = "near_duplicates"
# A copy made by monocular depth lies along the same viewing rays as the original, offset by the
# depth disagreement of the keyframes that saw each: the frame-agreement target allows a p90 of
# 10 %, i.e. 0.3 m at 3 m, the distance of the farthest objects of the example sequence.
NEAR_DUPLICATE_GAP_M = 0.3


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


def match_objects(a: list[tuple[DocObject, OBB]], b: list[tuple[DocObject, OBB]],
                  label_penalty: float = LABEL_PENALTY) -> list[tuple[int, int, float, float]]:
    """One-to-one pairs ``(i, j, iou, centre distance)`` of boxes ``a[i]`` and ``b[j]``
    (``label_penalty`` added for incompatible labels; 0: label-blind)."""
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
                    cost[i, j] += label_penalty
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
    blind = match_objects(a, b, label_penalty=0.0)
    agree = [compatible(a[i][0].label, b[j][0].label) for i, j, *_ in blind]
    aware = sum(compatible(r["single_label"], r["split_label"]) for r in rows)
    m.add(ids["label_agreement"], sum(agree) / len(agree) if agree else None,
          {"pairs": len(agree), "label_aware": round(aware / k, 4) if k else None,
           "disagreeing": [[a[i][0].id, a[i][0].label, b[j][0].id, b[j][0].label]
                           for (i, j, *_), ok in zip(blind, agree, strict=True) if not ok]},
          error=None if agree else none)
    m.add(ids["id_agreement"], sum(r["single_id"] == r["split_id"] for r in rows) / k
          if k else None, error=None if k else none)
    for key, col in (("centre_delta_median_m", "centre_delta_m"),
                     ("extent_delta_median_rel", "extent_delta_rel"), ("obb_iou_median", "iou")):
        m.add(ids[key], float(np.median([r[col] for r in rows])) if k else None,
              error=None if k else none)
    return rows


# ------------------------------------------------------------------------------------------------
# duplicated objects

# ``OBB.corners`` lists corner k at the signs of bits (4, 2, 1) of k along the box x, y, z axes:
# the 12 edges join the corners that differ in one bit
_EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]


def _point_box_distance(p: NDArray[Any], box: OBB) -> NDArray[np.float64]:
    local = np.abs((np.asarray(p, np.float64) - box.center) @ box.R) - box.size / 2
    return np.asarray(np.linalg.norm(np.maximum(local, 0.0), axis=1), np.float64)


def _segment_distances(p0: NDArray[Any], p1: NDArray[Any], q0: NDArray[Any], q1: NDArray[Any]
                       ) -> NDArray[np.float64]:
    """Distance between segments ``p0[i]p1[i]`` and ``q0[j]q1[j]`` for every (i, j) (closest
    points of two segments, Ericson, Real-Time Collision Detection 5.1.9)."""
    d1 = (p1 - p0)[:, None, :]
    d2 = (q1 - q0)[None, :, :]
    r = p0[:, None, :] - q0[None, :, :]
    a = np.maximum(np.sum(d1 * d1, axis=2), 1e-12)
    e = np.maximum(np.sum(d2 * d2, axis=2), 1e-12)
    f = np.sum(d2 * r, axis=2)
    c = np.sum(d1 * r, axis=2)
    b = np.sum(d1 * d2, axis=2)
    denom = a * e - b * b
    s = np.where(denom > 1e-12, np.clip((b * f - c * e) / np.maximum(denom, 1e-12), 0, 1), 0.0)
    t = (b * s + f) / e
    s = np.where(t < 0, np.clip(-c / a, 0, 1), np.where(t > 1, np.clip((b - c) / a, 0, 1), s))
    t = np.clip(t, 0, 1)
    diff = r + d1 * s[..., None] - d2 * t[..., None]
    return np.asarray(np.linalg.norm(diff, axis=2), np.float64)


def _boxes_overlap(a: OBB, b: OBB) -> bool:
    """Separating-axis test of two oriented boxes (face normals and edge cross products)."""
    axes = [a.R[:, i] for i in range(3)] + [b.R[:, i] for i in range(3)]
    axes += [np.cross(a.R[:, i], b.R[:, j]) for i in range(3) for j in range(3)]
    d = b.center - a.center
    for ax in axes:
        n = float(np.linalg.norm(ax))
        if n < 1e-9:
            continue
        ax = ax / n
        ra = float(np.sum(np.abs(a.R.T @ ax) * a.size / 2))
        rb = float(np.sum(np.abs(b.R.T @ ax) * b.size / 2))
        if abs(float(d @ ax)) > ra + rb + 1e-12:
            return False
    return True


def box_gap(a: OBB, b: OBB) -> float:
    """Smallest distance between two oriented boxes, 0 when they overlap. Two separate convex
    polyhedra are closest at a vertex of one and the other (point-to-box distance) or at an edge
    of each (segment-to-segment distance), so the minimum over those is exact."""
    if _boxes_overlap(a, b):
        return 0.0
    ca, cb = a.corners(), b.corners()
    ia, ja = np.array(_EDGES).T
    return float(min(_point_box_distance(ca, b).min(), _point_box_distance(cb, a).min(),
                     _segment_distances(ca[ia], ca[ja], cb[ia], cb[ja]).min()))


def near_duplicates(objs: list[DocObject]) -> list[dict[str, Any]]:
    """Pairs of objects with compatible labels that no keyframe observed together and whose
    boxes overlap or lie within ``NEAR_DUPLICATE_GAP_M`` (``box_gap``), by ascending ids."""
    boxes = [(o, box) for o in objs if (box := o.obb()) is not None]
    out = []
    for (a, ba), (b, bb) in itertools.combinations(boxes, 2):
        if not compatible(a.label, b.label) or a.frames & b.frames:
            continue
        reach = float(np.linalg.norm(ba.size) + np.linalg.norm(bb.size)) / 2
        if float(np.linalg.norm(ba.center - bb.center)) > reach + NEAR_DUPLICATE_GAP_M:
            continue  # the boxes are further apart than the gap
        gap = box_gap(ba, bb)
        if gap <= NEAR_DUPLICATE_GAP_M:
            out.append({"ids": [a.id, b.id], "labels": [a.label, b.label],
                        "gap_m": round(gap, 3),
                        "centre_distance_m": round(float(np.linalg.norm(ba.center - bb.center)),
                                                   3)})
    return out


def duplicate_metrics(m: Metrics, prefix: str, objs: list[DocObject]) -> list[dict[str, Any]]:
    """``<prefix>.near_duplicates``: the number of ``near_duplicates`` pairs of a map's objects;
    returns the pairs for the report."""
    rows = near_duplicates(objs)
    m.add(f"{prefix}.{DUPLICATE_METRIC}", len(rows),
          {"objects": len(objs), "gap_m": NEAR_DUPLICATE_GAP_M, "pairs": rows})
    return rows
