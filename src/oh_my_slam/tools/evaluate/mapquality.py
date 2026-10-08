"""Map quality (spec §5): point-cloud consistency across overlapping keyframes, and the stability
of the objects when the same sequence is mapped in one update versus split across several.

Consistency method: keyframe ``i``'s stored depth is back-projected into keyframe ``j`` with the
map's poses; where it lands on a valid pixel of ``j`` that shows the same surface (relative
difference below ``SAME_SURFACE``), ``|z_i→j / z_j - 1|`` is the disagreement. Stacked copies of a
surface (frames merged without agreeing) show up here. Two groups of pairs:

* the sequence's same-heading pairs. On ``ainex-captures`` the spec's pairs (§5: 001/053 and
  026/078, ``names.same_heading_pairs``) are revisits, which see the same surfaces at the same
  image positions: judged by the worst pair (``frame_agreement_*``: its median and its p90). On
  ``camera`` the tilts of one pan position (``names.tilt_pairs``) share only a strip, at the top
  of one frame and the bottom of the other: judged like the overlapping pairs below, over the
  tilt pairs alone (``frame_agreement_tilt_*``: ``pair_stats`` of the pairs' medians; user
  ruling 2026-10-08);
* every overlapping pair (``frame_agreement_pairs_*``): optical axes less than
  ``PAIRS_MAX_ANGLE_DEG`` apart and at least ``MIN_OVERLAP_PX`` shared pixels, whatever their
  distance in capture order — sequence neighbours, where keyframe-by-keyframe depth disagrees
  locally, and loop closures, where a depth scale that drifted along the sequence shows.
  Reported: the median and the p90 over the pairs of each pair's median disagreement, the share
  of pairs whose median disagreement exceeds ``GROSS_PCT`` and the worst pair; the detail repeats
  them for the neighbours (at most ``FAR_GAP`` keyframes apart) and the far pairs (more).

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
  construction). Its detail also reports the agreement over the label-aware pairs.

mapper.md allows two differences between the split map and the one-update map, both about what
an earlier update of the split map published (``Published``: that update's ``-t full`` objects):
an ``id`` may differ where that update had published it *for this object*, and the split map may
keep an object that update published. "For this object" is ``published_for``: an object of that
update with the same id, a compatible label and a box that overlaps or nearly coincides with the
split map's once carried into its frame (``same_object``). An id another object had (a swap) is no
such allowance, and an extra object of the split map that no earlier update published counts
against it (``unexcused_extra``)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from oh_my_slam.core.geometry import project, unproject_pixels
from oh_my_slam.core.types import Pose
from oh_my_slam.mapping.frame import align_by_poses
from oh_my_slam.mapping.store import FrameRecord, MapReader
from oh_my_slam.segmentation.detect import compatible
from oh_my_slam.segmentation.obb import OBB, obb_iou_upright
from oh_my_slam.tools.evaluate.metrics import Metrics
from oh_my_slam.tools.evaluate.names import AnyCapture
from oh_my_slam.tools.evaluate.scene import DocObject

MATCH_IOU = 0.05
MATCH_CENTRE_M = 0.25
LABEL_PENALTY = 1.0  # label-aware pairing (see the module docstring)
MIN_OVERLAP_PX = 500  # fewer shared pixels: the pair is not compared
SAME_SURFACE = 0.3  # larger relative differences are occlusions, not the same surface
PIXEL_STEP = 7  # every 7th valid pixel of the source keyframe is back-projected
SAME_HEADING_METRICS = ("frame_agreement_median_pct", "frame_agreement_p90_pct")  # worst pair
TILT_METRICS = ("frame_agreement_tilt_median_pct", "frame_agreement_tilt_p90_pct",
                "frame_agreement_tilt_over10_pct", "frame_agreement_tilt_max_pct")
PAIRS_METRICS = ("frame_agreement_pairs_median_pct", "frame_agreement_pairs_p90_pct",
                 "frame_agreement_pairs_over10_pct", "frame_agreement_pairs_max_pct")
STATS = ("median_pct", "p90_pct", "over10_pct", "max_pct")  # pair_stats: TILT_ / PAIRS_METRICS
AGREEMENT_METRICS = (*SAME_HEADING_METRICS, *PAIRS_METRICS)
TILT_AGREEMENT_METRICS = (*TILT_METRICS, *PAIRS_METRICS)
PAIRS_MAX_ANGLE_DEG = 45.0
FAR_GAP = 10  # pairs more than this many keyframes apart are "far" (loop closures) in the detail
GROSS_PCT = 10.0  # a pair disagreeing by more than this is grossly inconsistent
WORST_LISTED = 10
STABILITY_METRICS = ("matched_fraction", "unexcused_extra", "label_agreement", "id_agreement",
                     "centre_delta_median_m", "extent_delta_median_rel", "obb_iou_median")


def pair_agreement(reader: MapReader, ri: FrameRecord, rj: FrameRecord
                   ) -> tuple[float, float] | None:
    """Median and p90 of |z_i→j / z_j - 1| with keyframe ``ri``'s depth back-projected into
    ``rj`` (same surfaces only); None when they share fewer than ``MIN_OVERLAP_PX`` pixels, and
    ``SAME_SURFACE`` when they share pixels but disagree beyond it everywhere."""
    di, vi = reader.depth(ri), reader.valid(ri)
    dj, vj = reader.depth(rj), reader.valid(rj)
    v, u = np.nonzero(vi & (di > 0))
    u, v = u[::PIXEL_STEP], v[::PIXEL_STEP]
    pc = unproject_pixels(u, v, di[v, u].astype(np.float64), ri.K_grid.K())
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
    if not len(r):
        return SAME_SURFACE, SAME_SURFACE
    return float(np.median(r)), float(np.percentile(r, 90))


def overlapping_pairs(frames: list[FrameRecord]) -> list[tuple[FrameRecord, FrameRecord]]:
    """Keyframe pairs (earlier first) whose optical axes are less than ``PAIRS_MAX_ANGLE_DEG``
    apart, whatever their distance in capture order."""
    fr = sorted(frames, key=lambda r: r.index)
    if len(fr) < 2:
        return []
    F = np.array([r.T_map_cam.R[:, 2] for r in fr])
    cos = F @ F.T
    near = cos > float(np.cos(np.radians(PAIRS_MAX_ANGLE_DEG)))
    return [(a, b) for i, a in enumerate(fr) for j, b in enumerate(fr) if j > i and near[i, j]]


def _pct(res: tuple[float, float]) -> dict[str, float]:
    return {"median_pct": round(res[0] * 100, 2), "p90_pct": round(res[1] * 100, 2)}


def agreement_metrics(m: Metrics, prefix: str, map_dir: Path,
                      same_heading: Sequence[tuple[AnyCapture, AnyCapture]], tilts: bool = False
                      ) -> list[dict[str, Any]]:
    """``<prefix>.frame_agreement_*``: depth disagreement of the ``same_heading`` keyframe pairs
    and of every overlapping pair (``pair_metrics``), in %. The same-heading pairs are judged by
    the worst pair (its median and p90: ``SAME_HEADING_METRICS``) or, with ``tilts`` (the tilt
    pairs of a pan position), like the overlapping pairs over them alone (``pair_stats`` of each
    pair's median: ``TILT_METRICS``). Returns the worst overlapping pairs."""
    ids = [f"{prefix}.{k}" for k in (TILT_METRICS if tilts else SAME_HEADING_METRICS)]
    reader = MapReader(map_dir)
    by_capture = {Path(r.source).name: r for r in reader.frames}
    pairs: dict[str, Any] = {}
    for a, b in same_heading:
        key = f"{a.name}~{b.name}"
        if a.name not in by_capture or b.name not in by_capture:
            pairs[key] = "not registered"
            continue
        res = pair_agreement(reader, by_capture[a.name], by_capture[b.name])
        pairs[key] = "no overlap" if res is None else _pct(res)
    done = [v for v in pairs.values() if isinstance(v, dict)]
    if not done:
        m.fail(ids, f"no same-heading pair could be compared: {pairs}")
    elif tilts:
        stats = pair_stats(done)
        for mid, key in zip(ids, STATS, strict=True):
            m.add(mid, stats[key], {**stats, "same_heading": pairs})
    else:
        m.add(ids[0], max(v["median_pct"] for v in done), pairs)
        m.add(ids[1], max(v["p90_pct"] for v in done), pairs)
    return pair_metrics(m, prefix, reader, pairs)


def pair_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Median, p90, share above ``GROSS_PCT`` (all in %) and worst of the pairs' medians."""
    med = np.array([r["median_pct"] for r in rows], np.float64)
    if not len(med):
        return {"pairs": 0}
    return {"pairs": len(med), "median_pct": round(float(np.median(med)), 3),
            "p90_pct": round(float(np.percentile(med, 90)), 3),
            "over10_pct": round(float(np.mean(med > GROSS_PCT) * 100), 3),
            "max_pct": round(float(med.max()), 2)}


def pair_metrics(m: Metrics, prefix: str, reader: MapReader, same_heading: dict[str, Any]
                 ) -> list[dict[str, Any]]:
    """``<prefix>.frame_agreement_pairs_*`` over ``overlapping_pairs`` (see the module
    docstring); the detail repeats them for the neighbours and the far pairs and lists the worst
    pairs and the same-heading pairs. Returns the worst pairs."""
    ids = [f"{prefix}.{k}" for k in PAIRS_METRICS]
    source = {r.name: Path(r.source).name for r in reader.frames}
    axis = {r.name: r.T_map_cam.R[:, 2] for r in reader.frames}
    rows: list[dict[str, Any]] = []
    candidates = overlapping_pairs(reader.frames)
    for a, b in candidates:
        res = pair_agreement(reader, a, b)
        if res is not None:
            cos = float(np.clip(axis[a.name] @ axis[b.name], -1.0, 1.0))
            rows.append({"pair": f"{source[a.name]}~{source[b.name]}", "gap": b.index - a.index,
                         "angle_deg": round(float(np.degrees(np.arccos(cos))), 1), **_pct(res)})
    if not rows:
        m.fail(ids, f"no overlapping keyframe pair to compare ({len(candidates)} candidates)")
        return []
    worst = sorted(rows, key=lambda r: (-r["median_pct"], r["pair"]))[:WORST_LISTED]
    stats = pair_stats(rows)
    detail = {
        **stats, "candidates": len(candidates),
        "definition": f"optical axes < {PAIRS_MAX_ANGLE_DEG:g} deg apart, >= {MIN_OVERLAP_PX} "
                      "shared pixels, any distance in capture order",
        "neighbours": {"definition": f"<= {FAR_GAP} keyframes apart",
                       **pair_stats([r for r in rows if r["gap"] <= FAR_GAP])},
        "far": {"definition": f"> {FAR_GAP} keyframes apart (loop closures)",
                **pair_stats([r for r in rows if r["gap"] > FAR_GAP])},
        "worst": worst, "same_heading": same_heading,
    }
    for mid, key in zip(ids, STATS, strict=True):
        m.add(mid, stats[key], detail)
    return worst


def split_alignment(single: dict[str, Pose], split: dict[str, Pose]) -> Pose:
    """``T_single_split`` from the captures registered in both maps."""
    shared = sorted(single.keys() & split.keys())
    if not shared:
        raise ValueError("the two maps share no registered capture")
    return align_by_poses([split[n] for n in shared], [single[n] for n in shared])


def update_alignment(later: dict[str, Pose], earlier: dict[str, Pose]) -> Pose:
    """``T_later_earlier`` between two updates of one map, from the captures registered in both
    (identity when none is)."""
    try:
        return split_alignment(later, earlier)
    except ValueError:
        return Pose.identity()


def same_object(a: DocObject, b: DocObject, T_b_a: Pose) -> bool:
    """``b`` is the object ``a`` was: a compatible label and a box that overlaps or nearly
    coincides with ``a``'s carried into ``b``'s map frame by ``T_b_a`` (the pairing's
    admissibility, ``MATCH_IOU`` / ``MATCH_CENTRE_M``)."""
    if not compatible(a.label, b.label):
        return False
    box_a, bb = a.obb(), b.obb()
    if box_a is None or bb is None:
        return True
    ba = box_a.transformed(T_b_a)
    d = float(np.linalg.norm(ba.center - bb.center))
    return d <= MATCH_CENTRE_M or obb_iou_upright(ba, bb, samples=4000) >= MATCH_IOU


@dataclass(frozen=True)
class Published:
    """What an earlier update of the split map published: its ``-t full`` objects, and
    ``T_split_update``, which carries their boxes into the split map's frame (its last update's,
    ``update_alignment``)."""

    objects: list[DocObject]
    T_split_update: Pose


def published_for(o: DocObject, published: list[Published]) -> bool:
    """An earlier update published ``o``'s id for this very object (``same_object``)."""
    return any(p.id == o.id and same_object(p, o, u.T_split_update)
               for u in published for p in u.objects)


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
                      T_single_split: Pose, published: list[Published] | None = None
                      ) -> list[dict[str, Any]]:
    """``<prefix>.*`` (see ``STABILITY_METRICS``); returns the matched pairs for the report.

    ``published``: what the earlier updates of the split map published. mapper.md lets an id
    differ from the one-update map's where one of them had published it for this object, and the
    split map keep an object one of them published (``published_for``; module docstring); the
    detail keeps the strict share of same ids (``same_id``) too."""
    ids = {k: f"{prefix}.{k}" for k in STABILITY_METRICS}
    earlier = published or []
    a = [(o, box) for o in single if (box := o.obb()) is not None]
    b = [(o, box.transformed(T_single_split)) for o in split if (box := o.obb()) is not None]
    pairs = match_objects(a, b)
    # matched_fraction: the share of the one-update map's objects the split map has. Its extra
    # objects are listed; one that no earlier update published counts (unexcused_extra)
    n = len(a)
    paired = {j for _, j, _, _ in pairs}
    extra = [o for j, (o, _) in enumerate(b) if j not in paired]
    kept_extra = [{"id": o.id, "label": o.label} for o in extra if published_for(o, earlier)]
    unexcused = [{"id": o.id, "label": o.label} for o in extra if not published_for(o, earlier)]
    m.add(ids["matched_fraction"], len(pairs) / n if n else None,
          {"single": len(a), "split": len(b), "matched": len(pairs),
           "extra_published": kept_extra, "extra_unexcused": unexcused},
          error=None if n else "the one-update map has no objects")
    m.add(ids["unexcused_extra"], len(unexcused), {"extra_unexcused": unexcused})
    rows: list[dict[str, Any]] = []
    kept: list[list[int]] = []  # [single id, split id]: an id published earlier for the object
    for i, j, iou, d in pairs:
        (oa, ba), (ob, bb) = a[i], b[j]
        rel = np.abs(ba.size - bb.size) / np.maximum(np.maximum(ba.size, bb.size), 1e-6)
        rows.append({"single_id": oa.id, "split_id": ob.id, "single_label": oa.label,
                     "split_label": ob.label, "iou": round(iou, 3), "centre_delta_m": round(d, 3),
                     "extent_delta_rel": round(float(rel.mean()), 3)})
        if oa.id != ob.id and published_for(ob, earlier):
            kept.append([oa.id, ob.id])
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
    same = sum(r["single_id"] == r["split_id"] for r in rows)
    m.add(ids["id_agreement"], (same + len(kept)) / k if k else None,
          {"same_id": round(same / k, 4) if k else None, "published_earlier": kept}
          if published is not None else None, error=None if k else none)
    for key, col in (("centre_delta_median_m", "centre_delta_m"),
                     ("extent_delta_median_rel", "extent_delta_rel"), ("obb_iou_median", "iou")):
        m.add(ids[key], float(np.median([r[col] for r in rows])) if k else None,
              error=None if k else none)
    return rows
