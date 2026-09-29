"""Map update (high_level_spec.md §2.3 / §5): mapping ``examples/office_sequence/``, whose scene
changes during the capture (a cup on the window sill is gone in the last images), must give a map
that reflects the latest observation. What changed is data: a ``map_update`` file under
``examples/ground_truth/`` (format: that folder's README.md).

Two maps are built. The first takes the whole sequence in one update (the spec's example: the
map after the full sequence). The second is built the way a map is extended: an update with the
early part of the sequence (``MapUpdatePlan.before_images``: the images up to the last one that
shows an absent object, where the scene still had it), then a second update of the same map with
the rest of the images. Its map frame is the one of the first update, so the objects of the two
updates compare without an alignment. Metrics (``map_update.*``):

* ``absent_fraction`` and ``incremental.absent_fraction``: the share of the annotated absent
  objects that the map of the whole sequence, and the extended map, no longer has. A remnant of
  an absent object is a map object with a compatible label (its own or one of its ``detected_as``)
  whose box, projected with that map's own poses and intrinsics into the images that show the
  absent object, covers the annotated region (``REGION_COVERAGE`` of it) in at least one of them.
  The test follows the map's pose drift because the projection uses the map's own camera poses; if
  none of those images is registered in the map the label alone decides.
* ``before_present_fraction``: the same test on the map after the first update — the control that
  makes the absence mean something: an object the detector never saw in the early images cannot be
  seen to disappear.
* ``stability.*`` (the metrics of ``mapquality.stability_metrics``, whose method is described
  there): the objects that never changed keep their ``id``s, labels and OBBs from the first update
  to the second. The extended map's objects that the early images observe (absent objects left
  out) are paired with the objects of the map after the first update.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from oh_my_slam.core.geometry import project
from oh_my_slam.core.types import Pose
from oh_my_slam.segmentation.detect import compatible
from oh_my_slam.segmentation.obb import OBB
from oh_my_slam.tools.evaluate.groundtruth import Absent, MapUpdatePlan
from oh_my_slam.tools.evaluate.mapquality import STABILITY_METRICS, stability_metrics
from oh_my_slam.tools.evaluate.metrics import Metrics
from oh_my_slam.tools.evaluate.poses import capture_poses, capture_sources
from oh_my_slam.tools.evaluate.scene import DocObject, Json, doc_objects

PREFIX = "map_update"
ABSENT_METRICS = ("absent_fraction", "incremental.absent_fraction")
BEFORE_METRIC = "before_present_fraction"
REGION_COVERAGE = 0.25  # share of the annotated region a projected box must cover
MIN_DEPTH_M = 0.05  # a box with a corner nearer than this (or behind) has no image footprint
IMAGE_SUFFIXES = (".jpg", ".jpeg")


def metric_ids() -> list[str]:
    return [*(f"{PREFIX}.{k}" for k in (*ABSENT_METRICS, BEFORE_METRIC)),
            *(f"{PREFIX}.stability.{k}" for k in STABILITY_METRICS)]


def sequence_images(folder: Path) -> list[str]:
    """The images of a sequence folder by name (capture timestamps sort in capture order)."""
    return sorted(p.name for p in Path(folder).iterdir()
                  if p.is_file() and not p.name.startswith(".")
                  and p.suffix.lower() in IMAGE_SUFFIXES)


@dataclass
class MapView:
    """What the judgement needs of one map's ``-t full`` scene."""

    objects: list[DocObject]
    poses: dict[str, Pose]  # camera-to-map, by capture file name
    sources: dict[int, str]  # frame key → capture file name
    camera: tuple[float, float, float, float, int, int] | None  # fx, fy, cx, cy, width, height

    @classmethod
    def of(cls, doc: Json, map_dir: Path) -> MapView:
        return cls(doc_objects(doc), capture_poses(doc, map_dir), capture_sources(doc, map_dir),
                   _camera(doc))


def _camera(doc: Json) -> tuple[float, float, float, float, int, int] | None:
    """The pinhole intrinsics of the first stream that has them (the mapper has one camera)."""
    for stream in (doc.get("openlabel", {}).get("streams") or {}).values():
        pin = (stream.get("stream_properties") or {}).get("intrinsics_pinhole")
        if pin and len(pin.get("camera_matrix", ())) >= 7:
            k = pin["camera_matrix"]  # 3 x 4, row-major
            return (float(k[0]), float(k[5]), float(k[2]), float(k[6]),
                    int(pin["width_px"]), int(pin["height_px"]))
    return None


def region_coverage(box: OBB, pose: Pose, camera: tuple[float, float, float, float, int, int],
                    region: tuple[float, float, float, float]) -> float | None:
    """The share of ``region`` (shares of the image size) that ``box``' image footprint (the
    bounding rectangle of its projected corners) covers; None when a corner is not in front of
    the camera (no footprint)."""
    fx, fy, cx, cy, w, h = camera
    cam = pose.inverse().apply(box.corners())
    if np.any(cam[:, 2] <= MIN_DEPTH_M):
        return None
    uv, _ = project(cam, np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]))
    (u0, v0), (u1, v1) = uv.min(axis=0), uv.max(axis=0)
    r0, s0, r1, s1 = region[0] * w, region[1] * h, region[2] * w, region[3] * h
    inter = max(0.0, min(u1, r1) - max(u0, r0)) * max(0.0, min(v1, s1) - max(v0, s0))
    return float(inter / ((r1 - r0) * (s1 - s0)))


def _labels(o: DocObject) -> tuple[str, ...]:
    return (o.label, *o.labels)


def remnants(view: MapView, item: Absent) -> list[dict[str, Any]]:
    """The objects of ``view`` that still are (or stand where) the absent ``item`` was: see the
    module docstring. Rows carry what is needed to find them in the map."""
    shown = {n: r for n, r in item.seen_in.items() if n in view.poses}
    rows = []
    for o in view.objects:
        if not any(compatible(label, item.label) for label in _labels(o)):
            continue
        row: dict[str, Any] = {"id": o.id, "label": o.label, "detected_as": list(o.labels),
                               "frames": sorted(view.sources[f] for f in o.frames
                                                if f in view.sources)}
        box = o.obb()
        if not shown or view.camera is None or box is None:
            rows.append({**row, "localised": False})  # nothing to project with: the label decides
            continue
        cover = {n: region_coverage(box, view.poses[n], view.camera, r) for n, r in shown.items()}
        seen = [(c, n) for n, c in cover.items() if c is not None]
        best = max(seen, key=lambda cn: cn[0]) if seen else None
        if best is not None and best[0] >= REGION_COVERAGE:
            rows.append({**row, "localised": True, "coverage": round(best[0], 3),
                         "image": best[1]})
    return rows


def _absent(m: Metrics, mid: str, rows: list[dict[str, Any]], key: str, view: MapView) -> None:
    """A map without any object proves nothing about one object being gone: no value."""
    left = [{"label": r["label"], "objects": r[key]} for r in rows if r[key]]
    m.add(mid, sum(not r[key] for r in rows) / len(rows) if view.objects else None,
          {"present_in_the_map": left, "map_objects": len(view.objects)},
          error="the map has no objects at all: the absence of one proves nothing")


def map_update_metrics(m: Metrics, plan: MapUpdatePlan, images: list[str], single: MapView,
                       early: MapView, extended: MapView) -> dict[str, Any]:
    """Record the ``map_update.*`` metrics: ``single`` is the map of the whole sequence in one
    update, ``early`` the map after the first update of the extended map, ``extended`` that map
    after its second update. Returns the rows for the report."""
    ids = metric_ids()
    rows: list[dict[str, Any]] = []
    with m.expect(*ids[:3]):
        rows = [{"label": a.label, "final": remnants(single, a),
                 "incremental": remnants(extended, a), "before": remnants(early, a)}
                for a in plan.absent]
        _absent(m, ids[0], rows, "final", single)
        _absent(m, ids[1], rows, "incremental", extended)
        m.add(ids[2], sum(bool(r["before"]) for r in rows) / len(rows),
              {"missing_before": [r["label"] for r in rows if not r["before"]]})
    stability: list[dict[str, Any]] = []
    with m.expect(*ids[3:]):
        stability = _stability(m, plan, images, extended, early)
    return {"absent": rows, "stability": stability,
            "before_images": plan.before_images(images)}


def _stability(m: Metrics, plan: MapUpdatePlan, images: list[str], extended: MapView,
               early: MapView) -> list[dict[str, Any]]:
    absent = [a.label for a in plan.absent]
    seen_early = set(plan.before_images(images))

    def unchanged(o: DocObject) -> bool:
        gone = any(compatible(label, a) for label in _labels(o) for a in absent)
        wanted = not plan.stable or any(compatible(o.label, s) for s in plan.stable)
        return wanted and not gone

    kept = [o for o in extended.objects
            if unchanged(o) and any(extended.sources.get(f) in seen_early for f in o.frames)]
    # one map, one frame: the two updates' objects need no alignment
    return stability_metrics(m, f"{PREFIX}.stability", kept,
                             [o for o in early.objects if unchanged(o)], Pose.identity())
