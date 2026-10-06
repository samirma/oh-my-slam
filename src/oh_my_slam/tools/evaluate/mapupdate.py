"""Map update (specs/mapper.md §2.3, specs/high_level_spec.md §5): mapping
``examples/office_sequence/``, whose scene changes during the capture (a cup on the window sill is
gone in the last images), must give a map that reflects the latest observation — without the cup
and with no hole where it stood — in one update or split across several, while every object that
never changed keeps its ``id``, label and OBB. What changed is data: a ``map_update`` file under
``examples/ground_truth/`` (format: that folder's README.md), which also says how the sequence is
split (``splits``: the sizes of consecutive updates, e.g. 4+4+5 and 6+7).

Maps: the whole sequence in one update, and one map per split (an update per part, in order). Each
update's ``-t full`` scene is kept, since a later update changes the map's frame records. Metrics
(``map_update.*``; ``<s>`` is a split's name, ``split_4_4_5``):

* ``absent_fraction`` / ``<s>.absent_fraction``: the share of the annotated absent objects that
  the one-update map, and each split map after its last update, no longer has. A remnant of an
  absent object is a map object with a compatible label (its own or one of its ``detected_as``)
  whose box, projected with that map's own poses and intrinsics into the images that show the
  absent object, covers the annotated region (``REGION_COVERAGE`` of it) in at least one of them.
  If none of those images is registered in the map the label alone decides.
* ``hole_fraction`` / ``<s>.hole_fraction``: no hole where the absent object stood. The map cloud
  is projected (z-buffer) into each registered image that showed it; the annotated region and a
  ring around it are cut into cells; a region cell is a hole when no map point falls in it or its
  nearest point lies more than ``HOLE_DEPTH_REL`` behind the ring's median nearest depth (the
  surface around it): the share of hole cells over those images.
* ``before_present_fraction``: the remnant test on the map after the first update of the split
  whose first part is exactly the images that show the absent object — the control that makes the
  absence mean something.
* ``<s>.stability.label_agreement`` / ``.id_agreement`` (``mapquality.stability_metrics``): the
  objects the first update's images observe and that never changed keep their labels and ``id``s
  from the first update to the last. A later update may re-gauge the map frame (a rebuild) and
  refine OBBs (mapper.md), so the two updates are first aligned by their common captures' camera
  poses (``split_alignment``) and the box figures stay in the detail: the OBB requirement is
  ``<s>.vs_one_update``'s.
* ``<s>.ids_persistent_fraction``: every id an update published (its ``-t full`` scene) for an
  object that never changed is, in every later update, still an object of a compatible label whose
  box, once the two updates are aligned, overlaps or nearly coincides with it
  (``mapquality.MATCH_IOU`` / ``MATCH_CENTRE_M``). An id the later map merged into another (its
  ``objects.json`` ``merged_into``, a lasting merge) resolves to that object first.
* ``<s>.vs_one_update.*``: one update versus that split, after the rigid alignment of the two
  maps' camera poses; ids may differ only where an earlier update of the split had published one
  (mapper.md), which ``id_agreement`` allows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.geometry import project
from oh_my_slam.core.types import Pose
from oh_my_slam.segmentation.detect import compatible
from oh_my_slam.segmentation.obb import OBB, obb_iou_upright
from oh_my_slam.tools.evaluate.groundtruth import Absent, MapUpdatePlan
from oh_my_slam.tools.evaluate.mapquality import (
    MATCH_CENTRE_M,
    MATCH_IOU,
    STABILITY_METRICS,
    split_alignment,
    stability_metrics,
)
from oh_my_slam.tools.evaluate.metrics import Metrics
from oh_my_slam.tools.evaluate.poses import capture_poses, capture_sources
from oh_my_slam.tools.evaluate.scene import DocObject, Json, doc_objects

PREFIX = "map_update"
BEFORE_METRIC = "before_present_fraction"
REGION_COVERAGE = 0.25  # share of the annotated region a projected box must cover
MIN_DEPTH_M = 0.05  # a box with a corner nearer than this (or behind) has no image footprint
HOLE_GRID = 8  # cells per side of the annotated region
HOLE_RING = 0.5  # the ring around the region, as a share of its size on each side
HOLE_DEPTH_REL = 0.25  # a region cell seeing this much farther than the ring sees through a hole
IMAGE_SUFFIXES = (".jpg", ".jpeg")
# first vs last update of a split: labels and ids only (boxes are refined, the frame re-gauged)
SPLIT_STABILITY = ("label_agreement", "id_agreement")
Camera = tuple[float, float, float, float, int, int]  # fx, fy, cx, cy, width, height


def split_name(sizes: tuple[int, ...] | list[int]) -> str:
    return "split_" + "_".join(str(n) for n in sizes)


def split_metric_ids(name: str) -> list[str]:
    return [f"{PREFIX}.{name}.{k}" for k in ("absent_fraction", "hole_fraction",
                                              "ids_persistent_fraction")] + \
        [f"{PREFIX}.{name}.stability.{k}" for k in SPLIT_STABILITY] + \
        [f"{PREFIX}.{name}.vs_one_update.{k}" for k in STABILITY_METRICS]


def metric_ids(splits: list[str] | None = None) -> list[str]:
    """Every ``map_update.*`` metric for the splits named ``splits``."""
    ids = [f"{PREFIX}.absent_fraction", f"{PREFIX}.hole_fraction", f"{PREFIX}.{BEFORE_METRIC}"]
    for s in splits or []:
        ids += split_metric_ids(s)
    return ids


def sequence_images(folder: Path) -> list[str]:
    """The images of a sequence folder by name (capture timestamps sort in capture order)."""
    return sorted(p.name for p in Path(folder).iterdir()
                  if p.is_file() and not p.name.startswith(".")
                  and p.suffix.lower() in IMAGE_SUFFIXES)


@dataclass
class MapView:
    """What the judgement needs of one map's ``-t full`` scene (and, for a final map, its cloud)."""

    objects: list[DocObject]
    poses: dict[str, Pose]  # camera-to-map, by capture file name
    sources: dict[int, str]  # frame key → capture file name
    camera: Camera | None
    cloud: NDArray[np.float64] | None = None  # map points (N, 3), map coordinates
    merged_into: dict[int, int] = field(default_factory=dict)  # lasting merges, old id → id

    @classmethod
    def of(cls, doc: Json, map_dir: Path, cloud: bool = False) -> MapView:
        return cls(doc_objects(doc), capture_poses(doc, map_dir), capture_sources(doc, map_dir),
                   _camera(doc), map_points(map_dir) if cloud else None, merges(map_dir))

    def resolve(self, oid: int) -> int:
        """``oid`` followed through the map's lasting merges."""
        seen: set[int] = set()
        while oid in self.merged_into and oid not in seen:
            seen.add(oid)
            oid = self.merged_into[oid]
        return oid

    def ids(self) -> set[int]:
        return {o.id for o in self.objects}


def merges(map_dir: Path) -> dict[int, int]:
    """The map's lasting merges (``objects.json`` → ``merged_into``), read as the update left
    them; provisional rebuild absorptions are not lasting and are not read."""
    import json

    try:
        data = json.loads((Path(map_dir) / "objects.json").read_text())
        return {int(k): int(v) for k, v in (data.get("merged_into") or {}).items()}
    except (OSError, ValueError, AttributeError, TypeError):
        return {}


def map_points(map_dir: Path) -> NDArray[np.float64] | None:
    """The stored map cloud's points (read-only), None when the map has none."""
    from oh_my_slam.core.errors import OhMySlamError
    from oh_my_slam.mapping import export, store

    try:
        xyz = export.map_cloud(store.MapReader(map_dir)).xyz
    except (OSError, ValueError, KeyError, OhMySlamError):
        return None
    return np.asarray(xyz, np.float64) if len(xyz) else None


def _camera(doc: Json) -> Camera | None:
    """The pinhole intrinsics of the first stream that has them (the mapper has one camera)."""
    for stream in (doc.get("openlabel", {}).get("streams") or {}).values():
        pin = (stream.get("stream_properties") or {}).get("intrinsics_pinhole")
        if pin and len(pin.get("camera_matrix", ())) >= 7:
            k = pin["camera_matrix"]  # 3 x 4, row-major
            return (float(k[0]), float(k[5]), float(k[2]), float(k[6]),
                    int(pin["width_px"]), int(pin["height_px"]))
    return None


def region_coverage(box: OBB, pose: Pose, camera: Camera,
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


def hole_cells(xyz: NDArray[np.float64], pose: Pose, camera: Camera,
               region: tuple[float, float, float, float]) -> dict[str, Any] | None:
    """The hole test of one image (see the module docstring): {"cells", "holes", "ring_depth_m"},
    None when no region cell is in the image or the ring sees no surface."""
    fx, fy, cx, cy, w, h = camera
    x0, y0, x1, y1 = region[0] * w, region[1] * h, region[2] * w, region[3] * h
    cw, ch = (x1 - x0) / HOLE_GRID, (y1 - y0) / HOLE_GRID
    ring = int(np.ceil(HOLE_GRID * HOLE_RING))
    n = HOLE_GRID + 2 * ring
    ox, oy = x0 - ring * cw, y0 - ring * ch
    cam = pose.inverse().apply(xyz)
    front = cam[:, 2] > MIN_DEPTH_M
    cam = cam[front]
    u = fx * cam[:, 0] / cam[:, 2] + cx
    v = fy * cam[:, 1] / cam[:, 2] + cy
    i = np.floor((u - ox) / cw).astype(np.int64)
    j = np.floor((v - oy) / ch).astype(np.int64)
    inside = (i >= 0) & (i < n) & (j >= 0) & (j < n) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    flat = np.full(n * n, np.inf)
    np.minimum.at(flat, (j[inside] * n + i[inside]), cam[inside, 2])
    zbuf = flat.reshape(n, n)
    centres_u = ox + (np.arange(n) + 0.5) * cw
    centres_v = oy + (np.arange(n) + 0.5) * ch
    in_image = (centres_v[:, None] >= 0) & (centres_v[:, None] < h) \
        & (centres_u[None, :] >= 0) & (centres_u[None, :] < w)
    core = np.zeros((n, n), bool)
    core[ring:ring + HOLE_GRID, ring:ring + HOLE_GRID] = True
    ring_z = zbuf[~core & in_image]
    ring_z = ring_z[np.isfinite(ring_z)]
    cells = zbuf[core & in_image]
    if not len(cells) or not len(ring_z):
        return None
    ref = float(np.median(ring_z))
    holes = int(np.sum(~np.isfinite(cells) | (cells > ref * (1.0 + HOLE_DEPTH_REL))))
    return {"cells": len(cells), "holes": holes, "ring_depth_m": round(ref, 3)}


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


def _holes(m: Metrics, mid: str, plan: MapUpdatePlan, view: MapView) -> None:
    """``hole_fraction`` of one final map (see the module docstring)."""
    if view.cloud is None or view.camera is None:
        m.add(mid, None, error="the map has no cloud or no camera intrinsics")
        return
    per: dict[str, Any] = {}
    for a in plan.absent:
        for name, region in a.seen_in.items():
            if name not in view.poses:
                per[f"{a.label} in {name}"] = "not registered"
                continue
            res = hole_cells(view.cloud, view.poses[name], view.camera, region)
            per[f"{a.label} in {name}"] = res if res is not None else "outside the image"
    done = [v for v in per.values() if isinstance(v, dict)]
    cells = sum(v["cells"] for v in done)
    m.add(mid, sum(v["holes"] for v in done) / cells if cells else None, per,
          error="no image that showed an absent object is registered in the map")


def unchanged(plan: MapUpdatePlan) -> Any:
    absent = [a.label for a in plan.absent]

    def keep(o: DocObject) -> bool:
        gone = any(compatible(label, a) for label in _labels(o) for a in absent)
        wanted = not plan.stable or any(compatible(o.label, s) for s in plan.stable)
        return wanted and not gone

    return keep


def _same_object(a: DocObject, b: DocObject, T_b_a: Pose) -> bool:
    """``b`` is the object ``a`` was: a compatible label and a box that overlaps or nearly
    coincides with ``a``'s carried into ``b``'s map frame by ``T_b_a`` (the stability pairing's
    admissibility)."""
    if not compatible(a.label, b.label):
        return False
    box_a, bb = a.obb(), b.obb()
    if box_a is None or bb is None:
        return True
    ba = box_a.transformed(T_b_a)
    d = float(np.linalg.norm(ba.center - bb.center))
    return d <= MATCH_CENTRE_M or obb_iou_upright(ba, bb, samples=4000) >= MATCH_IOU


def ids_persistent(m: Metrics, mid: str, plan: MapUpdatePlan, views: list[MapView]
                   ) -> list[dict[str, Any]]:
    """``<s>.ids_persistent_fraction`` (see the module docstring); returns the broken ids."""
    keep = unchanged(plan)
    checks, broken = 0, []
    for k, view in enumerate(views[:-1]):
        for o in view.objects:
            if not keep(o):
                continue
            for j, later in enumerate(views[k + 1:], start=k + 2):
                checks += 1
                oid = later.resolve(o.id)
                now = next((x for x in later.objects if x.id == oid), None)
                if now is None or not _same_object(o, now, _aligned(later, view)):
                    broken.append({"id": o.id, "label": o.label, "published_by_update": k + 1,
                                   "update": j, "now": None if now is None else now.label,
                                   **({"merged_into": oid} if oid != o.id else {})})
    m.add(mid, 1.0 - len(broken) / checks if checks else None,
          {"checks": checks, "broken": broken[:20]},
          error="no update published an unchanged object")
    return broken


def _aligned(to: MapView, frm: MapView) -> Pose:
    """``T_to_from`` from the captures registered in both updates (identity when none is)."""
    try:
        return split_alignment(to.poses, frm.poses)
    except ValueError:
        return Pose.identity()


@dataclass
class SplitMaps:
    """One split of the sequence: the images of each update and the map after each update."""

    sizes: tuple[int, ...]
    parts: list[list[str]]
    views: list[MapView]  # after each update (the last one with its cloud)

    @property
    def name(self) -> str:
        return split_name(self.sizes)


def parts_of(images: list[str], sizes: tuple[int, ...]) -> list[list[str]]:
    if sum(sizes) != len(images):
        raise ValueError(f"split {'+'.join(map(str, sizes))} does not cover the {len(images)} "
                         "images of the sequence")
    edges = [int(e) for e in np.cumsum((0, *sizes))]
    return [images[a:b] for a, b in pairwise(edges)]


def map_update_metrics(m: Metrics, plan: MapUpdatePlan, images: list[str], single: MapView,
                       splits: list[SplitMaps], failed: dict[str, str] | None = None
                       ) -> dict[str, Any]:
    """Record the ``map_update.*`` metrics: ``single`` is the map of the whole sequence in one
    update (with its cloud), ``splits`` the split maps that were built (``failed``: those that
    were not, with the reason). Returns the rows for the report."""
    early = plan.before_images(images)
    keep = unchanged(plan)
    out: dict[str, Any] = {"before_images": early, "splits": {}}
    with m.expect(f"{PREFIX}.absent_fraction", f"{PREFIX}.hole_fraction"):
        rows = [{"label": a.label, "final": remnants(single, a)} for a in plan.absent]
        _absent(m, f"{PREFIX}.absent_fraction", rows, "final", single)
        _holes(m, f"{PREFIX}.hole_fraction", plan, single)
        out["absent"] = rows
    control = next((s for s in splits if s.parts[0] == early), None)
    with m.expect(f"{PREFIX}.{BEFORE_METRIC}"):
        if control is None:
            m.add(f"{PREFIX}.{BEFORE_METRIC}", None, error="no split map's first update is "
                  "exactly the images that show the absent objects")
        else:
            before = [{"label": a.label, "before": remnants(control.views[0], a)}
                      for a in plan.absent]
            m.add(f"{PREFIX}.{BEFORE_METRIC}", sum(bool(r["before"]) for r in before) / len(before),
                  {"split": control.name,
                   "missing_before": [r["label"] for r in before if not r["before"]]})
            out["before"] = before
    for name, why in (failed or {}).items():
        m.fail(split_metric_ids(name), why)
    for s in splits:
        ids = split_metric_ids(s.name)
        row: dict[str, Any] = {"sizes": list(s.sizes), "parts": s.parts}
        final = s.views[-1]
        with m.expect(*ids):
            rows = [{"label": a.label, "final": remnants(final, a)} for a in plan.absent]
            _absent(m, ids[0], rows, "final", final)
            _holes(m, ids[1], plan, final)
            row["absent"] = rows
            row["ids_broken"] = ids_persistent(m, ids[2], plan, s.views)
            first_images = set(s.parts[0])
            kept = [o for o in final.objects
                    if keep(o) and any(final.sources.get(f) in first_images for f in o.frames)]
            scratch = Metrics()  # the box figures of an update-to-update comparison: detail only
            row["stability"] = stability_metrics(
                scratch, "x", kept, [o for o in s.views[0].objects if keep(o)],
                _aligned(final, s.views[0]))
            boxes = {k: scratch.items[f"x.{k}"].value for k in STABILITY_METRICS
                     if k not in SPLIT_STABILITY}
            for k in SPLIT_STABILITY:
                x = scratch.items[f"x.{k}"]
                m.add(f"{PREFIX}.{s.name}.stability.{k}", x.value,
                      {**(x.detail or {}), "boxes_after_alignment": boxes}, error=x.error)
            published = set().union(*(v.ids() for v in s.views[:-1]))
            row["vs_one_update"] = stability_metrics(
                m, f"{PREFIX}.{s.name}.vs_one_update", single.objects, final.objects,
                split_alignment(single.poses, final.poses), published)
        out["splits"][s.name] = row
    return out
