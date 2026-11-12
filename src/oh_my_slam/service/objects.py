"""The objects read (O8): ``GET /api/maps/{name}/objects`` (map.md §5) [``Objects``].

An anchored map commits each version's object records and surfaces with it
(``versions/<n>.objects.json``, written by ``prior_first`` from ``mapping.anchored_objects``);
this module adds each object's ``region`` from the version's default grid (``nav.object_regions``,
5 cm cells, the default band, the answer a default ``GET grid`` shares) and keeps the answer per
version (``derived/<n>/objects.json``, and in memory: ``DerivedVersion.answer``), computed when
the version is prepared, before it is published, so a read only returns it.

A map of the older mapper has no such records: its objects (``derived`` meta) are listed from
what that mapper records (README "Objects of the older mapper"): its score and label votes (the
``detection_votes`` shares: each label's votes over its observing frames); the
observed colour and the centre's spread from the object's points in the version's cloud; the
versions from the mapper's update ids, translated by the version record's ``update_versions``
(each update id of the version's line and the version that committed it: a version with no
mapper update and a revert do not shift them, ``update.update_versions``). That mapper does not record which views bound a box, so
every extent is ``partial`` (not seen in full). An object without those records, or without
points in the cloud, is not listed: the read never makes a field up."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from oh_my_slam.core.atomic import atomic_write_json
from oh_my_slam.mapping.anchored_objects import extent_from
from oh_my_slam.segmentation.api import color_for_id
from oh_my_slam.segmentation.detect import PASSAGE_LABELS, door_leaf
from oh_my_slam.service import nav
from oh_my_slam.service.derived import Derived, DerivedVersion
from oh_my_slam.service.store import Store

CACHE = "objects.json"


def _mapper_records(dv: DerivedVersion) -> dict[str, Any]:
    """The objects of a map of the older mapper, from what it records."""
    cloud = dv.cloud()
    sig = [float(f["sigma_pos_m"]) for f in dv.record["frames"] if f.get("sigma_pos_m")]
    pose_sigma = float(np.median(sig)) if sig else 0.0
    # each keyframe's camera centre by the mapper's frame name: the views of its objects
    cams = {f["internal"]: np.asarray(f["T_map_cam"], np.float64)[:3, 3]
            for f in dv.record["frames"] if f.get("internal")}
    # the mapper's update ids as the versions of the history that committed them (a version
    # recorded before this: the ids themselves)
    updates = {int(k): int(v) for k, v in (dv.record.get("update_versions") or {}).items()}

    def as_version(u: int) -> int:
        return updates.get(u, u)

    out = []
    for o in dv.meta.get("objects", []):
        oid = int(o.get("id") or 0)
        if oid < 1 or o.get("score") is None or not o.get("created_update"):
            continue
        if o["label"] in PASSAGE_LABELS or (o["label"] == "door" and not door_leaf(o["size_m"])):
            continue  # an opening is a passage, and a door object is a leaf (map.md §5)
        mine = cloud["object_id"] == oid
        if not mine.any():
            continue
        rgb = np.median(cloud["rgb"][mine].astype(np.float64), axis=0)
        xyz = cloud["xyz"][mine].astype(np.float64)
        spread = float(np.sqrt(((xyz - xyz.mean(0)) ** 2).sum(1).mean()))
        n = max(int(o.get("observations") or 1), 1)
        votes = o.get("label_votes") or {}
        labels = sorted(votes, key=lambda k: (-float(votes[k]), k)) or [o["label"]]
        # that mapper's per-label votes over its observing frames (map.md §5)
        shares = [{"label": k, "share": round(min(max(float(votes[k]) / n, 0.0), 1.0), 4)}
                  for k in votes]
        first = min(max(as_version(int(o["created_update"])), 1), dv.version)
        last = min(max(as_version(int(o.get("last_seen_update") or o["created_update"])),
                       first), dv.version)
        out.append({"id": oid, "label": o["label"], "detected_as": labels,
                    "score": round(min(max(float(o["score"]), 0.0), 1.0), 4),
                    "centre_m": o["centre_m"], "size_m": o["size_m"],
                    "yaw_deg": round(float(o["yaw_deg"]), 3),
                    "sigma_centre_m": round(math.hypot(pose_sigma, spread / math.sqrt(n)), 4),
                    "extent_seen": _mapper_extent(o, cams),
                    "mean_rgb": [round(c) for c in rgb],
                    "detection_votes": sorted(shares, key=lambda e: (-e["share"], e["label"])),
                    "state": "confirmed", "moved_from_m": None,
                    "first_seen_version": first, "last_seen_version": last,
                    "supported_by": None, "inside_of": None, "seen_from": []})
    return {"objects": out, "surfaces": []}


def _mapper_extent(o: dict[str, Any], cams: dict[str, Any]) -> dict[str, str]:
    """``extent_seen`` of an older-mapper object from the views it recorded (``views``: each its
    keyframe, its points' centroid and whether its mask kept off the image border), the rule of
    the prior-first engine (``anchored_objects.extent_from``) on its yaw's axes, with the
    support plane that mapper measured under it (``support_z``) bounding its bottom; ``partial``
    on every axis when the builder recorded none (a map derived before they were read)."""
    views = []
    for v in o.get("views", []):
        c = cams.get(v["frame"])
        if c is None:
            continue
        ray = np.asarray(v["centroid"], np.float64) - c
        n = float(np.linalg.norm(ray))
        if n > 0:
            views.append((ray / n, [not v["uncut"]] * 4))
    a = math.radians(float(o["yaw_deg"]))
    R = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0],
                  [0.0, 0.0, 1.0]])
    return extent_from(R, views, bool(o.get("supported")))


def compute(store: Store, ids: nav.Ids, dv: DerivedVersion,
            files: dict[str, Any] | None = None) -> dict[str, Any]:
    """The ``Objects`` body of a derived version (without ``schema``), from its committed
    records (``files``, before it is published; else the store's)."""
    p = store.version_file(dv.name, dv.version, CACHE)
    data = (files or {}).get(CACHE)
    if data is None:
        data = json.loads(p.read_text()) if p.exists() else _mapper_records(dv)
    objects = data["objects"]
    regions: dict[int, int | None] = {}
    if objects:
        answer = nav.grid_answer(dv, ids, nav.DEFAULT_CELL_M, *nav.DEFAULT_Z_BAND, None,
                                 nav.HEIGHT_CAP_M)
        regions = nav.object_regions(answer, objects,
                                     {o["id"]: o.get("seen_from", []) for o in objects})
    return {"map": dv.name, "map_version": dv.version,
            "scale_source": dv.record["scale_source"],
            "scale_sigma_rel": dv.record["scale_sigma_rel"],
            "objects": [{**{k: v for k, v in o.items() if k != "seen_from"},
                         "color": list(color_for_id(o["id"])), "region": regions.get(o["id"])}
                        for o in objects],
            "surfaces": data["surfaces"]}


def body(store: Store, derived: Derived, ids: nav.Ids, dv: DerivedVersion,
         files: dict[str, Any] | None = None) -> dict[str, Any]:
    """The answer of a version: computed once (``prepare``'s hook, with its ``files``), kept
    on disk and in memory."""
    def load() -> dict[str, Any]:
        path = derived.dir(dv.name) / str(dv.version) / CACHE
        if path.exists():
            out: dict[str, Any] = json.loads(path.read_text())
            return out
        out = compute(store, ids, dv, files)
        atomic_write_json(path, out)
        return out

    got: dict[str, Any] = dv.answer(("objects",), load)
    return got
