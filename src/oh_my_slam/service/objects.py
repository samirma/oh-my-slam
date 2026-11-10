"""The objects read (O8): ``GET /api/maps/{name}/objects`` (map.md §5) [``Objects``].

An anchored map commits each version's object records and surfaces with it
(``versions/<n>.objects.json``, written by ``prior_first`` from ``mapping.anchored_objects``);
this module adds each object's ``region`` from the version's default grid (``nav.object_regions``,
5 cm cells, the default band) and keeps the answer per version (``derived/<n>/objects.json``),
computed by the derivation worker after each commit, so a read only loads it.

A map of the older mapper has no such records: its objects (``derived`` meta) are listed from
what that mapper records (README "Objects of the older mapper"): its score and label votes; the
observed colour and the centre's spread from the object's points in the version's cloud; the
versions from the mapper's update ids, which are the versions on a map the service built (one
mapper update per committed version). That mapper does not record which views bound a box, so
every extent is ``partial`` (not seen in full). An object without those records, or without
points in the cloud, is not listed: the read never makes a field up."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from oh_my_slam.core.atomic import atomic_write_json
from oh_my_slam.segmentation.api import color_for_id
from oh_my_slam.service import nav
from oh_my_slam.service.derived import Derived, DerivedVersion
from oh_my_slam.service.store import Store

CACHE = "objects.json"


def _mapper_records(dv: DerivedVersion) -> dict[str, Any]:
    """The objects of a map of the older mapper, from what it records."""
    cloud = dv.cloud()
    sig = [float(f["sigma_pos_m"]) for f in dv.record["frames"] if f.get("sigma_pos_m")]
    pose_sigma = float(np.median(sig)) if sig else 0.0
    out = []
    for o in dv.meta.get("objects", []):
        oid = int(o.get("id") or 0)
        if oid < 1 or o.get("score") is None or not o.get("created_update"):
            continue
        mine = cloud["object_id"] == oid
        if not mine.any():
            continue
        rgb = np.median(cloud["rgb"][mine].astype(np.float64), axis=0)
        xyz = cloud["xyz"][mine].astype(np.float64)
        spread = float(np.sqrt(((xyz - xyz.mean(0)) ** 2).sum(1).mean()))
        n = max(int(o.get("observations") or 1), 1)
        votes = o.get("label_votes") or {}
        labels = sorted(votes, key=lambda k: (-float(votes[k]), k)) or [o["label"]]
        first = min(max(int(o["created_update"]), 1), dv.version)
        last = min(max(int(o.get("last_seen_update") or first), first), dv.version)
        out.append({"id": oid, "label": o["label"], "detected_as": labels,
                    "score": round(min(max(float(o["score"]), 0.0), 1.0), 4),
                    "centre_m": o["centre_m"], "size_m": o["size_m"],
                    "yaw_deg": round(float(o["yaw_deg"]), 3),
                    "sigma_centre_m": round(math.hypot(pose_sigma, spread / math.sqrt(n)), 4),
                    "extent_seen": {"x": "partial", "y": "partial", "z": "partial"},
                    "mean_rgb": [round(c) for c in rgb],
                    "state": "confirmed", "moved_from_m": None,
                    "first_seen_version": first, "last_seen_version": last,
                    "supported_by": None, "inside_of": None, "seen_from": []})
    return {"objects": out, "surfaces": []}


def compute(store: Store, ids: nav.Ids, dv: DerivedVersion) -> dict[str, Any]:
    """The ``Objects`` body of a derived version (without ``schema``)."""
    p = store.version_file(dv.name, dv.version, "objects.json")
    data = json.loads(p.read_text()) if p.exists() else _mapper_records(dv)
    objects = data["objects"]
    regions: dict[int, int | None] = {}
    if objects:
        answer = nav.grid(dv, ids, nav.DEFAULT_CELL_M, *nav.DEFAULT_Z_BAND, None)
        regions = nav.object_regions(answer, objects,
                                     {o["id"]: o.get("seen_from", []) for o in objects})
    return {"map": dv.name, "map_version": dv.version,
            "scale_source": dv.record["scale_source"],
            "scale_sigma_rel": dv.record["scale_sigma_rel"],
            "objects": [{**{k: v for k, v in o.items() if k != "seen_from"},
                         "color": list(color_for_id(o["id"])), "region": regions.get(o["id"])}
                        for o in objects],
            "surfaces": data["surfaces"]}


def body(store: Store, derived: Derived, ids: nav.Ids, dv: DerivedVersion) -> dict[str, Any]:
    """The answer of a version: kept per version once computed."""
    path = derived.dir(dv.name) / str(dv.version) / CACHE
    if path.exists():
        out: dict[str, Any] = json.loads(path.read_text())
        return out
    out = compute(store, ids, dv)
    atomic_write_json(path, out)
    return out
