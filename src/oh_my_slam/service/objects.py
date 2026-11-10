"""The objects read (O8): ``GET /api/maps/{name}/objects`` (map.md §5) [``Objects``].

An anchored map commits each version's object records and surfaces with it
(``versions/<n>.objects.json``, written by ``prior_first`` from ``mapping.anchored_objects``);
this module adds each object's ``region`` from the version's default grid (``nav.object_regions``,
5 cm cells, the default band) and keeps the answer per version (``derived/<n>/objects.json``),
computed by the derivation worker after each commit, so a read only loads it. A map of the older
mapper has no such records: its objects (``derived`` meta) are listed with what it knows."""

from __future__ import annotations

import json
from typing import Any

from oh_my_slam.core.atomic import atomic_write_json
from oh_my_slam.segmentation.api import color_for_id
from oh_my_slam.service import nav
from oh_my_slam.service.derived import Derived, DerivedVersion
from oh_my_slam.service.store import Store

CACHE = "objects.json"


def _mapper_records(dv: DerivedVersion) -> dict[str, Any]:
    """The objects of a map of the older mapper, without what it does not keep."""
    out = []
    for o in dv.meta.get("objects", []):
        oid = int(o.get("id") or 0)
        if oid < 1:
            continue
        out.append({"id": oid, "label": o["label"], "detected_as": [o["label"]], "score": 0.5,
                    "centre_m": o["centre_m"], "size_m": o["size_m"],
                    "yaw_deg": round(float(o["yaw_deg"]), 3), "sigma_centre_m": 0.05,
                    "extent_seen": {"x": "partial", "y": "partial", "z": "partial"},
                    "mean_rgb": [128, 128, 128],
                    "state": "confirmed", "moved_from_m": None,
                    "first_seen_version": dv.version, "last_seen_version": dv.version,
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
