"""Object and surface ids across a map's whole version history (map.md §2 Revert, §5 ``id``).

Each engine numbers new objects and surfaces from counters kept in its mapper map: the prior-first
engine's ``objects.json`` ``next_id`` and ``next_surface_id``
(``mapping.anchored_objects.ObjectMap``), the older mapper's ``map.json`` ``next_object_id`` and
``objects.json`` ``next_id`` (``mapping.objects``; it numbers no surfaces). A revert clones the
mapper map of its target, whose counters are the target's: an object or surface created after
the revert would take an id a reverted-away version held. ``Store.revert`` therefore raises the
clone's counters to the map's high-water marks (``Ids``: those of the map it replaces, of its
target's, and the marks an earlier revert recorded in ``map.json``), so ids are unique over the
whole history and never reused. A revert whose target has no stored mapper map (a map from
before mapper maps were kept per version) leaves none to raise: the marks stay recorded, and the
next job that builds a new mapper map seeds it with them (``seed``)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oh_my_slam.core.atomic import atomic_write_json

OBJECTS_JSON = "objects.json"  # both engines' (``anchored_objects``, ``mapping.store``)
MAP_JSON = "map.json"  # the older mapper's (``mapping.store.MAP_JSON``)


@dataclass(frozen=True)
class Ids:
    """The next object and surface ids of a mapper map, or a map's high-water marks."""

    objects: int = 1
    surfaces: int = 1

    def max(self, other: Ids) -> Ids:
        return Ids(max(self.objects, other.objects), max(self.surfaces, other.surfaces))


def _read(p: Path) -> dict[str, Any] | None:
    try:
        out: dict[str, Any] = json.loads(p.read_text())
        return out
    except (OSError, ValueError):
        return None


def next_ids(mapper: Path | None) -> Ids:
    """The ids the mapper map ``mapper`` gives its next new object and surface (1 when none)."""
    if mapper is None:
        return Ids()
    objects, surfaces = 1, 1
    objs = _read(mapper / OBJECTS_JSON)
    if objs is not None:
        objects = max(objects, int(objs.get("next_id") or 1))
        surfaces = max(surfaces, int(objs.get("next_surface_id") or 1))
    meta = _read(mapper / MAP_JSON)
    if meta is not None:
        objects = max(objects, int(meta.get("next_object_id") or 1))
    return Ids(objects, surfaces)


def next_id(mapper: Path | None) -> int:
    """The id the mapper map ``mapper`` gives its next new object."""
    return next_ids(mapper).objects


def _empty_object_map(mapper: Path, first: Ids) -> None:
    from oh_my_slam.mapping.anchored_objects import ObjectMap

    om = ObjectMap(mapper)
    om.next_id, om.next_surface_id = first.objects, first.surfaces
    om.save()


def reserve(mapper: Path, first: Ids | int) -> None:
    """Number the new objects and surfaces of the mapper map ``mapper`` (a clone a revert
    commits) from at least ``first``."""
    from oh_my_slam.service.backend import is_anchored

    first = first if isinstance(first, Ids) else Ids(first)
    have = next_ids(mapper)
    if have.objects >= first.objects and have.surfaces >= first.surfaces:
        return
    want = have.max(first)
    objs = _read(mapper / OBJECTS_JSON)
    if objs is not None:
        out = {**objs, "next_id": want.objects}
        if "next_surface_id" in objs or is_anchored(mapper):
            out["next_surface_id"] = want.surfaces
        atomic_write_json(mapper / OBJECTS_JSON, out)
    elif is_anchored(mapper):  # no object yet: an empty object map numbering from ``first``
        _empty_object_map(mapper, want)
    meta = _read(mapper / MAP_JSON)
    if meta is not None:  # the older mapper's count, read when it has no objects file
        atomic_write_json(mapper / MAP_JSON, {**meta, "next_object_id": want.objects})


def seed(mapper: Path, first: Ids, anchored: bool) -> None:
    """A new mapper map a job is about to build in ``mapper`` (not there yet, or empty),
    numbering its objects and surfaces from ``first`` (the map's high-water marks)."""
    mapper.mkdir(parents=True, exist_ok=True)
    if anchored:
        _empty_object_map(mapper, first)
        return
    from oh_my_slam.version import MAP_FORMAT_VERSION

    # the older mapper's new map (``mapping.store.read_meta_or_default``), its count seeded
    atomic_write_json(mapper / MAP_JSON, {
        "format_version": MAP_FORMAT_VERSION, "created_at": time.time(), "update_count": 0,
        "next_object_id": first.objects, "next_frame_index": 0, "updates": []})
