"""Object ids across a map's whole version history (map.md §2 Revert, §5 ``id``).

Each engine numbers new objects from a counter kept in its mapper map: the prior-first engine's
``objects.json`` ``next_id`` (``mapping.anchored_objects.ObjectMap``), the older mapper's
``map.json`` ``next_object_id`` and ``objects.json`` ``next_id`` (``mapping.objects``). A revert
clones the mapper map of its target, whose counter is the target's: an object created after the
revert would take an id a reverted-away version held. ``Store.revert`` therefore raises the
clone's counter to the map's high-water mark (``next_id`` of the map it replaces, and the mark an
earlier revert recorded), so ids are unique over the whole history and never reused."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oh_my_slam.core.atomic import atomic_write_json

OBJECTS_JSON = "objects.json"  # both engines' (``anchored_objects``, ``mapping.store``)
MAP_JSON = "map.json"  # the older mapper's (``mapping.store.MAP_JSON``)


def _read(p: Path) -> dict[str, Any] | None:
    try:
        out: dict[str, Any] = json.loads(p.read_text())
        return out
    except (OSError, ValueError):
        return None


def next_id(mapper: Path | None) -> int:
    """The id the mapper map ``mapper`` gives its next new object (1 when it has none)."""
    if mapper is None:
        return 1
    out = 1
    objs = _read(mapper / OBJECTS_JSON)
    if objs is not None:
        out = max(out, int(objs.get("next_id") or 1))
    meta = _read(mapper / MAP_JSON)
    if meta is not None:
        out = max(out, int(meta.get("next_object_id") or 1))
    return out


def reserve(mapper: Path, first: int) -> None:
    """Number the new objects of the mapper map ``mapper`` (a clone a revert commits) from at
    least ``first``."""
    from oh_my_slam.service.backend import is_anchored

    if next_id(mapper) >= first:
        return
    objs = _read(mapper / OBJECTS_JSON)
    if objs is not None:
        atomic_write_json(mapper / OBJECTS_JSON, {**objs, "next_id": first})
    elif is_anchored(mapper):  # no object yet: an empty object map numbering from ``first``
        from oh_my_slam.mapping.anchored_objects import ObjectMap

        om = ObjectMap(mapper)
        om.next_id = first
        om.save()
    meta = _read(mapper / MAP_JSON)
    if meta is not None:  # the older mapper's count, read when it has no objects file
        atomic_write_json(mapper / MAP_JSON, {**meta, "next_object_id": first})
