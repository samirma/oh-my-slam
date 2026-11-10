"""A map's published objects and keyframes as the tests read them, from the map folder itself
(the mapper's result carries no scene document any more)."""

from __future__ import annotations

from pathlib import Path

from oh_my_slam.mapping import store
from oh_my_slam.mapping.export import map_objects
from oh_my_slam.segmentation.api import SceneObject


def objects(mdir: Path) -> list[SceneObject]:
    """The map's published objects, in id order."""
    return sorted(map_objects(store.MapReader(mdir))[1], key=lambda o: o.id)


def by_id(mdir: Path) -> dict[int, SceneObject]:
    return {o.id: o for o in objects(mdir)}


def by_label(mdir: Path) -> dict[str, list[SceneObject]]:
    out: dict[str, list[SceneObject]] = {}
    for o in objects(mdir):
        out.setdefault(o.label, []).append(o)
    return out


def frames(mdir: Path) -> list[store.FrameRecord]:
    """The map's keyframes, in index order."""
    return sorted(store.MapReader(mdir).frames, key=lambda r: r.index)
