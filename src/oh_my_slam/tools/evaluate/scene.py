"""Reading the commands' OpenLABEL outputs: objects (label, score, colour, cuboid, observing
frames) and camera poses (``camera_*_to_map`` transforms of the frames)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.geometry import quat_to_rot
from oh_my_slam.core.types import Pose
from oh_my_slam.segmentation.obb import OBB

Json = dict[str, Any]


@dataclass(frozen=True)
class DocObject:
    id: int
    label: str
    score: float | None
    color_hex: str | None
    color: tuple[int, ...] | None
    cuboid: tuple[float, ...] | None  # x, y, z, qx, qy, qz, qw, sx, sy, sz
    frames: frozenset[int] = field(default_factory=frozenset)  # frames that observed it
    labels: tuple[str, ...] = ()  # ``detected_as``: every label it was detected as (maps)

    def obb(self) -> OBB | None:
        if self.cuboid is None or len(self.cuboid) != 10:
            return None
        v = np.asarray(self.cuboid, dtype=np.float64)
        return OBB(v[:3], quat_to_rot(v[3:7]), v[7:10])


def _named(items: list[Json] | None, name: str) -> Any:
    return next((it.get("val") for it in items or [] if it.get("name") == name), None)


def _frames(intervals: list[Json] | None) -> frozenset[int]:
    out: set[int] = set()
    for fi in intervals or []:
        out.update(range(int(fi["frame_start"]), int(fi["frame_end"]) + 1))
    return frozenset(out)


def doc_objects(doc: Json) -> list[DocObject]:
    """The objects of an OpenLABEL document, by ascending id."""
    out = []
    for key, obj in (doc.get("openlabel", {}).get("objects") or {}).items():
        data = obj.get("object_data") or {}
        score = _named(data.get("num"), "score")
        vec = _named(data.get("vec"), "color")
        cub = (data.get("cuboid") or [{}])[0].get("val")
        out.append(DocObject(
            id=int(key), label=str(obj.get("type", "")),
            score=None if score is None else float(score),
            color_hex=_named(data.get("text"), "color_hex"),
            color=None if vec is None else tuple(int(v) for v in vec),
            cuboid=None if cub is None else tuple(float(v) for v in cub),
            frames=_frames(obj.get("frame_intervals")),
            labels=tuple(str(v) for v in _named(data.get("vec"), "detected_as") or ())))
    return sorted(out, key=lambda o: o.id)


def _pose(data: Json) -> Pose:
    if "matrix4x4" in data:
        return Pose.from_matrix(np.asarray(data["matrix4x4"], dtype=np.float64).reshape(4, 4))
    return Pose(quat_to_rot(np.asarray(data["quaternion"], dtype=np.float64)),
                np.asarray(data["translation"], dtype=np.float64))


def frame_poses(doc: Json, target: str = "map") -> dict[int, Pose]:
    """Camera-to-``target`` pose of every frame that carries one, by frame key."""
    out = {}
    for key, fr in (doc.get("openlabel", {}).get("frames") or {}).items():
        for tr in ((fr.get("frame_properties") or {}).get("transforms") or {}).values():
            if tr.get("dst") == target and "transform_src_to_dst" in tr:
                out[int(key)] = _pose(tr["transform_src_to_dst"])
                break
    return out


def frame_property(doc: Json, name: str) -> dict[int, Any]:
    """``frame_properties[name]`` of every frame that has it, by frame key."""
    return {int(k): fr["frame_properties"][name]
            for k, fr in (doc.get("openlabel", {}).get("frames") or {}).items()
            if name in (fr.get("frame_properties") or {})}


def forward(R: NDArray[Any]) -> NDArray[np.float64]:
    """Viewing direction (camera +z, OpenCV axes) in the parent frame."""
    return np.asarray(R, dtype=np.float64)[:, 2]


def yaw_deg(R: NDArray[Any]) -> float:
    """Heading of the viewing direction about the parent's z (up) axis, counter-clockwise (left)
    positive, from the parent's +x axis."""
    f = forward(R)
    return float(np.degrees(np.arctan2(f[1], f[0])))


def pitch_deg(R: NDArray[Any]) -> float:
    """Elevation of the viewing direction above the parent's horizontal plane (up positive)."""
    return float(np.degrees(np.arcsin(np.clip(forward(R)[2], -1.0, 1.0))))
