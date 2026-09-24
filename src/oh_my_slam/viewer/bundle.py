"""Everything the browser shows, gathered only through the owning packages' APIs.

* ``view.sh -i``: one reconstruction + segmentation run (``segmentation.api``, through the
  inference server) gives the OpenLABEL scene, the catalogue, the segmented image and the cloud
  *source* of the image.
* ``view.sh -m``: the read-only map store (``mapping.export``) gives the same for a map; no
  inference server is involved and nothing is written.

No point cloud is stored here: every cloud the page asks for is derived on request from the source
kept in memory, by ``segmentation.cloud.derive_cloud`` with the §2.2 attributes the page sends
(validated by ``core.cloud_attrs``), so changing a control never re-runs inference. Camera poses are
read from the scene description, so the viewer shows exactly the poses the JSON states. No
inference, point-cloud generation, OBB fitting, identity or colour logic lives here.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.cloud_attrs import (
    CloudAttrs,
    CloudScope,
    applicable,
    parse_cloud_attrs,
)
from oh_my_slam.core.errors import UsageError
from oh_my_slam.core.geometry import quat_to_rot, rotation_between
from oh_my_slam.core.ply import PointCloud
from oh_my_slam.reconstruction.gravity import DEFAULT_UP_CAM
from oh_my_slam.segmentation.cloud import CloudSource, derive_cloud, scope_of

Json = dict[str, Any]

# Browsers stay responsive up to a few million points; a larger derived cloud is thinned for display
# only, deterministically (every k-th point in derivation order), and the page says so.
MAX_DISPLAY_POINTS = 3_000_000

# Spec §2.5: these attributes concern PLY files only and have no control in the viewer.
PLY_ONLY = frozenset({"label", "encoding"})

# Presentation of the numeric controls only — slider range, step, unit and the value that means
# "filter off". Which controls exist, their accepted values, defaults and validation all come from
# core.cloud_attrs; ``None`` as a maximum means "the image's depth extent".
_SLIDERS: dict[str, tuple[float, float | None, float, str, float | None]] = {
    "stride": (1, 16, 1, "px", None),
    "min-depth": (0.0, None, 0.05, "m", 0.0),
    "max-depth": (0.05, None, 0.05, "m", math.inf),
    "edge": (0.0, 0.5, 0.005, "", 0.0),
    "voxel": (0.0, 0.5, 0.005, "m", 0.0),
}


@dataclass
class DisplayCloud:
    """A derived cloud as the page shows it."""

    cloud: PointCloud
    attrs: CloudAttrs  # the attributes the page asked for (label/encoding at their defaults)
    total: int  # points of the derived cloud before display thinning
    step: int  # display thinning: every ``step``-th point (1 = none)
    seconds: float  # derivation time


@dataclass
class ViewBundle:
    mode: str  # "image" | "map"
    title: str
    scene: Json  # the OpenLABEL document, as the commands emit it
    source: CloudSource  # data already computed; every displayed cloud is derived from it
    catalog: list[Json]
    segmented_png: bytes | None = None
    display_transform: list[list[float]] = field(default_factory=lambda: np.eye(4).tolist())
    stats: Json = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def scope(self) -> CloudScope:
        return scope_of(self.source)

    @property
    def cameras(self) -> list[Json]:
        return scene_cameras(self.scene)

    # -- point-cloud attributes -------------------------------------------------------------------

    @cached_property
    def controls(self) -> list[Json]:
        """One control per attribute that applies to this scope and affects the display."""
        defaults = CloudAttrs.defaults(self.scope)
        out = []
        for a in applicable(self.scope):
            if a.key in PLY_ONLY:
                continue
            value = getattr(defaults, a.field)
            c: Json = {"key": a.key, "help": a.effect, "values": a.values,
                       "default": a.format(value)}
            if isinstance(value, bool):
                c["kind"] = "toggle"
            elif isinstance(value, str):
                c.update(kind="choice", options=a.values.split("|"))
            elif isinstance(value, int | float) and a.key in _SLIDERS:
                lo, hi, step, unit, off = _SLIDERS[a.key]
                c.update(kind="int" if isinstance(value, int) else "float", min=lo,
                         max=self._depth_extent() if hi is None else hi, step=step, unit=unit,
                         off=None if off is None else a.format(off))
            else:
                c["kind"] = "text"
            out.append(c)
        return out

    def _depth_extent(self) -> float:
        """Slider maximum for the depth range: the farthest valid depth, rounded up."""
        depth = getattr(self.source, "depth", None)
        if depth is None:
            return 10.0
        d = np.asarray(depth)[np.asarray(self.source.valid) & (np.asarray(depth) > 0)]
        top = float(d.max()) if d.size else 10.0
        return max(0.1, math.ceil(top * 10.0) / 10.0)

    def parse_attrs(self, pairs: Iterable[tuple[str, str]]) -> CloudAttrs:
        """The attributes of a request (e.g. URL query pairs), validated by ``core.cloud_attrs``;
        keys that are not given keep their defaults. Raises :class:`UsageError`."""
        given: dict[str, str] = {}
        for key, value in pairs:
            if key in PLY_ONLY:
                raise UsageError(f"{key} concerns PLY files only; the viewer has no {key} control")
            if key in given:
                raise UsageError(f"point-cloud attribute {key!r} is given twice")
            given[key] = value
        return parse_cloud_attrs(given, self.scope)

    def describe(self, attrs: CloudAttrs) -> str:
        """``key=value,…`` of the controls (the ``-p`` syntax)."""
        keys = {c["key"] for c in self.controls}
        return ",".join(f"{k}={v}" for k, v in attrs.items(self.scope) if k in keys)

    def cloud(self, attrs: CloudAttrs) -> DisplayCloud:
        """The cloud ``attrs`` describe, derived from the in-memory source (no inference), with
        each point's object id; thinned for display beyond ``MAX_DISPLAY_POINTS``. Raises
        ``ValueError`` when the source cannot provide ``attrs`` (e.g. ``color=height`` without
        an estimated gravity)."""
        with self._lock:  # one derivation at a time; sources cache their normals
            t0 = time.perf_counter()
            full = derive_cloud(self.source, replace(attrs, label=self.source.labels is not None))
            seconds = time.perf_counter() - t0
        total = len(full)
        step = max(1, math.ceil(total / MAX_DISPLAY_POINTS))
        shown = full if step == 1 else full.subset(np.arange(0, total, step))
        return DisplayCloud(shown, attrs, total, step, seconds)

    def meta(self) -> Json:
        return {
            "mode": self.mode,
            "title": self.title,
            "stats": self.stats,
            "display_transform": self.display_transform,
            "cameras": self.cameras,
            "has_segmented": self.segmented_png is not None,
            "controls": self.controls,
            "defaults": self.describe(CloudAttrs.defaults(self.scope)),
            "max_points": MAX_DISPLAY_POINTS,
        }


# ------------------------------------------------------------------------------------------------
# camera poses and display frame


def _pose_matrix(transform: Json) -> NDArray[np.float64]:
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(np.asarray(transform["quaternion"], np.float64))
    T[:3, 3] = np.asarray(transform["translation"], np.float64)
    return T


def scene_cameras(scene: Json) -> list[Json]:
    """The camera of every frame of an OpenLABEL scene, in the objects' coordinate system: the
    frame's ``<stream> → <cs>`` transform (a map keyframe), or the identity for a stream whose
    sensor frame is itself a root coordinate system (a single image, whose scene is in its camera
    frame). ``T`` is camera-to-scene (4 x 4, row-major); ``K`` is ``fx, fy, cx, cy`` of the stream's
    pinhole intrinsics at ``size`` (width, height)."""
    root = scene.get("openlabel", {})
    streams: Json = root.get("streams", {})
    systems: Json = root.get("coordinate_systems", {})
    out: list[Json] = []
    for fid, fr in sorted(root.get("frames", {}).items(), key=lambda kv: int(kv[0])):
        props = fr.get("frame_properties", {})
        by_src = {t["src"]: t for t in props.get("transforms", {}).values()}
        for name in props.get("streams", {}):
            pin = streams.get(name, {}).get("stream_properties", {}).get("intrinsics_pinhole")
            if pin is None:
                continue
            if name in by_src:
                T = _pose_matrix(by_src[name]["transform_src_to_dst"])
            elif systems.get(name, {}).get("parent", "") == "":
                T = np.eye(4)
            else:
                continue
            m = pin["camera_matrix"]
            out.append({
                "name": props.get("keyframe", name), "frame": int(fid), "T": T.tolist(),
                "K": [m[0], m[5], m[2], m[6]], "size": [pin["width_px"], pin["height_px"]],
                "update": props.get("update_id"),
            })
    return out


def upright_transform(up_cam: NDArray[Any]) -> NDArray[np.float64]:
    """Display frame of a single image: its camera frame rotated so that the estimated up is +z
    and the camera looks along +y."""
    R = rotation_between(up_cam, np.array([0.0, 0.0, 1.0]))
    fwd = R @ np.array([0.0, 0.0, 1.0])
    yaw = np.arctan2(fwd[0], fwd[1])  # rotate the projected forward direction onto +y
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = Rz @ R
    return T


# ------------------------------------------------------------------------------------------------
# bundles


def image_bundle(image: Path, client: Any = None) -> ViewBundle:
    """Reconstruct and segment ``image`` once (inference server); keep its cloud source."""
    from oh_my_slam.core.images import png_bytes
    from oh_my_slam.segmentation.api import reconstruct_and_detect, segment_frame
    from oh_my_slam.segmentation.catalog import catalog_rows
    from oh_my_slam.segmentation.cloud import image_cloud_source
    from oh_my_slam.segmentation.render import segmented_image
    from oh_my_slam.segmentation.scene import single_image_scene

    if client is None:
        from oh_my_slam.reconstruction.api import connect_server

        client = connect_server()
    frame, dets = reconstruct_and_detect(Path(image), client)
    seg = segment_frame(frame, client=client, detections=dets)
    source = image_cloud_source(frame, seg)
    up = source.up if source.up is not None else DEFAULT_UP_CAM
    return ViewBundle(
        mode="image",
        title=Path(image).name,
        scene=single_image_scene(seg, tool="view"),
        source=source,
        catalog=catalog_rows(seg.objects),
        segmented_png=png_bytes(segmented_image(frame.rgb, seg.label_map)),
        display_transform=upright_transform(up).tolist(),
        stats={"objects": len(seg.objects), "frames": 1},
    )


def map_bundle(map_dir: Path) -> ViewBundle:
    """Open a persisted map read-only (no inference server, nothing written)."""
    import json

    from oh_my_slam.mapping import store
    from oh_my_slam.mapping.export import map_objects, reader_source, scene_bytes
    from oh_my_slam.segmentation.catalog import catalog_rows

    reader = store.MapReader(Path(map_dir))
    _, objs = map_objects(reader)
    source = reader_source(reader, objs)
    return ViewBundle(
        mode="map",
        title=reader.root.name,
        scene=json.loads(scene_bytes(reader)),
        source=source,
        catalog=catalog_rows(objs),
        stats={"objects": len(objs), "frames": len(reader.frames), "map_points": len(source.xyz)},
    )
