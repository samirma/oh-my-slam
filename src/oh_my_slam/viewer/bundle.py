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

:func:`bundle_of` builds the bundle of ``view.sh``'s validated arguments.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from oh_my_slam.core.log import get_logger
from oh_my_slam.core.ply import PointCloud
from oh_my_slam.reconstruction.gravity import DEFAULT_UP_CAM
from oh_my_slam.segmentation.cloud import CloudSource, ImageCloudSource, derive_thinned, scope_of

if TYPE_CHECKING:
    from oh_my_slam.mapping.store import MapReader

Json = dict[str, Any]
log = get_logger("oh_my_slam.viewer")

# Spec §2.5 display budget: the page draws every point of a cloud of at most this many points. A
# larger cloud is shown as a voxel-grid selection — one original point per occupied voxel of the
# smallest edge that fits (``derive_thinned``, the shared derivation) — and the page states
# "showing X of Y points" with the edge. PLY outputs and the map are never thinned.
DISPLAY_POINT_BUDGET = 16_000_000

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
    total: int  # points of the derived cloud before display thinning
    voxel: float  # display thinning: one point per voxel of this edge, metres (0 = every point)
    seconds: float  # derivation time
    owned_bytes: int | None = None  # memory of its arrays not shared with the source (None: all)


def owned_bytes(cloud: PointCloud, source: CloudSource) -> int:
    """Bytes of ``cloud``'s arrays that are not views of ``source``'s arrays: what keeping the
    cloud costs beyond the source (a map cloud with every point shares positions, colours and
    labels with its source, see ``derive_thinned``)."""
    shared = [a for a in (getattr(source, n, None) for n in ("xyz", "rgb", "labels"))
              if isinstance(a, np.ndarray)]
    return sum(a.nbytes for a in (cloud.xyz, cloud.rgb, cloud.label, cloud.normals)
               if a is not None and not any(np.may_share_memory(a, s) for s in shared))


@dataclass
class ViewBundle:
    mode: str  # "image" | "map"
    title: str
    scene: Json  # the OpenLABEL document, as the commands emit it
    source: CloudSource  # data already computed; every displayed cloud is derived from it
    catalog: list[Json]
    segmented_png: bytes | None = None
    display_transform: list[list[float]] = field(default_factory=lambda: np.eye(4).tolist())
    camera_sources: dict[str, str] = field(default_factory=dict)  # camera name → input file name
    point_budget: int = DISPLAY_POINT_BUDGET  # points drawn at most (tests set a small one)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def scope(self) -> CloudScope:
        return scope_of(self.source)

    @cached_property
    def cameras(self) -> list[Json]:
        """Every camera of the scene JSON at its pose there (see :func:`scene_cameras`)."""
        return [c | {"source": self.camera_sources.get(c["name"], c["source"])}
                for c in scene_cameras(self.scene)]

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
        if not isinstance(self.source, ImageCloudSource):
            return 10.0
        depth = np.asarray(self.source.depth)
        d = depth[np.asarray(self.source.valid) & (depth > 0) & np.isfinite(depth)]
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
        each point's object id; beyond ``point_budget`` points its voxel-grid selection
        (``derive_thinned``: normals only for those). Raises ``ValueError`` when the source cannot
        provide ``attrs`` (e.g. ``color=height`` without an estimated gravity)."""
        with self._lock:  # one derivation at a time; sources keep their normals
            t0 = time.perf_counter()
            thin = derive_thinned(self.source, replace(attrs, label=self.source.labels is not None),
                                  self.point_budget)
            seconds = time.perf_counter() - t0
        return DisplayCloud(thin.cloud, thin.total, thin.voxel, seconds,
                            owned_bytes(thin.cloud, self.source))

    def prepare(self) -> None:
        """Derive the default cloud once (its display selection is then kept by the source)."""
        try:
            self.cloud(CloudAttrs.defaults(self.scope))
        except Exception as exc:  # the page's own request reports it
            log.warning("viewer: preparing the default cloud failed: %s", exc)

    def meta(self) -> Json:
        return {
            "mode": self.mode,
            "title": self.title,
            "display_transform": self.display_transform,
            "cameras": self.cameras,
            "has_segmented": self.segmented_png is not None,
            "controls": self.controls,
            "defaults": self.describe(CloudAttrs.defaults(self.scope)),
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
    frame). ``T`` is camera-to-scene (4 x 4, row-major) and ``position`` its translation, the
    camera centre in the scene frame (metres); ``K`` is ``fx, fy, cx, cy`` of the stream's pinhole
    intrinsics at ``size`` (width, height); ``source`` the file name of the frame's image;
    ``located`` true for a frame ``mapper.sh locate`` marks as a located input image. The page's
    ``cameras.js`` ``sceneCameras`` mirrors this for scene files opened in the browser."""
    root = scene.get("openlabel", {})
    streams: Json = root.get("streams", {})
    systems: Json = root.get("coordinate_systems", {})
    out: list[Json] = []
    for fid, fr in sorted(root.get("frames", {}).items(), key=lambda kv: int(kv[0])):
        props = fr.get("frame_properties", {})
        by_src = {t["src"]: t for t in props.get("transforms", {}).values()}
        for name, stream in props.get("streams", {}).items():
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
                "position": T[:3, 3].tolist(),
                "K": [m[0], m[5], m[2], m[6]], "size": [pin["width_px"], pin["height_px"]],
                "update": props.get("update_id"),
                "source": Path(str(stream.get("uri", ""))).name,
                # a camera ``mapper.sh locate`` placed (not one of the map's own frames)
                "located": props.get("located") is True,
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


def image_bundle(image: Path, client: Any = None, min_score: float | None = None) -> ViewBundle:
    """Reconstruct and segment ``image`` once (inference server), keeping detections of at least
    ``min_score`` (default: segmentation's); keep its cloud source."""
    from oh_my_slam.core.images import png_bytes
    from oh_my_slam.segmentation.api import reconstruct_and_detect, segment_frame
    from oh_my_slam.segmentation.catalog import catalog_rows
    from oh_my_slam.segmentation.cloud import image_cloud_source
    from oh_my_slam.segmentation.render import segmented_image
    from oh_my_slam.segmentation.scene import single_image_scene

    if client is None:
        from oh_my_slam.reconstruction.api import connect_server

        client = connect_server()
    score = {} if min_score is None else {"min_score": min_score}
    frame, dets = reconstruct_and_detect(Path(image), client, **score)
    seg = segment_frame(frame, client=client, detections=dets, **score)
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
    )


def map_bundle(map_dir: Path, reader: MapReader | None = None) -> ViewBundle:
    """Open a persisted map read-only (no inference server, nothing written); ``reader``: the
    ``mapping.store.MapReader`` of ``map_dir`` when already opened (view.sh's map rule)."""
    import json

    from oh_my_slam.mapping import store
    from oh_my_slam.mapping.export import map_objects, reader_source, scene_bytes
    from oh_my_slam.segmentation.catalog import catalog_rows

    if reader is None:
        reader = store.MapReader(Path(map_dir))
    _, objs = map_objects(reader)
    source = reader_source(reader, objs)
    bundle = ViewBundle(
        mode="map",
        title=reader.root.name,
        scene=json.loads(scene_bytes(reader)),
        source=source,
        catalog=catalog_rows(objs),
        # keyframe images are copies (frames/fNNNNNN.jpg); name the input they came from
        camera_sources={r.name: Path(r.source).name for r in reader.frames if r.source},
    )
    if len(source.xyz) > bundle.point_budget:
        # a cloud above the display budget: find its selection while the browser starts, so that
        # the page's first cloud request finds it ready (the request waits for it otherwise)
        threading.Thread(target=bundle.prepare, name="display-selection", daemon=True).start()
    return bundle


def bundle_of(values: Any, client: Any = None) -> ViewBundle:
    """The bundle of a command's validated arguments (``commands.spec.validate``): ``map`` (and
    its opened ``reader``) gives the map's, else ``image`` the image's, with ``min_score`` when the
    command has one. ``view.sh`` serves it."""
    if getattr(values, "map", None) is not None:
        return map_bundle(values.map, getattr(values, "reader", None))
    return image_bundle(values.image, client, getattr(values, "min_score", None))
