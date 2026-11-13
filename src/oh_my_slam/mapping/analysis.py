"""Frame analysis (O7; api.md §7): what one frame shows, without changing any map.

The frame is a view (``anchored_objects.View``: a pose, a depth grid in the output's metres, the
image and the detections on that grid, the ignored pixels) in a gravity-aligned frame with z up:
the map's, the client's own, or the camera's (``gravity_pose``). Everything is what an update
does with a keyframe, on one frame and kept nowhere:

* **Objects** (``objects``): the view's detections lifted and filtered as a keyframe's
  (``anchored_objects.observe``: exclusive masks, no support fringe, no floor named as an object,
  no opening, nothing on the camera, no background), those scoring at least ``min_score`` boxed
  as a map object's sightings are (``segmentation.api.fit_object_obb``; a fragment under
  ``MIN_OBJECT_M`` and a door that is no leaf are no objects, ``anchored_objects.listed``). Each
  takes the id of the map object it matches (``anchored_objects.match`` against
  ``known_objects``), else a local id from ``LOCAL_ID_START``, in the detections' order. Its
  ``sigma_centre_m`` and ``extent_seen`` are a map object's with this one sighting.
* **Points** (``frame_points``, ``thinned``): every pixel that becomes a point
  (``reconstruction.pointcloud.pixel_mask``: valid, not ignored, no flying pixel), with the id of
  the object whose points it was lifted into (0: none) and its colour.
* **Range scan** (``range_scan``): the points by bearing around the camera.
* **Scale** of a monocular depth: onto the map (``map_factor``: against the map's cloud seen
  from the pose), or onto a known camera height (``height_factor``)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.geometry import voxel_downsample_indices
from oh_my_slam.core.types import Intrinsics
from oh_my_slam.mapping.anchored import (
    ALIGN_RANGE,
    MIN_DEPTH_M,
    SCALE_RANGE,
    first_camera,
    own_floor,
)
from oh_my_slam.mapping.anchored_objects import (
    MIN_OBJECT_M,
    UP,
    Box,
    MapObject,
    View,
    _thin,
    extent_seen,
    match,
    observe,
    sigma_centre,
)
from oh_my_slam.reconstruction.pointcloud import pixel_mask, pixel_points
from oh_my_slam.segmentation.api import DOOR_LABEL, color_for_id, door_leaf, fit_object_obb

Array = NDArray[np.float64]

LOCAL_ID_START = 1_000_000  # api.md §7: a detection that is no map object
KNOWN_POINTS = 2000  # cloud points kept per map object for the association
RANGE_MIN_POINTS = 3  # a beam's range: its third point (one or two flying pixels are no surface)
MIN_BEARING_STEP_DEG = 0.1  # at most 3600 beams
MIN_CAMERA_HEIGHT_M = 0.05  # a camera this near the floor measures no scale from its height
LOWEST_PLANE_M = 1e6  # ``own_floor``'s expected height that picks the lowest plane
RENDER_SIDE = 64  # map_factor: the map's cloud is rendered on a grid of this side
VISIBLE_TOL = 0.05  # ... a point this far (relative) behind its cell's nearest one is hidden
MAP_ALIGN_MIN_POINTS = 200  # ... and this many visible points with depth give the factor


def gravity_pose(up_cam: Any) -> Array:
    """The pose of a frame without one (api.md §7, ``gravity_camera``): the camera at the origin,
    z up along the gravity it sees (``up_cam``, level when unknown), x its viewing direction on
    the horizontal plane (``anchored.first_camera``)."""
    return first_camera(up_cam)


# -- scale ---------------------------------------------------------------------------------------


def height_factor(depth: NDArray[np.float32], K: Intrinsics, R: Array, height_m: float | None
                  ) -> float | None:
    """The factor bringing a frame's monocular depth onto a known camera height above the floor
    (``height_m``), as the floor check gives a monocular frame its own scale (map.md §2): the
    height over the camera's height above its own floor plane (``anchored.own_floor``, the depth
    levelled by ``R``). None without a height above ``MIN_CAMERA_HEIGHT_M``, without a measurable
    floor plane, or outside ``anchored.SCALE_RANGE``."""
    if height_m is None or height_m <= MIN_CAMERA_HEIGHT_M:
        return None
    h = own_floor(depth, K, R, height_m)
    if h is None:
        return None
    f = height_m / h
    return f if SCALE_RANGE[0] <= f <= SCALE_RANGE[1] else None


def own_floor_z(depth: NDArray[np.float32], K: Intrinsics, T: Array, height_m: float | None
                ) -> float | None:
    """The floor's z in the frame of the pose ``T``, from the frame's own depth: the large
    horizontal plane below the camera nearest ``height_m`` under it, the lowest one without a
    height; None without such a plane."""
    h = own_floor(depth, K, T[:3, :3], height_m if height_m else LOWEST_PLANE_M)
    return None if h is None else float(T[2, 3]) - h


def map_factor(depth: NDArray[np.float32], K: Intrinsics, T: Array, cloud_xyz: NDArray[Any]
               ) -> float | None:
    """The factor bringing a frame's monocular depth (on its grid, intrinsics ``K``) at the map
    pose ``T`` onto the map: the median ratio of the range of the map's cloud points the pose
    sees to the frame's depth at their pixels. A cloud point is seen when it is the nearest of
    its cell of a ``RENDER_SIDE`` grid, within ``VISIBLE_TOL`` (what lies behind it is hidden).
    None with fewer than ``MAP_ALIGN_MIN_POINTS`` such points on valid depth, or a factor
    outside ``anchored.ALIGN_RANGE`` (the depth and the pose disagree)."""
    h, w = depth.shape
    pc = (np.asarray(cloud_xyz, np.float64).reshape(-1, 3) - T[:3, 3]) @ T[:3, :3]
    pc = pc[pc[:, 2] > MIN_DEPTH_M]
    u = np.rint(K.fx * pc[:, 0] / pc[:, 2] + K.cx).astype(np.int64)
    v = np.rint(K.fy * pc[:, 1] / pc[:, 2] + K.cy).astype(np.int64)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[inside], v[inside], pc[inside, 2]
    s = RENDER_SIDE / max(w, h)
    cols = math.ceil(w * s)
    cell = (v * s).astype(np.int64) * cols + (u * s).astype(np.int64)
    nearest = np.full(math.ceil(h * s) * cols, np.inf)
    np.minimum.at(nearest, cell, z)
    d = depth[v, u].astype(np.float64)
    ok = (z <= nearest[cell] * (1.0 + VISIBLE_TOL)) & np.isfinite(d) & (d > 0)
    if int(ok.sum()) < MAP_ALIGN_MIN_POINTS:
        return None
    f = float(np.median(z[ok] / d[ok]))
    return f if ALIGN_RANGE[0] <= f <= ALIGN_RANGE[1] else None


# -- objects -------------------------------------------------------------------------------------


@dataclass
class Seen:
    """An object of an analysed frame: its id (the map object's, else local), the label it was
    detected as and its score, its upright box in the view's frame with a map object's
    uncertainty fields, its exclusive mask on the view's grid without the ignored pixels, and
    the flat grid pixels its points were lifted from."""

    id: int
    label: str
    score: float
    box: Box
    sigma_centre_m: float
    extent_seen: dict[str, str]
    mean_rgb: list[int]
    mask: NDArray[np.bool_]
    pixels: NDArray[np.int64]


def known_objects(records: list[dict[str, Any]], cloud: dict[str, NDArray[Any]]
                  ) -> list[MapObject]:
    """A version's objects as the association takes them (``anchored_objects.match``): each
    record of its objects read that is not struck (map.md §5), with its box, the labels it was
    detected as and its points, the cloud's points that carry its id (``KNOWN_POINTS`` at most,
    evenly)."""
    ids = np.asarray(cloud["object_id"])
    order = np.argsort(ids, kind="stable")
    ranked = ids[order]
    out = []
    for r in records:
        if r["state"] == "struck":
            continue
        lo, hi = np.searchsorted(ranked, [r["id"], r["id"] + 1])
        idx = order[lo:hi]
        if len(idx) > KNOWN_POINTS:
            idx = idx[np.linspace(0, len(idx) - 1, KNOWN_POINTS).astype(np.intp)]
        box = Box(np.asarray(r["centre_m"], np.float64), np.asarray(r["size_m"], np.float64),
                  float(r["yaw_deg"]))
        out.append(MapObject(int(r["id"]), {k: 1.0 for k in [r["label"], *r["detected_as"]]},
                             points=np.asarray(cloud["xyz"][idx], np.float32), box=box))
    return out


def objects(view: View, floor_z: float | None, floor_sigma: float, min_score: float,
            known: list[MapObject]) -> list[Seen]:
    """The objects of a view (see the module docstring), in its detections' order; ``floor_z``:
    the floor in the view's frame (None: unknown, nothing is grounded on it), ``known``: the map
    objects a detection may be (empty: every id is local)."""
    kept = []
    for ob in observe(view, 0, floor_z, floor_sigma):
        if ob.score < min_score:
            continue
        box = Box.of(fit_object_obb(ob.points, ob.label, UP, floor_z))
        if float(np.max(box.size)) < MIN_OBJECT_M or (
                ob.label == DOOR_LABEL and not door_leaf(box.size)):
            continue
        kept.append((ob, box))
    ids: list[int | None] = [None] * len(kept)
    if known:
        ids = match([(ob.label, _thin(ob.points), ob.centre) for ob, _ in kept], known,
                    [ob.sighting.truncated for ob, _ in kept])
    out = []
    local = LOCAL_ID_START
    for (ob, box), oid in zip(kept, ids, strict=True):
        if oid is None:
            oid, local = local, local + 1
        one = MapObject(oid, {ob.label: ob.score}, [ob.sighting], box=box)
        mask = ob.mask if view.ignore is None else ob.mask & ~view.ignore
        out.append(Seen(oid, ob.label, ob.score, box, sigma_centre(one),
                        extent_seen(one, support=False), [round(c) for c in ob.sighting.rgb],
                        mask, ob.pixels))
    return out


def color_of(object_id: int) -> tuple[int, int, int]:
    """An analysis object's display colour, a pure function of its id: a map object's own
    (``segmentation.api.color_for_id``); a local id's is the colour of its rank among the local
    ids (``LOCAL_ID_START`` takes id 1's: the palette is built id by id, so the millionth colour
    is never asked)."""
    return color_for_id(object_id - LOCAL_ID_START + 1 if object_id >= LOCAL_ID_START
                        else object_id)


# -- points --------------------------------------------------------------------------------------


@dataclass
class FramePoints:
    """A view's points in its frame, each with the id of its object (0: none) and its colour."""

    xyz: Array
    object_id: NDArray[np.uint32]
    rgb: NDArray[np.uint8]


def frame_points(view: View, seen: list[Seen]) -> FramePoints:
    """Every pixel of a view that becomes a point (valid depth, not ignored, no flying pixel),
    in the view's frame."""
    depth = np.where(np.isfinite(view.depth), view.depth, 0.0).astype(np.float32)
    valid = depth > 0
    if view.ignore is not None:
        valid &= ~view.ignore
    pts, pix = pixel_points(depth, view.K, pixel_mask(depth, valid))
    owner = np.zeros(depth.size, np.uint32)
    for s in seen:
        owner[s.pixels] = s.id
    return FramePoints(pts @ view.T[:3, :3].T + view.centre, owner[pix],
                       np.asarray(view.rgb, np.uint8).reshape(-1, 3)[pix])


def thinned(points: FramePoints, voxel: float) -> FramePoints:
    """One point per ``voxel`` (the first of each); every point with a voxel of 0."""
    if voxel <= 0 or not len(points.xyz):
        return points
    sel = voxel_downsample_indices(points.xyz, voxel, keep="first")
    return FramePoints(points.xyz[sel], points.object_id[sel], points.rgb[sel])


# -- range scan ----------------------------------------------------------------------------------


def _kth(bins: NDArray[np.int64], values: Array, n: int) -> Array:
    """Per bin (0..n-1), the ``RANGE_MIN_POINTS``-th smallest of its values; NaN with fewer."""
    out = np.full(n, np.nan)
    order = np.lexsort((values, bins))
    b, v = bins[order], values[order]
    start = np.searchsorted(b, np.arange(n))
    ok = np.searchsorted(b, np.arange(n), side="right") - start >= RANGE_MIN_POINTS
    out[ok] = v[start[ok] + RANGE_MIN_POINTS - 1]
    return out


def range_scan(xyz: Array, origin: Array, floor_z: float, floor_band: float,
               z_band: tuple[float, float], step_deg: float) -> list[dict[str, Any]]:
    """The range scan of a frame's points ``xyz`` (api.md §7) around the camera at ``origin``,
    both in a frame with z up: one beam per bearing k x ``step_deg`` in (-180, 180],
    counter-clockwise from the frame's +x, each holding the points nearest its bearing (the two
    end beams share what is left at ±180 when the step does not divide 360). ``range_m`` is
    horizontal, from the camera. A beam is

    * ``obstacle``: the nearest surface within ``z_band`` (metres above the floor ``floor_z``)
      and above the floor band (``floor_band``: what lies within it is the floor, map.md §6);
    * else ``floor_end``: the farthest floor it sees;
    * else ``unseen`` (``range_m`` null).

    Nearest and farthest are the ``RANGE_MIN_POINTS``-th point of the beam, so a flying pixel is
    neither."""
    k_min, k_max = math.floor(-180.0 / step_deg) + 1, math.floor(180.0 / step_deg)
    n = k_max - k_min + 1
    d = np.asarray(xyz, np.float64).reshape(-1, 3)[:, :2] - np.asarray(origin, np.float64)[:2]
    r = np.hypot(d[:, 0], d[:, 1])
    bearing = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    k = np.rint(bearing / step_deg)
    end = (k < k_min) | (k > k_max)  # beyond the end beams: the nearer of the two, around ±180
    to_first = np.abs((bearing[end] - k_min * step_deg + 180.0) % 360.0 - 180.0)
    to_last = np.abs((bearing[end] - k_max * step_deg + 180.0) % 360.0 - 180.0)
    k[end] = np.where(to_first < to_last, k_min, k_max)
    k = k.astype(np.int64) - k_min
    h = np.asarray(xyz, np.float64).reshape(-1, 3)[:, 2] - floor_z
    solid = (h >= z_band[0]) & (h <= z_band[1]) & (h > floor_band)
    floor = np.abs(h) <= floor_band
    near = _kth(k[solid], r[solid], n)
    far = -_kth(k[floor], -r[floor], n)
    out = []
    for i in range(n):
        beam: dict[str, Any] = {"bearing_deg": round((i + k_min) * step_deg, 6),
                                "range_m": None, "kind": "unseen"}
        if math.isfinite(near[i]):
            beam.update(range_m=round(float(near[i]), 4), kind="obstacle")
        elif math.isfinite(far[i]):
            beam.update(range_m=round(float(far[i]), 4), kind="floor_end")
        out.append(beam)
    return out
