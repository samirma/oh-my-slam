"""Frame analysis (O7; api.md §7): what one frame shows, without changing any map.

The frame is a view (``anchored_objects.View``: a pose, a depth grid in the output's metres, the
image and the detections on that grid, the ignored pixels) in a gravity-aligned frame with z up:
the map's, the client's own, or the camera's (``gravity_pose``). Everything is what an update
does with a keyframe, on one frame and kept nowhere:

* **Objects** (``objects``): the view's detections lifted and filtered as a keyframe's
  (``anchored_objects.observe``: exclusive masks, no support fringe, no floor named as an object,
  no opening, nothing on the camera, no background), those scoring at least ``min_score`` boxed
  as a map object's sightings are (``anchored_objects.fitted_box``; a fragment under
  ``MIN_OBJECT_M`` and a door that is no leaf are no objects, ``anchored_objects.listed``). Each
  takes the id of the map object it matches (``anchored_objects.match`` against
  ``known_objects``), else a local id from ``LOCAL_ID_START``, in the detections' order. Its
  ``sigma_centre_m`` and ``extent_seen`` are a map object's with this one sighting.
* **Points** (``frame_points``, ``thinned``): every pixel that becomes a point
  (``reconstruction.pointcloud.pixel_mask``: valid, not ignored, no flying pixel), with the id of
  the object whose points it was lifted into (0: none) and its colour.
* **Range scan** (``range_scan``): the points by bearing around the camera, a beam's return
  held only by supported pixels (``reconstruction.pointcloud.supported_pixels``, oms/2.10).
* **Scale** of a monocular depth: onto the map (``map_factor``: against the map's cloud seen
  from the pose), or onto a known camera height (``height_factor``).
* **Horizontal** (``horizontal``, oms/2.10): with a pose, its up axis is the horizontal's; a
  monocular depth's floor plane tilted from it beyond ``TILT_K_SIGMA`` x the request's
  ``sigma_tilt_deg`` is turned onto it about the camera centre (``onto_up``) before any other
  use, as the rotation the depth is lifted with."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.geometry import project, voxel_downsample_indices
from oh_my_slam.core.types import Intrinsics
from oh_my_slam.mapping.anchored import (
    ALIGN_RANGE,
    MIN_DEPTH_M,
    SCALE_RANGE,
    first_camera,
    floor_band,
    floor_share,
    own_floor,
    own_floor_plane,
)
from oh_my_slam.mapping.anchored_objects import (
    MIN_OBJECT_M,
    Box,
    MapObject,
    View,
    _thin,
    extent_seen,
    fitted_box,
    match,
    observe,
    sigma_centre,
)
from oh_my_slam.reconstruction.pointcloud import pixel_mask, pixel_points, supported_pixels
from oh_my_slam.segmentation.api import DOOR_LABEL, door_leaf

Array = NDArray[np.float64]

LOCAL_ID_START = 1_000_000  # api.md §7: a detection that is no map object
KNOWN_POINTS = 2000  # cloud points kept per map object for the association
RANGE_MIN_POINTS = 3  # a beam's range: its third point (one or two flying pixels are no surface)
MIN_BEARING_STEP_DEG = 0.1  # at most 3600 beams
MIN_CAMERA_HEIGHT_M = 0.05  # a camera this near the floor measures no scale from its height
RENDER_SIDE = 64  # map_factor: the map's cloud is rendered on a grid of this side
VISIBLE_TOL = 0.05  # ... a point this far (relative) behind its cell's nearest one is hidden
MAP_ALIGN_MIN_POINTS = 200  # ... and this many visible points with depth give the factor
SIGMA_TILT_DEFAULT_DEG = 2.0  # api.md §7 (oms/2.10): the pose's tilt sigma when none is sent
TILT_K_SIGMA = 3.0  # ... a depth's floor beyond this many of them from the pose's up is turned
MIN_BEAM_SUPPORT = 12  # api.md §7 "Unsupported beams": supported pixels a beam's return needs
BEAM_MARGIN_PX = 3  # ... none within this of a low-confidence pixel or a depth edge
BEAM_BORDER_REL = 0.02  # ... nor in the image's outer share of its larger side, on each side
RETURN_TOL_REL = 0.05  # a return's pixels: within this share of its range behind its nearest
RETURN_TOL_M = 0.05  # ... and at least this far


def gravity_pose(up_cam: Any) -> Array:
    """The pose of a frame without one (api.md §7, ``gravity_camera``): the camera at the origin,
    z up along the gravity it sees (``up_cam``, level when unknown), x its viewing direction on
    the horizontal plane (``anchored.first_camera``)."""
    return first_camera(up_cam)


# -- the horizontal (oms/2.10) -------------------------------------------------------------------


@dataclass
class Horizontal:
    """How a frame's depth meets the horizontal of its pose (api.md §7 "Horizontality from the
    pose"): the rotation it is lifted with (``R``: the pose's, its floor turned onto the pose's
    up axis when ``rotated``), the raw depth's floor plane in the pose's levelled frame (its
    distance to the camera and unit normal, ``anchored.own_floor_plane``; None without a
    measurable one) and that normal's angle to the up axis (``tilt_deg``)."""

    R: Array
    plane: tuple[float, Array] | None
    tilt_deg: float | None
    rotated: bool


def tilt_deg(normal: Array) -> float:
    """The angle (degrees) between a unit normal with z up and the up axis +z."""
    return math.degrees(math.acos(min(1.0, max(-1.0, float(normal[2])))))


def onto_up(normal: Array) -> Array:
    """The smallest rotation taking the unit ``normal`` onto +z (about their common
    perpendicular; the identity when they agree)."""
    n = np.asarray(normal, np.float64) / float(np.linalg.norm(normal))
    axis = np.cross(n, [0.0, 0.0, 1.0])
    s, c = float(np.linalg.norm(axis)), float(n[2])
    if s < 1e-12:
        return np.eye(3)
    k = axis / s
    Kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    out: Array = np.eye(3) + s * Kx + (1.0 - c) * (Kx @ Kx)
    return out


def horizontal(depth: NDArray[np.float32], K: Intrinsics, R: Array, expected: float | None,
               kind: str, sigma_m: float | None, sigma_tilt_deg: float, rotate: bool
               ) -> Horizontal:
    """A frame's ``Horizontal`` at the rotation ``R`` of its pose: its floor plane as the floor
    check chooses it (``anchored.own_floor_plane``: near the ``expected`` camera height within
    ``anchored.floor_band`` of the depth's ``kind`` and the height's sigma ``sigma_m``, else the
    widest plausible; the lowest without a height), in the raw ``depth``. With ``rotate`` (a
    monocular depth with a pose) and that plane's normal beyond ``TILT_K_SIGMA`` x
    ``sigma_tilt_deg`` of the up axis, the depth is turned about the camera centre so that its
    floor's normal is the up axis: it is lifted with ``onto_up(normal) @ R``. A rotation about
    the camera keeps every range and the plane's distance, so only the floor's height is left
    to fit, along the up axis. Otherwise, and without a plane, ``R`` as it is."""
    if expected:
        plane = own_floor_plane(depth, K, R, expected, floor_band(kind, expected, sigma_m))
    else:
        plane = own_floor_plane(depth, K, R, None, 0.0)
    if plane is None:
        return Horizontal(R, None, None, False)
    tilt = tilt_deg(plane[1])
    if rotate and tilt > TILT_K_SIGMA * sigma_tilt_deg:
        return Horizontal(onto_up(plane[1]) @ R, plane, tilt, True)
    return Horizontal(R, plane, tilt, False)


# -- scale ---------------------------------------------------------------------------------------


def height_factor(depth: NDArray[np.float32], K: Intrinsics, R: Array, height_m: float | None,
                  sigma_m: float | None = None) -> float | None:
    """The factor bringing a frame's monocular depth onto a known camera height above the floor
    (``height_m``, its sigma ``sigma_m``), as the floor check gives a monocular frame its own
    scale (map.md §2): the height over the camera's height above its own floor plane
    (``anchored.own_floor``, the depth levelled by ``R``: the supported plane near that height,
    ``anchored.floor_band``, else the widest plausible one, since the model's metres do not say
    which plane lies at that height). None without a height above ``MIN_CAMERA_HEIGHT_M``, without a
    measurable floor plane, or outside ``anchored.SCALE_RANGE``."""
    if height_m is None or height_m <= MIN_CAMERA_HEIGHT_M:
        return None
    h = own_floor(depth, K, R, height_m, floor_band("mono", height_m, sigma_m))
    if h is None:
        return None
    f = height_m / h
    return f if SCALE_RANGE[0] <= f <= SCALE_RANGE[1] else None


def scale_check(depth: NDArray[np.float32], K: Intrinsics, R: Array, height_m: float,
                source: str, sigma_m: float | None = None) -> dict[str, Any]:
    """A frame's ``scale_check`` (api.md §7, oms/2.8): the sent camera height ``height_m``
    against the camera's height above the frame's own floor plane (``anchored.own_floor_plane``,
    the floor check's rule, in the raw ``depth`` levelled by ``R``), the ratio of the two and
    the share of the frame's valid depth pixels on that plane (``anchored.floor_share``).
    Without a measurable plane: no model height or ratio, a share of 0. ``source``: the raw
    depth's, ``sensor`` or ``monocular``; ``sigma_m``: the sent height's sigma (the band in
    which a plane is the one at that height)."""
    return checked(own_floor_plane(depth, K, R, height_m, floor_band(
        "sensor" if source == "sensor" else "mono", height_m, sigma_m)), depth, K, R, height_m,
        source)


def checked(plane: tuple[float, Array] | None, depth: NDArray[np.float32], K: Intrinsics,
            R: Array, height_m: float, source: str) -> dict[str, Any]:
    """``scale_check`` on the floor ``plane`` already found in the raw ``depth`` levelled by
    ``R`` (its distance and unit normal; None: no measurable plane), with ``floor_tilt_deg``
    (oms/2.10): the angle between that normal and the up axis of ``R``, null without a plane."""
    out: dict[str, Any] = {"sent_height_m": round(float(height_m), 4), "model_height_m": None,
                           "ratio": None, "floor_inlier_share": 0.0, "depth_source": source,
                           "floor_tilt_deg": None}
    if plane is not None:
        h, normal = plane
        out.update(model_height_m=round(h, 4), ratio=round(height_m / h, 4),
                   floor_inlier_share=round(floor_share(depth, K, R, h, normal), 4),
                   floor_tilt_deg=round(tilt_deg(normal), 3))
    return out


def own_floor_z(depth: NDArray[np.float32], K: Intrinsics, T: Array, height_m: float | None,
                kind: str, sigma_m: float | None = None) -> float | None:
    """The floor's z in the frame of the pose ``T``, from the frame's own depth: its own floor
    plane (``anchored.own_floor``: with real support, nearest ``height_m`` within the band
    ``height_factor`` and ``scale_check`` take it in, ``anchored.floor_band`` of the depth's
    ``kind`` and the height's sigma ``sigma_m``, else the widest plausible one), the lowest one
    without a height; None without such a plane."""
    if not height_m:
        h = own_floor(depth, K, T[:3, :3], None, 0.0)
    else:
        h = own_floor(depth, K, T[:3, :3], height_m, floor_band(kind, height_m, sigma_m))
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
    uv, z = project(pc[pc[:, 2] > MIN_DEPTH_M], K.K())
    u, v = np.rint(uv[:, 0]).astype(np.int64), np.rint(uv[:, 1]).astype(np.int64)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[inside], v[inside], z[inside]
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
        box = fitted_box(ob.points, ob.label, floor_z)
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


# -- points --------------------------------------------------------------------------------------


@dataclass
class FramePoints:
    """A view's points in its frame, each with the id of its object (0: none), its colour and
    whether its pixel is a supported one (``supported_pixels``: a beam's return may rest on
    it)."""

    xyz: Array
    object_id: NDArray[np.uint32]
    rgb: NDArray[np.uint8]
    supported: NDArray[np.bool_]


def beam_border_px(shape: tuple[int, ...]) -> int:
    """The image-border band (pixels, on each side) of a grid of ``shape`` whose pixels hold no
    beam's return: ``BEAM_BORDER_REL`` of its larger side."""
    return round(BEAM_BORDER_REL * max(shape[0], shape[1]))


def frame_points(view: View, seen: list[Seen]) -> FramePoints:
    """Every pixel of a view that becomes a point (valid depth, not ignored, no flying pixel),
    in the view's frame, with whether it is a supported pixel (``supported_pixels`` of the
    depth without the ignored pixels: ``BEAM_MARGIN_PX`` from a pixel without depth and from a
    depth edge, outside ``beam_border_px``)."""
    depth = np.where(np.isfinite(view.depth), view.depth, 0.0).astype(np.float32)
    valid = depth > 0
    if view.ignore is not None:
        valid &= ~view.ignore
    pts, pix = pixel_points(depth, view.K, pixel_mask(depth, valid))
    owner = np.zeros(depth.size, np.uint32)
    for s in seen:
        owner[s.pixels] = s.id
    held = supported_pixels(np.where(valid, depth, 0.0), BEAM_MARGIN_PX,
                            beam_border_px(depth.shape)).reshape(-1)
    return FramePoints(pts @ view.T[:3, :3].T + view.centre, owner[pix],
                       np.asarray(view.rgb, np.uint8).reshape(-1, 3)[pix], held[pix])


def thinned(points: FramePoints, voxel: float) -> FramePoints:
    """One point per ``voxel`` (the first of each); every point with a voxel of 0."""
    if voxel <= 0 or not len(points.xyz):
        return points
    sel = voxel_downsample_indices(points.xyz, voxel, keep="first")
    return FramePoints(points.xyz[sel], points.object_id[sel], points.rgb[sel],
                       points.supported[sel])


# -- range scan ----------------------------------------------------------------------------------


def _farthest(bins: NDArray[np.int64], values: Array, n: int
              ) -> tuple[Array, NDArray[np.int64]]:
    """Per bin (0..n-1), the ``RANGE_MIN_POINTS``-th largest of its values and its number of
    values; NaN with fewer than ``MIN_BEAM_SUPPORT``."""
    out = np.full(n, np.nan)
    count = np.bincount(bins, minlength=n).astype(np.int64)
    order = np.lexsort((-values, bins))
    b, v = bins[order], values[order]
    start = np.searchsorted(b, np.arange(n))
    ok = count >= max(MIN_BEAM_SUPPORT, RANGE_MIN_POINTS)
    out[ok] = v[start[ok] + RANGE_MIN_POINTS - 1]
    return out, count


def _nearest_return(bins: NDArray[np.int64], values: Array, n: int
                    ) -> tuple[Array, NDArray[np.int64]]:
    """Per bin (0..n-1), its nearest return and the number of values holding it: the nearest
    value whose window (itself and the values up to max(``RETURN_TOL_M``, ``RETURN_TOL_REL`` x
    it) beyond it) holds at least ``MIN_BEAM_SUPPORT`` values, its range the window's
    ``RANGE_MIN_POINTS``-th value; NaN and 0 without one."""
    out = np.full(n, np.nan)
    support = np.zeros(n, np.int64)
    if not len(values):
        return out, support
    tol = np.maximum(RETURN_TOL_M, RETURN_TOL_REL * values)
    span = float(values.max() + tol.max()) + 1.0  # bins apart on one key
    order = np.lexsort((values, bins))
    b, v, t = bins[order], values[order], tol[order]
    key = b * span + v
    held = np.searchsorted(key, key + t, side="right") - np.arange(len(key))
    ok = np.flatnonzero(held >= max(MIN_BEAM_SUPPORT, RANGE_MIN_POINTS))
    first, at = np.unique(b[ok], return_index=True)  # sorted by bin, then value: the nearest
    j = ok[at]
    out[first] = v[j + RANGE_MIN_POINTS - 1]
    support[first] = held[j]
    return out, support


def range_scan(xyz: Array, origin: Array, floor_z: float, floor_band: float,
               z_band: tuple[float, float], step_deg: float,
               supported: NDArray[np.bool_] | None = None) -> list[dict[str, Any]]:
    """The range scan of a frame's points ``xyz`` (api.md §7) around the camera at ``origin``,
    both in a frame with z up: one beam per bearing k x ``step_deg`` in (-180, 180],
    counter-clockwise from the frame's +x, each holding the points nearest its bearing (the two
    end beams share what is left at ±180 when the step does not divide 360). ``range_m`` is
    horizontal, from the camera. Only the ``supported`` points (supported pixels,
    ``frame_points``; every point when None) hold a return (oms/2.10, "Unsupported beams"). A
    beam is

    * ``obstacle``: its nearest return within ``z_band`` (metres above the floor ``floor_z``)
      and above the floor band (``floor_band``: what lies within it is the floor, map.md §6):
      the nearest surface held by at least ``MIN_BEAM_SUPPORT`` points within its return
      window (``_nearest_return``);
    * else ``floor_end``: the farthest floor it sees, held by at least ``MIN_BEAM_SUPPORT``
      floor points (all of them hold it);
    * else ``unseen`` (``range_m`` null): a beam whose returns are all unsupported is never a
      short-range obstacle.

    ``support`` is the number of points holding the return (0 for ``unseen``). Nearest and
    farthest are the ``RANGE_MIN_POINTS``-th point, so a flying pixel is neither."""
    k_min, k_max = math.floor(-180.0 / step_deg) + 1, math.floor(180.0 / step_deg)
    n = k_max - k_min + 1
    p = np.asarray(xyz, np.float64).reshape(-1, 3)
    d = p[:, :2] - np.asarray(origin, np.float64)[:2]
    r = np.hypot(d[:, 0], d[:, 1])
    bearing = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    k = np.rint(bearing / step_deg)
    end = (k < k_min) | (k > k_max)  # beyond the end beams: the nearer of the two, around ±180
    to_first = np.abs((bearing[end] - k_min * step_deg + 180.0) % 360.0 - 180.0)
    to_last = np.abs((bearing[end] - k_max * step_deg + 180.0) % 360.0 - 180.0)
    k[end] = np.where(to_first < to_last, k_min, k_max)
    k = k.astype(np.int64) - k_min
    h = p[:, 2] - floor_z
    held = np.ones(len(p), bool) if supported is None else np.asarray(supported, bool)
    solid = (h >= z_band[0]) & (h <= z_band[1]) & (h > floor_band) & held
    floor = (np.abs(h) <= floor_band) & held
    near, near_n = _nearest_return(k[solid], r[solid], n)
    far, far_n = _farthest(k[floor], r[floor], n)
    out = []
    for i in range(n):
        beam: dict[str, Any] = {"bearing_deg": round((i + k_min) * step_deg, 6),
                                "range_m": None, "kind": "unseen", "support": 0}
        if math.isfinite(near[i]):
            beam.update(range_m=round(float(near[i]), 4), kind="obstacle",
                        support=int(near_n[i]))
        elif math.isfinite(far[i]):
            beam.update(range_m=round(float(far[i]), 4), kind="floor_end",
                        support=int(far_n[i]))
        out.append(beam)
    return out
