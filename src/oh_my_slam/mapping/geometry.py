"""Map geometry for an update: the coloured cloud (surface of a TSDF fusion of the valid, aligned
depth maps; colour and object id per point from the latest update that sees it), built with the
reconstruction package's fusion code.

Object ids come from the keyframes' instance masks, which a detector draws generously: a "carpet"
mask that covers a counter top and the floor beyond it, of which only the counter was lifted into
the object (lifting keeps a mask's largest spatial cluster). A keyframe's vote for an object
therefore counts only for points inside the object's box grown by the depth noise at its viewing
distance (``attribution_margin``), so an object's points in the cloud (``segments.ply``,
``color=segment``, its ``point_count``) coincide with its box. The vote needs a third of the
keyframes that see a point, and an object detected in fewer of them wins only part of its
surface — a refrigerator detected in 7 of the ~15 keyframes that see its front, half of it — or
nothing — a dishwasher detected in 2 of ~13, a light switch on a wall. Each confirmed object
therefore also takes the unlabelled cloud points of its gate nearest to its own lifted points
(``support_labels``) — if a keyframe that detected it fused it (``fused_objects``: not a car
detected only 40-100 m away, beyond the fused depth) — and it is exported only when that makes it
visibly drawn (``objects.min_cloud_points``, at the map's sampling at its nearest detection)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core import rle, timing
from oh_my_slam.core.geometry import project
from oh_my_slam.core.images import load_rgb
from oh_my_slam.core.ply import PointCloud, ply_bytes
from oh_my_slam.mapping import store
from oh_my_slam.mapping.objects import (
    ObjectState,
    Vacated,
    fit_points,
    label_map_for,
    load_vacated,
)
from oh_my_slam.reconstruction.depth import MAX_FACTOR
from oh_my_slam.reconstruction.fusion import TsdfFusion, choose_voxel_size, fusion_step
from oh_my_slam.reconstruction.pointcloud import pixel_mask
from oh_my_slam.segmentation.api import OBB, UNSEGMENTED

# Map cloud = surface of a fine TSDF (voxel/2, wide band so frames that disagree by a few
# centimetres still average into one surface), attributed from the latest update that sees it.
CLOUD_TRUNC_VOXELS = 8.0
# A surface voxel must be seen by this many frames (fewer in tiny maps), or by every frame that has
# it in view when fewer do (``_few_views``: a laptop at the corner of two photos).
CLOUD_MIN_VIEWS = 3
# A surface that fewer than CLOUD_MIN_VIEWS keyframes see is drawn only from keyframes whose depth
# fits the map: a scale measured with a spread (IQR / median of the per-point depth ratios) of at
# most FEW_VIEWS_MAX_SPREAD. A keyframe placed or scaled worse has nothing to check it there.
FEW_VIEWS_MAX_SPREAD = 0.1
FEW_VIEWS_CELL = 0.25  # m: frustum culling of the candidate points, per keyframe
VIS_TOL_MIN = 0.02
VIS_TOL_REL = 0.03
LABEL_SHARE_DIVISOR = 3  # an object id needs the votes of >= 1/3 of the views that see a point
ATTRIBUTE_CHUNK = 1_000_000  # points attributed at a time (bounds the vote's memory)
# A vote for an object counts only within its box grown by max(ATTRIBUTE_MARGIN_M,
# ATTRIBUTE_MARGIN_REL · its viewing distance): the fused surface lies within a few centimetres of
# the points the box was fitted to (TSDF band 4 cm, box at the 2-98 % extent), plus the depth
# disagreement of the keyframes that saw it (p90 ~3 % of the distance after the global depth
# adjustment), which matters most across a thin box (a dishwasher front 6 cm deep at 3 m).
ATTRIBUTE_MARGIN_M = 0.05
ATTRIBUTE_MARGIN_REL = 0.03
# The surface of a confirmed object beyond its votes: the SUPPORT_NEIGHBOURS unlabelled cloud points
# nearest each of its own lifted points (at least SUPPORT_MIN_POINTS of them), within
# max(SUPPORT_RADIUS_MIN, SUPPORT_RADIUS_REL · its viewing distance) — the depth disagreement of
# the keyframes that saw it and of the fused surface — and inside its attribution gate; a point
# several objects pick goes to the one whose own points are nearest.
SUPPORT_MIN_POINTS = 30
SUPPORT_NEIGHBOURS = 4
SUPPORT_RADIUS_MIN = 0.03
SUPPORT_RADIUS_REL = 0.02
# The place of a removed object in a keyframe that detected it: its retired pixels, at or behind its
# surface there up to the depth noise max(VACATED_MARGIN_M, VACATED_MARGIN_REL · depth) in front
# (``objects.absence_tau``'s floor).
VACATED_MARGIN_M = 0.05
VACATED_MARGIN_REL = 0.08


@dataclass
class FrameData:
    rec: store.FrameRecord
    depth: NDArray[np.float32]
    valid: NDArray[np.bool_]
    rgb: NDArray[np.uint8]
    labels: NDArray[np.int32]
    is_new: bool


@dataclass
class MapGeometry:
    cloud: PointCloud  # map cloud, label = object id
    new_cloud: PointCloud  # points of this update's keyframes
    stats: dict[str, Any]
    nearest: dict[int, float] = field(default_factory=dict)  # nearest_detections


@dataclass
class FusedCloud:
    """The fused surface of the map's confident keyframes (``fuse_map``), before object ids are
    attributed to it: the objects of the update use it (``objects.update_objects``)."""

    frames: list[FrameData]  # the confident keyframes (their labels are set by build_geometry)
    xyz: NDArray[np.float64]
    voxel: float
    seconds: float
    focal: float = 0.0  # focal length (px) of the fused depth grids (median over the keyframes)
    depth_max: float = float("inf")  # the fusion's depth cut (``fusion_depth_max`` per keyframe)
    vacated: list[Vacated] = field(default_factory=list)  # places of removed objects (``_vacated``)
    retired: list[FrameData] = field(default_factory=list)  # every keyframe, fused or not


def _frame_data(ctx: Any, rec: store.FrameRecord, new_by_name: dict[str, Any]) -> FrameData:
    """A keyframe's aligned depth, validity and colour (object ids not yet set)."""
    tx = ctx.tx
    nf = new_by_name.get(rec.name)
    if nf is not None and nf.depth is not None:
        depth = nf.depth
        rgb = nf.frame.rgb
    else:
        depth = store.load_depth(tx.current, rec.name)
        rgb = load_rgb(tx.current(rec.image), max_side=max(rec.grid_width, rec.grid_height))
    valid = store.load_valid(tx.current, rec.name, depth)
    return FrameData(rec, depth, valid, rgb, np.zeros(depth.shape, np.int32), nf is not None)


def _frame_labels(ctx: Any, rec: store.FrameRecord, shape: tuple[int, ...], objs: ObjectState
                  ) -> NDArray[np.int32]:
    """Per-pixel persistent object ids of a keyframe (its instances as stored or staged)."""
    inst_p = ctx.tx.current(store.frame_file(rec.name, "instances.json"))
    insts = json.loads(inst_p.read_text()).get("instances", []) if inst_p.exists() else []
    return label_map_for(insts, (int(shape[0]), int(shape[1])), objs)


def fusion_depth_max(fd: FrameData, depth_max: float) -> float:
    """How deep keyframe ``fd`` is fused: ``depth_max`` where the keyframe placed it before its
    near/far correction (``mapping.api._adjust_depth_scales``: about its median depth m,
    d' = m · (d / m) ** exponent, the factor within [1/MAX_FACTOR, MAX_FACTOR]).

    The correction moves a keyframe's surfaces; it does not change which of them the keyframe
    contributes. Outdoors the keyframes tilt by exponents up to ~1.2 (far field 15-35 % deeper):
    cut at a fixed ``depth_max``, the keyframes that see a facade from 25-30 m no longer counted
    for it, and the facades 10-20 m from a street walk fell below ``CLOUD_MIN_VIEWS`` (street.mp4:
    8.2 M cloud points untilted, 6.9 M tilted)."""
    e = float(fd.rec.stats.get("depth_exponent", 1.0))
    d = fd.depth[fd.valid & (fd.depth > 0)]
    if e == 1.0 or not len(d):
        return depth_max
    med = float(np.median(d))
    return float(np.clip(med * (depth_max / med) ** e, depth_max / MAX_FACTOR,
                         depth_max * MAX_FACTOR))


def fused_cloud_points(frames: list[FrameData], voxel: float, depth_max: float,
                       vacated: list[Vacated] | None = None,
                       retired: list[FrameData] | None = None) -> NDArray[np.float64]:
    """Surface points of a fine TSDF of all frames' valid, edge-free depth, each frame up to
    ``depth_max`` as it placed it before its near/far correction (``fusion_depth_max``).

    Each keyframe's monocular depth disagrees with its neighbours by a few percent even after
    alignment, so back-projecting every frame leaves one offset copy of each surface per view;
    the TSDF averages them into a single surface. A surface is kept when ``CLOUD_MIN_VIEWS``
    frames fused it, or, where fewer frames have it in view, when all of them did
    (``_few_views``): speckle that other frames look at and do not see is dropped, a laptop at
    the corner of two photos is not. Frames are fused in ``FrameRecord.order_key`` order and the
    points are returned sorted, so the cloud does not depend on the keyframes' order within an
    update.

    ``vacated``: the places of removed objects, drawn from the keyframes that saw through them
    (``_vacated``; ``retired``: the keyframes whose pixels were retired, fused or not).
    """
    fusion = TsdfFusion(voxel, depth_max, trunc_voxels=CLOUD_TRUNC_VOXELS)
    for fd in sorted(frames, key=lambda fd: fd.rec.order_key):
        m = pixel_mask(fd.depth, fd.valid)
        fusion.integrate(np.where(m, fd.depth, 0.0), fd.rec.K_grid.K(), fd.rec.T_map_cam,
                         depth_max=fusion_depth_max(fd, depth_max))
    views = max(1, min(CLOUD_MIN_VIEWS, fusion.stats.frames))
    pts = _extract(fusion, views)
    if views > 1:
        # the surfaces that fewer than ``views`` frames fused (Open3D's weight counts the frames
        # that updated a voxel: the ones that saw its surface or saw through it)
        low = _extract(fusion, 1)
        few = low[~_member(low, pts)]
        weight = np.where(_member(few, _extract(fusion, 2)), 2, 1) if views > 2 else np.ones(
            len(few), np.int64)
        add = _few_views(few, weight, frames, depth_max, views)
        keep = np.ones(len(pts), bool)
        if vacated:
            keep, witnessed = _vacated(pts, few, frames, vacated, retired or frames)
            add |= witnessed
        pts = np.concatenate([pts[keep], few[add]])
    # Open3D's parallel hash map returns the points in no fixed order
    return pts[np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))]


def _extract(fusion: TsdfFusion, views: int) -> NDArray[np.float64]:
    """The fused surface where at least ``views`` frames updated the voxels."""
    pts = fusion.extract_points(weight_threshold=views - 0.5)  # Open3D keeps weight > threshold
    return np.asarray(pts, dtype=np.float64).reshape(-1, 3)


def _member(a: NDArray[Any], b: NDArray[Any]) -> NDArray[np.bool_]:
    """Whether each point of ``a`` is one of ``b`` (the same float32 coordinates: surface points
    that TSDF extractions at two weight thresholds share are bit-identical)."""
    def key(p: NDArray[Any]) -> NDArray[np.uint64]:
        u = np.ascontiguousarray(p, np.float32).view(np.uint32).astype(np.uint64).reshape(-1, 3)
        with np.errstate(over="ignore"):
            return ((u[:, 0] * np.uint64(0x9E3779B97F4A7C15))
                    ^ (u[:, 1] * np.uint64(0xC2B2AE3D27D4EB4F))
                    ^ (u[:, 2] * np.uint64(0x165667B19E3779F9)) ^ (u[:, 0] << np.uint64(32)))

    if not len(a) or not len(b):
        return np.zeros(len(a), bool)
    return np.isin(key(a), key(b))


def _trusted(fd: FrameData) -> bool:
    """Whether a keyframe may draw alone what fewer than ``CLOUD_MIN_VIEWS`` keyframes see: its
    depth scale was measured with a spread of at most ``FEW_VIEWS_MAX_SPREAD``."""
    spread = fd.rec.stats.get("depth_scale_spread")
    return spread is None or float(spread) <= FEW_VIEWS_MAX_SPREAD


def _few_views(pts: NDArray[np.float64], weight: NDArray[Any], frames: list[FrameData],
               depth_max: float, views: int) -> NDArray[np.bool_]:
    """Which surface points that fewer than ``views`` frames fused (``weight`` frames) stay: every
    frame that has the point in view fused it — its image holds a fused pixel there (``valid``,
    edge-free) and the point lies within the frame's fused depth (``fusion_depth_max``), whatever
    the pixel shows — and a frame that sees it (``_visible``) is ``_trusted``.

    A surface seen by one or two keyframes because no other keyframe looks there (a laptop at the
    corner of two photos, the half of a monitor that one photo shows, the sill that the latest
    photos show where a removed cup stood) is kept; one that other keyframes look at without
    fusing it (they see a nearer surface, or through it) needs ``views`` of them, as before."""
    if not len(pts):
        return np.zeros(0, bool)
    cells = np.floor(pts / FEW_VIEWS_CELL).astype(np.int64)
    uc, inv = np.unique(cells, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    centres = (uc + 0.5) * FEW_VIEWS_CELL
    rad = FEW_VIEWS_CELL * float(np.sqrt(3.0)) / 2
    in_view = np.zeros(len(pts), np.int32)
    seen = np.zeros(len(pts), bool)
    for fd in frames:
        cut = fusion_depth_max(fd, depth_max)
        K = fd.rec.K_grid.K()
        cam = fd.rec.T_map_cam.inverse()
        h, w = fd.depth.shape
        uvc, zc = project(centres @ cam.R.T + cam.t, K)
        marg = rad * float(K[0, 0]) / np.maximum(zc - rad, 1e-3)
        with np.errstate(invalid="ignore"):
            near = (zc > -rad) & (zc - rad < cut)
            framed = ((uvc[:, 0] > -marg) & (uvc[:, 0] < w + marg)
                      & (uvc[:, 1] > -marg) & (uvc[:, 1] < h + marg))
        sel = np.flatnonzero((near & (framed | (zc < rad)))[inv])
        if not len(sel):
            continue
        uv, z = project(pts[sel] @ cam.R.T + cam.t, K)
        with np.errstate(invalid="ignore"):
            u = np.floor(uv[:, 0] + 0.5)
            v = np.floor(uv[:, 1] + 0.5)
            ok = (z > 0) & (z < cut) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        sel, z = sel[ok], z[ok]
        uu, vv = u[ok].astype(np.int64), v[ok].astype(np.int64)
        fused = pixel_mask(fd.depth, fd.valid)[vv, uu]
        in_view[sel[fused]] += 1
        if _trusted(fd):
            d = fd.depth[vv, uu]
            sees = fd.valid[vv, uu] & (np.abs(z - d) < np.maximum(VIS_TOL_MIN, VIS_TOL_REL * z))
            seen[sel[sees]] = True
    return (np.asarray(weight) >= np.minimum(views, in_view)) & seen


def _vacated_region(pts: NDArray[np.float64], place: Vacated, by_name: dict[str, FrameData]
                    ) -> NDArray[np.bool_]:
    """Whether each point lies in a removed object's place: it projects onto a pixel retired from
    a keyframe that detected the object, at or behind the object's surface there (up to the depth
    noise, max(VACATED_MARGIN_M, VACATED_MARGIN_REL · depth), in front of it): the object itself
    and what it hid from that keyframe."""
    out = np.zeros(len(pts), bool)
    for name, enc in place.masks.items():
        fd = by_name.get(name)
        if fd is None:
            continue
        mask = rle.decode(enc)
        if mask.shape != fd.depth.shape:
            continue
        cam = fd.rec.T_map_cam.inverse()
        uv, z = project(pts @ cam.R.T + cam.t, fd.rec.K_grid.K())
        h, w = mask.shape
        with np.errstate(invalid="ignore"):
            u = np.floor(uv[:, 0] + 0.5)
            v = np.floor(uv[:, 1] + 0.5)
            idx = np.flatnonzero((z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h))
        uu, vv = u[idx].astype(np.int64), v[idx].astype(np.int64)
        d = fd.depth[vv, uu].astype(np.float64)
        hit = mask[vv, uu] & (d > 0)
        hit &= z[idx] >= d - np.maximum(VACATED_MARGIN_M, VACATED_MARGIN_REL * d)
        out[idx[hit]] = True
    return out


def _seen_through(fd: FrameData, pts: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Whether keyframe ``fd`` sees clearly behind each point: it projects onto a valid pixel whose
    depth lies beyond it by more than the visibility tolerance (``_visible``)."""
    cam = fd.rec.T_map_cam.inverse()
    uv, z = project(pts @ cam.R.T + cam.t, fd.rec.K_grid.K())
    h, w = fd.depth.shape
    with np.errstate(invalid="ignore"):
        u = np.floor(uv[:, 0] + 0.5)
        v = np.floor(uv[:, 1] + 0.5)
        idx = np.flatnonzero((z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h))
    uu, vv = u[idx].astype(np.int64), v[idx].astype(np.int64)
    zi = z[idx]
    hit = fd.valid[vv, uu] & (fd.depth[vv, uu] - zi >= np.maximum(VIS_TOL_MIN, VIS_TOL_REL * zi))
    out = np.zeros(len(pts), bool)
    out[idx[hit]] = True
    return out


def _vacated(pts: NDArray[np.float64], few: NDArray[np.float64], frames: list[FrameData],
             vacated: list[Vacated], retired: list[FrameData]
             ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Latest wins in the places of removed objects (``Vacated``): what the keyframes that saw
    through an object (its witnesses) see there is the map's surface. Returns (which of the
    well-seen surface points ``pts`` stay, which of the surface points fewer frames fused
    ``few`` are added).

    The pixels of the keyframes that detected the object are retired, so the surface it stood on
    or hid is seen only by its witnesses, where a surface usually needs ``CLOUD_MIN_VIEWS``
    views. In the place, a point that a witness sees is kept however few frames fused it, and a
    point that a witness sees through, which neither a witness nor a keyframe of a later update
    sees, is dropped: what is left of the object, drawn by keyframes that did not detect it or by
    pixels beside its masks. A later update that sees the place again is fused like any other."""
    by_name = {fd.rec.name: fd for fd in retired}
    fused = {fd.rec.name: fd for fd in frames}
    keep = np.ones(len(pts), bool)
    add = np.zeros(len(few), bool)
    for place in vacated:
        witnesses = [fused[n] for n in place.witnesses if n in fused]
        if not witnesses:
            continue
        later = [fd for fd in frames if fd.rec.update_id > place.update]
        for cloud, is_few in ((pts, False), (few, True)):
            region = np.flatnonzero(_vacated_region(cloud, place, by_name))
            if not len(region):
                continue
            q = cloud[region]
            seen = np.zeros(len(q), bool)
            through = np.zeros(len(q), bool)
            for fd in witnesses:
                seen[_visible(fd, q)[0]] = True
                through |= _seen_through(fd, q)
            if is_few:
                add[region[seen]] = True
                continue
            again = np.zeros(len(q), bool)
            for fd in later:
                again[_visible(fd, q)[0]] = True
            keep[region[through & ~seen & ~again]] = False
    return keep, add


def _visible(fd: FrameData, pts: NDArray[np.float64]
             ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64],
                        NDArray[np.float64]]:
    """(point indices, pixel rows, pixel cols, footprint in m/px) of the points ``fd`` sees: they
    project onto a valid pixel whose depth agrees within max(VIS_TOL_MIN, VIS_TOL_REL·z)."""
    cam = fd.rec.T_map_cam.inverse()
    pc = pts @ cam.R.T + cam.t
    K = fd.rec.K_grid
    uv, z = project(pc, K.K())
    h, w = fd.depth.shape
    with np.errstate(invalid="ignore"):
        u = np.floor(uv[:, 0] + 0.5)
        v = np.floor(uv[:, 1] + 0.5)
        idx = np.nonzero((z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h))[0]
    uu = u[idx].astype(np.int64)
    vv = v[idx].astype(np.int64)
    dz = fd.depth[vv, uu]
    vis = fd.valid[vv, uu] & (np.abs(z[idx] - dz) < np.maximum(VIS_TOL_MIN, VIS_TOL_REL * z[idx]))
    idx, uu, vv = idx[vis], uu[vis], vv[vis]
    return idx, vv, uu, z[idx] / K.fx


def attribution_margin(obs_depth: float) -> float:
    """How far beyond its box an object's cloud points may lie (its attribution gate), for an
    object seen from ``obs_depth`` metres."""
    return max(ATTRIBUTE_MARGIN_M, ATTRIBUTE_MARGIN_REL * float(obs_depth))


Gates = dict[int, tuple[OBB, float]]  # object id -> (box, margin of its attribution gate)


def attribution_gates(objs: ObjectState) -> Gates:
    """Each object's box and attribution margin (``attribution_margin``)."""
    return {o.id: (o.obb, attribution_margin(o.obs_depth)) for o in objs.objects
            if o.obb is not None}


def _in_boxes(pts: NDArray[np.float64], lab: NDArray[Any], gates: Gates) -> NDArray[np.bool_]:
    """Whether each point lies in the attribution gate of the object it is labelled with;
    objects without a box have none."""
    out = np.zeros(len(pts), bool)
    for oid in np.unique(lab).tolist():
        gate = gates.get(int(oid))
        if gate is not None:
            sel = lab == oid
            out[sel] = gate[0].contains(pts[sel], gate[1])
    return out


def _attribute_update(pts: NDArray[np.float64], frames: list[FrameData],
                      boxes: Gates | None = None
                      ) -> tuple[NDArray[np.bool_], NDArray[np.uint8], NDArray[np.int32]]:
    """(seen, colour, object id) of each point from the keyframes of one update, whatever their
    order: the colour of the finest view (smallest footprint; ties: larger colour, then larger id)
    and the object id with most votes among the views that see the point (ties: finest view),
    kept only if at least a third of those views give it. With ``boxes`` (object id → box and
    margin), a vote counts only for a point inside the voted object's gate (``_in_boxes``)."""
    n = len(pts)
    views = np.zeros(n, np.int32)
    best = np.full(n, np.inf)
    best_key = np.full(n, -1, np.int64)
    rgb = np.zeros((n, 3), np.uint8)
    p_idx, p_lab, p_fp = [], [], []
    for fd in frames:
        idx, vv, uu, fp = _visible(fd, pts)
        views[idx] += 1
        col = fd.rgb[vv, uu]
        lab = fd.labels[vv, uu]
        key = ((col[:, 0].astype(np.int64) << 16) | (col[:, 1].astype(np.int64) << 8)
               | col[:, 2].astype(np.int64)) * (1 << 31) + lab
        better = (fp < best[idx]) | ((fp == best[idx]) & (key > best_key[idx]))
        sel = idx[better]
        best[sel], best_key[sel], rgb[sel] = fp[better], key[better], col[better]
        on = lab > 0
        if boxes is not None and on.any():
            on[on] = _in_boxes(pts[idx[on]], lab[on], boxes)
        p_idx.append(idx[on])
        p_lab.append(lab[on])
        p_fp.append(fp[on])
    label = np.zeros(n, np.int32)
    P = np.concatenate(p_idx) if p_idx else np.zeros(0, np.int64)
    if len(P):
        L, F = np.concatenate(p_lab), np.concatenate(p_fp)
        o = np.lexsort((F, L, P))
        P, L, F = P[o], L[o], F[o]
        start = np.flatnonzero(np.r_[True, (P[1:] != P[:-1]) | (L[1:] != L[:-1])])
        votes = np.diff(np.r_[start, len(P)])
        gp, gl, gf = P[start], L[start], F[start]  # gf: finest view voting for (point, id)
        o = np.lexsort((gl, gf, -votes, gp))
        gp, gl, votes = gp[o], gl[o], votes[o]
        win = np.r_[True, gp[1:] != gp[:-1]]
        wp, wl, wv = gp[win], gl[win], votes[win]
        ok = wv * LABEL_SHARE_DIVISOR >= views[wp]
        label[wp[ok]] = wl[ok]
    return views > 0, rgb, label


def attribute_points(xyz: NDArray[Any], frames: list[FrameData],
                     boxes: Gates | None = None, vacated: list[Vacated] | None = None,
                     retired: list[FrameData] | None = None
                     ) -> tuple[NDArray[np.uint8], NDArray[np.int32], NDArray[np.bool_]]:
    """Colour and object id of each point from the latest update whose keyframes see it.

    Updates are applied oldest → newest, so a later update wins wherever it sees a point. The
    keyframes of one update are one observation: within it, their order never matters
    (``_attribute_update``; ``boxes`` gate the votes). In the place of an object an update
    removed (``vacated``; ``retired``: the keyframes whose pixels were retired), the keyframes
    that saw through it are the latest observation: they colour what they see there, unless a
    later update sees it too (what its detecting keyframes still see beside their retired masks,
    such as its shadow, is not its latest state). Returns (rgb, label, seen by a new frame);
    unseen points keep mid-grey and label 0.
    """
    n = len(xyz)
    rgb = np.full((n, 3), UNSEGMENTED, np.uint8)
    label = np.zeros(n, np.int32)
    seen_new = np.zeros(n, bool)
    pts_all = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    by_update: dict[int, list[FrameData]] = {}
    for fd in frames:
        by_update.setdefault(fd.rec.update_id, []).append(fd)
    for start in range(0, n, ATTRIBUTE_CHUNK):
        sl = slice(start, min(n, start + ATTRIBUTE_CHUNK))
        pts = pts_all[sl]
        for uid in sorted(by_update):
            seen, col, lab = _attribute_update(pts, by_update[uid], boxes)
            where = np.flatnonzero(seen) + start
            rgb[where] = col[seen]
            label[where] = lab[seen]
            if any(fd.is_new for fd in by_update[uid]):
                seen_new[where] = True
        _attribute_vacated(pts, start, frames, vacated or [], retired or frames, boxes, rgb, label)
    return rgb, label, seen_new


def _attribute_vacated(pts: NDArray[np.float64], start: int, frames: list[FrameData],
                       vacated: list[Vacated], retired: list[FrameData], boxes: Gates | None,
                       rgb: NDArray[np.uint8], label: NDArray[np.int32]) -> None:
    """Colour and object id, in the places of removed objects, from the keyframes that saw
    through them (``attribute_points``); ``pts`` are the points from index ``start``."""
    by_name = {fd.rec.name: fd for fd in retired}
    fused = {fd.rec.name: fd for fd in frames}
    for place in sorted(vacated, key=lambda v: (v.update, v.object)):
        witnesses = [fused[n] for n in place.witnesses if n in fused]
        if not witnesses:
            continue
        region = np.flatnonzero(_vacated_region(pts, place, by_name))
        if not len(region):
            continue
        q = pts[region]
        seen, col, lab = _attribute_update(q, witnesses, boxes)
        for fd in frames:
            if fd.rec.update_id > place.update:
                seen[_visible(fd, q)[0]] = False
        rgb[region[seen] + start] = col[seen]
        label[region[seen] + start] = lab[seen]


def _sighting_depths(objs: ObjectState, frames: list[FrameData], depth_max: float
                     ) -> dict[int, list[tuple[float, float]]]:
    """Per object: (depth in the keyframe's camera, that keyframe's fused depth) of each of its
    sightings in the fused keyframes ``frames``."""
    cams = {fd.rec.index: (fd.rec.T_map_cam, fusion_depth_max(fd, depth_max)
                           if np.isfinite(depth_max) else depth_max) for fd in frames}
    out: dict[int, list[tuple[float, float]]] = {}
    for o in objs.objects:
        for sg in o.sightings:
            cam = cams.get(sg.frame)
            if cam is not None:
                z = float((np.asarray(sg.centroid, np.float64) - cam[0].t) @ cam[0].R[:, 2])
                out.setdefault(o.id, []).append((z, cam[1]))
    return out


def nearest_detections(objs: ObjectState, frames: list[FrameData]) -> dict[int, float]:
    """Per object, the depth (in that keyframe's camera) of its nearest sighting among the fused
    keyframes ``frames``: its finest view, which sampled it most densely
    (``objects.min_cloud_points``). Objects without such a sighting are left out."""
    ahead = {oid: [z for z, _ in zs if z > 0]
             for oid, zs in _sighting_depths(objs, frames, float("inf")).items()}
    return {oid: min(zs) for oid, zs in ahead.items() if zs}


def fused_objects(objs: ObjectState, frames: list[FrameData], depth_max: float) -> set[int]:
    """Ids of the objects that a keyframe which detected them fused: one of their sightings (its
    centroid, in that keyframe's camera) lies within the keyframe's fused depth
    (``fusion_depth_max``) — ``frames`` are the fused keyframes.

    Only these take support (``support_labels``). An object detected only beyond that depth (a
    car 40-100 m down a street) has no surface of its own in the cloud: its lifted points are
    placed by far monocular depth, its gate and support radius grow to 1-3 m, and the cloud
    points near them are other surfaces (the road, a facade). Its votes still count: a point
    that its keyframes see at the depth where they detected it."""
    return {oid for oid, zs in _sighting_depths(objs, frames, depth_max).items()
            if any(0.0 < z < cut for z, cut in zs)}


def support_labels(xyz: NDArray[Any], label: NDArray[np.int32], objs: ObjectState,
                   fused: set[int] | None = None) -> int:
    """Give the confirmed objects the unlabelled cloud points of their own surface: each of an
    object's own lifted points (``objects.fit_points``) picks the ``SUPPORT_NEIGHBOURS`` nearest
    unlabelled cloud points inside the object's attribution gate, within ``support_radius``; a
    point that several objects pick takes the id of the object whose own point is nearest (ties:
    the lower id), so the order of the objects does not matter. With ``fused``, only the objects
    it holds take points (``fused_objects``). ``label`` is modified in place. Returns how many
    objects took points."""
    from scipy.spatial import cKDTree

    pts_all = np.asarray(xyz, np.float64).reshape(-1, 3)
    free = label == 0
    best_d = np.full(len(pts_all), np.inf)
    best_id = np.zeros(len(pts_all), np.int32)
    for o in sorted(objs.objects, key=lambda o: o.id):
        if not o.confirmed or o.obb is None or (fused is not None and o.id not in fused):
            continue
        own = np.asarray(fit_points(o), np.float64)
        if len(own) < SUPPORT_MIN_POINTS:
            continue
        cand = np.flatnonzero(free & o.obb.contains(pts_all, attribution_margin(o.obs_depth)))
        if not len(cand):
            continue
        k = min(SUPPORT_NEIGHBOURS, len(cand))
        d, j = cKDTree(pts_all[cand]).query(own, k=k,
                                            distance_upper_bound=support_radius(o.obs_depth))
        d, j = np.reshape(d, -1), np.reshape(j, -1)
        ok = np.isfinite(d)
        picked, dist = cand[j[ok]], d[ok]
        if not len(picked):
            continue
        order = np.lexsort((dist, picked))  # per picked point, its nearest own point first
        picked, dist = picked[order], dist[order]
        first = np.r_[True, picked[1:] != picked[:-1]]
        picked, dist = picked[first], dist[first]
        better = dist < best_d[picked]  # strict: a tie keeps the lower id
        best_d[picked[better]] = dist[better]
        best_id[picked[better]] = o.id
    take = best_id > 0
    label[take] = best_id[take]
    return len(np.unique(best_id[take]))


def support_radius(obs_depth: float) -> float:
    """How far a cloud point may lie from an object's own lifted points to be its surface, for
    an object seen from ``obs_depth`` metres."""
    return max(SUPPORT_RADIUS_MIN, SUPPORT_RADIUS_REL * float(obs_depth))


def fuse_map(ctx: Any, records: list[store.FrameRecord]) -> FusedCloud:
    """The fused surface of the map's confident keyframes (timed as the stage ``cloud``)."""
    t0 = time.perf_counter()
    with timing.stage("cloud"):
        new_by_name = {nf.kf.name: nf for nf in ctx.new if nf.record is not None}
        frames = [_frame_data(ctx, r, new_by_name)
                  for r in sorted(records, key=lambda r: r.order_key)]
        depths = [np.median(fd.depth[fd.valid & (fd.depth > 0)]) for fd in frames
                  if (fd.valid & (fd.depth > 0)).any()]
        med = float(np.median(depths)) if depths else 2.0
        voxel = choose_voxel_size(med)
        depth_max = float(np.clip(2.5 * med, 3.0, 30.0))
        cloud_voxel = max(0.005, voxel / 2)
        confident = [fd for fd in frames if not fd.rec.low_confidence]
        vacated = load_vacated(ctx.tx.current)
        xyz = fused_cloud_points(confident, cloud_voxel, depth_max, vacated, frames)
        focal = float(np.median([fd.rec.K_grid.fx / fusion_step(fd.depth.shape)
                                 for fd in confident])) if confident else 0.0
    return FusedCloud(confident, xyz, cloud_voxel, time.perf_counter() - t0, focal, depth_max,
                      vacated, frames)


def build_geometry(ctx: Any, records: list[store.FrameRecord], objs: ObjectState,
                   progress: Any, fused: FusedCloud | None = None) -> MapGeometry:
    """The map cloud with its object ids (timed as the stage ``cloud``): the fused surface
    (``fused``, else fused here) attributed from the keyframes' instances."""
    tx = ctx.tx
    fused = fused if fused is not None else fuse_map(ctx, records)
    t0 = time.perf_counter()
    with timing.stage("cloud"):
        for fd in fused.frames:
            fd.labels = _frame_labels(ctx, fd.rec, fd.depth.shape, objs)
        xyz = fused.xyz
        rgb, label, seen_new = attribute_points(xyz, fused.frames, attribution_gates(objs),
                                                fused.vacated, fused.retired)
        supported = support_labels(xyz, label, objs,
                                   fused_objects(objs, fused.frames, fused.depth_max))
        cloud = PointCloud(xyz, rgb, label)
        new_cloud = cloud.subset(np.nonzero(seen_new)[0])
        assert cloud.label is not None
        tx.write_bytes(store.CLOUD_PLY, ply_bytes(PointCloud(cloud.xyz, cloud.rgb),
                                                  comments=["oh-my-slam map cloud, metres, z up"]))
        tx.save_npy(store.CLOUD_OBJECTS, cloud.label.astype(np.int32))
    progress(f"cloud: {len(cloud)} points (voxel {fused.voxel * 100:.1f} cm) in "
             f"{fused.seconds + time.perf_counter() - t0:.0f} s")
    stats = {"cloud_points": len(cloud), "voxel": fused.voxel, "focal_px": round(fused.focal, 2),
             "support_objects": supported}
    ctx.notes["geometry"] = stats
    return MapGeometry(cloud, new_cloud, stats, nearest_detections(objs, fused.frames))
