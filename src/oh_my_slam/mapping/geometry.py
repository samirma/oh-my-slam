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

from oh_my_slam.core import timing
from oh_my_slam.core.geometry import project
from oh_my_slam.core.images import load_rgb
from oh_my_slam.core.ply import PointCloud, ply_bytes
from oh_my_slam.mapping import store
from oh_my_slam.mapping.objects import ObjectState, fit_points, label_map_for
from oh_my_slam.reconstruction.depth import MAX_FACTOR
from oh_my_slam.reconstruction.fusion import TsdfFusion, choose_voxel_size, fusion_step
from oh_my_slam.reconstruction.pointcloud import pixel_mask
from oh_my_slam.segmentation.api import OBB, UNSEGMENTED

# Map cloud = surface of a fine TSDF (voxel/2, wide band so frames that disagree by a few
# centimetres still average into one surface), attributed from the latest update that sees it.
CLOUD_TRUNC_VOXELS = 8.0
CLOUD_MIN_VIEWS = 3  # a surface voxel must be seen by this many frames (fewer in tiny maps)
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


def fused_cloud_points(frames: list[FrameData], voxel: float, depth_max: float
                       ) -> NDArray[np.float64]:
    """Surface points of a fine TSDF of all frames' valid, edge-free depth, each frame up to
    ``depth_max`` as it placed it before its near/far correction (``fusion_depth_max``).

    Each keyframe's monocular depth disagrees with its neighbours by a few percent even after
    alignment, so back-projecting every frame leaves one offset copy of each surface per view;
    the TSDF averages them into a single surface. Speckle seen by one view only is dropped once
    enough frames are fused. Frames are fused in ``FrameRecord.order_key`` order and the points
    are returned sorted, so the cloud does not depend on the keyframes' order within an update.
    """
    fusion = TsdfFusion(voxel, depth_max, trunc_voxels=CLOUD_TRUNC_VOXELS)
    for fd in sorted(frames, key=lambda fd: fd.rec.order_key):
        m = pixel_mask(fd.depth, fd.valid)
        fusion.integrate(np.where(m, fd.depth, 0.0), fd.rec.K_grid.K(), fd.rec.T_map_cam,
                         depth_max=fusion_depth_max(fd, depth_max))
    views = max(1, min(CLOUD_MIN_VIEWS, fusion.stats.frames))
    pts = fusion.extract_points(weight_threshold=views - 0.5)  # Open3D keeps weight > threshold
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    # Open3D's parallel hash map returns the points in no fixed order
    return pts[np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))]


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
                     boxes: Gates | None = None
                     ) -> tuple[NDArray[np.uint8], NDArray[np.int32], NDArray[np.bool_]]:
    """Colour and object id of each point from the latest update whose keyframes see it.

    Updates are applied oldest → newest, so a later update wins wherever it sees a point. The
    keyframes of one update are one observation: within it, their order never matters
    (``_attribute_update``; ``boxes`` gate the votes). Returns (rgb, label, seen by a new frame);
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
    return rgb, label, seen_new


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
        xyz = fused_cloud_points(confident, cloud_voxel, depth_max)
        focal = float(np.median([fd.rec.K_grid.fx / fusion_step(fd.depth.shape)
                                 for fd in confident])) if confident else 0.0
    return FusedCloud(confident, xyz, cloud_voxel, time.perf_counter() - t0, focal, depth_max)


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
        rgb, label, seen_new = attribute_points(xyz, fused.frames, attribution_gates(objs))
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
