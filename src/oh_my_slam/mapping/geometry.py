"""Map geometry for an update: the coloured cloud (surface of a TSDF fusion of the valid, aligned
depth maps; colour and object id per point from the latest update that sees it), built with the
reconstruction package's fusion code.

Object ids come from the keyframes' instance masks, which a detector draws generously: a "carpet"
mask that covers a counter top and the floor beyond it, of which only the counter was lifted into
the object (lifting keeps a mask's largest spatial cluster). A keyframe's vote for an object
therefore counts only for points inside the object's box grown by ``ATTRIBUTE_MARGIN_M``, so an
object's points in the cloud (``segments.ply``, ``color=segment``, its ``point_count``) coincide
with its box. A confirmed object that wins no point this way — its detecting keyframes are fewer
than a third of those that see its surface, like a light switch on a wall — takes the unlabelled
cloud points nearest to its own lifted points (``support_labels``)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core import timing
from oh_my_slam.core.geometry import project
from oh_my_slam.core.images import load_rgb
from oh_my_slam.core.ply import PointCloud, ply_bytes
from oh_my_slam.mapping import store
from oh_my_slam.mapping.objects import (
    EXPORT_MIN_CLOUD_POINTS,
    ObjectState,
    fit_points,
    label_map_for,
)
from oh_my_slam.reconstruction.fusion import TsdfFusion, choose_voxel_size
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
# A vote for an object counts only within its box grown by this: the fused surface lies within a
# few centimetres of the points the box was fitted to (TSDF band 4 cm, box at the 2-98 % extent).
ATTRIBUTE_MARGIN_M = 0.05
# Fallback of a confirmed object without cloud points: the SUPPORT_NEIGHBOURS unlabelled cloud
# points nearest each of its own lifted points (at least SUPPORT_MIN_POINTS of them), within
# max(SUPPORT_RADIUS_MIN, SUPPORT_RADIUS_REL · its viewing distance) — the depth disagreement of
# the keyframes that saw it and of the fused surface — and inside its grown box.
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


@dataclass
class FusedCloud:
    """The fused surface of the map's confident keyframes (``fuse_map``), before object ids are
    attributed to it: the objects of the update use it (``objects.update_objects``)."""

    frames: list[FrameData]  # the confident keyframes (their labels are set by build_geometry)
    xyz: NDArray[np.float64]
    voxel: float
    seconds: float


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


def fused_cloud_points(frames: list[FrameData], voxel: float, depth_max: float
                       ) -> NDArray[np.float64]:
    """Surface points of a fine TSDF of all frames' valid, edge-free depth.

    Each keyframe's monocular depth disagrees with its neighbours by a few percent even after
    alignment, so back-projecting every frame leaves one offset copy of each surface per view;
    the TSDF averages them into a single surface. Speckle seen by one view only is dropped once
    enough frames are fused. Frames are fused in ``FrameRecord.order_key`` order and the points
    are returned sorted, so the cloud does not depend on the keyframes' order within an update.
    """
    fusion = TsdfFusion(voxel, depth_max, trunc_voxels=CLOUD_TRUNC_VOXELS)
    for fd in sorted(frames, key=lambda fd: fd.rec.order_key):
        m = pixel_mask(fd.depth, fd.valid)
        fusion.integrate(np.where(m, fd.depth, 0.0), fd.rec.K_grid.K(), fd.rec.T_map_cam)
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


def _in_boxes(pts: NDArray[np.float64], lab: NDArray[Any], boxes: dict[int, OBB]
              ) -> NDArray[np.bool_]:
    """Whether each point lies in the box (grown by ``ATTRIBUTE_MARGIN_M``) of the object it is
    labelled with; objects without a box have none."""
    out = np.zeros(len(pts), bool)
    for oid in np.unique(lab).tolist():
        box = boxes.get(int(oid))
        if box is not None:
            sel = lab == oid
            out[sel] = box.contains(pts[sel], ATTRIBUTE_MARGIN_M)
    return out


def _attribute_update(pts: NDArray[np.float64], frames: list[FrameData],
                      boxes: dict[int, OBB] | None = None
                      ) -> tuple[NDArray[np.bool_], NDArray[np.uint8], NDArray[np.int32]]:
    """(seen, colour, object id) of each point from the keyframes of one update, whatever their
    order: the colour of the finest view (smallest footprint; ties: larger colour, then larger id)
    and the object id with most votes among the views that see the point (ties: finest view),
    kept only if at least a third of those views give it. With ``boxes`` (object id → OBB), a
    vote counts only for a point inside the voted object's grown box (``_in_boxes``)."""
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
                     boxes: dict[int, OBB] | None = None
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


def support_labels(xyz: NDArray[Any], label: NDArray[np.int32], objs: ObjectState) -> int:
    """Give each confirmed object with a box but fewer than ``EXPORT_MIN_CLOUD_POINTS`` cloud
    points the unlabelled cloud points its own lifted points (``objects.fit_points``) pick: the
    ``SUPPORT_NEIGHBOURS`` nearest to each, within the support radius and inside its grown box.
    Objects are served in id order; ``label`` is modified in place. Returns how many objects
    took points."""
    from scipy.spatial import cKDTree

    pts_all = np.asarray(xyz, np.float64).reshape(-1, 3)
    counts = np.bincount(label[label > 0]) if (label > 0).any() else np.zeros(1, np.int64)
    served = 0
    for o in sorted(objs.objects, key=lambda o: o.id):
        have = int(counts[o.id]) if o.id < len(counts) else 0
        if not o.confirmed or o.obb is None or have >= EXPORT_MIN_CLOUD_POINTS:
            continue
        own = np.asarray(fit_points(o), np.float64)
        if len(own) < SUPPORT_MIN_POINTS:
            continue
        cand = np.flatnonzero(o.obb.contains(pts_all, ATTRIBUTE_MARGIN_M) & (label == 0))
        if not len(cand):
            continue
        radius = max(SUPPORT_RADIUS_MIN, SUPPORT_RADIUS_REL * o.obs_depth)
        k = min(SUPPORT_NEIGHBOURS, len(cand))
        d, j = cKDTree(pts_all[cand]).query(own, k=k, distance_upper_bound=radius)
        d, j = np.reshape(d, (len(own), k)), np.reshape(j, (len(own), k))
        take = cand[np.unique(j[np.isfinite(d)])]
        if len(take):
            label[take] = o.id
            served += 1
    return served


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
    return FusedCloud(confident, xyz, cloud_voxel, time.perf_counter() - t0)


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
        boxes = {o.id: o.obb for o in objs.objects if o.obb is not None}
        rgb, label, seen_new = attribute_points(xyz, fused.frames, boxes)
        supported = support_labels(xyz, label, objs)
        cloud = PointCloud(xyz, rgb, label)
        new_cloud = cloud.subset(np.nonzero(seen_new)[0])
        assert cloud.label is not None
        tx.write_bytes(store.CLOUD_PLY, ply_bytes(PointCloud(cloud.xyz, cloud.rgb),
                                                  comments=["oh-my-slam map cloud, metres, z up"]))
        tx.save_npy(store.CLOUD_OBJECTS, cloud.label.astype(np.int32))
    progress(f"cloud: {len(cloud)} points (voxel {fused.voxel * 100:.1f} cm) in "
             f"{fused.seconds + time.perf_counter() - t0:.0f} s")
    stats = {"cloud_points": len(cloud), "voxel": fused.voxel, "support_fallback": supported}
    ctx.notes["geometry"] = stats
    return MapGeometry(cloud, new_cloud, stats)
