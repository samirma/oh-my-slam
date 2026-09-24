"""Map geometry for an update: the coloured cloud (surface of a TSDF fusion of the valid, aligned
depth maps; latest colour wins, object id per point), built with the reconstruction package's
fusion code."""

from __future__ import annotations

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
from oh_my_slam.mapping.objects import ObjectState, label_map_for, load_valid
from oh_my_slam.reconstruction.fusion import TsdfFusion, choose_voxel_size
from oh_my_slam.reconstruction.pointcloud import pixel_mask
from oh_my_slam.segmentation.colors import UNSEGMENTED

# Map cloud = surface of a fine TSDF (voxel/2, wide band so frames that disagree by a few
# centimetres still average into one surface), attributed from the latest frame that sees it.
CLOUD_TRUNC_VOXELS = 8.0
CLOUD_MIN_VIEWS = 3  # a surface voxel must be seen by this many frames (fewer in tiny maps)
VIS_TOL_MIN = 0.02
VIS_TOL_REL = 0.03


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


def _frame_data(ctx: Any, rec: store.FrameRecord, objs: ObjectState,
                new_by_name: dict[str, Any]) -> FrameData:
    tx = ctx.tx
    d = f"per_frame/{rec.name}"
    nf = new_by_name.get(rec.name)
    if nf is not None and nf.depth is not None:
        depth = nf.depth
        rgb = nf.frame.rgb
    else:
        depth = np.load(tx.current(f"{d}/depth.npy")).astype(np.float32)
        rgb = load_rgb(tx.current(rec.image), max_side=max(rec.grid_width, rec.grid_height))
    valid = load_valid(tx.current(f"{d}/valid.png"), depth)
    inst_p = tx.current(f"{d}/instances.json")
    import json

    insts = json.loads(inst_p.read_text()).get("instances", []) if inst_p.exists() else []
    labels = label_map_for(insts, depth.shape, objs)
    return FrameData(rec, depth, valid, rgb, labels, nf is not None)


def fused_cloud_points(frames: list[FrameData], voxel: float, depth_max: float
                       ) -> NDArray[np.float64]:
    """Surface points of a fine TSDF of all frames' valid, edge-free depth.

    Each keyframe's monocular depth disagrees with its neighbours by a few percent even after
    alignment, so back-projecting every frame leaves one offset copy of each surface per view;
    the TSDF averages them into a single surface. Speckle seen by one view only is dropped once
    enough frames are fused.
    """
    fusion = TsdfFusion(voxel, depth_max, trunc_voxels=CLOUD_TRUNC_VOXELS)
    for fd in frames:
        m = pixel_mask(fd.depth, fd.valid)
        fusion.integrate(np.where(m, fd.depth, 0.0), fd.rec.K_grid.K(), fd.rec.T_map_cam)
    views = max(1, min(CLOUD_MIN_VIEWS, fusion.stats.frames))
    pts = fusion.extract_points(weight_threshold=views - 0.5)  # Open3D keeps weight > threshold
    return np.asarray(pts, dtype=np.float64).reshape(-1, 3)


def attribute_points(xyz: NDArray[Any], frames: list[FrameData]
                     ) -> tuple[NDArray[np.uint8], NDArray[np.int32], NDArray[np.bool_]]:
    """Colour and object id of each point from the latest frame that sees it (oldest → newest).

    A frame sees a point when it projects onto a valid pixel whose depth agrees within
    max(VIS_TOL_MIN, VIS_TOL_REL·z). Returns (rgb, label, seen by a new frame); unseen points
    keep mid-grey and label 0.
    """
    n = len(xyz)
    rgb = np.full((n, 3), UNSEGMENTED, np.uint8)
    label = np.zeros(n, np.int32)
    seen_new = np.zeros(n, bool)
    pts = np.asarray(xyz, dtype=np.float64)
    for fd in frames:
        cam = fd.rec.T_map_cam.inverse()
        pc = pts @ cam.R.T + cam.t
        uv, z = project(pc, fd.rec.K_grid.K())
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
        rgb[idx] = fd.rgb[vv, uu]
        label[idx] = fd.labels[vv, uu]
        if fd.is_new:
            seen_new[idx] = True
    return rgb, label, seen_new


def build_geometry(ctx: Any, records: list[store.FrameRecord], objs: ObjectState,
                   progress: Any) -> MapGeometry:
    """The map cloud with its object ids (timed as the stage ``cloud``)."""
    tx = ctx.tx
    t0 = time.perf_counter()
    with timing.stage("cloud"):
        new_by_name = {nf.kf.name: nf for nf in ctx.new if nf.record is not None}
        frames = [_frame_data(ctx, r, objs, new_by_name)
                  for r in sorted(records, key=lambda r: r.index)]
        depths = [np.median(fd.depth[fd.valid & (fd.depth > 0)]) for fd in frames
                  if (fd.valid & (fd.depth > 0)).any()]
        med = float(np.median(depths)) if depths else 2.0
        voxel = choose_voxel_size(med)
        depth_max = float(np.clip(2.5 * med, 3.0, 30.0))
        cloud_voxel = max(0.005, voxel / 2)
        confident = [fd for fd in frames if not fd.rec.low_confidence]
        xyz = fused_cloud_points(confident, cloud_voxel, depth_max)
        rgb, label, seen_new = attribute_points(xyz, confident)
        cloud = PointCloud(xyz, rgb, label)
        new_cloud = cloud.subset(np.nonzero(seen_new)[0])
        assert cloud.label is not None
        tx.write_bytes(store.CLOUD_PLY, ply_bytes(PointCloud(cloud.xyz, cloud.rgb),
                                                  comments=["oh-my-slam map cloud, metres, z up"]))
        tx.save_npy(store.CLOUD_OBJECTS, cloud.label.astype(np.int32))
    progress(f"cloud: {len(cloud)} points (voxel {cloud_voxel * 100:.1f} cm) in "
             f"{time.perf_counter() - t0:.0f} s")
    stats = {"cloud_points": len(cloud), "voxel": cloud_voxel}
    ctx.notes["geometry"] = stats
    return MapGeometry(cloud, new_cloud, stats)
