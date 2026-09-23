"""Map geometry for an update: the coloured cloud (latest colour wins, object id per point) and
the textured mesh (TSDF fusion of the valid, aligned depth maps, oldest first) — built with the
reconstruction package's fusion/mesh/texture code."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core import timing
from oh_my_slam.core.geometry import voxel_downsample_indices
from oh_my_slam.core.images import load_rgb
from oh_my_slam.core.ply import PointCloud, ply_bytes
from oh_my_slam.mapping import store
from oh_my_slam.mapping.objects import ObjectState, label_map_for, load_valid
from oh_my_slam.reconstruction.fusion import TsdfFusion, choose_voxel_size
from oh_my_slam.reconstruction.mesh import clean_mesh
from oh_my_slam.reconstruction.pointcloud import cloud_mask, frame_cloud
from oh_my_slam.reconstruction.texture import TextureView, texture_mesh

CLOUD_STRIDE = 2
MIN_TEXTURE_VALID = 0.7
MAX_TEXTURE_VIEWS = 300
MESH_MAX_FACES = 200_000


@dataclass
class FrameData:
    rec: store.FrameRecord
    depth: NDArray[np.float32]
    valid: NDArray[np.bool_]
    rgb: NDArray[np.uint8]
    labels: NDArray[np.int32]
    image_path: Any
    is_new: bool


@dataclass
class MapGeometry:
    cloud: PointCloud  # map cloud, label = object id
    new_cloud: PointCloud  # points of this update's keyframes
    mesh_method: str
    voxel: float
    stats: dict[str, Any]


def _frame_data(ctx: Any, rec: store.FrameRecord, objs: ObjectState,
                new_by_name: dict[str, Any]) -> FrameData:
    tx = ctx.tx
    d = f"per_frame/{rec.name}"
    nf = new_by_name.get(rec.name)
    if nf is not None and nf.depth is not None:
        depth = nf.depth
        rgb = nf.frame.rgb
        image_path = tx.current(rec.image)
    else:
        depth = np.load(tx.current(f"{d}/depth.npy")).astype(np.float32)
        image_path = tx.current(rec.image)
        rgb = load_rgb(image_path, max_side=max(rec.grid_width, rec.grid_height))
    valid = load_valid(tx.current(f"{d}/valid.png"), depth)
    inst_p = tx.current(f"{d}/instances.json")
    import json

    insts = json.loads(inst_p.read_text()).get("instances", []) if inst_p.exists() else []
    labels = label_map_for(insts, depth.shape, objs)
    return FrameData(rec, depth, valid, rgb, labels, image_path, nf is not None)


def frames_cloud(frames: list[FrameData], voxel: float) -> PointCloud:
    """All frames' valid points (oldest first), voxel-downsampled keeping the latest point."""
    parts = []
    for fd in frames:
        m = cloud_mask(fd.depth, fd.valid)
        sub = np.zeros_like(m)
        sub[::CLOUD_STRIDE, ::CLOUD_STRIDE] = True
        m &= sub
        cloud, idx = frame_cloud(fd.depth, fd.rgb, fd.rec.K_grid, m, fd.rec.T_map_cam)
        cloud.label = fd.labels.reshape(-1)[idx].astype(np.int32)
        parts.append(cloud)
    allc = PointCloud.concat(parts)
    if len(allc) == 0:
        return allc
    keep = voxel_downsample_indices(allc.xyz, voxel, keep="last")
    return allc.subset(keep)


def build_geometry(ctx: Any, records: list[store.FrameRecord], objs: ObjectState,
                   progress: Any) -> MapGeometry:
    """Cloud, TSDF mesh and texture (timed as the stages cloud / fusion / mesh / texture)."""
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
        cloud_voxel = max(0.005, voxel / 2)
        cloud = frames_cloud(frames, cloud_voxel)
        new_cloud = frames_cloud([fd for fd in frames if fd.is_new], cloud_voxel)
        assert cloud.label is not None
        tx.write_bytes(store.CLOUD_PLY, ply_bytes(PointCloud(cloud.xyz, cloud.rgb),
                                                  comment="oh-my-slam map cloud, metres, z up"))
        tx.save_npy(store.CLOUD_OBJECTS, cloud.label.astype(np.int32))
    progress(f"cloud: {len(cloud)} points (voxel {cloud_voxel * 100:.1f} cm) in "
             f"{time.perf_counter() - t0:.0f} s")
    t1 = time.perf_counter()
    with timing.stage("fusion"):
        depth_max = float(np.clip(2.5 * med, 3.0, 30.0))
        fusion = TsdfFusion(voxel, depth_max)
        for fd in frames:
            if fd.rec.low_confidence:
                continue
            fusion.integrate(np.where(fd.valid, fd.depth, 0.0), fd.rgb, fd.rec.K_grid.K(),
                             fd.rec.T_map_cam)
    # OpenMVS view selection is superlinear in the face count (100 k faces: 2 s, 250 k: 32 s,
    # 740 k: > 10 min on a synthetic room); the texture carries the detail, so the textured
    # mesh is decimated to MESH_MAX_FACES.
    with timing.stage("mesh"):
        mesh = clean_mesh(fusion.extract_mesh(), max_faces=MESH_MAX_FACES)
    with timing.stage("texture"):
        views = [
            TextureView(fd.image_path, fd.rec.K.K(), fd.rec.width, fd.rec.height,
                        fd.rec.T_map_cam)
            for fd in frames
            if not fd.rec.low_confidence and fd.valid.mean() >= MIN_TEXTURE_VALID
        ]
        if len(views) > MAX_TEXTURE_VIEWS:
            keep = np.linspace(0, len(views) - 1, MAX_TEXTURE_VIEWS).round().astype(int)
            views = [views[i] for i in keep]
        out = tx.stage(store.MESH_GLB)
        method = texture_mesh(mesh, views, out, ctx.work / "texture") if len(mesh.triangles) \
            else "empty"
    progress(f"mesh: {len(mesh.triangles)} faces, textured with {method} from {len(views)} "
             f"views in {time.perf_counter() - t1:.0f} s")
    stats = {"cloud_points": len(cloud), "mesh_faces": len(mesh.triangles), "voxel": voxel,
             "texture": method, "texture_views": len(views)}
    ctx.notes["geometry"] = stats
    return MapGeometry(cloud, new_cloud, method, voxel, stats)
