"""Everything the browser shows, gathered only through the owning packages' APIs:
reconstruction + segmentation for an image, the read-only map store + export for a map. No
inference, fitting or colour logic lives here — colours come from segmentation's outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.geometry import rotation_between
from oh_my_slam.core.images import png_bytes
from oh_my_slam.core.ply import PointCloud

MAX_DISPLAY_POINTS = 3_000_000


@dataclass
class ViewBundle:
    mode: str  # "image" | "map"
    title: str
    scene: dict[str, Any]
    cloud: PointCloud  # display points (rgb)
    segments: NDArray[np.uint8]  # per-point segment colour (same order as cloud)
    labels: NDArray[np.int32]  # per-point object id (0 = none)
    catalog: list[dict[str, Any]]
    frustums: list[dict[str, Any]] = field(default_factory=list)
    mesh_path: Path | None = None
    segmented_png: bytes | None = None
    display_transform: list[list[float]] = field(default_factory=lambda: np.eye(4).tolist())
    stats: dict[str, Any] = field(default_factory=dict)

    def meta(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "title": self.title,
            "stats": self.stats,
            "display_transform": self.display_transform,
            "frustums": self.frustums,
            "catalog": self.catalog,
            "has_mesh": self.mesh_path is not None and self.mesh_path.exists(),
            "has_segmented": self.segmented_png is not None,
            "points": len(self.cloud),
        }

    def points_bytes(self) -> bytes:
        return np.ascontiguousarray(self.cloud.xyz, dtype="<f4").tobytes()

    def colors_bytes(self) -> bytes:
        return np.ascontiguousarray(self.cloud.rgb, dtype=np.uint8).tobytes()

    def segments_bytes(self) -> bytes:
        return np.ascontiguousarray(self.segments, dtype=np.uint8).tobytes()

    def labels_bytes(self) -> bytes:
        return np.ascontiguousarray(self.labels, dtype="<i4").tobytes()


def _thin(cloud: PointCloud, extra: list[NDArray[Any]]) -> tuple[PointCloud, list[NDArray[Any]]]:
    """Display-only thinning to MAX_DISPLAY_POINTS (deterministic)."""
    n = len(cloud)
    if n <= MAX_DISPLAY_POINTS:
        return cloud, extra
    idx = np.sort(np.random.default_rng(0).choice(n, MAX_DISPLAY_POINTS, replace=False))
    return cloud.subset(idx), [e[idx] for e in extra]


def _upright_transform(up_cam: NDArray[Any]) -> NDArray[np.float64]:
    """Camera frame → display frame with z up and the camera looking along +y."""
    R = rotation_between(up_cam, np.array([0.0, 0.0, 1.0]))
    fwd = R @ np.array([0.0, 0.0, 1.0])
    yaw = np.arctan2(fwd[0], fwd[1])  # rotate the projected forward direction onto +y
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = Rz @ R
    return T


def image_bundle(image: Path, client: Any = None) -> ViewBundle:
    from oh_my_slam.segmentation.api import reconstruct_and_detect, segment_frame
    from oh_my_slam.segmentation.catalog import catalog_rows
    from oh_my_slam.segmentation.render import segmented_image
    from oh_my_slam.segmentation.scene import single_image_scene

    if client is None:
        from oh_my_slam.reconstruction.api import connect_server

        client = connect_server()
    frame, dets = reconstruct_and_detect(Path(image), client)
    seg = segment_frame(frame, client=client, detections=dets)
    scene = single_image_scene(seg, tool="view")
    segcloud = seg.segments_cloud()
    cloud, _ = frame.camera_cloud()
    assert segcloud.label is not None
    cloud, (segs, labels) = _thin(cloud, [segcloud.rgb, segcloud.label])
    up = frame.gravity.up_cam if frame.gravity is not None else np.array([0.0, -1.0, 0.0])
    return ViewBundle(
        mode="image",
        title=Path(image).name,
        scene=scene,
        cloud=cloud,
        segments=segs,
        labels=labels,
        catalog=catalog_rows(seg.objects),
        segmented_png=png_bytes(segmented_image(frame.rgb, seg.label_map)),
        display_transform=_upright_transform(up).tolist(),
        stats={"points": len(cloud), "objects": len(seg.objects), "frames": 1, "triangles": 0},
    )


def map_bundle(map_dir: Path) -> ViewBundle:
    import json

    from oh_my_slam.mapping import store
    from oh_my_slam.mapping.export import map_cloud, map_objects, scene_bytes
    from oh_my_slam.segmentation.api import colorize_cloud
    from oh_my_slam.segmentation.catalog import catalog_rows

    reader = store.MapReader(Path(map_dir))
    scene = json.loads(scene_bytes(reader))
    _, objs = map_objects(reader)
    cloud = map_cloud(reader)
    seg = colorize_cloud(cloud, {o.id for o in objs})
    assert seg.label is not None
    disp, (segs, labels) = _thin(PointCloud(cloud.xyz, cloud.rgb), [seg.rgb, seg.label])
    frustums = [
        {
            "name": f.name,
            "T": f.T_map_cam.matrix().tolist(),
            "K": [f.K.fx, f.K.fy, f.K.cx, f.K.cy],
            "size": [f.width, f.height],
            "update": f.update_id,
        }
        for f in reader.frames
    ]
    mesh = reader.path(store.MESH_GLB)
    triangles = 0
    if mesh.exists():
        try:
            import trimesh

            sc = trimesh.load(str(mesh), force="scene")
            triangles = int(sum(len(g.faces) for g in sc.geometry.values()))
        except Exception:
            triangles = 0
    return ViewBundle(
        mode="map",
        title=reader.root.name,
        scene=scene,
        cloud=disp,
        segments=segs,
        labels=labels,
        catalog=catalog_rows(objs),
        frustums=frustums,
        mesh_path=mesh if mesh.exists() else None,
        stats={"points": len(disp), "objects": len(objs), "frames": len(reader.frames),
               "triangles": triangles, "map_points": len(cloud)},
    )
