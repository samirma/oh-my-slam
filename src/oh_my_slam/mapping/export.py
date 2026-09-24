"""Map scene export (OpenLABEL, map frame) for ``-t full`` / ``-t single``, the PLY payloads, and
the read-only inputs for ``segment.sh -m``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from oh_my_slam.core.images import load_rgb
from oh_my_slam.core.log import json_payload_bytes
from oh_my_slam.core.ply import PointCloud, ply_bytes, read_ply
from oh_my_slam.mapping import store
from oh_my_slam.mapping.objects import ObjectState, label_map_for, load_state
from oh_my_slam.schema import openlabel as ol
from oh_my_slam.segmentation.api import KeyframeLabels, SceneObject, export_map
from oh_my_slam.segmentation.scene import objects_block, ontology_labels

Json = dict[str, Any]


def camera_key(camera_id: int) -> str:
    return f"camera_{camera_id}"


def full_scene(root: Path, meta: dict[str, Any], records: list[store.FrameRecord],
               objects: list[SceneObject], current: Any = None,
               frame_filter: set[str] | None = None) -> Json:
    """OpenLABEL document of the map (or of the frames in ``frame_filter``)."""
    recs = [r for r in sorted(records, key=lambda r: r.index)
            if frame_filter is None or r.name in frame_filter]
    cams: dict[int, store.FrameRecord] = {}
    for r in sorted(records, key=lambda r: r.index):
        cams.setdefault(r.camera_id, r)
    used = sorted({r.camera_id for r in recs} or set(cams))
    css: dict[str, Json] = {"map": ol.map_cs([camera_key(c) for c in used])}
    streams: dict[str, Json] = {}
    for c in used:
        css[camera_key(c)] = ol.sensor_cs("map")
        streams[camera_key(c)] = ol.camera_stream(cams[c].K, description=f"camera {c}")
    frames = {}
    for r in recs:
        key = camera_key(r.camera_id)
        frames[str(r.index)] = ol.frame(
            timestamp=float(r.index),
            stream_uris={key: r.image},
            transforms={f"{key}_to_map": ol.transform(key, "map", r.T_map_cam)},
            keyframe=r.name, pose_source=r.pose_source, update_id=r.update_id,
            low_confidence=r.low_confidence,
        )
    md = ol.metadata(
        Path(root).name, tagged_file=str(root), tool="mapper",
        map_frame=meta.get("map_frame"), scale=meta.get("scale"),
        update_count=meta.get("update_count"), keyframes=len(records),
        floor_z=meta.get("floor_z"),
    )
    return ol.document(md, objects_block(objects, "map"), coordinate_systems=css,
                       streams=streams, frames=frames,
                       labels_for_ontology=ontology_labels(objects))


def scene_payload(scene_full: Json, mode: str, new_names: list[str],
                  records: list[store.FrameRecord], objs: ObjectState) -> bytes:
    if mode == "full":
        return json_payload_bytes(scene_full)
    new = set(new_names)
    keep_objs = [o for o in objs.exported() if o.id in objs.observed]
    doc = json.loads(json.dumps(scene_full))
    root = doc["openlabel"]
    new_idx = {str(r.index) for r in records if r.name in new}
    root["frames"] = {k: v for k, v in root["frames"].items() if k in new_idx}
    root["frame_intervals"] = ol.frame_intervals([int(k) for k in root["frames"]])
    root["objects"] = objects_block(keep_objs, "map")
    root["metadata"]["scope"] = "single"
    return json_payload_bytes(doc)


def ply_payload(geo: Any, mode: str, new_names: set[str]) -> bytes:
    cloud = geo.cloud if mode == "full" else geo.new_cloud
    return ply_bytes(PointCloud(cloud.xyz, cloud.rgb), comment=f"oh-my-slam map ({mode})")


# ------------------------------------------------------------------------------------------------
# read-only map access (segment -m, view -m)


def map_objects(reader: store.MapReader) -> tuple[ObjectState, list[SceneObject]]:
    state = load_state(reader.path, reader.meta)
    return state, state.exported()


def map_cloud(reader: store.MapReader) -> PointCloud:
    if not reader.exists(store.CLOUD_PLY):
        return PointCloud(np.zeros((0, 3)), np.zeros((0, 3)))
    cloud = read_ply(reader.path(store.CLOUD_PLY))
    lp = reader.path(store.CLOUD_OBJECTS)
    labels = np.load(lp).astype(np.int32) if lp.exists() else np.zeros(len(cloud), np.int32)
    return PointCloud(cloud.xyz, cloud.rgb, labels)


def keyframe_labels(reader: store.MapReader, state: ObjectState) -> list[KeyframeLabels]:
    out = []
    for r in reader.frames:
        rgb = load_rgb(reader.image_path(r), max_side=max(r.grid_width, r.grid_height))
        lab = label_map_for(reader.instances(r), rgb.shape[:2], state)
        out.append(KeyframeLabels(r.name, rgb, lab))
    return out


def scene_bytes(reader: store.MapReader) -> bytes:
    if reader.exists(store.SCENE_JSON):
        return json_payload_bytes(reader.read_json(store.SCENE_JSON))
    _, objs = map_objects(reader)
    return json_payload_bytes(full_scene(reader.root, reader.meta, reader.frames, objs))


def map_segment_outputs(map_dir: Path, out_dir: Path | None) -> tuple[bytes, bytes]:
    """``segment.sh -m``: (scene JSON bytes, segments PLY bytes); artefacts when ``out_dir``."""
    from oh_my_slam.segmentation.artifacts import write_artifacts

    reader = store.MapReader(map_dir)
    state, objs = map_objects(reader)
    scene = scene_bytes(reader)
    need_sheet = out_dir is not None
    kfs = keyframe_labels(reader, state) if need_sheet else []
    seg = export_map(objs, kfs, map_cloud(reader))
    ply = ply_bytes(seg.segments, comment="oh-my-slam map segments")
    if out_dir is not None:
        write_artifacts(out_dir, scene, seg.segmented, objs, seg.segments,
                        title=f"Objects in map {reader.root.name}")
    return scene, ply
