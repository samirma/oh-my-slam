"""Map scene export (OpenLABEL, map frame) for ``-t full`` / ``-t single``, the PLY payloads
(derived with the point-cloud attributes by ``segmentation.cloud``), and the read-only access of
``mapper.sh locate`` and ``view.sh -m`` to a persisted map."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from oh_my_slam.core.cloud_attrs import CloudAttrs
from oh_my_slam.core.log import json_payload_bytes
from oh_my_slam.core.ply import PointCloud, read_ply
from oh_my_slam.mapping import store
from oh_my_slam.mapping.objects import ObjectState, load_state
from oh_my_slam.schema import openlabel as ol
from oh_my_slam.segmentation.api import SceneObject
from oh_my_slam.segmentation.cloud import MapCloudSource, cloud_ply, map_cloud_source
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
        map_frame=ol.rounded(meta.get("map_frame")), scale=ol.rounded(meta.get("scale")),
        update_count=meta.get("update_count"), keyframes=len(records),
        floor_z=ol.rounded(meta.get("floor_z")),
    )
    return ol.document(md, objects_block(objects, "map"), coordinate_systems=css,
                       streams=streams, frames=frames,
                       labels_for_ontology=ontology_labels(objects))


def payload_objects(objs: ObjectState, mode: str) -> list[SceneObject]:
    """Objects a ``-t full`` (all) or ``-t single`` (observed by the new input) payload covers."""
    exported = objs.exported()
    return exported if mode == "full" else [o for o in exported if o.id in objs.observed]


def scene_payload(scene_full: Json, mode: str, new_names: list[str],
                  records: list[store.FrameRecord], objs: ObjectState) -> bytes:
    if mode == "full":
        return json_payload_bytes(scene_full)
    new = set(new_names)
    keep_objs = payload_objects(objs, mode)
    doc = json.loads(json.dumps(scene_full))
    root = doc["openlabel"]
    new_idx = {str(r.index) for r in records if r.name in new}
    root["frames"] = {k: v for k, v in root["frames"].items() if k in new_idx}
    root["frame_intervals"] = ol.frame_intervals([int(k) for k in root["frames"]])
    root["objects"] = objects_block(keep_objs, "map")
    root["metadata"]["scope"] = "single"
    return json_payload_bytes(doc)


def map_source(cloud: PointCloud, objects: list[SceneObject], records: list[store.FrameRecord]
               ) -> MapCloudSource:
    """Cloud source of a map: point labels restricted to ``objects``, normals oriented towards
    the keyframe camera centres."""
    rgb = cloud.rgb if cloud.rgb is not None else np.zeros((len(cloud), 3), np.uint8)
    return map_cloud_source(cloud.xyz, rgb, cloud.label, {o.id for o in objects},
                            np.array([r.T_map_cam.t for r in records]).reshape(-1, 3))


def ply_payload(geo: Any, mode: str, records: list[store.FrameRecord], objs: ObjectState,
                attrs: CloudAttrs) -> bytes:
    """``-f ply``: the whole map cloud (full) or the new keyframes' points (single)."""
    cloud = geo.cloud if mode == "full" else geo.new_cloud
    return cloud_ply(map_source(cloud, payload_objects(objs, mode), records), attrs)


# ------------------------------------------------------------------------------------------------
# read-only map access (mapper.sh locate, view.sh -m)


def map_objects(reader: store.MapReader) -> tuple[ObjectState, list[SceneObject]]:
    state = load_state(reader.path, reader.meta)
    return state, state.exported()


def map_cloud(reader: store.MapReader) -> PointCloud:
    """The stored map cloud with its object id per point."""
    if not reader.exists(store.CLOUD_PLY):
        return PointCloud(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))
    cloud = read_ply(reader.path(store.CLOUD_PLY))
    lp = reader.path(store.CLOUD_OBJECTS)
    labels = np.asarray(np.load(lp), np.int32) if lp.exists() else np.zeros(len(cloud), np.int32)
    return PointCloud(cloud.xyz, cloud.rgb, labels)


def reader_source(reader: store.MapReader, objects: list[SceneObject]) -> MapCloudSource:
    """Cloud source of a persisted map (``view.sh -m``)."""
    return map_source(map_cloud(reader), objects, reader.frames)


def scene_bytes(reader: store.MapReader) -> bytes:
    """The map's scene JSON (``mapper.sh locate -t full``, ``view.sh -m``), built from its
    persisted state by the code that builds ``mapper.sh update``'s result: the object colours are
    a pure function of the ids *now* (§2.4), so they agree with the clouds derived from the map
    even for a map written before a palette change."""
    _, objs = map_objects(reader)
    return json_payload_bytes(full_scene(reader.root, reader.meta, reader.frames, objs))
