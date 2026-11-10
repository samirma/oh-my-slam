"""Read-only access to a persisted map's objects and cloud (the bird's-eye renderer, the
service's later reads)."""

from __future__ import annotations

import numpy as np

from oh_my_slam.core.ply import PointCloud, read_ply
from oh_my_slam.mapping import store
from oh_my_slam.mapping.objects import ObjectState, load_state
from oh_my_slam.reconstruction.cloud import MapCloudSource
from oh_my_slam.segmentation.api import SceneObject, map_cloud_source


def map_source(cloud: PointCloud, objects: list[SceneObject], records: list[store.FrameRecord]
               ) -> MapCloudSource:
    """Cloud source of a map: point labels restricted to ``objects``, normals oriented towards
    the keyframe camera centres."""
    rgb = cloud.rgb if cloud.rgb is not None else np.zeros((len(cloud), 3), np.uint8)
    return map_cloud_source(cloud.xyz, rgb, cloud.label, {o.id for o in objects},
                            np.array([r.T_map_cam.t for r in records]).reshape(-1, 3))


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
