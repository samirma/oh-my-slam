"""Builders for ASAM OpenLABEL 1.0.0 scene descriptions.

The schema URL goes in ``metadata.schema_url`` (a root ``$schema`` key fails validation because
the root only allows ``openlabel``). Cuboids use the 10-value form
``(x, y, z, qx, qy, qz, qw, sx, sy, sz)`` with a scalar-last quaternion.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.geometry import rot_to_quat
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.version import __version__

SCHEMA_VERSION = "1.0.0"
SCHEMA_URL = "https://openlabel.asam.net/V1-0-0/schema/openlabel_json_schema.json"
LVIS_ONTOLOGY_URI = "https://www.lvisdataset.org/"
ONTOLOGY_UID = "0"

Json = dict[str, Any]


def _r(x: float, nd: int = 6) -> float:
    return float(round(float(x), nd))


def metadata(name: str, tagged_file: str | None = None, **extra: Any) -> Json:
    md: Json = {
        "schema_version": SCHEMA_VERSION,
        "schema_url": SCHEMA_URL,
        "name": name,
        "annotator": f"oh-my-slam {__version__}",
    }
    if tagged_file is not None:
        md["tagged_file"] = tagged_file
    md.update(extra)
    return md


def ontology(labels: Sequence[str]) -> Json:
    return {
        ONTOLOGY_UID: {
            "uri": LVIS_ONTOLOGY_URI,
            "boundary_list": sorted(set(labels)),
            "boundary_mode": "include",
        }
    }


def camera_matrix_3x4(K: NDArray[Any]) -> list[float]:
    P = np.zeros((3, 4))
    P[:, :3] = K
    return [_r(v) for v in P.ravel()]


def camera_stream(intr: Intrinsics, uri: str | None = None, description: str | None = None) -> Json:
    stream: Json = {
        "type": "camera",
        "stream_properties": {
            "intrinsics_pinhole": {
                "width_px": int(intr.width),
                "height_px": int(intr.height),
                "camera_matrix": camera_matrix_3x4(intr.K()),
                "distortion_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
            },
            "intrinsics_source": intr.source,
        },
    }
    if uri is not None:
        stream["uri"] = uri
    if description is not None:
        stream["description"] = description
    return stream


def sensor_cs(parent: str = "", pose: Pose | None = None) -> Json:
    cs: Json = {"type": "sensor_cs", "parent": parent, "children": []}
    if pose is not None:
        cs["pose_wrt_parent"] = transform_data(pose)
    return cs


def map_cs(children: Sequence[str]) -> Json:
    return {
        "type": "scene_cs",
        "parent": "",
        "children": list(children),
        "axes": "x-forward,y-left,z-up",
        "gravity_aligned": True,
        "units": "m",
    }


def transform_data(pose: Pose) -> Json:
    return {
        "quaternion": [_r(v, 8) for v in rot_to_quat(pose.R)],
        "translation": [_r(v) for v in pose.t],
    }


def transform(src: str, dst: str, pose_src_to_dst: Pose) -> Json:
    return {"src": src, "dst": dst, "transform_src_to_dst": transform_data(pose_src_to_dst)}


def frame(
    timestamp: float | str | None = None,
    stream_uris: dict[str, str] | None = None,
    transforms: dict[str, Json] | None = None,
    **extra: Any,
) -> Json:
    props: Json = {}
    if timestamp is not None:
        props["timestamp"] = timestamp
    if stream_uris:
        props["streams"] = {k: {"uri": v} for k, v in stream_uris.items()}
    if transforms:
        props["transforms"] = transforms
    props.update(extra)
    return {"frame_properties": props}


def frame_intervals(frame_ids: Sequence[int]) -> list[Json]:
    """Closed intervals covering the sorted frame ids."""
    ids = sorted(set(int(i) for i in frame_ids))
    out: list[Json] = []
    for i in ids:
        if out and i == out[-1]["frame_end"] + 1:
            out[-1]["frame_end"] = i
        else:
            out.append({"frame_start": i, "frame_end": i})
    return out


def cuboid_val(center: NDArray[Any], R: NDArray[Any], size: NDArray[Any]) -> list[float]:
    q = rot_to_quat(R)
    return (
        [_r(v) for v in center]
        + [_r(v, 8) for v in q]
        + [_r(max(float(v), 0.0)) for v in size]
    )


def num(name: str, val: float) -> Json:
    return {"name": name, "val": _r(val)}


def text(name: str, val: str) -> Json:
    return {"name": name, "val": val}


def vec(name: str, val: Sequence[float | int]) -> Json:
    return {"name": name, "val": list(val)}


def boolean(name: str, val: bool) -> Json:
    return {"name": name, "val": bool(val)}


def object_entry(
    name: str,
    type_: str,
    coordinate_system: str,
    cuboid: Json | None,
    nums: Sequence[Json] = (),
    texts: Sequence[Json] = (),
    vecs: Sequence[Json] = (),
    booleans: Sequence[Json] = (),
    frame_ids: Sequence[int] | None = None,
) -> Json:
    data: Json = {}
    if cuboid is not None:
        data["cuboid"] = [cuboid]
    if nums:
        data["num"] = list(nums)
    if texts:
        data["text"] = list(texts)
    if vecs:
        data["vec"] = list(vecs)
    if booleans:
        data["boolean"] = list(booleans)
    obj: Json = {
        "name": name,
        "type": type_,
        "ontology_uid": ONTOLOGY_UID,
        "coordinate_system": coordinate_system,
        "object_data": data,
    }
    if frame_ids:
        obj["frame_intervals"] = frame_intervals(frame_ids)
    return obj


def cuboid(
    center: NDArray[Any],
    R: NDArray[Any],
    size: NDArray[Any],
    coordinate_system: str,
    attributes_num: Sequence[Json] = (),
    name: str = "obb",
) -> Json:
    c: Json = {
        "name": name,
        "val": cuboid_val(center, R, size),
        "coordinate_system": coordinate_system,
    }
    if attributes_num:
        c["attributes"] = {"num": list(attributes_num)}
    return c


def document(
    md: Json,
    objects: dict[str, Json],
    coordinate_systems: dict[str, Json] | None = None,
    streams: dict[str, Json] | None = None,
    frames: dict[str, Json] | None = None,
    labels_for_ontology: Sequence[str] | None = None,
) -> Json:
    root: Json = {"metadata": md}
    if labels_for_ontology is not None:
        root["ontologies"] = ontology(labels_for_ontology)
    if coordinate_systems:
        root["coordinate_systems"] = coordinate_systems
    if streams:
        root["streams"] = streams
    if frames is not None:
        root["frames"] = frames
        root["frame_intervals"] = frame_intervals([int(k) for k in frames])
    root["objects"] = objects
    return {"openlabel": root}
