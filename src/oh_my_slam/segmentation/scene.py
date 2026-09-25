"""Objects → OpenLABEL entries (colours, OBB cuboids, attributes), and the single-image scene."""

from __future__ import annotations

from typing import Any

from oh_my_slam.schema import openlabel as ol
from oh_my_slam.segmentation.api import FrameSegmentation, SceneObject
from oh_my_slam.segmentation.detect import default_vocabulary

Json = dict[str, Any]


def object_entry(obj: SceneObject, coordinate_system: str) -> Json:
    w, d, h = (float(v) for v in obj.obb.size)
    cub = ol.cuboid(
        obj.obb.center, obj.obb.R, obj.obb.size, coordinate_system,
        attributes_num=[ol.num("width_m", w), ol.num("depth_m", d), ol.num("height_m", h),
                        ol.num("volume_m3", w * d * h)],
    )
    return ol.object_entry(
        name=f"{obj.label} {obj.id}",
        type_=obj.label,
        coordinate_system=coordinate_system,
        cuboid=cub,
        nums=[ol.num("score", obj.score), ol.num("pixel_count", obj.pixel_count),
              ol.num("point_count", obj.point_count), ol.num("observations", obj.observations)],
        texts=[ol.text("color_hex", obj.color_hex)],
        vecs=[ol.vec("color", list(obj.color))]
        + ([ol.vec("detected_as", list(obj.labels))] if len(obj.labels) > 1 else []),
        booleans=[ol.boolean("confirmed", obj.confirmed)],
        frame_ids=obj.frames or None,
    )


def objects_block(objects: list[SceneObject], coordinate_system: str) -> dict[str, Json]:
    return {str(o.id): object_entry(o, coordinate_system) for o in sorted(objects,
                                                                          key=lambda o: o.id)}


def ontology_labels(objects: list[SceneObject]) -> list[str]:
    return sorted(set(default_vocabulary()) | {o.label for o in objects})


def single_image_scene(seg: FrameSegmentation, tool: str) -> Json:
    f = seg.frame
    name = f.image_path.name
    md_extra: Json = {
        "tool": tool,
        "intrinsics_source": f.intrinsics.source,
        "depth_grid": {"width": f.grid_size[0], "height": f.grid_size[1]},
    }
    if f.gravity is not None:
        md_extra["gravity"] = f.gravity.to_dict()
    return ol.document(
        ol.metadata(name, tagged_file=str(f.image_path), **md_extra),
        objects_block(seg.objects, "camera"),
        coordinate_systems={"camera": ol.sensor_cs()},
        streams={"camera": ol.camera_stream(f.intrinsics, uri=str(f.image_path))},
        frames={"0": ol.frame(timestamp=0.0, stream_uris={"camera": str(f.image_path)})},
        labels_for_ontology=ontology_labels(seg.objects),
    )
