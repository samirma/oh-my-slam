from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from oh_my_slam.core.geometry import rot_z
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.schema import openlabel as ol
from oh_my_slam.schema.validate import (
    SCHEMA_SHA256,
    extra_errors,
    schema_bytes,
    schema_errors,
    validation_errors,
)


def single_image_example() -> dict:
    intr = Intrinsics(2658.0, 2658.0, 2000.0, 1500.0, 4000, 3000, "exif")
    cub = ol.cuboid(
        np.array([0.5, 0.2, 4.0]),
        np.eye(3),
        np.array([0.5, 0.45, 0.9]),
        "camera",
        [ol.num("width_m", 0.5), ol.num("depth_m", 0.45), ol.num("height_m", 0.9),
         ol.num("volume_m3", 0.2025)],
    )
    obj = ol.object_entry(
        "chair 1", "chair", "camera", cub,
        nums=[ol.num("score", 0.91), ol.num("pixel_count", 1200), ol.num("point_count", 800)],
        texts=[ol.text("color_hex", "#e6194b")],
        vecs=[ol.vec("color", [230, 25, 75])],
        booleans=[ol.boolean("confirmed", True)],
    )
    return ol.document(
        ol.metadata("restaurant", tagged_file="restaurant.jpg", intrinsics_source="exif"),
        {"1": obj},
        coordinate_systems={"camera": ol.sensor_cs()},
        streams={"camera": ol.camera_stream(intr, uri="restaurant.jpg")},
        frames={"0": ol.frame(timestamp=0.0, stream_uris={"camera": "restaurant.jpg"})},
        labels_for_ontology=["chair"],
    )


def map_example() -> dict:
    intr = Intrinsics(500.0, 500.0, 320.0, 240.0, 640, 480, "colmap")
    poses = [Pose(rot_z(0.1 * k), np.array([0.1 * k, 0.0, 1.2])) for k in range(3)]
    frames = {
        str(k): ol.frame(
            timestamp=0.5 * k,
            stream_uris={"camera_0": f"frames/f{k:06d}.jpg"},
            transforms={"camera_0_to_map": ol.transform("camera_0", "map", p)},
        )
        for k, p in enumerate(poses)
    }
    cub = ol.cuboid(np.array([1.0, 0.0, 0.4]), rot_z(0.3), np.array([1.8, 0.9, 0.8]), "map")
    obj = ol.object_entry("sofa 7", "sofa", "map", cub, nums=[ol.num("score", 0.8)],
                          frame_ids=[0, 1, 2])
    return ol.document(
        ol.metadata("map", keyframes=3),
        {"7": obj},
        coordinate_systems={"map": ol.map_cs(["camera_0"]), "camera_0": ol.sensor_cs("map")},
        streams={"camera_0": ol.camera_stream(intr)},
        frames=frames,
        labels_for_ontology=["sofa"],
    )


def test_vendored_schema_is_pinned() -> None:
    assert hashlib.sha256(schema_bytes()).hexdigest() == SCHEMA_SHA256
    schema = json.loads(schema_bytes())
    assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"


@pytest.mark.parametrize("make", [single_image_example, map_example])
def test_examples_validate(make) -> None:  # type: ignore[no-untyped-def]
    doc = make()
    assert validation_errors(doc) == []
    json.loads(json.dumps(doc))
    md = doc["openlabel"]["metadata"]
    assert md["schema_url"] == ol.SCHEMA_URL and md["schema_version"] == "1.0.0"


def test_root_schema_key_is_rejected() -> None:
    doc = single_image_example()
    doc["$schema"] = ol.SCHEMA_URL
    assert schema_errors(doc)


def test_cuboid_is_ten_values_scalar_last() -> None:
    val = ol.cuboid_val(np.zeros(3), rot_z(np.pi / 2), np.ones(3))
    assert len(val) == 10
    np.testing.assert_allclose(val[3:7], [0, 0, np.sqrt(0.5), np.sqrt(0.5)], atol=1e-6)


def test_extra_checks_catch_what_the_schema_misses() -> None:
    doc = single_image_example()
    bad = copy.deepcopy(doc)
    pin = bad["openlabel"]["streams"]["camera"]["stream_properties"]["intrinsics_pinhole"]
    pin["camera_matrix"] = [1, 2, 3]
    assert schema_errors(bad) == []  # the schema does not validate intrinsics
    assert any("camera_matrix" in e for e in extra_errors(bad))

    bad = copy.deepcopy(doc)
    bad["openlabel"]["objects"]["1"]["object_data"]["cuboid"][0]["val"][3:7] = [0, 0, 0, 2]
    assert any("quaternion norm" in e for e in extra_errors(bad))

    bad = copy.deepcopy(doc)
    bad["openlabel"]["objects"]["1"]["object_data"]["cuboid"][0]["val"] = [0] * 9
    assert any("10 numbers" in e for e in extra_errors(bad))

    bad = copy.deepcopy(doc)
    bad["openlabel"]["frame_intervals"] = [{"frame_start": 3, "frame_end": 1}]
    assert any("frame_intervals" in e for e in extra_errors(bad))

    bad = copy.deepcopy(doc)
    bad["openlabel"]["frame_intervals"] = [{"frame_start": 0, "frame_end": 4}]
    assert any("do not match" in e for e in extra_errors(bad))

    bad = copy.deepcopy(doc)
    del bad["openlabel"]["frame_intervals"]
    assert any("without frame_intervals" in e for e in extra_errors(bad))

    bad = copy.deepcopy(doc)
    bad["openlabel"]["metadata"]["schema_url"] = "https://example.com"
    assert extra_errors(bad)

    bad = copy.deepcopy(doc)
    bad["openlabel"]["objects"]["1"]["coordinate_system"] = "nowhere"
    assert any("unknown coordinate system" in e for e in extra_errors(bad))

    bad = copy.deepcopy(doc)
    bad["openlabel"]["objects"]["1"]["object_data"]["cuboid"][0]["val"][7] = -1
    assert any("negative size" in e for e in extra_errors(bad))

    bad = copy.deepcopy(doc)
    del bad["openlabel"]["streams"]["camera"]["stream_properties"]["intrinsics_pinhole"]
    assert any("without intrinsics" in e for e in extra_errors(bad))
    assert any("without intrinsics" in e for e in validation_errors(bad))  # schema + extra checks


def test_map_transform_checks() -> None:
    doc = map_example()
    bad = copy.deepcopy(doc)
    tr = bad["openlabel"]["frames"]["1"]["frame_properties"]["transforms"]["camera_0_to_map"]
    tr["transform_src_to_dst"]["quaternion"] = [0, 0, 0, 0.5]
    assert any("quaternion norm" in e for e in extra_errors(bad))
    tr["dst"] = "elsewhere"
    assert any("unknown dst" in e for e in extra_errors(bad))


def test_frame_intervals_merge() -> None:
    assert ol.frame_intervals([3, 1, 2, 7, 8, 10]) == [
        {"frame_start": 1, "frame_end": 3},
        {"frame_start": 7, "frame_end": 8},
        {"frame_start": 10, "frame_end": 10},
    ]
    assert ol.frame_intervals([]) == []
