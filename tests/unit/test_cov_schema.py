"""OpenLABEL scene descriptions (spec §3) beyond the shapes the commands emit: optional parts of
the builders, and every check the schema itself does not make (stream intrinsics, top-level frame
intervals, quaternions), plus the pin on the vendored schema."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from oh_my_slam.core.geometry import rot_z
from oh_my_slam.core.types import Pose
from oh_my_slam.schema import openlabel as ol
from oh_my_slam.schema import validate
from tests.unit.test_openlabel import map_example, single_image_example


@pytest.fixture
def fresh_validator() -> Iterator[None]:
    validate._validator.cache_clear()
    try:
        yield
    finally:
        validate._validator.cache_clear()


def test_a_changed_vendored_schema_is_refused(fresh_validator: None,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    pinned = validate.schema_bytes()
    monkeypatch.setattr(validate, "schema_bytes", lambda: pinned + b"\n")
    with pytest.raises(RuntimeError, match="vendored OpenLABEL schema changed: sha256 "):
        validate.validation_errors(single_image_example())


def test_a_sensor_pose_and_sparse_objects_are_valid() -> None:
    """A camera mounted on a parent frame (``pose_wrt_parent``), an object without a cuboid or
    numbers, and a document without coordinate systems."""
    mount = Pose(rot_z(0.25), np.array([0.1, -0.2, 1.5]))
    cs = ol.sensor_cs("base", mount)
    assert cs["pose_wrt_parent"] == ol.transform_data(mount)
    assert cs["pose_wrt_parent"]["translation"] == [0.1, -0.2, 1.5]
    obj = ol.object_entry("cup 1", "cup", "base", None, texts=[ol.text("color_hex", "#3cb44b")])
    assert obj["object_data"] == {"text": [{"name": "color_hex", "val": "#3cb44b"}]}
    doc = ol.document(ol.metadata("sparse"), {"1": obj},
                      coordinate_systems={"base": ol.map_cs(["camera"]), "camera": cs})
    assert validate.validation_errors(doc) == []
    bare = ol.document(ol.metadata("bare"), {"1": obj})
    assert "coordinate_systems" not in bare["openlabel"] and "frames" not in bare["openlabel"]
    assert validate.validation_errors(bare) == []  # no systems: nothing to resolve names against


def _pinhole(doc: dict[str, Any]) -> dict[str, Any]:
    return doc["openlabel"]["streams"]["camera"]["stream_properties"]["intrinsics_pinhole"]


@pytest.mark.parametrize(("field", "value", "message"), [
    ("width_px", 0, "width_px/height_px must be positive integers"),
    ("height_px", 480.0, "width_px/height_px must be positive integers"),
    ("camera_matrix", [-500.0, 0, 320, 0, 0, 500, 240, 0, 0, 0, 1, 0],
     "focal lengths must be positive"),
    ("camera_matrix", [500.0, 0, 320, 0, 0, 500, 240, 0, 0, 0, 2, 0],
     "camera_matrix last row must be [0, 0, 1, 0]"),
    ("distortion_coeffs", [0.0, 0.0, 0.0], "distortion_coeffs must be a list of numbers"),
    ("distortion_coeffs", [0.0, 0.0, 0.0, "x"], "distortion_coeffs must be a list of numbers"),
])
def test_stream_intrinsics_the_schema_does_not_check(field: str, value: Any, message: str) -> None:
    doc = single_image_example()
    _pinhole(doc)[field] = value
    assert validate.schema_errors(doc) == []
    assert validate.extra_errors(doc) == [f"streams/camera: {message}"]


def test_well_formed_intrinsics_variants_are_accepted() -> None:
    doc = single_image_example()
    pin = _pinhole(doc)
    pin["camera_matrix"] = [500.0, 0.0, 320.0, 0.0, 0.0, 500.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    pin["distortion_coeffs"] = [0.1, -0.05, 0.0, 0.0]
    assert validate.validation_errors(doc) == []
    del pin["distortion_coeffs"]
    assert validate.validation_errors(doc) == []


def test_a_stream_that_is_not_a_camera_needs_no_intrinsics() -> None:
    doc = single_image_example()
    doc["openlabel"]["streams"]["imu"] = {"type": "other", "description": "not a camera"}
    assert validate.validation_errors(doc) == []


def test_a_transform_quaternion_must_be_four_numbers() -> None:
    doc = map_example()
    tr = doc["openlabel"]["frames"]["2"]["frame_properties"]["transforms"]["camera_0_to_map"]
    for bad in ([0.0, 0.0, 1.0], [0.0, 0.0, 0.0, "1"], [0.0, 0.0, 0.0, float("nan")]):
        tr["transform_src_to_dst"]["quaternion"] = bad
        assert "frames/2/transforms/camera_0_to_map: quaternion must be 4 numbers" in \
            validate.extra_errors(doc)


def test_top_level_frame_intervals_must_be_a_list() -> None:
    doc = copy.deepcopy(map_example())
    doc["openlabel"]["frame_intervals"] = {"frame_start": 0, "frame_end": 2}
    assert "frame_intervals must be a list" in validate.extra_errors(doc)


def test_metadata_must_name_the_schema_version() -> None:
    doc = single_image_example()
    doc["openlabel"]["metadata"]["schema_version"] = "1.0.1"
    assert validate.extra_errors(doc) == ["metadata.schema_version must be 1.0.0"]
    assert validate.extra_errors({"not": "openlabel"})[:2] == [
        "metadata.schema_version must be 1.0.0",
        "metadata.schema_url must be the canonical OpenLABEL schema URL"]
