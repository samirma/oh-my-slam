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
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.schema import openlabel as ol
from oh_my_slam.schema import validate
from tests.unit.test_openlabel import map_example, single_image_example

LENS = Intrinsics(1392.0, 1392.0, 960.0, 444.0, 1920, 888, "colmap", -0.524)


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


_DIST = "distortion_coeffs must be a list of 5, 8, 12 or 14 numbers"


@pytest.mark.parametrize(("field", "value", "message"), [
    ("width_px", 0, "width_px/height_px must be positive integers"),
    ("height_px", 480.0, "width_px/height_px must be positive integers"),
    ("camera_matrix", [-500.0, 0, 320, 0, 0, 500, 240, 0, 0, 0, 1, 0],
     "focal lengths must be positive"),
    ("camera_matrix", [500.0, 0, 320, 0, 0, 500, 240, 0, 0, 0, 2, 0],
     "camera_matrix last row must be [0, 0, 1, 0]"),
    ("distortion_coeffs", [0.0, 0.0, 0.0], _DIST),
    ("distortion_coeffs", [0.1, -0.05, 0.0, 0.0], _DIST),  # the schema's minItems is 5
    ("distortion_coeffs", [0.0] * 6, _DIST),  # no OpenCV model has 6
    ("distortion_coeffs", [0.0, 0.0, 0.0, 0.0, "x"], _DIST),
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
    for n in (5, 8, 12, 14):
        pin["distortion_coeffs"] = [0.1, -0.05] + [0.0] * (n - 2)
        assert validate.validation_errors(doc) == []
    del pin["distortion_coeffs"]
    assert validate.validation_errors(doc) == []


def _lens_doc() -> dict[str, Any]:
    """A document whose camera stream is a lens with distortion: OpenCV's 8 rational
    coefficients and the division model as ``intrinsics_custom``."""
    doc = single_image_example()
    doc["openlabel"]["streams"]["camera"] = ol.camera_stream(LENS, uri="img_008_p03_mid.jpg")
    return doc


def test_a_lens_stream_with_its_custom_model_is_valid() -> None:
    doc = _lens_doc()
    props = doc["openlabel"]["streams"]["camera"]["stream_properties"]
    assert len(props["intrinsics_pinhole"]["distortion_coeffs"]) == 8
    assert props["intrinsics_custom"]["model"] == "division"
    assert validate.validation_errors(doc) == []
    assert ol.stream_intrinsics(props) == LENS
    props["intrinsics_custom"] = {"model": "fisheye-of-another-tool", "anything": [1, 2]}
    assert validate.validation_errors(doc) == []  # another model: the stream defines it freely


@pytest.mark.parametrize(("change", "message"), [
    (lambda c, p: 5, "intrinsics_custom must be an object"),
    (lambda c, p: {**c, "k": "oops"}, "intrinsics_custom: the division model needs a finite "
                                      "number k"),
    (lambda c, p: {**c, "k": float("inf")}, "intrinsics_custom: the division model needs a "
                                            "finite number k"),
    (lambda c, p: {**c, "focal_length_px": 1000.0},
     "intrinsics_custom: focal_length_px must be the camera_matrix's (1392)"),
    (lambda c, p: {k: v for k, v in c.items() if k != "center_x_px"},
     "intrinsics_custom: center_x_px must be the camera_matrix's (960)"),
    (lambda c, p: {**c, "center_y_px": True},
     "intrinsics_custom: center_y_px must be the camera_matrix's (444)"),
    (lambda c, p: p.update(distortion_coeffs=[0.0] * 8) or c,
     "intrinsics_custom: a division model with k != 0 needs the OpenCV distortion_coeffs that "
     "approximate it"),
    (lambda c, p: p.pop("distortion_coeffs") and c,
     "intrinsics_custom: a division model with k != 0 needs the OpenCV distortion_coeffs that "
     "approximate it"),
])
def test_a_malformed_custom_lens_is_refused(change: Any, message: str) -> None:
    doc = _lens_doc()
    props = doc["openlabel"]["streams"]["camera"]["stream_properties"]
    props["intrinsics_custom"] = change(props["intrinsics_custom"], props["intrinsics_pinhole"])
    assert validate.extra_errors(doc) == [f"streams/camera: {message}"]


def test_a_custom_model_is_checked_without_a_pinhole_too() -> None:
    """A division model with k = 0 needs no coefficients; without a pinhole there is no matrix
    to compare it with, and a camera stream still needs one."""
    doc = _lens_doc()
    props = doc["openlabel"]["streams"]["camera"]["stream_properties"]
    props["intrinsics_custom"]["k"] = 0.0
    props["intrinsics_pinhole"]["distortion_coeffs"] = [0.0] * 5
    assert validate.validation_errors(doc) == []
    del props["intrinsics_pinhole"]
    assert validate.extra_errors(doc) == ["streams/camera: camera stream without intrinsics_pinhole"]
    doc["openlabel"]["streams"]["camera"]["type"] = "other"
    props["intrinsics_custom"] = []
    assert validate.extra_errors(doc) == ["streams/camera: intrinsics_custom must be an object"]


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


def test_a_transform_gives_its_pose_back() -> None:
    """``transform_pose`` reads what ``transform_data`` writes (quaternion and translation), and
    the 4 x 4 ``matrix4x4`` form the schema also allows: the one reader of the viewer and the
    evaluator."""
    pose = Pose(rot_z(0.7), np.array([1.0, -2.0, 0.5]))
    back = ol.transform_pose(ol.transform_data(pose))
    np.testing.assert_allclose(back.matrix(), pose.matrix(), atol=1e-7)
    matrix = {"matrix4x4": pose.matrix().reshape(-1).tolist()}
    np.testing.assert_array_equal(ol.transform_pose(matrix).matrix(), pose.matrix())
