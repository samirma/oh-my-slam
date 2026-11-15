"""Frame analysis (api.md §7, oms/2.10): the sent pose's up axis defines the horizontal, a
monocular floor tilted from it is turned onto it, ``scale_check.floor_tilt_deg``, and the range
scan's unsupported beams. Through the Starlette app with ``tests.fakes.service.FakeBackend`` (the
fixtures of ``test_service_analyse``), on a humanoid's view: its head camera 0.33 m above the
floor, pitched 20 degrees down, a wall 2.5 m ahead. Every body is checked against
api.schema.json."""

from __future__ import annotations

import io
import math
from typing import Any

import numpy as np
import pytest
from PIL import Image

from oh_my_slam.core.types import Intrinsics
from oh_my_slam.mapping import analysis
from oh_my_slam.reconstruction.pointcloud import supported_pixels
from tests.unit.test_service_analyse import An, _knobs, an, decode, png

__all__ = ["_knobs", "an"]  # the fixtures

HUMANOID = {"fx": 270.0, "fy": 270.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480}
HEAD_M = 0.33
WALL_M = 2.5  # map x of the wall; the camera at the origin looks along +x


def _tilted(tilt_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """The pose the humanoid sends, and the depth of its view from a camera pitched ``tilt_deg``
    further down: a depth whose floor, lifted with the sent pose, tilts that much from its up
    axis (rising away from the camera), as MoGe's does on the robot's head camera."""
    from tests.synth.rooms import Scene, camera

    scene = Scene().add((-5.0, -5.0, -0.1), (5.0, 5.0, 0.0)).add(
        (WALL_M, -5.0, 0.0), (WALL_M + 0.1, 5.0, 3.0))
    K = np.array([[270.0, 0.0, 320.0], [0.0, 270.0, 240.0], [0.0, 0.0, 1.0]])
    seen = scene.depth(camera(0.0, 0.0, HEAD_M, 0.0, -20.0 - tilt_deg), K, 640, 480)
    return camera(0.0, 0.0, HEAD_M, 0.0, -20.0), seen.astype(np.float32)


def jpeg(grey: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((480, 640, 3), grey, np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


def to_wall(doc: dict[str, Any], half_deg: int = 40) -> dict[int, tuple[str, float, float]]:
    """The central beams: their kind, range and the wall's true range along them."""
    beams = {b["bearing_deg"]: b for b in doc["range_scan"]}
    return {k: (beams[float(k)]["kind"], beams[float(k)]["range_m"],
                WALL_M / math.cos(math.radians(k))) for k in range(-half_deg, half_deg + 1, 5)}


def test_a_monocular_floor_tilted_from_the_pose_is_turned_onto_its_up_axis(an: An) -> None:
    """A monocular depth whose floor tilts 7 degrees from the sent pose's up axis (beyond
    3 x 2 degrees) is turned about the camera before any use: the wall's ranges come back true,
    where the depth as measured puts a phantom obstacle on its rising floor at a fifth of the
    wall's range."""
    T, z = _tilted(7.0)
    an.backend.other_depth = z * 0.8  # the model's metres
    img = jpeg(70)
    req = {"image": "i", "camera": {**HUMANOID, "height_above_floor_m": HEAD_M},
           "T_map_cam": T.tolist(), "outputs": ["points", "range_scan", "depth"], "voxel_m": 0}
    doc = an.analyse(req, {"i": img})
    sc = doc["scale_check"]
    assert (doc["frame"], doc["scale_source"]) == ("client", "camera_height")
    assert sc["floor_tilt_deg"] == pytest.approx(7.0, abs=0.3)  # the raw depth's, from the pose
    assert sc["model_height_m"] == pytest.approx(0.8 * HEAD_M, abs=0.01)  # a turn keeps it
    for k, (kind, got, truth) in to_wall(doc).items():
        assert kind == "obstacle" and got == pytest.approx(truth, rel=0.03), k
    beams = {b["bearing_deg"]: b for b in doc["range_scan"]}
    assert all(beams[float(k)]["support"] > 10 * analysis.MIN_BEAM_SUPPORT for k in to_wall(doc))
    assert all(b["support"] >= analysis.MIN_BEAM_SUPPORT for b in doc["range_scan"]
               if b["kind"] != "unseen")
    # the points: the floor level at the pose's z 0, the wall upright at its range
    xyz = decode(doc["points"]["xyz"])
    floor, wall = xyz[xyz[:, 0] < WALL_M - 0.2], xyz[xyz[:, 2] > 0.1]
    assert np.percentile(np.abs(floor[:, 2]), 95) < 0.02
    assert np.percentile(np.abs(wall[:, 0] - WALL_M), 95) < 0.04
    # the depth output is the model's per pixel, at the height's scale
    assert np.abs(decode(doc["depth"]) * 0.001 - z).max() < 0.02
    # 3 x 5 degrees: the floor lies within it, the depth is used as measured, and its floor
    # rises into the obstacle band half a metre ahead
    before = an.analyse({**req, "sigma_tilt_deg": 5.0}, {"i": img})
    assert before["scale_check"] == sc
    for k, (kind, got, truth) in to_wall(before).items():
        assert kind == "obstacle" and got < truth / 4, k


def test_the_depth_is_used_as_measured_within_3_sigma_from_a_sensor_or_without_a_pose(
        an: An) -> None:
    """Within 3 x sigma_tilt_deg of the pose's up axis, from a depth sensor and in the
    gravity_camera frame, nothing is turned."""
    cam = {**HUMANOID, "height_above_floor_m": HEAD_M}
    outputs = {"outputs": ["points", "range_scan"], "voxel_m": 0}

    def same(req: dict[str, Any], files: dict[str, bytes], tilt: float) -> dict[str, Any]:
        """The answer: the one of a tilt sigma no floor lies beyond; its floor's tilt."""
        doc = an.analyse(req, files)
        assert an.analyse({**req, "sigma_tilt_deg": 60.0}, files) == doc
        assert doc["scale_check"]["floor_tilt_deg"] == pytest.approx(tilt, abs=0.3)
        return doc

    # a monocular floor 4 degrees from the pose's up: within 3 x 2 degrees
    T, z = _tilted(4.0)
    an.backend.other_depth = z * 0.8
    req = {"image": "i", "camera": cam, "T_map_cam": T.tolist(), **outputs}
    same(req, {"i": jpeg(40)}, 4.0)
    # a depth image 7 degrees off, whatever the sigma
    T, z = _tilted(7.0)
    dep = png(np.rint(z / 0.001).astype(np.uint16))
    doc = same({**req, "T_map_cam": T.tolist(), "depth": "d", "sigma_tilt_deg": 0.5},
               {"i": jpeg(41), "d": dep}, 7.0)
    assert doc["scale_check"]["depth_source"] == "sensor"
    # no pose: the gravity the model sees is the up axis (here the sent pose's), nothing turned
    an.backend.other_depth = z * 0.8
    an.backend.up = T[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    doc = same({"image": "i", "camera": cam, "sigma_tilt_deg": 0.5, **outputs},
               {"i": jpeg(42)}, 7.0)
    assert doc["frame"] == "gravity_camera"


def test_sigma_tilt_deg_is_validated(an: An) -> None:
    T, z = _tilted(0.0)
    an.backend.other_depth = z
    ok = {"image": "i", "camera": HUMANOID, "T_map_cam": T.tolist(), "outputs": ["range_scan"]}
    for bad in (0, -1.0, "2"):
        err = an.analyse({**ok, "sigma_tilt_deg": bad}, {"i": jpeg(44)}, 400)["error"]
        assert err["code"] == "invalid_request" and err["details"]["field"] == "sigma_tilt_deg"
    assert an.analyse({**ok, "sigma_tilt_deg": 0.01}, {"i": jpeg(44)})["frame"] == "client"


def test_the_horizontal_s_rotation() -> None:
    assert np.array_equal(analysis.onto_up(np.array([0.0, 0.0, 1.0])), np.eye(3))
    n = np.array([math.sin(math.radians(7.0)), 0.0, math.cos(math.radians(7.0))])
    Q = analysis.onto_up(n)
    assert Q @ n == pytest.approx([0.0, 0.0, 1.0])
    assert np.allclose(Q @ Q.T, np.eye(3))
    assert analysis.tilt_deg(n) == pytest.approx(7.0)
    # without a height: the lowest plane, turned all the same
    T, z = _tilted(7.0)
    K = Intrinsics(270.0, 270.0, 320.0, 240.0, 640, 480, "model")
    hz = analysis.horizontal(z, K, T[:3, :3], None, "mono", None, 2.0, True)
    assert hz.rotated and hz.plane is not None and hz.tilt_deg == pytest.approx(7.0, abs=0.3)
    turned = hz.R @ T[:3, :3].T  # the turn, in the pose's levelled frame
    assert turned @ hz.plane[1] == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)
    assert not analysis.horizontal(z, K, T[:3, :3], None, "mono", None, 2.0, False).rotated
    flat = analysis.horizontal(np.full_like(z, 2.0), K, T[:3, :3], HEAD_M, "mono", None, 2.0,
                               True)
    assert (flat.plane, flat.tilt_deg, flat.rotated) == (None, None, False)
    assert np.array_equal(flat.R, T[:3, :3])


def test_border_edge_and_low_confidence_pixels_hold_no_beam(an: An) -> None:
    """"Unsupported beams" through the API: the model's depth leaves phantoms at 0.4 m on the
    image's border, on a sliver of depth edges and beside its low-confidence pixels (no
    depth); a beam that those alone hold is unseen with support 0, never a short obstacle, and
    the wall the other beams see keeps its range, held by many pixels."""
    cam = {"fx": 300.0, "fy": 300.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480}
    d = np.zeros((480, 640), np.float32)
    d[:, 100:540] = 3.0  # a wall 3 m ahead in the middle of the image; no depth elsewhere ...
    d[:, :8] = 0.4  # ... but an 8 px band on the image's border at 0.4 m
    d[:, 400:403] = 0.4  # ... a sliver of depth edges (3 px) at 0.4 m on the wall
    d[:, 560:565] = 0.4  # ... and a phantom beside the model's low-confidence pixels
    an.backend.other_depth = d
    doc = an.analyse({"image": "i", "camera": cam, "outputs": ["range_scan"]}, {"i": jpeg(43)})
    beams = {b["bearing_deg"]: b for b in doc["range_scan"]}

    def at(col: float) -> dict[str, Any]:  # gravity_camera: counter-clockwise from the view
        return beams[float(round(math.degrees(math.atan2(-(col - 320.0), 300.0))))]

    for col in (2.0, 401.0, 562.0):
        assert (at(col)["kind"], at(col)["range_m"], at(col)["support"]) == ("unseen", None, 0)
    centre = beams[0.0]
    assert centre["kind"] == "obstacle" and centre["range_m"] == pytest.approx(3.0, abs=0.01)
    assert centre["support"] > analysis.MIN_BEAM_SUPPORT


def test_supported_pixels() -> None:
    d = np.full((40, 60), 2.0)
    d[10, 10] = 0.0  # no depth
    d[30:, 40:] = 1.0  # a step: a depth edge along it
    ok = supported_pixels(d, 2, 3)
    assert not (ok[:3].any() or ok[-3:].any() or ok[:, :3].any() or ok[:, -3:].any())
    # the hole and the pixels around it it makes edges of (3 x 3), 2 px further
    assert not ok[7:14, 7:14].any() and ok[6, 6] and ok[14, 14] and ok[20, 20]
    assert not ok[28:33, 45].any() and not ok[35, 38:43].any() and ok[35, 50] and ok[25, 45]
    assert supported_pixels(d, 0, 0).sum() > ok.sum()
    tiny = np.full((2, 2), 1.0)
    tiny[0, 0] = 0.0
    assert not supported_pixels(tiny, 5, 0).any()
