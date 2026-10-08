"""``view.sh`` (spec §2.5) beyond its normal pages: the controls the attribute table gives, a map
above the display budget prepared ahead, failures the page did not cause, cameras the scene does
not place, ``-i`` without the inference server, and a browser that hangs up mid-answer."""

from __future__ import annotations

import json
import logging
import socket
import threading
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oh_my_slam.core.cloud_attrs import ATTRIBUTES, CloudAttrs, CloudScope, applicable
from oh_my_slam.core.errors import ExitCode, ServerUnavailableError
from oh_my_slam.core.geometry import rot_z
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.reconstruction.cloud import ImageCloudSource
from oh_my_slam.schema import openlabel as ol
from oh_my_slam.segmentation.api import POINT_COLORS, map_cloud_source
from oh_my_slam.viewer import bundle as vb
from oh_my_slam.viewer.bundle import ViewBundle, scene_cameras
from oh_my_slam.viewer.routes import ViewerRoutes
from oh_my_slam.viewer.server import make_handler

K = Intrinsics(100.0, 100.0, 2.5, 2.0, 5, 4)


@contextmanager
def logged(logger: logging.Logger) -> Iterator[list[str]]:
    lines: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda r: lines.append(r.getMessage())  # type: ignore[method-assign]
    logger.addHandler(handler)
    try:
        yield lines
    finally:
        logger.removeHandler(handler)


def image_source(depth: np.ndarray) -> ImageCloudSource:
    return ImageCloudSource(depth=depth.astype(np.float32), valid=np.ones(depth.shape, bool),
                            rgb=np.zeros((*depth.shape, 3), np.uint8), K=K, colors=POINT_COLORS)


def small_map(n: int = 4) -> ViewBundle:
    source = map_cloud_source(np.arange(3 * n, dtype=np.float32).reshape(n, 3),
                              np.zeros((n, 3), np.uint8), None, set(), np.zeros((1, 3)))
    return ViewBundle(mode="map", title="t", scene={}, source=source)


# --- controls ------------------------------------------------------------------------------------


def test_every_numeric_attribute_is_a_slider_and_the_depth_range_follows_the_image() -> None:
    depth = np.full((4, 5), 1.5)
    depth[0, 0], depth[0, 1], depth[1, 1] = 3.47, np.inf, 0.0  # farthest valid: 3.47 m
    b = ViewBundle(mode="image", title="i", scene={}, source=image_source(depth), catalog=[])
    by_key = {c["key"]: c for c in b.controls}
    assert {c["kind"] for c in b.controls} == {"choice", "int", "float", "toggle"}
    numeric = {a.key for a in ATTRIBUTES if a.key not in vb.PLY_ONLY
               and isinstance(getattr(CloudAttrs(), a.field), int | float)
               and not isinstance(getattr(CloudAttrs(), a.field), bool)}
    assert numeric == set(vb._SLIDERS) == {k for k, c in by_key.items() if c["kind"] in
                                           ("int", "float")}
    assert by_key["min-depth"]["max"] == by_key["max-depth"]["max"] == 3.5  # rounded up
    assert by_key["edge"]["max"] == 0.5 and by_key["stride"]["kind"] == "int"
    empty = ViewBundle(mode="image", title="i", scene={}, source=image_source(np.zeros((4, 5))),
                       catalog=[])
    assert {c["key"]: c for c in empty.controls}["max-depth"]["max"] == 10.0  # no valid depth
    # a map has no pixel-level control, so no depth range
    assert [c["key"] for c in small_map().controls] == [
        a.key for a in applicable(CloudScope.MAP) if a.key not in vb.PLY_ONLY]


# --- a map above the display budget --------------------------------------------------------------


class _Reader:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.frames = [types.SimpleNamespace(name="f000000", source="/in/IMG_1.jpg"),
                       types.SimpleNamespace(name="f000001", source=None)]


@pytest.fixture
def opened_maps(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """``map_bundle`` over a stand-in reader (the map store's own tests cover reading a map);
    ``prepare`` records the thread it runs on instead of searching a selection."""
    from oh_my_slam.mapping import export

    scene = ol.document(ol.metadata("m"), {})
    monkeypatch.setattr(export, "map_objects", lambda reader: (None, []))
    monkeypatch.setattr(export, "scene_bytes", lambda reader: json.dumps(scene).encode())
    prepared: list[str] = []
    monkeypatch.setattr(ViewBundle, "prepare",
                        lambda self: prepared.append(threading.current_thread().name))
    return prepared


def test_a_map_above_the_budget_finds_its_selection_while_the_browser_starts(
        opened_maps: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from oh_my_slam.mapping import export

    n = vb.DISPLAY_POINT_BUDGET + 1  # positions broadcast from one row: no memory needed
    source = map_cloud_source(np.broadcast_to(np.zeros(3, np.float32), (n, 3)),
                              np.broadcast_to(np.zeros(3, np.uint8), (n, 3)), None, set(),
                              np.zeros((1, 3)))
    monkeypatch.setattr(export, "reader_source", lambda reader, objs: source)
    b = vb.map_bundle(tmp_path / "office", reader=_Reader(tmp_path / "office"))  # type: ignore[arg-type]
    for t in threading.enumerate():
        if t.name == "display-selection":
            t.join(10)
    assert opened_maps == ["display-selection"]  # on its own thread, started by map_bundle
    assert b.mode == "map" and b.title == "office" and b.catalog is None  # an image's only
    assert b.camera_sources == {"f000000": "IMG_1.jpg"}  # keyframe copies name their input
    assert b.scene["openlabel"]["metadata"]["name"] == "m"


def test_a_map_within_the_budget_is_not_prepared_ahead(opened_maps: list[str],
                                                      monkeypatch: pytest.MonkeyPatch,
                                                      tmp_path: Path) -> None:
    from oh_my_slam.mapping import export

    monkeypatch.setattr(export, "reader_source", lambda reader, objs: small_map().source)
    vb.map_bundle(tmp_path / "m", reader=_Reader(tmp_path / "m"))  # type: ignore[arg-type]
    assert opened_maps == []


# --- failures the page did not cause -------------------------------------------------------------


def test_a_failed_preparation_is_logged_and_the_page_gets_the_error(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def out_of_memory(*args: Any) -> Any:
        raise MemoryError("cannot allocate 3.2 GB")

    monkeypatch.setattr(vb, "derive_thinned", out_of_memory)
    b = small_map()
    with logged(vb.log) as lines:
        b.prepare()  # on view.sh's background thread: never raises
        routes = ViewerRoutes(b)
        r = routes.handle("GET", "/api/cloud", "voxel=0.1")
    assert lines[0] == "viewer: preparing the default cloud failed: cannot allocate 3.2 GB"
    assert r.status == 500 and json.loads(r.tobytes()) == {
        "error": "MemoryError: cannot allocate 3.2 GB"}
    assert lines[1] == "viewer: /api/cloud failed: cannot allocate 3.2 GB"
    assert routes.handle("GET", "/api/scene").status == 200  # the server keeps serving


# --- cameras -------------------------------------------------------------------------------------


def test_cameras_the_scene_does_not_place_are_not_shown() -> None:
    """Only a camera stream with pinhole intrinsics, placed by its frame's transform or being a
    root frame itself, is a camera of the page."""
    pose = Pose(rot_z(0.2), np.array([1.0, 2.0, 0.5]))
    frames = {"0": ol.frame(0.0, stream_uris={"camera_0": "a.jpg", "camera_1": "b.jpg",
                                              "imu": "imu.csv"},
                            transforms={"c0": ol.transform("camera_0", "map", pose)})}
    scene = ol.document(
        ol.metadata("m"), {},
        coordinate_systems={"map": ol.map_cs(["camera_0", "camera_1"]),
                            "camera_0": ol.sensor_cs("map"), "camera_1": ol.sensor_cs("map")},
        streams={"camera_0": ol.camera_stream(K), "camera_1": ol.camera_stream(K),
                 "imu": {"type": "other"}},
        frames=frames)
    (cam,) = scene_cameras(scene)  # camera_1 has no pose in this frame; imu is no camera
    assert cam["name"] == "camera_0" and cam["source"] == "a.jpg"
    np.testing.assert_allclose(cam["T"], pose.matrix(), atol=1e-6)


# --- view.sh -i needs the inference server -------------------------------------------------------


def test_an_image_without_the_inference_server_fails_with_exit_3(tmp_path: Path) -> None:
    from argparse import Namespace

    from PIL import Image

    from oh_my_slam.cli.view import make_bundle

    img = tmp_path / "room.png"
    Image.fromarray(np.zeros((24, 32, 3), np.uint8)).save(img)
    with pytest.raises(ServerUnavailableError) as err:  # no server in this test session
        make_bundle(Namespace(map=None, image=img))
    assert err.value.exit_code == ExitCode.SERVER_UNAVAILABLE
    assert "./start_inference_server.sh" in str(err.value)


# --- a browser that hangs up ---------------------------------------------------------------------


def test_a_browser_that_hangs_up_mid_answer_is_ignored() -> None:
    """The browser closes the connection before reading the answer (a reload, a closed tab):
    the handler finishes quietly instead of raising."""
    handler = make_handler(small_map())
    server_side, browser = socket.socketpair()
    browser.sendall(b"GET /api/meta HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
    browser.close()
    try:
        handler(server_side, ("127.0.0.1", 0), types.SimpleNamespace())
    finally:
        server_side.close()
