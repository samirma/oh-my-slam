"""Edges of kept modules that the removed commands' tests used to reach: the inference client's
clone and an image sent as it is, the map-scope attribute rule, the mapper's default progress,
incremental mapping restricted to some images, normals asked for twice and in chunks, map point
labels, detections on the caller's own client, and the inference server's state file and pids."""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from PIL import Image

from oh_my_slam.client import protocol as p
from oh_my_slam.client.client import InferenceClient


def test_a_clone_is_a_separate_client_and_a_small_image_is_sent_as_it_is(tmp_path: Path) -> None:
    sock = tmp_path / "srv.sock"
    sock.touch()
    img = tmp_path / "small.png"
    Image.fromarray(np.full((48, 64, 3), 90, np.uint8)).save(img)
    c = InferenceClient(sock, timeout=3.0)
    twin = c.clone()
    assert twin is not c and twin.socket_path == sock and twin.timeout == 3.0
    body = p.GeometryResponse(width=64, height=48, orig_width=64, orig_height=48,
                              intrinsics=p.PixelIntrinsics(fx=50, fy=50, cx=32, cy=24),
                              fov_x_deg=60.0, depth_path="d", mask_path="m").model_dump()
    c._client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body)),
                             base_url="http://x")
    res = c.geometry(p.GeometryRequest(image_path=str(img), out_dir=str(tmp_path)))
    assert (res.orig_width, res.orig_height) == (64, 48)


def test_pixel_level_attributes_are_refused_for_a_map() -> None:
    from oh_my_slam.core.cloud_attrs import CloudScope, parse_cloud_attrs
    from oh_my_slam.core.errors import UsageError

    with pytest.raises(UsageError, match="pixel-level"):
        parse_cloud_attrs("stride=2", CloudScope.MAP)


def test_the_mapper_s_default_progress_logs(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    from oh_my_slam.mapping import api

    logger = logging.getLogger("oh_my_slam")
    logger.propagate = True
    try:
        with caplog.at_level(logging.INFO, logger="oh_my_slam"):
            api._progress("mapping")
    finally:
        logger.propagate = False
    assert "mapping" in caplog.text


def test_incremental_mapping_registers_only_the_images_named(tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    from oh_my_slam.mapping.sfm import Sfm

    seen: dict[str, Any] = {}

    class Options:
        pass

    def incremental_mapping(db: str, images: str, out: str, opts: Any, input_path: str) -> dict:
        seen["names"] = opts.image_names
        return {}

    fake = types.SimpleNamespace(IncrementalPipelineOptions=Options,
                                 incremental_mapping=incremental_mapping)
    sfm = Sfm(tmp_path / "db.db", tmp_path / "frames", tmp_path / "work", only=["b.jpg", "a.jpg"])
    monkeypatch.setitem(sys.modules, "pycolmap", fake)
    assert sfm.map_incremental(tmp_path / "out", seed=7) is None
    assert seen["names"] == ["a.jpg", "b.jpg"]


def test_normals_are_kept_and_computed_in_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    from oh_my_slam.reconstruction import pointcloud

    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(400, 3))
    one = pointcloud.PointNormals(xyz, np.zeros((0, 3)))
    first = one.at(np.arange(400))
    assert np.array_equal(one.at(np.arange(10)), first[:10])  # nothing computed again
    monkeypatch.setattr(pointcloud, "_NORMAL_CHUNK", 64)
    chunked = pointcloud.PointNormals(xyz, np.zeros((0, 3)))
    assert np.allclose(chunked.at(np.arange(400)), first)


def test_map_points_without_labels() -> None:
    from oh_my_slam.segmentation.api import map_cloud_source

    src = map_cloud_source(np.zeros((3, 3)), np.zeros((3, 3), np.uint8), None, {1},
                           np.zeros((1, 3)))
    assert src is not None


def test_detections_alongside_on_the_caller_s_client(monkeypatch: pytest.MonkeyPatch,
                                                     tmp_path: Path) -> None:
    from oh_my_slam.segmentation import api

    class Client:
        def clone(self) -> Client:
            return self  # one connection only: nothing extra to close

        def close(self) -> None:
            raise AssertionError("the caller's own client is not closed")

    monkeypatch.setattr(api, "detect", lambda *a, **k: ["det"])
    closed: list[bool] = []

    class Own(Client):
        def close(self) -> None:
            closed.append(True)

    class Pooled(Client):
        def clone(self) -> Client:
            return Own()

    api.detect_alongside(tmp_path / "a.jpg", Pooled(), lambda client: "frame",  # type: ignore[arg-type]
                         max_side=64)
    assert closed == [True]  # the second connection is closed
    c = Client()
    frame, dets = api.detect_alongside(tmp_path / "a.jpg", c, lambda client: "frame",  # type: ignore[arg-type]
                                       max_side=64)
    assert (frame, dets) == ("frame", ["det"])


def test_state_file_and_pids(monkeypatch: pytest.MonkeyPatch) -> None:
    from oh_my_slam.core import paths
    from oh_my_slam.server import lifecycle

    paths.state_file().write_text("{")
    assert lifecycle.read_state() is None
    paths.state_file().write_text(json.dumps({"pid": 1}))
    assert lifecycle.read_state() == {"pid": 1}
    paths.state_file().unlink()

    def denied(pid: int, sig: int) -> None:
        raise PermissionError("another user's process")

    monkeypatch.setattr(os, "kill", denied)
    assert lifecycle.pid_alive(1)

    def gone(pid: int, sig: int) -> None:
        raise ProcessLookupError("no such process")

    monkeypatch.setattr(os, "kill", gone)
    assert not lifecycle.pid_alive(1)
