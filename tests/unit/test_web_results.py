"""``server.sh`` operations against the commands themselves (spec §2.6 "Results", "Timings",
"Requests"): with the stub inference server, a response's body is byte-identical to a direct run
of the command, its ``Server-Timing`` names the command's own stages, a map made by the service is
a ``mapper.sh`` map, and a map update whose client disconnects leaves the map's tree hash
unchanged."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from oh_my_slam.commands import spec
from oh_my_slam.mapping import store
from oh_my_slam.web.app import Service, create_app
from oh_my_slam.web.runner import Runner
from oh_my_slam.web.workspace import Workspace
from tests.fakes.stub_server import start_stub_server
from tests.mapsnap import full_tree_hash
from tests.unit.test_view_cli import sh
from tests.unit.test_web_api import _test_client_host, jpeg  # noqa: F401
from tests.unit.test_web_requests import Call, send, started, until

REPO = Path(__file__).resolve().parents[2]
FRAMES = sorted((REPO / "examples" / "ainex-captures").glob("*.jpg"))
needs_colmap = pytest.mark.skipif(shutil.which("colmap") is None, reason="needs Homebrew colmap")


@pytest.fixture(scope="module")
def stub_server() -> Iterator[None]:
    start_stub_server()
    try:
        yield
    finally:
        sh("start_inference_server.sh", "--stop")


@pytest.fixture
def svc(tmp_path: Path) -> Iterator[tuple[Service, TestClient]]:
    ws = Workspace(tmp_path / "data")
    ws.create()
    service = Service(ws, Runner(ws), url="http://0.0.0.0:0/")
    with TestClient(create_app(service)) as client:
        yield service, client
    service.runner.shutdown()


def cli(script: str, *args: str) -> subprocess.CompletedProcess[bytes]:
    res = subprocess.run([str(REPO / script), *args], capture_output=True, timeout=600,
                         env=os.environ.copy())
    assert res.returncode == 0, res.stderr.decode()
    return res


def stages(header: str, label: str) -> list[str]:
    """The stage names of a ``Server-Timing`` header, checked against the command's own."""
    names = [m.split(";dur=")[0] for m in header.split(", ")]
    declared = {str(s) for p, c, m in spec.operations() if c.label(m) == label for s in m.stages}
    assert names[-1] == "total" and names[:-1] and set(names[:-1]) <= declared, (names, label)
    return names


def test_responses_are_byte_identical_to_the_commands(
        stub_server: None, svc: tuple[Service, TestClient], tmp_path: Path) -> None:
    service, client = svc
    ws = service.workspace
    image = jpeg(ws.root / "inputs" / "photo.jpg")
    real = str(image.resolve())

    r = client.post("/api/ops/segment", json={"image": "inputs/photo.jpg"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/json"
    assert r.content == cli("segment.sh", "-i", real).stdout
    stages(r.headers["server-timing"], "segment.sh")

    r = client.post("/api/ops/reconstruct", json={"image": "inputs/photo.jpg", "format": "ply",
                                                  "attrs": "normals=on"})
    assert r.status_code == 200 and r.headers["content-type"] == "application/octet-stream"
    assert r.content == cli("reconstruct.sh", "-i", real, "-f", "ply", "-p", "normals=on").stdout
    stages(r.headers["server-timing"], "reconstruct.sh")

    # the images: the depth image and the segmented image, PNGs byte for byte
    r = client.post("/api/ops/reconstruct", json={"image": "inputs/photo.jpg", "format": "depth"})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.content == cli("reconstruct.sh", "-i", real, "-f", "depth").stdout
    assert "segment" not in stages(r.headers["server-timing"], "reconstruct.sh")
    r = client.post("/api/ops/segment", json={"image": "inputs/photo.jpg", "format": "png"})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.content == cli("segment.sh", "-i", real, "-f", "png").stdout
    stages(r.headers["server-timing"], "segment.sh")

    # a map made by the service is a mapper.sh map
    r = client.post("/api/ops/mapper-update", json={"inputs": ["inputs/photo.jpg"], "map": "m"})
    assert r.status_code == 200, r.text
    assert "commit" in stages(r.headers["server-timing"], "mapper.sh update")
    assert store.classify(ws.maps / "m") == "map"
    summary = client.get("/api/maps").json()[0]
    assert summary["name"] == "m" and summary["frames"] == 1 and summary["update_count"] == 1
    assert list(ws.requests.iterdir()) == []  # the service keeps no result


def test_the_default_scene_and_the_mapper_bodies_are_the_commands(
        stub_server: None, svc: tuple[Service, TestClient]) -> None:
    """Byte for byte what the command writes for the same arguments: reconstruct's scene
    description (its default JSON), the body of a map update, and that of locating an image in
    the map. A map's scene names the map's folder, so the command runs on the same folder: the
    map the service made is moved aside and the command makes it again."""
    service, client = svc
    ws = service.workspace
    real = str(jpeg(ws.root / "inputs" / "photo.jpg").resolve())

    r = client.post("/api/ops/reconstruct", json={"image": "inputs/photo.jpg"})
    assert r.status_code == 200 and r.headers["content-type"] == "application/json"
    assert r.content == cli("reconstruct.sh", "-i", real).stdout

    folder = ws.maps / "m"
    r = client.post("/api/ops/mapper-update", json={"inputs": ["inputs/photo.jpg"], "map": "m"})
    assert r.status_code == 200 and r.headers["content-type"] == "application/json", r.text
    folder.rename(ws.root / "made-by-the-service")
    assert r.content == cli("mapper.sh", "update", "-i", real, "-m", str(folder)).stdout

    r = client.post("/api/ops/mapper-locate", json={"inputs": ["inputs/photo.jpg"], "map": "m"})
    assert r.status_code == 200 and r.headers["content-type"] == "application/json", r.text
    assert r.content == cli("mapper.sh", "locate", "-i", real, "-m", str(folder)).stdout
    assert json.loads(r.content)["openlabel"]["metadata"]["tagged_file"] == str(folder)


@needs_colmap
def test_an_interrupted_map_update_leaves_the_map_unchanged(stub_server: None,
                                                            tmp_path: Path) -> None:
    """The client of a map update disconnects while the command runs: the command is interrupted
    as Ctrl-C would, and the map is the one before the update."""
    ws = Workspace(tmp_path / "data")
    ws.create()
    service = Service(ws, Runner(ws), url="http://0.0.0.0:0/")
    inputs = ws.root / "inputs"
    inputs.mkdir()
    for f in FRAMES[:6]:
        shutil.copy(f, inputs / f.name)

    async def scenario(app: Callable[..., Awaitable[None]]) -> None:
        first = await Call(app, "/api/ops/mapper-update",
                           {"inputs": [f"inputs/{FRAMES[0].name}"], "map": "m"})()
        assert first.status == 200, first.body
        before = full_tree_hash(ws.maps / "m")
        update, task = await send(app, "/api/ops/mapper-update",
                                  {"inputs": [f"inputs/{f.name}" for f in FRAMES[1:6]],
                                   "map": "m"})
        await until(started(service.runner, 1), 60)
        running = service.runner.running[0]
        await until(lambda: (ws.maps / "m" / store.STAGING).exists(), 120)  # inside the update
        await asyncio.sleep(0.5)
        update.leave()
        await task
        assert update.status == 499
        assert running.proc is not None and running.proc.returncode == 130
        assert full_tree_hash(ws.maps / "m") == before
        assert json.loads((ws.maps / "m" / "map.json").read_text())["update_count"] == 1
        assert not (ws.maps / "m" / store.STAGING).exists()

    t0 = time.monotonic()
    try:
        asyncio.run(scenario(create_app(service)))
    finally:
        service.runner.shutdown()
    assert time.monotonic() - t0 < 600
