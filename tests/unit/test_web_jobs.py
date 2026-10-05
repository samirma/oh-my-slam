"""``server.sh`` jobs against the commands themselves (spec §2.6 "Results", "Jobs", "Viewer"): with
the stub inference server, a job's result and its ``-d`` files are byte-identical to a direct run of
the command; a cancelled map update leaves the map's tree hash unchanged; the viewer of a map is
the viewer's own routes, and a ``view.sh`` job's viewer is proxied."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from oh_my_slam.mapping import store
from oh_my_slam.web.app import Service, create_app
from oh_my_slam.web.jobs import Runner
from oh_my_slam.web.workspace import Workspace
from tests.fakes.stub_server import start_stub_server
from tests.unit.test_view_cli import minimal_map, sh
from tests.unit.test_web_api import jpeg

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
    runner = Runner(ws, stop_grace_s=60)
    service = Service(ws, runner, url="http://0.0.0.0:0/")
    with TestClient(create_app(service)) as client:
        yield service, client
    runner.shutdown()


def run_job(service: Service, client: TestClient, op: str, params: dict, timeout: float = 300
            ) -> dict:
    r = client.post(f"/api/ops/{op}", json=params)
    assert r.status_code == 202, r.json()
    job = service.runner.wait(r.json()["id"], timeout)
    assert job.state == "succeeded", (job.error, job.log_tail)
    return client.get(f"/api/jobs/{job.id}").json()


def cli(script: str, *args: str) -> subprocess.CompletedProcess[bytes]:
    res = subprocess.run([str(REPO / script), *args], capture_output=True, timeout=600,
                         env=os.environ.copy())
    assert res.returncode == 0, res.stderr.decode()
    return res


def job_files(client: TestClient, jid: str, folder: str) -> dict[str, bytes]:
    files = client.get(f"/api/jobs/{jid}/files").json()
    return {f["path"].removeprefix(folder + "/"): client.get(f["url"]).content
            for f in files if f["path"].startswith(folder + "/")}


def test_results_and_files_are_byte_identical_to_the_commands(
        stub_server: None, svc: tuple[Service, TestClient], tmp_path: Path) -> None:
    service, client = svc
    ws = service.workspace
    image = jpeg(ws.root / "inputs" / "photo.jpg")
    real = str(image.resolve())

    job = run_job(service, client, "segment-image",
                  {"image": "inputs/photo.jpg", "artifacts": "files", "attrs": "voxel=0.02"})
    assert job["result"]["name"] == "result.json"
    assert job["stages"] and {s["stage"] for s in job["stages"]} <= {
        "connect", "inference", "segment", "export", "artifacts", "write"}
    assert client.get(f"/api/jobs/{job['id']}/timings").json()["stages_s"]
    res = cli("segment.sh", "-i", real, "-d", str(tmp_path / "cli"), "-p", "voxel=0.02")
    assert client.get(job["result"]["url"]).content == res.stdout
    files = job_files(client, job["id"], "files")
    assert sorted(files) == sorted(p.name for p in (tmp_path / "cli").iterdir())
    for name, data in files.items():
        assert data == (tmp_path / "cli" / name).read_bytes(), name

    job = run_job(service, client, "reconstruct",
                  {"image": "inputs/photo.jpg", "format": "ply", "attrs": "normals=on"})
    assert job["result"]["name"] == "result.ply"
    res = cli("reconstruct.sh", "-i", real, "-f", "ply", "-p", "normals=on")
    assert client.get(job["result"]["url"]).content == res.stdout

    # a map made by the service is a mapper.sh map: the commands read it, and the same
    # read-only export through the service is byte-identical
    run_job(service, client, "mapper-update", {"inputs": ["inputs/photo.jpg"], "map": "m"})
    assert store.classify(ws.maps / "m") == "map"
    summary = client.get("/api/maps").json()[0]
    assert summary["name"] == "m" and summary["frames"] == 1 and summary["update_count"] == 1
    assert client.get(f"/api/maps/m/files/{summary['thumbnail']}").status_code == 200
    job = run_job(service, client, "segment-map", {"map": "m", "artifacts": "files",
                                                   "format": "ply", "output": "objects.ply"})
    assert job["result"]["name"] == "objects.ply"
    res = cli("segment.sh", "-m", str((ws.maps / "m").resolve()), "-f", "ply",
              "-d", str(tmp_path / "cli_m"))
    assert client.get(job["result"]["url"]).content == res.stdout
    for name, data in job_files(client, job["id"], "files").items():
        assert data == (tmp_path / "cli_m" / name).read_bytes(), name


@needs_colmap
def test_cancelled_map_update_leaves_the_map_unchanged(stub_server: None,
                                                       svc: tuple[Service, TestClient]) -> None:
    service, client = svc
    ws = service.workspace
    inputs = ws.root / "inputs"
    inputs.mkdir()
    for f in FRAMES[:6]:
        shutil.copy(f, inputs / f.name)
    run_job(service, client, "mapper-update", {"inputs": [f"inputs/{FRAMES[0].name}"],
                                               "map": "m"})
    before = store.full_tree_hash(ws.maps / "m")
    r = client.post("/api/ops/mapper-update",
                    json={"inputs": [f"inputs/{f.name}" for f in FRAMES[1:6]], "map": "m"})
    assert r.status_code == 202
    jid = r.json()["id"]
    deadline = time.monotonic() + 120
    while service.runner.get(jid).stage in (None, "setup", "ingest") \
            and time.monotonic() < deadline:
        time.sleep(0.05)  # well inside the update: inference or reconstruction running
    assert service.runner.get(jid).state == "running"
    assert client.post(f"/api/jobs/{jid}/cancel").status_code == 200
    job = service.runner.wait(jid, 120)
    assert job.state == "cancelled", job.log_tail
    assert store.full_tree_hash(ws.maps / "m") == before
    assert json.loads((ws.maps / "m" / "map.json").read_text())["update_count"] == 1


def test_map_viewer_and_view_job_viewer(svc: tuple[Service, TestClient]) -> None:
    service, client = svc
    minimal_map(service.workspace.maps / "m")
    page = client.get("/viewer/map/m/")
    assert page.status_code == 200 and b"<html" in page.content.lower()
    assert client.get("/viewer/map/m/api/meta").json()["mode"] == "map"
    assert client.get("/viewer/map/m/api/cloud?label=on").status_code == 400  # the viewer's own
    assert client.head("/viewer/map/m/api/meta").status_code == 200

    r = client.post("/api/ops/view-map", json={"map": "m"})
    assert r.status_code == 202, r.json()
    assert "--no-browser" in r.json()["command"]  # the service shows the viewer itself
    job = service.runner.wait(r.json()["id"], 120)
    assert job.state == "succeeded" and job.viewer == f"/viewer/job/{job.id}/"
    meta = client.get(f"/viewer/job/{job.id}/api/meta")
    assert meta.status_code == 200 and meta.json()["mode"] == "map"
    assert client.get(f"/viewer/job/{job.id}/").status_code == 200
    service.runner.shutdown()  # the viewer process closes with the service
    assert service.runner.get(job.id).viewer is None
    assert client.get(f"/viewer/job/{job.id}/api/meta").status_code == 410
