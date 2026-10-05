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

import numpy as np
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
    service = Service(ws, runner, url="http://0.0.0.0:0/", extra_hosts={"testserver"})
    with TestClient(create_app(service)) as client:
        yield service, client
    runner.shutdown()


def run_job(service: Service, client: TestClient, op: str, params: dict, timeout: float = 300,
            query: str = "") -> dict:
    r = client.post(f"/api/ops/{op}{query}", json=params)
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
    assert client.post(f"/api/jobs/{jid}/cancel", json={}).status_code == 200
    job = service.runner.wait(jid, 120)
    assert job.state == "cancelled", job.log_tail
    assert store.full_tree_hash(ws.maps / "m") == before
    assert json.loads((ws.maps / "m" / "map.json").read_text())["update_count"] == 1


def test_map_viewer_and_saved_job_viewers(stub_server: None,
                                          svc: tuple[Service, TestClient]) -> None:
    """The viewer of a map, of a view.sh -i / -m job and of an image job that asked for it, all
    served in-process by the viewer's own routes; a job's saved viewer survives a restart."""
    service, client = svc
    ws = service.workspace
    minimal_map(ws.maps / "m")
    page = client.get("/viewer/map/m/")
    assert page.status_code == 200 and b"<html" in page.content.lower()
    assert client.get("/viewer/map/m/api/meta").json()["mode"] == "map"
    assert client.get("/api/maps/m/viewer/api/meta").json()["mode"] == "map"
    assert client.get("/viewer/map/m/api/cloud?label=on").status_code == 400  # the viewer's own
    assert client.head("/viewer/map/m/api/meta").status_code == 200

    job = run_job(service, client, "view-map", {"map": "m"})
    assert client.get(f"/viewer/job/{job['id']}/api/meta").json()["mode"] == "map"

    image = jpeg(ws.root / "inputs" / "photo.jpg").read_bytes()
    octet = {"content-type": "application/octet-stream"}
    up = client.post("/api/uploads?name=photo.jpg", headers=octet, content=image).json()
    job = run_job(service, client, "view-image", {"image": up["path"]})
    assert job["viewer"] == f"/viewer/job/{job['id']}/"
    assert not (ws.uploads / up["id"]).exists()  # the viewer keeps what it needs, not the upload
    meta = client.get(f"/api/jobs/{job['id']}/viewer/api/meta").json()
    assert meta["mode"] == "image" and meta["title"] == "photo.jpg"
    cloud = client.get(f"/viewer/job/{job['id']}/api/cloud?stride=2&normals=on")
    assert cloud.status_code == 200  # the live controls re-derive from the saved source
    seg_png = client.get(f"/viewer/job/{job['id']}/api/segmented.png")
    assert seg_png.status_code == 200 and seg_png.content[:4] == b"\x89PNG"

    # one upload, one job: the command's result and the image's viewer
    up = client.post("/api/uploads?name=photo.jpg", headers=octet, content=image).json()
    both = run_job(service, client, "segment-image", {"image": up["path"], "min_score": 0.2},
                   query="?viewer=true")
    assert both["result"]["name"] == "result.json" and both["viewer"]
    assert client.get(f"/viewer/job/{both['id']}/api/meta").json()["mode"] == "image"
    assert not (ws.uploads / up["id"]).exists()
    # the viewer replays the command's inference: the same objects and ids as the result
    scene = json.loads(client.get(both["result"]["url"]).content)
    objects = {int(k): o["type"] for k, o in scene["openlabel"].get("objects", {}).items()}
    catalog = client.get(f"/viewer/job/{both['id']}/api/catalog").json()
    assert {r["id"]: r["label"] for r in catalog} == objects
    assert not (ws.jobs / both["id"] / "inference").exists()  # deleted once the viewer is saved
    # the job's stages are the command's own (its timings record); the viewer step's are not
    timings = client.get(f"/api/jobs/{both['id']}/timings").json()
    assert sorted(s["stage"] for s in both["stages"]) == sorted(timings["stages_s"])
    assert both["viewer_progress"] is None
    viewer_events = (ws.jobs / both["id"] / "viewer_progress.jsonl").read_text().splitlines()
    assert any(json.loads(e).get("event") == "stage_start" for e in viewer_events)

    # after a restart (a new runner and service on the same workspace) the viewer is still there
    again = Runner(ws)
    again.load()
    fresh = Service(ws, again, extra_hosts={"testserver"})
    with TestClient(create_app(fresh)) as c2:
        r = c2.get(f"/viewer/job/{job['id']}/api/catalog")
        assert r.status_code == 200
        assert r.json() == client.get(f"/viewer/job/{job['id']}/api/catalog").json()
    again.shutdown()


def test_recorded_inference_replays_identically(stub_server: None, tmp_path: Path) -> None:
    """``client.replay``: a command run on a recording gives the same bytes without the server;
    without a recording the requests go to the server (exit 3 when there is none)."""
    image = jpeg(tmp_path / "photo.jpg")
    rec = tmp_path / "rec"
    first = subprocess.run([str(REPO / "segment.sh"), "-i", str(image), "--min-score", "0.3"],
                           capture_output=True, timeout=300,
                           env={**os.environ, "OH_MY_SLAM_INFERENCE_RECORD": str(rec)})
    assert first.returncode == 0, first.stderr.decode()
    assert (rec / "responses.jsonl").is_file()
    nowhere = tmp_path / "no-server"  # a runtime dir with no inference server in it
    nowhere.mkdir()
    env = {**os.environ, "OH_MY_SLAM_INFERENCE_REPLAY": str(rec),
           "OH_MY_SLAM_RUNTIME_DIR": str(nowhere)}
    again = subprocess.run([str(REPO / "segment.sh"), "-i", str(image), "--min-score", "0.3"],
                           capture_output=True, timeout=300, env=env)
    assert again.returncode == 0, again.stderr.decode()
    assert again.stdout == first.stdout
    missing = subprocess.run([str(REPO / "segment.sh"), "-i", str(image)], capture_output=True,
                             timeout=300, env={**env, "OH_MY_SLAM_INFERENCE_REPLAY":
                                               str(tmp_path / "absent")})
    assert missing.returncode == 3, missing.stderr.decode()
    assert b"./start_inference_server.sh" in missing.stderr


def test_a_saved_image_bundle_serves_the_same_viewer(tmp_path: Path) -> None:
    """save_bundle / load_bundle: /api/meta byte-identical, and every -i control's cloud the same
    points, colours, labels and normals."""
    from oh_my_slam.viewer.bundle import image_bundle, load_bundle, save_bundle
    from oh_my_slam.viewer.routes import ViewerRoutes, parse_cloud_payload
    from tests.browser.scenes import synthetic_image

    (tmp_path / "img").mkdir()
    img, client = synthetic_image(tmp_path / "img")
    live = image_bundle(img, client)
    save_bundle(live, tmp_path / "saved")
    a, b = ViewerRoutes(live), ViewerRoutes(load_bundle(tmp_path / "saved"))
    for path in ("/api/meta", "/api/scene", "/api/catalog", "/api/segmented.png", "/"):
        assert a.handle("GET", path).tobytes() == b.handle("GET", path).tobytes(), path
    for query in ("", "color=segment&normals=on", "color=height&voxel=0.05", "stride=3",
                  "min-depth=0.5&max-depth=3&edge=0", "color=none"):
        ra, rb = a.handle("GET", "/api/cloud", query), b.handle("GET", "/api/cloud", query)
        assert ra.status == rb.status == 200, query
        (ha, xa), (hb, xb) = (parse_cloud_payload(r.tobytes()) for r in (ra, rb))
        ha.pop("seconds"), hb.pop("seconds")
        assert ha == hb, query
        assert xa.keys() == xb.keys() and all(np.array_equal(xa[k], xb[k]) for k in xa), query


@pytest.mark.parametrize("attrs", [None, "color=height,voxel=0.02"])
def test_a_ply_reconstruction_with_its_viewer(stub_server: None, svc: tuple[Service, TestClient],
                                              tmp_path: Path, attrs: str | None) -> None:
    """``reconstruct -f ply ?viewer=true``: the command asks no segmentation, so the viewer step
    replays what it recorded and forwards the rest; the result is the command's bytes."""
    service, client = svc
    ws = service.workspace
    image = jpeg(ws.root / "inputs" / "photo.jpg")
    params = {"image": "inputs/photo.jpg", "format": "ply"} | ({"attrs": attrs} if attrs else {})
    job = run_job(service, client, "reconstruct", params, query="?viewer=true")
    assert job["viewer_error"] is None and job["viewer"] == f"/viewer/job/{job['id']}/"
    res = cli("reconstruct.sh", "-i", str(image.resolve()), "-f", "ply",
              *(["-p", attrs] if attrs else []))
    assert client.get(job["result"]["url"]).content == res.stdout
    assert client.get(f"/viewer/job/{job['id']}/api/meta").json()["mode"] == "image"
    assert client.get(f"/viewer/job/{job['id']}/api/cloud?color=height").status_code == 200
    assert not (ws.jobs / job["id"] / "inference").exists()  # the recording is gone
