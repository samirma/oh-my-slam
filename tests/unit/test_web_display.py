"""What the service gives the 3D scene viewer (http_server.md "3D scene viewer"): a job's PLY within
the display budget of spec §2.5, by the shared selection (``segmentation.cloud.thin_cloud``, each
kept point with exactly its values), with the file's header comments; and the viewer's own display
transform of a scene. Plus the guard that keeps the browser's port of the checks beyond the schema
in step with ``schema/validate.py``."""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from starlette.testclient import TestClient

from oh_my_slam.core.cloud_attrs import CloudAttrs, CloudScope
from oh_my_slam.core.geometry import budget_voxel_indices
from oh_my_slam.core.ply import PointCloud, ply_bytes
from oh_my_slam.reconstruction.gravity import DEFAULT_UP_CAM
from oh_my_slam.schema import validate
from oh_my_slam.segmentation.cloud import IMAGE_FRAME, MAP_FRAME, thin_cloud
from oh_my_slam.viewer import bundle
from oh_my_slam.viewer.routes import parse_cloud_payload
from oh_my_slam.web.app import Service, create_app
from oh_my_slam.web.jobs import Job, Runner
from oh_my_slam.web.workspace import Workspace

N = 5000


def cloud() -> PointCloud:
    rng = np.random.default_rng(3)
    normals = rng.normal(size=(N, 3))
    return PointCloud(rng.uniform(-2, 2, (N, 3)), rng.integers(0, 255, (N, 3), np.uint8),
                      rng.integers(0, 4, N).astype(np.int32),
                      normals / np.linalg.norm(normals, axis=1, keepdims=True))


@pytest.fixture
def svc(tmp_path: Path) -> Iterator[tuple[Service, TestClient, Path]]:
    ws = Workspace(tmp_path / "data")
    ws.create()
    runner = Runner(ws)
    service = Service(ws, runner, url="http://0.0.0.0:0/", extra_hosts={"testserver"})
    job = Job("j1", "segment-map", "segment.sh -m", {}, [], [], False, result_name="result.ply",
              state="succeeded")
    runner.jobs[job.id] = job
    out = ws.job_dir(job.id) / "out"
    out.mkdir(parents=True)
    with TestClient(create_app(service)) as client:
        yield service, client, out
    runner.shutdown()


def test_thin_cloud_is_the_shared_selection_with_exact_values() -> None:
    c = cloud()
    assert thin_cloud(c, N).cloud is c and thin_cloud(c, N).voxel == 0.0
    t = thin_cloud(c, 1000)
    keep, edge = budget_voxel_indices(c.xyz, 1000)
    assert t.total == N and len(t.cloud) == len(keep) <= 1000 and t.voxel == edge > 0
    for a, b in ((t.cloud.xyz, c.xyz), (t.cloud.rgb, c.rgb), (t.cloud.label, c.label),
                 (t.cloud.normals, c.normals)):
        assert np.array_equal(a, b[keep])


def test_a_job_ply_above_the_budget_is_served_thinned(svc: tuple[Service, TestClient, Path],
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    service, client, out = svc
    c = cloud()
    comments = [MAP_FRAME, f"attributes {CloudAttrs(normals=True, label=True).describe(CloudScope.MAP)}",
                'located_0 {"image": "a.jpg", "located": false}']
    (out / "result.ply").write_bytes(ply_bytes(c, comments=comments))
    (out / "files").mkdir()
    (out / "files" / "segments.ply").write_bytes(ply_bytes(c, encoding="ascii", comments=comments))
    monkeypatch.setattr(bundle, "DISPLAY_POINT_BUDGET", 1000)
    for url in ("/api/jobs/j1/display-cloud", "/api/jobs/j1/display-cloud?file=files/segments.ply"):
        r = client.get(url)
        assert r.status_code == 200, r.text
        head, arrays = parse_cloud_payload(r.content)
        keep, edge = budget_voxel_indices(c.xyz, 1000)
        assert head["total"] == N and head["count"] == len(keep) and head["voxel"] == edge
        assert head["comments"] == comments and head["attrs"] == comments[1].removeprefix("attributes ")
        assert np.array_equal(arrays["position"], c.xyz[keep])
        assert np.array_equal(arrays["color"], c.rgb[keep]) and np.array_equal(arrays["label"], c.label[keep])
        assert np.allclose(arrays["normal"], c.normals[keep], atol=1e-6)
    # within the budget: every point
    monkeypatch.setattr(bundle, "DISPLAY_POINT_BUDGET", N)
    (out / "small.ply").write_bytes(ply_bytes(c, comments=comments))
    head, _ = parse_cloud_payload(client.get("/api/jobs/j1/display-cloud?file=small.ply").content)
    assert head["count"] == head["total"] == N and head["voxel"] == 0
    # what is not such a file is refused with the reason; a path never leaves the job's folder
    (out / "bad.ply").write_bytes(b"ply\nformat binary_big_endian 1.0\nend_header\n")
    r = client.get("/api/jobs/j1/display-cloud?file=bad.ply")
    assert r.status_code == 400 and "binary_big_endian" in r.json()["error"]["message"]
    assert client.get("/api/jobs/j1/display-cloud?file=../../x.ply").status_code in (400, 404)
    assert client.get("/api/jobs/j1/display-cloud?file=nope.ply").status_code == 404
    # the header alone can be read first (a Range request)
    r = client.get("/api/jobs/j1/result", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100


def test_display_transform_is_the_viewers_own(svc: tuple[Service, TestClient, Path]) -> None:
    _, client, _ = svc
    eye = np.eye(4).tolist()
    r = client.get("/api/display-transform").json()
    assert r == {"camera_frame": False, "display_transform": eye}
    r = client.get("/api/display-transform", params={"camera": "true"}).json()
    assert r["camera_frame"] and np.allclose(r["display_transform"], bundle.upright_transform(DEFAULT_UP_CAM))
    up = [0.1, -0.9, 0.2]
    r = client.get("/api/display-transform", params={"camera": "true", "up": "0.1,-0.9,0.2"}).json()
    u = np.array(up) / np.linalg.norm(up)
    assert np.allclose(r["display_transform"], bundle.upright_transform(u))
    r = client.get("/api/display-transform", params=[("comment", IMAGE_FRAME), ("comment", "attributes x")]).json()
    assert r["camera_frame"]
    r = client.get("/api/display-transform", params=[("comment", MAP_FRAME)]).json()
    assert not r["camera_frame"]
    assert client.get("/api/display-transform", params={"camera": "1", "up": "1,2"}).status_code == 400
    assert client.get("/api/display-transform", params={"camera": "1", "up": "0,0,0"}).status_code == 400
    assert "/api/display-transform" in client.get("/api/openapi.json").json()["paths"]


# The browser's port of these checks (web/static/js/scene/openlabel.js extraErrors) must be revisited
# whenever they change: update the port, then this hash.
EXTRA_CHECKS_SHA256 = "78e5d44ce722bfdc942020edfd1452f861a19529c673c54b9dcc90b34082ec75"


def test_the_browser_port_of_the_extra_checks_is_in_step() -> None:
    src = "".join(inspect.getsource(f) for f in (validate.extra_errors, validate._check_quat,
                                                 validate._check_intrinsics, validate._is_num))
    assert hashlib.sha256(src.encode()).hexdigest() == EXTRA_CHECKS_SHA256, (
        "schema/validate.py's checks beyond the schema changed: port the change to "
        "web/static/js/scene/openlabel.js, then update EXTRA_CHECKS_SHA256")
