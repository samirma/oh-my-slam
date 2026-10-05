"""``server.sh`` API without inference (spec §2.6): operations and OpenAPI derived from the commands'
registry (a new option or mode appears with no web change), synchronous per-field validation,
workspace confinement (path escapes and symlinks refused), uploads, the job runner's states,
order, progress, cancellation, events and persistence, and the inference server down."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import sys
import time
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from starlette.testclient import TestClient

from oh_my_slam.commands import spec
from oh_my_slam.web import operations as web_ops
from oh_my_slam.web.app import Service, create_app
from oh_my_slam.web.jobs import Job, Runner
from oh_my_slam.web.openapi import document
from oh_my_slam.web.operations import Prepared
from oh_my_slam.web.workspace import Workspace
from tests.unit.test_view_cli import minimal_map, sh

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _repo_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Job subprocesses run in the workspace; the test helpers (tests.fakes) must import there."""
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
        filter(None, [str(REPO), os.environ.get("PYTHONPATH")])))


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    w = Workspace(tmp_path / "data")
    w.create()
    return w


@dataclasses.dataclass
class Svc:
    service: Service
    client: TestClient

    @property
    def ws(self) -> Workspace:
        return self.service.workspace

    @property
    def runner(self) -> Runner:
        return self.service.runner


@pytest.fixture
def svc(ws: Workspace) -> Iterator[Svc]:
    runner = Runner(ws, stop_grace_s=30)
    service = Service(ws, runner, url="http://0.0.0.0:0/")
    with TestClient(create_app(service)) as client:
        yield Svc(service, client)
    runner.shutdown()


def fields(body: dict[str, Any]) -> list[str]:
    """The parameters a validation flags (the inference server's state concerns none)."""
    return [k for k in body["by_parameter"] if k]


def jpeg(path: Path, seed: int = 5) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)).save(path, quality=95)
    return path


# -- single source of truth --------------------------------------------------------------------------


def test_one_operation_per_command_mode_with_the_commands_parameters() -> None:
    ops = web_ops.operations()
    described = spec.describe()["operations"]
    assert sorted(op.label for op in ops.values()) == sorted(d["id"] for d in described)
    doc = document(ops)
    for d in described:
        op = next(o for o in ops.values() if o.label == d["id"])
        post = doc["paths"][f"/api/ops/{op.id}"]["post"]
        schema = post["requestBody"]["content"]["application/json"]["schema"]
        assert list(schema["properties"]) == [p["name"] for p in d["parameters"]]
        assert schema["required"] == [p["name"] for p in d["parameters"] if p["required"]]
        for p in d["parameters"]:
            prop = schema["properties"][p["name"]]
            assert prop["x-oms"] == p  # names, kinds, defaults, help, bounds: the registry's
            if p["choices"]:
                assert prop["enum"] == p["choices"]
        assert post["x-oms"] == d  # outputs, stages, rules, errors, inference need
        assert f"/api/ops/{op.id}/validate" in doc["paths"]
    assert doc["x-oms"]["exit_codes"] == spec.describe()["exit_codes"]
    assert {op.id for op in ops.values()} == {"reconstruct", "mapper-update", "mapper-locate",
                                              "segment-image", "segment-map", "view-image",
                                              "view-map"}


def test_each_operation_runs_the_commands_own_entry_point() -> None:
    import importlib.util

    for op in web_ops.operations().values():
        assert importlib.util.find_spec(op.module) is not None, op.module
        assert (REPO / f"{op.program.prog}").is_file()
        assert f"oms_exec {op.module.rsplit('.', 1)[1]}" in (REPO / op.program.prog).read_text()


def test_a_new_option_and_a_new_mode_reach_the_api_without_web_changes(
        monkeypatch: pytest.MonkeyPatch, svc: Svc) -> None:
    """Monkeypatch the registry: an extra option on segment.sh and a whole new command appear in
    the OpenAPI document and are accepted by the API, with no change to oh_my_slam.web."""
    seg = spec.SEGMENT.command()
    extra = spec.Option("--shade", "shade", spec.Kind.ENUM, "a new option", default="dark",
                        choices=("dark", "light"))
    seg2 = dataclasses.replace(seg, options=(*seg.options, extra))
    new_mode = spec.Mode(None, None, (), "never", "needs nothing", (spec.Stage.EXPORT,),
                         (spec.Output("result", "stdout", "json", "an echo"),))
    echo_cmd = spec.Command("echo.sh", None, "echo", (
        spec.Option("--word", "word", spec.Kind.ENUM, "what to say", default="hi",
                    choices=("hi", "bye")),), (new_mode,))
    programs = (*[p for p in spec.PROGRAMS if p is not spec.SEGMENT],
                dataclasses.replace(spec.SEGMENT, commands=(seg2,)),
                spec.Program("echo.sh", "echo", (echo_cmd,)))
    monkeypatch.setattr(spec, "PROGRAMS", programs)
    svc.service.ops = web_ops.operations()

    doc = svc.client.get("/api/openapi.json").json()
    props = doc["paths"]["/api/ops/segment-map"]["post"]["requestBody"]["content"][
        "application/json"]["schema"]["properties"]
    assert props["shade"]["enum"] == ["dark", "light"] and props["shade"]["default"] == "dark"
    assert "/api/ops/echo" in doc["paths"]

    minimal_map(svc.ws.maps / "m")
    r = svc.client.post("/api/ops/segment-map/validate", json={"map": "m", "shade": "light"})
    assert r.json()["valid"], r.json()
    assert "--shade=light" in r.json()["command"]
    bad = svc.client.post("/api/ops/segment-map/validate", json={"map": "m", "shade": "blue"})
    assert list(bad.json()["by_parameter"]) == ["shade"]
    r = svc.client.post("/api/ops/echo", json={"word": "bye"})
    assert r.status_code == 202, r.json()
    assert r.json()["command"] == ["echo.sh", "--word=bye"]
    job = svc.runner.wait(r.json()["id"], 60)
    assert job.state == "failed"  # there is no oh_my_slam.cli.echo to run, but it was accepted


# -- validation and the workspace --------------------------------------------------------------------


def test_invalid_requests_get_per_field_errors_and_queue_nothing(svc: Svc) -> None:
    minimal_map(svc.ws.maps / "m")
    r = svc.client.post("/api/ops/segment-map", json={"map": "m", "min_score": 0.2,
                                                      "attrs": "color=rgb", "format": "xml"})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "usage" and err["exit_code"] == 2
    assert set(err["by_parameter"]) >= {"format", "min_score"}
    assert "invalid choice: 'xml'" in err["by_parameter"]["format"][0]  # argparse's message
    r = svc.client.post("/api/ops/segment-image", json={})
    assert r.status_code == 400 and "one of the arguments -i -m is required" in r.json()[
        "error"]["message"]
    r = svc.client.post("/api/ops/segment-map", json={"map": "absent"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "not_a_map"
    r = svc.client.post("/api/ops/segment-map", json={"map": "m", "artifacts": "../x"})
    assert r.status_code == 400 and list(r.json()["error"]["by_parameter"]) == ["artifacts"]
    r = svc.client.post("/api/ops/segment-map", content=b"[1, 2]")
    assert r.status_code == 400
    r = svc.client.post("/api/ops/nope", json={})
    assert r.status_code == 404
    assert svc.client.get("/api/jobs").json() == []
    assert not list(svc.ws.jobs.iterdir())


def test_paths_outside_the_workspace_are_refused(svc: Svc, tmp_path: Path) -> None:
    outside = jpeg(tmp_path / "outside" / "photo.jpg")
    jpeg(svc.ws.root / "inputs" / "ok.jpg")
    (svc.ws.root / "inputs" / "link.jpg").symlink_to(outside)
    (svc.ws.root / "inputs" / "dirlink").symlink_to(outside.parent)
    (svc.ws.maps / "evil").symlink_to(tmp_path / "outside")
    for image in (str(outside), "../outside/photo.jpg", "inputs/../../outside/photo.jpg",
                  "inputs/link.jpg", "inputs/dirlink/photo.jpg", "~/photo.jpg"):
        r = svc.client.post("/api/ops/segment-image/validate", json={"image": image})
        body = r.json()
        assert not body["valid"], image
        assert fields(body) == ["image"], (image, body)
        assert "outside the workspace" in body["by_parameter"]["image"][0], image
    for m in ("../m", "evil", "inputs", "/tmp"):
        r = svc.client.post("/api/ops/segment-map/validate", json={"map": m})
        assert fields(r.json()) == ["map"], m
    r = svc.client.post("/api/ops/mapper-update/validate",
                        json={"inputs": ["inputs/ok.jpg", "inputs/link.jpg"], "map": "new"})
    assert fields(r.json()) == ["inputs"]
    # inside the workspace, absolute or relative, is fine
    for image in ("inputs/ok.jpg", str(svc.ws.root / "inputs" / "ok.jpg")):
        r = svc.client.post("/api/ops/segment-image/validate", json={"image": image})
        assert r.json()["by_parameter"].get("image") is None, r.json()
    # map files never leave the map, nor show its hidden entries
    minimal_map(svc.ws.maps / "m")
    assert svc.client.get("/api/maps/m/files/map.json").status_code == 200
    for rel in ("../evil/photo.jpg", ".staging/x", "%2e%2e/%2e%2e/server.lock"):
        assert svc.client.get(f"/api/maps/m/files/{rel}").status_code == 404, rel
    assert svc.client.get("/api/maps/evil").status_code == 404


def test_inference_operations_get_503_while_read_only_ones_work(svc: Svc) -> None:
    sh("start_inference_server.sh", "--stop")  # no inference server in this test
    jpeg(svc.ws.root / "inputs" / "ok.jpg")
    health = svc.client.get("/api/health").json()
    assert health["status"] == "ok"
    assert health["inference"]["status"] == "down"
    assert health["inference"]["start_command"] == "./start_inference_server.sh"
    for op, params in (("reconstruct", {"image": "inputs/ok.jpg"}),
                       ("segment-image", {"image": "inputs/ok.jpg"}),
                       ("view-image", {"image": "inputs/ok.jpg"}),
                       ("mapper-update", {"inputs": ["inputs/ok.jpg"], "map": "new"})):
        r = svc.client.post(f"/api/ops/{op}", json=params)
        assert r.status_code == 503, (op, r.json())
        err = r.json()["error"]
        assert err["code"] == "server_unavailable" and err["exit_code"] == 3
        assert "./start_inference_server.sh" in err["message"]
    assert not (svc.ws.maps / "new").exists()
    minimal_map(svc.ws.maps / "m")
    r = svc.client.post("/api/ops/segment-map", json={"map": "m", "artifacts": "files"})
    assert r.status_code == 202, r.json()
    job = svc.runner.wait(r.json()["id"], 120)
    assert job.state == "succeeded", job.log_tail
    assert svc.client.get(f"/api/jobs/{job.id}/result").status_code == 200
    names = {f["path"] for f in svc.client.get(f"/api/jobs/{job.id}/files").json()}
    assert {"files/segmentation.json", "files/catalog.csv", "result.json"} <= names
    maps = svc.client.get("/api/maps").json()
    assert [m["name"] for m in maps] == ["m"] and maps[0]["update_count"] == 1


# -- uploads -----------------------------------------------------------------------------------------


def test_upload_is_deleted_when_its_job_ends(svc: Svc) -> None:
    minimal_map(svc.ws.maps / "m")
    r = svc.client.post("/api/uploads?name=photo.jpg", content=b"not really a jpeg")
    assert r.status_code == 201
    up = r.json()
    assert up["path"] == f"uploads/{up['id']}/photo.jpg" and up["size"] == 17
    assert (svc.ws.root / up["path"]).read_bytes() == b"not really a jpeg"
    assert [u["id"] for u in svc.client.get("/api/uploads").json()] == [up["id"]]
    # a job that consumes it (fails: not an image) — whatever the state, the upload goes
    runner = svc.runner
    op = web_ops.operations()["segment-image"]
    prep = web_ops.prepare(op, {"image": up["path"]}, svc.ws, svc.ws.job_dir("x"))
    assert prep.uploads == [up["id"]] and not prep.problems
    slow = _slow_op(inference=False)
    job = runner.submit(slow, {"image": up["path"]}, Prepared(argv=["0.2", "2"], uploads=[
        up["id"]]), runner.new_id())
    # a second job may not consume the same upload
    r = svc.client.post("/api/ops/segment-image", json={"image": up["path"]})
    assert r.status_code in (409, 503), r.json()
    job = runner.wait(job.id, 60)
    assert job.state == "failed" and job.error["message"] == "asked to fail"
    assert not (svc.ws.uploads / up["id"]).exists()
    r = svc.client.post("/api/ops/segment-image/validate", json={"image": up["path"]})
    assert "upload the file again" in r.json()["by_parameter"]["image"][0]
    # a queued job that is cancelled releases its upload too
    up2 = svc.client.post("/api/uploads?name=b.jpg", content=b"x").json()
    blocker = runner.submit(_slow_op(), {}, Prepared(argv=["3"]), runner.new_id())
    queued = runner.submit(_slow_op(), {}, Prepared(argv=["0"], uploads=[up2["id"]]),
                           runner.new_id())
    assert runner.get(queued.id).state == "queued"
    assert svc.client.post(f"/api/jobs/{queued.id}/cancel").json()["state"] == "cancelled"
    assert not (svc.ws.uploads / up2["id"]).exists()
    svc.client.post(f"/api/jobs/{blocker.id}/cancel")
    # unconsumed uploads can be discarded; bad names are refused
    up3 = svc.client.post("/api/uploads?name=c.jpg", content=b"x").json()
    assert svc.client.delete(f"/api/uploads/{up3['id']}").status_code == 204
    for name in ("", "../x.jpg", ".hidden.jpg", "a/b.jpg"):
        assert svc.client.post(f"/api/uploads?name={name}", content=b"x").status_code == 400
    runner.wait(blocker.id, 30)
    assert svc.client.get("/api/uploads").json() == []


def test_an_interrupted_upload_is_deleted_at_once(svc: Svc) -> None:
    app = create_app(svc.service)
    messages = [{"type": "http.request", "body": b"first part", "more_body": True},
                {"type": "http.disconnect"}]
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return messages.pop(0)

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": "/api/uploads", "raw_path": b"/api/uploads",
             "query_string": b"name=big.mp4", "headers": [], "http_version": "1.1",
             "scheme": "http", "server": ("test", 80), "client": ("test", 1), "root_path": ""}
    asyncio.run(app(scope, receive, send))
    assert list(svc.ws.uploads.iterdir()) == []


def test_uploads_are_cleared_at_start_and_stop(ws: Workspace) -> None:
    uid, target = ws.new_upload("left.jpg")
    target.write_bytes(b"x")
    ws.clear_uploads()
    assert list(ws.uploads.iterdir()) == []
    runner = Runner(ws)
    uid, target = ws.new_upload("left.jpg")
    target.write_bytes(b"x")
    runner.shutdown()
    assert list(ws.uploads.iterdir()) == []


# -- jobs --------------------------------------------------------------------------------------------


def _slow_op(inference: bool = True, browser: bool = False) -> Any:
    return types.SimpleNamespace(id="slow", label="slow_command", module="tests.fakes.slow_command",
                                 program=types.SimpleNamespace(prog="slow_command"),
                                 uses_inference=inference, browser=browser)


def test_inference_jobs_run_one_at_a_time_in_order_others_meanwhile(svc: Svc) -> None:
    runner = svc.runner
    a = runner.submit(_slow_op(), {}, Prepared(argv=["1.5"], command=["a"]), runner.new_id())
    b = runner.submit(_slow_op(), {}, Prepared(argv=["0.3"], command=["b"]), runner.new_id())
    c = runner.submit(_slow_op(inference=False), {}, Prepared(argv=["0.3"]), runner.new_id())
    assert runner.get(a.id).state == "running"
    assert runner.get(b.id).state == "queued"
    assert runner.get(c.id).state == "running"  # read-only work is not queued behind inference
    assert svc.client.get("/api/health").json()["service"]["jobs"] == {"queued": 1, "running": 2}
    a, b, c = (runner.wait(j.id, 60) for j in (a, b, c))
    assert [j.state for j in (a, b, c)] == ["succeeded"] * 3
    assert b.started_at >= a.ended_at
    assert c.ended_at < a.ended_at
    assert a.stages[0]["stage"] == "setup" and a.stages[0]["seconds"] > 1.0
    assert a.progress == {"stage": "setup", "done": 10, "total": 10}
    assert a.log_tail[-1] == "slow_command: slept 1.5 s"
    assert svc.client.get(f"/api/jobs/{a.id}/log").text.endswith("slept 1.5 s\n")
    timings = svc.client.get(f"/api/jobs/{a.id}/timings")
    assert timings.status_code == 200


def test_two_jobs_never_write_the_same_map(svc: Svc) -> None:
    runner = svc.runner
    w1 = runner.submit(_slow_op(inference=False), {}, Prepared(argv=["1"], writes="/m"),
                       runner.new_id())
    w2 = runner.submit(_slow_op(inference=False), {}, Prepared(argv=["0.1"], writes="/m"),
                       runner.new_id())
    other = runner.submit(_slow_op(inference=False), {}, Prepared(argv=["0.1"], writes="/n"),
                          runner.new_id())
    assert [runner.get(j.id).state for j in (w1, w2, other)] == ["running", "queued", "running"]
    w1, w2 = runner.wait(w1.id, 60), runner.wait(w2.id, 60)
    assert w2.started_at >= w1.ended_at


def test_cancel_a_running_job_interrupts_the_command(svc: Svc) -> None:
    runner = svc.runner
    job = runner.submit(_slow_op(), {}, Prepared(argv=["30"]), runner.new_id())
    deadline = time.monotonic() + 30
    while runner.get(job.id).progress is None and time.monotonic() < deadline:
        time.sleep(0.05)  # the command is in its stage
    r = svc.client.post(f"/api/jobs/{job.id}/cancel")
    assert r.status_code == 200
    job = runner.wait(job.id, 30)
    assert job.state == "cancelled" and job.exit_code == 130  # the command's own Ctrl-C exit
    assert svc.client.post(f"/api/jobs/{job.id}/cancel").status_code == 409
    assert svc.client.get(f"/api/jobs/{job.id}/result").status_code == 404


def test_progress_events_stream_until_the_job_ends(svc: Svc) -> None:
    runner = svc.runner
    job = runner.submit(_slow_op(), {}, Prepared(argv=["0.5"]), runner.new_id())
    states, stages = [], set()
    with svc.client.stream("GET", f"/api/jobs/{job.id}/events") as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if line.startswith("data: "):
                ev = json.loads(line[6:])
                states.append(ev["state"])
                stages.add(ev["stage"])
    assert states[-1] == "succeeded" and "setup" in stages
    assert svc.client.get(f"/api/jobs/{job.id}").json()["state"] == "succeeded"
    assert svc.client.get("/api/jobs/missing").status_code == 404


def test_job_list_survives_a_restart(ws: Workspace) -> None:
    runner = Runner(ws)
    done = runner.submit(_slow_op(), {"x": 1}, Prepared(argv=["0"], command=["slow"]),
                         runner.new_id())
    runner.wait(done.id, 30)
    runner.shutdown()
    stale = Job(id="20000101-000000-abcdef", operation="slow", label="slow", params={},
                command=[], argv=[], module="m", prog="p", inference=True, browser=False,
                state="running")
    (ws.jobs / stale.id).mkdir()
    (ws.jobs / stale.id / "job.json").write_text(json.dumps(dataclasses.asdict(stale)))
    again = Runner(ws)
    again.load()
    assert again.get(done.id).state == "succeeded" and again.get(done.id).params == {"x": 1}
    assert again.get(stale.id).state == "cancelled"
    assert json.loads((ws.jobs / stale.id / "job.json").read_text())["state"] == "cancelled"


def test_resubmit_uses_the_same_parameters(svc: Svc) -> None:
    minimal_map(svc.ws.maps / "m")
    r = svc.client.post("/api/ops/segment-map", json={"map": "m", "format": "ply"})
    first = svc.runner.wait(r.json()["id"], 120)
    assert first.state == "succeeded", first.log_tail
    r = svc.client.post(f"/api/jobs/{first.id}/resubmit", json={"format": "json"})
    assert r.status_code == 202
    second = svc.runner.wait(r.json()["id"], 120)
    assert second.resubmitted_from == first.id and second.params == {"map": "m", "format": "json"}
    assert first.result_name == "result.ply" and second.result_name == "result.json"


def test_service_shutdown_cancels_running_and_queued_jobs(ws: Workspace) -> None:
    runner = Runner(ws, stop_grace_s=30)
    running = runner.submit(_slow_op(), {}, Prepared(argv=["30"]), runner.new_id())
    queued = runner.submit(_slow_op(), {}, Prepared(argv=["30"]), runner.new_id())
    time.sleep(1.0)
    runner.shutdown()
    assert runner.get(running.id).state == "cancelled"
    assert runner.get(queued.id).state == "cancelled"


def test_the_web_process_never_loads_torch_or_open3d(tmp_path: Path) -> None:
    import subprocess

    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from starlette.testclient import TestClient\n"
        "from oh_my_slam.web.app import Service, create_app\n"
        "from oh_my_slam.web.jobs import Runner\n"
        "from oh_my_slam.web.workspace import Workspace\n"
        "from tests.unit.test_view_cli import minimal_map\n"
        f"ws = Workspace(Path({str(tmp_path)!r}))\n"
        "ws.create()\n"
        "minimal_map(ws.maps / 'm')\n"
        "c = TestClient(create_app(Service(ws, Runner(ws))))\n"
        "for u in ('/api/health', '/api/openapi.json', '/api/maps', '/api/maps/m'):\n"
        "    assert c.get(u).status_code == 200, u\n"
        "c.post('/api/ops/segment-map/validate', json={'map': 'm'})\n"
        "print(sorted(m for m in ('torch', 'open3d', 'pycolmap') if m in sys.modules))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO,
                         timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


# -- viewer of a map ---------------------------------------------------------------------------------


def test_map_viewer_is_served_by_the_viewers_own_routes(svc: Svc,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """``/viewer/map/<name>/…`` hands the path below the prefix and the query to
    ``viewer.routes.ViewerRoutes`` built over the map's bundle (``viewer.bundle.map_bundle``)."""
    calls: list[tuple[str, str, str]] = []

    class FakeRoutes:
        def __init__(self, bundle: Any) -> None:
            self.bundle = bundle

        def handle(self, method: str, path: str, query: str = "") -> Any:
            calls.append((method, path, query))
            body = (b"<html>", memoryview(b"page</html>"))
            return types.SimpleNamespace(status=200, headers=(("Content-Type", "text/html"),),
                                         body=body)

    monkeypatch.setitem(sys.modules, "oh_my_slam.viewer.routes",
                        types.SimpleNamespace(ViewerRoutes=FakeRoutes))
    minimal_map(svc.ws.maps / "m")
    r = svc.client.get("/viewer/map/m", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/viewer/map/m/"
    r = svc.client.get("/viewer/map/m/api/cloud?voxel=0.1")
    assert r.status_code == 200 and r.content == b"<html>page</html>"
    assert calls == [("GET", "/api/cloud", "voxel=0.1")]
    assert svc.client.get("/viewer/map/absent/").status_code == 404
    assert svc.client.get("/viewer/job/none/").status_code == 404
