"""``server.sh`` API without inference (spec §2.6): operations and OpenAPI derived from the commands'
registry (a new option or mode appears with no web change), synchronous per-field validation,
workspace confinement (path escapes, symlinks and hidden entries refused), the Host / Origin /
content-type guard, uploads (lifecycle, size cap), the inference server down or loading, and the
viewer routes."""

from __future__ import annotations

import asyncio
import dataclasses
import os
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from starlette.testclient import TestClient

from oh_my_slam.commands import spec
from oh_my_slam.web import app as web_app
from oh_my_slam.web import operations as web_ops
from oh_my_slam.web.app import Service, create_app
from oh_my_slam.web.jobs import Runner
from oh_my_slam.web.openapi import document
from oh_my_slam.web.operations import Prepared, Step
from oh_my_slam.web.workspace import Workspace
from tests.fakes import slow_command
from tests.unit.test_view_cli import minimal_map, sh

REPO = Path(__file__).resolve().parents[2]
OCTET = {"content-type": "application/octet-stream"}


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


def make_svc(ws: Workspace, **kw: Any) -> Service:
    return Service(ws, Runner(ws, stop_grace_s=30), url="http://0.0.0.0:0/",
                   extra_hosts={"testserver"}, **kw)


@pytest.fixture
def svc(ws: Workspace) -> Iterator[Svc]:
    service = make_svc(ws)
    with TestClient(create_app(service)) as client:
        yield Svc(service, client)
    service.runner.shutdown()


def fields(body: dict[str, Any]) -> list[str]:
    """The parameters a validation flags (the inference server's state concerns none)."""
    return [k for k in body["by_parameter"] if k]


def jpeg(path: Path, seed: int = 5) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)).save(path, quality=95)
    return path


def slow_op(inference: bool = True) -> Any:
    return types.SimpleNamespace(id="slow", label="slow.sh")


def slow(seconds: float, *flags: str, inference: bool = True, **kw: Any) -> Prepared:
    """A job of the stand-in command (``tests.fakes.slow_command``)."""
    return Prepared(steps=[Step("slow.sh", slow_command.MODULE, [f"--seconds={seconds}", *flags])],
                    command=["slow.sh"], inference=inference, **kw)


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
        params = [p for p in d["parameters"] if p["service"]]  # command-line-only ones left out
        assert list(schema["properties"]) == [p["name"] for p in params]
        assert schema["required"] == [p["name"] for p in params if p["required"]]
        for p in params:
            prop = schema["properties"][p["name"]]
            assert prop["x-oms"] == p  # names, kinds, defaults, help, bounds: the registry's
            if p["choices"]:
                assert prop["enum"] == p["choices"]
        assert post["x-oms"] == d  # outputs, stages, rules, errors, inference need
        assert f"/api/ops/{op.id}/validate" in doc["paths"]
        # single-image modes may ask for their viewer
        takes_image = any(p["kind"] == "image" for p in d["parameters"])
        assert bool(post["parameters"]) == (takes_image and op.label.split()[0] != "view.sh")
    assert doc["x-oms"]["exit_codes"] == spec.describe()["exit_codes"]
    view_props = doc["paths"]["/api/ops/view-map"]["post"]["requestBody"]["content"][
        "application/json"]["schema"]["properties"]
    assert "no_browser" not in view_props  # spec marks it as meaning nothing to the service
    assert {"/api/maps/{name}/viewer/{path}", "/api/jobs/{id}/viewer/{path}"} <= set(doc["paths"])
    assert {op.id for op in ops.values()} == {"reconstruct", "mapper-update", "mapper-locate",
                                              "segment-image", "segment-map", "view-image",
                                              "view-map"}


def test_each_operation_runs_the_commands_own_entry_point(ws: Workspace) -> None:
    import importlib.util

    jpeg(ws.root / "in" / "a.jpg")
    minimal_map(ws.maps / "m")
    for op in web_ops.operations().values():
        assert importlib.util.find_spec(op.module) is not None, op.module
        assert f"oms_exec {op.module.rsplit('.', 1)[1]}" in (REPO / op.program.prog).read_text()
    seg = web_ops.prepare(web_ops.operations()["segment-image"], {"image": "in/a.jpg"}, ws,
                          ws.job_dir("j"))
    assert [s.module for s in seg.steps] == ["oh_my_slam.cli.segment"]
    assert seg.steps[0].argv == [f"-i={(ws.root / 'in' / 'a.jpg')}",
                                 f"-o={ws.job_dir('j') / 'out' / 'result.json'}"]
    # a browser mode runs the viewer step on view.sh's own command line; it saves into viewer/
    view = web_ops.prepare(web_ops.operations()["view-map"], {"map": "m"}, ws, ws.job_dir("j"))
    assert [s.module for s in view.steps] == ["oh_my_slam.cli.view_save"]
    assert view.steps[0].argv == [str(ws.job_dir("j") / "viewer"), "view.sh",
                                  f"-m={ws.maps / 'm'}"]
    assert view.viewer and not view.inference and view.steps[0].env == {}
    refused = web_ops.prepare(web_ops.operations()["view-map"], {"map": "m", "no_browser": True},
                              ws, ws.job_dir("j"))
    assert [p.parameters for p in refused.problems] == [("no_browser",)]
    # an image request with its viewer: the command records its inference, the viewer replays it
    both = web_ops.prepare(web_ops.operations()["segment-image"],
                           {"image": "in/a.jpg", "min_score": 0.7}, ws, ws.job_dir("j"),
                           viewer=True)
    assert [s.module for s in both.steps] == ["oh_my_slam.cli.segment", "oh_my_slam.cli.view_save"]
    rec = str(ws.job_dir("j") / "inference")
    assert both.steps[0].env == {"OH_MY_SLAM_INFERENCE_RECORD": rec}
    assert both.steps[1].env == {"OH_MY_SLAM_INFERENCE_REPLAY": rec}
    assert both.steps[1].argv == [str(ws.job_dir("j") / "viewer"), "segment.sh",
                                  *both.steps[0].argv]  # the command's own options, --min-score
    assert "--min-score=0.7" in both.steps[1].argv
    none = web_ops.prepare(web_ops.operations()["segment-map"], {"map": "m"}, ws,
                           ws.job_dir("j"), viewer=True)
    assert none.problems and "no single image" in none.problems[0].message


def test_a_new_option_and_a_new_mode_reach_the_api_without_web_changes(
        monkeypatch: pytest.MonkeyPatch, svc: Svc) -> None:
    """Monkeypatch the registry: an extra option on segment.sh and a whole new command appear in
    the OpenAPI document and are accepted by the API, with no change to oh_my_slam.web."""
    seg = spec.SEGMENT.command()
    extra = spec.Option("--shade", "shade", spec.Kind.ENUM, "a new option", default="dark",
                        choices=("dark", "light"))
    seg2 = dataclasses.replace(seg, options=(*seg.options, extra))
    programs = (*[p for p in spec.PROGRAMS if p is not spec.SEGMENT],
                dataclasses.replace(spec.SEGMENT, commands=(seg2,)),
                slow_command.registry_program())
    monkeypatch.setattr(spec, "PROGRAMS", programs)
    monkeypatch.setattr(web_ops.Operation, "module", property(
        lambda op: slow_command.MODULE if op.program.prog == "slow.sh"
        else f"oh_my_slam.cli.{op.program.prog.removesuffix('.sh')}"))
    svc.service.ops = web_ops.operations()

    doc = svc.client.get("/api/openapi.json").json()
    props = doc["paths"]["/api/ops/segment-map"]["post"]["requestBody"]["content"][
        "application/json"]["schema"]["properties"]
    assert props["shade"]["enum"] == ["dark", "light"] and props["shade"]["default"] == "dark"
    assert "/api/ops/slow" in doc["paths"]

    minimal_map(svc.ws.maps / "m")
    r = svc.client.post("/api/ops/segment-map/validate", json={"map": "m", "shade": "light"})
    assert r.json()["valid"], r.json()
    assert "--shade=light" in r.json()["command"]
    bad = svc.client.post("/api/ops/segment-map/validate", json={"map": "m", "shade": "blue"})
    assert fields(bad.json()) == ["shade"]
    r = svc.client.post("/api/ops/slow", json={"seconds": 0.2})
    assert r.status_code == 202, r.json()
    assert r.json()["command"] == ["slow.sh", "--seconds=0.2"] and not r.json()["inference"]
    job = svc.runner.wait(r.json()["id"], 60)
    assert job.state == "succeeded" and job.stages[0]["stage"] == "setup"


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
    assert r.status_code == 400 and fields(r.json()["error"]) == ["artifacts"]
    # the result file and the -d folder never share a name
    r = svc.client.post("/api/ops/segment-map", json={"map": "m", "artifacts": "result.json"})
    assert r.status_code == 400 and fields(r.json()["error"]) == ["artifacts"]
    r = svc.client.post("/api/ops/segment-map", json={"map": "m", "artifacts": "x",
                                                      "output": "x"})
    assert r.status_code == 400 and fields(r.json()["error"]) == ["artifacts"]
    r = svc.client.post("/api/ops/segment-map", content=b"[1, 2]",
                        headers={"content-type": "application/json"})
    assert r.status_code == 400
    r = svc.client.post("/api/ops/nope", json={})
    assert r.status_code == 404
    assert svc.client.get("/api/jobs").json() == []
    assert not list(svc.ws.jobs.iterdir())


def test_paths_outside_the_workspace_are_refused(svc: Svc, tmp_path: Path) -> None:
    outside = jpeg(tmp_path / "outside" / "photo.jpg")
    jpeg(svc.ws.root / "inputs" / "ok.jpg")
    jpeg(svc.ws.root / "inputs" / ".hidden" / "h.jpg")
    (svc.ws.uploads / "abc").mkdir()
    (svc.ws.uploads / "abc" / ".x.jpg.part").write_bytes(b"arriving")
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
    for image in ("inputs/.hidden/h.jpg", "uploads/abc/.x.jpg.part"):
        r = svc.client.post("/api/ops/segment-image/validate", json={"image": image})
        assert "hidden entries" in r.json()["by_parameter"]["image"][0], image
    for m in ("../m", "evil", "inputs", "/tmp", ".staging"):
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


def test_host_origin_and_content_type_are_checked(ws: Workspace) -> None:
    """CSRF and DNS rebinding: a foreign Host is refused for every request; a state-changing one
    needs no foreign Origin and a non-form content type (uploads and cancel included)."""
    service = make_svc(ws)
    minimal_map(ws.maps / "m")
    with TestClient(create_app(service)) as c:
        assert c.get("/api/health").status_code == 200
        assert c.get("/api/health", headers={"host": "evil.example:80"}).status_code == 403
        assert c.get("/api/health", headers={"host": "127.0.0.1:1234"}).status_code == 200
        assert c.get("/api/health", headers={"host": "[::1]:1234"}).status_code == 200
        assert c.get("/api/health", headers={"host": "localhost"}).status_code == 200
        foreign = {"origin": "http://evil.example"}
        r = c.post("/api/ops/segment-map", json={"map": "m"}, headers=foreign)
        assert r.status_code == 403
        assert c.post("/api/uploads?name=a.jpg", content=b"x",
                      headers={**OCTET, **foreign}).status_code == 403
        assert c.post("/api/jobs/x/cancel", json={}, headers=foreign).status_code == 403
        assert c.delete("/api/uploads/x", headers=foreign).status_code == 403
        # a form post (what a cross-site page can send without a preflight) is refused
        for ctype in ("text/plain", "application/x-www-form-urlencoded", "multipart/form-data"):
            r = c.post("/api/ops/segment-map", content=b'{"map": "m"}',
                       headers={"content-type": ctype})
            assert r.status_code == 415, ctype
            r = c.post("/api/uploads?name=a.jpg", content=b"x", headers={"content-type": ctype})
            assert r.status_code == 415, ctype
        assert c.post("/api/jobs/x/cancel").status_code == 415
        assert not list(ws.uploads.iterdir()) and not list(ws.jobs.iterdir())
        assert c.post("/api/uploads?name=a.jpg", content=b"x").status_code == 415  # no type
        # the same origin is fine
        r = c.post("/api/ops/segment-map/validate", json={"map": "m"},
                   headers={"origin": "http://testserver"})
        assert r.status_code == 200
        # the full origin counts: another port or scheme of this machine is another site
        for other in ("http://testserver:9999", "https://testserver", "http://127.0.0.1:8000"):
            r = c.post("/api/ops/segment-map/validate", json={"map": "m"},
                       headers={"origin": other})
            assert r.status_code == 403, other
        # an upload may carry the file's own media type
        r = c.post("/api/uploads?name=a.jpg", content=b"x", headers={"content-type": "image/jpeg"})
        assert r.status_code == 201
    service.runner.shutdown()
    assert {"localhost", "127.0.0.1", "::1"} <= web_app.machine_hosts()


def test_an_unknown_host_refreshes_the_machines_names(ws: Workspace,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    names = {"localhost"}
    monkeypatch.setattr(web_app, "machine_hosts", lambda: set(names))
    service = make_svc(ws)
    assert service.knows_host("localhost") and not service.knows_host("new.local")
    names.add("new.local")  # e.g. joined another network
    assert not service.knows_host("new.local")  # rate-limited
    service._hosts_at -= web_app.HOSTS_REFRESH_S + 1
    assert service.knows_host("new.local")
    service.runner.shutdown()


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
    # view.sh -m saves its viewer without inference, served in-process from the map
    r = svc.client.post("/api/ops/view-map", json={"map": "m"})
    assert r.status_code == 202, r.json()
    job = svc.runner.wait(r.json()["id"], 120)
    assert job.state == "succeeded" and job.viewer == f"/viewer/job/{job.id}/", job.log_tail


def test_submissions_wait_for_a_loading_inference_server(monkeypatch: pytest.MonkeyPatch) -> None:
    from oh_my_slam.client.client import InferenceClient

    state = {"status": "loading"}
    monkeypatch.setattr(InferenceClient, "health", lambda self, timeout=1.0: types.SimpleNamespace(
        status=state["status"], models={}))
    assert web_app.inference_problem() is None  # the command waits for the models itself
    state["status"] = "ready"
    assert web_app.inference_problem() is None
    state["status"] = "error"
    p = web_app.inference_problem()
    assert p is not None and p.exit_code == 3


def test_a_locate_that_needs_no_inference_skips_the_inference_queue(ws: Workspace) -> None:
    minimal_map(ws.maps / "m")
    loc = spec.MAPPER.command("locate")
    args = spec.parse(loc, loc.modes[0], {"inputs": ["a.jpg"], "map": str(ws.maps / "m")})
    assert spec.needs_inference(loc.modes[0], args) is False  # 0 keyframes: matched exhaustively
    big = dataclasses.replace(loc.modes[0], inference_condition={"map_keyframes_greater_than": -1})
    assert spec.needs_inference(big, args) is True
    odd = dataclasses.replace(loc.modes[0], inference_condition={"unknown_condition": 1})
    assert spec.needs_inference(odd, args) is True  # what cannot be evaluated needs it
    assert spec.needs_inference(spec.SEGMENT_MAP, args) is False
    assert spec.needs_inference(spec.SEGMENT_IMAGE, args) is True


# -- uploads -----------------------------------------------------------------------------------------


def test_upload_is_deleted_when_its_job_ends(svc: Svc) -> None:
    r = svc.client.post("/api/uploads?name=photo.jpg", content=b"not really a jpeg",
                        headers=OCTET)
    assert r.status_code == 201
    up = r.json()
    assert up["path"] == f"uploads/{up['id']}/photo.jpg" and up["size"] == 17
    assert (svc.ws.root / up["path"]).read_bytes() == b"not really a jpeg"
    assert [u["id"] for u in svc.client.get("/api/uploads").json()] == [up["id"]]
    op = web_ops.operations()["segment-image"]
    prep = web_ops.prepare(op, {"image": up["path"]}, svc.ws, svc.ws.job_dir("x"))
    assert prep.uploads == [up["id"]] and not prep.problems
    runner = svc.runner
    job = runner.submit(slow_op(), {}, slow(0.2, "--code=2", uploads=[up["id"]]),
                        runner.new_id())
    # a second job may not consume the same upload
    with pytest.raises(Exception, match="already the input of job"):
        runner.submit(slow_op(), {}, slow(0, uploads=[up["id"]]), runner.new_id())
    job = runner.wait(job.id, 60)
    assert job.state == "failed" and job.error["message"] == "asked to fail"
    assert not (svc.ws.uploads / up["id"]).exists()  # a failed job's upload goes too
    r = svc.client.post("/api/ops/segment-image/validate", json={"image": up["path"]})
    assert "upload the file again" in r.json()["by_parameter"]["image"][0]
    # a queued job that is cancelled releases its upload too
    up2 = svc.client.post("/api/uploads?name=b.jpg", content=b"x", headers=OCTET).json()
    blocker = runner.submit(slow_op(), {}, slow(3), runner.new_id())
    queued = runner.submit(slow_op(), {}, slow(0, uploads=[up2["id"]]), runner.new_id())
    assert runner.get(queued.id).state == "queued"
    assert svc.client.post(f"/api/jobs/{queued.id}/cancel", json={}).json()["state"] \
        == "cancelled"
    assert not (svc.ws.uploads / up2["id"]).exists()
    svc.client.post(f"/api/jobs/{blocker.id}/cancel", json={})
    # unconsumed uploads can be discarded; bad names are refused
    up3 = svc.client.post("/api/uploads?name=c.jpg", content=b"x", headers=OCTET).json()
    assert svc.client.delete(f"/api/uploads/{up3['id']}").status_code == 204
    for name in ("", "../x.jpg", ".hidden.jpg", "a/b.jpg"):
        r = svc.client.post(f"/api/uploads?name={name}", content=b"x", headers=OCTET)
        assert r.status_code == 400, name
    runner.wait(blocker.id, 30)
    assert svc.client.get("/api/uploads").json() == []


def test_uploads_are_capped_and_need_free_space(ws: Workspace) -> None:
    service = make_svc(ws, max_upload_bytes=1000)
    with TestClient(create_app(service)) as c:
        assert c.post("/api/uploads?name=a.mp4", content=b"x" * 1000,
                      headers=OCTET).status_code == 201
        r = c.post("/api/uploads?name=b.mp4", content=b"x" * 1001, headers=OCTET)
        assert r.status_code == 413 and r.json()["error"]["code"] == "too_large"

        def chunks() -> Iterator[bytes]:  # no Content-Length: refused as it grows
            yield from (b"x" * 600, b"x" * 600)

        r = c.post("/api/uploads?name=c.mp4", content=chunks(), headers=OCTET)
        assert r.status_code == 413
        service.max_upload_bytes = 1 << 40
        service.min_free_bytes = 1 << 62  # more than any disk has free
        r = c.post("/api/uploads?name=d.mp4", content=b"x", headers=OCTET)
        assert r.status_code == 413 and r.json()["error"]["code"] == "insufficient_storage"
    assert [u.name for u in ws.list_uploads()] == ["a.mp4"]
    service.runner.shutdown()


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
             "query_string": b"name=big.mp4", "http_version": "1.1",
             "headers": [(b"host", b"localhost"), (b"content-type", b"application/octet-stream")],
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
        "c = TestClient(create_app(Service(ws, Runner(ws), extra_hosts={'testserver'})))\n"
        "for u in ('/api/health', '/api/openapi.json', '/api/maps', '/api/maps/m',\n"
        "          '/viewer/map/m/', '/api/maps/m/viewer/api/meta'):\n"
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
    """``/viewer/map/<name>/…`` and ``/api/maps/<name>/viewer/…`` hand the path below the prefix
    and the query to ``viewer.routes.ViewerRoutes`` over the map's bundle; loaded bundles are
    kept in a small LRU."""
    calls: list[tuple[str, str, str]] = []
    built: list[Any] = []

    class FakeRoutes:
        def __init__(self, bundle: Any) -> None:
            built.append(bundle)

        def handle(self, method: str, path: str, query: str = "") -> Any:
            calls.append((method, path, query))
            body = (b"<html>", memoryview(b"page</html>"))
            return types.SimpleNamespace(status=200, headers=(("Content-Type", "text/html"),),
                                         body=body)

    monkeypatch.setitem(sys.modules, "oh_my_slam.viewer.routes",
                        types.SimpleNamespace(ViewerRoutes=FakeRoutes))
    monkeypatch.setattr("oh_my_slam.viewer.bundle.map_bundle", lambda root: root)
    for n in ("a", "b", "c"):
        minimal_map(svc.ws.maps / n)
    r = svc.client.get("/viewer/map/a", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/viewer/map/a/"
    r = svc.client.get("/viewer/map/a/api/cloud?voxel=0.1")
    assert r.status_code == 200 and r.content == b"<html>page</html>"
    assert svc.client.get("/api/maps/a/viewer/api/meta").status_code == 200
    assert calls == [("GET", "/api/cloud", "voxel=0.1"), ("GET", "/api/meta", "")]
    assert len(built) == 1  # one bundle for both URLs
    for n in ("b", "c", "a"):
        svc.client.get(f"/viewer/map/{n}/")
    assert len(built) == 4  # at most two kept: a was evicted by b and c
    assert svc.client.get("/viewer/map/absent/").status_code == 404
    assert svc.client.get("/viewer/job/none/").status_code == 404
    assert svc.client.get("/api/jobs/none/viewer/").status_code == 404
