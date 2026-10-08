"""``server.sh`` API without inference (spec §2.6): the operations and the OpenAPI document derived
from the commands' registry — every mode of the programs it offers (reconstruct.sh, mapper.sh,
segment.sh; not view.sh), without the options that only choose where the command writes — so a new
option or mode appears with no web change; exactly the routes the spec lists; synchronous
per-field validation; workspace confinement (path escapes, symlinks and hidden entries refused);
the Host / Origin / content-type guard; uploads (lifecycle, size cap); the inference server down
or loading."""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
import os
import re
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
from oh_my_slam.web.openapi import document
from oh_my_slam.web.runner import Runner
from oh_my_slam.web.workspace import Workspace
from tests.fakes import slow_command
from tests.unit.test_view_cli import minimal_map, sh

REPO = Path(__file__).resolve().parents[2]
OCTET = {"content-type": "application/octet-stream"}
OFFERED = {"reconstruct", "mapper-update", "mapper-locate", "segment"}
API_ROUTES = {"GET /api/health", "GET /api/openapi.json", "POST /api/ops/{op}",
              "POST /api/ops/{op}/validate", "POST /api/uploads", "PUT /api/uploads",
              "DELETE /api/uploads/{id}", "GET /api/maps", "GET /api/maps/{name}"}


@pytest.fixture(autouse=True)
def _repo_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Commands run in the workspace; the test helpers (tests.fakes) must import there."""
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
        filter(None, [str(REPO), os.environ.get("PYTHONPATH")])))


@pytest.fixture(autouse=True)
def _test_client_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """The test client's ``Host`` (``testserver``) names this machine, as a browser's would."""
    names = web_app.machine_hosts
    monkeypatch.setattr(web_app, "machine_hosts", lambda: names() | {"testserver"})


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
    return Service(ws, Runner(ws, interrupt_grace_s=5), url="http://0.0.0.0:0/", **kw)


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


def with_programs(monkeypatch: pytest.MonkeyPatch, service: Service, *programs: Any) -> None:
    """Add stand-in programs (``tests.fakes.slow_command``) to the registry and the service."""
    monkeypatch.setattr(spec, "PROGRAMS", (*spec.PROGRAMS, *programs))
    fakes = {p.prog for p in programs}
    monkeypatch.setattr(web_ops.Operation, "module", property(
        lambda op: slow_command.MODULE if op.program.prog in fakes
        else f"oh_my_slam.cli.{op.program.prog.removesuffix('.sh')}"))
    service.ops = web_ops.operations()


def api_routes(service: Service) -> set[str]:
    """``METHOD /path`` of every route of the app under ``/api/`` (HEAD aside)."""
    app = inspect.getclosurevars(create_app(service)).nonlocals["app"]  # inside the guard
    return {f"{m} " + re.sub(r"\{(\w+):\w+\}", r"{\1}", r.path)
            for r in app.routes for m in r.methods - {"HEAD"} if r.path.startswith("/api/")}


# -- single source of truth --------------------------------------------------------------------------


def test_the_routes_are_exactly_the_specs(svc: Svc) -> None:
    """Spec §2.6 "API": health, the OpenAPI document, uploads (create, discard), maps (list, one)
    and the operations with their validation — no job, download or viewer endpoint."""
    assert api_routes(svc.service) == API_ROUTES
    doc = svc.client.get("/api/openapi.json").json()
    fixed = {p for p in doc["paths"] if not p.startswith("/api/ops/")}
    assert fixed == {"/api/health", "/api/uploads", "/api/uploads/{id}", "/api/maps",
                     "/api/maps/{name}"}
    for gone in ("/api/jobs", "/api/jobs/x", "/viewer/map/m/", "/api/maps/m/files/map.json",
                 "/api/maps/m/viewer/api/meta", "/api/display-transform"):
        assert svc.client.get(gone).status_code == 404, gone


def test_one_operation_per_mode_of_the_offered_commands() -> None:
    ops = web_ops.operations()
    assert set(ops) == OFFERED  # view.sh is a command only
    assert {p.prog for p in spec.PROGRAMS if not p.service} == {"view.sh"}
    described = {d["id"]: d for d in spec.describe()["operations"]}
    doc = document(ops)
    assert {p.removeprefix("/api/ops/").removesuffix("/validate") for p in doc["paths"]
            if p.startswith("/api/ops/")} == OFFERED
    for op in ops.values():
        d = described[op.label]
        post = doc["paths"][f"/api/ops/{op.id}"]["post"]
        schema = post["requestBody"]["content"]["application/json"]["schema"]
        # every option but those that only choose where the command writes (-o, -d)
        params = [p for p in d["parameters"] if p["kind"] not in ("file_out", "folder_out")]
        assert list(schema["properties"]) == [p["name"] for p in params]
        assert not {"output", "artifacts"} & set(schema["properties"])
        assert schema["required"] == [p["name"] for p in params if p["required"]]
        for p in params:
            prop = schema["properties"][p["name"]]
            assert prop["x-oms"]["name"] == p["name"] and prop["x-oms"]["help"] == p["help"]
            if p["choices"]:
                assert prop["enum"] == p["choices"]
        oms = post["x-oms"]
        assert oms == op.entry(d) and oms["stages"] == d["stages"]
        assert {o["via"] for o in oms["outputs"]} <= {"stdout", "-m"}  # no -d artefacts
        assert all(r["parameters"] and set(r["parameters"]) <= set(schema["properties"])
                   for r in oms["rules"])
        ok = post["responses"]["200"]
        assert "Server-Timing" in ok["headers"]
        assert set(ok["content"]) == {o["media_type"] for o in d["outputs"]
                                      if o["via"] == "stdout"}
        assert f"/api/ops/{op.id}/validate" in doc["paths"]
        assert "parameters" not in post  # no ?viewer
    assert doc["x-oms"]["exit_codes"] == spec.describe()["exit_codes"]
    assert set(doc["paths"]["/api/uploads"]) == {"post", "put"}  # put: curl -T
    for op in ("reconstruct", "mapper-update", "mapper-locate"):  # one -p: a string, no list
        attrs = doc["paths"][f"/api/ops/{op}"]["post"]["requestBody"]["content"][
            "application/json"]["schema"]["properties"]["attrs"]
        assert attrs["type"] == "string" and "oneOf" not in attrs and "items" not in attrs


def test_each_operation_runs_the_commands_own_entry_point(ws: Workspace) -> None:
    import importlib.util

    jpeg(ws.root / "in" / "a.jpg")
    minimal_map(ws.maps / "m")
    for op in web_ops.operations().values():
        assert importlib.util.find_spec(op.module) is not None, op.module
        assert f"oms_exec {op.module.rsplit('.', 1)[1]}" in (REPO / op.program.prog).read_text()
    ops = web_ops.operations()
    seg = web_ops.prepare(ops["segment"], {"image": "in/a.jpg"}, ws)
    assert seg.argv == [f"-i={(ws.root / 'in' / 'a.jpg')}"]  # the result is its stdout: no -o
    assert seg.command == ["segment.sh", "-i=in/a.jpg"] and seg.result_format == "json"
    # each result in the media type of its format: the segmented image and the depth image are
    # PNGs, the point cloud a PLY
    for op, params, fmt, media in (
            ("segment", {"format": "png"}, "png", "image/png"),
            ("reconstruct", {"format": "depth"}, "png", "image/png"),
            ("reconstruct", {"format": "ply", "attrs": "normals=on"}, "ply",
             "application/octet-stream"),
            ("reconstruct", {}, "json", "application/json")):
        prep = web_ops.prepare(ops[op], {"image": "in/a.jpg", **params}, ws)
        assert prep.problems == [] and prep.inference, (op, params)
        assert (prep.result_format, web_ops.media_of(prep.result_format)) == (fmt, media)
    loc = web_ops.prepare(ops["mapper-locate"], {"inputs": ["in/a.jpg"], "map": "m"}, ws)
    assert loc.argv[-1] == f"-m={ws.maps / 'm'}" and loc.result_format == "json"
    assert not loc.inference  # a small map is matched exhaustively
    for name, value in (("output", "x.json"), ("artifacts", "files")):
        refused = web_ops.prepare(ops["segment"], {"image": "in/a.jpg", name: value}, ws)
        assert [p.parameters for p in refused.problems] == [(name,)]
        assert "chooses where the command writes" in refused.problems[0].message
    # segment.sh has no -m and no -p any more: the command's parser refuses them
    gone = web_ops.prepare(ops["segment"], {"image": "in/a.jpg", "map": "m",
                                                  "attrs": "voxel=1"}, ws)
    assert [p.parameters for p in gone.problems] == [("attrs", "map")]
    refused = web_ops.prepare(ops["reconstruct"], {"image": "in/a.jpg", "format": "depth",
                                                   "attrs": "voxel=1"}, ws)
    assert refused.problems[0].message == ("-p sets point-cloud attributes, which only the PLY "
                                           "output has: use -f ply")
    # one -p (spec §2.2): its key=value[,key=value…] text, never a list
    listed = web_ops.prepare(ops["reconstruct"], {"image": "in/a.jpg", "format": "ply",
                                                  "attrs": ["voxel=1", "normals=on"]}, ws)
    assert [(p.parameters, p.message) for p in listed.problems] == [
        (("attrs",), "-p takes one value")]


def test_a_new_option_and_a_new_mode_reach_the_api_without_web_changes(
        monkeypatch: pytest.MonkeyPatch, svc: Svc) -> None:
    """Monkeypatch the registry: an extra option on segment.sh and a whole new command appear in
    the OpenAPI document and are accepted by the API, with no change to oh_my_slam.web (the
    inference server counts as up: the check concerns the request)."""
    seg = spec.SEGMENT.command()
    extra = spec.Option("--shade", "shade", spec.Kind.ENUM, "a new option", default="dark",
                        choices=("dark", "light"))
    seg2 = dataclasses.replace(seg, options=(*seg.options, extra))
    monkeypatch.setattr(spec, "PROGRAMS", (
        *[p for p in spec.PROGRAMS if p is not spec.SEGMENT],
        dataclasses.replace(spec.SEGMENT, commands=(seg2,))))
    with_programs(monkeypatch, svc.service, slow_command.registry_program())
    monkeypatch.setattr(svc.service, "inference_check", lambda: None)

    doc = svc.client.get("/api/openapi.json").json()
    props = doc["paths"]["/api/ops/segment"]["post"]["requestBody"]["content"][
        "application/json"]["schema"]["properties"]
    assert props["shade"]["enum"] == ["dark", "light"] and props["shade"]["default"] == "dark"
    assert "/api/ops/slow" in doc["paths"]

    jpeg(svc.ws.root / "inputs" / "a.jpg")
    image = {"image": "inputs/a.jpg"}
    r = svc.client.post("/api/ops/segment/validate", json={**image, "shade": "light"})
    assert r.json()["valid"], r.json()
    assert "--shade=light" in r.json()["command"]
    bad = svc.client.post("/api/ops/segment/validate", json={**image, "shade": "blue"})
    assert fields(bad.json()) == ["shade"]
    v = svc.client.post("/api/ops/slow/validate", json={"seconds": 0.2}).json()
    assert v["command"] == ["slow.sh", "--seconds=0.2"] and v["inference"] is False
    r = svc.client.post("/api/ops/slow", json={"seconds": 0.2})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/json" and r.json()["slept"] == 0.2
    assert re.fullmatch(r"setup;dur=\d+\.\d, total;dur=\d+\.\d", r.headers["server-timing"])


# -- validation and the workspace --------------------------------------------------------------------


def test_invalid_requests_get_per_field_errors_and_run_nothing(svc: Svc) -> None:
    jpeg(svc.ws.root / "inputs" / "ok.jpg")
    image = {"image": "inputs/ok.jpg"}
    r = svc.client.post("/api/ops/segment", json={**image, "min_score": "abc",
                                                        "format": "xml"})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "usage" and err["exit_code"] == 2
    assert set(err["by_parameter"]) >= {"format", "min_score"}
    assert "invalid choice: 'xml'" in err["by_parameter"]["format"][0]  # argparse's message
    r = svc.client.post("/api/ops/segment", json={})
    assert r.status_code == 400 and "the following arguments are required: -i" in r.json()[
        "error"]["message"]
    (svc.ws.maps / "junk").mkdir()
    (svc.ws.maps / "junk" / "notes.txt").write_text("x")
    r = svc.client.post("/api/ops/mapper-update", json={"inputs": ["inputs/ok.jpg"],
                                                        "map": "junk"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "not_a_map"
    r = svc.client.post("/api/ops/segment", json={**image, "artifacts": "x",
                                                        "output": "x"})
    assert r.status_code == 400 and fields(r.json()["error"]) == ["artifacts", "output"]
    r = svc.client.post("/api/ops/segment", content=b"[1, 2]",
                        headers={"content-type": "application/json"})
    assert r.status_code == 400
    r = svc.client.post("/api/ops/nope", json={})
    assert r.status_code == 404
    r = svc.client.post("/api/ops/view-map", json={"map": "m"})
    assert r.status_code == 404 and "no operation view-map" in r.json()["error"]["message"]
    assert not svc.ws.requests.exists() or not list(svc.ws.requests.iterdir())


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
        r = svc.client.post("/api/ops/segment/validate", json={"image": image})
        body = r.json()
        assert not body["valid"], image
        assert fields(body) == ["image"], (image, body)
        assert "outside the workspace" in body["by_parameter"]["image"][0], image
    for image in ("inputs/.hidden/h.jpg", "uploads/abc/.x.jpg.part", ".requests/x/stdout"):
        r = svc.client.post("/api/ops/segment/validate", json={"image": image})
        assert "hidden entries" in r.json()["by_parameter"]["image"][0], image
    for m in ("../m", "evil", "inputs", "/tmp", ".staging"):
        r = svc.client.post("/api/ops/mapper-locate/validate",
                            json={"inputs": ["inputs/ok.jpg"], "map": m})
        assert fields(r.json()) == ["map"], m
    r = svc.client.post("/api/ops/mapper-update/validate",
                        json={"inputs": ["inputs/ok.jpg", "inputs/link.jpg"], "map": "new"})
    assert fields(r.json()) == ["inputs"]
    # inside the workspace, absolute or relative, is fine
    for image in ("inputs/ok.jpg", str(svc.ws.root / "inputs" / "ok.jpg")):
        r = svc.client.post("/api/ops/segment/validate", json={"image": image})
        assert r.json()["by_parameter"].get("image") is None, r.json()
    assert svc.client.get("/api/maps/evil").status_code == 404


def test_host_origin_and_content_type_are_checked(ws: Workspace) -> None:
    """CSRF and DNS rebinding: a foreign Host is refused for every request; a state-changing one
    needs no foreign Origin and a non-form content type (uploads included)."""
    service = make_svc(ws)
    minimal_map(ws.maps / "m")
    with TestClient(create_app(service)) as c:
        assert c.get("/api/health").status_code == 200
        assert c.get("/api/health", headers={"host": "evil.example:80"}).status_code == 403
        assert c.get("/api/health", headers={"host": "127.0.0.1:1234"}).status_code == 200
        assert c.get("/api/health", headers={"host": "[::1]:1234"}).status_code == 200
        assert c.get("/api/health", headers={"host": "localhost"}).status_code == 200
        # the address of the listening line, used as printed, and a page loaded from it
        printed = {"host": "0.0.0.0:1234"}
        assert c.get("/api/health", headers=printed).status_code == 200
        r = c.post("/api/ops/mapper-locate/validate", json={"inputs": ["a.jpg"], "map": "m"},
                   headers={**printed, "origin": "http://0.0.0.0:1234"})
        assert r.status_code == 200, r.text
        foreign = {"origin": "http://evil.example"}
        locate = {"inputs": ["a.jpg"], "map": "m"}
        r = c.post("/api/ops/mapper-locate", json=locate, headers=foreign)
        assert r.status_code == 403
        assert c.post("/api/uploads?name=a.jpg", content=b"x",
                      headers={**OCTET, **foreign}).status_code == 403
        assert c.delete("/api/uploads/x", headers=foreign).status_code == 403
        # a form post (what a cross-site page can send without a preflight) is refused
        for ctype in ("text/plain", "application/x-www-form-urlencoded", "multipart/form-data"):
            r = c.post("/api/ops/mapper-locate", content=json.dumps(locate).encode(),
                       headers={"content-type": ctype})
            assert r.status_code == 415, ctype
            r = c.post("/api/uploads?name=a.jpg", content=b"x", headers={"content-type": ctype})
            assert r.status_code == 415, ctype
        assert not list(ws.uploads.iterdir())
        assert c.post("/api/uploads?name=a.jpg", content=b"x").status_code == 415  # no type
        # the same origin is fine
        r = c.post("/api/ops/mapper-locate/validate", json=locate,
                   headers={"origin": "http://testserver"})
        assert r.status_code == 200
        # the full origin counts: another port or scheme of this machine is another site
        for other in ("http://testserver:9999", "https://testserver", "http://127.0.0.1:8000"):
            r = c.post("/api/ops/mapper-locate/validate", json=locate,
                       headers={"origin": other})
            assert r.status_code == 403, other
        # an upload may carry the file's own media type
        r = c.post("/api/uploads?name=a.jpg", content=b"x", headers={"content-type": "image/jpeg"})
        assert r.status_code == 201
        # curl -T: a PUT of the raw file, with no type of its own (a PUT always needs a CORS
        # preflight, which is never granted); a form is still no raw file
        r = c.put("/api/uploads?name=b.jpg", content=b"xy")
        assert r.status_code == 201 and r.json()["size"] == 2, r.text
        assert (ws.root / r.json()["path"]).read_bytes() == b"xy"
        for ctype in ("multipart/form-data", "application/x-www-form-urlencoded"):
            r = c.put("/api/uploads?name=c.jpg", content=b"x", headers={"content-type": ctype})
            assert r.status_code == 415 and r.json()["error"]["code"] == "unsupported_media_type"
        assert c.put("/api/uploads?name=d.jpg", content=b"x", headers=foreign).status_code == 403
        assert sorted(p.name for p in ws.uploads.glob("*/*")) == ["a.jpg", "b.jpg"]
    service.runner.shutdown()
    assert {"localhost", "127.0.0.1", "::1", "0.0.0.0"} <= web_app.machine_hosts()


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


def test_inference_operations_get_503_while_the_others_work(
        monkeypatch: pytest.MonkeyPatch, svc: Svc) -> None:
    sh("start_inference_server.sh", "--stop")  # no inference server in this test
    jpeg(svc.ws.root / "inputs" / "ok.jpg")
    health = svc.client.get("/api/health").json()
    assert health["status"] == "ok" and health["service"]["requests"] == {"running": 0,
                                                                          "waiting": 0}
    assert health["inference"]["status"] == "down"
    assert health["inference"]["start_command"] == "./start_inference_server.sh"
    up = svc.client.post("/api/uploads?name=b.jpg", headers=OCTET,  # a readable image
                         content=(svc.ws.root / "inputs" / "ok.jpg").read_bytes()).json()
    for op, params in (("reconstruct", {"image": "inputs/ok.jpg"}),
                       ("reconstruct", {"image": "inputs/ok.jpg", "format": "depth"}),
                       ("segment", {"image": up["path"]}),
                       ("segment", {"image": "inputs/ok.jpg", "format": "png"}),
                       ("mapper-update", {"inputs": ["inputs/ok.jpg"], "map": "new"})):
        r = svc.client.post(f"/api/ops/{op}", json=params)
        assert r.status_code == 503, (op, r.json())
        err = r.json()["error"]
        assert err["code"] == "server_unavailable" and err["exit_code"] == 3
        assert "./start_inference_server.sh" in err["message"]
    assert not (svc.ws.maps / "new").exists()
    assert not (svc.ws.uploads / up["id"]).exists()  # the refused request consumed its upload
    minimal_map(svc.ws.maps / "m")
    # mapper.sh locate on a small map needs no inference server: it is not refused
    v = svc.client.post("/api/ops/mapper-locate/validate",
                        json={"inputs": ["inputs/ok.jpg"], "map": "m"}).json()
    assert v["valid"] and v["inference"] is False, v
    # an operation that needs none runs, and answers with its result
    with_programs(monkeypatch, svc.service, slow_command.map_registry_program())
    r = svc.client.post("/api/ops/slow", json={"map": "m"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/json" and r.json()["slept"] == 0
    assert "setup;dur=" in r.headers["server-timing"]
    maps = svc.client.get("/api/maps").json()
    assert [m["name"] for m in maps] == ["m"] and maps[0]["update_count"] == 1
    assert "thumbnail" not in maps[0]  # no map-file download to show it with
    assert svc.client.get("/api/maps/m").json()["meta"]["update_count"] == 1


def test_requests_wait_for_a_loading_inference_server(monkeypatch: pytest.MonkeyPatch) -> None:
    from oh_my_slam.client.client import InferenceClient
    from oh_my_slam.client.protocol import Health

    state = {"status": "loading"}
    monkeypatch.setattr(InferenceClient, "health", lambda self, timeout=1.0: Health(
        status=state["status"]))  # type: ignore[arg-type]
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
    never = slow_command.registry_program().commands[0].modes[0]  # type: ignore[attr-defined]
    assert spec.needs_inference(never, args) is False
    assert spec.needs_inference(spec.SEGMENT_IMAGE, args) is True
    jpeg(ws.root / "q.jpg")
    op = web_ops.operations()["mapper-locate"]
    assert web_ops.prepare(op, {"inputs": ["q.jpg"], "map": "m"}, ws).inference is False


# -- uploads -----------------------------------------------------------------------------------------


def test_an_upload_is_consumed_by_the_request_it_is_given_to(
        monkeypatch: pytest.MonkeyPatch, svc: Svc) -> None:
    """Deleted when that request ends, whatever its outcome — done, failed, refused — but not by
    a validation; an unconsumed one can be discarded."""
    with_programs(monkeypatch, svc.service, slow_command.registry_program())
    c = svc.client

    def upload(name: str = "photo.jpg") -> dict[str, Any]:
        r = c.post(f"/api/uploads?name={name}", content=b"not really a jpeg", headers=OCTET)
        assert r.status_code == 201
        return dict(r.json())

    up = upload()
    assert up["path"] == f"uploads/{up['id']}/photo.jpg" and up["size"] == 17
    assert (svc.ws.root / up["path"]).read_bytes() == b"not really a jpeg"
    assert c.post("/api/ops/slow/validate", json={"image": up["path"]}).json()["valid"]
    assert (svc.ws.root / up["path"]).is_file()  # validating consumes nothing
    r = c.post("/api/ops/slow", json={"image": up["path"], "seconds": 0})
    assert r.status_code == 200 and not (svc.ws.uploads / up["id"]).exists()
    r = c.post("/api/ops/slow/validate", json={"image": up["path"]})
    assert "upload the file again" in r.json()["by_parameter"]["image"][0]
    up = upload()
    r = c.post("/api/ops/slow", json={"image": up["path"], "code": 2})
    assert r.status_code == 400 and r.json()["error"] == {
        "code": "usage", "exit_code": 2, "message": "asked to fail", "http_status": 400}
    assert not (svc.ws.uploads / up["id"]).exists()  # a failed request's upload goes too
    up = upload()
    r = c.post("/api/ops/slow", json={"image": up["path"], "seconds": "soon"})
    assert r.status_code == 400 and not (svc.ws.uploads / up["id"]).exists()  # and a refused one
    up = upload()
    assert c.delete(f"/api/uploads/{up['id']}").status_code == 204
    assert c.delete(f"/api/uploads/{up['id']}").status_code == 404
    for name in ("", "../x.jpg", ".hidden.jpg", "a/b.jpg"):
        r = c.post(f"/api/uploads?name={name}", content=b"x", headers=OCTET)
        assert r.status_code == 400, name
    assert list(svc.ws.uploads.iterdir()) == []
    assert list(svc.ws.requests.iterdir()) == []  # nor is any result kept


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
    assert [p.name for p in ws.uploads.glob("*/*")] == ["a.mp4"]
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


def test_uploads_and_request_folders_are_cleared_at_start_and_stop(ws: Workspace) -> None:
    uid, target = ws.new_upload("left.jpg")
    target.write_bytes(b"x")
    ws.request_dir("r1").mkdir(parents=True)
    ws.clear_uploads()
    ws.clear_requests()
    assert list(ws.uploads.iterdir()) == [] and list(ws.requests.iterdir()) == []
    runner = Runner(ws)
    uid, target = ws.new_upload("left.jpg")
    target.write_bytes(b"x")
    runner.shutdown()
    assert list(ws.uploads.iterdir()) == []


def test_the_web_process_never_loads_torch_or_open3d(tmp_path: Path) -> None:
    import subprocess
    import sys

    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from starlette.testclient import TestClient\n"
        "from oh_my_slam.web.app import Service, create_app\n"
        "from oh_my_slam.web.runner import Runner\n"
        "from oh_my_slam.web.workspace import Workspace\n"
        "from tests.unit.test_view_cli import minimal_map\n"
        f"ws = Workspace(Path({str(tmp_path)!r}))\n"
        "ws.create()\n"
        "minimal_map(ws.maps / 'm')\n"
        "c = TestClient(create_app(Service(ws, Runner(ws))), base_url='http://localhost')\n"
        "for u in ('/api/health', '/api/openapi.json', '/api/maps', '/api/maps/m'):\n"
        "    assert c.get(u).status_code == 200, u\n"
        "(ws.root / 'a.jpg').write_bytes(b'x')\n"
        "locate = {'inputs': ['a.jpg'], 'map': 'm'}\n"
        "assert c.post('/api/ops/mapper-locate/validate', json=locate).json()['valid']\n"
        "c.post('/api/ops/segment/validate', json={'image': 'a.jpg', 'format': 'png'})\n"
        "c.post('/api/ops/reconstruct', json={'image': 'a.jpg', 'format': 'depth'})\n"
        "print(sorted(m for m in ('torch', 'open3d', 'pycolmap') if m in sys.modules))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO,
                         timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


def test_the_service_never_deletes_a_map(ws: Workspace) -> None:
    """No request deletes or changes a map outside the mapping operation: no route deletes one,
    read-only requests (whatever their outcome) and a failing update leave it as it was, and so
    do a stop and a restart."""
    from tests.mapsnap import snapshot

    root = minimal_map(ws.maps / "m")
    before = snapshot(root)
    service = make_svc(ws)
    with TestClient(create_app(service)) as c:
        assert c.delete("/api/maps/m").status_code == 405
        assert c.delete("/api/uploads/m").status_code == 404
        (ws.root / "bad.jpg").write_bytes(b"not an image")
        for params in ({"map": "m"}, {"map": "maps/m", "format": "ply"}):
            r = c.post("/api/ops/mapper-locate", json={"inputs": ["bad.jpg"], **params})
            assert r.status_code == 400, r.text  # the command's own refusal (an unreadable image)
            assert "cannot read image" in r.json()["error"]["message"]
        r = c.post("/api/ops/mapper-update", json={"inputs": ["bad.jpg"], "map": "m"})
        assert r.status_code >= 400
        assert c.get("/api/maps").json()[0]["name"] == "m"
    service.runner.shutdown()
    make_svc(ws).runner.shutdown()
    assert snapshot(root) == before
    assert json.loads((root / "map.json").read_text())["update_count"] == 1


# -- the answers' shape ------------------------------------------------------------------------------


def test_an_unknown_route_or_method_answers_the_error_shape(svc: Svc) -> None:
    r = svc.client.get("/api/nope")
    assert r.status_code == 404 and r.json() == {"error": {
        "code": "not_found", "message": "no route GET /api/nope; see /api/openapi.json",
        "http_status": 404}}
    r = svc.client.get("/api/uploads")
    assert r.status_code == 405 and r.json()["error"]["code"] == "method_not_allowed"
    assert set(r.headers["allow"].split(", ")) == {"POST", "PUT"}
    assert "GET /api/uploads: the route takes" in r.json()["error"]["message"]


def test_a_crash_answers_the_internal_error_and_consumes_the_uploads(ws: Workspace) -> None:
    """Spec §2.6 "Errors", "Workspace": a check or a route that crashes — here on a corrupt
    ``map.json`` — answers the command's internal error (exit 1: 500 ``internal``, with its
    message as the command tells it) in the error shape of every other answer, and the request's
    uploads go all the same."""
    from jsonschema import Draft202012Validator

    bad = ws.maps / "bad"
    bad.mkdir()
    (bad / "map.json").write_text("{not json")
    photo = jpeg(ws.root / "inputs" / "q.jpg").read_bytes()
    service = make_svc(ws, inference_check=lambda: None)
    with TestClient(create_app(service), raise_server_exceptions=False) as c:
        doc = c.get("/api/openapi.json").json()
        error = Draft202012Validator({**doc, "$ref": "#/components/schemas/Error"})
        assert all("500" in route["responses"] for item in doc["paths"].values()
                   for route in item.values())

        def internal(r: Any) -> dict[str, Any]:
            assert r.status_code == 500 and r.headers["content-type"] == "application/json", r.text
            error.validate(r.json())
            err: dict[str, Any] = r.json()["error"]
            assert err["code"] == "internal" and err["http_status"] == 500, err
            return err

        up = c.put("/api/uploads?name=q.jpg", content=photo).json()
        locate = {"inputs": [up["path"]], "map": "bad"}
        v = c.post("/api/ops/mapper-locate/validate", json=locate).json()
        (p,) = v["problems"]
        assert not v["valid"] and p["rule"] == "existing_map" and p["exit_code"] == 1, v
        assert p["code"] == "internal" and p["message"].startswith("JSONDecodeError: ")
        assert v["by_parameter"] == {"map": [p["message"]]}
        assert (ws.root / up["path"]).is_file()  # validating consumes nothing
        err = internal(c.post("/api/ops/mapper-locate", json=locate))
        assert err["exit_code"] == 1 and err["message"] == p["message"]
        assert not (ws.uploads / up["id"]).exists()  # the refused request consumed its upload
        err = internal(c.get("/api/maps/bad"))
        assert err["message"].startswith("JSONDecodeError: ")
        assert c.get("/api/maps").json() == []  # the list leaves it out

        # a check that crashes outside the command's rules (here the inference server's)
        def crashes() -> None:
            raise RuntimeError("boom")

        service.inference_check = crashes  # type: ignore[assignment]
        up = c.put("/api/uploads?name=q.jpg", content=photo).json()
        v = c.post("/api/ops/segment/validate", json={"image": up["path"]}).json()
        assert v["problems"] == [{"rule": "internal", "parameters": [], "code": "internal",
                                  "message": "RuntimeError: boom", "exit_code": 1,
                                  "http_status": 500}]
        err = internal(c.post("/api/ops/segment", json={"image": up["path"]}))
        assert err["message"] == "RuntimeError: boom"
        assert list(ws.uploads.iterdir()) == []
    service.runner.shutdown()


def test_the_services_own_refusals_are_one_table(ws: Workspace,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """Every refusal that is no command's error answers with the code and HTTP status of
    ``web.app.refusals()`` (no status is written anywhere else), and the OpenAPI document exports
    that table and the upload limits."""

    service = make_svc(ws, max_upload_bytes=10)
    uid, target = ws.new_upload("a.jpg")
    target.write_bytes(b"x")
    got: dict[str, int] = {}
    octet = {"content-type": "application/octet-stream"}
    with TestClient(create_app(service)) as c:
        for r in (c.get("/api/maps/nope"), c.get("/api/uploads"),
                  c.get("/api/maps", headers={"host": "elsewhere.example"}),
                  c.post("/api/ops/reconstruct/validate", content=b"{}",
                         headers={"content-type": "text/plain"}),
                  c.post("/api/uploads?name=a.jpg", content=b"x" * 100, headers=octet)):
            got[r.json()["error"]["code"]] = r.status_code
        service.max_upload_bytes, service.min_free_bytes = 1 << 40, 1 << 62
        r = c.post("/api/uploads?name=a.jpg", content=b"x", headers=octet)
        got[r.json()["error"]["code"]] = r.status_code
        monkeypatch.setattr(service.runner, "in_use", lambda _uid: True)
        r = c.delete(f"/api/uploads/{uid}")
        got[r.json()["error"]["code"]] = r.status_code
        service.runner.stopping = True
        r = c.post("/api/ops/reconstruct", json={})
        got[r.json()["error"]["code"]] = r.status_code
        doc = c.get("/api/openapi.json").json()
    service.runner.shutdown()
    table = web_app.refusals()
    assert got == {code: r.http_status for code, r in table.items()}
    assert doc["x-oms"]["refusals"] == [dataclasses.asdict(r) for r in table.values()]
    assert doc["x-oms"]["limits"] == {"max_upload_bytes": web_app.MAX_UPLOAD_BYTES,
                                      "min_free_bytes": web_app.MIN_FREE_BYTES}
    assert f"an upload over {web_app.MAX_UPLOAD_BYTES / 2**30:g} GiB" == table["too_large"].when
    source = (REPO / "src" / "oh_my_slam" / "web" / "app.py").read_text()
    assert not re.search(r"_error\(\d", source)  # every status of a refusal is the table's


def test_every_json_answer_follows_its_schema_in_the_document(
        ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    """``/api/openapi.json`` gives the schema of every JSON answer — health (the inference server
    up or down), an upload, validation, the maps, and every error — and the service's answers
    follow them."""
    from jsonschema import Draft202012Validator

    from oh_my_slam.client.protocol import Health
    from oh_my_slam.web.runner import STOPPING, Outcome

    minimal_map(ws.maps / "m")
    health = {"status": "ready", "health": Health(status="ready").model_dump(),
              "start_command": None}
    service = make_svc(ws, inference_check=lambda: None, inference_health=lambda: health,
                       max_upload_bytes=10)
    with TestClient(create_app(service)) as c:
        doc = c.get("/api/openapi.json").json()

        def check(path: str, method: str, r: Any) -> None:
            responses = doc["paths"][path][method]["responses"]
            declared = responses.get(str(r.status_code)) or responses[f"{r.status_code // 100}XX"]
            if "$ref" in declared:
                declared = doc["components"]["responses"][declared["$ref"].rsplit("/", 1)[1]]
            schema = declared["content"]["application/json"]["schema"]
            Draft202012Validator({**doc, **schema}).validate(r.json())

        check("/api/health", "get", c.get("/api/health"))
        health = {"status": "down", "message": "not running", "start_command": "./x.sh"}
        check("/api/health", "get", c.get("/api/health"))
        put = c.put("/api/uploads?name=a.jpg", content=b"x")
        check("/api/uploads", "put", put)
        check("/api/uploads", "post", c.post("/api/uploads?name=a.jpg", content=b"x" * 11,
                                             headers=OCTET))
        image = {"image": put.json()["path"]}
        for body in (image, {}, {"image": "../x.jpg", "format": "xml"}):
            check("/api/ops/segment/validate", "post",
                  c.post("/api/ops/segment/validate", json=body))
        check("/api/ops/segment", "post", c.post("/api/ops/segment", json={}))
        check("/api/maps", "get", c.get("/api/maps"))
        check("/api/maps/{name}", "get", c.get("/api/maps/m"))
        check("/api/maps/{name}", "get", c.get("/api/maps/nope"))
        check("/api/uploads/{id}", "delete", c.delete("/api/uploads/nope"))
    service.runner.shutdown()
    error = Draft202012Validator({**doc, "$ref": "#/components/schemas/Error"})
    for outcome in (Outcome(5, "no overlap"), Outcome(130, None, STOPPING)):
        error.validate(web_app.failure(outcome)[1])
    for schema in doc["components"]["schemas"].values():
        Draft202012Validator.check_schema(schema)
