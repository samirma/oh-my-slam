"""``server.sh``'s HTTP layer (spec §2.6 "API") at its edges: the inference server's health as the
service reports it (ready, loading, one that became ready while it was checked), the machine's
names when the system cannot list them all, a command's timings that cannot be read, a request
body that is empty or not JSON, a static path that cannot be resolved, a service that stops while
a request is validated, an upload whose declared length is not a number."""

from __future__ import annotations

import asyncio
import json
import socket
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from oh_my_slam.client import protocol as p
from oh_my_slam.client.client import InferenceClient
from oh_my_slam.web import app as web_app
from oh_my_slam.web.app import Service, create_app
from oh_my_slam.web.workspace import Workspace
from tests.fakes import slow_command
from tests.unit.test_web_api import (  # noqa: F401
    _repo_importable,
    _test_client_host,
    make_svc,
    with_programs,
)
from tests.unit.test_web_requests import Call


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    w = Workspace(tmp_path / "data")
    w.create()
    return w


@pytest.fixture
def svc(ws: Workspace) -> Iterator[tuple[Service, TestClient]]:
    service = make_svc(ws, inference_health=lambda: {})
    with TestClient(create_app(service)) as client:
        yield service, client
    service.runner.shutdown()


def answers(monkeypatch: pytest.MonkeyPatch, *statuses: str) -> list[str]:
    """The inference server's health answers ``statuses`` in turn (the last one repeats)."""
    left = list(statuses)

    def health(self: InferenceClient, timeout: float = 1.0) -> p.Health:
        status = left.pop(0) if len(left) > 1 else left[0]
        return p.Health(status=status, device="mps")  # type: ignore[arg-type]

    monkeypatch.setattr(InferenceClient, "health", health)
    return left


# -- the inference server ------------------------------------------------------------------------


def test_a_ready_inference_server_needs_no_start_command(monkeypatch: pytest.MonkeyPatch) -> None:
    answers(monkeypatch, "ready")
    h = web_app.inference_health()
    assert h["status"] == "ready" and h["start_command"] is None
    assert h["health"]["device"] == "mps"
    answers(monkeypatch, "loading")  # still loading: the top bar keeps the command that starts it
    h = web_app.inference_health()
    assert h["status"] == "loading" and h["start_command"] == web_app.START_COMMAND


def test_a_server_that_became_ready_while_it_was_checked_is_no_problem(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The commands' own check reads the health again: a server that was stopping or failing at
    the first look and is ready at the second is fine."""
    answers(monkeypatch, "stopping", "ready")
    assert web_app.inference_problem() is None
    answers(monkeypatch, "stopping")
    problem = web_app.inference_problem()
    assert problem is not None and problem.exit_code == 3 and "stopping" in problem.message


def test_the_machines_names_without_its_full_name_or_its_addresses(
        monkeypatch: pytest.MonkeyPatch) -> None:
    import psutil

    def fails(*_a: Any) -> Any:
        raise OSError("no resolver")

    monkeypatch.setattr(socket, "getfqdn", fails)
    monkeypatch.setattr(psutil, "net_if_addrs", fails)
    names = web_app.machine_hosts()
    host = socket.gethostname().lower()
    assert {"localhost", "127.0.0.1", "::1", host, f"{host.split('.')[0]}.local"} <= names


# -- what a command left ------------------------------------------------------------------------


def test_timings_that_cannot_be_read_give_no_server_timing(tmp_path: Path) -> None:
    assert web_app.server_timing(tmp_path) is None  # no record
    for record in ("{not json", "[]", '{"stages_s": {"setup": 1}}', '{"stages_s": 3, '
                   '"total_s": 1}'):
        (tmp_path / "timings.json").write_text(record)
        assert web_app.server_timing(tmp_path) is None, record
    (tmp_path / "timings.json").write_text(json.dumps({"stages_s": {"setup": 0.25},
                                                       "total_s": 0.5}))
    assert web_app.server_timing(tmp_path) == "setup;dur=250.0, total;dur=500.0"


def test_a_client_is_gone_only_at_its_disconnect() -> None:
    messages = [{"type": "http.request", "body": b"", "more_body": False},
                {"type": "http.request", "body": b"", "more_body": False},
                {"type": "http.disconnect"}]

    async def receive() -> dict[str, Any]:
        return messages.pop(0)

    asyncio.run(web_app.disconnected(types.SimpleNamespace(receive=receive)))  # type: ignore[arg-type]
    assert messages == []


# -- requests ------------------------------------------------------------------------------------


def test_an_empty_body_is_no_parameters_and_a_body_that_is_not_json_is_refused(
        svc: tuple[Service, TestClient]) -> None:
    _, c = svc
    json_type = {"content-type": "application/json"}
    r = c.post("/api/ops/segment/validate", content=b"", headers=json_type)
    assert r.status_code == 200 and not r.json()["valid"]
    assert "the following arguments are required: -i" in r.json()["problems"][0]["message"]
    r = c.post("/api/ops/segment/validate", content=b"{not json", headers=json_type)
    assert r.json()["problems"][0]["message"] == ("the request body must be a JSON object of "
                                                  "parameters")
    r = c.post("/api/ops/segment", content=b"{not json", headers=json_type)
    assert r.status_code == 400 and r.json()["error"]["code"] == "usage"


def test_a_static_path_that_cannot_be_resolved_is_not_found(svc: tuple[Service, TestClient]
                                                             ) -> None:
    service, _ = svc
    app = create_app(service)
    for path in ("/static/app\0.js", "/static/viewer/lib/x\0.js"):
        call = asyncio.run(Call(app, path, method="GET")())
        assert call.status == 404 and call.json()["error"]["code"] == "not_found", path


def test_a_service_that_stops_while_a_request_is_validated_refuses_it(
        ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    """The request is refused with 503 ``stopping`` instead of starting, and the upload it was
    given goes with it."""
    service = make_svc(ws, inference_health=lambda: {})

    def stops_meanwhile() -> None:
        service.runner.stopping = True  # e.g. Ctrl-C while the inference server is checked

    service.inference_check = stops_meanwhile
    with_programs(monkeypatch, service, slow_command.registry_program("infer.sh", "required"))
    uid, target = ws.new_upload("a.jpg")
    target.write_bytes(b"x")
    with TestClient(create_app(service)) as c:
        r = c.post("/api/ops/infer", json={"image": f"uploads/{uid}/a.jpg"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "stopping"
    assert not (ws.uploads / uid).exists() and service.runner.counts() == {"running": 0,
                                                                         "waiting": 0}
    service.runner.shutdown()


def test_an_upload_whose_declared_length_is_not_a_number_is_measured_as_it_arrives(
        svc: tuple[Service, TestClient]) -> None:
    service, _ = svc
    app = create_app(service)
    sent: list[dict[str, Any]] = []
    messages = [{"type": "http.request", "body": b"abc", "more_body": False}]

    async def receive() -> dict[str, Any]:
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": "/api/uploads", "raw_path": b"/api/uploads",
             "query_string": b"name=a.jpg", "http_version": "1.1", "scheme": "http",
             "headers": [(b"host", b"localhost"), (b"content-type", b"image/jpeg"),
                         (b"content-length", b"three")],
             "server": ("localhost", 80), "client": ("test", 1), "root_path": ""}
    asyncio.run(app(scope, receive, send))
    assert sent[0]["status"] == 201
    body = json.loads(b"".join(m.get("body", b"") for m in sent[1:]))
    assert body["size"] == 3 and (service.workspace.root / body["path"]).read_bytes() == b"abc"


def test_a_request_the_stop_interrupted_answers_the_commands_own_error() -> None:
    """Spec §2.6 "Requests", "Errors": a request ends with the command's own error by the generic
    rule, also when the service's stop interrupted it (exit 130: 499 ``interrupted``) or it ended
    otherwise while the service stopped."""
    from oh_my_slam.web.runner import STOPPING, Outcome

    for code, name, http in ((130, "interrupted", 499), (5, "not_registered", 422),
                             (1, "internal", 500), (2, "usage", 400)):
        status, body = web_app.failure(Outcome(code, "its own message", STOPPING))
        assert (status, body) == (http, {"error": {"code": name, "exit_code": code,
                                                   "message": "its own message",
                                                   "http_status": http}})
    status, body = web_app.failure(Outcome(130, "slow.sh: interrupted"))  # its client left
    assert (status, body["error"]["code"]) == (499, "interrupted")
