"""``server.sh`` requests (spec §2.6 "Requests", "Workspace"): an operation runs within its own
request and answers when its command ends. Requests that use the inference server run one at a
time, in arrival order, while the others are answered at once; two requests never write the same
map at once; a client that disconnects interrupts its request — waiting or running, as Ctrl-C
would — and frees the queue; stopping the service interrupts them all; an upload goes with the
request it was given to. Driven through the ASGI app itself, with the stand-in command
(``tests.fakes.slow_command``), so a disconnect is the client's ``http.disconnect``."""

from __future__ import annotations

import asyncio
import json
import signal
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from oh_my_slam.web.app import Service, create_app
from oh_my_slam.web.runner import Run, Runner
from oh_my_slam.web.workspace import Workspace
from tests.fakes import slow_command
from tests.unit.test_web_api import _repo_importable, make_svc, with_programs  # noqa: F401

INFER = "infer.sh"  # a stand-in that needs the inference server


class Call:
    """One HTTP request to the ASGI app; ``leave()`` disconnects its client."""

    def __init__(self, app: Callable[..., Awaitable[None]], path: str, body: Any = None,
                 method: str = "POST") -> None:
        self.app, self.path, self.method = app, path, method
        self.data = json.dumps(body).encode() if body is not None else b""
        self.gone = asyncio.Event()
        self.status: int | None = None
        self.headers: dict[str, str] = {}
        self.body = b""
        self.ended: float | None = None

    def leave(self) -> None:
        self.gone.set()

    def json(self) -> Any:
        return json.loads(self.body)

    async def __call__(self) -> Call:
        sent = False

        async def receive() -> dict[str, Any]:
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": self.data, "more_body": False}
            await self.gone.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                self.status = message["status"]
                self.headers = {k.decode().lower(): v.decode() for k, v in message["headers"]}
            elif message["type"] == "http.response.body":
                self.body += message.get("body", b"")

        path, _, query = self.path.partition("?")
        headers = [(b"host", b"localhost")]
        if self.method == "POST":
            headers.append((b"content-type", b"application/json"))
        scope = {"type": "http", "method": self.method, "path": path, "raw_path": path.encode(),
                 "query_string": query.encode(), "http_version": "1.1", "headers": headers,
                 "scheme": "http", "server": ("localhost", 80), "client": ("test", 1),
                 "root_path": ""}
        await self.app(scope, receive, send)
        self.ended = time.monotonic()
        return self


@pytest.fixture
def svc(ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> Service:
    service = make_svc(ws, inference_check=lambda: None, inference_health=lambda: {})
    with_programs(monkeypatch, service, slow_command.registry_program(),
                  slow_command.registry_program(INFER, "required"))
    return service


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    w = Workspace(tmp_path / "data")
    w.create()
    return w


def run(service: Service, scenario: Callable[[Callable[..., Awaitable[None]]], Awaitable[None]]
        ) -> None:
    """Run ``scenario(app)`` in an event loop, then stop the runner."""
    try:
        asyncio.run(scenario(create_app(service)))
    finally:
        service.runner.shutdown()


async def until(check: Callable[[], bool], timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.02)


def started(runner: Runner, n: int) -> Callable[[], bool]:
    """Whether ``n`` commands run (their process started)."""
    return lambda: sum(r.proc is not None for r in runner.running) >= n


async def send(app: Callable[..., Awaitable[None]], path: str, body: Any = None,
               method: str = "POST") -> tuple[Call, asyncio.Task[Call]]:
    """Send a request (it takes its place in the arrival order before this returns)."""
    call = Call(app, path, body, method)
    task = asyncio.ensure_future(call())
    await asyncio.sleep(0.05)
    return call, task


def upload(ws: Workspace, name: str = "photo.jpg") -> str:
    uid, target = ws.new_upload(name)
    target.write_bytes(b"x")
    return f"uploads/{uid}/{name}"


def test_inference_requests_run_one_at_a_time_in_arrival_order_others_at_once(
        svc: Service) -> None:
    async def scenario(app: Callable[..., Awaitable[None]]) -> None:
        a, ta = await send(app, f"/api/ops/{INFER.removesuffix('.sh')}", {"seconds": 1.5})
        b, tb = await send(app, "/api/ops/infer", {"seconds": 0.2})
        c, tc = await send(app, "/api/ops/infer", {"seconds": 0.2})
        await until(started(svc.runner, 1))
        assert svc.runner.counts() == {"running": 1, "waiting": 2}
        health = await Call(app, "/api/health", method="GET")()
        assert health.json()["service"]["requests"] == {"running": 1, "waiting": 2}
        # each request in progress, in arrival order, with the command line its validation gave
        # (how a page tells whether its own request runs or waits for its turn)
        listed = health.json()["service"]["in_progress"]
        assert [(r["operation"], r["state"]) for r in listed] == [
            ("infer", "running"), ("infer", "waiting"), ("infer", "waiting")]
        assert [r["command"] for r in listed] == [
            ["infer.sh", f"--seconds={s:g}"] for s in (1.5, 0.2, 0.2)]
        assert listed[0]["started_at"] >= listed[0]["arrived_at"] and listed[1]["started_at"] is None
        assert listed[0]["arrived_at"] <= listed[1]["arrived_at"] <= listed[2]["arrived_at"]
        checked = await Call(app, "/api/ops/infer/validate", {"seconds": 0.2})()
        assert checked.json()["command"] == listed[1]["command"]
        d, td = await send(app, "/api/ops/slow", {"seconds": 0.2})  # answered at once
        await td
        assert d.status == 200 and not ta.done()  # while the first inference request runs
        await asyncio.gather(ta, tb, tc)
        for call in (a, b, c):
            assert call.status == 200, call.body
        ra, rb, rc, rd = (x.json() for x in (a, b, c, d))
        assert ra["end"] <= rb["start"] and rb["end"] <= rc["start"]  # one at a time, in order
        assert ra["start"] < rd["start"] < rd["end"] < ra["end"]
        assert svc.runner.counts() == {"running": 0, "waiting": 0}

    run(svc, scenario)


def test_arrival_order_holds_while_an_earlier_request_is_still_validated(
        ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    """A request that arrived first runs first even when its validation (the inference server's
    check) takes longer than that of one that arrived after it."""
    checks: list[float] = []

    def slow_first_check() -> None:
        checks.append(time.monotonic())
        if len(checks) == 1:
            time.sleep(0.8)  # the first arrival's check is slow; the second's is not

    service = make_svc(ws, inference_check=slow_first_check, inference_health=lambda: {})
    with_programs(monkeypatch, service, slow_command.registry_program(INFER, "required"))

    async def scenario(app: Callable[..., Awaitable[None]]) -> None:
        a, ta = await send(app, "/api/ops/infer", {"seconds": 0.2})
        b, tb = await send(app, "/api/ops/infer", {"seconds": 0.2})
        await asyncio.gather(ta, tb)
        assert a.status == b.status == 200, (a.body, b.body)
        assert len(checks) == 2 and checks[0] < checks[1]
        assert a.json()["end"] <= b.json()["start"]  # in arrival order

    run(service, scenario)


def test_two_requests_never_write_the_same_map_at_once(svc: Service) -> None:
    async def scenario(app: Callable[..., Awaitable[None]]) -> None:
        a, ta = await send(app, "/api/ops/slow", {"seconds": 1.0, "map": "m"})
        b, tb = await send(app, "/api/ops/slow", {"seconds": 0.1, "map": "maps/m"})
        c, tc = await send(app, "/api/ops/slow", {"seconds": 0.1, "map": "n"})
        await asyncio.gather(ta, tb, tc)
        ra, rb, rc = (x.json() for x in (a, b, c))
        assert ra["end"] <= rb["start"]  # the second writer of m waited for the first
        assert rc["end"] < ra["end"]  # another map's writer did not

    run(svc, scenario)


def test_a_client_that_disconnects_interrupts_its_request(svc: Service) -> None:
    """A waiting request never starts; a running one is interrupted as Ctrl-C would (exit 130)
    and frees the queue; their uploads are deleted at once; nothing of them is kept."""
    ws = svc.workspace

    async def scenario(app: Callable[..., Awaitable[None]]) -> None:
        up_a, up_b = upload(ws, "a.jpg"), upload(ws, "b.jpg")
        a, ta = await send(app, "/api/ops/infer", {"seconds": 60, "image": up_a})
        b, tb = await send(app, "/api/ops/infer", {"seconds": 0.1, "image": up_b})
        c, tc = await send(app, "/api/ops/infer", {"seconds": 0.1})
        await until(started(svc.runner, 1))
        running: Run = svc.runner.running[0]
        assert running.uploads and svc.runner.counts() == {"running": 1, "waiting": 2}
        b.leave()  # waiting: never starts
        await tb
        assert b.status == 499 and not (ws.root / up_b).parent.exists()
        assert svc.runner.counts() == {"running": 1, "waiting": 1}
        t0 = time.monotonic()
        a.leave()  # running: SIGINT to its command
        await ta
        assert a.status == 499 and time.monotonic() - t0 < 10
        assert running.proc is not None and running.proc.returncode == 130
        assert not (ws.root / up_a).parent.exists()
        await tc  # the queue moved on
        assert c.status == 200 and c.json()["start"] >= t0
        assert list(ws.uploads.iterdir()) == [] and list(ws.requests.iterdir()) == []

    run(svc, scenario)


def test_a_command_deaf_to_ctrl_c_is_terminated(ws: Workspace,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    service = Service(ws, Runner(ws, interrupt_grace_s=0.5), extra_hosts={"testserver"},
                      inference_check=lambda: None)
    with_programs(monkeypatch, service, slow_command.registry_program())

    async def scenario(app: Callable[..., Awaitable[None]]) -> None:
        a, ta = await send(app, "/api/ops/slow", {"seconds": 60, "ignore_sigint": True})
        await until(started(service.runner, 1))
        running = service.runner.running[0]
        a.leave()  # at once: deaf from its start (tests.fakes.slow_command)
        await ta
        assert running.proc is not None and running.proc.returncode == -15  # SIGTERM

    run(service, scenario)


def test_stopping_the_service_interrupts_every_request(svc: Service) -> None:
    """Each request still gets its answer: 503 ``stopping``; its upload is deleted, and so is
    every other one when the runner shuts down."""
    ws = svc.workspace

    async def scenario(app: Callable[..., Awaitable[None]]) -> None:
        up = upload(ws)
        a, ta = await send(app, "/api/ops/infer", {"seconds": 60, "image": up})
        b, tb = await send(app, "/api/ops/infer", {"seconds": 0.1})
        await until(started(svc.runner, 1))
        running = svc.runner.running[0]
        svc.runner.stop()
        await asyncio.gather(ta, tb)
        for call in (a, b):
            assert call.status == 503 and call.json()["error"]["code"] == "stopping"
        assert running.proc is not None and running.proc.returncode == 130
        assert not (ws.root / up).exists()
        late = await Call(app, "/api/ops/slow", {"seconds": 0})()
        assert late.status == 503 and late.json()["error"]["code"] == "stopping"

    upload(ws, "left.jpg")  # given to no request
    run(svc, scenario)
    assert list(ws.uploads.iterdir()) == []


def test_an_interrupt_while_its_command_starts_is_the_commands_interrupt(ws: Workspace) -> None:
    """A request interrupted as soon as its command started: the SIGINT may reach the command
    before Python handles it, which kills a process started as usual (-2). The runner starts it
    with SIGINT blocked (``core.process.sigint_blocked``), so the interrupt waits for the
    command's ``run_main`` and ends as the command's own (130). Here the command sends itself the
    SIGINT before ``run_main``, so the moment is certain."""
    argv = ["--seconds=30", "--interrupt-at-start"]
    plain = subprocess.run([sys.executable, "-m", slow_command.MODULE, *argv],
                           capture_output=True, timeout=60)
    assert plain.returncode == -signal.SIGINT  # started as usual: killed by the signal
    runner = Runner(ws)
    run_ = Run(1, "slow.sh", slow_command.MODULE, argv, inference=False)
    t0 = time.monotonic()
    outcome = runner._command(run_, ws.request_dir(run_.id))
    assert (outcome.code, outcome.message) == (130, "slow.sh: interrupted")
    assert time.monotonic() - t0 < 30  # interrupted, not slept


def test_an_upload_in_use_is_neither_given_twice_nor_deleted(svc: Service) -> None:
    ws = svc.workspace

    async def scenario(app: Callable[..., Awaitable[None]]) -> None:
        up = upload(ws)
        uid = up.split("/")[1]
        a, ta = await send(app, "/api/ops/slow", {"seconds": 1, "image": up})
        await until(started(svc.runner, 1))
        b = await Call(app, "/api/ops/slow", {"image": up})()
        assert b.status == 409 and b.json()["error"]["code"] == "upload_in_use"
        bad = await Call(app, "/api/ops/slow", {"image": up, "seconds": "x"})()
        assert bad.status == 400  # refused, and the upload is not its to delete
        d = await Call(app, f"/api/uploads/{uid}", method="DELETE")()
        assert d.status == 409 and (ws.root / up).exists()
        await ta
        assert a.status == 200 and not (ws.root / up).exists()

    run(svc, scenario)


def test_the_response_is_the_commands_stdout_with_its_timings(svc: Service) -> None:
    """The body is what the command wrote to stdout, in the result's media type; the command's
    own stages are in ``Server-Timing``; a failure is the command's message and exit code."""
    async def scenario(app: Callable[..., Awaitable[None]]) -> None:
        ok = await Call(app, "/api/ops/slow", {"seconds": 0.3})()
        assert ok.status == 200 and ok.headers["content-type"] == "application/json"
        assert ok.body.endswith(b"}\n") and ok.json()["slept"] == 0.3
        assert int(ok.headers["content-length"]) == len(ok.body)
        timing = dict(m.split(";dur=") for m in ok.headers["server-timing"].split(", "))
        assert list(timing) == ["setup", "total"] and float(timing["setup"]) >= 300
        failed = await Call(app, "/api/ops/slow", {"code": 6})()
        assert failed.status == 409 and failed.json()["error"] == {
            "code": "map_locked", "exit_code": 6, "message": "slow.sh: slept 0 s",
            "http_status": 409}
        assert "server-timing" not in failed.headers
        assert list(svc.workspace.requests.iterdir()) == []  # no result is kept

    run(svc, scenario)
