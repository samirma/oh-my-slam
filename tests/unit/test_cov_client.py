"""The inference client's failure handling (spec §2.1: every inference-requiring operation fails
with an actionable error when the server is not running, exit 3): an unreachable or unhealthy
server, a server still loading or whose models failed, queue-full answers retried with back-off,
input and server errors, timeouts. The server is an ``httpx.MockTransport`` on the client's
connection (no process, no socket)."""

from __future__ import annotations

import json
import logging
import socket
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from PIL import Image

from oh_my_slam.client import client as client_mod
from oh_my_slam.client import protocol as p
from oh_my_slam.client.client import InferenceClient
from oh_my_slam.core import timing
from oh_my_slam.core.errors import (
    ExitCode,
    InferenceError,
    InputError,
    ServerBusyError,
    ServerModelsFailedError,
    ServerUnavailableError,
)

Handler = Callable[[httpx.Request], httpx.Response]

GRAVITY = {"up_cam": [0.0, -1.0, 0.0], "roll_deg": 0.0, "pitch_deg": 0.0, "roll_unc_deg": 1.0,
           "pitch_unc_deg": 1.0, "focal_px": 500.0, "focal_unc_px": 2.0, "vfov_deg": 50.0,
           "timings": {"queue_s": 0.25, "compute_s": 0.5}}


@pytest.fixture
def sock(tmp_path: Path) -> Path:
    path = tmp_path / "srv.sock"
    path.touch()  # the client checks that the socket exists before any request
    return path


@pytest.fixture
def image(tmp_path: Path) -> Path:
    path = tmp_path / "small.png"  # sent as it is (no side above the server's read sizes)
    Image.fromarray(np.full((48, 64, 3), 90, np.uint8)).save(path)
    return path


def served(sock: Path, handler: Handler) -> InferenceClient:
    client = InferenceClient(sock, timeout=7.0)
    client._client = httpx.Client(transport=httpx.MockTransport(handler),
                                  base_url="http://oh-my-slam")
    return client


def health(status: str, **models: str | None) -> Handler:
    body = p.Health(status=status, models={k: p.ModelStatus(name=k.upper(), error=v)  # type: ignore[arg-type]
                                           for k, v in models.items()}).model_dump()
    return lambda req: httpx.Response(200, json=body)


def test_the_client_closes_its_connection_on_exit(sock: Path) -> None:
    with served(sock, health("ready")) as client:
        assert client.health().status == "ready"
        assert client._client is not None
    assert client._client is None
    client.close()  # closing again is harmless


def test_a_socket_nobody_listens_on_means_not_running() -> None:
    path = Path("/tmp") / f"oms-dead-{time.monotonic_ns()}.sock"
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(str(path))
    s.close()  # the file stays, nobody listens (a server that crashed)
    try:
        with pytest.raises(ServerUnavailableError) as err:
            InferenceClient(path).health()
    finally:
        path.unlink()
    assert err.value.exit_code == ExitCode.SERVER_UNAVAILABLE
    assert "ConnectError" in str(err.value) and "./start_inference_server.sh" in str(err.value)


def test_an_unhealthy_answer_means_not_running(sock: Path) -> None:
    client = served(sock, lambda req: httpx.Response(502, text="bad gateway"))
    with pytest.raises(ServerUnavailableError, match="health returned HTTP 502"):
        client.health()


class Clock:
    """The client's clock: ``sleep`` advances it (no real waiting)."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    perf_counter = monotonic

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    c = Clock()
    monkeypatch.setattr(client_mod, "time", c)
    return c


def test_ready_waits_for_a_loading_server(sock: Path, clock: Clock) -> None:
    answers = iter(["loading", "loading", "ready"])
    lines: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: lines.append(record.getMessage())  # type: ignore[method-assign]
    client_mod.log.addHandler(handler)
    try:
        assert served(sock, lambda req: health(next(answers))(req)).require_ready().status == \
            "ready"
    finally:
        client_mod.log.removeHandler(handler)
    assert clock.sleeps == [1.0, 1.0]
    assert lines == ["inference server is still loading models; waiting"]  # said once


def test_the_wait_for_a_loading_server_is_bounded(sock: Path, clock: Clock) -> None:
    with pytest.raises(ServerUnavailableError, match="server status 'loading'"):
        served(sock, health("loading")).require_ready(wait_loading_s=3.0)
    assert clock.sleeps == [1.0, 1.0, 1.0]


def test_a_stopping_server_is_unavailable(sock: Path) -> None:
    with pytest.raises(ServerUnavailableError, match="server status 'stopping'"):
        served(sock, health("stopping")).require_ready()


def test_failed_models_without_detail_are_still_actionable(sock: Path) -> None:
    with pytest.raises(ServerModelsFailedError, match=r"models failed to load \(no detail\)"):
        served(sock, health("error", geometry=None)).require_ready()


def test_requests_need_the_socket(tmp_path: Path, image: Path) -> None:
    with pytest.raises(ServerUnavailableError, match="no socket"):
        InferenceClient(tmp_path / "none.sock").gravity(p.GravityRequest(image_path=str(image)))


def test_a_queue_full_answer_is_retried_with_back_off(sock: Path, image: Path,
                                                     clock: Clock) -> None:
    calls: list[float] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(clock.now)
        if len(calls) < 3:
            return httpx.Response(503, json={"error": "busy", "detail": "queue full (8 jobs)"})
        return httpx.Response(200, json=GRAVITY)

    with timing.collect(sample_every=None) as t:
        res = served(sock, handler).gravity(p.GravityRequest(image_path=str(image)))
    assert res.focal_px == 500.0 and len(calls) == 3
    assert clock.sleeps == [0.2, pytest.approx(0.3)]  # 0.2 s, then 1.5 times longer each time
    # the request is recorded once, with the server's queue and compute time
    assert t.requests["gravity"] == {"count": 1, "wall_s": pytest.approx(0.5), "queue_s": 0.25,
                                     "compute_s": 0.5}


def test_a_server_busy_for_too_long_is_an_error(sock: Path, image: Path, clock: Clock) -> None:
    client = served(sock, lambda req: httpx.Response(503, json={"error": "busy"}))
    with pytest.raises(ServerBusyError, match=r"/v1/gravity: server busy \(busy\)"):
        client.gravity(p.GravityRequest(image_path=str(image)))
    # back-off up to 2 s between attempts, for at least BUSY_RETRY_S (300 s) in all
    assert max(clock.sleeps) == 2.0 and clock.sleeps[-1] == 2.0
    assert client_mod.BUSY_RETRY_S <= sum(clock.sleeps) < client_mod.BUSY_RETRY_S + 2.5


@pytest.mark.parametrize(("status", "body", "error", "message"), [
    (400, {"error": "InputError", "detail": "cannot read x.jpg"}, InputError,
     "/v1/segment: InputError: cannot read x.jpg"),
    (500, {"error": "RuntimeError", "detail": "MPS out of memory"}, InferenceError,
     "/v1/segment failed: RuntimeError: MPS out of memory"),
    (500, {"error": "model 'segment_yoloe' unavailable", "detail": None}, InferenceError,
     "/v1/segment failed: model 'segment_yoloe' unavailable"),
    (500, "Internal Server Error", InferenceError, "/v1/segment failed: Internal Server Error"),
])
def test_server_errors_carry_its_message(sock: Path, image: Path, status: int, body: Any,
                                         error: type[Exception], message: str) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if isinstance(body, str):
            return httpx.Response(status, text=body)
        return httpx.Response(status, json=body)

    with pytest.raises(error) as exc:
        served(sock, handler).segment_image(p.SegmentRequest(image_path=str(image),
                                                             labels=["cup"]))
    assert str(exc.value) == message


def test_a_request_that_times_out_is_an_inference_error(sock: Path, image: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=req)

    with pytest.raises(InferenceError, match="/v1/gravity timed out after 7 s"):
        served(sock, handler).gravity(p.GravityRequest(image_path=str(image)))


def test_a_connection_lost_mid_request_means_not_running(sock: Path, image: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("Server disconnected without sending a response.",
                                        request=req)

    with pytest.raises(ServerUnavailableError, match="RemoteProtocolError"):
        served(sock, handler).gravity(p.GravityRequest(image_path=str(image)))


def test_multiview_without_known_intrinsics_sends_none(sock: Path, image: Path) -> None:
    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={"views": []})

    served(sock, handler).multiview(p.MultiviewRequest(image_paths=[str(image)] * 2,
                                                       out_dir="/tmp/x"))
    assert seen[0]["intrinsics"] is None and seen[0]["image_paths"] == [str(image)] * 2


def test_connect_shares_one_client_and_checks_the_server_only_when_required(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_mod, "_shared", None)
    first = client_mod.connect(require=False)  # no server in this session: not checked
    assert client_mod.connect(require=False) is first
    with pytest.raises(ServerUnavailableError):
        client_mod.connect()
