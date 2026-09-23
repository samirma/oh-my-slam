"""Server lifecycle and concurrency against a real server process running stub models
(AC1, AC3). Uses the isolated runtime directory from conftest."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

from oh_my_slam.client import protocol as p
from oh_my_slam.client.client import InferenceClient
from oh_my_slam.core import paths
from oh_my_slam.core.errors import ServerUnavailableError
from oh_my_slam.server.lifecycle import (
    ServerLock,
    cleanup_stale_socket,
    pid_alive,
    read_state,
    socket_is_live,
)

REPO = Path(__file__).resolve().parents[2]
START = REPO / "start_inference_server.sh"


def run_start(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    e = os.environ.copy()
    e.update(env or {})
    return subprocess.run([str(START), *args], capture_output=True, text=True, env=e, timeout=120)


@pytest.fixture
def stub_server() -> Iterator[None]:
    res = run_start("--stub", env={"OH_MY_SLAM_STUB_DELAY": "0.02"})
    assert res.returncode == 0, res.stderr
    try:
        yield
    finally:
        run_start("--stop")


def test_start_is_idempotent_and_stop_cleans_up(stub_server: None) -> None:
    sock = paths.socket_path()
    assert sock.exists() and oct(sock.stat().st_mode & 0o777) == "0o600"
    state = read_state()
    assert state is not None and pid_alive(state["pid"])
    t0 = time.monotonic()
    again = run_start("--stub")
    assert again.returncode == 0 and "already running" in again.stderr
    assert time.monotonic() - t0 < 2.0
    status = run_start("--status")
    assert status.returncode == 0
    h = p.Health.model_validate_json(status.stdout)
    assert h.status == "ready" and h.pid == state["pid"]
    t0 = time.monotonic()
    stopped = run_start("--stop")
    assert stopped.returncode == 0
    assert time.monotonic() - t0 < 15.0
    assert not sock.exists() and not paths.state_file().exists()
    assert not pid_alive(state["pid"])
    down = run_start("--status")
    assert down.returncode == 3 and "./start_inference_server.sh" in down.stderr
    assert run_start("--stop").stderr.strip().endswith("not running")


def test_concurrent_clients_never_crash_the_server(stub_server: None, tmp_path: Path) -> None:
    """AC3: 8 clients x 25 mixed requests, every response 200 or 503."""
    img = tmp_path / "img.png"
    Image.fromarray(np.full((120, 160, 3), 90, np.uint8)).save(img)
    sock = str(paths.socket_path())
    codes: list[int] = []
    lock = threading.Lock()

    def client(i: int) -> None:
        with httpx.Client(transport=httpx.HTTPTransport(uds=sock), base_url="http://x",
                          timeout=60) as c:
            for j in range(25):
                kind = (i + j) % 4
                out = tmp_path / f"c{i}_{j}"
                out.mkdir()
                if kind == 0:
                    r = c.post(p.ROUTE_GEOMETRY, json=p.GeometryRequest(
                        image_path=str(img), out_dir=str(out)).model_dump())
                elif kind == 1:
                    r = c.post(p.ROUTE_GRAVITY, json=p.GravityRequest(
                        image_path=str(img)).model_dump())
                elif kind == 2:
                    r = c.post(p.ROUTE_SEGMENT, json=p.SegmentRequest(
                        image_path=str(img), labels=["chair"]).model_dump())
                else:
                    r = c.get(p.ROUTE_HEALTH)
                with lock:
                    codes.append(r.status_code)

    threads = [threading.Thread(target=client, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(codes) == 200
    assert set(codes) <= {200, 503}
    assert codes.count(200) > 0
    assert InferenceClient().health().status == "ready"


def test_second_server_process_refuses_to_start(stub_server: None) -> None:
    res = subprocess.run([sys.executable, "-m", "oh_my_slam.server.main", "--stub"],
                         capture_output=True, text=True, timeout=60)
    assert res.returncode == 1 and "another server" in res.stderr
    assert ServerLock.is_held()


def test_stale_socket_is_cleaned(tmp_path: Path) -> None:
    sock_path = Path("/tmp") / f"oms-stale-{os.getpid()}.sock"
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(str(sock_path))
    s.close()  # file remains, nobody listens
    assert sock_path.exists() and not socket_is_live(sock_path)
    assert cleanup_stale_socket(sock_path)
    assert not sock_path.exists()
    assert not cleanup_stale_socket(sock_path)


def test_client_fails_fast_when_down() -> None:
    t0 = time.monotonic()
    with pytest.raises(ServerUnavailableError) as err:
        InferenceClient().require_ready()
    assert time.monotonic() - t0 < 2.0
    assert "./start_inference_server.sh" in str(err.value)
