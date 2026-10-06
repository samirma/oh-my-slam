"""``server.sh`` as a process (spec §2.6): exactly one stderr line once listening, nothing on
stdout, ``--status`` health JSON (exit 3 when not running), one service per workspace, ``--stop``
and SIGINT/SIGTERM as the normal stop — interrupting the requests in progress, which still get
their answer — a client that disconnects interrupting its request, unconsumed uploads cleared at
start and stop, and with the inference server down a 503 for inference operations while the
others work."""

from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import psutil
import pytest

from tests.unit.test_view_cli import PROBE, minimal_map, probe_records, sh

REPO = Path(__file__).resolve().parents[2]
LINE = re.compile(r"^server\.sh: listening on http://0\.0\.0\.0:(\d+)/$")


def server(*args: str, timeout: float = 60) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run([str(REPO / "server.sh"), *args], capture_output=True, timeout=timeout,
                          env=os.environ.copy())


def start(data: Path, probe: Path | None = None, browser: bool = False
          ) -> tuple[subprocess.Popen[bytes], str]:
    """Start server.sh on ``data``. With ``probe``, it runs with ``webbrowser.open`` replaced by
    ``tests/fakes/browser_probe.py`` (recording into ``probe``), opening the browser if
    ``browser``."""
    args = ["--data", str(data), *(() if browser else ("--no-browser",))]
    cmd = ([str(REPO / "server.sh"), *args] if probe is None else
           [sys.executable, str(PROBE), str(probe), "oh_my_slam.web.main", *args])
    proc = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy())
    assert proc.stderr is not None
    line = proc.stderr.readline().decode().rstrip("\n")
    m = LINE.match(line)
    if not m:
        proc.kill()
        raise AssertionError(f"unexpected first stderr line: {line!r}")
    return proc, f"http://127.0.0.1:{m.group(1)}/"


@pytest.fixture
def data(tmp_path: Path) -> Iterator[Path]:
    d = tmp_path / "data"
    yield d
    server("--data", str(d), "--stop")


def test_server_sh_lifecycle(data: Path) -> None:
    sh("start_inference_server.sh", "--stop")  # the inference server is down in this test
    stray = data / "uploads" / "abc" / "left.jpg"
    stray.parent.mkdir(parents=True)
    stray.write_bytes(b"x")
    res = server("--data", str(data), "--status")
    assert res.returncode == 3 and res.stdout == b""
    assert b"no service is running" in res.stderr

    proc, url = start(data)
    assert not stray.parent.exists()  # unconsumed uploads are deleted at start
    status = server("--data", str(data), "--status")
    assert status.returncode == 0, status.stderr
    health = json.loads(status.stdout)
    assert health["status"] == "ok" and health["inference"]["status"] == "down"
    assert health["service"]["data"] == str(data.resolve())

    second = server("--data", str(data), "--no-browser")
    assert second.returncode == 0 and second.stdout == b""
    assert b"already running" in second.stderr
    assert health["service"]["url"].encode() in second.stderr

    image = data / "inputs" / "a.jpg"
    image.parent.mkdir()
    image.write_bytes(b"\xff\xd8\xff")
    r = httpx.post(url + "api/ops/reconstruct", json={"image": "inputs/a.jpg"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "server_unavailable"
    minimal_map(data / "maps" / "m")
    assert httpx.get(url + "api/maps").json()[0]["name"] == "m"
    r = httpx.post(url + "api/ops/segment-map", json={"map": "m"}, timeout=60)
    assert r.status_code == 200 and r.json()["openlabel"]
    assert r.headers["server-timing"].startswith("export;dur=")
    assert httpx.get(url).status_code == 200
    assert httpx.get(url + "api/openapi.json").json()["openapi"].startswith("3.")
    up = httpx.post(url + "api/uploads?name=b.jpg", content=b"x",
                    headers={"content-type": "application/octet-stream"}).json()
    assert (data / up["path"]).is_file()

    stop = server("--data", str(data), "--stop")
    assert stop.returncode == 0, stop.stderr
    out, err = proc.communicate(timeout=60)
    assert proc.returncode == 0
    assert out == b""
    assert err == b""  # the listening line was the only one
    assert list((data / "uploads").iterdir()) == []  # and cleared at stop
    assert not (data / "server.json").exists()
    assert server("--data", str(data), "--status").returncode == 3


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_ctrl_c_and_sigterm_are_the_normal_stop(data: Path, sig: signal.Signals) -> None:
    proc, url = start(data)
    assert httpx.get(url + "api/health").status_code == 200
    proc.send_signal(sig)
    out, err = proc.communicate(timeout=60)
    assert proc.returncode == 0 and out == b"" and err == b""


@pytest.mark.parametrize("browser", [True, False], ids=["browser", "no-browser"])
def test_browser_opens_once_the_service_accepts_connections(data: Path, tmp_path: Path,
                                                            browser: bool) -> None:
    """The browser is opened on the service once it accepts connections (on the loopback
    address of the listening line's port), and never with ``--no-browser``."""
    record = tmp_path / "browser.jsonl"
    proc, url = start(data, probe=record, browser=browser)
    try:
        opened = probe_records(record, 10.0 if browser else 1.0)
        assert httpx.get(url).status_code == 200
    finally:
        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=60)
    assert proc.returncode == 0 and out == b"" and err == b""
    assert opened == ([{"url": url, "connected": True}] if browser else [])


def test_malformed_requests_never_reach_stderr(data: Path) -> None:
    """stderr stays the one listening line: uvicorn's warning about an invalid HTTP request (and
    anything else logged after that line) goes to ``<data>/server.log``."""
    import socket
    from urllib.parse import urlparse

    proc, url = start(data)
    try:
        u = urlparse(url)
        for garbage in (b"NOT HTTP AT ALL\r\n\r\n", b"GET / HTTP/1.1\r\nHost: \x00\r\n\r\n",
                        b"GET /" + b"x" * 70000 + b" HTTP/1.1\r\n\r\n"):
            with socket.create_connection((u.hostname, u.port), timeout=5) as sock:
                sock.sendall(garbage)
                sock.settimeout(5)
                with contextlib.suppress(OSError):
                    sock.recv(4096)
        assert httpx.get(url + "api/health").status_code == 200  # still serving
    finally:
        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=60)
    assert proc.returncode == 0 and out == b""
    assert err == b"", err.decode(errors="replace")
    assert "Invalid HTTP request" in (data / "server.log").read_text()


FAILING_STOP = """
import sys
from oh_my_slam.web import runner
def boom(self):
    raise RuntimeError("boom at shutdown")
runner.Runner.shutdown = boom
from oh_my_slam.web.main import entry
sys.argv = ["server.sh", *sys.argv[1:]]
entry()
"""


def test_a_failure_after_listening_is_one_line_pointing_at_the_log(data: Path) -> None:
    """A non-zero exit once listening: stderr has the listening line and one error line naming
    <data>/server.log, which holds the details."""
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    proc = subprocess.Popen([sys.executable, "-c", FAILING_STOP, "--data", str(data),
                             "--no-browser"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=env, cwd=REPO)
    assert proc.stderr is not None
    assert LINE.match(proc.stderr.readline().decode().rstrip("\n"))
    proc.send_signal(signal.SIGTERM)
    out, err = proc.communicate(timeout=60)
    log = data.resolve() / "server.log"
    assert proc.returncode == 1 and out == b""
    assert err.decode().splitlines() == [
        f"server.sh: error: internal error: RuntimeError: boom at shutdown (see {log})"]
    assert "boom at shutdown" in log.read_text()


def test_bad_options_are_usage_errors(tmp_path: Path) -> None:
    for args in (["--port", "-1"], ["--status", "--stop"], ["--status", "--port", "8"],
                 ["--stop", "--port", "0"], ["--status", "--no-browser"], ["--bogus"]):
        res = server("--data", str(tmp_path / "d"), *args)
        assert res.returncode == 2, args
        assert res.stdout == b""


SLOW_SERVER = """
import sys
from oh_my_slam.commands import spec
from oh_my_slam.web import operations
from tests.fakes import slow_command
spec.PROGRAMS = (*spec.PROGRAMS, slow_command.registry_program())
operations.Operation.module = property(
    lambda op: slow_command.MODULE if op.program.prog == "slow.sh"
    else f"oh_my_slam.cli.{op.program.prog.removesuffix('.sh')}")
from oh_my_slam.web.main import entry
sys.argv = ["server.sh", *sys.argv[1:]]
entry()
"""


def slow_server(data: Path) -> tuple[subprocess.Popen[bytes], str]:
    """server.sh with the stand-in command ``slow.sh`` (tests.fakes.slow_command) in its
    registry."""
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    proc = subprocess.Popen([sys.executable, "-c", SLOW_SERVER, "--data", str(data),
                             "--no-browser"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=env, cwd=REPO)
    assert proc.stderr is not None
    m = LINE.match(proc.stderr.readline().decode().rstrip("\n"))
    assert m, "no listening line"
    return proc, f"http://127.0.0.1:{m.group(1)}/"


def in_background(fn: Any) -> tuple[threading.Thread, dict[str, Any]]:
    out: dict[str, Any] = {}

    def target() -> None:
        try:
            out["value"] = fn()
        except Exception as exc:
            out["error"] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t, out


def command_of(proc: subprocess.Popen[bytes], timeout: float = 60) -> Any:
    """The command the service runs for a request (its child process)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        children = psutil.Process(proc.pid).children()
        if children:
            return children[0]
        time.sleep(0.05)
    raise AssertionError("no command started")


def test_a_client_that_disconnects_interrupts_its_command(data: Path) -> None:
    proc, url = slow_server(data)
    try:
        t, out = in_background(lambda: httpx.post(url + "api/ops/slow", json={"seconds": 120},
                                                  timeout=None))
        child = command_of(proc)
        assert httpx.get(url + "api/health").json()["service"]["requests"]["running"] == 1
        with pytest.raises(httpx.ReadTimeout):  # a client that gives up after 1 s
            httpx.post(url + "api/ops/slow", json={"seconds": 120}, timeout=1)
        deadline = time.monotonic() + 15
        while httpx.get(url + "api/health").json()["service"]["requests"]["running"] != 1:
            assert time.monotonic() < deadline, "the request was not interrupted"
            time.sleep(0.1)
        assert psutil.Process(proc.pid).children() == [child]  # its command is gone
    finally:
        proc.send_signal(signal.SIGTERM)
        out_, err = proc.communicate(timeout=60)
    t.join(30)
    assert proc.returncode == 0 and err == b""
    assert out["value"].status_code == 503 and out["value"].json()["error"]["code"] == "stopping"
    assert not child.is_running()


def test_stop_interrupts_the_requests_in_progress(data: Path) -> None:
    """``--stop``: the running command gets Ctrl-C, its client the answer 503 ``stopping``."""
    proc, url = slow_server(data)
    t, out = in_background(lambda: httpx.post(url + "api/ops/slow", json={"seconds": 120},
                                              timeout=None))
    child = command_of(proc)
    stop = server("--data", str(data), "--stop")
    assert stop.returncode == 0, stop.stderr
    t.join(30)
    _, err = proc.communicate(timeout=60)
    assert proc.returncode == 0 and err == b""
    r = out["value"]
    assert r.status_code == 503 and r.json()["error"]["code"] == "stopping"
    assert "interrupted" in r.json()["error"]["message"]
    assert not child.is_running()


def test_a_second_signal_kills_the_commands_and_exits(data: Path) -> None:
    """The first SIGTERM waits for a running command (here one deaf to SIGINT); a second one kills
    every command's process group and exits at once."""
    proc, url = slow_server(data)
    in_background(lambda: httpx.post(url + "api/ops/slow",
                                     json={"seconds": 120, "ignore_sigint": True}, timeout=None))
    child = command_of(proc)
    time.sleep(0.5)  # its SIGINT is ignored by now
    proc.send_signal(signal.SIGTERM)
    time.sleep(1.5)
    assert proc.poll() is None  # still waiting for the command
    t0 = time.monotonic()
    proc.send_signal(signal.SIGTERM)
    _, err = proc.communicate(timeout=30)
    assert proc.returncode == 130 and time.monotonic() - t0 < 10
    # after the listening line, the non-zero exit says so in one line pointing at the log
    assert err.decode().splitlines() == [
        f"server.sh: error: stopped by a second signal; every command's processes were killed "
        f"(see {data.resolve() / 'server.log'})"]
    child.wait(10)
