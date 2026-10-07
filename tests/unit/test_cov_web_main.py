"""``server.sh`` (spec §2.6) beyond ``test_web_server_sh``: the running service's state when its
file cannot be read, ``--status`` of a service that holds the workspace but does not answer,
``--stop`` of one that ignores SIGTERM (a second SIGTERM, then SIGKILL), the one error line on the
original stderr once listening, a port that cannot be bound, a second ``server.sh`` opening the
browser on the running one, and a clean exit that says nothing."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from oh_my_slam.core.errors import ServiceNotRunningError, UsageError
from oh_my_slam.server.lifecycle import ServerLock
from oh_my_slam.web import main as web_main
from oh_my_slam.web.workspace import Workspace

REPO = Path(__file__).resolve().parents[2]

HOLDER = """
import signal, sys, time
from pathlib import Path
from oh_my_slam.server.lifecycle import ServerLock
ServerLock(Path(sys.argv[1])).acquire()
ignore = int(sys.argv[2])  # how many SIGTERMs it ignores
seen = 0
def on_term(signum, frame):
    global seen
    seen += 1
    if seen > ignore:
        sys.exit(0)
signal.signal(signal.SIGTERM, on_term)
print("ready", flush=True)
while True:
    time.sleep(0.02)
"""


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    w = Workspace(tmp_path / "data")
    w.create()
    return w


@pytest.fixture
def held(ws: Workspace) -> Iterator[ServerLock]:
    """The workspace's lock, held as a running service holds it."""
    lock = ServerLock(ws.root / web_main.LOCK)
    lock.acquire()
    try:
        yield lock
    finally:
        lock.release()


def holder(ws: Workspace, ignore: int) -> subprocess.Popen[bytes]:
    """A process standing in for a running service: it holds the workspace's lock and ignores
    the first ``ignore`` SIGTERMs."""
    proc = subprocess.Popen([sys.executable, "-c", HOLDER, str(ws.root / web_main.LOCK),
                             str(ignore)], stdout=subprocess.PIPE, cwd=REPO)
    assert proc.stdout is not None and proc.stdout.readline() == b"ready\n"
    (ws.root / web_main.STATE).write_text(json.dumps({"pid": proc.pid,
                                                      "url": "http://0.0.0.0:1/"}))
    return proc


def test_a_state_file_that_cannot_be_read_is_no_running_service(ws: Workspace,
                                                                held: ServerLock) -> None:
    state = ws.root / web_main.STATE
    assert web_main._state(ws) is None  # held, but no state file yet (still starting)
    for text in ("{not json", "[1, 2]"):
        state.write_text(text)
        assert web_main._state(ws) is None, text
    state.write_text('{"pid": 1, "url": "http://0.0.0.0:5/"}')
    assert web_main._state(ws) == {"pid": 1, "url": "http://0.0.0.0:5/"}


def test_status_of_a_service_that_does_not_answer(ws: Workspace, held: ServerLock) -> None:
    with socket.socket() as s:  # a port nobody listens on any more
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    url = f"http://0.0.0.0:{port}/"
    (ws.root / web_main.STATE).write_text(json.dumps({"pid": os.getpid(), "url": url}))
    with pytest.raises(ServiceNotRunningError, match=f"the service at {url} does not answer"):
        web_main.status(ws)


@pytest.mark.parametrize(("ignore", "how"), [(1, "a second SIGTERM"), (99, "SIGKILL")])
def test_stop_forces_a_service_that_does_not_stop(ws: Workspace, monkeypatch: pytest.MonkeyPatch,
                                                  capsys: pytest.CaptureFixture[str], ignore: int,
                                                  how: str) -> None:
    """``--stop`` waits ``STOP_TIMEOUT_S``, then sends a second SIGTERM (which kills the commands'
    process groups and exits), and SIGKILL as the last resort."""
    monkeypatch.setattr(web_main, "STOP_TIMEOUT_S", 0.4)
    # a service that obeys the second SIGTERM gets all the time it needs to exit (the wait ends
    # as soon as it released the workspace); one that never obeys is killed after a short wait
    monkeypatch.setattr(web_main, "FORCE_TIMEOUT_S", 30.0 if how == "a second SIGTERM" else 0.4)
    proc = holder(ws, ignore)
    try:
        assert web_main.stop(ws) == 0
        code = proc.wait(10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert code == (0 if how == "a second SIGTERM" else -signal.SIGKILL)
    err = capsys.readouterr().err
    assert f"pid {proc.pid} did not stop within 0 s; killing its commands" in err
    assert err.strip().endswith(f"stopped (pid {proc.pid}, forced)")
    assert not ServerLock.is_held(ws.root / web_main.LOCK)


def test_the_error_line_goes_to_the_original_stderr_once_listening(
        ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web_main, "_terminal", None)
    web_main._tell_terminal(ws, "boom")  # not listening yet: run_main prints the error itself
    r, w = os.pipe()
    try:
        monkeypatch.setattr(web_main, "_terminal", w)
        web_main._tell_terminal(ws, "boom")
        os.close(w)
        line = os.read(r, 4096).decode()
    finally:
        os.close(r)
    assert line == f"server.sh: error: boom (see {ws.root / web_main.LOG})\n"


def test_a_failure_is_told_in_the_commands_words() -> None:
    assert web_main._failure(ServiceNotRunningError("no service")) == "no service"
    assert web_main._failure(KeyboardInterrupt()) == "interrupted"
    assert web_main._failure(ValueError("bad")) == "internal error: ValueError: bad"


def test_a_port_that_cannot_be_bound_is_a_usage_error() -> None:
    with socket.socket() as taken:
        taken.bind(("0.0.0.0", 0))
        taken.listen(1)
        port = taken.getsockname()[1]
        with pytest.raises(UsageError, match=f"--port {port}: cannot bind .*use another port or "
                                             "--port 0"):
            web_main._bind(port)


def test_a_second_server_sh_opens_the_browser_on_the_running_one(
        ws: Workspace, held: ServerLock, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    opened: list[str] = []
    monkeypatch.setattr(web_main.webbrowser, "open", opened.append)
    (ws.root / web_main.STATE).write_text(json.dumps({"pid": os.getpid(),
                                                      "url": "http://0.0.0.0:5555/"}))
    assert web_main.serve(ws, 0, True) == 0
    assert opened == ["http://127.0.0.1:5555/"]
    assert f"already running for {ws.root} at http://0.0.0.0:5555/" in capsys.readouterr().err
    (ws.root / web_main.STATE).unlink()  # still starting: no URL to open yet
    assert web_main.serve(ws, 0, True) == 0 and opened == ["http://127.0.0.1:5555/"]
    assert "at (starting)" in capsys.readouterr().err


@pytest.mark.parametrize("code", [0, None, 3])
def test_only_a_failed_exit_is_told_on_the_terminal(tmp_path: Path, code: int | None,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    told: list[str] = []

    def serve(*_a: object) -> int:
        raise SystemExit(code)

    monkeypatch.setattr(web_main, "serve", serve)
    monkeypatch.setattr(web_main, "_tell_terminal", lambda ws, what: told.append(what))
    with pytest.raises(SystemExit):
        web_main.main(["--data", str(tmp_path / "d"), "--no-browser"])
    assert told == ([] if code in (0, None) else ["internal error: SystemExit: 3"])
