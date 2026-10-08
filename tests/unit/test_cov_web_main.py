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
from typing import Any

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


def test_a_port_that_cannot_be_bound_is_a_usage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    with socket.socket() as taken:
        taken.bind(("0.0.0.0", 0))
        taken.listen(1)
        port = taken.getsockname()[1]
        with pytest.raises(UsageError, match=f"--port {port}: cannot bind .*use another port or "
                                             "--port 0"):
            web_main._bind(port)
        # taken after the check (or where it cannot look): the bind's own error, in those words
        monkeypatch.setattr(web_main, "_in_use", lambda port: None)
        with pytest.raises(UsageError, match=f"--port {port}: cannot bind \\(Address already in "
                                             "use\\); use another port or --port 0"):
            web_main._bind(port)


@pytest.mark.parametrize("family, address", [(socket.AF_INET, "127.0.0.1"),
                                             (socket.AF_INET6, "::1")])
def test_a_port_another_process_listens_on_at_any_address_is_refused(family: int,
                                                                     address: str) -> None:
    """With SO_REUSEADDR, macOS binds 0.0.0.0:<n> even while another process listens on
    127.0.0.1:<n>, which then gets the browser's and --status's connections; and a browser's
    ``localhost`` may be ::1. Either is refused, naming the address."""
    with socket.socket(family) as taken:
        taken.bind((address, 0))
        taken.listen(1)
        port = taken.getsockname()[1]
        with pytest.raises(UsageError, match=f"--port {port}: cannot bind \\(another process "
                                             f"listens on it at {address}\\); use another port "
                                             "or --port 0"):
            web_main._bind(port)


def test_a_port_that_only_closed_connections_hold_is_bound_again_at_once(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A service that just stopped leaves its port in TIME_WAIT, which only SO_REUSEADDR binds
    over: it is no other process, so the port binds at once; so does a free one."""
    from oh_my_slam.web import app as web_app

    # the unit tests run offline: the machine's own addresses are its loopback ones (and names)
    monkeypatch.setattr(web_app, "machine_hosts",
                        lambda: {"localhost", "127.0.0.1", "::1", "0.0.0.0"})
    with socket.socket() as server:
        server.bind(("0.0.0.0", 0))
        server.listen(1)
        port = server.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port)) as client:
            accepted, _ = server.accept()
            accepted.close()  # the server closes first: its side is left in TIME_WAIT
            client.recv(1)
    assert not web_main._binds(socket.AF_INET, port)
    assert web_main._in_use(port) is None
    with web_main._bind(port) as sock:
        assert sock.getsockname()[1] == port
    with socket.socket() as probe:  # a free port
        probe.bind(("0.0.0.0", 0))
        port = probe.getsockname()[1]
    assert web_main._binds(socket.AF_INET, port) and web_main._binds(socket.AF_INET6, port)
    with web_main._bind(port) as sock:
        assert sock.getsockname()[1] == port


def test_a_machine_without_ipv6_has_nothing_listening_there(
        monkeypatch: pytest.MonkeyPatch) -> None:
    real = socket.socket

    def no_ipv6(family: int = socket.AF_INET, *args: Any) -> socket.socket:
        if family == socket.AF_INET6:
            raise OSError("Address family not supported by protocol")
        return real(family, *args)

    monkeypatch.setattr(web_main.socket, "socket", no_ipv6)
    assert web_main._binds(socket.AF_INET6, 0)


def test_a_second_server_sh_reports_the_running_one_and_exits(
        ws: Workspace, held: ServerLock, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    """Spec §2.6: a second server.sh on the same --data reports the running one's URL and exits
    (it opens no browser)."""
    opened: list[str] = []
    monkeypatch.setattr(web_main.webbrowser, "open", opened.append)
    (ws.root / web_main.STATE).write_text(json.dumps({"pid": os.getpid(),
                                                      "url": "http://0.0.0.0:5555/"}))
    assert web_main.serve(ws, 0, True) == 0
    assert f"already running for {ws.root} at http://0.0.0.0:5555/" in capsys.readouterr().err
    (ws.root / web_main.STATE).unlink()  # still starting: no URL yet
    assert web_main.serve(ws, 0, True) == 0 and opened == []
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
