"""``start_inference_server.sh`` (spec §2.1) beyond the happy path of ``test_server_lifecycle``:
starting launches the server module in the background and waits until it is ready, joins a server
that is already starting, gives up with the reason (the server exited, a model failed, too slow)
and stops a server that ignores SIGTERM. The launched server is the stub one (no weights); each
test has a runtime directory of its own."""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from oh_my_slam.cli import server as server_cli
from oh_my_slam.client import protocol as p
from oh_my_slam.core import paths
from oh_my_slam.core.errors import ServerUnavailableError
from oh_my_slam.server.lifecycle import ServerLock, pid_alive, read_state
from tests.fakes.stub_server import REPO, SERVER_CMD


@pytest.fixture(autouse=True)
def runtime(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A runtime directory (socket, lock, state, log) of this test's own."""
    short = Path(tempfile.mkdtemp(prefix="oms-", dir="/tmp"))
    monkeypatch.setenv("OH_MY_SLAM_RUNTIME_DIR", str(short))
    try:
        yield short
    finally:
        shutil.rmtree(short, ignore_errors=True)


class Clock:
    """``time`` for ``cli.server``: ``sleep`` advances ``monotonic`` instead of waiting."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.now += s

    @staticmethod
    def strftime(fmt: str) -> str:
        import time

        return time.strftime(fmt)


class Scripted:
    """An inference client whose ``health`` answers in turn from ``answers`` (the last one
    repeats); an exception class or instance is raised."""

    def __init__(self, *answers: Any) -> None:
        self.answers = list(answers)
        self.calls = 0

    def health(self, timeout: float = 0.5) -> p.Health:
        self.calls += 1
        a = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(a, BaseException) or (isinstance(a, type) and issubclass(a, BaseException)):
            raise a
        return p.Health(status=a, device="cpu")


def sleeper(*code: str) -> subprocess.Popen[bytes]:
    """A child process standing in for the server process (it sleeps unless ``code`` exits)."""
    return subprocess.Popen([sys.executable, "-c", "\n".join(code or ("import time",
                                                                       "time.sleep(60)"))])


def test_the_command_launches_the_server_module() -> None:
    """What start_inference_server.sh launches is the server's own entry point (which
    ``test_cov_server_process`` runs as such), with the interpreter running the command."""
    assert server_cli._server_cmd() == [sys.executable, "-m", "oh_my_slam.server.main"]


def test_start_launches_the_server_in_the_background_and_waits_until_ready(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(server_cli, "_server_cmd", lambda: SERVER_CMD)
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    try:
        assert server_cli.start(60.0) == 0
        err = capsys.readouterr().err
        state = read_state()
        assert state is not None and pid_alive(state["pid"])
        threading.Thread(target=reap, args=(state["pid"],), daemon=True).start()
        assert f"starting (pid {state['pid']}, log {paths.server_log()})" in err
        assert "ready after" in err and "(ready on cpu)" in err
        assert "=== start " in paths.server_log().read_text()  # when each start began
        # once running, starting again changes nothing, and --stop stops it
        assert server_cli.start(60.0) == 0
        assert "already running (ready on cpu)" in capsys.readouterr().err
    finally:
        assert server_cli.stop() == 0
    assert capsys.readouterr().err.strip().endswith(f"stopped (pid {state['pid']})")
    assert not paths.socket_path().exists() and not paths.state_file().exists()


def reap(pid: int) -> None:
    """Wait for a child this process started (the stopped server must not linger as a zombie,
    which counts as alive)."""
    with contextlib.suppress(ChildProcessError):
        os.waitpid(pid, 0)


def test_start_joins_a_server_that_is_already_starting(monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture[str]
                                                         ) -> None:
    client = Scripted("loading", "loading", "ready")
    monkeypatch.setattr(server_cli, "InferenceClient", lambda: client)
    monkeypatch.setattr(server_cli, "time", Clock())
    assert server_cli.start(60.0) == 0
    err = capsys.readouterr().err
    assert "already starting; waiting until ready" in err and "ready after 0.5 s" in err
    assert not paths.server_log().exists()  # nothing was launched


def test_start_waits_for_a_server_that_holds_the_lock_but_does_not_answer_yet(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    client = Scripted(ServerUnavailableError("no socket"), ServerUnavailableError("no socket"),
                      "ready")
    monkeypatch.setattr(server_cli, "InferenceClient", lambda: client)
    monkeypatch.setattr(server_cli, "time", Clock())
    lock = ServerLock()
    lock.acquire()  # another server process, still starting
    try:
        assert server_cli.start(60.0) == 0
    finally:
        lock.release()
    err = capsys.readouterr().err
    assert "holds the lock but does not answer yet; waiting" in err and "ready after" in err
    assert not paths.server_log().exists()


def test_a_server_that_exits_while_starting_fails_with_its_last_log_lines(
        capsys: pytest.CaptureFixture[str]) -> None:
    paths.server_log().write_text("".join(f"line {i}\n" for i in range(40)) + "boom: no weights\n")
    proc = sleeper("import sys", "sys.exit(7)")
    proc.wait(10)
    assert server_cli._wait_ready(Scripted(ServerUnavailableError("no socket")), proc, 60) == 1
    err = capsys.readouterr().err
    assert "server exited with code 7; last log lines:" in err
    assert "boom: no weights" in err and "line 16\n" in err and "line 15\n" not in err  # 25 lines


def test_a_missing_log_has_no_last_lines(runtime: Path) -> None:
    assert server_cli._tail(runtime / "server.log") == ""


@pytest.mark.parametrize("launched", [True, False], ids=["launched", "joined"])
def test_a_model_that_fails_to_load_ends_the_start(launched: bool,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    """Health ``error``: the start fails naming each model's error, and the server it launched
    is terminated (one it only joined is left to its owner)."""

    class Failed(Scripted):
        def health(self, timeout: float = 0.5) -> p.Health:
            return p.Health(status="error", models={
                "geometry": p.ModelStatus(name="moge", error="weights missing"),
                "segment": p.ModelStatus(name="yoloe", loaded=True)})

    proc = sleeper() if launched else None
    try:
        assert server_cli._wait_ready(Failed(), proc, 60) == 1
        assert "model loading failed: moge: weights missing" in capsys.readouterr().err
        if proc is not None:
            assert proc.wait(10) == -signal.SIGTERM
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()


@pytest.mark.parametrize("launched", [True, False], ids=["launched", "joined"])
def test_a_server_not_ready_in_time_ends_the_start(launched: bool, monkeypatch: pytest.MonkeyPatch,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    """Loading is reported every 10 s; past the timeout the start fails pointing at the log, and
    terminates the server it launched."""
    monkeypatch.setattr(server_cli, "time", Clock())
    proc = sleeper() if launched else None
    try:
        assert server_cli._wait_ready(Scripted("loading"), proc, 25.0) == 1
        err = capsys.readouterr().err
        assert "loading models… 10 s" in err and "loading models… 20 s" in err
        assert f"not ready after 25 s; see {paths.server_log()}" in err
        if proc is not None:
            assert proc.wait(10) == -signal.SIGTERM
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()


def test_stop_kills_a_server_that_ignores_sigterm(monkeypatch: pytest.MonkeyPatch,
                                                  capsys: pytest.CaptureFixture[str]) -> None:
    proc = subprocess.Popen(
        [sys.executable, "-c", "import signal, sys, time\n"
         "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
         "print('ready', flush=True)\n"
         "time.sleep(60)"], stdout=subprocess.PIPE)
    try:
        assert proc.stdout is not None and proc.stdout.readline() == b"ready\n"
        paths.state_file().write_text(f'{{"pid": {proc.pid}}}')
        paths.socket_path().touch()
        monkeypatch.setattr(server_cli, "STOP_TIMEOUT_S", 0.3)
        assert server_cli.stop() == 0
        assert proc.wait(10) == -signal.SIGKILL
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    err = capsys.readouterr().err
    assert "did not stop in time; killing" in err and f"stopped (pid {proc.pid})" in err
    assert not paths.socket_path().exists() and not paths.state_file().exists()


def test_starting_next_to_a_server_whose_models_failed_reports_the_failure(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """A server that answers ``error`` still holds the lock: no second one is launched, and the
    start fails with the models' errors."""

    class Failed(Scripted):
        def health(self, timeout: float = 0.5) -> p.Health:
            return p.Health(status="error", models={
                "multiview": p.ModelStatus(name="mapanything", error="out of memory")})

    monkeypatch.setattr(server_cli, "InferenceClient", Failed)
    lock = ServerLock()
    lock.acquire()
    try:
        assert server_cli.start(60.0) == 1
    finally:
        lock.release()
    err = capsys.readouterr().err
    assert "holds the lock but does not answer yet" in err
    assert "model loading failed: mapanything: out of memory" in err
    assert not paths.server_log().exists()


def test_the_server_pid_comes_from_the_state_file_or_the_held_lock(
        capsys: pytest.CaptureFixture[str]) -> None:
    assert server_cli._server_pid() is None  # nothing: no state file, no lock file
    assert server_cli.stop() == 0 and capsys.readouterr().err.strip().endswith("not running")
    paths.lock_file().write_text("4242")  # a lock file left behind: nobody holds it
    assert server_cli._server_pid() is None
    lock = ServerLock()
    lock.acquire()  # the lock holder writes its pid into the lock file
    try:
        assert server_cli._server_pid() == os.getpid()
        paths.lock_file().write_text("not a pid")
        assert server_cli._server_pid() is None
        paths.state_file().write_text('{"pid": 4243}')  # the state file comes first
        assert server_cli._server_pid() == 4243
        paths.state_file().write_text('{"pid": "4243"}')  # not a pid: the lock file decides
        assert server_cli._server_pid() is None
    finally:
        lock.release()
