"""``start_inference_server.sh`` (spec §2.1) beyond the happy path of ``test_server_lifecycle``:
starting launches the server module in the background and waits until it is ready, joins a server
that is already starting, replaces one that is stopping or that died, joins the server of a start
that won the race for the lock, gives up with the reason (the server exited, a model failed, too
slow), and ``--stop`` stops a server that ignores SIGTERM but never signals the pid of a stale
state file. The launched server is the stub one (no weights); each test has a runtime directory of
its own."""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from oh_my_slam.cli import server as server_cli
from oh_my_slam.client import protocol as p
from oh_my_slam.core import paths
from oh_my_slam.core.errors import ServerUnavailableError
from oh_my_slam.server.lifecycle import ServerLock, pid_alive, read_state
from oh_my_slam.version import PROTOCOL_VERSION
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
    repeats); an exception class or instance is raised, a function is called for the answer."""

    def __init__(self, *answers: Any) -> None:
        self.answers = list(answers)
        self.calls = 0

    def health(self, timeout: float = 0.5) -> p.Health:
        self.calls += 1
        a = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(a, types.FunctionType):
            a = a()
        if isinstance(a, BaseException) or (isinstance(a, type) and issubclass(a, BaseException)):
            raise a
        return p.Health(status=a, device="cpu")


class Launches:
    """``_launch`` replaced: each launch is recorded and is a sleeping child process."""

    def __init__(self) -> None:
        self.procs: list[subprocess.Popen[bytes]] = []

    def __call__(self) -> subprocess.Popen[bytes]:
        self.procs.append(sleeper())
        return self.procs[-1]


@pytest.fixture
def launches(monkeypatch: pytest.MonkeyPatch) -> Iterator[Launches]:
    launched = Launches()
    monkeypatch.setattr(server_cli, "_launch", launched)
    try:
        yield launched
    finally:
        for proc in launched.procs:
            proc.kill()
            proc.wait()


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
    """Health ``error``: the start fails naming each model's error and the log. The server it
    launched is terminated; one it only joined is left to its owner, with the restart command."""

    class Failed(Scripted):
        def health(self, timeout: float = 0.5) -> p.Health:
            return p.Health(status="error", models={
                "geometry": p.ModelStatus(name="moge", error="weights missing"),
                "segment": p.ModelStatus(name="yoloe", loaded=True)})

    proc = sleeper() if launched else None
    try:
        assert server_cli._wait_ready(Failed(), proc, 60) == 1
        err = capsys.readouterr().err
        if proc is not None:
            assert f"model loading failed: moge: weights missing; see {paths.server_log()}" in err
            assert proc.wait(10) == -signal.SIGTERM
        else:
            assert "models failed to load (moge: weights missing)" in err
            assert str(paths.server_log()) in err and RESTART in err
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()


RESTART = "./start_inference_server.sh --stop && ./start_inference_server.sh"


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


def test_start_replaces_a_server_that_is_stopping(monkeypatch: pytest.MonkeyPatch,
                                                  capsys: pytest.CaptureFixture[str],
                                                  launches: Launches) -> None:
    """A server that is stopping still holds the lock: the start waits until it has exited (the
    lock is free), then launches a new one and waits until it is ready."""
    monkeypatch.setattr(server_cli, "time", Clock())
    lock = ServerLock()
    lock.acquire()  # the stopping server

    def exits() -> Any:
        lock.release()
        return ServerUnavailableError("no socket")

    client = Scripted("stopping", "stopping", exits, "ready")
    monkeypatch.setattr(server_cli, "InferenceClient", lambda: client)
    try:
        assert server_cli.start(60.0) == 0
    finally:
        lock.release()
    err = capsys.readouterr().err
    assert "the running server is stopping; a new one starts once it has exited" in err
    assert "the server it waited for is gone; starting a new one" in err and "ready after" in err
    assert len(launches.procs) == 1 and client.calls == 4


def test_start_replaces_a_joined_server_that_dies(monkeypatch: pytest.MonkeyPatch,
                                                  capsys: pytest.CaptureFixture[str],
                                                  launches: Launches) -> None:
    """The server the start joined while it loaded dies (no answer, the lock is free): a new one
    is launched instead of waiting for the timeout."""
    monkeypatch.setattr(server_cli, "time", Clock())
    client = Scripted("loading", "loading", ServerUnavailableError("ConnectError"), "ready")
    monkeypatch.setattr(server_cli, "InferenceClient", lambda: client)
    assert server_cli.start(60.0) == 0
    err = capsys.readouterr().err
    assert "already starting; waiting until ready" in err
    assert "the server it waited for is gone; starting a new one" in err and "ready after" in err
    assert len(launches.procs) == 1


def test_a_start_whose_server_lost_the_lock_joins_the_winner(monkeypatch: pytest.MonkeyPatch,
                                                             capsys: pytest.CaptureFixture[str]
                                                             ) -> None:
    """Two starts at once each launch a server; the one that did not get the lock exits 1. Its
    start then waits for the other's server instead of failing."""
    monkeypatch.setattr(server_cli, "time", Clock())
    proc = sleeper("import sys", "sys.exit(1)")  # "another server holds …"
    proc.wait(10)
    lock = ServerLock()
    lock.acquire()  # the winner's server
    try:
        client = Scripted(ServerUnavailableError("no socket"), "loading", "ready")
        assert server_cli._wait_ready(client, proc, 60) == 0
    finally:
        lock.release()
    err = capsys.readouterr().err
    assert "another start launched a server first; waiting for it" in err and "ready after" in err
    assert "server exited with code 1" not in err


def test_two_starts_at_once_both_end_with_one_ready_server(monkeypatch: pytest.MonkeyPatch,
                                                           capsys: pytest.CaptureFixture[str]
                                                           ) -> None:
    """Both starts find no server and launch the (stub) server process; one process gets the
    lock, and both starts succeed with that one server."""
    monkeypatch.setattr(server_cli, "_server_cmd", lambda: SERVER_CMD)
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    both = threading.Barrier(2, timeout=60)
    launch = server_cli._launch

    def racing() -> subprocess.Popen[bytes]:
        both.wait()  # both starts have found no server
        return launch()

    monkeypatch.setattr(server_cli, "_launch", racing)
    codes: list[int] = []
    starts = [threading.Thread(target=lambda: codes.append(server_cli.start(60.0)))
              for _ in range(2)]
    try:
        for t in starts:
            t.start()
        for t in starts:
            t.join(120)
        assert codes == [0, 0]
        state = read_state()
        assert state is not None and pid_alive(state["pid"]) and ServerLock.is_held()
        threading.Thread(target=reap, args=(state["pid"],), daemon=True).start()
        assert capsys.readouterr().err.count("starting (pid ") == 2
    finally:
        assert server_cli.stop() == 0
    assert not ServerLock.is_held()


def test_starting_next_to_a_server_whose_models_failed_says_how_to_restart_it(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """A running server that answers ``error``: no second one is launched, and the start fails
    with the models' errors, the log and the restart command."""

    class Failed(Scripted):
        def health(self, timeout: float = 0.5) -> p.Health:
            return p.Health(status="error", models={
                "multiview": p.ModelStatus(name="mapanything", error="out of memory")})

    monkeypatch.setattr(server_cli, "InferenceClient", Failed)
    assert server_cli.start(60.0) == 1
    err = capsys.readouterr().err
    assert ("already running, but inference server models failed to load (mapanything: out of "
            "memory)") in err
    assert str(paths.server_log()) in err and RESTART in err
    assert not paths.server_log().exists()  # nothing launched


@pytest.mark.parametrize("status", ["ready", "loading", "error"])
def test_starting_next_to_a_server_of_another_protocol_says_how_to_restart_it(
        status: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """A server started before an upgrade is not joined (every command would refuse it): no
    second one is launched, and the start fails with the restart command."""

    class Old(Scripted):
        def health(self, timeout: float = 0.5) -> p.Health:
            return p.Health(status=status, protocol=PROTOCOL_VERSION + 1)  # type: ignore[arg-type]

    monkeypatch.setattr(server_cli, "InferenceClient", Old)
    assert server_cli.start(60.0) == 1
    err = capsys.readouterr().err
    assert (f"already running, but the running inference server speaks protocol "
            f"{PROTOCOL_VERSION + 1}, this version needs {PROTOCOL_VERSION}") in err
    assert RESTART in err
    assert not paths.server_log().exists()  # nothing launched


def _lock_holder(*code: str) -> subprocess.Popen[bytes]:
    """A child process that takes the server lock (as a server does: its pid in the lock file),
    runs ``code``, says ``ready`` and sleeps."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "\n".join([
            "import time",
            "from oh_my_slam.server.lifecycle import ServerLock",
            *code,
            "ServerLock().acquire()",
            "print('ready', flush=True)",
            "time.sleep(60)"])], stdout=subprocess.PIPE, cwd=REPO)
    assert proc.stdout is not None and proc.stdout.readline() == b"ready\n"
    return proc


def test_stop_kills_a_server_that_ignores_sigterm(monkeypatch: pytest.MonkeyPatch,
                                                  capsys: pytest.CaptureFixture[str]) -> None:
    proc = _lock_holder("import signal", "signal.signal(signal.SIGTERM, signal.SIG_IGN)")
    try:
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


def test_stop_never_signals_the_pid_of_a_stale_state_file(capsys: pytest.CaptureFixture[str]
                                                          ) -> None:
    """A server that crashed leaves its state file, lock file and socket; its pid may name
    another process since. Nobody holds the lock: not running, the files are removed, and that
    process is left alone."""
    other = sleeper()
    try:
        paths.state_file().write_text(f'{{"pid": {other.pid}}}')
        paths.lock_file().write_text(str(other.pid))
        paths.socket_path().touch()
        assert server_cli.stop() == 0
        assert other.poll() is None  # not signalled
    finally:
        other.kill()
        other.wait()
    assert capsys.readouterr().err.strip().endswith("not running")
    assert not paths.socket_path().exists() and not paths.state_file().exists()


def test_the_server_pid_is_the_lock_holders(capsys: pytest.CaptureFixture[str]) -> None:
    """Only the process holding the lock is the server: its pid from the lock file; a state file
    naming another pid is stale and removed, one naming it is kept."""
    assert server_cli._server_pid() is None  # nothing: no state file, no lock file
    assert server_cli.stop() == 0 and capsys.readouterr().err.strip().endswith("not running")
    paths.lock_file().write_text("4242")  # a lock file left behind: nobody holds it
    paths.state_file().write_text('{"pid": 4242}')
    assert server_cli._server_pid() is None and not paths.state_file().exists()
    lock = ServerLock()
    lock.acquire()  # the lock holder writes its pid into the lock file
    try:
        paths.state_file().write_text(f'{{"pid": {os.getpid()}}}')
        assert server_cli._server_pid() == os.getpid() and paths.state_file().exists()
        paths.state_file().write_text('{"pid": 4243}')  # another server's, which crashed
        assert server_cli._server_pid() == os.getpid() and not paths.state_file().exists()
        paths.lock_file().write_text("not a pid")
        assert server_cli._server_pid() is None
        assert server_cli.stop() == 1  # someone holds the lock, but who: nothing is signalled
        assert "without its pid in it; not stopping an unknown process" in capsys.readouterr().err
    finally:
        lock.release()
