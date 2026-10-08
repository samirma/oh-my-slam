"""``start_inference_server.sh`` — start (idempotent), stop or query the inference server.

    start_inference_server.sh           start in the background, wait until ready
    start_inference_server.sh --status  print /health as JSON (exit 3 if not running)
    start_inference_server.sh --stop    stop it; socket and state file removed

Starting leaves a ready server as it is and joins one that is still starting. It waits for a
server that is stopping to exit and starts a new one, and starts a new one when the server it
joined dies. Of two starts at once, the one whose server lost the single-instance lock joins the
other's. A running server whose models failed to load, or that speaks another protocol (it was
started before an upgrade), is reported, with the restart command.

The running server is the process that holds the single-instance lock (``ServerLock``), whose pid
is in the lock file. The state file survives a server that crashed, and its pid may since name
another process, so ``--stop`` never trusts it on its own.

Its options and modes are defined in ``commands.entry_points`` (``INFERENCE_SERVER``).
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from oh_my_slam.client import protocol as p
from oh_my_slam.client.client import InferenceClient
from oh_my_slam.commands import spec
from oh_my_slam.commands.entry_points import INFERENCE_SERVER
from oh_my_slam.commands.parser import ArgumentParser, run_main
from oh_my_slam.core import paths
from oh_my_slam.core.errors import (
    ServerModelsFailedError,
    ServerProtocolError,
    ServerUnavailableError,
)
from oh_my_slam.core.log import PayloadWriter, claim_stdout
from oh_my_slam.server.lifecycle import ServerLock, pid_alive, read_state
from oh_my_slam.version import PROTOCOL_VERSION

PROG = INFERENCE_SERVER.prog
START_TIMEOUT_S = 1200.0  # the first start downloads the model weights
STOP_TIMEOUT_S = 12.0


def _say(msg: str) -> None:
    print(f"{PROG}: {msg}", file=sys.stderr, flush=True)


def _server_cmd() -> list[str]:
    return [sys.executable, "-m", "oh_my_slam.server.main"]


def _describe(h: p.Health) -> str:
    return f"{h.status} on {h.device}"


def _models_failed(h: p.Health) -> str:
    """What to do about a running server whose models failed to load: its log and the restart."""
    return str(ServerModelsFailedError(h.failures() or "no detail", str(paths.server_log())))


def _tail(path: Path, n: int = 25) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return ""
    return "\n".join(lines[-n:])


def _health(client: InferenceClient, timeout: float) -> p.Health | None:
    """The server's health, or None when it does not answer."""
    try:
        return client.health(timeout=timeout)
    except ServerUnavailableError:
        return None


def _launch() -> subprocess.Popen[bytes]:
    """Start the server process in the background, its output appended to the log."""
    log_path = paths.server_log()
    with log_path.open("ab") as logf:
        logf.write(f"\n=== start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n".encode())
        logf.flush()
        proc = subprocess.Popen(
            _server_cmd(),
            stdin=subprocess.DEVNULL,
            stdout=logf,
            stderr=logf,
            start_new_session=True,
            env=os.environ.copy(),
        )
    _say(f"starting (pid {proc.pid}, log {log_path})")
    return proc


def start(timeout: float) -> int:
    client = InferenceClient()
    h = _health(client, 0.5)
    if h is not None and h.status != "stopping" and h.protocol != PROTOCOL_VERSION:
        _say(f"already running, but {ServerProtocolError(h.protocol, PROTOCOL_VERSION)}")
        return 1
    if h is not None and h.status == "ready":
        _say(f"already running ({_describe(h)})")
        return 0
    if h is not None and h.status == "error":
        _say(f"already running, but {_models_failed(h)}")
        return 1
    if h is not None and h.status == "loading":
        _say("already starting; waiting until ready")
    elif h is not None:  # stopping
        _say("the running server is stopping; a new one starts once it has exited")
    elif ServerLock.is_held():
        _say("a server process holds the lock but does not answer yet; waiting")
    else:
        return _wait_ready(client, _launch(), timeout)
    return _wait_ready(client, None, timeout)


def _wait_ready(client: InferenceClient, proc: subprocess.Popen[bytes] | None, timeout: float) -> int:
    """Wait until the server answers ``ready``; ``proc`` is the server process this command
    started, None for one it joined. A joined server that is gone (it stopped or died: no answer,
    or ``stopping``, and the lock is free) is replaced by a new one. A started one that exited
    while another server holds the lock lost a race with another start: that server is joined."""
    t0 = time.monotonic()
    last_note = 0.0
    while True:
        elapsed = time.monotonic() - t0
        if proc is not None and proc.poll() is not None:
            if not ServerLock.is_held():
                _say(f"server exited with code {proc.returncode}; last log lines:")
                print(_tail(paths.server_log()), file=sys.stderr)
                return 1
            _say("another start launched a server first; waiting for it")
            proc = None
        h = _health(client, 1.0)
        if h is not None and h.status == "ready":
            _say(f"ready after {elapsed:.1f} s ({_describe(h)})")
            return 0
        if h is not None and h.status == "error":
            if proc is None:
                _say(_models_failed(h))
                return 1
            _say(f"model loading failed: {h.failures()}; see {paths.server_log()}")
            proc.terminate()
            return 1
        if proc is None and (h is None or h.status == "stopping") and not ServerLock.is_held():
            _say("the server it waited for is gone; starting a new one")
            proc = _launch()
        if elapsed > timeout:
            _say(f"not ready after {timeout:.0f} s; see {paths.server_log()}")
            if proc is not None:
                proc.terminate()
            return 1
        if elapsed - last_note >= 10.0:
            last_note = elapsed
            _say(f"loading models… {elapsed:.0f} s")
        time.sleep(0.5)


def _server_pid() -> int | None:
    """The pid of the running server: the one the lock holder wrote into the lock file, None when
    no process holds the lock (or its pid cannot be read). A state file whose pid is not that one
    is stale (left by a server that crashed) and is removed."""
    pid = None
    if ServerLock.is_held():
        try:
            pid = int(paths.lock_file().read_text().strip())
        except (FileNotFoundError, ValueError):
            pass
    state = read_state()
    if state is not None and state.get("pid") != pid:
        paths.state_file().unlink(missing_ok=True)
    return pid


def stop() -> int:
    pid = _server_pid()
    sock = paths.socket_path()
    if pid is None:
        if ServerLock.is_held():
            _say(f"a process holds {paths.lock_file()} without its pid in it; not stopping an "
                 "unknown process")
            return 1
        sock.unlink(missing_ok=True)  # left by a server that crashed: nobody holds the lock
        paths.state_file().unlink(missing_ok=True)
        _say("not running")
        return 0
    with contextlib.suppress(ProcessLookupError):  # it exited meanwhile
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + STOP_TIMEOUT_S
    while time.monotonic() < deadline:
        if not pid_alive(pid) and not sock.exists():
            break
        time.sleep(0.1)
    if pid_alive(pid):
        _say("did not stop in time; killing")
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.2)
    sock.unlink(missing_ok=True)
    paths.state_file().unlink(missing_ok=True)
    _say(f"stopped (pid {pid})")
    return 0


def status(out: PayloadWriter) -> int:
    h = InferenceClient().health(timeout=1.0)
    out.write_json(h.model_dump())
    return 0


def build_parser() -> ArgumentParser:
    return spec.build_parser(INFERENCE_SERVER)


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    out = claim_stdout()  # stdout: --status's health JSON only
    if args.stop:
        return stop()
    if args.status:
        return status(out)
    return start(START_TIMEOUT_S)


def entry() -> None:
    run_main(PROG, main)


if __name__ == "__main__":
    entry()
