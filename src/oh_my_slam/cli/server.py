"""``start_inference_server.sh`` — start (idempotent), stop or query the inference server.

    start_inference_server.sh           start in the background, wait until ready
    start_inference_server.sh --status  print /health as JSON (exit 3 if not running)
    start_inference_server.sh --stop    stop it; socket and state file removed

Its options and modes are defined in ``commands.entry_points`` (``INFERENCE_SERVER``).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from oh_my_slam.cli.common import ArgumentParser, run_main
from oh_my_slam.client.client import InferenceClient
from oh_my_slam.commands import spec
from oh_my_slam.commands.entry_points import INFERENCE_SERVER
from oh_my_slam.core import paths
from oh_my_slam.core.errors import ServerUnavailableError
from oh_my_slam.core.log import claim_stdout
from oh_my_slam.server.lifecycle import ServerLock, pid_alive, read_state

PROG = INFERENCE_SERVER.prog
START_TIMEOUT_S = 1200.0  # the first start downloads the model weights
STOP_TIMEOUT_S = 12.0


def _say(msg: str) -> None:
    print(f"{PROG}: {msg}", file=sys.stderr, flush=True)


def _server_cmd() -> list[str]:
    return [sys.executable, "-m", "oh_my_slam.server.main"]


def _describe(h: object) -> str:
    return f"{getattr(h, 'status', '?')} on {getattr(h, 'device', '?')}"


def _tail(path: Path, n: int = 25) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return ""
    return "\n".join(lines[-n:])


def start(timeout: float) -> int:
    client = InferenceClient()
    try:
        h = client.health()
        if h.status == "ready":
            _say(f"already running ({_describe(h)})")
            return 0
        if h.status == "loading":
            _say("already starting; waiting until ready")
            return _wait_ready(client, None, timeout)
    except ServerUnavailableError:
        pass
    if ServerLock.is_held():
        _say("a server process holds the lock but does not answer yet; waiting")
        return _wait_ready(client, None, timeout)
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
    return _wait_ready(client, proc, timeout)


def _wait_ready(client: InferenceClient, proc: subprocess.Popen[bytes] | None, timeout: float) -> int:
    t0 = time.monotonic()
    last_note = 0.0
    while True:
        elapsed = time.monotonic() - t0
        if proc is not None and proc.poll() is not None:
            _say(f"server exited with code {proc.returncode}; last log lines:")
            print(_tail(paths.server_log()), file=sys.stderr)
            return 1
        try:
            h = client.health(timeout=1.0)
            if h.status == "ready":
                _say(f"ready after {elapsed:.1f} s ({_describe(h)})")
                return 0
            if h.status == "error":
                errors = "; ".join(f"{m.name}: {m.error}" for m in h.models.values() if m.error)
                _say(f"model loading failed: {errors}")
                if proc is not None:
                    proc.terminate()
                return 1
        except ServerUnavailableError:
            pass
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
    state = read_state()
    if state and isinstance(state.get("pid"), int):
        return int(state["pid"])
    try:
        text = paths.lock_file().read_text().strip()
        return int(text) if text and ServerLock.is_held() else None
    except (FileNotFoundError, ValueError):
        return None


def stop() -> int:
    pid = _server_pid()
    sock = paths.socket_path()
    if pid is None or not pid_alive(pid):
        sock.unlink(missing_ok=True)
        paths.state_file().unlink(missing_ok=True)
        _say("not running")
        return 0
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


def status() -> int:
    out = claim_stdout()
    h = InferenceClient().health(timeout=1.0)
    out.write_json(h.model_dump())
    return 0


def build_parser() -> ArgumentParser:
    return spec.build_parser(INFERENCE_SERVER)


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    if args.stop:
        return stop()
    if args.status:
        return status()
    return start(START_TIMEOUT_S)


def entry() -> None:
    run_main(PROG, main)


if __name__ == "__main__":
    entry()
