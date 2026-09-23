"""``start_inference_server.sh`` — start (idempotent), stop or query the inference server.

    start_inference_server.sh               start in the background, wait until ready
    start_inference_server.sh --foreground  run in this terminal (Ctrl-C stops)
    start_inference_server.sh --status      print /health as JSON (exit 3 if not running)
    start_inference_server.sh --stop        stop it; socket and state file removed
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
from oh_my_slam.core import paths
from oh_my_slam.core.errors import ServerUnavailableError
from oh_my_slam.core.log import claim_stdout
from oh_my_slam.server.lifecycle import ServerLock, pid_alive, read_state

PROG = "start_inference_server.sh"
DEFAULT_TIMEOUT_S = 1200.0
STOP_TIMEOUT_S = 12.0


def _say(msg: str) -> None:
    print(f"{PROG}: {msg}", file=sys.stderr, flush=True)


def _server_cmd(stub: bool) -> list[str]:
    cmd = [sys.executable, "-m", "oh_my_slam.server.main"]
    if stub:
        cmd.append("--stub")
    return cmd


def _describe(h: object) -> str:
    status = getattr(h, "status", "?")
    device = getattr(h, "device", "?")
    models = getattr(h, "models", {})
    missing = [m.name for m in models.values() if not m.loaded]
    extra = f"; unavailable: {', '.join(missing)}" if missing else ""
    return f"{status} on {device}{extra}"


def _tail(path: Path, n: int = 25) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return ""
    return "\n".join(lines[-n:])


def start(timeout: float, stub: bool) -> int:
    client = InferenceClient()
    try:
        h = client.health()
        if h.status in ("ready", "degraded"):
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
            _server_cmd(stub),
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
            if h.status in ("ready", "degraded"):
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


def main(argv: list[str]) -> int:
    parser = ArgumentParser(prog=PROG, description=__doc__.split("\n\n")[0] if __doc__ else None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--foreground", action="store_true", help="run in this terminal")
    group.add_argument("--stop", action="store_true", help="stop the running server")
    group.add_argument("--status", action="store_true", help="print health JSON to stdout")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                        help="seconds to wait for the models to load (default %(default)s)")
    parser.add_argument("--stub", action="store_true", help=argparse_suppress())
    args = parser.parse_args(argv)
    if args.stop:
        return stop()
    if args.status:
        return status()
    if args.foreground:
        cmd = _server_cmd(args.stub)
        os.execv(cmd[0], cmd)
    return start(args.timeout, args.stub)


def argparse_suppress() -> str:
    import argparse

    return argparse.SUPPRESS


def entry() -> None:
    run_main(PROG, main)


if __name__ == "__main__":
    entry()
