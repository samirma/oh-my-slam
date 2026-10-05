"""``server.sh`` — the local web service (spec §2.6).

    server.sh [--port <n>] [--data <folder>] [--no-browser]
    server.sh --status | --stop

Binds ``0.0.0.0:<port>`` (default 0: a free port), serves the API and the web application until
Ctrl-C or SIGTERM, and once it accepts connections writes exactly one stderr line::

    server.sh: listening on http://0.0.0.0:<port>/

Nothing goes to stdout except ``--status``'s health JSON. One service runs per workspace (a lock
and a state file in ``<data>``): a second ``server.sh`` on the same ``--data`` reports the running
one's URL and exits 0. ``--stop`` stops it the way SIGTERM does: queued jobs are cancelled and
running ones interrupted (SIGINT, as Ctrl-C) and waited for. A second Ctrl-C or SIGTERM while it
stops kills every job's process group and exits at once (130); ``--stop`` sends it after waiting
``STOP_TIMEOUT_S``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import socket
import sys
import time
import webbrowser
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from oh_my_slam.commands.parser import ArgumentParser, run_main
from oh_my_slam.core.atomic import atomic_write_json
from oh_my_slam.core.errors import ExitCode, OhMySlamError, UsageError
from oh_my_slam.core.log import claim_stdout
from oh_my_slam.core.process import default_sigint
from oh_my_slam.server.lifecycle import AlreadyRunningError, ServerLock, pid_alive
from oh_my_slam.version import __version__
from oh_my_slam.web.workspace import DEFAULT_DATA, Workspace

PROG = "server.sh"
LOCK = "server.lock"
STATE = "server.json"
STOP_TIMEOUT_S = 180.0  # running jobs get the runner's grace period to stop
FORCE_TIMEOUT_S = 10.0


class ServiceNotRunningError(OhMySlamError):
    """``--status`` without a running service: exit 3, like ``start_inference_server.sh
    --status``."""

    exit_code = ExitCode.SERVER_UNAVAILABLE


def _say(msg: str) -> None:
    print(f"{PROG}: {msg}", file=sys.stderr, flush=True)


def build_parser() -> ArgumentParser:
    ap = ArgumentParser(prog=PROG, description="Local web service: the commands as an HTTP API "
                        "and a browser application.")
    ap.add_argument("--port", type=int, default=None,
                    help="port to bind on 0.0.0.0 (default: 0, a free port)")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA,
                    help=f"workspace folder for maps, uploads and results (default: "
                         f"{DEFAULT_DATA}/)")
    ap.add_argument("--no-browser", action="store_true",
                    help="do not open the default browser once listening")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--status", action="store_true",
                       help="print the running service's health JSON on stdout")
    group.add_argument("--stop", action="store_true", help="stop the running service")
    return ap


# -- state of the running service -------------------------------------------------------------------


def _state(ws: Workspace) -> dict[str, Any] | None:
    """The running service's state file, or None when no service holds the workspace's lock."""
    if not ServerLock.is_held(ws.root / LOCK):
        return None
    try:
        state = json.loads((ws.root / STATE).read_text())
    except (OSError, ValueError):
        return None
    return state if isinstance(state, dict) else None


def _local(url: str) -> str:
    """The service's URL as this machine reaches it (the browser and ``--status``)."""
    return url.replace("://0.0.0.0:", "://127.0.0.1:")


def status(ws: Workspace) -> int:
    import httpx

    state = _state(ws)
    if state is None:
        raise ServiceNotRunningError(f"no service is running for {ws.root} — start it with "
                                     f"./server.sh --data {ws.root}")
    try:
        r = httpx.get(_local(state["url"]) + "api/health", timeout=5.0)
        r.raise_for_status()
    except httpx.HTTPError as exc:
        raise ServiceNotRunningError(f"the service at {state['url']} does not answer ({exc})"
                                     ) from exc
    claim_stdout().write_json(r.json())
    return 0


def stop(ws: Workspace) -> int:
    state = _state(ws)
    pid = state.get("pid") if state else None
    if not isinstance(pid, int) or not pid_alive(pid):
        _say(f"not running for {ws.root}")
        return 0
    os.kill(pid, signal.SIGTERM)
    if _released(ws, STOP_TIMEOUT_S):
        _say(f"stopped (pid {pid})")
        return 0
    # a second signal kills the jobs' process groups and exits at once; SIGKILL as the last resort
    _say(f"pid {pid} did not stop within {STOP_TIMEOUT_S:.0f} s; killing its jobs")
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGTERM)
    if not _released(ws, FORCE_TIMEOUT_S):
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
        _released(ws, FORCE_TIMEOUT_S)
    _say(f"stopped (pid {pid}, forced)")
    return 0


def _released(ws: Workspace, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while ServerLock.is_held(ws.root / LOCK) and time.monotonic() < deadline:
        time.sleep(0.1)
    return not ServerLock.is_held(ws.root / LOCK)


# -- serving ----------------------------------------------------------------------------------------


def _bind(port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("0.0.0.0", port))
    except OSError as exc:
        sock.close()
        raise UsageError(f"--port {port}: cannot bind ({exc.strerror or exc}); use another port "
                         "or --port 0") from exc
    sock.listen(128)
    sock.set_inheritable(False)
    return sock


def serve(ws: Workspace, port: int, open_browser: bool) -> int:
    import uvicorn

    from oh_my_slam.web.app import Service, create_app
    from oh_my_slam.web.jobs import Runner

    default_sigint()  # jobs inherit a default SIGINT even if this process was started ignoring it
    lock = ServerLock(ws.root / LOCK)
    try:
        lock.acquire()
    except AlreadyRunningError:
        state = _state(ws) or {}
        url = state.get("url", "(starting)")
        _say(f"already running for {ws.root} at {url}")
        if open_browser and "url" in state:
            webbrowser.open(_local(url))
        return 0
    try:
        claim_stdout()  # nothing reaches stdout while serving
        sock = _bind(port)
        url = f"http://0.0.0.0:{sock.getsockname()[1]}/"
        ws.clear_uploads()  # no job can consume what an earlier run left
        runner = Runner(ws)
        runner.load()
        service = Service(ws, runner, url=url)
        app = create_app(service)

        def hard_stop(signum: int = 0, frame: object = None) -> None:
            """A second Ctrl-C or SIGTERM: kill every job's process group and exit now."""
            runner.kill_all()
            ws.clear_uploads()
            with contextlib.suppress(OSError):
                (ws.root / STATE).unlink()
            lock.release()
            os._exit(int(ExitCode.INTERRUPTED))

        class _Server(uvicorn.Server):
            async def startup(self, sockets: list[socket.socket] | None = None) -> None:
                await super().startup(sockets)
                if self.started:
                    atomic_write_json(ws.root / STATE, {
                        "pid": os.getpid(), "url": url, "port": sock.getsockname()[1],
                        "data": str(ws.root), "version": __version__,
                        "started_at": service.started_at})
                    _say(f"listening on {url}")
                    if open_browser:
                        webbrowser.open(_local(url))

            @contextlib.contextmanager
            def capture_signals(self) -> Iterator[None]:
                """Ctrl-C and SIGTERM stop the service normally (uvicorn would re-raise them
                after its shutdown, skipping the jobs' and the workspace's clean-up)."""
                for s in (signal.SIGINT, signal.SIGTERM):
                    signal.signal(s, self.stop_signal)
                try:
                    yield
                finally:  # straight to the hard stop: no default handler in between
                    for s in (signal.SIGINT, signal.SIGTERM):
                        signal.signal(s, hard_stop)

            def stop_signal(self, signum: int, frame: Any) -> None:
                if self.should_exit:
                    hard_stop()
                self.should_exit = True

            async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
                runner.stopping = True  # event streams end, so connections can close
                await super().shutdown(sockets)

        config = uvicorn.Config(app, log_config=None, log_level="warning", access_log=False,
                                lifespan="off", timeout_graceful_shutdown=5)
        server = _Server(config)
        try:
            server.run(sockets=[sock])
        finally:
            for s in (signal.SIGINT, signal.SIGTERM):  # a signal while the jobs stop: hard stop
                signal.signal(s, hard_stop)
            runner.shutdown()
            sock.close()
    finally:
        with contextlib.suppress(OSError):
            (ws.root / STATE).unlink()
        lock.release()
    return 0


def main(argv: list[str]) -> int:
    args: argparse.Namespace = build_parser().parse_args(argv)
    if (args.status or args.stop) and (args.port is not None or args.no_browser):
        raise UsageError("--status and --stop take only --data")
    port = 0 if args.port is None else args.port
    if not 0 <= port <= 65535:
        raise UsageError(f"--port must be between 0 and 65535, got {port}")
    ws = Workspace(args.data)
    if args.status:
        return status(ws)
    if args.stop:
        return stop(ws)
    ws.create()
    return serve(ws, port, not args.no_browser)


def entry() -> None:
    run_main(PROG, main)


if __name__ == "__main__":
    entry()
