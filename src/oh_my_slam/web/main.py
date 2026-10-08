"""``server.sh`` — the local web service (spec §2.6).

    server.sh [--port <n>] [--data <folder>] [--no-browser]
    server.sh --status | --stop

Binds ``0.0.0.0:<port>`` (default 0: a free port), serves the API and the web application until
Ctrl-C or SIGTERM, and once it accepts connections writes exactly one stderr line::

    server.sh: listening on http://0.0.0.0:<port>/

Nothing goes to stdout except ``--status``'s health JSON. One service runs per workspace (a lock
and a state file in ``<data>``): a second ``server.sh`` on the same ``--data`` reports the running
one's URL and exits 0. ``--stop`` stops it the way SIGTERM does: every request in progress is
interrupted — a waiting one never starts, a running command gets SIGINT, as Ctrl-C — and the
running ones are waited for, so each still gets its answer. A second Ctrl-C or SIGTERM while it
stops kills every command's process group and exits at once (130); ``--stop`` sends it after
waiting ``STOP_TIMEOUT_S``. Its options and modes are defined in ``commands.entry_points``
(``WEB_SERVICE``).
"""

from __future__ import annotations

import argparse
import contextlib
import ipaddress
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

from oh_my_slam.commands import spec
from oh_my_slam.commands.entry_points import WEB_SERVICE
from oh_my_slam.commands.parser import ArgumentParser, run_main
from oh_my_slam.core.atomic import atomic_write_json
from oh_my_slam.core.constants import CHECKOUT
from oh_my_slam.core.errors import (
    ExitCode,
    OhMySlamError,
    ServiceNotRunningError,
    UsageError,
    internal_message,
)
from oh_my_slam.core.log import claim_stdout
from oh_my_slam.core.process import default_sigint, exit_now
from oh_my_slam.server.lifecycle import AlreadyRunningError, ServerLock, pid_alive
from oh_my_slam.version import __version__
from oh_my_slam.web.workspace import DEFAULT_DATA, Workspace

PROG = WEB_SERVICE.prog
LOCK = "server.lock"
STATE = "server.json"
LOG = "server.log"
LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access", "oh_my_slam", "")  # "" is the root
_terminal: int | None = None  # the original stderr, kept once logging went to server.log
STOP_TIMEOUT_S = 180.0  # interrupted commands get the runner's grace periods to stop
FORCE_TIMEOUT_S = 10.0
PROBE_TIMEOUT_S = 0.5  # a connection to one of this machine's own addresses answers within this


def _say(msg: str) -> None:
    print(f"{PROG}: {msg}", file=sys.stderr, flush=True)


def build_parser() -> ArgumentParser:
    return spec.build_parser(WEB_SERVICE)


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
                                     f"{CHECKOUT}{PROG} --data {ws.root}")
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
    # a second signal kills the commands' process groups and exits at once; SIGKILL as the last
    # resort
    _say(f"pid {pid} did not stop within {STOP_TIMEOUT_S:.0f} s; killing its commands")
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


def _log_to(path: Path) -> None:
    """After the listening line nothing more reaches stderr (spec §2.6: exactly one line): the
    uvicorn and oh_my_slam loggers (and the root, for asyncio and the rest) write to ``path``, and
    so does anything else written to file descriptors 1 and 2 — warnings, a thread's traceback,
    a library's own output. The commands are unaffected: their stderr is a pipe the service
    reads."""
    import logging

    global _terminal
    stream = path.open("a", buffering=1, encoding="utf-8", errors="backslashreplace")
    _terminal = os.dup(2)  # for the one line of a failed exit (``_tell_terminal``)
    os.dup2(stream.fileno(), 2)
    os.dup2(stream.fileno(), 1)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    for name in LOGGERS:
        logger = logging.getLogger(name)
        logger.handlers = [handler]
        logger.propagate = False
    logging.getLogger().setLevel(logging.WARNING)
    logging.captureWarnings(True)  # py.warnings → the root logger → the file


def _tell_terminal(ws: Workspace, what: str) -> None:
    """A non-zero exit after the listening line: one ``server.sh: error:`` line on the original
    stderr, pointing at the log that has the details."""
    if _terminal is None:
        return
    line = f"{PROG}: error: {what} (see {ws.root / LOG})\n"
    with contextlib.suppress(OSError):
        os.write(_terminal, line.encode("utf-8", "backslashreplace"))


def _failure(exc: BaseException) -> str:
    if isinstance(exc, OhMySlamError):
        return str(exc)
    if isinstance(exc, KeyboardInterrupt):
        return "interrupted"
    return f"internal error: {internal_message(exc)}"


def _binds(family: int, port: int) -> bool:
    """Whether ``port`` binds on every address of ``family`` without ``SO_REUSEADDR``: nothing
    holds it on any of them, not even the closed connections of a service that just stopped."""
    try:
        sock = socket.socket(family, socket.SOCK_STREAM)
    except OSError:  # no IPv6 on this machine: nothing listens there
        return True
    with sock:
        if family == socket.AF_INET6:
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        try:
            sock.bind(("::" if family == socket.AF_INET6 else "0.0.0.0", port))
        except OSError:
            return False
    return True


def _listener(port: int) -> str | None:
    """The first of this machine's addresses (loopback first) at which another process accepts
    connections on ``port``, if any."""
    from oh_my_slam.web.app import machine_hosts

    addresses = []
    for host in machine_hosts():
        with contextlib.suppress(ValueError):  # a name, not an address
            addresses.append(ipaddress.ip_address(host))
    for ip in sorted(addresses, key=lambda a: (not a.is_loopback, a.version, str(a))):
        if ip.is_unspecified:
            continue
        family = socket.AF_INET6 if ip.version == 6 else socket.AF_INET
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.settimeout(PROBE_TIMEOUT_S)
            if probe.connect_ex((str(ip), port)) == 0:
                return str(ip)
    return None


def _in_use(port: int) -> str | None:
    """The address at which another process listens on ``port``, IPv4 or IPv6 (a browser's
    ``localhost`` may be either), else None. A port that binds without ``SO_REUSEADDR`` in both
    families is free; one that does not may hold only the closed connections of a service that
    just stopped (TIME_WAIT), so each address of the machine is then asked whether it accepts."""
    if _binds(socket.AF_INET, port) and _binds(socket.AF_INET6, port):
        return None
    return _listener(port)


def _bind(port: int) -> socket.socket:
    """The service's socket on ``0.0.0.0:<port>``, with ``SO_REUSEADDR`` so that a service that
    just stopped can be started again on its port at once. macOS then binds it even while another
    process listens on one of the machine's addresses, which would get the connections to that
    address (the browser's and ``--status``'s ``127.0.0.1``): an explicit port another process
    listens on, on any address, is refused."""
    where = _in_use(port) if port else None
    if where is not None:
        raise UsageError(f"--port {port}: cannot bind (another process listens on it at {where}); "
                         "use another port or --port 0")
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
    from oh_my_slam.web.runner import Runner

    default_sigint()  # commands inherit a default SIGINT even if this was started ignoring it
    lock = ServerLock(ws.root / LOCK)
    try:
        lock.acquire()
    except AlreadyRunningError:
        url = (_state(ws) or {}).get("url", "(starting)")
        _say(f"already running for {ws.root} at {url}")
        return 0
    try:
        claim_stdout()  # nothing reaches stdout while serving
        sock = _bind(port)
        url = f"http://0.0.0.0:{sock.getsockname()[1]}/"
        ws.clear_uploads()  # no request can consume what an earlier run left
        ws.clear_requests()
        runner = Runner(ws)
        service = Service(ws, runner, url=url)
        app = create_app(service)

        def hard_stop(signum: int = 0, frame: object = None) -> None:
            """A second Ctrl-C or SIGTERM: kill every command's process group and exit now."""
            runner.kill_all()
            ws.clear_uploads()
            ws.clear_requests()
            with contextlib.suppress(OSError):
                (ws.root / STATE).unlink()
            lock.release()
            _tell_terminal(ws, "stopped by a second signal; every command's processes were "
                           "killed")
            exit_now(int(ExitCode.INTERRUPTED))

        class _Server(uvicorn.Server):
            async def startup(self, sockets: list[socket.socket] | None = None) -> None:
                await super().startup(sockets)  # returns once started (else it raises)
                atomic_write_json(ws.root / STATE, {
                    "pid": os.getpid(), "url": url, "port": sock.getsockname()[1],
                    "data": str(ws.root), "version": __version__,
                    "started_at": service.started_at})
                _say(f"listening on {url}")
                _log_to(ws.root / LOG)
                if open_browser:
                    webbrowser.open(_local(url))

            @contextlib.contextmanager
            def capture_signals(self) -> Iterator[None]:
                """Ctrl-C and SIGTERM stop the service normally (uvicorn would re-raise them
                after its shutdown, skipping the requests' and the workspace's clean-up)."""
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
                runner.stop()  # every request in progress is interrupted, then answered
                await super().shutdown(sockets)

        # the open requests get their answer once their interrupted commands end (SIGTERM and
        # SIGKILL follow a command that ignores SIGINT); a task still open after that is cancelled
        config = uvicorn.Config(app, log_config=None, log_level="warning", access_log=False,
                                lifespan="off",
                                timeout_graceful_shutdown=int(2 * runner.interrupt_grace_s) + 5)
        server = _Server(config)
        try:
            server.run(sockets=[sock])
        finally:
            for s in (signal.SIGINT, signal.SIGTERM):  # a signal while requests stop: hard stop
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
    spec.validate(WEB_SERVICE.command(), args)  # --port and --no-browser: the serve mode's only
    port = 0 if args.port is None else args.port
    if not 0 <= port <= 65535:
        raise UsageError(f"--port must be between 0 and 65535, got {port}")
    ws = Workspace(args.data or DEFAULT_DATA)
    if args.status:
        return status(ws)
    if args.stop:
        return stop(ws)
    ws.create()
    try:
        return serve(ws, port, not args.no_browser)
    except BaseException as exc:  # run_main logs the details (to server.log once listening)
        if not (isinstance(exc, SystemExit) and exc.code in (0, None)):
            _tell_terminal(ws, _failure(exc))
        raise


def entry() -> None:
    run_main(PROG, main)


if __name__ == "__main__":
    entry()
