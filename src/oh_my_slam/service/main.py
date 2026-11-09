"""``server.sh`` — the oh-my-slam HTTP service (api.md §1).

    server.sh [--port <n>] [--host <addr>] [--data <folder>] [--keep-versions <n>]
    server.sh --status | --stop

Binds ``--host`` (default ``0.0.0.0``) on ``--port`` (default 9091; 0: a free port), starts the
inference server if it is not running, and once it accepts connections writes one stderr line::

    server.sh: listening on http://<host>:<port>/

Its log goes to ``<data>/oms/service.log``. One service runs per data folder (a lock in
``<data>/oms``). ``--status`` prints ``GET /api/health``; ``--stop`` stops the service, and the
inference server when this service started it."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from oh_my_slam.core.atomic import atomic_write_json
from oh_my_slam.core.constants import SERVICE_PORT
from oh_my_slam.server.lifecycle import AlreadyRunningError, ServerLock, pid_alive
from oh_my_slam.service.store import Store
from oh_my_slam.version import __version__

PROG = "server.sh"
DEFAULT_DATA = Path("~/oh-my-slam-data")
DEFAULT_KEEP_VERSIONS = 20
STOP_TIMEOUT_S = 60.0
LOCK, STATE, LOG, INFERENCE_LOG = "service.lock", "service.json", "service.log", "inference.log"


def _say(msg: str) -> None:
    print(f"{PROG}: {msg}", file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=PROG, description="oh-my-slam HTTP service (specs/api.md)")
    p.add_argument("--port", type=int, default=SERVICE_PORT,
                   help=f"TCP port (default {SERVICE_PORT}; 0: a free port)")
    p.add_argument("--host", default="0.0.0.0", help="address to bind (default 0.0.0.0)")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA,
                   help="data folder (default ~/oh-my-slam-data)")
    p.add_argument("--keep-versions", type=int, default=DEFAULT_KEEP_VERSIONS,
                   help="versions kept per map (default 20; retention comes with O2)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--status", action="store_true", help="print GET /api/health")
    mode.add_argument("--stop", action="store_true", help="stop the service")
    return p


def git_version() -> str:
    """The checkout's commit, else the package version."""
    root = Path(__file__).resolve().parents[3]
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return __version__
    return out.stdout.strip() or __version__


def _state(store: Store) -> dict[str, Any] | None:
    if not ServerLock.is_held(store.root / LOCK):
        return None
    try:
        state = json.loads((store.root / STATE).read_text())
    except (OSError, ValueError):
        return None
    return state if isinstance(state, dict) else None


def status(store: Store) -> int:
    import httpx

    state = _state(store)
    if state is None:
        _say(f"not running for {store.root.parent}")
        return 3
    try:
        r = httpx.get(f"http://127.0.0.1:{state['port']}/api/health", timeout=5.0)
    except httpx.HTTPError as exc:
        _say(f"the service on port {state['port']} does not answer ({exc})")
        return 3
    sys.stdout.write(json.dumps(r.json(), indent=2) + "\n")
    return 0


def stop(store: Store) -> int:
    from oh_my_slam.service.inference import stop_inference

    state = _state(store)
    pid = state.get("pid") if state else None
    if not isinstance(pid, int) or not pid_alive(pid):
        _say(f"not running for {store.root.parent}")
        return 0
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + STOP_TIMEOUT_S
    while ServerLock.is_held(store.root / LOCK) and time.monotonic() < deadline:
        time.sleep(0.1)
    if ServerLock.is_held(store.root / LOCK):
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
    _say(f"stopped (pid {pid})")
    if state and state.get("started_inference"):
        stop_inference(store.root / INFERENCE_LOG)
        _say("stopped the inference server it started")
    return 0


def _bind(host: str, port: int) -> socket.socket:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError as exc:
        sock.close()
        raise SystemExit(f"{PROG}: error: cannot bind {host}:{port} ({exc.strerror or exc}); "
                         "use another --port") from exc
    sock.listen(128)
    return sock


def _log_to(path: Path) -> None:
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "oh_my_slam", ""):
        logger = logging.getLogger(name)
        logger.handlers = [handler]
        logger.propagate = False
    logging.getLogger().setLevel(logging.INFO)


def serve(store: Store, host: str, port: int) -> int:
    import uvicorn

    from oh_my_slam.service.app import Service, create_app
    from oh_my_slam.service.backend import MapperBackend
    from oh_my_slam.service.inference import Inference
    from oh_my_slam.service.jobs import Jobs
    from oh_my_slam.service.update import Updater

    store.create()
    lock = ServerLock(store.root / LOCK)
    try:
        lock.acquire()
    except AlreadyRunningError:
        _say(f"already running for {store.root.parent} on port "
             f"{(_state(store) or {}).get('port', '?')}")
        return 1
    try:
        sock = _bind(host, port)
        _log_to(store.root / LOG)
        store.clear_scratch()
        backend = MapperBackend()
        jobs = Jobs(store.jobs, Updater(store, backend))
        jobs.load()
        inference = Inference(store.root / INFERENCE_LOG)
        inference.ensure_started()
        service = Service(store, jobs, inference, backend.require_inference,
                          version=git_version())
        app = create_app(service)
        bound = sock.getsockname()[1]
        url = f"http://{host}:{bound}/"

        class _Server(uvicorn.Server):
            async def startup(self, sockets: list[socket.socket] | None = None) -> None:
                await super().startup(sockets)
                atomic_write_json(store.root / STATE, {
                    "pid": os.getpid(), "url": url, "host": host, "port": bound,
                    "data": str(store.root.parent), "version": service.version,
                    "started_inference": inference.started, "started_at": time.time()})
                _say(f"listening on {url}")

            @contextlib.contextmanager
            def capture_signals(self) -> Iterator[None]:
                """Ctrl-C and SIGTERM stop the service normally (uvicorn would raise them
                again after its shutdown, skipping the jobs' clean-up)."""
                for s in (signal.SIGINT, signal.SIGTERM):
                    signal.signal(s, lambda signum, frame: setattr(self, "should_exit", True))
                yield

            async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
                service.stopping = True
                await super().shutdown(sockets)

        jobs.start()
        server = _Server(uvicorn.Config(app, log_config=None, access_log=False,
                                        lifespan="off", timeout_graceful_shutdown=10))
        try:
            server.run(sockets=[sock])
        finally:
            backend.terminate()
            jobs.stop(timeout=STOP_TIMEOUT_S)
            sock.close()
    finally:
        with contextlib.suppress(OSError):
            (store.root / STATE).unlink()
        lock.release()
    return 0


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    if not 0 <= args.port <= 65535:
        _say(f"error: --port must be between 0 and 65535, got {args.port}")
        return 2
    store = Store(args.data)
    if args.status:
        return status(store)
    if args.stop:
        return stop(store)
    return serve(store, args.host, args.port)


def entry() -> None:
    sys.exit(main(sys.argv[1:]))


if __name__ == "__main__":
    entry()
