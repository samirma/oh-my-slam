"""Server process entry point: ``python -m oh_my_slam.server.main [--stub]``.

Takes the single-instance lock, binds the 0600 Unix socket, starts the HTTP server immediately
(``/health`` says ``loading``) and loads the models on the GPU worker thread.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from importlib import metadata
from types import FrameType

from oh_my_slam.core import paths
from oh_my_slam.server.app import ServerState, create_app
from oh_my_slam.server.gpu_worker import GpuWorker
from oh_my_slam.server.lifecycle import (
    AlreadyRunningError,
    ServerLock,
    bind_socket,
    cleanup_stale_socket,
    clear_state,
    write_state,
)
from oh_my_slam.server.models import build_registry, select_device
from oh_my_slam.version import __version__

log = logging.getLogger("oh_my_slam.server")

_VERSION_PACKAGES = ("torch", "ultralytics", "transformers", "moge", "geocalib", "mapanything")
MEMORY_FRACTION = 0.7


def _versions() -> dict[str, str]:
    out = {"oh-my-slam": __version__}
    for pkg in _VERSION_PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def _load_models(state: ServerState, stub: bool) -> None:
    """Runs on the GPU thread: device selection, memory cap, model loading, warm-up."""
    t0 = time.perf_counter()
    try:
        device = "cpu" if stub else select_device()
        if device == "mps" and not stub:
            import torch

            torch.mps.set_per_process_memory_fraction(MEMORY_FRACTION)
        state.registry.load_all(device, log=log.info)
    finally:
        state.loading = False
        log.info("models loaded in %.1f s: status %s", time.perf_counter() - t0, state.status())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="oh_my_slam.server.main")
    parser.add_argument("--stub", action="store_true", help="deterministic stand-in models")
    args = parser.parse_args(argv)
    stub = args.stub or os.environ.get("OH_MY_SLAM_SERVER_STUB") == "1"

    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    lock = ServerLock()
    try:
        lock.acquire()
    except AlreadyRunningError as exc:
        log.error("%s", exc)
        return 1
    sock_path = paths.socket_path()
    try:
        cleanup_stale_socket(sock_path)
        os.chdir(paths.weights_dir())  # Ultralytics downloads relative assets into the cwd
        registry = build_registry(stub=stub)
        worker = GpuWorker(max_queue=int(os.environ.get("OH_MY_SLAM_QUEUE", "8")))
        worker.start()
        state = ServerState(registry=registry, worker=worker, versions=_versions())
        app = create_app(state)

        import uvicorn

        sock = bind_socket(sock_path)
        config = uvicorn.Config(app, log_level="warning", access_log=False, lifespan="off")
        server = uvicorn.Server(config)

        def on_signal(signum: int, frame: FrameType | None) -> None:
            state.stopping = True
            server.should_exit = True

        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGINT, on_signal)
        worker.submit(_load_models, state, stub)
        write_state(sock_path, __version__, paths.server_log())
        log.info("listening on %s (pid %d)", sock_path, os.getpid())
        # uvicorn swaps in its own SIGINT/SIGTERM handlers while serving and re-raises the signal
        # to ours afterwards, which only marks the state as stopping; cleanup runs below.
        server.run(sockets=[sock])
        state.stopping = True
        worker.stop(timeout=10.0)
        return 0
    finally:
        clear_state(sock_path)
        lock.release()
        log.info("server stopped")


if __name__ == "__main__":
    sys.exit(main())
