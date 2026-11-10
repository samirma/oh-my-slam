"""The inference server as the service sees it (inference.md): its state for ``/api/health``,
started with the service when it is not running (``python -m oh_my_slam.server.main``, its log
``core.paths.server_log()``), and stopped by ``server.sh --stop`` when the service started it.
The service talks to it through its client and its lock (``server.lifecycle``) only."""

from __future__ import annotations

import contextlib
import os
import shlex
import signal
import subprocess
import sys
import time
from typing import Any

from oh_my_slam.client.client import InferenceClient
from oh_my_slam.core import paths
from oh_my_slam.core.errors import ServerUnavailableError
from oh_my_slam.server.lifecycle import ServerLock
from oh_my_slam.service.errors import ApiError

GIB_TO_GB = 1.073741824
START_GRACE_S = 1200.0  # a first start downloads the weights
STATES = {"ready": "ready", "degraded": "degraded", "loading": "starting", "error": "down",
          "stopping": "down"}


STOP_TIMEOUT_S = 12.0


def inference_cmd() -> list[str]:
    """The inference server's command line; ``OH_MY_SLAM_INFERENCE_CMD`` replaces it (the
    tests, which never start the real server)."""
    env = os.environ.get("OH_MY_SLAM_INFERENCE_CMD")
    return shlex.split(env) if env else [sys.executable, "-m", "oh_my_slam.server.main"]


def inference_ready() -> None:
    """``inference_unavailable`` unless the inference server answers ``ready`` or
    ``degraded``."""
    try:
        h = InferenceClient().health(timeout=2.0)
    except ServerUnavailableError as exc:
        raise ApiError("inference_unavailable", f"the inference server is not running ({exc}); "
                       "server.sh starts it — retry in a moment", {"state": "down"}) from exc
    if h.status not in ("ready", "degraded"):
        raise ApiError("inference_unavailable", f"the inference server is {h.status}; retry in a "
                       "moment", {"state": STATES.get(h.status, "down")})


class Inference:
    def __init__(self) -> None:
        self.launcher: subprocess.Popen[bytes] | None = None
        self.launched_at: float | None = None

    @property
    def started(self) -> bool:
        """Whether this service started the inference server."""
        return self.launched_at is not None

    def ensure_started(self) -> None:
        """Start the inference server in the background unless it answers or is starting."""
        try:
            InferenceClient().health(timeout=1.0)
            return
        except ServerUnavailableError:
            pass
        if ServerLock.is_held():
            return  # its process holds the lock and loads its models
        with paths.server_log().open("ab") as out:
            self.launcher = subprocess.Popen(inference_cmd(), stdin=subprocess.DEVNULL, stdout=out,
                                             stderr=out, start_new_session=True)
        self.launched_at = time.time()

    def _starting(self) -> bool:
        if ServerLock.is_held():
            return True
        return (self.launcher is not None and self.launcher.poll() is None
                and self.launched_at is not None and time.time() - self.launched_at < START_GRACE_S)

    def state(self) -> dict[str, Any]:
        """The ``Health.inference`` object."""
        try:
            h = InferenceClient().health(timeout=1.0)
        except ServerUnavailableError as exc:
            return {"state": "starting" if self._starting() else "down", "device": "unknown",
                    "memory_used_gb": None, "memory_cap_gb": None, "error": str(exc)}
        mem = h.memory
        error = None
        if h.status == "error":
            error = h.failures() or "models failed to load"
        elif h.status == "degraded":
            error = h.memory_note()
        return {"state": STATES.get(h.status, "down"), "device": h.device,
                "memory_used_gb": None if mem is None else round(mem.allocated_gib * GIB_TO_GB, 3),
                "memory_cap_gb": None if mem is None else round(mem.limit_gib * GIB_TO_GB, 3),
                "error": error}


def stop_inference() -> str:
    """Stop the inference server: SIGTERM to the process that holds its lock (its pid is in the
    lock file), SIGKILL after ``STOP_TIMEOUT_S``. Returns what happened."""
    if not ServerLock.is_held():
        paths.socket_path().unlink(missing_ok=True)
        paths.state_file().unlink(missing_ok=True)
        return "the inference server was not running"
    try:
        pid = int(paths.lock_file().read_text().strip())
    except (OSError, ValueError):
        return f"a process holds {paths.lock_file()} without its pid; not stopping it"
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + STOP_TIMEOUT_S
    while ServerLock.is_held() and time.monotonic() < deadline:
        time.sleep(0.1)
    if ServerLock.is_held():
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
        return f"stopped the inference server (pid {pid}, killed)"
    return f"stopped the inference server (pid {pid})"
