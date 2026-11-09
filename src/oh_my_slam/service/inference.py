"""The inference server as the service sees it (inference.md): its state for ``/api/health``,
started with the service when it is not running, and stopped by ``server.sh --stop`` when the
service started it. The service talks to it only through its client and its command line
(``start_inference_server`` in ``oh_my_slam.cli.server``)."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from oh_my_slam.client.client import InferenceClient
from oh_my_slam.core.errors import ServerUnavailableError
from oh_my_slam.server.lifecycle import ServerLock
from oh_my_slam.service.errors import ApiError

GIB_TO_GB = 1.073741824
START_GRACE_S = 1200.0  # a first start downloads the weights
STATES = {"ready": "ready", "degraded": "degraded", "loading": "starting", "error": "down",
          "stopping": "down"}


def inference_cmd() -> list[str]:
    """The inference server's command line; ``OH_MY_SLAM_INFERENCE_CMD`` replaces it (the
    tests, which never start the real server)."""
    env = os.environ.get("OH_MY_SLAM_INFERENCE_CMD")
    return shlex.split(env) if env else [sys.executable, "-m", "oh_my_slam.cli.server"]


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
    def __init__(self, log: Path) -> None:
        self.log = log
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
        with self.log.open("ab") as out:
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


def stop_inference(log: Path) -> int:
    """Stop the inference server (its command line's ``--stop``)."""
    with log.open("ab") as out:
        return subprocess.run([*inference_cmd(), "--stop"], stdin=subprocess.DEVNULL, stdout=out,
                              stderr=out).returncode
