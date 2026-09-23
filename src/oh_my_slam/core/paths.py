"""Well-known locations: server runtime files, model weights, external tools, data."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

# macOS limits AF_UNIX paths to 104 bytes including the terminator.
_MAX_SOCKET_PATH = 100


def runtime_dir() -> Path:
    env = os.environ.get("OH_MY_SLAM_RUNTIME_DIR")
    base = Path(env) if env else Path.home() / "Library" / "Caches" / "oh-my-slam"
    base.mkdir(parents=True, exist_ok=True)
    return base


def socket_path() -> Path:
    """Unix socket of the inference server; falls back to a short /tmp path if too long."""
    path = runtime_dir() / "srv.sock"
    if len(str(path).encode()) <= _MAX_SOCKET_PATH:
        return path
    digest = hashlib.sha256(str(runtime_dir()).encode()).hexdigest()[:10]
    short = Path("/tmp") / f"oms-{digest}"
    short.mkdir(mode=0o700, parents=True, exist_ok=True)
    return short / "srv.sock"


def state_file() -> Path:
    return runtime_dir() / "server.json"


def lock_file() -> Path:
    return runtime_dir() / "server.lock"


def server_log() -> Path:
    return runtime_dir() / "server.log"


def weights_dir() -> Path:
    env = os.environ.get("OH_MY_SLAM_WEIGHTS_DIR")
    base = Path(env) if env else Path.home() / "Library" / "Caches" / "oh-my-slam" / "weights"
    base.mkdir(parents=True, exist_ok=True)
    return base


def tools_dir() -> Path:
    env = os.environ.get("OH_MY_SLAM_TOOLS_DIR")
    return Path(env) if env else Path.home() / ".local" / "share" / "oh-my-slam" / "tools"


def openmvs_dir() -> Path:
    env = os.environ.get("OH_MY_SLAM_OPENMVS_DIR")
    return Path(env) if env else tools_dir() / "openmvs-2.4.0"


def data_dir() -> Path:
    env = os.environ.get("OH_MY_SLAM_DATA_DIR")
    return Path(env) if env else Path.home() / "oh-my-slam-data"


def scratch_dir() -> Path:
    """Per-user scratch space for request files (depth maps written by the server)."""
    base = runtime_dir() / "scratch"
    base.mkdir(parents=True, exist_ok=True)
    return base
