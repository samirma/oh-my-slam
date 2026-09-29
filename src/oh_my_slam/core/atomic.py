"""Crash-safe file writes: temp file in the same directory, fsync, rename."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from oh_my_slam.core.errors import UsageError

_UMASK = os.umask(0)
os.umask(_UMASK)
_FILE_MODE = 0o666 & ~_UMASK  # mkstemp creates 0600; published files get the usual mode


def preflight_dir(folder: Path, option: str) -> None:
    """Create ``folder`` (with its parents) and prove that a file can be written in it, so that
    ``-o`` / ``-d`` targets fail before any work starts; a usage error (exit 2) names the path."""
    folder = Path(folder)
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=folder):
            pass
    except OSError as exc:
        raise UsageError(f"{option} {folder}: cannot write there ({exc.strerror or exc})") from exc


def preflight_file(path: Path, option: str) -> None:
    """``preflight_dir`` for the folder of a file the command will write, which must not be a
    folder itself."""
    path = Path(path)
    if path.is_dir():
        raise UsageError(f"{option} {path} is a folder; give the path of the file to write")
    preflight_dir(path.parent, option)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.fchmod(fd, _FILE_MODE)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, obj: Any) -> None:
    from oh_my_slam.core.log import json_payload_bytes

    atomic_write_bytes(path, json_payload_bytes(obj))


def atomic_save_npy(path: Path, array: np.ndarray[Any, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npy", dir=path.parent)
    os.fchmod(fd, _FILE_MODE)
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, array, allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
