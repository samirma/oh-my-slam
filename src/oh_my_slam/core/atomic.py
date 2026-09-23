"""Crash-safe file writes: temp file in the same directory, fsync, rename."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

_UMASK = os.umask(0)
os.umask(_UMASK)
_FILE_MODE = 0o666 & ~_UMASK  # mkstemp creates 0600; published files get the usual mode


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


def replace_dir(src: Path, dst: Path) -> None:
    """Move directory ``src`` over ``dst`` (dst removed first; not atomic across the two steps)."""
    dst = Path(dst)
    if dst.exists():
        trash = dst.with_name(f".{dst.name}.old")
        if trash.exists():
            shutil.rmtree(trash)
        dst.rename(trash)
        Path(src).rename(dst)
        shutil.rmtree(trash, ignore_errors=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        Path(src).rename(dst)


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
