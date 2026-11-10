"""Crash-safe file writes: temp file in the same directory, fsync, rename."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

_UMASK = os.umask(0)
os.umask(_UMASK)
_FILE_MODE = 0o666 & ~_UMASK  # mkstemp creates 0600; published files get the usual mode


def clone_file(src: Path, dst: Path) -> None:
    """Copy ``src`` to ``dst`` as an APFS clone (``cp -c``: instant, no extra space) when the
    filesystem allows it, else as a plain copy."""
    res = subprocess.run(["cp", "-c", str(src), str(dst)], capture_output=True)
    if res.returncode != 0:
        shutil.copyfile(src, dst)


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
    atomic_write_bytes(path, json_payload_bytes(obj))


def json_payload_bytes(obj: Any) -> bytes:
    """Canonical JSON serialisation of every JSON file the package writes."""
    return (json.dumps(obj, ensure_ascii=False, indent=None, separators=(",", ":")) + "\n"
            ).encode("utf-8")


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
