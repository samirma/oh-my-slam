"""Crash-safe file writes: temp file in the same directory, fsync, rename."""

from __future__ import annotations

import errno
import os
import shutil
import subprocess
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
    if _check_file(Path(path), option):
        preflight_dir(Path(path).parent, option)


def check_dir(folder: Path, option: str) -> None:
    """``preflight_dir`` without touching the filesystem: the same usage errors for the cases it
    can see (a file in the way, a folder it may not write); ``preflight_dir`` still has the last
    word when the command runs."""
    folder = Path(folder)
    probe = folder
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if probe.exists() and not probe.is_dir():
        code = errno.EEXIST if probe == folder else errno.ENOTDIR
        raise UsageError(f"{option} {folder}: cannot write there ({os.strerror(code)})")
    if not os.access(probe, os.W_OK | os.X_OK):
        raise UsageError(f"{option} {folder}: cannot write there ({os.strerror(errno.EACCES)})")


def check_file(path: Path, option: str) -> None:
    """``preflight_file`` without touching the filesystem (see ``check_dir``)."""
    if _check_file(Path(path), option):
        check_dir(Path(path).parent, option)


def _check_file(path: Path, option: str) -> bool:
    """The checks of a file target itself; True when its folder must still be checked."""
    if path.is_dir():
        raise UsageError(f"{option} {path} is a folder; give the path of the file to write")
    if _is_special(path):
        if not os.access(path, os.W_OK):
            raise UsageError(f"{option} {path}: cannot write there (permission denied)")
        return False
    return True


def _is_special(path: Path) -> bool:
    """An existing file that is not a regular file (``/dev/null``, a FIFO): written in place, since
    it cannot be replaced by a rename."""
    return path.exists() and not path.is_file()


def clone_file(src: Path, dst: Path) -> None:
    """Copy ``src`` to ``dst`` as an APFS clone (``cp -c``: instant, no extra space) when the
    filesystem allows it, else as a plain copy."""
    res = subprocess.run(["cp", "-c", str(src), str(dst)], capture_output=True)
    if res.returncode != 0:
        shutil.copyfile(src, dst)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    if _is_special(path):
        with path.open("wb") as f:
            f.write(data)
        return
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
