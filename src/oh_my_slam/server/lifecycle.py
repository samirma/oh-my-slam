"""Single-instance lock, state file and stale-socket cleanup for the inference server."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

from oh_my_slam.core import paths
from oh_my_slam.core.atomic import atomic_write_json


class AlreadyRunningError(RuntimeError):
    pass


class ServerLock:
    """Exclusive ``flock`` on the lock file, held for the lifetime of the server process."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or paths.lock_file()
        self._fd: int | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise AlreadyRunningError(f"another server holds {self.path}") from exc
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        self._fd = fd

    def release(self) -> None:
        if self._fd is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None

    @staticmethod
    def is_held(path: Path | None = None) -> bool:
        path = path or paths.lock_file()
        if not path.exists():
            return False
        fd = os.open(path, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
        finally:
            os.close(fd)


def socket_is_live(sock_path: Path, timeout: float = 0.5) -> bool:
    if not sock_path.exists():
        return False
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(str(sock_path))
        return True
    except OSError:
        return False
    finally:
        s.close()


def cleanup_stale_socket(sock_path: Path) -> bool:
    """Remove a socket nobody listens on. Call only while holding the server lock."""
    if sock_path.exists() and not socket_is_live(sock_path):
        sock_path.unlink(missing_ok=True)
        return True
    return False


def bind_socket(sock_path: Path) -> socket.socket:
    """Bind the listening Unix socket with mode 0600 (umask applied at bind time)."""
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    old = os.umask(0o177)
    try:
        s.bind(str(sock_path))
    finally:
        os.umask(old)
    os.chmod(sock_path, 0o600)
    s.listen(64)
    return s


def write_state(sock_path: Path, version: str, log_path: Path | None = None) -> None:
    atomic_write_json(
        paths.state_file(),
        {
            "pid": os.getpid(),
            "socket": str(sock_path),
            "version": version,
            "started_at": time.time(),
            "log": str(log_path) if log_path else None,
        },
    )


def read_state() -> dict[str, Any] | None:
    try:
        return json.loads(paths.state_file().read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def clear_state(sock_path: Path) -> None:
    state = read_state()
    if state is None or state.get("pid") == os.getpid():
        paths.state_file().unlink(missing_ok=True)
    sock_path.unlink(missing_ok=True)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
