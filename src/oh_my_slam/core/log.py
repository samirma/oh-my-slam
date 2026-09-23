"""stderr logging and the single stdout payload writer.

Commands call :func:`claim_stdout` first. It duplicates the real stdout file descriptor for the
payload and points fd 1 (and ``sys.stdout``) at stderr, so banners and progress printed by any
library — including C++ code in Open3D or COLMAP — land on stderr. The returned
:class:`PayloadWriter` accepts exactly one payload (one JSON document or one PLY).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, BinaryIO

_LOGGER_NAME = "oh_my_slam"


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """Return a logger writing to stderr (configured once)."""
    root = logging.getLogger(_LOGGER_NAME)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[oh-my-slam] %(message)s"))
        root.addHandler(handler)
        level = os.environ.get("OH_MY_SLAM_LOG", "INFO").upper()
        root.setLevel(getattr(logging, level, logging.INFO))
        root.propagate = False
    if name == _LOGGER_NAME:
        return root
    return root.getChild(name.removeprefix(_LOGGER_NAME + "."))


def json_payload_bytes(obj: Any) -> bytes:
    """Canonical JSON serialisation used for stdout and every JSON file we write."""
    return (json.dumps(obj, ensure_ascii=False, indent=None, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


class PayloadWriter:
    """Writes the single stdout payload to the saved real-stdout descriptor."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._written = False

    @property
    def written(self) -> bool:
        return self._written

    def write_bytes(self, data: bytes) -> None:
        if self._written:
            raise RuntimeError("stdout payload already written")
        self._written = True
        try:
            self._stream.write(data)
            self._stream.flush()
        except BrokenPipeError:
            # Consumer closed the pipe (e.g. `| head`); nothing more to do.
            pass

    def write_json(self, obj: Any) -> None:
        self.write_bytes(json_payload_bytes(obj))


_claimed: PayloadWriter | None = None


def claim_stdout() -> PayloadWriter:
    """Redirect fd 1 to stderr and return the writer for the real stdout (idempotent)."""
    global _claimed
    if _claimed is not None:
        return _claimed
    sys.stdout.flush()
    saved_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    stream = os.fdopen(saved_fd, "wb", buffering=0)
    _claimed = PayloadWriter(stream)
    return _claimed


@contextmanager
def timed(label: str, logger: logging.Logger | None = None) -> Iterator[None]:
    """Log the wall time of a block at DEBUG level."""
    start = time.perf_counter()
    try:
        yield
    finally:
        (logger or get_logger()).debug("%s: %.3f s", label, time.perf_counter() - start)
