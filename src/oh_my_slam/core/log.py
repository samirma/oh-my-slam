"""stderr logging and the single payload writer.

Commands call :func:`claim_stdout` once their arguments are parsed and checked, before any
inference, mapping or serving. It duplicates the real stdout file descriptor for the payload and
points fd 1 (and ``sys.stdout``) at stderr, so banners and progress printed from then on by any
library — including C++ code in Open3D or COLMAP — land on stderr. The returned
:class:`PayloadWriter` accepts exactly one payload (one JSON document, one PLY or one PNG) and
writes it to the real stdout, or — for ``-o <file>`` — atomically to that file, leaving stdout
empty.

A command's warnings (records of level WARNING and above) go to stderr like every other line;
when ``OH_MY_SLAM_WARNINGS=<path>`` is set, each one is also appended to that file as one JSON
line ``{"message": …}``, which is how the web service (spec §2.6) passes them on with a result.
"""

from __future__ import annotations

import json
import logging
import os
import select
import sys
from pathlib import Path
from typing import Any, BinaryIO

from oh_my_slam.core.atomic import atomic_write_bytes, preflight_file

_LOGGER_NAME = "oh_my_slam"
WARNINGS_ENV = "OH_MY_SLAM_WARNINGS"


class WarningsFile(logging.Handler):
    """Appends each warning to ``path``, one JSON line ``{"message": …}`` each."""

    def __init__(self, path: str) -> None:
        super().__init__(logging.WARNING)
        self.path = path

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = json.dumps({"message": record.getMessage()}, ensure_ascii=False)
            with Path(self.path).open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            self.handleError(record)


def read_warnings(path: Path) -> list[str]:
    """The messages a :class:`WarningsFile` appended to ``path`` (none when there is no file); a
    line that is no such record is taken as it is."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    out = []
    for line in filter(None, lines):
        try:
            out.append(str(json.loads(line)["message"]))
        except (ValueError, KeyError, TypeError):
            out.append(line)
    return out


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """Return a logger writing to stderr (configured once), and its warnings to the file
    ``OH_MY_SLAM_WARNINGS`` names, if any."""
    root = logging.getLogger(_LOGGER_NAME)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[oh-my-slam] %(message)s"))
        root.addHandler(handler)
        target = os.environ.get(WARNINGS_ENV)
        if target:
            root.addHandler(WarningsFile(target))
        root.setLevel(logging.INFO)
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
    """Writes the single payload: to a stream (the saved real stdout) or, when ``path`` is given,
    atomically to that file (replaced if it exists; its missing parent folders are created only
    then, so a command that fails first leaves none behind). A file that cannot be written is a
    usage error, raised when the writer is made, which creates nothing."""

    def __init__(self, stream: BinaryIO | None = None, path: Path | None = None) -> None:
        if (stream is None) == (path is None):
            raise ValueError("give exactly one of stream and path")
        if path is not None:
            preflight_file(Path(path), "-o")  # before any work: exit 2 rather than a late failure
        self._stream = stream
        self._path = None if path is None else Path(path)
        self._written = False

    def write_bytes(self, data: bytes) -> None:
        if self._written:
            raise RuntimeError("payload already written")
        self._written = True
        if self._path is not None:
            atomic_write_bytes(self._path, data)
            return
        assert self._stream is not None
        try:
            _write_all(self._stream, data)
        except BrokenPipeError:
            # Consumer closed the pipe (e.g. `| head`); nothing more to do.
            pass

    def write_json(self, obj: Any) -> None:
        self.write_bytes(json_payload_bytes(obj))


def _write_all(stream: BinaryIO, data: bytes) -> None:
    """Write every byte of ``data``. The real stdout is a raw stream, whose write may take only
    part of it (a signal; a non-blocking stdout that is full takes none and returns None, and is
    waited for until it can take more)."""
    view = memoryview(data)
    while view:
        n: int | None = stream.write(view)
        if n is None:
            select.select([], [stream], [])
        else:
            view = view[n:]
    stream.flush()


_claimed: PayloadWriter | None = None


def claim_stdout(output: Path | None = None) -> PayloadWriter:
    """Redirect fd 1 to stderr (idempotent) and return the payload writer: the real stdout, or
    the file ``output`` (``-o``), in which case nothing is ever written to stdout."""
    global _claimed
    if _claimed is None:
        sys.stdout.flush()
        saved_fd = os.dup(1)
        os.dup2(2, 1)
        sys.stdout = sys.stderr
        _claimed = PayloadWriter(os.fdopen(saved_fd, "wb", buffering=0))
    return _claimed if output is None else PayloadWriter(path=output)
