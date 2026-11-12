"""stderr logging of the package (``oh_my_slam`` logger), and the context a line is logged in
(``log_context``: the job or locate it belongs to, which the service log prints:
``ContextFilter``)."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_LOGGER_NAME = "oh_my_slam"


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """Return a logger writing to stderr (configured once)."""
    root = logging.getLogger(_LOGGER_NAME)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[oh-my-slam] %(message)s"))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
    if name == _LOGGER_NAME:
        return root
    return root.getChild(name.removeprefix(_LOGGER_NAME + "."))




_CONTEXT: ContextVar[str | None] = ContextVar("oh_my_slam_log_context", default=None)


@contextmanager
def log_context(tag: str) -> Iterator[None]:
    """Lines logged inside (on this thread or context) belong to ``tag`` (``job <id>``,
    ``locate <map>``): the service log prints it (``ContextFilter``)."""
    token = _CONTEXT.set(tag)
    try:
        yield
    finally:
        _CONTEXT.reset(token)


class ContextFilter(logging.Filter):
    """Sets a record's ``context``: `` [<tag>]`` inside ``log_context``, else empty."""

    def filter(self, record: logging.LogRecord) -> bool:
        tag = _CONTEXT.get()
        record.context = f" [{tag}]" if tag else ""
        return True
