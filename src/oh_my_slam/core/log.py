"""stderr logging of the package (``oh_my_slam`` logger)."""

from __future__ import annotations

import logging
import sys

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


