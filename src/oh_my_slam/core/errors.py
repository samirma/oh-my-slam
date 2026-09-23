"""Exit codes and the exception hierarchy that maps onto them.

Every command maps an uncaught :class:`OhMySlamError` to its ``exit_code`` and prints one
human-readable line to stderr. Anything else is an internal error (exit 1).
"""

from __future__ import annotations

from enum import IntEnum


class ExitCode(IntEnum):
    OK = 0
    INTERNAL = 1
    USAGE = 2
    SERVER_UNAVAILABLE = 3
    NOT_A_MAP = 4
    NOT_REGISTERED = 5
    MAP_LOCKED = 6


SERVER_HINT = "start it with ./start_inference_server.sh"


class OhMySlamError(Exception):
    """Base class; carries the process exit code."""

    exit_code: ExitCode = ExitCode.INTERNAL


class UsageError(OhMySlamError):
    exit_code = ExitCode.USAGE


class InputError(OhMySlamError):
    """An input file is missing, unreadable or of an unsupported type."""

    exit_code = ExitCode.USAGE


class ServerUnavailableError(OhMySlamError):
    exit_code = ExitCode.SERVER_UNAVAILABLE

    def __init__(self, detail: str = "") -> None:
        suffix = f" ({detail})" if detail else ""
        super().__init__(f"inference server is not running{suffix} — {SERVER_HINT}")


class ServerBusyError(OhMySlamError):
    """The server queue is full (HTTP 503) and retries were exhausted."""

    exit_code = ExitCode.INTERNAL


class InferenceError(OhMySlamError):
    """The server returned an error for a request."""

    exit_code = ExitCode.INTERNAL


class NotAMapError(OhMySlamError):
    exit_code = ExitCode.NOT_A_MAP


class RegistrationError(OhMySlamError):
    """No input frame could be placed in the map (e.g. no overlap)."""

    exit_code = ExitCode.NOT_REGISTERED


class MapLockedError(OhMySlamError):
    exit_code = ExitCode.MAP_LOCKED
