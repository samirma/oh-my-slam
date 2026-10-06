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
    INTERRUPTED = 130  # Ctrl-C / SIGINT


# The one generic exit status → HTTP status rule of the web service (spec §2.6 "Errors": input
# errors → 4xx, inference server unavailable → 503, internal → 500). The machine-readable error
# code is the exit code's lower-case name (``error_code``).
HTTP_STATUS: dict[ExitCode, int] = {
    ExitCode.OK: 200,
    ExitCode.INTERNAL: 500,
    ExitCode.USAGE: 400,  # bad option or input
    ExitCode.SERVER_UNAVAILABLE: 503,
    ExitCode.NOT_A_MAP: 422,  # the folder exists but is not a map
    ExitCode.NOT_REGISTERED: 422,  # valid request, nothing could be placed in the map
    ExitCode.MAP_LOCKED: 409,  # another update holds the map
    ExitCode.INTERRUPTED: 499,  # the request was withdrawn (Ctrl-C, a client that left)
}


def http_status(exit_code: int) -> int:
    """HTTP status of a command's exit status; any other status (a crash, a signal) is 500."""
    try:
        return HTTP_STATUS[ExitCode(exit_code)]
    except ValueError:
        return 500


def error_code(exit_code: int) -> str:
    """Machine-readable code of a command's exit status (``usage``, ``not_a_map``, …)."""
    try:
        return ExitCode(exit_code).name.lower()
    except ValueError:
        return ExitCode.INTERNAL.name.lower()


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


class ServerModelsFailedError(ServerUnavailableError):
    """The server runs but its models failed to load (health status ``error``)."""

    def __init__(self, failed: str, log_path: str) -> None:
        OhMySlamError.__init__(
            self,
            f"inference server models failed to load ({failed}) — see {log_path}, fix the cause, "
            "then restart with ./start_inference_server.sh --stop && ./start_inference_server.sh",
        )


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
