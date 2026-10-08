"""Exit codes and the exception hierarchy that maps onto them.

Every command maps an uncaught :class:`OhMySlamError` to its ``exit_code`` and prints one
human-readable line to stderr. Anything else is an internal error (exit 1).
"""

from __future__ import annotations

from enum import IntEnum

from oh_my_slam.core.constants import START_INFERENCE_SERVER


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


_DOWN = "not running, or its models failed to load"

# What each exit status means, for the people and agents that read it (README "Output contract and
# exit codes", the agent skill).
MEANING: dict[ExitCode, str] = {
    ExitCode.OK: "success",
    ExitCode.INTERNAL: "internal error (also: inference failed, or COLMAP is missing or the "
                       "wrong version)",
    ExitCode.USAGE: "usage or input error: a bad option or value, a missing or unsupported input "
                    "file",
    ExitCode.SERVER_UNAVAILABLE: "a server it needs does not answer: the inference server "
                                 f"({_DOWN}), or for --status the server it queries",
    ExitCode.NOT_A_MAP: "the map folder is not a map (and not empty, for an update)",
    ExitCode.NOT_REGISTERED: "nothing could be placed in the map (no overlap); the map is "
                             "unchanged",
    ExitCode.MAP_LOCKED: "another update holds the map",
    ExitCode.INTERRUPTED: "interrupted (Ctrl-C, or a server.sh request whose client left)",
}

# What each exit status means to a client of the web service's API (the agent skill): the meanings
# above without what only a local run sees (``--status``, Ctrl-C) or no client reads (the answer
# to a client that left): a client reads 130 only when the service's stop interrupted its request.
API_MEANING: dict[ExitCode, str] = {
    **MEANING,
    ExitCode.SERVER_UNAVAILABLE: f"the inference server does not answer ({_DOWN})",
    ExitCode.INTERRUPTED: "interrupted: the service stopped while the request waited or ran; send "
                          "it again once the service runs",
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


def internal_message(exc: BaseException) -> str:
    """How an internal error (an unexpected exception, exit 1) is told: its type and message, what
    follows ``<prog>: internal error:`` on a command's stderr."""
    return f"{type(exc).__name__}: {exc}"


SERVER_HINT = f"start it with {START_INFERENCE_SERVER}"


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
            f"then restart with {START_INFERENCE_SERVER} --stop && {START_INFERENCE_SERVER}",
        )


class ServerLoadingError(ServerUnavailableError):
    """The server runs but is still loading its models after the command's bounded wait (health
    status ``loading``)."""

    def __init__(self, waited_s: float, log_path: str) -> None:
        OhMySlamError.__init__(
            self,
            f"inference server is still loading its models after {waited_s:.0f} s — run "
            f"{START_INFERENCE_SERVER}, which waits until they are loaded (progress in "
            f"{log_path}), then retry",
        )


class ServerStoppingError(ServerUnavailableError):
    """The server is shutting down (health status ``stopping``)."""

    def __init__(self) -> None:
        OhMySlamError.__init__(
            self,
            f"inference server is stopping — start it again with {START_INFERENCE_SERVER} "
            "(it waits until the old one has exited), then retry",
        )


class ServerProtocolError(ServerUnavailableError):
    """The running server speaks another protocol version than this code (health ``protocol``):
    it was started before an upgrade."""

    def __init__(self, server: int, client: int) -> None:
        OhMySlamError.__init__(
            self,
            f"the running inference server speaks protocol {server}, this version needs "
            f"{client} (it was started before an upgrade) — restart it with "
            f"{START_INFERENCE_SERVER} --stop && {START_INFERENCE_SERVER}",
        )


class ServiceNotRunningError(OhMySlamError):
    """``server.sh --status`` without a running service: exit 3, like ``start_inference_server.sh
    --status``."""

    exit_code = ExitCode.SERVER_UNAVAILABLE


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
