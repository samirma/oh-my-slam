"""Exit codes and the exception hierarchy that maps onto them.

The mapper process of an update job (``service.mapper_proc``) exits with an uncaught
:class:`OhMySlamError`'s ``exit_code``; the service maps errors to API error codes.
"""

from __future__ import annotations

from enum import IntEnum

from oh_my_slam.core.constants import RESTART_INFERENCE_SERVER, START_INFERENCE_SERVER


class ExitCode(IntEnum):
    OK = 0
    INTERNAL = 1
    USAGE = 2
    SERVER_UNAVAILABLE = 3
    NOT_A_MAP = 4
    NOT_REGISTERED = 5
    MAP_LOCKED = 6
    INTERRUPTED = 130  # Ctrl-C / SIGINT


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
            f"then restart with {RESTART_INFERENCE_SERVER}",
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
            f"{RESTART_INFERENCE_SERVER}",
        )


class ServerBusyError(OhMySlamError):
    """The server queue is full (HTTP 503) and retries were exhausted."""

    exit_code = ExitCode.INTERNAL


class InferenceError(OhMySlamError):
    """The server returned an error for a request."""

    exit_code = ExitCode.INTERNAL


class DeviceMemoryError(InferenceError):
    """The server's device ran out of memory for a request; a smaller one may fit."""


class NotAMapError(OhMySlamError):
    exit_code = ExitCode.NOT_A_MAP


class RegistrationError(OhMySlamError):
    """No input frame could be placed in the map (e.g. no overlap)."""

    exit_code = ExitCode.NOT_REGISTERED


class MapLockedError(OhMySlamError):
    exit_code = ExitCode.MAP_LOCKED
