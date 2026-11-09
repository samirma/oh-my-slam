"""Typed API errors (api.md §8): a fixed code, its HTTP status, whether a retry can succeed, a
message that says what to do, and details."""

from __future__ import annotations

from typing import Any

# code -> (HTTP status, retryable)
ERRORS: dict[str, tuple[int, bool]] = {
    "invalid_request": (400, False),
    "camera_conflict": (409, False),
    "map_not_found": (404, False),
    "job_not_found": (404, False),
    "version_gone": (410, False),
    "no_pose": (404, False),
    "map_busy": (409, True),
    "inference_unavailable": (503, True),
    "out_of_memory": (503, True),
    "payload_too_large": (413, False),
    "internal": (500, False),
}


class ApiError(Exception):
    """An error answer: ``code`` from ``ERRORS``; ``status`` overrides its HTTP status (an
    unknown route is a 404 ``invalid_request``)."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None,
                 status: int | None = None) -> None:
        super().__init__(message)
        if code not in ERRORS:
            raise ValueError(f"unknown error code {code!r}")
        self.code = code
        self.message = message
        self.details = details or {}
        self.status = status or ERRORS[code][0]

    @property
    def retryable(self) -> bool:
        return ERRORS[self.code][1]

    def body(self) -> dict[str, Any]:
        """The ``Error`` object."""
        return {"code": self.code, "message": self.message, "retryable": self.retryable,
                "details": self.details}


def invalid(field: str, message: str, **details: Any) -> ApiError:
    """``invalid_request`` naming the failing ``field`` (``details.field``)."""
    return ApiError("invalid_request", message, {"field": field, **details})
