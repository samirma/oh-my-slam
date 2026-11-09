"""The endpoint definitions: ``app`` routes them, ``openapi`` documents them, both from here."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Param:
    name: str
    where: str  # "path" | "query"
    type: str  # JSON Schema type
    description: str
    required: bool = False
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class Endpoint:
    method: str
    path: str
    handler: str  # the ``app.Api`` method
    summary: str
    responses: dict[int, str | None]  # status -> $defs name (None: no body)
    errors: tuple[str, ...] = ()
    request: str | None = None  # $defs name of the body
    multipart: bool = False  # the body is the ``request`` part of a multipart form
    params: tuple[Param, ...] = field(default_factory=tuple)


MAP = Param("name", "path", "string", "Map name: 1-64 of letters, digits, '.', '_', '-'.",
            required=True)
VERSION = Param("version", "query", "integer", "A kept version (default: the latest).",
                minimum=1)

ENDPOINTS: tuple[Endpoint, ...] = (
    Endpoint("GET", "/api/health", "health", "Service, inference server and lane states",
             {200: "Health"}),
    Endpoint("GET", "/api/openapi.json", "openapi", "This API (OpenAPI 3.1)", {200: None}),
    Endpoint("POST", "/api/maps/{name}/updates", "post_update",
             "Start an update job (creates the map)", {202: "Job"},
             ("invalid_request", "camera_conflict", "inference_unavailable", "payload_too_large"),
             request="UpdateRequest", multipart=True, params=(MAP,)),
    Endpoint("GET", "/api/jobs/{id}", "get_job", "Job state and result", {200: "Job"},
             ("job_not_found", "invalid_request"),
             params=(Param("id", "path", "string", "Job id.", required=True),
                     Param("wait_s", "query", "number", "Hold the answer until the state "
                           "changes or this many seconds pass (at most 30).", minimum=0,
                           maximum=30))),
    Endpoint("GET", "/api/maps/{name}", "get_map", "Map summary", {200: "MapSummary"},
             ("map_not_found", "version_gone", "invalid_request"), params=(MAP, VERSION)),
    Endpoint("GET", "/api/maps/{name}/frames", "get_frames", "Keyframes and their poses",
             {200: "Frames"}, ("map_not_found", "version_gone", "invalid_request"),
             params=(MAP, VERSION)),
    Endpoint("DELETE", "/api/maps/{name}", "delete_map", "Delete the map and its versions",
             {204: None}, ("map_not_found", "map_busy", "invalid_request"), params=(MAP,)),
)
