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
    media: str | None = None  # the media type of a body that is not JSON (``responses``: None)


MAP = Param("name", "path", "string", "Map name: 1-64 of letters, digits, '.', '_', '-'.",
            required=True)
VERSION = Param("version", "query", "integer", "A kept version (default: the latest).",
                minimum=1)

ENDPOINTS: tuple[Endpoint, ...] = (
    Endpoint("GET", "/api/health", "health", "Service, inference server and lane states",
             {200: "Health"}),
    Endpoint("GET", "/api/openapi.json", "openapi", "This API (OpenAPI 3.1)", {200: None}),
    Endpoint("POST", "/api/maps/{name}/updates", "post_update",
             "Start an update job (creates the map); with dry_run=1, check the frames and write "
             "nothing", {202: "Job", 200: "DryRun"},
             ("invalid_request", "camera_conflict", "inference_unavailable", "payload_too_large"),
             request="UpdateRequest", multipart=True,
             params=(MAP, Param("dry_run", "query", "integer", "1: answer a DryRun "
                                "synchronously, write nothing (O10).", minimum=0, maximum=1))),
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
    Endpoint("GET", "/api/maps/{name}/objects", "get_objects", "Object records and support "
             "surfaces", {200: "Objects"}, ("map_not_found", "version_gone", "invalid_request"),
             params=(MAP, VERSION)),
    Endpoint("GET", "/api/maps/{name}/grid", "get_grid", "Navigation grid", {200: "Grid"},
             ("map_not_found", "version_gone", "invalid_request"),
             params=(MAP, VERSION,
                     Param("cell_m", "query", "number", "Cell size in metres (default 0.05).",
                           minimum=0.01, maximum=1.0),
                     Param("z_min", "query", "number", "Bottom of the height band, metres above "
                           "the floor estimate floor_z_m (default 0.05)."),
                     Param("z_max", "query", "number", "Top of the height band, metres above "
                           "the floor estimate (default 1.8): the robot's height plus a margin "
                           "marks everything it could hit."),
                     Param("footprint_radius_m", "query", "number", "The robot's radius: its "
                           "footprint is free where cameras stood, and passages narrower than "
                           "twice it are kind narrow.", minimum=0))),
    Endpoint("GET", "/api/maps/{name}/points", "get_points", "Fused points inside a box",
             {200: "Points"}, ("map_not_found", "version_gone", "invalid_request"),
             params=(MAP, VERSION,
                     Param("min", "query", "string", "Box corner x,y,z (metres, map frame).",
                           required=True),
                     Param("max", "query", "string", "Opposite box corner x,y,z.",
                           required=True),
                     Param("voxel_m", "query", "number", "Thinning voxel in metres (default "
                           "0.01).", minimum=0.001, maximum=1.0),
                     Param("max_points", "query", "integer", "Cap on the points returned "
                           "(default 200000); truncated says whether it applied.", minimum=1,
                           maximum=2000000))),
    Endpoint("GET", "/api/maps/{name}/birdseye.png", "get_birdseye_png",
             "Bird's-eye image: the cloud, the objects' footprints with their ids and the "
             "keyframe cameras seen from above; its text metadata records map_version and "
             "pixel_to_map", {200: None}, ("map_not_found", "version_gone", "invalid_request"),
             params=(MAP, VERSION), media="image/png"),
    Endpoint("GET", "/api/maps/{name}/birdseye", "get_birdseye",
             "The bird's-eye image's metadata: size and pixel_to_map", {200: "Birdseye"},
             ("map_not_found", "version_gone", "invalid_request"), params=(MAP, VERSION)),
    Endpoint("POST", "/api/maps/{name}/locate", "post_locate", "Locate frames in the map",
             {200: "Locate"}, ("invalid_request", "camera_conflict", "map_not_found",
                               "version_gone", "payload_too_large"),
             request="LocateRequest", multipart=True, params=(MAP,)),
    Endpoint("GET", "/api/maps/{name}/pose", "get_pose", "Last recorded pose", {200: "Pose"},
             ("map_not_found", "no_pose", "invalid_request"), params=(MAP,)),
    Endpoint("DELETE", "/api/maps/{name}", "delete_map", "Delete the map and its versions",
             {204: None}, ("map_not_found", "map_busy", "invalid_request"), params=(MAP,)),
)
