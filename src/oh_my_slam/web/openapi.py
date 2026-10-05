"""The OpenAPI 3.1 document at ``/api/openapi.json`` (spec §2.6 "API"): one ``POST
/api/ops/<operation>`` (submit a job) and one ``POST /api/ops/<operation>/validate`` (check
without queuing) per command mode, their parameters generated from ``spec.describe()`` — names,
kinds, defaults, choices, bounds and help — plus the service's fixed endpoints. Each operation
also carries its whole ``describe()`` entry under ``x-oms`` (outputs, stages, rules, errors,
inference need), so a client can render forms and result pages from this document alone."""

from __future__ import annotations

from typing import Any

from oh_my_slam.commands import spec
from oh_my_slam.version import __version__
from oh_my_slam.web.operations import OUT_DIR, Operation

Json = dict[str, Any]


def _schema(p: Json) -> Json:
    """JSON schema of one parameter from its ``describe()`` entry."""
    kind = p["kind"]
    s: Json
    if kind == str(spec.Kind.FLAG):
        s = {"type": "boolean"}
    elif kind == str(spec.Kind.NUMBER):
        s = {"type": "number"}
        if p["minimum"] is not None:
            s["minimum"] = p["minimum"]
        if p["exclusive_minimum"] is not None:
            s["exclusiveMinimum"] = p["exclusive_minimum"]
    elif kind == str(spec.Kind.ENUM):
        s = {"type": "string", "enum": p["choices"]}
    elif kind == str(spec.Kind.ATTRS):
        s = {"type": "string",
             "description": "key=value[,key=value…]: " + ", ".join(
                 a["key"] for a in p.get("attributes", []))}
    elif kind == str(spec.Kind.MAP):
        s = {"type": "string", "format": "map-name",
             "description": "a map of the workspace: <name> or maps/<name>"}
    elif kind in (str(spec.Kind.FILE_OUT), str(spec.Kind.FOLDER_OUT)):
        s = {"type": "string", "format": "file-name",
             "description": f"a plain name in the job's {OUT_DIR}/ folder"}
    else:  # a path input
        s = {"type": "string", "format": "workspace-path",
             "description": "a path inside the workspace (an upload is uploads/<id>/<file>)"}
    if p["multiple"] or p["repeatable"]:
        item = s
        s = {"type": "array", "items": item, "minItems": 1}
        if p["repeatable"]:
            s = {"oneOf": [item, s]}
    if p["default"] is not None and kind != str(spec.Kind.ATTRS):
        s["default"] = p["default"]
    help_text = p["help"] + (f" ({p['applies_text']})" if p["applies_text"] else "")
    s["description"] = f"{p['flag']}: {help_text}" + (
        f" — {s['description']}" if "description" in s else "")
    s["x-oms"] = p
    return s


def _body(d: Json) -> Json:
    """The request body: one property per parameter that means something to the service."""
    params = [p for p in d["parameters"] if p.get("service", True)]
    return {"required": True, "content": {"application/json": {"schema": {
        "type": "object", "additionalProperties": False,
        "properties": {p["name"]: _schema(p) for p in params},
        "required": [p["name"] for p in params if p["required"]],
    }}}}


_VIEWER_PARAM = [{"name": "viewer", "in": "query", "required": False,
                  "schema": {"type": "boolean", "default": False},
                  "description": "also save the viewer of the request's image (one more step of "
                                 "the same job, which runs inference for it once more)"}]
_ERROR = {"$ref": "#/components/responses/Error"}
_JOB = {"description": "the job", "content": {"application/json": {
    "schema": {"$ref": "#/components/schemas/Job"}}}}


def _ok(description: str, media: str = "application/json") -> Json:
    return {"200": {"description": description, "content": {media: {}}}}


def _fixed() -> Json:
    """The service's own endpoints (not per command)."""
    job_id = [{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}]
    name = [{"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}]
    path = [{"name": "path", "in": "path", "required": True, "schema": {"type": "string"}}]
    return {
        "/api/health": {"get": {"summary": "service and inference-server health",
                                "responses": _ok("health")}},
        "/api/operations": {"get": {"summary": "the commands' definitions (spec.describe())",
                                    "responses": _ok("every operation as data")}},
        "/api/uploads": {
            "get": {"summary": "uploads not yet consumed by a finished job",
                    "responses": _ok("uploads")},
            "post": {"summary": "upload one input file (raw request body)",
                     "parameters": [{"name": "name", "in": "query", "required": True,
                                     "schema": {"type": "string"},
                                     "description": "the file name (its suffix tells its type)"}],
                     "requestBody": {"content": {"application/octet-stream": {}}},
                     "responses": {"201": {"description": "the upload: use its path as an input"},
                                   "413": _ERROR, "4XX": _ERROR}}},
        "/api/uploads/{id}": {"delete": {"summary": "discard an unconsumed upload",
                                         "parameters": job_id,
                                         "responses": {"204": {"description": "deleted"},
                                                       "4XX": _ERROR}}},
        "/api/maps": {"get": {"summary": "the workspace's maps with their summaries",
                              "responses": _ok("maps")}},
        "/api/maps/{name}": {"get": {"summary": "a map's summary and metadata (map.json)",
                                     "parameters": name, "responses": _ok("the map")}},
        "/api/maps/{name}/files/{path}": {"get": {"summary": "a file of a map (read-only)",
                                                  "parameters": [*name, *path],
                                                  "responses": _ok("the file", "*/*")}},
        "/api/maps/{name}/viewer/{path}": {"get": {
            "summary": "the viewer (spec 2.5) of a map: its page, scripts and data, served by the "
                       "viewer's own routes; also at /viewer/map/{name}/",
            "parameters": [*name, *path], "responses": _ok("viewer page or data", "*/*")}},
        "/api/jobs/{id}/viewer/{path}": {"get": {
            "summary": "the viewer a job saved (view.sh jobs, or ?viewer=true); also at "
                       "/viewer/job/{id}/",
            "parameters": [*job_id, *path], "responses": _ok("viewer page or data", "*/*")}},
        "/api/jobs": {"get": {"summary": "every job, oldest first", "responses": _ok("jobs")}},
        "/api/jobs/events": {"get": {"summary": "server-sent events: every job change",
                                     "responses": _ok("events", "text/event-stream")}},
        "/api/jobs/{id}": {"get": {"summary": "a job", "parameters": job_id,
                                   "responses": {"200": _JOB, "4XX": _ERROR}}},
        "/api/jobs/{id}/events": {"get": {"summary": "server-sent events of one job until it ends",
                                          "parameters": job_id,
                                          "responses": _ok("events", "text/event-stream")}},
        "/api/jobs/{id}/cancel": {"post": {"summary": "cancel a queued or running job (the effect "
                                                      "of interrupting the command)",
                                           "parameters": job_id,
                                           "responses": {"200": _JOB, "4XX": _ERROR}}},
        "/api/jobs/{id}/resubmit": {"post": {
            "summary": "submit a new job with the same parameters (body: parameters to replace)",
            "parameters": job_id, "responses": {"202": _JOB, "4XX": _ERROR, "503": _ERROR}}},
        "/api/jobs/{id}/result": {"get": {"summary": "the result: the command's stdout / -o file",
                                          "parameters": job_id,
                                          "responses": _ok("the result", "*/*")}},
        "/api/jobs/{id}/files": {"get": {"summary": "every file the job wrote",
                                         "parameters": job_id, "responses": _ok("files")}},
        "/api/jobs/{id}/files/{path}": {"get": {"summary": "one file the job wrote",
                                                "parameters": [*job_id, *path],
                                                "responses": _ok("the file", "*/*")}},
        "/api/jobs/{id}/display-cloud": {"get": {
            "summary": "a PLY file of the job (?file=<path>, else its result) as the viewer draws "
                       "it: the viewer's binary cloud document, within the display budget of spec "
                       "2.5 (a voxel-grid selection above it), with the file's header comments",
            "parameters": [*job_id, {"name": "file", "in": "query", "required": False,
                                     "schema": {"type": "string"}}],
            "responses": {"200": {"description": "the cloud document", "content": {
                "application/octet-stream": {}}}, "4XX": _ERROR}}},
        "/api/display-transform": {"get": {
            "summary": "the viewer's display transform of a scene: identity for map coordinates, "
                       "view.sh -i's upright transform for a single image's camera frame",
            "parameters": [
                {"name": "camera", "in": "query", "required": False,
                 "schema": {"type": "boolean"}, "description": "the scene is in a camera frame"},
                {"name": "cs_types", "in": "query", "required": False,
                 "schema": {"type": "string"},
                 "description": "a scene JSON's coordinate-system types, comma-separated: a camera "
                                "frame when none is a scene_cs"},
                {"name": "comment", "in": "query", "required": False,
                 "schema": {"type": "array", "items": {"type": "string"}},
                 "description": "a PLY's header comments (they name its frame)"},
                {"name": "up", "in": "query", "required": False, "schema": {"type": "string"},
                 "description": "x,y,z: the estimated up direction in the camera frame"}],
            "responses": {**_ok("{camera_frame, display_transform (4 x 4, rows)}"),
                          "4XX": _ERROR}}},
        "/api/jobs/{id}/log": {"get": {"summary": "the lines the command printed to stderr",
                                       "parameters": job_id,
                                       "responses": _ok("the log", "text/plain")}},
        "/api/jobs/{id}/timings": {"get": {"summary": "the command's timings record",
                                           "parameters": job_id, "responses": _ok("timings")}},
    }


def document(ops: dict[str, Operation]) -> Json:
    described = spec.describe()
    by_label = {op.label: op for op in ops.values()}
    paths: Json = {}
    for d in described["operations"]:
        op = by_label[d["id"]]
        summary = f"{d['id']} — {d['description']}"
        paths[f"/api/ops/{op.id}"] = {"post": {
            "operationId": op.id, "summary": summary, "tags": [d["prog"]],
            "description": f"Runs `{d['id']}` as a job. Inference: {d['inference']} "
                           f"({d['inference_text']}).",
            "parameters": _VIEWER_PARAM if op.viewer_input is not None and not op.browser else [],
            "requestBody": _body(d),
            "responses": {"202": _JOB, "4XX": _ERROR, "503": _ERROR},
            "x-oms": d,
        }}
        paths[f"/api/ops/{op.id}/validate"] = {"post": {
            "operationId": f"{op.id}-validate", "summary": f"check a {d['id']} request",
            "tags": [d["prog"]], "requestBody": _body(d),
            "responses": {"200": {"description": "the problems (empty: valid)"}},
        }}
    paths.update(_fixed())
    return {
        "openapi": "3.1.0",
        "info": {"title": "oh-my-slam server.sh", "version": __version__,
                 "description": "Every mode of reconstruct.sh, mapper.sh, segment.sh and view.sh "
                                "as a job (spec §2.6)."},
        "paths": paths,
        "components": {
            "schemas": {"Job": {"type": "object", "properties": {
                "id": {"type": "string"}, "operation": {"type": "string"},
                "state": {"enum": ["queued", "running", "succeeded", "failed", "cancelled"]},
                "stage": {"type": ["string", "null"], "description": "the command's own stage"},
                "progress": {"type": ["object", "null"]},
                "stages": {"type": "array", "description": "the command's own stages with "
                           "their seconds (never those of a viewer step)"},
                "viewer_progress": {"type": ["object", "null"], "description": (
                    "{stage, done, total} of the viewer step that follows the command of a "
                    "?viewer=true job, while it runs; null otherwise")},
                "viewer": {"type": ["string", "null"],
                           "description": "the page of the job's saved viewer"},
                "viewer_error": {"type": ["object", "null"], "description": (
                    "why a ?viewer=true job has no viewer although its result stands: "
                    "{code, exit_code, http_status, message}; code 'cancelled' when the job "
                    "was cancelled during the viewer step, else the code of its exit status "
                    "(e.g. 'server_unavailable')")},
            }}},
            "responses": {"Error": {"description": "the command's message and the code of its "
                                                   "exit status", "content": {
                "application/json": {}}}},
        },
        "x-oms": {"exit_codes": described["exit_codes"], "stages": described["stages"]},
    }
