"""The OpenAPI 3.1 document at ``/api/openapi.json`` (spec §2.6 "API"): one ``POST
/api/ops/<operation>`` (run it within the request) and one ``POST /api/ops/<operation>/validate``
(check it, nothing runs) per mode of the commands the service offers, their parameters generated
from ``spec.describe()`` — names, kinds, defaults, choices, bounds and help — plus the service's
fixed endpoints. Each operation also carries its ``describe()`` entry as the API offers it
(``Operation.entry``: outputs, stages, rules, errors, inference need) under ``x-oms``, so a client
can render forms and result pages from this document alone.

Every JSON answer has its schema (``components/schemas``): the upload record, validation's answer,
the error shape, the health (the inference server's from its own model) and the maps. The
service's own refusals and its upload limits are exported under ``x-oms`` from ``web.app``, beside
the commands' exit codes."""

from __future__ import annotations

from typing import Any, get_args

from oh_my_slam.client.protocol import Health, ServerStatus
from oh_my_slam.commands import spec
from oh_my_slam.version import __version__
from oh_my_slam.web.operations import RESULT, Operation

Json = dict[str, Any]
_REFS = "#/components/schemas/{model}"


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
    else:  # a path input
        s = {"type": "string", "format": "workspace-path",
             "description": "a path inside the workspace (an upload is uploads/<id>/<file>)"}
    if p["multiple"]:
        s = {"type": "array", "items": s, "minItems": 1}
    if p["default"] is not None and kind != str(spec.Kind.ATTRS):
        s["default"] = p["default"]
    help_text = p["help"] + (f" ({p['applies_text']})" if p["applies_text"] else "")
    s["description"] = f"{p['flag']}: {help_text}" + (
        f" — {s['description']}" if "description" in s else "")
    s["x-oms"] = p
    return s


def _body(d: Json) -> Json:
    """The request body: one property per API parameter."""
    params = d["parameters"]
    return {"required": True, "content": {"application/json": {"schema": {
        "type": "object", "additionalProperties": False,
        "properties": {p["name"]: _schema(p) for p in params},
        "required": [p["name"] for p in params if p["required"]],
    }}}}


_ERROR = {"$ref": "#/components/responses/Error"}
_SERVER_TIMING = {"Server-Timing": {
    "description": "the command's per-stage timings: each stage under the command's own name, "
                   "then total, in milliseconds (name;dur=<ms>, comma-separated)",
    "schema": {"type": "string"}}}
_STR: Json = {"type": "string"}
_INT: Json = {"type": "integer"}
_NUM: Json = {"type": "number"}
_STRS: Json = {"type": "array", "items": _STR}


def _ref(name: str) -> Json:
    return {"$ref": _REFS.format(model=name)}


def _object(required: Json, optional: Json | None = None, more: bool = False) -> Json:
    """A JSON object with these properties (``more``: and others)."""
    return {"type": "object", "properties": {**required, **(optional or {})},
            "required": list(required), "additionalProperties": more}


def _schemas() -> Json:
    """The schema of every JSON answer but an operation's result (the command's own format)."""
    inference = Health.model_json_schema(ref_template=_REFS)
    by_parameter = {"type": "object", "additionalProperties": _STRS,
                    "description": "the messages per parameter (under \"\" those that concern "
                                   "none)"}
    problems = {"type": "array", "items": _ref("Problem")}
    either = {"type": ["string", "null"]}
    return {
        **inference.pop("$defs", {}),
        "InferenceServerHealth": inference,
        "Problem": _object({"rule": _STR, "parameters": _STRS, "message": _STR, "code": _STR,
                            "exit_code": _INT, "http_status": _INT}),
        "Error": _object({"error": _object(
            {"code": {**_STR, "description": "the code of the command's exit status "
                                             "(x-oms.exit_codes) or one of the service's own "
                                             "refusals (x-oms.refusals)"},
             "message": {**_STR, "description": "the command's own message"},
             "http_status": _INT},
            {"exit_code": _INT, "rule": _STR, "parameters": _STRS, "problems": problems,
             "by_parameter": by_parameter})}),
        "Upload": _object({"id": _STR, "name": _STR, "size": _INT,
                           "path": {**_STR, "description": "uploads/<id>/<name>: the value to "
                                                           "give one request's parameter"}}),
        "Validation": _object({"valid": {"type": "boolean"}, "command": _STRS,
                               "inference": {"type": ["boolean", "null"]}, "problems": problems,
                               "by_parameter": by_parameter}),
        "Request": _object({"operation": _STR, "command": _STRS, "state": _STR,
                            "arrived_at": _NUM, "started_at": {"type": ["number", "null"]}}),
        "Health": _object({
            "status": _STR,
            "service": _object({"version": _STR, "url": _STR, "pid": _INT, "workspace": _STR,
                                "data": _STR, "started_at": _NUM,
                                "requests": _object({"running": _INT, "waiting": _INT}),
                                "in_progress": {"type": "array", "items": _ref("Request")}}),
            "inference": _object(
                {"status": {"enum": ["down", *get_args(ServerStatus)]}, "start_command": either},
                {"message": _STR, "health": _ref("InferenceServerHealth")})}),
        "MapSummary": _object(
            {"name": _STR, "path": _STR, "frames": _INT},
            {"objects": {"type": ["integer", "null"]},
             "last_update": _object({}, {"frames_added": _INT,
                                         "total_s": {"type": ["number", "null"]}}, more=True)},
            more=True),
        "Map": {"allOf": [_ref("MapSummary"), _object({"meta": {"type": "object"}}, more=True)]},
    }


def _ok(description: str, schema: Json, status: str = "200") -> Json:
    return {status: {"description": description,
                     "content": {"application/json": {"schema": schema}}}}


def _result(d: Json) -> Json:
    """The response of a successful run: the command's stdout in the result's media type."""
    media = dict.fromkeys(o["media_type"] for o in d["outputs"] if o["via"] == RESULT)
    return {"description": "the command's result, byte-identical to what it writes to stdout or "
                           "to -o: " + "; ".join(o["text"] for o in d["outputs"]
                                                 if o["via"] == RESULT),
            "headers": _SERVER_TIMING,
            "content": {m: {} for m in media} or {"application/octet-stream": {}}}


def _upload(summary: str) -> Json:
    return {"summary": summary,
            "parameters": [{"name": "name", "in": "query", "required": True,
                            "schema": {"type": "string"},
                            "description": "the file name the command sees (its suffix tells its "
                                           "type)"}],
            "requestBody": {"content": {"application/octet-stream": {}}},
            "responses": {**_ok("the upload: give its path to one request, which consumes it",
                                _ref("Upload"), "201"), "4XX": _ERROR}}


def _fixed() -> Json:
    """The service's own endpoints (not per command)."""
    upload_id = [{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}]
    name = [{"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}]
    return {
        "/api/health": {"get": {"summary": "service and inference-server health, and the requests "
                                           "running or waiting for their turn",
                                "responses": _ok("health", _ref("Health"))}},
        "/api/uploads": {
            "put": _upload("upload one input file: the raw file as the body (curl -T <file>)"),
            "post": _upload("upload one input file: the raw file as the body, with its own media "
                            "type")},
        "/api/uploads/{id}": {"delete": {"summary": "discard an upload no request was given",
                                         "parameters": upload_id,
                                         "responses": {"204": {"description": "deleted"},
                                                       "4XX": _ERROR}}},
        "/api/maps": {"get": {"summary": "the workspace's maps with their summaries",
                              "responses": _ok("maps", {"type": "array",
                                                        "items": _ref("MapSummary")})}},
        "/api/maps/{name}": {"get": {"summary": "a map's summary and metadata (map.json)",
                                     "parameters": name,
                                     "responses": {**_ok("the map", _ref("Map")),
                                                   "4XX": _ERROR}}},
    }


def document(ops: dict[str, Operation]) -> Json:
    from oh_my_slam.web.app import MAX_UPLOAD_BYTES, MIN_FREE_BYTES, refusals  # app imports this

    described = spec.describe()
    by_label = {op.label: op for op in ops.values()}
    paths: Json = {}
    for raw in described["operations"]:
        op = by_label.get(raw["id"])
        if op is None:  # a command the service does not offer (view.sh)
            continue
        d = op.entry(raw)
        summary = f"{d['id']} — {d['description']}"
        paths[f"/api/ops/{op.id}"] = {"post": {
            "operationId": op.id, "summary": summary, "tags": [d["prog"]],
            "description": f"Runs `{d['id']}` within this request and answers when it ends. "
                           f"Inference: {d['inference']} ({d['inference_text']}); requests that "
                           "use the inference server run one at a time, in arrival order, each "
                           "waiting for its turn with its connection open. Disconnecting "
                           "interrupts the command as Ctrl-C would.",
            "requestBody": _body(d),
            "responses": {"200": _result(d), "4XX": _ERROR, "500": _ERROR, "503": _ERROR},
            "x-oms": d,
        }}
        paths[f"/api/ops/{op.id}/validate"] = {"post": {
            "operationId": f"{op.id}-validate", "summary": f"check a {d['id']} request",
            "description": "The request's checks (the command's own, and the inference server's "
                           "when the request needs it); nothing runs.",
            "tags": [d["prog"]], "requestBody": _body(d),
            "responses": {**_ok("the problems (empty: valid)", _ref("Validation")),
                          "4XX": _ERROR},
        }}
    paths.update(_fixed())
    for item in paths.values():  # anything else that fails is the internal error, in that shape
        for route in item.values():
            route["responses"].setdefault("500", _ERROR)
    return {
        "openapi": "3.1.0",
        "info": {"title": "oh-my-slam server.sh", "version": __version__,
                 "description": "Every mode of " + ", ".join(
                     dict.fromkeys(op.program.prog for op in ops.values()))
                 + ", run within its own request (spec §2.6)."},
        "paths": paths,
        "components": {
            "responses": {"Error": {"description": "the command's message and the code of its "
                                                   "exit status, or one of the service's own "
                                                   "refusals", "content": {
                "application/json": {"schema": _ref("Error")}}}},
            "schemas": _schemas(),
        },
        "x-oms": {"exit_codes": described["exit_codes"], "stages": described["stages"],
                  "refusals": [{"code": r.code, "http_status": r.http_status, "when": r.when}
                               for r in refusals().values()],
                  "limits": {"max_upload_bytes": MAX_UPLOAD_BYTES,
                             "min_free_bytes": MIN_FREE_BYTES}},
    }
