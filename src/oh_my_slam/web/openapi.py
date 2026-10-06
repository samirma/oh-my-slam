"""The OpenAPI 3.1 document at ``/api/openapi.json`` (spec §2.6 "API"): one ``POST
/api/ops/<operation>`` (run it within the request) and one ``POST /api/ops/<operation>/validate``
(check it, nothing runs) per mode of the commands the service offers, their parameters generated
from ``spec.describe()`` — names, kinds, defaults, choices, bounds and help — plus the service's
fixed endpoints. Each operation also carries its ``describe()`` entry as the API offers it
(``Operation.entry``: outputs, stages, rules, errors, inference need) under ``x-oms``, so a client
can render forms and result pages from this document alone."""

from __future__ import annotations

from typing import Any

from oh_my_slam.commands import spec
from oh_my_slam.version import __version__
from oh_my_slam.web.operations import RESULT, Operation

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


def _ok(description: str, media: str = "application/json") -> Json:
    return {"200": {"description": description, "content": {media: {}}}}


def _result(d: Json) -> Json:
    """The response of a successful run: the command's stdout in the result's media type."""
    media = dict.fromkeys(o["media_type"] for o in d["outputs"] if o["via"] == RESULT)
    return {"description": "the command's result, byte-identical to what it writes to stdout or "
                           "to -o: " + "; ".join(o["text"] for o in d["outputs"]
                                                 if o["via"] == RESULT),
            "headers": _SERVER_TIMING,
            "content": {m: {} for m in media} or {"application/octet-stream": {}}}


def _fixed() -> Json:
    """The service's own endpoints (not per command)."""
    upload_id = [{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}]
    name = [{"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}]
    return {
        "/api/health": {"get": {"summary": "service and inference-server health, and the requests "
                                           "running or waiting for their turn",
                                "responses": _ok("health")}},
        "/api/uploads": {
            "post": {"summary": "upload one input file (raw request body)",
                     "parameters": [{"name": "name", "in": "query", "required": True,
                                     "schema": {"type": "string"},
                                     "description": "the file name (its suffix tells its type)"}],
                     "requestBody": {"content": {"application/octet-stream": {}}},
                     "responses": {"201": {"description": "the upload: use its path as an input "
                                                          "of one request"},
                                   "413": _ERROR, "4XX": _ERROR}}},
        "/api/uploads/{id}": {"delete": {"summary": "discard an upload no request was given",
                                         "parameters": upload_id,
                                         "responses": {"204": {"description": "deleted"},
                                                       "4XX": _ERROR}}},
        "/api/maps": {"get": {"summary": "the workspace's maps with their summaries",
                              "responses": _ok("maps")}},
        "/api/maps/{name}": {"get": {"summary": "a map's summary and metadata (map.json)",
                                     "parameters": name,
                                     "responses": {**_ok("the map"), "4XX": _ERROR}}},
    }


def document(ops: dict[str, Operation]) -> Json:
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
            "responses": {"200": {"description": "the problems (empty: valid)"}},
        }}
    paths.update(_fixed())
    return {
        "openapi": "3.1.0",
        "info": {"title": "oh-my-slam server.sh", "version": __version__,
                 "description": "Every mode of " + ", ".join(
                     dict.fromkeys(op.program.prog for op in ops.values()))
                 + ", run within its own request (spec §2.6)."},
        "paths": paths,
        "components": {
            "responses": {"Error": {"description": "the command's message and the code of its "
                                                   "exit status", "content": {
                "application/json": {}}}},
        },
        "x-oms": {"exit_codes": described["exit_codes"], "stages": described["stages"]},
    }
