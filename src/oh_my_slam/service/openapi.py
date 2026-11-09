"""``GET /api/openapi.json``: OpenAPI 3.1 generated from ``routes.ENDPOINTS``, its schemas the
``$defs`` of ``api.schema.json`` (references rewritten to ``#/components/schemas``)."""

from __future__ import annotations

import json
from typing import Any

from oh_my_slam.service.errors import ERRORS
from oh_my_slam.service.routes import ENDPOINTS, Param
from oh_my_slam.service.schema import SCHEMA_VERSION, defs, document

UPLOAD_NOTE = ("multipart/form-data: the part named `request` holds the JSON body; every other "
               "part is a file, referenced from the JSON by its part name.")


def _ref(name: str) -> dict[str, str]:
    return {"$ref": f"#/components/schemas/{name}"}


def _components() -> dict[str, Any]:
    text = json.dumps(defs()).replace('"#/$defs/', '"#/components/schemas/')
    out: dict[str, Any] = json.loads(text)
    return out


def _param(p: Param) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": p.type}
    if p.minimum is not None:
        schema["minimum"] = p.minimum
    if p.maximum is not None:
        schema["maximum"] = p.maximum
    return {"name": p.name, "in": p.where, "required": p.required, "description": p.description,
            "schema": schema}


def build(version: str) -> dict[str, Any]:
    paths: dict[str, Any] = {}
    for e in ENDPOINTS:
        op: dict[str, Any] = {"operationId": e.handler, "summary": e.summary,
                              "parameters": [_param(p) for p in e.params], "responses": {}}
        if e.request:  # every request body so far carries images: multipart
            op["description"] = UPLOAD_NOTE
            op["requestBody"] = {"required": True, "content": {"multipart/form-data": {
                "schema": {"type": "object", "required": ["request"],
                           "properties": {"request": _ref(e.request)},
                           "additionalProperties": {"type": "string", "format": "binary"}},
                "encoding": {"request": {"contentType": "application/json"}}}}}
        for status, body in e.responses.items():
            resp: dict[str, Any] = {"description": e.summary}
            if body is not None:
                resp["content"] = {"application/json": {"schema": _ref(body)}}
            elif e.handler == "openapi":
                resp["content"] = {"application/json": {"schema": {"type": "object"}}}
            if status == 202:
                resp["headers"] = {"Location": {"schema": {"type": "string"},
                                                "description": "/api/jobs/{job_id}"}}
            op["responses"][str(status)] = resp
        for status in sorted({ERRORS[c][0] for c in (*e.errors, "internal")}):
            codes = [c for c in (*e.errors, "internal") if ERRORS[c][0] == status]
            op["responses"][str(status)] = {
                "description": "Error: " + ", ".join(codes),
                "content": {"application/json": {"schema": _ref("ErrorBody")}}}
        paths.setdefault(e.path, {})[e.method.lower()] = op
    return {
        "openapi": "3.1.0",
        "info": {"title": "oh-my-slam", "version": version,
                 "description": f"Schema {SCHEMA_VERSION}. {document()['description']} Every "
                 "response carries per-stage timings in a Server-Timing header."},
        "jsonSchemaDialect": "https://json-schema.org/draft/2020-12/schema",
        "paths": paths,
        "components": {"schemas": _components()},
    }
