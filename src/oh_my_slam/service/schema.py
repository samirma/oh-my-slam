"""The API's JSON Schema (``api.schema.json``, a copy of ``specs/api.schema.json``) and its
validators: every request body is validated against it, and the tests check every response body
against it."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from oh_my_slam.service.errors import invalid

SCHEMA_FILE = Path(__file__).with_name("api.schema.json")
SCHEMA_VERSION = "oms/2.5"


@cache
def document() -> dict[str, Any]:
    doc: dict[str, Any] = json.loads(SCHEMA_FILE.read_text("utf-8"))
    return doc


def defs() -> dict[str, Any]:
    out: dict[str, Any] = document()["$defs"]
    return out


@cache
def _validator(name: str) -> Draft202012Validator:
    doc = document()
    if name not in doc["$defs"]:
        raise KeyError(f"no $defs/{name} in {SCHEMA_FILE.name}")
    registry: Registry[Any] = Registry().with_resource(doc["$id"], Resource.from_contents(doc))
    return Draft202012Validator({"$ref": f"{doc['$id']}#/$defs/{name}"}, registry=registry)


def field_path(path: Any) -> str:
    """A JSON path as ``frames[0].camera_id`` (``body`` for the body itself)."""
    out = ""
    for part in path:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out or "body"


def problems(name: str, body: Any) -> list[str]:
    """Every way ``body`` fails ``$defs/name``, as ``path: message`` (empty when it is valid)."""
    return [f"{field_path(e.absolute_path)}: {e.message}"
            for e in sorted(_validator(name).iter_errors(body), key=lambda e: list(e.path))]


def validate(name: str, body: Any) -> None:
    """``invalid_request`` naming the first failing field when ``body`` fails ``$defs/name``."""
    errors = sorted(_validator(name).iter_errors(body), key=lambda e: list(e.absolute_path))
    if errors:
        # the deepest error of the first failing path: oneOf/if-then report their branches
        e = errors[0]
        while e.context:
            e = sorted(e.context, key=lambda c: -len(c.absolute_path))[0]
        raise invalid(field_path(e.absolute_path), f"{field_path(e.absolute_path)}: {e.message}",
                      schema=name)
