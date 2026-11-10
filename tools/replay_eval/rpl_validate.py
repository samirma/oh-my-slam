"""Validate response bodies against specs/api.schema.json.

With the optional ``jsonschema`` package the whole schema (draft 2020-12) is checked. Without it,
a minimal check: the definition's required keys (its own and those of a top-level ``allOf``) and
the ``schema`` version pattern.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

try:  # optional
    import jsonschema
except ImportError:  # covered by test_validate_without_jsonschema
    jsonschema = None

DEFAULT_SCHEMA = Path(__file__).resolve().parents[2] / "specs" / "api.schema.json"


class Validator:
    def __init__(self, path: Path = DEFAULT_SCHEMA, use_jsonschema: bool = True) -> None:
        self.schema: dict[str, Any] = json.loads(path.read_text())
        self.defs: dict[str, Any] = self.schema["$defs"]
        self.full = use_jsonschema and jsonschema is not None
        self._validators: dict[str, Any] = {}

    def _validator(self, name: str) -> Any:
        if name not in self._validators:
            sub = {"$schema": self.schema["$schema"], "$defs": self.defs,
                   "$ref": f"#/$defs/{name}"}
            cls = jsonschema.validators.validator_for(sub)
            self._validators[name] = cls(sub)
        return self._validators[name]

    def errors(self, name: str, body: Any) -> list[str]:
        """Violations of ``$defs/<name>`` by ``body`` (empty when valid)."""
        if name not in self.defs:
            return [f"unknown schema definition '{name}'"]
        if self.full:
            return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
                    for e in self._validator(name).iter_errors(body)][:20]
        return self._minimal(name, body)

    def _minimal(self, name: str, body: Any) -> list[str]:
        d = self.defs[name]
        if not isinstance(body, dict):
            return [f"<root>: expected an object for {name}"]
        required = list(d.get("required", []))
        for part in d.get("allOf", []):
            ref = part.get("$ref", "")
            if ref.startswith("#/$defs/"):
                required += self.defs[ref.split("/")[-1]].get("required", [])
        out = [f"<root>: '{k}' is a required property" for k in required if k not in body]
        pattern = self.defs["SchemaVersion"]["pattern"]
        if "schema" in body and not re.match(pattern, str(body["schema"])):
            out.append(f"schema: '{body['schema']}' does not match {pattern}")
        return out
