"""Validation against the vendored, sha256-pinned OpenLABEL schema plus the checks the schema
does not express (stream intrinsics, top-level ``frame_intervals``, cuboid shape, quaternion norm).
"""

from __future__ import annotations

import hashlib
import json
import math
from functools import cache
from importlib import resources
from typing import Any

from oh_my_slam.schema.openlabel import SCHEMA_URL, SCHEMA_VERSION

SCHEMA_SHA256 = "22879dd20878d4fec02f96eccd401896e40988c3a77ded269b18505854bcdaa8"
_QUAT_TOL = 1e-3


class SceneValidationError(ValueError):
    """Raised with every problem found, one per line."""


def schema_bytes() -> bytes:
    return resources.files("oh_my_slam.schema").joinpath("openlabel_json_schema.json").read_bytes()


@cache
def _validator() -> Any:
    import jsonschema

    raw = schema_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SCHEMA_SHA256:
        raise RuntimeError(f"vendored OpenLABEL schema changed: sha256 {digest}")
    schema = json.loads(raw)
    return jsonschema.Draft7Validator(schema)


def schema_errors(doc: Any) -> list[str]:
    return [
        f"schema: {'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
        for e in _validator().iter_errors(doc)
    ]


def _is_num(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def _check_quat(q: Any, where: str, errors: list[str]) -> None:
    if not (isinstance(q, list) and len(q) == 4 and all(_is_num(v) for v in q)):
        errors.append(f"{where}: quaternion must be 4 numbers")
        return
    n = math.sqrt(sum(v * v for v in q))
    if abs(n - 1.0) > _QUAT_TOL:
        errors.append(f"{where}: quaternion norm {n:.6f} is not 1")


def _check_intrinsics(name: str, stream: dict[str, Any], errors: list[str]) -> None:
    props = stream.get("stream_properties", {})
    pin = props.get("intrinsics_pinhole")
    if stream.get("type") == "camera" and pin is None:
        errors.append(f"streams/{name}: camera stream without intrinsics_pinhole")
        return
    if pin is None:
        return
    w, h = pin.get("width_px"), pin.get("height_px")
    if not (isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0):
        errors.append(f"streams/{name}: width_px/height_px must be positive integers")
    cm = pin.get("camera_matrix")
    if not (isinstance(cm, list) and len(cm) == 12 and all(_is_num(v) for v in cm)):
        errors.append(f"streams/{name}: camera_matrix must be 12 numbers (3x4)")
    else:
        fx, fy = cm[0], cm[5]
        if fx <= 0 or fy <= 0:
            errors.append(f"streams/{name}: focal lengths must be positive")
        if cm[8:12] != [0, 0, 1, 0] and [float(v) for v in cm[8:12]] != [0.0, 0.0, 1.0, 0.0]:
            errors.append(f"streams/{name}: camera_matrix last row must be [0, 0, 1, 0]")
    dist = pin.get("distortion_coeffs")
    if dist is not None and not (
        isinstance(dist, list) and len(dist) in (4, 5, 8, 12, 14) and all(_is_num(v) for v in dist)
    ):
        errors.append(f"streams/{name}: distortion_coeffs must be a list of numbers")


def extra_errors(doc: Any) -> list[str]:
    errors: list[str] = []
    ol = doc.get("openlabel", {}) if isinstance(doc, dict) else {}
    md = ol.get("metadata", {})
    if md.get("schema_version") != SCHEMA_VERSION:
        errors.append("metadata.schema_version must be 1.0.0")
    if md.get("schema_url") != SCHEMA_URL:
        errors.append("metadata.schema_url must be the canonical OpenLABEL schema URL")
    for name, stream in (ol.get("streams") or {}).items():
        _check_intrinsics(name, stream, errors)
    frames = ol.get("frames")
    intervals = ol.get("frame_intervals")
    if intervals is not None:
        if not isinstance(intervals, list):
            errors.append("frame_intervals must be a list")
        else:
            covered: set[int] = set()
            for i, fi in enumerate(intervals):
                s, e = (fi or {}).get("frame_start"), (fi or {}).get("frame_end")
                if not (isinstance(s, int) and isinstance(e, int) and s <= e):
                    errors.append(f"frame_intervals[{i}]: needs integer frame_start <= frame_end")
                    continue
                covered.update(range(s, e + 1))
            if frames is not None and covered != {int(k) for k in frames}:
                errors.append("frame_intervals do not match the frame keys")
    elif frames:
        errors.append("frames present without frame_intervals")
    css = ol.get("coordinate_systems") or {}
    for key, fr in (frames or {}).items():
        for tname, tr in ((fr.get("frame_properties") or {}).get("transforms") or {}).items():
            data = tr.get("transform_src_to_dst", {})
            if "quaternion" in data:
                _check_quat(data["quaternion"], f"frames/{key}/transforms/{tname}", errors)
            for end in ("src", "dst"):
                if css and tr.get(end) not in css:
                    errors.append(f"frames/{key}/transforms/{tname}: unknown {end} {tr.get(end)}")
    for oid, obj in (ol.get("objects") or {}).items():
        cs = obj.get("coordinate_system")
        if cs is not None and css and cs not in css:
            errors.append(f"objects/{oid}: unknown coordinate system {cs}")
        for j, cub in enumerate((obj.get("object_data") or {}).get("cuboid", [])):
            val = cub.get("val")
            where = f"objects/{oid}/cuboid[{j}]"
            if not (isinstance(val, list) and len(val) == 10 and all(_is_num(v) for v in val)):
                errors.append(f"{where}: val must be 10 numbers (x,y,z,qx,qy,qz,qw,sx,sy,sz)")
                continue
            _check_quat(val[3:7], where, errors)
            if any(v < 0 for v in val[7:10]):
                errors.append(f"{where}: negative size")
    return errors


def validation_errors(doc: Any) -> list[str]:
    return schema_errors(doc) + extra_errors(doc)


def validate_scene(doc: Any) -> None:
    errors = validation_errors(doc)
    if errors:
        raise SceneValidationError("\n".join(errors))
