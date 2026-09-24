"""The contracts every command must keep: stdout purity, OpenLABEL validity and the colour
contract (high_level_spec.md §2.4, §3, §4). Each checker returns a list of problems (empty = kept);
:class:`ContractLog` counts them per contract and entry point."""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.ply import PointCloud, parse_header, parse_ply
from oh_my_slam.schema.validate import validation_errors
from oh_my_slam.segmentation.artifacts import ARTIFACT_NAMES
from oh_my_slam.segmentation.catalog import CSV_HEADER
from oh_my_slam.segmentation.colors import (
    UNSEGMENTED,
    color_for_id,
    color_hex_for_id,
    hex_to_rgb,
    segment_colors,
)
from oh_my_slam.segmentation.render import DIM
from oh_my_slam.tools.evaluate.scene import DocObject

MAX_LISTED = 5
# Contract → the subjects (entry points, or the pair of outputs compared) it is reported for.
CONTRACTS: dict[str, tuple[str, ...]] = {
    "stdout": ("server", "reconstruct", "mapper", "segment", "view"),
    "openlabel": ("reconstruct", "mapper", "segment", "view"),
    "colour": ("reconstruct", "mapper", "segment", "view"),
    "artifacts": ("segment",),
    "same_objects": ("image", "map"),
    "readonly": ("map",),
    "console_errors": ("view",),
}
# Brightest channel of the dimmed background of segmented.png; every palette colour is brighter.
_DIMMED_MAX = int(255 * DIM)


def subject_of(entry: str) -> str:
    """``reconstruct.sh`` → ``reconstruct``; ``start_inference_server.sh`` → ``server``."""
    name = Path(entry).name.removesuffix(".sh")
    return "server" if name == "start_inference_server" else name


def _rgb(c: Any) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(v) for v in c))


# ------------------------------------------------------------------------------------------------
# stdout purity


def payload_problems(data: bytes, kind: str) -> list[str]:
    """``data`` must be exactly one JSON document, one PLY, or nothing (``kind`` "empty")."""
    if kind not in ("empty", "json", "ply"):
        raise ValueError(f"unknown payload kind {kind!r}")
    if kind == "empty":
        return [] if not data else [f"expected no output, got {len(data)} bytes: {data[:60]!r}"]
    if not data:
        return [f"expected one {kind.upper()} payload, got nothing"]
    return _json_problems(data) if kind == "json" else _ply_problems(data)


def _json_problems(data: bytes) -> list[str]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [f"not UTF-8: {exc}"]
    if not text.lstrip().startswith("{"):
        return [f"does not start with a JSON object: {text[:60]!r}"]
    try:
        json.loads(text)  # a banner or a second document fails here
    except json.JSONDecodeError as exc:
        return [f"not exactly one JSON document: {exc}"]
    return []


def _ply_problems(data: bytes) -> list[str]:
    try:
        head = parse_header(data)
    except (ValueError, KeyError, UnicodeDecodeError) as exc:
        return [f"not a PLY: {exc}"]
    body = data[head.body_offset:]
    if head.encoding == "binary":
        want = head.count * np.dtype(head.fields).itemsize
        if len(body) != want:
            return [f"binary body is {len(body)} bytes, {head.count} vertices need {want}"]
        return []
    rows = body.splitlines()
    if len(rows) != head.count or (head.count and not body.endswith(b"\n")):
        return [f"ASCII body has {len(rows)} lines for {head.count} vertices"]
    try:
        parse_ply(data)
    except ValueError as exc:
        return [f"ASCII body does not parse: {exc}"]
    return []


# ------------------------------------------------------------------------------------------------
# OpenLABEL


def parse_scene(data: bytes) -> tuple[dict[str, Any] | None, list[str]]:
    """The document and its OpenLABEL problems (schema + the checks the schema cannot express)."""
    try:
        doc = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, [f"not JSON: {exc}"]
    return doc, validation_errors(doc)


# ------------------------------------------------------------------------------------------------
# colour contract


def scene_colour_problems(objs: list[DocObject]) -> list[str]:
    """Every object's ``color`` and ``color_hex`` are the sRGB triple of its id."""
    out = []
    for o in objs:
        if o.id < 1:
            out.append(f"object id {o.id} is not positive")
            continue
        want = color_for_id(o.id)
        if o.color_hex is None or hex_to_rgb(o.color_hex) != want:
            out.append(f"id {o.id}: color_hex {o.color_hex} != {_rgb(want)}")
        if o.color is None or tuple(o.color) != want:
            out.append(f"id {o.id}: color {o.color} != {list(want)}")
    return out


def cloud_colour_problems(cloud: PointCloud, ids: set[int] | None) -> list[str]:
    """A ``color=segment`` cloud: with a ``label`` property every point has exactly its object's
    colour (unsegmented mid-grey); without one every colour is an object colour of ``ids`` or
    the grey. Labels must be objects of ``ids`` when that is known."""
    if cloud.rgb is None:
        return ["no colour properties"]
    out = []
    if cloud.label is not None:
        bad = np.any(cloud.rgb != segment_colors(cloud.label), axis=1)
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            out.append(f"{int(bad.sum())} points not in their object's colour (e.g. label "
                       f"{int(cloud.label[i])}: {_rgb(cloud.rgb[i])})")
        if ids is not None:
            unknown = sorted(set(np.unique(cloud.label).tolist()) - {0} - ids)
            if unknown:
                out.append(f"labels not in the scene: {unknown[:MAX_LISTED]}")
        return out
    if ids is None:
        return ["no label property and no object list to check the colours against"]
    allowed = {UNSEGMENTED} | {color_for_id(i) for i in ids}
    foreign = [c for c in np.unique(cloud.rgb, axis=0) if tuple(int(v) for v in c) not in allowed]
    if foreign:
        out.append(f"{len(foreign)} colours are neither object colours nor grey: "
                   f"{[_rgb(c) for c in foreign[:MAX_LISTED]]}")
    return out


def png_colour_problems(rgb: NDArray[np.uint8], objs: list[DocObject], sheet: bool) -> list[str]:
    """``segmented.png``: every mask pixel (brighter than the dimmed background and not a grey of
    the contact sheet's headers) is exactly an object colour — blended or foreign colours are
    violations; the objects' colours appear (image: ≥ 90 % of them; map sheet: at least one)."""
    uniq = np.unique(np.asarray(rgb, dtype=np.uint8).reshape(-1, 3), axis=0)
    wanted = {color_for_id(o.id) for o in objs}
    grey = (uniq[:, 0] == uniq[:, 1]) & (uniq[:, 1] == uniq[:, 2])
    masks = uniq[(uniq.max(axis=1) > _DIMMED_MAX) & ~grey]
    out = []
    foreign = [c for c in masks if tuple(int(v) for v in c) not in wanted]
    if foreign:
        out.append(f"{len(foreign)} mask colours are not object colours of this output: "
                   f"{[_rgb(c) for c in foreign[:MAX_LISTED]]}")
    present = {tuple(int(v) for v in c) for c in masks}
    shown = sum(color_for_id(o.id) in present for o in objs)
    need = min(1, len(objs)) if sheet else int(np.ceil(0.9 * len(objs)))
    if shown < need:
        out.append(f"only {shown} of {len(objs)} object colours appear in the image")
    return out


def catalog_csv_problems(text: str, objs: list[DocObject]) -> list[str]:
    lines = text.splitlines()
    out = []
    if not lines or lines[0] != ",".join(CSV_HEADER):
        out.append(f"header {lines[0] if lines else ''!r} != {','.join(CSV_HEADER)!r}")
    rows = list(csv.DictReader(io.StringIO(text)))
    ids = {o.id for o in objs}
    got = {int(r["id"]) for r in rows if r.get("id", "").isdigit()}
    if got != ids:
        out.append(f"catalog.csv ids differ from the scene: {sorted(got ^ ids)[:MAX_LISTED]}")
    for r in rows:
        if r.get("id", "").isdigit() and r.get("color_hex") != color_hex_for_id(int(r["id"])):
            out.append(f"catalog.csv id {r['id']}: {r.get('color_hex')} != "
                       f"{color_hex_for_id(int(r['id']))}")
    return out


_HEX = r"#[0-9a-fA-F]{6}"


def catalog_md_problems(text: str, objs: list[DocObject]) -> list[str]:
    """Each table row's swatch and ``color_hex`` code are the colour of the row's id."""
    out, got = [], set()
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        oid = next((int(c) for c in cells if c.isdigit()), None)
        if not line.startswith("|") or oid is None:
            continue
        got.add(oid)
        want = color_hex_for_id(oid).lower()
        colours = re.findall(rf"color:\s*({_HEX})", line) + re.findall(rf"`({_HEX})`", line)
        if not colours or any(c.lower() != want for c in colours):
            out.append(f"catalog.md id {oid}: {colours} != {want}")
    ids = {o.id for o in objs}
    if got != ids:
        out.append(f"catalog.md ids differ from the scene: {sorted(got ^ ids)[:MAX_LISTED]}")
    return out


def same_objects_problems(a: list[DocObject], b: list[DocObject], geometry: bool) -> list[str]:
    """``a`` and ``b`` hold the same objects: ids, labels and colours (and cuboids if
    ``geometry``)."""
    def key(o: DocObject) -> tuple[Any, ...]:
        cub = tuple(round(v, 5) for v in o.cuboid) if geometry and o.cuboid else None
        return o.label, o.color_hex, cub

    ka, kb = {o.id: key(o) for o in a}, {o.id: key(o) for o in b}
    diff = sorted(i for i in ka.keys() | kb.keys() if ka.get(i) != kb.get(i))
    if not diff:
        return []
    return [f"{len(diff)} objects differ (ids {diff[:MAX_LISTED]}): "
            + "; ".join(f"{i}: {ka.get(i)} vs {kb.get(i)}" for i in diff[:2])]


def artifact_problems(folder: Path, stdout: bytes | None) -> list[str]:
    """Exactly the five artefacts; ``segmentation.json`` is byte-identical to the stdout JSON."""
    names = sorted(p.name for p in Path(folder).iterdir() if not p.name.startswith("."))
    out = []
    if names != sorted(ARTIFACT_NAMES):
        out.append(f"artefacts {names} != {sorted(ARTIFACT_NAMES)}")
    if stdout is not None and (Path(folder) / "segmentation.json").read_bytes() != stdout:
        out.append("segmentation.json differs from the JSON on stdout")
    return out


# ------------------------------------------------------------------------------------------------


@dataclass
class ContractLog:
    """Checks per (contract, subject); a metric value is the number of failed checks."""

    checks: dict[tuple[str, str], dict[str, list[str]]] = field(default_factory=dict)

    def check(self, contract: str, subject: str, name: str, problems: list[str]) -> None:
        if subject not in CONTRACTS.get(contract, ()):
            raise ValueError(f"no contract {contract}.{subject}")
        self.checks.setdefault((contract, subject), {})[name] = problems

    def results(self) -> list[tuple[str, float | None, dict[str, Any], str | None]]:
        """(metric id, value, detail, error) for every contract and subject."""
        out = []
        for contract, subjects in CONTRACTS.items():
            for subject in subjects:
                done = self.checks.get((contract, subject), {})
                failed = {k: v[:MAX_LISTED] for k, v in done.items() if v}
                detail = {"checked": len(done), "failed": failed}
                error = None if done else "nothing was checked (the commands did not run or failed)"
                out.append((f"contract.{contract}.{subject}",
                            float(len(failed)) if done else None, detail, error))
        return out
