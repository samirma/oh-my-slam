"""The contracts every command must keep: stdout purity, OpenLABEL validity and the colour
contract (specs/segment.md §2.4, high_level_spec.md §3, §4). Each checker returns a list of
problems (empty = kept); :class:`ContractLog` counts them per contract and entry point."""

from __future__ import annotations

import csv
import io
import json
import re
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

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
from oh_my_slam.viewer.routes import parse_cloud_payload

MAX_LISTED = 5
# Contract → the subjects (entry points, or the pair of outputs compared) it is reported for.
CONTRACTS: dict[str, tuple[str, ...]] = {
    "stdout": ("server", "reconstruct", "mapper", "segment", "view", "server_sh"),
    "openlabel": ("reconstruct", "mapper", "segment", "view"),
    "colour": ("reconstruct", "mapper", "segment", "view"),
    "artifacts": ("segment",),
    "same_objects": ("image", "map"),
    "readonly": ("map",),
}
# Brightest channel of the dimmed background of segmented.png; every palette colour is brighter.
_DIMMED_MAX = int(255 * DIM)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# segment.sh -d: the artefact byte-identical to the result of each -f format
RESULT_ARTEFACT = {"json": "segmentation.json", "png": "segmented.png"}


def subject_of(entry: str) -> str:
    """``reconstruct.sh`` → ``reconstruct``; ``start_inference_server.sh`` → ``server``;
    ``server.sh`` (the web service) → ``server_sh``."""
    name = Path(entry).name.removesuffix(".sh")
    return {"start_inference_server": "server", "server": "server_sh"}.get(name, name)


def _rgb(c: Any) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(v) for v in c))


# ------------------------------------------------------------------------------------------------
# stdout purity


def payload_problems(data: bytes, kind: str) -> list[str]:
    """``data`` must be exactly one JSON document, one PLY, one PNG, or nothing (``kind``
    "empty")."""
    checks = {"json": _json_problems, "ply": _ply_problems, "png": _png_problems}
    if kind != "empty" and kind not in checks:
        raise ValueError(f"unknown payload kind {kind!r}")
    if kind == "empty":
        return [] if not data else [f"expected no output, got {len(data)} bytes: {data[:60]!r}"]
    if not data:
        return [f"expected one {kind.upper()} payload, got nothing"]
    return checks[kind](data)


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


def _png_problems(data: bytes) -> list[str]:
    """Exactly one PNG: the signature, then chunks — each its length, type, data and a matching
    CRC — from ``IHDR`` to ``IEND``, nothing after it, and image data that decodes."""
    if not data.startswith(PNG_SIGNATURE):
        return [f"does not start with the PNG signature: {data[:16]!r}"]
    pos, kinds = len(PNG_SIGNATURE), list[bytes]()
    while pos < len(data) and kinds[-1:] != [b"IEND"]:
        (length,) = struct.unpack(">I", data[pos:pos + 4].rjust(4, b"\0"))
        kind, end = data[pos + 4:pos + 8], pos + 12 + length
        if end > len(data):
            return [f"the chunk at byte {pos} is cut short"]
        if zlib.crc32(data[pos + 4:end - 4]) != struct.unpack(">I", data[end - 4:end])[0]:
            return [f"the {kind!r} chunk at byte {pos} fails its CRC"]
        kinds.append(kind)
        pos = end
    if kinds[:1] != [b"IHDR"] or kinds[-1:] != [b"IEND"]:
        return [f"the chunks run {kinds[:1]} … {kinds[-1:]}, not IHDR … IEND (cut short?)"]
    if pos != len(data):
        return [f"{len(data) - pos} bytes after the PNG's IEND chunk: {data[pos:pos + 60]!r}"]
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
    except Exception as exc:  # PIL raises many types
        return [f"the PNG does not decode: {type(exc).__name__}: {exc}"]
    return []


def depth_image_problems(data: bytes, size: tuple[int, int]) -> list[str]:
    """``reconstruct.sh -f depth``: one 16-bit single-channel (greyscale) PNG with the pixel size
    of the input, ``size`` = (width, height) of the upright image."""
    problems = _png_problems(data)
    if problems:
        return problems
    width, height, bits, colour = struct.unpack(">IIBB", data[16:26])
    out = []
    if (bits, colour) != (16, 0):
        out.append(f"bit depth {bits} and colour type {colour}: not a 16-bit single-channel "
                   "(greyscale) image")
    if (width, height) != tuple(size):
        out.append(f"{width}x{height} pixels; the input has {size[0]}x{size[1]}")
    return out


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


def served_cloud_problems(body: bytes | None, ids: set[int] | None) -> list[str]:
    """The viewer's ``/api/cloud?color=segment`` payload (``viewer.server.cloud_payload``) holds
    exactly the object colours of its points (the object ids of ``ids``; unsegmented grey)."""
    if body is None:
        return ["the viewer served no color=segment cloud"]
    try:
        _, arrays = parse_cloud_payload(body)
        cloud = PointCloud(arrays["position"], arrays.get("color"), arrays.get("label"))
    except (ValueError, KeyError, TypeError, struct.error) as exc:
        return [f"unreadable cloud payload: {type(exc).__name__}: {exc}"]
    return cloud_colour_problems(cloud, ids)


def png_colour_problems(rgb: NDArray[np.uint8], objs: list[DocObject]) -> list[str]:
    """The segmented image (``segmented.png``, ``segment.sh -f png``): every mask pixel (brighter
    than the dimmed background) is exactly an object colour — blended or foreign colours are
    violations; at least 90 % of the objects' colours appear."""
    uniq = np.unique(np.asarray(rgb, dtype=np.uint8).reshape(-1, 3), axis=0)
    wanted = {color_for_id(o.id) for o in objs}
    masks = uniq[uniq.max(axis=1) > _DIMMED_MAX]
    out = []
    foreign = [c for c in masks if tuple(int(v) for v in c) not in wanted]
    if foreign:
        out.append(f"{len(foreign)} mask colours are not object colours of this output: "
                   f"{[_rgb(c) for c in foreign[:MAX_LISTED]]}")
    present = {tuple(int(v) for v in c) for c in masks}
    shown = sum(color_for_id(o.id) in present for o in objs)
    if shown < int(np.ceil(0.9 * len(objs))):
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


def artifact_problems(folder: Path, stdout: bytes | None, fmt: str = "json") -> list[str]:
    """Exactly the four artefacts; the one of the result's format ``fmt`` (``segmentation.json``
    for ``-f json``, ``segmented.png`` for ``-f png``) is byte-identical to the result on
    stdout."""
    names = sorted(p.name for p in Path(folder).iterdir() if not p.name.startswith("."))
    out = []
    if names != sorted(ARTIFACT_NAMES):
        out.append(f"artefacts {names} != {sorted(ARTIFACT_NAMES)}")
    same = RESULT_ARTEFACT[fmt]
    if stdout is not None and (Path(folder) / same).read_bytes() != stdout:
        out.append(f"{same} differs from the -f {fmt} result on stdout")
    return out


def tree_digest(root: Path) -> str:
    """Every entry under ``root`` — hidden ones (``.staging/``, ``.lock``) included: its path,
    type, size, modification time and contents. Equal digests: nothing in the folder changed."""
    import hashlib

    h = hashlib.sha256()
    root = Path(root)
    for p in sorted(root.rglob("*")):
        st = p.lstat()
        kind = "l" if p.is_symlink() else "d" if p.is_dir() else "f"
        h.update(f"{p.relative_to(root)}\0{kind}\0{st.st_size}\0{st.st_mtime_ns}\0".encode())
        if kind == "f":
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
    return h.hexdigest()


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
