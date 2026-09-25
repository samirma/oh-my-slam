"""PLY point clouds, binary little-endian or ASCII: xyz, optional normals, colour and object label.

Vertex properties, in this order: ``float x y z``, ``float nx ny nz`` (normals), ``uchar red green
blue`` (colour), ``int label`` (object id, 0 = unsegmented). The output is a pure function of the
cloud, the encoding and the comments, so identical inputs give byte-identical files.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

BINARY = "binary_little_endian 1.0"
ASCII = "ascii 1.0"


@dataclass
class PointCloud:
    """xyz (N, 3) float32; optional rgb (N, 3) uint8, label (N,) int32, normals (N, 3) float32."""

    xyz: NDArray[np.float32]
    rgb: NDArray[np.uint8] | None = None
    label: NDArray[np.int32] | None = None
    normals: NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        self.xyz = np.ascontiguousarray(self.xyz, dtype=np.float32).reshape(-1, 3)
        n = len(self.xyz)
        if self.rgb is not None:
            self.rgb = np.ascontiguousarray(self.rgb, dtype=np.uint8).reshape(-1, 3)
            if len(self.rgb) != n:
                raise ValueError("xyz and rgb must have the same length")
        if self.label is not None:
            self.label = np.ascontiguousarray(self.label, dtype=np.int32).reshape(-1)
            if len(self.label) != n:
                raise ValueError("label must have one entry per point")
        if self.normals is not None:
            self.normals = np.ascontiguousarray(self.normals, dtype=np.float32).reshape(-1, 3)
            if len(self.normals) != n:
                raise ValueError("normals must have one entry per point")

    def __len__(self) -> int:
        return len(self.xyz)

    def subset(self, index: NDArray[Any]) -> PointCloud:
        def pick(a: NDArray[Any] | None) -> Any:
            return None if a is None else a[index]

        return PointCloud(self.xyz[index], pick(self.rgb), pick(self.label), pick(self.normals))


# (name, numpy type, PLY type, ASCII format) per property group
_XYZ = [("x", "<f4", "float", "%.9g"), ("y", "<f4", "float", "%.9g"), ("z", "<f4", "float", "%.9g")]
_NORMALS = [("nx", "<f4", "float", "%.9g"), ("ny", "<f4", "float", "%.9g"),
            ("nz", "<f4", "float", "%.9g")]
_RGB = [("red", "u1", "uchar", "%d"), ("green", "u1", "uchar", "%d"), ("blue", "u1", "uchar", "%d")]
_LABEL = [("label", "<i4", "int", "%d")]


def _properties(cloud: PointCloud) -> list[tuple[str, str, str, str]]:
    props = list(_XYZ)
    if cloud.normals is not None:
        props += _NORMALS
    if cloud.rgb is not None:
        props += _RGB
    if cloud.label is not None:
        props += _LABEL
    return props


def ply_bytes(cloud: PointCloud, *, encoding: str = "binary", comments: Sequence[str] = ()
              ) -> bytes:
    """Serialise ``cloud``; ``encoding`` is ``binary`` or ``ascii``, one ``comment`` line each."""
    if encoding not in ("binary", "ascii"):
        raise ValueError(f"unknown PLY encoding {encoding!r}")
    props = _properties(cloud)
    header = ["ply", f"format {BINARY if encoding == 'binary' else ASCII}"]
    for c in comments:
        if "\n" in c or "\r" in c:
            raise ValueError("a PLY comment is a single line")
        header.append(f"comment {c}")
    header.append(f"element vertex {len(cloud)}")
    header += [f"property {ptype} {name}" for name, _, ptype, _ in props]
    header.append("end_header")
    arr = np.empty(len(cloud), dtype=np.dtype([(name, np_t) for name, np_t, _, _ in props]))
    for i, axis in enumerate("xyz"):
        arr[axis] = cloud.xyz[:, i]
        if cloud.normals is not None:
            arr[f"n{axis}"] = cloud.normals[:, i]
    if cloud.rgb is not None:
        for i, channel in enumerate(("red", "green", "blue")):
            arr[channel] = cloud.rgb[:, i]
    if cloud.label is not None:
        arr["label"] = cloud.label
    head = ("\n".join(header) + "\n").encode("ascii")
    if encoding == "binary":
        return head + arr.tobytes()
    body = io.BytesIO()
    if len(arr):
        np.savetxt(body, arr, fmt=[f for _, _, _, f in props], delimiter=" ", newline="\n")
    return head + body.getvalue()


_PLY_TYPES = {
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
    "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
}


@dataclass(frozen=True)
class PlyHeader:
    encoding: str  # "binary" | "ascii"
    comments: list[str]
    count: int
    fields: list[tuple[str, str]]  # vertex properties (name, numpy type)
    body_offset: int


def parse_header(data: bytes) -> PlyHeader:
    """Header of a binary little-endian or ASCII PLY whose first element is ``vertex``."""
    end = data.find(b"end_header\n")
    if not data.startswith(b"ply\n") or end < 0:
        raise ValueError("not a PLY file")
    lines = data[:end].decode("ascii").splitlines()
    fmt = next((ln.split(None, 1)[1] for ln in lines if ln.startswith("format ")), "")
    if fmt not in (BINARY, ASCII):
        raise ValueError(f"unsupported PLY format {fmt!r} (binary_little_endian or ascii 1.0)")
    comments = [ln[len("comment "):] for ln in lines if ln.startswith("comment ")]
    count, fields, element = 0, [], None
    for ln in lines:
        parts = ln.split()
        if parts[:1] == ["element"]:
            if element is None and parts[1] != "vertex":
                raise ValueError("the first PLY element must be vertex")
            element = parts[1]
            if element == "vertex":
                count = int(parts[2])
        elif parts[:1] == ["property"] and element == "vertex":
            if parts[1] == "list":
                raise ValueError("list properties are not supported on vertices")
            fields.append((parts[2], _PLY_TYPES[parts[1]]))
    if element is None:
        raise ValueError("PLY has no vertex element")
    return PlyHeader("binary" if fmt == BINARY else "ascii", comments, count, fields,
                     end + len(b"end_header\n"))


def parse_ply(data: bytes) -> PointCloud:
    """The vertex element of a PLY (later elements are ignored)."""
    h = parse_header(data)
    dtype = np.dtype(h.fields)
    if h.encoding == "binary":
        arr = np.frombuffer(data, dtype=dtype, count=h.count, offset=h.body_offset)
    elif h.count == 0:
        arr = np.zeros(0, dtype=dtype)
    else:
        arr = np.loadtxt(io.BytesIO(data[h.body_offset:]), dtype=dtype, max_rows=h.count,
                         ndmin=1)
    names = set(dtype.names or ())

    def stack(*cols: str) -> Any:
        return np.stack([arr[c] for c in cols], axis=1) if set(cols) <= names else None

    return PointCloud(stack("x", "y", "z"), stack("red", "green", "blue"),
                      arr["label"].astype(np.int32) if "label" in names else None,
                      stack("nx", "ny", "nz"))


READ_CHUNK_BYTES = 1 << 22  # binary PLY body read per step (``read_ply``)
_HEADER_READ = 1 << 16
_HEADER_MAX = 1 << 24


def read_ply(path: Path) -> PointCloud:
    """``parse_ply`` of a file. A binary body is read in chunks of ``READ_CHUNK_BYTES`` straight
    into the cloud's arrays, so the file is never held in memory as a whole (a large map cloud
    would otherwise need its size again on top of the cloud)."""
    with Path(path).open("rb") as f:
        head = b""
        while b"end_header\n" not in head:
            more = f.read(_HEADER_READ)
            if not more or len(head) > _HEADER_MAX:
                break
            head += more
        h = parse_header(head)
        if h.encoding != "binary":
            return parse_ply(head + f.read())
        dtype = np.dtype(h.fields)
        names = set(dtype.names or ())
        if not {"x", "y", "z"} <= names:
            raise ValueError("PLY vertices have no x y z")
        groups = [(cols, np.empty((h.count, 3), np_t)) for cols, np_t in (
            (("x", "y", "z"), np.float32), (("red", "green", "blue"), np.uint8),
            (("nx", "ny", "nz"), np.float32)) if set(cols) <= names]
        label = np.empty(h.count, np.int32) if "label" in names else None
        step = max(1, READ_CHUNK_BYTES // dtype.itemsize)
        buf = np.empty(min(step, h.count), dtype)
        f.seek(h.body_offset)
        for start in range(0, h.count, step):
            part = buf[:min(step, h.count - start)]
            if f.readinto(part.view(np.uint8)) != part.nbytes:
                raise ValueError(f"PLY body is shorter than its {h.count} vertices")
            rows = slice(start, start + len(part))
            for cols, out in groups:
                for i, c in enumerate(cols):
                    out[rows, i] = part[c]
            if label is not None:
                label[rows] = part["label"]
    arrays = {cols[0]: out for cols, out in groups}
    return PointCloud(arrays["x"], arrays.get("red"), label, arrays.get("nx"))
