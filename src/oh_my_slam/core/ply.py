"""Binary little-endian PLY for coloured point clouds (optionally with an object label)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class PointCloud:
    """xyz (N, 3) float32, rgb (N, 3) uint8, optional per-point label (N,) int32 (0 = none)."""

    xyz: NDArray[np.float32]
    rgb: NDArray[np.uint8]
    label: NDArray[np.int32] | None = None

    def __post_init__(self) -> None:
        self.xyz = np.ascontiguousarray(self.xyz, dtype=np.float32).reshape(-1, 3)
        self.rgb = np.ascontiguousarray(self.rgb, dtype=np.uint8).reshape(-1, 3)
        if len(self.xyz) != len(self.rgb):
            raise ValueError("xyz and rgb must have the same length")
        if self.label is not None:
            self.label = np.ascontiguousarray(self.label, dtype=np.int32).reshape(-1)
            if len(self.label) != len(self.xyz):
                raise ValueError("label must have one entry per point")

    def __len__(self) -> int:
        return len(self.xyz)

    def subset(self, index: NDArray[Any]) -> PointCloud:
        return PointCloud(
            self.xyz[index], self.rgb[index], None if self.label is None else self.label[index]
        )

    @staticmethod
    def concat(clouds: list[PointCloud]) -> PointCloud:
        if not clouds:
            return PointCloud(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8))
        labels = None
        if all(c.label is not None for c in clouds):
            labels = np.concatenate([c.label for c in clouds if c.label is not None])
        return PointCloud(
            np.concatenate([c.xyz for c in clouds]), np.concatenate([c.rgb for c in clouds]), labels
        )


def _dtype(with_label: bool) -> np.dtype[Any]:
    fields: list[tuple[str, str]] = [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    if with_label:
        fields.append(("label", "<i4"))
    return np.dtype(fields)


def ply_bytes(cloud: PointCloud, comment: str | None = None) -> bytes:
    with_label = cloud.label is not None
    header = ["ply", "format binary_little_endian 1.0"]
    if comment:
        header.append(f"comment {comment}")
    header += [
        f"element vertex {len(cloud)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
    ]
    if with_label:
        header.append("property int label")
    header.append("end_header")
    arr = np.empty(len(cloud), dtype=_dtype(with_label))
    arr["x"], arr["y"], arr["z"] = cloud.xyz[:, 0], cloud.xyz[:, 1], cloud.xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = cloud.rgb[:, 0], cloud.rgb[:, 1], cloud.rgb[:, 2]
    if with_label and cloud.label is not None:
        arr["label"] = cloud.label
    return ("\n".join(header) + "\n").encode("ascii") + arr.tobytes()


def write_ply(path: Path, cloud: PointCloud, comment: str | None = None) -> None:
    from oh_my_slam.core.atomic import atomic_write_bytes

    atomic_write_bytes(path, ply_bytes(cloud, comment))


_PLY_TYPES = {
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
    "uchar": "u1",
    "uint8": "u1",
    "char": "i1",
    "int8": "i1",
    "short": "<i2",
    "ushort": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
}


def parse_ply(data: bytes) -> PointCloud:
    """Parse a binary little-endian PLY with a vertex element (extra elements are ignored)."""
    end = data.find(b"end_header\n")
    if not data.startswith(b"ply\n") or end < 0:
        raise ValueError("not a PLY file")
    header = data[:end].decode("ascii").splitlines()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError("only binary_little_endian PLY is supported")
    count = 0
    fields: list[tuple[str, str]] = []
    in_vertex = False
    for line in header:
        parts = line.split()
        if parts[:2] == ["element", "vertex"]:
            count = int(parts[2])
            in_vertex = True
        elif parts and parts[0] == "element":
            in_vertex = False
        elif in_vertex and parts and parts[0] == "property":
            fields.append((parts[2], _PLY_TYPES[parts[1]]))
    arr = np.frombuffer(data, dtype=np.dtype(fields), count=count, offset=end + len(b"end_header\n"))
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1)
    names = {n for n, _ in fields}
    if {"red", "green", "blue"} <= names:
        rgb = np.stack([arr["red"], arr["green"], arr["blue"]], axis=1)
    else:
        rgb = np.zeros((count, 3), np.uint8)
    label = arr["label"].astype(np.int32) if "label" in names else None
    return PointCloud(xyz, rgb, label)


def read_ply(path: Path) -> PointCloud:
    return parse_ply(Path(path).read_bytes())
