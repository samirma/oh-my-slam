"""Test doubles of the service's backend (``oh_my_slam.service.backend.Backend``) and helpers to
build update requests.

``FakeBackend`` plays the mapper: each frame's image bytes name its true camera pose in a world
frame (``truth``), and the mapper's internal frame is ``gauge`` applied to that world (an
arbitrary similarity, which a "rebuild" may change). Its map is a JSON file in the mapper
folder, so a clone of the folder is a copy of the map, as with the real mapper."""

from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from oh_my_slam.core.types import Intrinsics
from oh_my_slam.service.anchor import Sim3
from oh_my_slam.service.backend import (
    InternalFrame,
    InternalMap,
    Mono,
    NothingRegisteredError,
    Registered,
)


def textured(seed: int, w: int = 320, h: int = 240) -> bytes:
    """A sharp, textured JPEG (features and texture well above the rejection thresholds)."""
    import cv2

    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, (h // 4, w // 4, 3), dtype=np.uint8)
    img = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    for _ in range(25):
        x, y = (int(v) for v in rng.integers(0, [w, h]))
        cv2.circle(img, (x, y), int(rng.integers(3, 20)), tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def plain(w: int = 320, h: int = 240) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 198, 190)).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def png16(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint16)).save(buf, format="PNG")
    return buf.getvalue()


def pose(yaw_deg: float, x: float, y: float, z: float = 1.0, tilt_deg: float = 0.0) -> np.ndarray:
    """A camera looking horizontally along ``yaw`` (map frame, z up), centre (x, y, z), tilted
    down by ``tilt``: optical z forward, x right, y down."""
    a, t = math.radians(yaw_deg), math.radians(tilt_deg)
    fwd = np.array([math.cos(a) * math.cos(t), math.sin(a) * math.cos(t), -math.sin(t)])
    right = np.array([math.sin(a), -math.cos(a), 0.0])
    down = np.cross(fwd, right)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([right, down, fwd])
    T[:3, 3] = [x, y, z]
    return T


CAMERA = {"fx": 300.0, "fy": 300.0, "cx": 160.0, "cy": 120.0, "width": 320, "height": 240}


@dataclass
class FakeBackend:
    truth: dict[str, np.ndarray] = field(default_factory=dict)  # sha256 of the image -> T_world_cam
    gauge: Sim3 = field(default_factory=lambda: Sim3(0.5, np.eye(3), np.array([1.0, 2.0, 0.0])))
    refuse: set[str] = field(default_factory=set)  # shas the mapper does not place
    depth_m: dict[str, float] = field(default_factory=dict)  # sha -> median depth (world m)
    floor_z: float | None = 0.0  # world floor
    noise: dict[str, np.ndarray] = field(default_factory=dict)  # sha -> world offset of the pose
    ready: bool = True
    mono: float | None = 2.0
    rebuild_with: Sim3 | None = None  # a new gauge applied to every stored frame on register
    calls: list[list[str]] = field(default_factory=list)
    masks: list[dict[str, str]] = field(default_factory=list)  # the ignore masks of each call
    points_z: np.ndarray | None = None  # the cloud's heights (world), for a mapper without floor
    nothing: bool = False
    stats: dict[str, Any] = field(default_factory=lambda: {"observations": 400.0,
                                                           "reproj_error": 0.6})
    mono_depth: dict[str, np.ndarray] = field(default_factory=dict)  # sha -> true z-depth
    mono_scale: float = 1.0
    descriptors: dict[str, np.ndarray] = field(default_factory=dict)
    infers: list[tuple[str, bool]] = field(default_factory=list)
    infer_error: bool = False

    def add(self, image: bytes, T: np.ndarray, depth_m: float = 2.0) -> bytes:
        sha = hashlib.sha256(image).hexdigest()
        self.truth[sha] = T
        self.depth_m[sha] = depth_m
        return image

    def require_inference(self) -> None:
        from oh_my_slam.service.errors import ApiError

        if not self.ready:
            raise ApiError("inference_unavailable", "down (test)")

    def median_depth(self, image: Path, K: Intrinsics) -> float | None:
        return self.mono

    def infer(self, image: Path, K: Intrinsics, gravity: bool = False) -> Mono:
        """Monocular depth: ``mono_depth`` of the image's bytes (times ``mono_scale``), else a
        flat 2 m; the retrieval descriptor ``descriptors`` gives; up when asked."""
        from oh_my_slam.service.errors import ApiError

        self.infers.append((image.name, gravity))
        if self.infer_error:
            raise ApiError("inference_unavailable", "down (test)")
        sha = hashlib.sha256(image.read_bytes()).hexdigest()
        d = self.mono_depth.get(sha)
        depth = (np.full((K.height, K.width), 2.0) if d is None else d * self.mono_scale)
        return Mono(depth.astype(np.float32), K, self.descriptors.get(sha),
                    np.array([0.0, -1.0, 0.0]) if gravity else None)

    def _internal(self, sha: str) -> np.ndarray:
        T = self.truth[sha].copy()
        T[:3, 3] = T[:3, 3] + self.noise.get(sha, np.zeros(3))
        return self.gauge.pose(T)

    def register(self, mapper_dir: Path, inputs: list[Path], intrinsics: dict[Path, Intrinsics],
                 log: Path, masks: dict[Path, Path] | None = None) -> Registered:
        self.calls.append([p.name for p in inputs])
        self.masks.append({p.name: m.name for p, m in (masks or {}).items()})
        if self.nothing:
            raise NothingRegisteredError("nothing (test)")
        mapper_dir.mkdir(parents=True, exist_ok=True)
        f = mapper_dir / "fake.json"
        state = json.loads(f.read_text()) if f.exists() else {"frames": [], "next": 0}
        if self.rebuild_with is not None:
            for fr in state["frames"]:
                old = Sim3.from_dict(state["gauge"])
                T = old.inverse().pose(np.array(fr["T"]))
                T[:3, 3] += self.noise.get(fr["sha"], np.zeros(3))
                fr["T"] = self.rebuild_with.pose(T).tolist()
            self.gauge = self.rebuild_with
        rejected = []
        for p in inputs:
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            if sha in self.refuse or sha not in self.truth:
                rejected.append(str(p))
                continue
            state["frames"].append({"name": f"f{state['next']:06d}", "source": str(p),
                                    "T": self._internal(sha).tolist(), "sha": sha,
                                    "fx": intrinsics[p].fx})
            state["next"] += 1
        state["gauge"] = self.gauge.to_dict()
        state["rebuilt"] = self.rebuild_with is not None
        f.write_text(json.dumps(state))
        return Registered(rejected, objects=3)

    def read(self, mapper_dir: Path) -> InternalMap:
        state = json.loads((mapper_dir / "fake.json").read_text())
        gauge = Sim3.from_dict(state["gauge"])
        frames = [InternalFrame(f["name"], f["source"], np.array(f["T"]), f["fx"],
                                dict(self.stats)) for f in state["frames"]]
        shas = {f["name"]: f["sha"] for f in state["frames"]}

        def depth(name: str) -> np.ndarray | None:
            sha = shas.get(name)
            if sha is None or sha not in self.depth_m:
                return None
            return np.full((24, 32), self.depth_m[sha] * gauge.s, np.float32)

        floor = None if self.floor_z is None else float(gauge.point(np.array([0, 0, self.floor_z]))[2])
        lo = gauge.point(np.array([-1.0, -1.0, 0.0]))
        hi = gauge.point(np.array([3.0, 3.0, 2.0]))
        pz = None if self.points_z is None else self.points_z * gauge.s + gauge.t[2]
        return InternalMap(frames, floor, (np.minimum(lo, hi).tolist(), np.maximum(lo, hi).tolist()),
                           bool(state.get("rebuilt")), depth, pz)

    def terminate(self) -> None:
        pass


class FakeInference:
    def __init__(self, state: str = "ready") -> None:
        self.value = state

    def state(self) -> dict[str, Any]:
        return {"state": self.value, "device": "mps", "memory_used_gb": 1.0,
                "memory_cap_gb": 20.0, "error": None}


def multipart(request: dict[str, Any], files: dict[str, bytes],
              extra: tuple[tuple[str, bytes], ...] = ()) -> tuple[bytes, str]:
    """A multipart/form-data body and its content type (``extra`` parts may repeat a name)."""
    boundary = "----omsboundary7MA4YWxkTrZu0gW"
    out = io.BytesIO()
    for name, data, ctype in [("request", json.dumps(request).encode(), "application/json"),
                              *((n, d, "application/octet-stream") for n, d in files.items()),
                              *((n, d, "application/octet-stream") for n, d in extra)]:
        out.write(f"--{boundary}\r\n".encode())
        out.write(f'Content-Disposition: form-data; name="{name}"; filename="{name}"\r\n'.encode())
        out.write(f"Content-Type: {ctype}\r\n\r\n".encode())
        out.write(data)
        out.write(b"\r\n")
    out.write(f"--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"
