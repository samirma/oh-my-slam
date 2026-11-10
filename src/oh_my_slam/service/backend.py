"""What an update job needs beyond the service: the mapper and the monocular depth model.

``MapperBackend`` runs the mapper (``mapping.api.update``) in a process of its own
(``mapper_proc``): its memory (COLMAP, Open3D) goes back to the system when it ends, a crash
fails one job and not the service, and a stop of the service ends it. It reads the mapper's map
through ``mapping.store``. Tests replace the backend with a double (the ``Backend`` protocol)."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.errors import (
    DeviceMemoryError,
    ExitCode,
    OhMySlamError,
    ServerUnavailableError,
)
from oh_my_slam.core.types import Intrinsics
from oh_my_slam.service.errors import ApiError

LOG_TAIL_LINES = 20


class NothingRegisteredError(Exception):
    """The mapper placed none of the frames (the map is unchanged)."""


@dataclass
class InternalFrame:
    """A keyframe of the mapper's map, in its internal frame."""

    name: str
    source: str
    T: NDArray[np.float64]  # 4x4 T_internal_cam
    fx: float
    stats: dict[str, Any] = field(default_factory=dict)
    low_confidence: bool = False


@dataclass
class InternalMap:
    frames: list[InternalFrame]
    floor_z: float | None
    bounds: tuple[list[float], list[float]] | None
    rebuilt: bool
    # a keyframe's aligned depth on its grid (internal units), None if it has none
    depth: Callable[[str], NDArray[np.float32] | None]
    points_z: NDArray[np.float64] | None = None  # heights of the cloud's points (internal)
    # what the derived reads (``derived``: grid, points, locate) need beyond registration; each
    # answers None (or nothing) when the mapper has no such data
    image: Callable[[str], Path | None] = field(default=lambda name: None)  # the keyframe image
    cloud: Callable[[], Cloud | None] = field(default=lambda: None)  # the fused cloud
    objects: Callable[[], list[dict[str, Any]]] = field(default=lambda: [])  # internal frame


@dataclass
class Cloud:
    """The fused cloud in the internal frame: positions, sRGB colours and object ids (0: none)."""

    xyz: NDArray[np.float32]
    rgb: NDArray[np.uint8]
    object_id: NDArray[np.uint32]


@dataclass
class Registered:
    sources_rejected: list[str]
    objects: int
    seconds: float = 0.0
    stages: dict[str, float] = field(default_factory=dict)  # the mapper's stage seconds


class Backend(Protocol):
    def require_inference(self) -> None:
        """``inference_unavailable`` unless the inference server is ready."""

    def median_depth(self, image: Path, K: Intrinsics) -> float | None:
        """The median monocular depth of an image (metres), or None."""

    def register(self, mapper_dir: Path, inputs: list[Path],
                 intrinsics: dict[Path, Intrinsics], log: Path,
                 masks: dict[Path, Path] | None = None) -> Registered:
        """Map ``inputs`` into the mapper map ``mapper_dir`` (created if missing), the pixels of
        each one's ignore mask (``masks``, PNG) neither fused nor detected; raises
        ``NothingRegisteredError`` or ``ApiError``."""

    def read(self, mapper_dir: Path) -> InternalMap:
        """The mapper map ``mapper_dir``."""

    def terminate(self) -> None:
        """End the mapper process of a running job (the service stops)."""


def api_error(exc: OhMySlamError) -> ApiError:
    """The API error of a pipeline error: the inference server down or short of memory."""
    if isinstance(exc, DeviceMemoryError):
        return ApiError("out_of_memory", f"{exc}; the inference server recovers by itself — send "
                        "the request again", {"stage": "inference"})
    if isinstance(exc, ServerUnavailableError):
        return ApiError("inference_unavailable", f"{exc}; it is restarted by server.sh — retry "
                        "in a moment", {"stage": "inference"})
    return ApiError("internal", f"internal error: {exc}", {"stage": "mapper"})


def _tail(path: Path) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-LOG_TAIL_LINES:])
    except OSError:
        return ""


MAPPER_CMD = (sys.executable, "-m", "oh_my_slam.service.mapper_proc")


class MapperBackend:
    def __init__(self, cmd: tuple[str, ...] = MAPPER_CMD) -> None:
        self.cmd = cmd
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()

    def require_inference(self) -> None:
        from oh_my_slam.service.inference import inference_ready

        inference_ready()

    def median_depth(self, image: Path, K: Intrinsics) -> float | None:
        from oh_my_slam.reconstruction.api import connect_server, reconstruct_image
        from oh_my_slam.service.checks import median_depth

        try:
            client = connect_server()
            try:
                fr = reconstruct_image(image, intrinsics=K, max_side=512, want_gravity=False,
                                       client=client)
            finally:
                client.close()
        except OhMySlamError as exc:
            raise api_error(exc) from exc
        return median_depth(np.where(fr.valid, fr.depth, 0.0).astype(np.float32))

    def register(self, mapper_dir: Path, inputs: list[Path],
                 intrinsics: dict[Path, Intrinsics], log: Path,
                 masks: dict[Path, Path] | None = None) -> Registered:
        args = log.with_suffix(".args.json")
        result = log.with_suffix(".result.json")
        args.write_text(json.dumps({
            "map_dir": str(mapper_dir), "inputs": [str(p) for p in inputs],
            "intrinsics": {str(p): K.to_dict() for p, K in intrinsics.items()},
            "masks": {str(p): str(m) for p, m in (masks or {}).items()},
            "result": str(result)}))
        with log.open("ab") as out, self._lock:
            self._proc = subprocess.Popen(
                [*self.cmd, str(args)],
                stdin=subprocess.DEVNULL, stdout=out, stderr=out)
        code = self._proc.wait()
        with self._lock:
            self._proc = None
        try:
            res: dict[str, Any] = json.loads(result.read_text())
        except (OSError, ValueError):
            res = {}
        if code == 0 and "rejected" in res:
            return Registered(res["rejected"], int(res["objects"]), float(res["seconds"]),
                              dict(res.get("stages", {})))
        err = res.get("error", {})
        message = err.get("message") or f"the mapper ended with exit code {code}"
        if code == ExitCode.NOT_REGISTERED:
            raise NothingRegisteredError(message)
        if code == ExitCode.SERVER_UNAVAILABLE:
            raise ApiError("inference_unavailable", f"{message} — retry in a moment",
                           {"stage": "mapper"})
        if err.get("out_of_memory"):
            raise ApiError("out_of_memory", f"{message} — send the update again",
                           {"stage": "mapper"})
        raise ApiError("internal", f"the mapper failed: {message}",
                       {"stage": "mapper", "exit_code": code, "log": _tail(log)})

    def read(self, mapper_dir: Path) -> InternalMap:
        from oh_my_slam.core.ply import read_ply
        from oh_my_slam.mapping.store import CLOUD_OBJECTS, CLOUD_PLY, OBJECTS_JSON, MapReader

        reader = MapReader(mapper_dir)
        frames = [InternalFrame(f.name, f.source, f.T_map_cam.matrix(), float(f.K.fx),
                                dict(f.stats), bool(f.low_confidence)) for f in reader.frames]
        by_name = {f.name: f for f in reader.frames}
        bounds, points_z = None, None
        if reader.exists(CLOUD_PLY):
            pts = read_ply(reader.path(CLOUD_PLY)).xyz
            if len(pts):
                bounds = (pts.min(0).tolist(), pts.max(0).tolist())
                points_z = np.asarray(pts[:, 2], np.float64)
        updates = reader.meta.get("updates", [])
        rebuilt = bool(updates and "restarted" in updates[-1].get("notes", {}))

        def depth(name: str) -> NDArray[np.float32] | None:
            fr = by_name.get(name)
            if fr is None:
                return None
            try:
                d = reader.depth(fr)
            except FileNotFoundError:
                return None
            return np.where(reader.valid(fr), d, 0.0).astype(np.float32)

        def image(name: str) -> Path | None:
            fr = by_name.get(name)
            return None if fr is None else reader.image_path(fr)

        def cloud() -> Cloud | None:
            if not reader.exists(CLOUD_PLY):
                return None
            pc = read_ply(reader.path(CLOUD_PLY))
            n = len(pc.xyz)
            rgb = np.zeros((n, 3), np.uint8) if pc.rgb is None else np.asarray(pc.rgb, np.uint8)
            ids = np.zeros(n, np.uint32)
            if reader.exists(CLOUD_OBJECTS):
                lab = np.load(reader.path(CLOUD_OBJECTS))
                if len(lab) == n:
                    ids = np.where(lab > 0, lab, 0).astype(np.uint32)
            return Cloud(np.asarray(pc.xyz, np.float32), rgb, ids)

        def objects() -> list[dict[str, Any]]:
            if not reader.exists(OBJECTS_JSON):
                return []
            out = []
            for o in reader.read_json(OBJECTS_JSON).get("objects", []):
                obb = o.get("obb") or {}
                if "center" in obb:
                    out.append({"id": o.get("id"), "label": str(o.get("label", "")),
                                "center": obb["center"], "R": obb.get("R"),
                                "size": obb.get("size")})
            return out

        floor = reader.meta.get("floor_z")
        return InternalMap(frames, None if floor is None else float(floor), bounds, rebuilt, depth,
                           points_z, image, cloud, objects)

    def terminate(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()
