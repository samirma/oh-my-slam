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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.client.protocol import INTERACTIVE
from oh_my_slam.core.errors import (
    DeviceMemoryError,
    ExitCode,
    OhMySlamError,
    ServerUnavailableError,
)
from oh_my_slam.core.types import Intrinsics
from oh_my_slam.mapping.register_ma import MvView
from oh_my_slam.mapping.sfm import ignore_mask_path
from oh_my_slam.service.errors import ApiError

if TYPE_CHECKING:
    from oh_my_slam.segmentation.api import Detection, Region

LOG_TAIL_LINES = 20
MONO_GRID_SIDE = 640  # the monocular depth grid of an anchored map's keyframes (README)
# a map update keeps detections from this score: below the default threshold (0.5), an object
# needs two keyframes to be listed (``mapping.anchored_objects.TRUSTED_SCORE``)
MAP_MIN_SCORE = 0.2


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
    # a keyframe's ignored pixels (map.md §2 "Ignored pixels"), on its image, None: none
    ignore: Callable[[str], NDArray[np.bool_] | None] = field(default=lambda name: None)
    # the older mapper's count of its updates, the id of its latest (None: another engine)
    update_id: int | None = None


@dataclass
class Cloud:
    """The fused cloud in the internal frame: positions, sRGB colours and object ids (0: none)."""

    xyz: NDArray[np.float32]
    rgb: NDArray[np.uint8]
    object_id: NDArray[np.uint32]


@dataclass
class Mono:
    """A frame's monocular depth: grid (metres, 0: none), its intrinsics, the retrieval
    descriptor and the camera's up direction (when asked), as ``mapping.anchored`` takes it."""

    depth: NDArray[np.float32]
    K_grid: Intrinsics
    descriptor: NDArray[np.float32] | None
    up_cam: NDArray[np.float64] | None

    def as_tuple(self) -> tuple[NDArray[np.float32], Intrinsics, Any, Any]:
        return self.depth, self.K_grid, self.descriptor, self.up_cam


@dataclass
class Registered:
    sources_rejected: list[str]
    objects: int
    seconds: float = 0.0
    stages: dict[str, float] = field(default_factory=dict)  # the mapper's stage seconds


class Backend(Protocol):
    def require_inference(self) -> None:
        """``inference_unavailable`` unless the inference server is ready."""

    def interactive(self) -> Backend:
        """This backend with its inference requests served ahead of fusion (dry runs:
        inference.md, single worker)."""

    def median_depth(self, image: Path, K: Intrinsics) -> float | None:
        """The median monocular depth of an image (metres), or None."""

    def infer(self, image: Path, K: Intrinsics, gravity: bool = False) -> Mono:
        """The monocular depth of a canonical image with its intrinsics ``K`` (and the camera's
        up direction when ``gravity``)."""

    def geometry_views(self, images: list[Path], intrinsics: list[Intrinsics],
                       poses: list[NDArray[np.float64] | None], out_dir: Path) -> list[MvView]:
        """MapAnything over ``images`` with their intrinsics and the given camera-to-world
        poses; ``ApiError`` ``out_of_memory`` when the call does not fit."""

    def detect(self, image: Path, grid: tuple[int, int], labels: Sequence[str] = ()
               ) -> list[Detection]:
        """The detections of an image, masks on ``grid`` (width, height); ``labels``: the map's
        vocabulary, prompted with the defaults (``segmentation.detect.prompts``)."""

    def look_closer(self, image: Path, regions: list[Region], dets: list[Detection],
                    grid: tuple[int, int], labels: Sequence[str] = ()) -> list[Detection]:
        """The detections ``dets`` of an image looking down at its support (masks on ``grid``),
        looked at closer (``segmentation.api.look_closer``): its raised areas ``regions`` that
        they leave uncovered labelled, the other labels of each."""

    def register(self, mapper_dir: Path, inputs: list[Path],
                 intrinsics: dict[Path, Intrinsics], log: Path,
                 masks: dict[Path, Path] | None = None, labels: Sequence[str] = ()
                 ) -> Registered:
        """Map ``inputs`` into the mapper map ``mapper_dir`` (created if missing), the pixels of
        each one's ignore mask (``masks``, PNG) neither fused nor detected; raises
        ``NothingRegisteredError`` or ``ApiError``."""

    def read(self, mapper_dir: Path) -> InternalMap:
        """The mapper map ``mapper_dir``."""

    def terminate(self) -> None:
        """End the mapper process of a running job (the service stops)."""


def is_anchored(mapper_dir: Path) -> bool:
    """Whether a mapper folder holds a prior-first map (``mapping.anchored``)."""
    from oh_my_slam.mapping.anchored import STATE_JSON

    return (mapper_dir / STATE_JSON).exists()


def read_anchored(mapper_dir: Path) -> InternalMap:
    """A prior-first map as the derived reads see it: its frame is the map frame (metres), its
    keyframes' ids are the internal names, provisional keyframes count as low confidence, the
    depth of a keyframe is its grid at the map's depth scale, the cloud its fused voxels, each
    with the id of the listed object whose box holds it (the smallest), and the listed objects
    that are not struck (``mapping.anchored_objects``)."""
    from oh_my_slam.mapping.anchored import AnchoredMap
    from oh_my_slam.mapping.anchored_objects import ObjectMap

    am = AnchoredMap.load(mapper_dir)
    frames = [InternalFrame(k.id, k.name, k.T.copy(), float(k.K.fx),
                            {"inliers": k.inliers, "method": k.method}, k.state == "provisional")
              for k in am.keyframes]
    floor, pts = am.floor_and_bounds()
    bounds = (pts.min(0).tolist(), pts.max(0).tolist()) if len(pts) else None
    xyz = am.cloud.points()
    live = [o for o in ObjectMap.load(mapper_dir).live() if o.box is not None]

    def cloud() -> Cloud | None:
        if not len(xyz):
            return None
        rgb = am.cloud.rgb / np.maximum(am.cloud.count, 1)[:, None]
        ids = np.zeros(len(xyz), np.uint32)
        vol = np.full(len(xyz), np.inf)
        for o in live:
            assert o.box is not None
            v = float(np.prod(o.box.size))
            r = float(np.linalg.norm(o.box.size)) / 2
            near = np.flatnonzero(np.all(np.abs(xyz - o.box.centre) <= r, axis=1))
            hit = near[o.box.contains(xyz[near]) & (vol[near] > v)]
            ids[hit], vol[hit] = o.id, v
        return Cloud(xyz.astype(np.float32), np.clip(np.round(rgb), 0, 255).astype(np.uint8),
                     ids)

    def objects() -> list[dict[str, Any]]:
        out = []
        for o in live:
            assert o.box is not None
            out.append({"id": o.id, "label": o.label, "center": o.box.centre.tolist(),
                        "R": o.box.R.tolist(), "size": o.box.size.tolist()})
        return out

    return InternalMap(frames, floor, bounds, False, am.content_depth,
                       np.asarray(xyz[:, 2], np.float64) if len(xyz) else None, am.image_path,
                       cloud, objects, am.ignore_of)


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
    def __init__(self, cmd: tuple[str, ...] = MAPPER_CMD, priority: str | None = None) -> None:
        self.cmd = cmd
        self.priority = priority  # of its inference requests (``client.protocol.PRIORITIES``)
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()

    def interactive(self) -> MapperBackend:
        """A backend whose inference requests the server serves before fusion's; it runs no
        mapper (a dry run's)."""
        return MapperBackend(self.cmd, INTERACTIVE)

    def require_inference(self) -> None:
        from oh_my_slam.service.inference import inference_ready

        inference_ready()

    def median_depth(self, image: Path, K: Intrinsics) -> float | None:
        from oh_my_slam.mapping.checks import median_depth
        from oh_my_slam.reconstruction.api import connect_server, reconstruct_image

        try:
            client = connect_server().clone(self.priority)  # a client of its own: asked from several threads
            try:
                fr = reconstruct_image(image, intrinsics=K, max_side=512, want_gravity=False,
                                       client=client)
            finally:
                client.close()
        except OhMySlamError as exc:
            raise api_error(exc) from exc
        return median_depth(np.where(fr.valid, fr.depth, 0.0).astype(np.float32))

    def infer(self, image: Path, K: Intrinsics, gravity: bool = False) -> Mono:
        from oh_my_slam.reconstruction.api import connect_server, reconstruct_image

        try:
            # a client of its own: jobs ask for several frames at once, and the shared client
            # is not for use from several threads
            client = connect_server().clone(self.priority)
            try:
                fr = reconstruct_image(image, intrinsics=K, max_side=MONO_GRID_SIDE,
                                       want_gravity=gravity, want_descriptor=True, client=client)
            finally:
                client.close()
        except OhMySlamError as exc:
            raise api_error(exc) from exc
        up = None if fr.gravity is None else np.asarray(fr.gravity.up_cam, np.float64)
        return Mono(np.where(fr.valid, fr.depth, 0.0).astype(np.float32), fr.K_grid,
                    None if fr.descriptor is None else np.asarray(fr.descriptor, np.float32), up)

    def geometry_views(self, images: list[Path], intrinsics: list[Intrinsics],
                       poses: list[NDArray[np.float64] | None], out_dir: Path) -> list[MvView]:
        from oh_my_slam.core.types import Pose
        from oh_my_slam.reconstruction.api import connect_server
        from oh_my_slam.reconstruction.multiview import run_multiview

        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            client = connect_server().clone(self.priority)
            try:
                res, _ = run_multiview(
                    images, out_dir, intrinsics=list(intrinsics),
                    poses=[None if T is None else Pose.from_matrix(T) for T in poses],
                    want_depth=True, client=client)
            finally:
                client.close()
        except OhMySlamError as exc:
            raise api_error(exc) from exc
        return [MvView(r.pose.matrix(), np.load(str(r.depth_path)).astype(np.float32), r.K)
                for r in res]

    def detect(self, image: Path, grid: tuple[int, int], labels: Sequence[str] = ()
               ) -> list[Detection]:
        from oh_my_slam.reconstruction.api import connect_server
        from oh_my_slam.segmentation.api import detect_on_grid

        try:
            client = connect_server().clone(self.priority)
            try:
                return detect_on_grid(image, grid, client, min_score=MAP_MIN_SCORE,
                                      labels=labels)
            finally:
                client.close()
        except OhMySlamError as exc:
            raise api_error(exc) from exc

    def look_closer(self, image: Path, regions: list[Region], dets: list[Detection],
                    grid: tuple[int, int], labels: Sequence[str] = ()) -> list[Detection]:
        from oh_my_slam.reconstruction.api import connect_server
        from oh_my_slam.segmentation.api import look_closer

        try:
            client = connect_server().clone(self.priority)
            try:
                return look_closer(image, regions, dets, grid, client, MAP_MIN_SCORE, labels)
            finally:
                client.close()
        except OhMySlamError as exc:
            raise api_error(exc) from exc

    def register(self, mapper_dir: Path, inputs: list[Path],
                 intrinsics: dict[Path, Intrinsics], log: Path,
                 masks: dict[Path, Path] | None = None, labels: Sequence[str] = ()
                 ) -> Registered:
        args = log.with_suffix(".args.json")
        result = log.with_suffix(".result.json")
        args.write_text(json.dumps({
            "map_dir": str(mapper_dir), "inputs": [str(p) for p in inputs],
            "intrinsics": {str(p): K.to_dict() for p, K in intrinsics.items()},
            "masks": {str(p): str(m) for p, m in (masks or {}).items()},
            "labels": list(labels), "result": str(result)}))
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
        if is_anchored(mapper_dir):
            return read_anchored(mapper_dir)
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
                                "size": obb.get("size"), "score": o.get("score"),
                                "label_votes": o.get("label_votes") or {},
                                "observations": o.get("observations"),
                                "created_update": o.get("created_update"),
                                "last_seen_update": o.get("last_seen_update"),
                                # the views it was detected in (``objects.Sighting.to_list``)
                                "sightings": o.get("sightings") or [],
                                # the views that are no view of it (map.md §5 "Ignored
                                # pixels") and the support its box rests on
                                "masked_frames": o.get("masked_frames") or [],
                                # the vote shares without them (``MapObject.view_votes``)
                                **{k: o[k] for k in ("view_votes", "view_observations")
                                   if k in o},
                                "support_z": o.get("support_z")})
            return out

        def ignore(name: str) -> NDArray[np.bool_] | None:
            p = image(name)
            m = None if p is None else ignore_mask_path(p.parent, p.name)
            if m is None or not m.exists():
                return None
            import cv2

            got = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
            return None if got is None else got > 0

        floor = reader.meta.get("floor_z")
        count = reader.meta.get("update_count")
        return InternalMap(frames, None if floor is None else float(floor), bounds, rebuilt, depth,
                           points_z, image, cloud, objects, ignore,
                           None if count is None else int(count))

    def terminate(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()
