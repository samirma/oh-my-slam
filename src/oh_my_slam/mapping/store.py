"""The on-disk map: layout, read-only access, the update lock and the staged commit.

Layout (all paths relative to the map folder)::

    map.json            format version, map frame, scale, next ids, update history (written last)
    frames/fNNNNNN.jpg  keyframe images
    frames.json         per keyframe: camera, K, T_map_cam, registration stats, pose source, update
    per_frame/fNNNNNN/  depth.npy (float16, metric, aligned), valid.png (latest wins),
                        instances.json (RLE masks, labels, scores, object ids), descriptor.npy
    sfm/database.db     COLMAP database;  sfm/model/  COLMAP model in map coordinates
    cloud.ply  cloud_objects.npy  objects.json  objects/points_NNNNNN.npy
    scene.json          cached full scene

Updates write every new or changed file into ``.staging/`` first. ``commit`` writes the list of
staged files to ``.staging/COMMIT`` (the commit point), moves them into place and writes
``map.json`` last. An update killed before the commit point leaves the map untouched (the staging
folder is discarded by the next update); one killed after it is rolled forward by the next update,
and read-only readers already see the committed files through an overlay.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.atomic import atomic_save_npy, atomic_write_bytes, atomic_write_json, fsync_dir
from oh_my_slam.core.errors import MapLockedError, NotAMapError
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.version import MAP_FORMAT_VERSION

MAP_JSON = "map.json"
FRAMES_JSON = "frames.json"
OBJECTS_JSON = "objects.json"
SCENE_JSON = "scene.json"
CLOUD_PLY = "cloud.ply"
CLOUD_OBJECTS = "cloud_objects.npy"
SFM_DB = "sfm/database.db"
SFM_MODEL = "sfm/model"
STAGING = ".staging"
LOCK = ".lock"
COMMIT = "COMMIT"


def frame_name(index: int) -> str:
    return f"f{index:06d}"


def frame_file(name: str, file: str) -> str:
    """Map-relative path of a keyframe's per-frame file (``depth.npy``, ``valid.png``, …)."""
    return f"per_frame/{name}/{file}"


def load_depth(path_of: Callable[[str], Path], name: str) -> NDArray[np.float32]:
    """A keyframe's aligned metric depth (``path_of`` resolves map-relative paths)."""
    return np.load(path_of(frame_file(name, "depth.npy"))).astype(np.float32)


def load_valid(path_of: Callable[[str], Path], name: str,
               depth: NDArray[Any] | None = None) -> NDArray[np.bool_]:
    """A keyframe's validity with latest wins applied (``valid.png``), or ``depth > 0`` for a
    keyframe stored without it."""
    from oh_my_slam.core.images import load_png

    p = path_of(frame_file(name, "valid.png"))
    if p.exists():
        return load_png(p) > 0
    return (load_depth(path_of, name) if depth is None else np.asarray(depth)) > 0


@dataclass
class FrameRecord:
    index: int
    name: str
    image: str  # relative path of the keyframe image
    source: str  # original input (file, or video + time in the video)
    camera_id: int
    width: int
    height: int
    K: Intrinsics  # full resolution
    T_map_cam: Pose
    grid_width: int
    grid_height: int
    pose_source: str = "sfm"
    update_id: int = 1
    depth_scale: float = 1.0
    low_confidence: bool = False
    stats: dict[str, Any] = field(default_factory=dict)
    up_cam: list[float] | None = None

    @property
    def K_grid(self) -> Intrinsics:
        return self.K.resized(self.grid_width, self.grid_height)

    @property
    def order_key(self) -> tuple[float, ...]:
        """Processing order for code that must visit keyframes one at a time: updates oldest
        first, and within an update an order defined by content (the pose) — the keyframes of
        one update are one observation. The index only separates bit-identical poses."""
        T = self.T_map_cam
        return (float(self.update_id), *T.t.tolist(), *T.R.reshape(-1).tolist(),
                float(self.index))

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "image": self.image,
            "source": self.source,
            "camera_id": self.camera_id,
            "width": self.width,
            "height": self.height,
            "K": self.K.to_dict(),
            "T_map_cam": self.T_map_cam.to_dict(),
            "grid": [self.grid_width, self.grid_height],
            "pose_source": self.pose_source,
            "update_id": self.update_id,
            "depth_scale": self.depth_scale,
            "low_confidence": self.low_confidence,
            "stats": self.stats,
            "up_cam": self.up_cam,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> FrameRecord:
        return FrameRecord(
            index=int(d["index"]),
            name=d["name"],
            image=d["image"],
            source=d.get("source", ""),
            camera_id=int(d.get("camera_id", 0)),
            width=int(d["width"]),
            height=int(d["height"]),
            K=Intrinsics.from_dict(d["K"]),
            T_map_cam=Pose.from_dict(d["T_map_cam"]),
            grid_width=int(d["grid"][0]),
            grid_height=int(d["grid"][1]),
            pose_source=d.get("pose_source", "sfm"),
            update_id=int(d.get("update_id", 1)),
            depth_scale=float(d.get("depth_scale", 1.0)),
            low_confidence=bool(d.get("low_confidence", False)),
            stats=d.get("stats", {}),
            up_cam=d.get("up_cam"),
        )


def classify(root: Path) -> str:
    """'missing', 'empty' (only hidden entries), 'map' or 'other'."""
    root = Path(root)
    if not root.exists():
        return "missing"
    if not root.is_dir():
        return "other"
    if (root / MAP_JSON).is_file():
        return "map"
    visible = [p for p in root.iterdir() if not p.name.startswith(".")]
    return "empty" if not visible else "other"


def _committed_manifest(root: Path) -> list[str] | None:
    marker = root / STAGING / COMMIT
    if not marker.is_file():
        return None
    try:
        return list(json.loads(marker.read_text())["files"])
    except (json.JSONDecodeError, KeyError):
        return None


class MapReader:
    """Read-only view of a map (never writes, never locks)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        if classify(self.root) != "map" and _committed_manifest(self.root) is None:
            raise NotAMapError(f"not a map folder: {self.root}")
        manifest = _committed_manifest(self.root)
        self._overlay = set(manifest or [])
        self.meta: dict[str, Any] = self.read_json(MAP_JSON)
        frames = self.read_json(FRAMES_JSON) if self.exists(FRAMES_JSON) else {"frames": []}
        self.frames = [FrameRecord.from_dict(f) for f in frames.get("frames", [])]

    # -- paths ---------------------------------------------------------------------------------

    def path(self, rel: str) -> Path:
        if rel in self._overlay:
            return self.root / STAGING / rel
        return self.root / rel

    def exists(self, rel: str) -> bool:
        return self.path(rel).exists()

    def read_json(self, rel: str) -> Any:
        return json.loads(self.path(rel).read_text())

    # -- per-frame data --------------------------------------------------------------------------

    def frame_dir(self, name: str) -> str:
        return f"per_frame/{name}"

    def depth(self, fr: FrameRecord) -> NDArray[np.float32]:
        return load_depth(self.path, fr.name)

    def valid(self, fr: FrameRecord) -> NDArray[np.bool_]:
        return load_valid(self.path, fr.name)

    def instances(self, fr: FrameRecord) -> list[dict[str, Any]]:
        rel = f"{self.frame_dir(fr.name)}/instances.json"
        if not self.exists(rel):
            return []
        return list(self.read_json(rel).get("instances", []))

    def descriptor(self, fr: FrameRecord) -> NDArray[np.float32] | None:
        p = self.path(f"{self.frame_dir(fr.name)}/descriptor.npy")
        return np.load(p) if p.exists() else None

    def image_path(self, fr: FrameRecord) -> Path:
        return self.path(fr.image)

    @property
    def update_id(self) -> int:
        return int(self.meta.get("update_count", 0))


class MapTransaction:
    """Exclusive, staged update of a map folder (created if missing or empty)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self._lock_fd: int | None = None
        self.staging = self.root / STAGING
        self.created = False
        self._deleted: set[str] = set()

    # -- lifecycle -------------------------------------------------------------------------------

    def __enter__(self) -> MapTransaction:
        state = classify(self.root)
        if state == "other":
            raise NotAMapError(
                f"{self.root} is not empty and not a map; use a new or empty folder")
        self.created = state in ("missing", "empty")
        self.root.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.root / LOCK, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise MapLockedError(f"another update is running on {self.root}") from exc
        self._lock_fd = fd
        self.recover()
        # a committed roll-forward may have turned an 'empty' folder into a map
        self.created = classify(self.root) != "map"
        self.staging.mkdir()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            if self.staging.exists() and not (self.staging / COMMIT).exists():
                shutil.rmtree(self.staging, ignore_errors=True)
        finally:
            if self._lock_fd is not None:
                with contextlib.suppress(OSError):
                    fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                os.close(self._lock_fd)
                self._lock_fd = None

    def recover(self) -> None:
        """Roll a committed staging forward; discard an uncommitted one."""
        if not self.staging.exists():
            return
        manifest = _committed_manifest(self.root)
        if manifest is None:
            shutil.rmtree(self.staging, ignore_errors=True)
            return
        self._apply(manifest)

    # -- staged writes ---------------------------------------------------------------------------

    def stage(self, rel: str) -> Path:
        p = self.staging / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def current(self, rel: str) -> Path:
        """Latest version of a file: staged if written in this update, else committed."""
        staged = self.staging / rel
        return staged if staged.exists() else self.root / rel

    def write_json(self, rel: str, obj: Any) -> None:
        atomic_write_json(self.stage(rel), obj)

    def write_bytes(self, rel: str, data: bytes) -> None:
        atomic_write_bytes(self.stage(rel), data)

    def save_npy(self, rel: str, arr: NDArray[Any]) -> None:
        atomic_save_npy(self.stage(rel), arr)

    def clone_for_edit(self, rel: str) -> Path:
        """Staged copy of a committed file (APFS clone when possible) to be modified in place."""
        dst = self.stage(rel)
        src = self.root / rel
        if dst.exists() or not src.exists():
            return dst
        res = subprocess.run(["cp", "-c", str(src), str(dst)], capture_output=True)
        if res.returncode != 0:
            shutil.copyfile(src, dst)
        return dst

    def delete(self, rel: str) -> None:
        """Remove a committed file at commit time."""
        self._deleted.add(rel)
        staged = self.staging / rel
        if staged.exists():
            staged.unlink()

    # -- commit ----------------------------------------------------------------------------------

    def commit(self, meta: dict[str, Any]) -> None:
        meta = {**meta, "format_version": MAP_FORMAT_VERSION, "updated_at": time.time()}
        self.write_json(MAP_JSON, meta)
        files = sorted(
            str(p.relative_to(self.staging))
            for p in self.staging.rglob("*")
            if p.is_file() and p.name != COMMIT
        )
        files.remove(MAP_JSON)
        files.append(MAP_JSON)  # map.json is applied last
        deleted = sorted(self._deleted - set(files))
        atomic_write_json(self.staging / COMMIT,
                          {"files": files, "delete": deleted, "at": time.time()})
        fsync_dir(self.staging)
        self._apply(files, deleted)

    def _apply(self, files: list[str], deleted: list[str] | None = None) -> None:
        if deleted is None:
            marker = self.staging / COMMIT
            deleted = json.loads(marker.read_text()).get("delete", []) if marker.exists() else []
        for rel in deleted:
            (self.root / rel).unlink(missing_ok=True)
        for rel in files:
            src = self.staging / rel
            if not src.exists():
                continue  # already applied before an interruption
            dst = self.root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.replace(dst)
        fsync_dir(self.root)
        shutil.rmtree(self.staging, ignore_errors=True)


def read_meta_or_default(tx: MapTransaction) -> dict[str, Any]:
    p = tx.root / MAP_JSON
    if p.exists():
        return dict(json.loads(p.read_text()))
    return {
        "format_version": MAP_FORMAT_VERSION,
        "created_at": time.time(),
        "update_count": 0,
        "next_object_id": 1,
        "next_frame_index": 0,
        "updates": [],
    }


def read_frames(tx: MapTransaction) -> list[FrameRecord]:
    p = tx.current(FRAMES_JSON)
    if not p.exists():
        return []
    return [FrameRecord.from_dict(f) for f in json.loads(p.read_text()).get("frames", [])]


def full_tree_hash(root: Path) -> str:
    """Hash of every visible file (paths + contents); hidden entries are ignored."""
    import hashlib

    h = hashlib.sha256()
    root = Path(root)
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if any(part.startswith(".") for part in rel.parts) or not p.is_file():
            continue
        h.update(str(rel).encode())
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()
