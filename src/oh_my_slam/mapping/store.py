"""The on-disk map: layout, read-only access, the update lock and the staged commit.

Layout (all paths relative to the map folder)::

    map.json            format version, map frame, scale, next ids, update history (written last)
    frames/fNNNNNN.jpg  keyframe images
    frames.json         per keyframe: camera, K, T_map_cam, registration stats, pose source, update
    per_frame/fNNNNNN/  depth.npy (float16, metric, aligned), valid.png (latest wins),
                        instances.json (RLE masks, labels, scores, object ids), descriptor.npy
    sfm/database.db     COLMAP database;  sfm/model/  COLMAP model in map coordinates
    cloud.ply  cloud_objects.npy  objects.json  objects/points_NNNNNN.npy

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
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.atomic import (
    atomic_save_npy,
    atomic_write_bytes,
    atomic_write_json,
    clone_file,
    fsync_dir,
)
from oh_my_slam.core.errors import MapLockedError, NotAMapError
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.version import MAP_FORMAT_VERSION

MAP_JSON = "map.json"
FRAMES_JSON = "frames.json"
OBJECTS_JSON = "objects.json"
CLOUD_PLY = "cloud.ply"
CLOUD_OBJECTS = "cloud_objects.npy"
SFM_DB = "sfm/database.db"
SFM_MODEL = "sfm/model"
STAGING = ".staging"
LOCK = ".lock"
COMMIT = "COMMIT"
# what the map layout owns (README "Map folder"): a rebuild replaces these, nothing else in the
# folder (an earlier ``-o`` result, the user's own files)
OWNED_DIRS = ("frames", "per_frame", "sfm", "objects")
OWNED_FILES = (FRAMES_JSON, OBJECTS_JSON, CLOUD_PLY, CLOUD_OBJECTS)


def frame_name(index: int) -> str:
    return f"f{index:06d}"


def frame_file(name: str, file: str) -> str:
    """Map-relative path of a keyframe's per-frame file (``depth.npy``, ``valid.png``, …)."""
    return f"per_frame/{name}/{file}"


def load_depth(path_of: Callable[[str], Path], name: str) -> NDArray[np.float32]:
    """A keyframe's aligned metric depth (``path_of`` resolves map-relative paths)."""
    return np.load(path_of(frame_file(name, "depth.npy"))).astype(np.float32)


def keyframe_rgb(path: Path, rec: FrameRecord, max_side: int) -> NDArray[np.uint8]:
    """A stored keyframe's image at ``max_side`` on the grid of its depth: undistorted for a
    camera with distortion (``FrameRecord.K_grid``, ``core.images.undistort_rgb``)."""
    from oh_my_slam.core.images import load_rgb, undistort_rgb

    rgb = load_rgb(path, max_side=max_side)
    if not rec.K.k:
        return rgb
    return undistort_rgb(rgb, rec.K.resized(rgb.shape[1], rgb.shape[0]))


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
        """The camera of the keyframe's depth, validity and masks: its pinhole on the grid (the
        keyframes of a camera with distortion are inferred undistorted)."""
        return self.K.pinhole().resized(self.grid_width, self.grid_height)

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


# Entries that may sit in a folder that still counts as empty: what macOS and this tool leave
# behind. Any other entry, hidden or not (a `.git`), makes the folder non-empty.
TOOL_ENTRIES = frozenset({".DS_Store", LOCK, STAGING})


def classify(root: Path) -> str:
    """'missing', 'empty' (no entry, or only ``TOOL_ENTRIES``), 'map' or 'other'. A folder whose
    staging holds a committed update is a map even before its ``map.json`` is in place (a first
    update killed while applying its commit): the next update rolls it forward, and readers see
    it through the overlay."""
    root = Path(root)
    if not root.exists():
        return "missing"
    if not root.is_dir():
        return "other"
    if (root / MAP_JSON).is_file() or _committed_manifest(root) is not None:
        return "map"
    return "other" if any(p.name not in TOOL_ENTRIES for p in root.iterdir()) else "empty"


def refuse_non_map(root: Path) -> str:
    """``classify(root)``, raising ``NotAMapError`` for a non-empty folder that is not a map (spec
    2.3: refused and left untouched). Cheap and server-free, so callers check it first."""
    state = classify(root)
    if state == "other":
        raise NotAMapError(f"{Path(root).resolve()} is not empty and not a map; "
                           "use a new or empty folder")
    return state


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
        if classify(self.root) != "map":
            raise NotAMapError(f"not a map folder: {self.root}")
        manifest = _committed_manifest(self.root)
        self._overlay = set(manifest or [])
        self.meta: dict[str, Any] = self.read_json(MAP_JSON)
        frames = self.read_json(FRAMES_JSON) if self.exists(FRAMES_JSON) else {"frames": []}
        self.frames = [FrameRecord.from_dict(f) for f in frames.get("frames", [])]

    # -- paths ---------------------------------------------------------------------------------

    def path(self, rel: str) -> Path:
        """Where ``rel`` is read: its staged copy when a committed update lists it, unless that
        copy is gone (an update killed partway through applying its commit had moved it in
        place already), else the map's own file."""
        if rel in self._overlay:
            staged = self.root / STAGING / rel
            if staged.exists():
                return staged
        return self.root / rel

    def exists(self, rel: str) -> bool:
        return self.path(rel).exists()

    def _again(self, read: Callable[[], Any]) -> Any:
        """``read()``, once more if a file it needs vanished while an update applied its commit
        (``COMMIT`` in the staging folder: the staged copy moved into place between ``path`` and
        the read; a rebuild replaces most of the map's files, so the window is wider)."""
        try:
            return read()
        except FileNotFoundError:
            if not (self.root / STAGING / COMMIT).exists():
                raise
            self._overlay = set(_committed_manifest(self.root) or [])
            return read()

    def read_json(self, rel: str) -> Any:
        return self._again(lambda: json.loads(self.path(rel).read_text()))

    # -- per-frame data --------------------------------------------------------------------------

    def frame_dir(self, name: str) -> str:
        return f"per_frame/{name}"

    def depth(self, fr: FrameRecord) -> NDArray[np.float32]:
        out: NDArray[np.float32] = self._again(lambda: load_depth(self.path, fr.name))
        return out

    def valid(self, fr: FrameRecord) -> NDArray[np.bool_]:
        out: NDArray[np.bool_] = self._again(lambda: load_valid(self.path, fr.name))
        return out

    def instances(self, fr: FrameRecord) -> list[dict[str, Any]]:
        rel = f"{self.frame_dir(fr.name)}/instances.json"
        if not self.exists(rel):
            return []
        return list(self.read_json(rel).get("instances", []))

    def descriptor(self, fr: FrameRecord) -> NDArray[np.float32] | None:
        def read() -> NDArray[np.float32] | None:
            p = self.path(f"{self.frame_dir(fr.name)}/descriptor.npy")
            return np.load(p) if p.exists() else None
        out: NDArray[np.float32] | None = self._again(read)
        return out

    def image_path(self, fr: FrameRecord) -> Path:
        return self.path(fr.image)


class MapTransaction:
    """Exclusive, staged update of a map folder (created if missing or empty)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self._lock_fd: int | None = None
        self.staging = self.root / STAGING
        self.created = False
        self._deleted: set[str] = set()
        self._fresh = False  # ``start_over``: nothing committed is read or kept but map.json
        self._started_over: set[str] = set()
        self._kept: set[str] = set()

    # -- lifecycle -------------------------------------------------------------------------------

    def __enter__(self) -> MapTransaction:
        state = refuse_non_map(self.root)
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
        """Latest version of a file: staged if written in this update, else committed (after
        ``start_over``: staged only)."""
        staged = self.staging / rel
        return staged if staged.exists() or self._dropped(rel) else self.root / rel

    def _dropped(self, rel: str) -> bool:
        return self._fresh and rel not in self._kept

    def start_over(self, keep: tuple[str, ...] = ()) -> None:
        """Rebuild the map in this update: every committed file of the map layout (``OWNED_*``)
        but ``keep`` is deleted at commit unless staged again, and none is read (``current``,
        ``clone_for_edit``) — what the update needs of the old map (its keyframe images) it
        stages first. Other files in the folder (an earlier ``-o`` result) stay."""
        self._fresh = True
        self._kept = set(keep)
        self._started_over = set()
        for p in self.root.rglob("*"):
            rel = p.relative_to(self.root)
            if (p.is_file() and (rel.parts[0] in OWNED_DIRS or str(rel) in OWNED_FILES)
                    and str(rel) not in self._kept):
                self._started_over.add(str(rel))
        self._deleted |= self._started_over

    def resume(self) -> None:
        """Undo ``start_over``: the committed files are read and kept again, and what the rebuild
        staged of the derived files (the SfM database and model, the keyframes' files but their
        images, the objects) is discarded."""
        self._fresh = False
        self._deleted -= self._started_over
        self._started_over = set()
        for rel in (SFM_DB, SFM_MODEL, "per_frame", "objects"):
            p = self.staging / rel
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
        for p in self.staging.glob(SFM_DB + "-*"):  # SQLite's side files
            p.unlink()

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
        if dst.exists() or not src.exists() or self._dropped(rel):
            return dst
        clone_file(src, dst)
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
