"""What the grid, points and locate reads need per version, derived once after each commit.

Under ``<data>/oms/maps/<name>/derived/``:

* ``<n>/volume.npz`` — the occupancy volume of version *n* (``mapping.navgrid.Volume``): the
  previous version's plus the rays of its frames that are new or moved (all of them again when a
  stored frame moved);
* ``<n>/cloud.npz`` — the fused cloud in the map frame (``xyz`` float32, ``rgb``, ``object_id``);
* ``<n>/locate.npz`` — per keyframe, the map-frame points of its keypoints;
* ``<n>/meta.json`` — the floor, door positions and objects in the map frame (written last: a
  version is derived once it exists);
* ``features/<key>.npz`` — each keyframe image's SIFT keypoints and descriptors, shared by every
  version.

``Derived`` subscribes to ``Store.preparing`` and derives each version before the store
publishes it (``prepare``, in the job, from the mapper map it is committed with): the files
above, its locate index (``index``, built once) and the answers of the ``prepared`` hooks (the
objects read's, with the default grid it uses), so a read never waits for a derivation and no
locate builds an index (high_level_spec.md §8). Each map's latest version stays in memory
(``_latest``) beside the ``CACHE_VERSIONS`` most recent others; a version's read answers are
kept per version and parameters (``DerivedVersion.answer``, one computation per key). A read
never takes a lock a derivation holds: derivations of one version are serialised by that
version's own lock. A version that preparation did not derive (an error, or a version committed
before this) is derived by its first read. A version's
mapper-map content
(depth, cloud, images, objects) comes from the mapper map it was committed with: before a commit
replaces that map, every version it serves that is not derived yet is derived from it
(``Store.replacing``); a version whose mapper map is gone and that was never derived answers
``version_gone``. Everything it needs comes through
two interfaces: the version record (``frames[]`` with ``name``, ``camera_id``, ``internal``,
``T_map_cam``, ``sigma_pos_m``, ``sigma_rot_deg``, ``sha256``; ``sim3``; ``floor_z_m``) and the
mapper map read by the backend (``backend.InternalMap``: ``depth``, ``image``, ``cloud``,
``objects`` by internal frame name)."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import queue
import shutil
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.atomic import atomic_write_bytes, atomic_write_json
from oh_my_slam.core.constants import FLOOR_SIGMA_DEFAULT_M
from oh_my_slam.core.errors import OhMySlamError
from oh_my_slam.core.images import on_grid
from oh_my_slam.core.log import get_logger
from oh_my_slam.mapping import navgrid, relocalize
from oh_my_slam.mapping.anchor import Sim3
from oh_my_slam.mapping.anchored import valid_depth
from oh_my_slam.mapping.store import frame_name
from oh_my_slam.segmentation.api import DOOR_LABEL, door_leaf
from oh_my_slam.service.backend import InternalMap
from oh_my_slam.service.cameras import Camera
from oh_my_slam.service.store import Store

log = get_logger("oh_my_slam.service.derived")

CACHE_VERSIONS = 3  # versions kept in memory beside each map's latest
ANSWERS = 16  # read answers kept per version (grid parameters, objects)
DEFAULT_CAMERA_HEIGHT_M = 1.0
# what the older mapper records of an object beyond its box, kept for the objects read
# the older mapper's sighting of an object is uncut (``objects.extent_from``) when none of its
# mask lies in the image-border band (``Sighting.border``: that share)
MAPPER_UNCUT_BORDER = 0.0
MAPPER_OBJECT_KEYS = ("score", "label_votes", "observations", "created_update",
                      "last_seen_update")  # a map with no floor at all: this far below its lowest camera


@dataclass
class DerivedVersion:
    """A derived version, its parts loaded on first use."""

    name: str
    version: int
    record: dict[str, Any]
    meta: dict[str, Any]
    dir: Path
    stands: list[navgrid.Stand]
    _volume: navgrid.Volume | None = None
    _cloud: dict[str, NDArray[Any]] | None = None
    _index: relocalize.Index | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    # read answers by key (``answer``) and the locks that compute each once
    _answers: OrderedDict[Any, Any] = field(default_factory=OrderedDict)
    _answer_locks: dict[Any, threading.Lock] = field(default_factory=dict)
    _index_lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def floor_z(self) -> float:
        return float(self.meta["floor_z"])

    @property
    def floor_sigma(self) -> float:
        """The floor's height sigma (a version derived before oms/2.2: the default)."""
        return float(self.meta.get("floor_sigma", FLOOR_SIGMA_DEFAULT_M))

    def answer(self, key: Any, compute: Callable[[], Any]) -> Any:
        """A read's answer of this version for ``key`` (its parameters): computed once, by the
        first read that needs it (the others wait for it), then kept (the ``ANSWERS`` most
        recent keys)."""
        with self._lock:
            if key in self._answers:
                self._answers.move_to_end(key)
                return self._answers[key]
            lock = self._answer_locks.setdefault(key, threading.Lock())
        with lock:
            with self._lock:
                if key in self._answers:
                    return self._answers[key]
            out = compute()
            with self._lock:
                self._answers[key] = out
                while len(self._answers) > ANSWERS:
                    gone, _ = self._answers.popitem(last=False)
                    self._answer_locks.pop(gone, None)
            return out

    def volume(self) -> navgrid.Volume:
        with self._lock:
            if self._volume is None:
                self._volume = navgrid.Volume.from_bytes((self.dir / "volume.npz").read_bytes())
            return self._volume

    def cloud(self) -> dict[str, NDArray[Any]]:
        with self._lock:
            if self._cloud is None:
                p = self.dir / "cloud.npz"
                if p.exists():
                    z = np.load(p)
                    self._cloud = {k: z[k] for k in ("xyz", "rgb", "object_id")}
                else:
                    self._cloud = {"xyz": np.zeros((0, 3), np.float32),
                                   "rgb": np.zeros((0, 3), np.uint8),
                                   "object_id": np.zeros(0, np.uint32)}
            return self._cloud


def _occupies(f: dict[str, Any]) -> bool:
    """Whether a version frame feeds occupancy and heights: confirmed by the images, or kept at
    its prior under the too-close exception (``content``, api.md §4), and not left out by the
    height gate (``height_inconsistent``, map.md §2)."""
    if f.get("height_inconsistent"):
        return False
    return f.get("pose_state") != "kept_at_prior" or bool(f.get("content"))


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def yaw_of(R: NDArray[Any]) -> float:
    """The yaw (deg, in (-90, 90]) of an upright box's x axis."""
    a = math.degrees(math.atan2(float(R[1][0]), float(R[0][0])))
    a = (a + 90.0) % 180.0 - 90.0
    return 90.0 if a == -90.0 else a


class Derived:
    def __init__(self, store: Store, read_map: Callable[[Path], InternalMap]) -> None:
        self.store = store
        self.read_map = read_map
        self._queue: queue.Queue[tuple[str, int] | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._cache: OrderedDict[tuple[str, str, int], DerivedVersion] = OrderedDict()
        self._latest: dict[str, tuple[tuple[str, str, int], DerivedVersion]] = {}
        self._prepared: dict[tuple[str, int], DerivedVersion] = {}  # derived, being committed
        self._lock = threading.Lock()
        self._version_locks: dict[tuple[str, int], threading.Lock] = {}
        self._pairs: dict[tuple[str, str], NDArray[np.int64]] = {}  # keyframe matches by image
        # what preparation also computes for each version before it is published (the objects
        # read's answer: ``objects.body``), with the version's committed JSON files
        self.prepared: list[Callable[[DerivedVersion, dict[str, Any]], Any]] = []
        store.listeners.append(self.committed)
        store.pruned.append(self.forget)
        store.replacing.append(self.replacing)
        store.preparing.append(self.prepare)

    # -- worker ---------------------------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._work, name="oms-derived", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout)
            self._thread = None

    def committed(self, name: str, n: int) -> None:
        """Version ``n`` is published: the one ``prepare`` derived is the map's latest; the
        worker derives it if preparation did not."""
        with self._lock:
            dv = self._prepared.pop((name, n), None)
        if dv is not None:
            meta = self.store.meta(name) or {}
            self._keep((name, str(meta.get("created_at")), n), dv, latest=True)
        self._queue.put((name, n))

    def forget(self, name: str, n: int) -> None:
        """Version ``n`` was deleted by retention: out of memory too."""
        with self._lock:
            for key in [k for k in self._cache if k[0] == name and k[2] == n]:
                del self._cache[key]
            self._version_locks.pop((name, n), None)

    def prepare(self, name: str, n: int, record: dict[str, Any], mapper: Path,
                cameras: dict[str, Any], files: dict[str, Any] | None = None) -> None:
        """Version ``n`` of map ``name`` derived before it is published, from its ``record``,
        the ``mapper`` map it is committed with, the map's ``cameras`` and its committed JSON
        ``files`` (what a version derived earlier under that number, never committed, held is
        replaced): its files, its locate index and the ``prepared`` hooks' answers."""
        vdir = self.dir(name) / str(n)
        with self._version_lock(name, n):
            shutil.rmtree(vdir, ignore_errors=True)
            internal = None
            if mapper.exists():
                try:
                    internal = self.read_map(mapper)
                except (OSError, OhMySlamError):
                    internal = None
            self._derive(name, n, record, vdir, internal, cameras, files)
            dv = DerivedVersion(name, n, record, json.loads((vdir / "meta.json").read_text()),
                                vdir, self._stands(record, cameras))
            self.index(dv, cameras)  # the locate index ready before the version is
            for hook in self.prepared:
                try:
                    hook(dv, files or {})
                except Exception:
                    log.exception("preparing %s version %d: a read answer failed", name, n)
            with self._lock:
                self._prepared[(name, n)] = dv

    def replacing(self, name: str, first: int, last: int) -> None:
        """Versions ``first``..``last`` derived from the mapper map a commit is about to
        replace."""
        for n in range(first, last + 1):
            if not (self.dir(name) / str(n) / "meta.json").exists():
                self.get(name, n)

    def _work(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                dv = self.get(*item)  # derived now if preparation did not
                self.index(dv)  # ... and its locate index (built once: ``index``)
            except Exception:
                log.exception("deriving %s version %d failed", *item)

    # -- reads ----------------------------------------------------------------------------------

    @contextlib.contextmanager
    def _version_lock(self, name: str, n: int) -> Iterator[None]:
        """The lock deriving version ``n`` of a map: a read of another version never waits."""
        with self._lock:
            lock = self._version_locks.setdefault((name, n), threading.Lock())
        with lock:
            yield

    def _keep(self, key: tuple[str, str, int], dv: DerivedVersion, latest: bool) -> None:
        """``dv`` in memory: as its map's latest (pinned until a later version is), else among
        the ``CACHE_VERSIONS`` most recent others."""
        name = key[0]
        with self._lock:
            old = self._latest.get(name)
            if latest and (old is None or old[0][1] != key[1] or old[0][2] <= key[2]):
                if old is not None and old[0] != key and old[0][1] == key[1]:
                    self._cache[old[0]] = old[1]  # no longer the latest: an ordinary entry
                self._latest[name] = (key, dv)
                self._cache.pop(key, None)
            elif old is None or old[0] != key:
                self._cache[key] = dv
                self._cache.move_to_end(key)
            while len(self._cache) > CACHE_VERSIONS:
                self._cache.popitem(last=False)

    def dir(self, name: str) -> Path:
        return self.store.dir(name) / "derived"

    def get(self, name: str, n: int) -> DerivedVersion:
        """Version ``n`` of map ``name`` derived (now, if it was not yet)."""
        meta = self.store.require(name)
        key = (name, str(meta.get("created_at")), n)
        latest = int(meta.get("latest_version") or 0) == n
        with self._lock:
            pinned = self._latest.get(name)
            if pinned is not None and pinned[0] == key:
                return pinned[1]
            dv = self._cache.get(key)
            prepared = self._prepared.get((name, n))
        if dv is None:
            record = self.store.version(name, n)
            if prepared is not None and prepared.record.get("job_id") == record.get("job_id"):
                dv = prepared  # published while its commit finishes
            else:
                dv = self._load(name, n, record)
        self._keep(key, dv, latest)
        return dv

    def _load(self, name: str, n: int, record: dict[str, Any]) -> DerivedVersion:
        vdir = self.dir(name) / str(n)
        with self._version_lock(name, n):  # held by a derivation of this version only
            if not (vdir / "meta.json").exists():
                # from the mapper map it was committed with (kept under ``mappers/``, O2);
                # ``version_gone`` when that one is gone
                path = self.store.mapper_for(name, n)
                if path == self.store.mapper_dir(name):
                    self._derive(name, n, record, vdir)
                else:
                    internal = None
                    if path is not None:
                        with contextlib.suppress(OSError, OhMySlamError):
                            internal = self.read_map(path)
                    self._derive(name, n, record, vdir, internal,
                                 record.get("cameras") or self.store.cameras(name))
        return DerivedVersion(name, n, record, json.loads((vdir / "meta.json").read_text()),
                              vdir, self._stands(record, self.store.cameras(name)))

    def _stands(self, record: dict[str, Any], cams: dict[str, Any]) -> list[navgrid.Stand]:
        out = []
        for f in record["frames"]:
            cam = cams.get(f["camera_id"])
            if cam is None:
                continue
            c = Camera.from_json(cam)
            K = c.canonical()
            h = f.get("height_above_floor_m", c.height_above_floor_m)
            out.append(navgrid.Stand(np.asarray(f["T_map_cam"], np.float64), K.K(), K.width,
                                     K.height, None if h is None else float(h)))
        return out

    # -- derivation -----------------------------------------------------------------------------

    def _read(self, name: str) -> InternalMap | None:
        """The mapper map, None when the map has none (no update placed a frame yet)."""
        try:
            return self.read_map(self.store.mapper_dir(name))
        except (OSError, OhMySlamError):
            return None

    def _base_volume(self, name: str, n: int) -> navgrid.Volume:
        for m in range(n - 1, 0, -1):
            p = self.dir(name) / str(m) / "volume.npz"
            if p.exists():
                return navgrid.Volume.from_bytes(p.read_bytes())
        return navgrid.Volume()

    def _derive(self, name: str, n: int, record: dict[str, Any], vdir: Path,
                internal: InternalMap | None = None, cameras: dict[str, Any] | None = None,
                files: dict[str, Any] | None = None) -> None:
        """Derive a version into ``vdir`` (meta.json last) from its record, its mapper map
        (``internal``; default: the map's current one), the map's cameras (default: the
        store's) and, before it is committed, its JSON ``files``."""
        sim = Sim3.from_dict(record["sim3"])
        if cameras is None:
            internal = self._read(name)
        cams = {cid: Camera.from_json(c) for cid, c in (
            self.store.cameras(name) if cameras is None else cameras).items()}
        frames = [f for f in record["frames"] if f.get("camera_id") in cams]
        vol = self._base_volume(name, n)
        by_name = {f["name"]: f for f in frames}
        stale = [k for k, T in vol.frames.items()
                 if k not in by_name or not _occupies(by_name[k])
                 or navgrid.pose_moved(T, np.asarray(
                     by_name[k]["T_map_cam"]).reshape(-1).tolist())]
        if stale:
            vol = navgrid.Volume()
        depth_of: dict[str, NDArray[np.float32] | None] = {}

        def depth(f: dict[str, Any]) -> NDArray[np.float32] | None:
            if f["name"] not in depth_of:
                d = None
                if internal is not None and f.get("internal"):
                    raw = internal.depth(f["internal"])
                    if raw is not None:
                        d = (np.asarray(raw, np.float32) * np.float32(sim.s))
                elif f.get("content") and f.get("added_version"):
                    a = int(f["added_version"])
                    d = self._kept_depth(name, a, f["name"],
                                         (files or {}).get("kept_depth.json") if a == n else None)
                depth_of[f["name"]] = d
            return depth_of[f["name"]]

        for f in frames:
            if f["name"] in vol.frames or not _occupies(f):
                # occupancy and heights come only from image-confirmed frames; a kept frame
                # counts for the free-by-rule cells alone (``_stands``, map.md §6), but for the
                # too-close exception (api.md §4: its depth in the valid range, at its prior)
                continue
            d = depth(f)
            if d is None:
                continue
            K = cams[f["camera_id"]].canonical().resized(d.shape[1], d.shape[0])
            ok = valid_depth(d) if f.get("content") else d > 0
            vol.integrate(f["name"], d, K.K(), np.asarray(f["T_map_cam"], np.float64), ok)
        # every camera stands inside the grid, even one whose depth gave nothing (all unknown)
        C: NDArray[np.float64] = np.array(
            [np.asarray(f["T_map_cam"])[:3, 3] for f in frames]) if frames else np.zeros((1, 3))
        vol.ensure(C.min(0) - np.array([0.0, 0.0, navgrid.BELOW_CAMERA_M]),
                   C.max(0) + np.array([0.0, 0.0, navgrid.ABOVE_CAMERA_M]))
        vdir.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(vdir / "volume.npz", vol.to_bytes())
        # the cloud, in the map frame
        cloud = internal.cloud() if internal is not None else None
        if cloud is not None and len(cloud.xyz):
            buf = io.BytesIO()
            np.savez(buf, xyz=sim.point(np.asarray(cloud.xyz, np.float64)).astype(np.float32),
                     rgb=cloud.rgb, object_id=cloud.object_id)
            atomic_write_bytes(vdir / "cloud.npz", buf.getvalue())
        # keyframe features and their points: triangulated with the stored poses, else depth
        kfs: list[relocalize.Keyframe] = []
        keys: dict[str, str] = {}
        for f in frames:
            if f.get("pose_state") == "kept_at_prior":
                continue  # a pose the images did not confirm: never a locate reference
            img = internal.image(f["internal"]) if internal is not None and f.get("internal") \
                else None
            if img is None or not img.exists():
                continue
            # no keypoint on its ignored pixels (map.md §2): the mask is part of the key
            ignore = internal.ignore(f["internal"]) if internal is not None else None
            if ignore is not None and not ignore.any():
                ignore = None
            key = f"{f.get('sha256') or _sha_file(img)}-{relocalize.FEATURES_TAG}"
            if ignore is not None:
                key += "-m" + hashlib.sha256(np.packbits(ignore).tobytes()).hexdigest()[:12]
            feat = self._features(name, key, img, ignore)
            if feat is None:
                continue
            uv, desc, w, h = feat
            Kf = cams[f["camera_id"]].canonical().resized(w, h).K()
            T = np.asarray(f["T_map_cam"], np.float64)
            kfs.append(relocalize.Keyframe(f["name"], T, Kf, uv, desc,
                                           relocalize.keyframe_points(uv, Kf, depth(f), T)))
            keys[f["name"]] = key

        def pair(i: int, j: int) -> NDArray[np.int64]:
            k = (keys[kfs[i].name], keys[kfs[j].name])
            with self._lock:
                hit = self._pairs.get(k)
            if hit is None:
                hit = relocalize._pair(kfs[i], kfs[j])
                with self._lock:
                    self._pairs[k] = hit
            return hit

        points = relocalize.triangulate(kfs, pair)
        buf = io.BytesIO()
        arrays: dict[str, Any] = {f"p{i}": p for i, p in enumerate(points)}
        np.savez(buf, **arrays)
        atomic_write_bytes(vdir / "locate.npz", buf.getvalue())
        # the floor: the version's one estimate (map.md §5), which its objects used, when its
        # record holds it (``floor_sigma_m``: oms/2.2 records); for an older record, the plane of
        # the volume's flat points below the cameras that the version's floor names — else the
        # lowest; else the version's; else from the camera heights
        floor_sigma = float(record.get("floor_sigma_m") or FLOOR_SIGMA_DEFAULT_M)
        floor = record.get("floor_z_m") if "floor_sigma_m" in record else None
        if floor is None and frames:
            floor = vol.estimate_floor(float(C[:, 2].min()), record.get("floor_z_m"))
        if floor is None:
            floor = record.get("floor_z_m")
        if floor is None:
            heights = [np.asarray(f["T_map_cam"])[2, 3] - cams[f["camera_id"]].height_above_floor_m
                       for f in frames if cams[f["camera_id"]].height_above_floor_m is not None]
            floor = float(np.median(heights)) if heights else (
                min(np.asarray(f["T_map_cam"])[2, 3] for f in frames) - DEFAULT_CAMERA_HEIGHT_M
                if frames else 0.0)
        objects = []
        for o in (internal.objects() if internal is not None else []):
            R = np.asarray(o.get("R") or np.eye(3), np.float64)
            size = np.asarray(o.get("size") or [0.0, 0.0, 0.0], np.float64) * sim.s
            views = [{"frame": frame_name(int(v[0])),
                      "centroid": sim.point(np.asarray(v[3:6], np.float64)).round(5).tolist(),
                      "uncut": float(v[2]) <= MAPPER_UNCUT_BORDER}
                     for v in o.get("sightings") or [] if len(v) >= 6]
            objects.append({"id": o.get("id"), "label": o["label"],
                            "centre_m": sim.point(np.asarray(o["center"], np.float64)).tolist(),
                            "size_m": size.tolist(), "yaw_deg": yaw_of(sim.R @ R),
                            **{k: o[k] for k in MAPPER_OBJECT_KEYS if k in o},
                            **({"views": views} if "sightings" in o else {})})
        atomic_write_json(vdir / "meta.json", {
            "version": n, "floor_z": float(floor), "floor_sigma": floor_sigma, "objects": objects,
            "doors": [o["centre_m"][:2] for o in objects if "door" in o["label"].lower()],
            # the door leaves a passage names (map.md §5-§6): door objects shaped as leaves
            "door_leaves": [{k: o[k] for k in ("id", "centre_m", "size_m", "yaw_deg")}
                            for o in objects if o["label"].lower() == DOOR_LABEL and o.get("id")
                            and door_leaf(o["size_m"])],
            "keyframes": [k.name for k in kfs], "keys": keys})

    def _kept_depth(self, name: str, n: int, frame: str, held: dict[str, Any] | None = None
                    ) -> NDArray[np.float32] | None:
        """A too-close kept frame's depth image on the older mapper (metres; the version file
        ``update.KEPT_DEPTH`` of the version that added it, ``held`` while that version is
        being committed), or None."""
        import base64

        import cv2

        if held is None:
            p = self.store.version_file(name, n, "kept_depth.json")
            if not p.exists():
                return None
            held = json.loads(p.read_text())
        b64 = held.get(frame)
        if not b64:
            return None
        mm = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_UNCHANGED)
        return None if mm is None else (mm.astype(np.float32) / 1000.0)

    def _features(self, name: str, key: str, img: Path, ignore: NDArray[np.bool_] | None = None
                  ) -> tuple[NDArray[np.float32], NDArray[np.uint8], int, int] | None:
        p = self.dir(name) / "features" / f"{key}.npz"
        if p.exists():
            z = np.load(p)
            return z["uv"], z["desc"], int(z["size"][0]), int(z["size"][1])
        import cv2

        gray = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None
        mask = on_grid(ignore, gray.shape)
        uv, desc = relocalize.extract(np.asarray(gray, np.uint8), mask)
        buf = io.BytesIO()
        np.savez(buf, uv=uv, desc=desc, size=np.array([gray.shape[1], gray.shape[0]]))
        atomic_write_bytes(p, buf.getvalue())
        return uv, desc, gray.shape[1], gray.shape[0]

    # -- locate index ---------------------------------------------------------------------------

    def index(self, dv: DerivedVersion, cameras: dict[str, Any] | None = None
              ) -> relocalize.Index:
        """The version's locate index, built once (the first caller builds it, the others
        wait): at commit by ``prepare``, else by the worker or the first locate."""
        with dv._index_lock:
            if dv._index is None:
                dv._index = self._build_index(dv, cameras)
            return dv._index

    def _build_index(self, dv: DerivedVersion, cameras: dict[str, Any] | None
                     ) -> relocalize.Index:
        cams = {cid: Camera.from_json(c) for cid, c in (
            self.store.cameras(dv.name) if cameras is None else cameras).items()}
        z = np.load(dv.dir / "locate.npz")
        by_name = {f["name"]: f for f in dv.record["frames"]}
        kfs = []
        for i, kname in enumerate(dv.meta["keyframes"]):
            f = by_name[kname]
            z_f = np.load(self.dir(dv.name) / "features" / f"{dv.meta['keys'][kname]}.npz")
            uv, desc = z_f["uv"], z_f["desc"]
            w, h = int(z_f["size"][0]), int(z_f["size"][1])
            kfs.append(relocalize.Keyframe(
                kname, np.asarray(f["T_map_cam"], np.float64),
                cams[f["camera_id"]].canonical().resized(w, h).K(), uv, desc, z[f"p{i}"],
                float(f.get("sigma_pos_m") or 0.0), float(f.get("sigma_rot_deg") or 0.0),
                f.get("depth_kind") == "sensor", f.get("height_sigma_m"),
                f.get("pose_state", "fused") != "kept_at_prior"))
        idx = relocalize.Index(kfs)
        idx.prepare()  # its nearest-neighbour index too: the first locate does not build it
        return idx
