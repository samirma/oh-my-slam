"""Maps on disk: cameras, versions and the mapper's own map.

Under ``<data>/oms/maps/<name>/``:

* ``map.json`` — the cameras (``Camera`` bodies as sent) of its committed versions, their ignore
  masks' digests, the creation time and the latest version; a job's new cameras are recorded
  with the version it commits, so a rejected or failed job leaves none;
* ``masks/<camera_id>.png`` — ignore masks;
* ``versions/<n>.json`` — version *n*, immutable once written: its frames with their map-frame
  poses and sigmas, the similarity from the mapper's internal frame (``anchor.Sim3``), the scale
  and its source, the cameras known at that version, and the counts a summary reads;
* ``mapper/`` — the mapper's map of the latest version (``mapping.store``), in its internal
  frame. An update works on a clone of it and moves the clone into place when it commits.

Every file is written atomically; a version is written before ``map.json`` names it, so a read
always sees a complete version."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from oh_my_slam.core.atomic import atomic_write_bytes, atomic_write_json
from oh_my_slam.service.errors import ApiError, invalid

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
ROOT = "oms"  # under --data, beside what the earlier tools keep there
# a version record's ``pose_states``: its frames' ``pose_state`` follows oms/2.2 (a frame kept at
# its prior is ``kept_at_prior``). Earlier records named such frames ``provisional``, before the
# O3 two-phase state existed: they are read as ``kept_at_prior`` (``version``)
POSE_STATES = "oms/2.2"


def utc(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int(t % 1 * 1000):03d}Z"


def check_name(name: str) -> str:
    if not NAME.match(name):
        raise invalid("name", f"map name {name!r}: 1-64 characters of letters, digits, '.', '_' "
                      "and '-', starting with a letter or digit")
    return name


def clone_tree(src: Path, dst: Path) -> None:
    """Copy a folder as APFS clones (``cp -c``: instant, no extra space), else a plain copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if subprocess.run(["cp", "-c", "-R", str(src), str(dst)], capture_output=True).returncode:
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst, symlinks=True)


class Store:
    def __init__(self, data: Path) -> None:
        self.root = Path(data).expanduser() / ROOT
        self.maps = self.root / "maps"
        self.jobs = self.root / "jobs"
        self.work = self.root / "work"
        self.trash = self.root / "trash"
        self._lock = threading.RLock()
        # called with (map, version) once a version is committed (``derived`` derives the grid,
        # points and locate inputs from it); a listener's failure never fails the commit
        self.listeners: list[Callable[[str, int], None]] = []
        # called with (map, first, last) before a commit replaces the mapper map that versions
        # first..last were committed with (``derived`` derives them from it first)
        self.replacing: list[Callable[[str, int, int], None]] = []
        # called with (map, version, record, mapper map, cameras) before a version is published
        # (``derived`` derives it there, so a read never derives: high_level_spec.md §8); a
        # listener's failure never fails the commit (the read then derives it)
        self.preparing: list[Callable[[str, int, dict[str, Any], Path, dict[str, Any]], None]] \
            = []

    def create(self) -> None:
        for d in (self.maps, self.jobs, self.work, self.trash):
            d.mkdir(parents=True, exist_ok=True)

    def clear_scratch(self) -> None:
        """What a stopped service left in ``work`` and ``trash``."""
        for d in (self.work, self.trash):
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)

    # -- maps ----------------------------------------------------------------------------------

    def dir(self, name: str) -> Path:
        return self.maps / check_name(name)

    def meta(self, name: str) -> dict[str, Any] | None:
        try:
            out: dict[str, Any] = json.loads((self.dir(name) / "map.json").read_text())
            return out
        except FileNotFoundError:
            return None

    def _write_meta(self, name: str, meta: dict[str, Any]) -> None:
        atomic_write_json(self.dir(name) / "map.json", meta)

    def latest(self, name: str) -> int | None:
        meta = self.meta(name)
        return None if meta is None else meta.get("latest_version")

    def require(self, name: str) -> dict[str, Any]:
        """The map's metadata; ``map_not_found`` unless a version was committed."""
        meta = self.meta(name)
        if meta is None or not meta.get("latest_version"):
            raise ApiError("map_not_found", f"no map named {name!r}: an update creates it "
                           f"(POST /api/maps/{name}/updates)", {"map": name})
        return meta

    def version(self, name: str, n: int | None = None) -> dict[str, Any]:
        """Version ``n`` (default: the latest committed) of a map."""
        meta = self.require(name)
        latest = int(meta["latest_version"])
        if n is None:
            n = latest
        if n < 1 or n > latest:
            raise invalid("version", f"version {n} of map {name!r} does not exist (latest is "
                          f"{latest})", latest_version=latest)
        try:
            out: dict[str, Any] = json.loads((self.dir(name) / "versions" / f"{n}.json")
                                             .read_text())
        except FileNotFoundError as exc:
            raise ApiError("version_gone", f"version {n} of map {name!r} is no longer kept",
                           {"map": name, "version": n, "latest_version": latest}) from exc
        if out.get("pose_states") != POSE_STATES:
            for r in out.get("frames", []):
                if r.get("pose_state") == "provisional":
                    r["pose_state"] = "kept_at_prior"
        return out

    def version_file(self, name: str, n: int, fname: str) -> Path:
        """A file committed with version ``n`` (``commit``'s ``files``)."""
        return self.dir(name) / "versions" / f"{n}.{fname}"

    def cameras(self, name: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = (self.meta(name) or {}).get("cameras", {})
        return out

    def mask_digest(self, name: str, camera_id: str) -> str | None:
        out: str | None = (self.meta(name) or {}).get("masks", {}).get(camera_id)
        return out

    def mask_path(self, name: str, camera_id: str) -> Path:
        return self.dir(name) / "masks" / f"{camera_id}.png"

    def add_cameras(self, name: str, cameras: dict[str, dict[str, Any]],
                    masks: dict[str, tuple[str, bytes]]) -> dict[str, Any]:
        """Record new cameras of a map (creating its folder), with their ignore masks
        (``camera_id -> (digest, PNG bytes)``); returns the map's metadata."""
        with self._lock:
            meta = self.meta(name) or {"name": name, "cameras": {}, "masks": {},
                                        "latest_version": None, "created_at": None}
            for cid, (digest, data) in masks.items():
                atomic_write_bytes(self.mask_path(name, cid), data)
                meta["masks"][cid] = digest
            meta["cameras"].update(cameras)
            self._write_meta(name, meta)
            return meta

    def mapper_dir(self, name: str) -> Path:
        return self.dir(name) / "mapper"

    def snapshot(self, name: str, dest: Path) -> dict[str, Any] | None:
        """The latest committed version of a map (None before any) and its mapper map cloned
        to ``dest``, both read under the commit lock: one version, even while a job commits
        the next (a dry run reads it while a job runs)."""
        with self._lock:
            if not self.latest(name):
                return None
            v = self.version(name)
            if self.mapper_dir(name).exists():
                clone_tree(self.mapper_dir(name), dest)
            return v

    def commit(self, name: str, record: dict[str, Any], mapper: Path | None,
               files: dict[str, Any] | None = None,
               cameras: dict[str, dict[str, Any]] | None = None,
               masks: dict[str, tuple[str, bytes]] | None = None) -> int:
        """Commit a new version: ``record`` (its ``version``, ``created_at`` and ``cameras``
        set here), the version's JSON ``files`` (``versions/<n>.<file name>``: the objects of an
        anchored map), the job's new ``cameras`` and their ignore ``masks`` (creating the map)
        and the mapper map ``mapper`` moved into place. Returns the version.

        Before the mapper map is replaced, ``replacing`` listeners get the versions the current
        one serves (``mapper_version`` to the latest): what is derived from the mapper map of a
        version is derived from that version's, never from a later one."""
        if mapper is not None:
            self._replacing(name)
        self._prepare(name, record, mapper, cameras or {})
        with self._lock:
            meta = self.add_cameras(name, cameras or {}, masks or {})
            n = int(meta.get("latest_version") or 0) + 1
            now = time.time()
            record = {**record, "version": n, "created_at": utc(now),
                      "cameras": dict(meta["cameras"])}
            if mapper is not None:
                current = self.mapper_dir(name)
                if current.exists():
                    current.rename(self.trash / f"{name}-{uuid.uuid4().hex}")
                mapper.rename(current)
                meta["mapper_version"] = n
            for fname, data in (files or {}).items():
                atomic_write_json(self.version_file(name, n, fname), data)
            atomic_write_json(self.dir(name) / "versions" / f"{n}.json", record)
            meta["latest_version"] = n
            meta["created_at"] = meta.get("created_at") or utc(now)
            self._write_meta(name, meta)
        shutil.rmtree(self.trash, ignore_errors=True)
        self.trash.mkdir(parents=True, exist_ok=True)
        self.committed(name, n)
        return n

    def mapper_version(self, name: str) -> int:
        """The first version the current mapper map serves: it holds the content of every
        version from it to the latest (a version committed without a mapper map keeps it). 0 for
        a map written before this was recorded."""
        return int((self.meta(name) or {}).get("mapper_version") or 0)

    def _replacing(self, name: str) -> None:
        latest = self.latest(name)
        if not latest:
            return
        first = self.mapper_version(name) or latest  # unknown: the latest at least
        for listener in self.replacing:
            try:
                listener(name, first, latest)
            except Exception:
                from oh_my_slam.core.log import get_logger

                get_logger("oh_my_slam.service.store").exception(
                    "replacing listener failed for %s versions %d-%d", name, first, latest)

    def _prepare(self, name: str, record: dict[str, Any], mapper: Path | None,
                 cameras: dict[str, dict[str, Any]]) -> None:
        """The ``preparing`` listeners on the version about to be committed: its number (one job
        writes a map at a time), its record as it will read (``version``, ``cameras``), the
        mapper map it is committed with (the current one when none) and the map's cameras."""
        n = int((self.meta(name) or {}).get("latest_version") or 0) + 1
        cams = {**self.cameras(name), **cameras}
        full = {**record, "version": n, "cameras": cams}
        for listener in self.preparing:
            try:
                listener(name, n, full, mapper or self.mapper_dir(name), cams)
            except Exception:
                from oh_my_slam.core.log import get_logger

                get_logger("oh_my_slam.service.store").exception(
                    "preparing listener failed for %s version %d", name, n)

    def committed(self, name: str, n: int) -> None:
        """Tell the listeners that version ``n`` of map ``name`` is committed."""
        from oh_my_slam.core.log import get_logger

        for listener in self.listeners:
            try:
                listener(name, n)
            except Exception:
                get_logger("oh_my_slam.service.store").exception(
                    "commit listener failed for %s version %d", name, n)

    def delete(self, name: str) -> bool:
        """Remove a map and its versions; False when there is none."""
        with self._lock:
            d = self.dir(name)
            if not d.exists():
                return False
            gone = self.trash / f"{name}-{uuid.uuid4().hex}"
            d.rename(gone)
        shutil.rmtree(gone, ignore_errors=True)
        return True
