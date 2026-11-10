"""Maps on disk: cameras, versions and the mapper's own map.

Under ``<data>/oms/maps/<name>/``:

* ``map.json`` — the cameras (``Camera`` bodies as sent), their ignore masks' digests, the
  creation time and the latest version;
* ``masks/<camera_id>.png`` — ignore masks;
* ``versions/<n>.json`` — version *n*, immutable once written: its frames with their map-frame
  poses and sigmas, the similarity from the mapper's internal frame (``anchor.Sim3``), the scale
  and its source, and the counts a summary reads;
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
        return out

    def cameras(self, name: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = (self.meta(name) or {}).get("cameras", {})
        return out

    def mask_digest(self, name: str, camera_id: str) -> str | None:
        out: str | None = (self.meta(name) or {}).get("masks", {}).get(camera_id)
        return out

    def mask_path(self, name: str, camera_id: str) -> Path:
        return self.dir(name) / "masks" / f"{camera_id}.png"

    def add_cameras(self, name: str, cameras: dict[str, dict[str, Any]],
                    masks: dict[str, tuple[str, bytes]]) -> None:
        """Record new cameras of a map (creating its folder), with their ignore masks
        (``camera_id -> (digest, PNG bytes)``)."""
        with self._lock:
            meta = self.meta(name) or {"name": name, "cameras": {}, "masks": {},
                                        "latest_version": None, "created_at": None}
            for cid, (digest, data) in masks.items():
                atomic_write_bytes(self.mask_path(name, cid), data)
                meta["masks"][cid] = digest
            meta["cameras"].update(cameras)
            self._write_meta(name, meta)

    def mapper_dir(self, name: str) -> Path:
        return self.dir(name) / "mapper"

    def commit(self, name: str, record: dict[str, Any], mapper: Path | None) -> int:
        """Commit a new version: ``record`` (its ``version`` and ``created_at`` set here) and
        the mapper map ``mapper`` moved into place. Returns the version."""
        with self._lock:
            meta = self.meta(name)
            assert meta is not None, "cameras are recorded before a job runs"
            n = int(meta.get("latest_version") or 0) + 1
            now = time.time()
            record = {**record, "version": n, "created_at": utc(now)}
            if mapper is not None:
                current = self.mapper_dir(name)
                if current.exists():
                    current.rename(self.trash / f"{name}-{uuid.uuid4().hex}")
                mapper.rename(current)
            atomic_write_json(self.dir(name) / "versions" / f"{n}.json", record)
            meta["latest_version"] = n
            meta["created_at"] = meta.get("created_at") or utc(now)
            self._write_meta(name, meta)
        shutil.rmtree(self.trash, ignore_errors=True)
        self.trash.mkdir(parents=True, exist_ok=True)
        self.committed(name, n)
        return n

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
