"""The service's workspace (spec §2.6 "Workspace"): ``<data>/maps/<name>/`` (maps in exactly the
format ``mapper.sh`` writes), ``<data>/uploads/<id>/<file>`` (transient inputs) and
``<data>/.requests/<id>/`` (what a running request's command writes — its stdout, the result, and
its timing record — deleted once the response is sent: the service keeps no results). Every path a
request names is resolved — symlinks included — and refused when it lands outside the workspace or
goes through a hidden entry."""

from __future__ import annotations

import re
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oh_my_slam.core import constants
from oh_my_slam.core.errors import InputError, UsageError

MAPS = "maps"
UPLOADS = "uploads"
REQUESTS = ".requests"
DEFAULT_DATA = Path(constants.DEFAULT_DATA)  # server.sh --data (commands.entry_points)

_NAME = re.compile(r"^[^/\\\0]+$")


class OutsideWorkspaceError(InputError):
    """A path that resolves outside the workspace (exit-code class: input error, HTTP 400)."""


class NotFoundError(UsageError):
    """A workspace object (map, upload, operation) that does not exist (HTTP 404)."""


def plain_name(name: str, what: str) -> str:
    """``name`` if it is one plain path component (no separator, not ``.``/``..``, not hidden)."""
    if not isinstance(name, str) or not _NAME.match(name) or name.startswith("."):
        raise UsageError(f"{what} must be a plain file or folder name (no '/', not hidden), "
                         f"got {name!r}")
    return name


def inside(root: Path, rel: str) -> Path:
    """``root / rel`` resolved (symlinks followed); :class:`NotFoundError` when it leaves
    ``root`` or names a hidden entry."""
    real_root = root.resolve()
    try:
        target = (real_root / rel).resolve()
    except (OSError, ValueError) as exc:
        raise NotFoundError(f"not found: {rel}") from exc
    if target != real_root and real_root not in target.parents:
        raise NotFoundError(f"not found: {rel}")
    if any(part.startswith(".") for part in target.relative_to(real_root).parts):
        raise NotFoundError(f"not found: {rel}")
    return target


@dataclass(frozen=True)
class Upload:
    id: str
    name: str
    path: Path  # <data>/uploads/<id>/<name>
    size: int

    def describe(self, root: Path) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "size": self.size,
                "path": str(self.path.relative_to(root))}


class Workspace:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.maps = self.root / MAPS
        self.uploads = self.root / UPLOADS
        self.requests = self.root / REQUESTS

    def create(self) -> None:
        for d in (self.root, self.maps, self.uploads):
            d.mkdir(parents=True, exist_ok=True)

    # -- paths -------------------------------------------------------------------------------------

    def resolve(self, value: str) -> Path:
        """A path a request names — relative to the workspace, or absolute inside it — resolved
        with its symlinks; refused when it lands outside the workspace."""
        if not isinstance(value, str) or not value or "\0" in value:
            raise UsageError(f"expected a path inside the workspace {self.root}, got {value!r}")
        p = Path(value).expanduser()
        real = (p if p.is_absolute() else self.root / p).resolve()
        if real != self.root and self.root not in real.parents:
            raise OutsideWorkspaceError(
                f"{value} resolves outside the workspace {self.root}; upload the file or use a "
                "path inside the workspace")
        if any(part.startswith(".") for part in real.relative_to(self.root).parts):
            raise OutsideWorkspaceError(
                f"{value}: hidden entries of the workspace (an upload still arriving, a map's "
                ".staging) are not inputs")
        return real

    def map_path(self, value: str) -> Path:
        """A map named by a request: ``<name>`` or ``maps/<name>``, always ``<data>/maps/<name>``
        (maps live nowhere else in the workspace)."""
        if isinstance(value, str) and "/" not in value.rstrip("/") and not value.startswith("~"):
            value = f"{MAPS}/{value.rstrip('/')}"
        real = self.resolve(value)
        if real.parent != self.maps.resolve() or real.name.startswith("."):
            raise OutsideWorkspaceError(
                f"{value}: a map is a folder directly in {self.maps} (give its name)")
        return real

    def upload_of(self, path: Path) -> str | None:
        """The upload id a resolved input path belongs to, if any."""
        up = self.uploads.resolve()
        if up in path.parents:
            return path.relative_to(up).parts[0]
        return None

    def relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    # -- uploads -----------------------------------------------------------------------------------

    def new_upload(self, name: str) -> tuple[str, Path]:
        """A fresh upload folder and the path the file is written to."""
        plain_name(name, "the upload's file name")
        uid = secrets.token_hex(8)
        folder = self.uploads / uid
        folder.mkdir(parents=True)
        return uid, folder / name

    def upload(self, uid: str) -> Upload:
        folder = inside(self.uploads, plain_name(uid, "an upload id"))
        files = [p for p in folder.iterdir() if p.is_file() and not p.name.startswith(".")] \
            if folder.is_dir() else []
        if len(files) != 1:
            raise NotFoundError(f"no upload {uid} (an upload is deleted when the request it was "
                                "given to ends)")
        return Upload(uid, files[0].name, files[0], files[0].stat().st_size)

    def delete_upload(self, uid: str) -> None:
        shutil.rmtree(self.uploads / uid, ignore_errors=True)

    def clear_uploads(self) -> None:
        """Delete every upload (at start and stop no request can consume them any more)."""
        _clear(self.uploads)

    # -- requests ----------------------------------------------------------------------------------

    def request_dir(self, rid: str) -> Path:
        """The folder of what a running request's command writes (created when it starts)."""
        return self.requests / plain_name(rid, "a request id")

    def clear_requests(self) -> None:
        """Delete what requests left (at start and stop none runs: a crash, a lost response)."""
        _clear(self.requests)

    # -- maps (read-only) --------------------------------------------------------------------------

    def map_dir(self, name: str) -> Path:
        d = inside(self.maps, plain_name(name, "a map name"))
        from oh_my_slam.mapping.store import classify

        if d.parent != self.maps.resolve() or classify(d) != "map":
            raise NotFoundError(f"no map {name} in {self.maps}")
        return d

    def list_maps(self) -> list[dict[str, Any]]:
        """Every map in ``<data>/maps`` with its summary (from its own metadata)."""
        from oh_my_slam.core.errors import OhMySlamError

        out = []
        for d in sorted(self.maps.iterdir()) if self.maps.is_dir() else []:
            if d.name.startswith(".") or not d.is_dir():
                continue
            try:
                out.append(self.map_summary(d.name))
            except (OhMySlamError, OSError, ValueError, KeyError):
                continue  # not a map (or not yet one)
        return out

    def map_summary(self, name: str, full: bool = False) -> dict[str, Any]:
        """A map's summary figures, read from its own metadata (``map.json``, the frame and object
        records) through the read-only map store; ``full`` adds the whole ``map.json``."""
        from oh_my_slam.mapping import store

        reader = store.MapReader(self.map_dir(name))
        meta = reader.meta
        summary: dict[str, Any] = {"name": name, "path": f"{MAPS}/{name}"}
        for k, v in meta.items():
            if isinstance(v, str | int | float | bool) or v is None:
                summary[k] = v
            else:  # the only other JSON values: a list or an object
                summary[f"{k}_count"] = len(v)
        summary["frames"] = len(reader.frames)
        if reader.exists(store.OBJECTS_JSON):
            objs = reader.read_json(store.OBJECTS_JSON)
            listed = objs.get("objects") if isinstance(objs, dict) else objs
            summary["objects"] = len(listed) if isinstance(listed, list | dict) else None
        updates = meta.get("updates") or []
        if updates:
            last = updates[-1]
            summary["last_update"] = {k: last.get(k) for k in ("id", "at", "kind")} | {
                "frames_added": len(last.get("frames_added") or []),
                "total_s": (last.get("timings") or {}).get("total_s")}
        if full:
            summary["meta"] = meta
        return summary


def _clear(folder: Path) -> None:
    """Delete everything in ``folder`` (the folder stays)."""
    if folder.is_dir():
        for d in folder.iterdir():
            if d.is_dir() and not d.is_symlink():
                shutil.rmtree(d, ignore_errors=True)
            else:
                d.unlink(missing_ok=True)
