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
  frame. An update works on a clone of it and moves the clone into place when it commits;
* ``mappers/<k>/`` — the mapper maps a commit replaced (O2, map.md §2), each named by the first
  version it served: it serves the versions from ``k`` to the next one's first, a revert clones
  it back. Kept while a kept version uses it; clones share their unchanged files (APFS), so a
  version costs only what changed.

Retention (``--keep-versions``, map.md §2): after each commit the versions older than the last
``keep`` are deleted (their record, their own files, their derived reads), except a held one
(``hold``: a job holds its base version, a read the version it answers from, until it ends; a held
version does not count toward ``keep`` and goes at the first commit after its last hold ended)
and the files a kept version still reads (``kept_depth.json`` of the version that added a kept
frame).

Commits of one map are serialised (``_commit_lock``: a job's, a revert's), so two reverts never
take the same version number; reads never take that lock.

Every file is written atomically; a version is written before ``map.json`` names it, so a read
always sees a complete version."""

from __future__ import annotations

import contextlib
import json
import re
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from oh_my_slam.core.atomic import atomic_write_bytes, atomic_write_json
from oh_my_slam.service import object_ids
from oh_my_slam.service.errors import ApiError, invalid

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
ROOT = "oms"  # under --data, beside what the earlier tools keep there
# a version record's ``pose_states``: its frames' ``pose_state`` follows oms/2.2 (a frame kept at
# its prior is ``kept_at_prior``). Earlier records named such frames ``provisional``, before the
# O3 two-phase state existed: they are read as ``kept_at_prior`` (``version``)
POSE_STATES = "oms/2.2"
# a version frame row's key: a frame kept at a prior its images contradict (``detail``
# ``conflict_unsupported``, whatever its reason) makes no camera stand and no near blind disc
# (map.md §6 "Unconfirmed frames", P30b/P30c; ``derived``) until a later job confirms it
NO_STAND = "no_stand"


def no_stand(conflict: bool) -> dict[str, bool]:
    """A version frame row's ``NO_STAND`` entry: present only on such a frame."""
    return {NO_STAND: True} if conflict else {}


# a version frame row's key: a keyframe the floor check did not judge (``floor_not_judged``),
# whose depth lifts no later frame, in registration or in locate (api.md §6, P30d; ``derived``)
DEPTH_UNUSED = "depth_unused"


def depth_unused(unjudged: bool) -> dict[str, bool]:
    """A version frame row's ``DEPTH_UNUSED`` entry: present only on such a frame."""
    return {DEPTH_UNUSED: True} if unjudged else {}


def utc(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int(t % 1 * 1000):03d}Z"


def check_name(name: str) -> str:
    if not NAME.match(name):
        raise invalid("name", f"map name {name!r}: 1-64 characters of letters, digits, '.', '_' "
                      "and '-', starting with a letter or digit")
    return name


def latest_plus(meta: dict[str, Any]) -> int:
    """One past a map's latest version."""
    return int(meta.get("latest_version") or 0) + 1


def clone_tree(src: Path, dst: Path) -> None:
    """Copy a folder as APFS clones (``cp -c``: instant, no extra space), else a plain copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if subprocess.run(["cp", "-c", "-R", str(src), str(dst)], capture_output=True).returncode:
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst, symlinks=True)


DEFAULT_KEEP_VERSIONS = 20  # versions kept per map (``--keep-versions``, README)
KEPT_DEPTH = "kept_depth.json"  # a version file later versions read (``update.KEPT_DEPTH``)


def _info(record: dict[str, Any]) -> dict[str, Any]:
    """A version's ``VersionInfo`` (api.md §5) from its record."""
    return {"version": record["version"], "created_at": record["created_at"],
            "kind": record.get("kind") or "update", "job_id": record.get("job_id"),
            "reverted_to": record.get("reverted_to"), "frames": len(record.get("frames", [])),
            "objects": int(record.get("objects") or 0)}


class Store:
    def __init__(self, data: Path, keep_versions: int = DEFAULT_KEEP_VERSIONS) -> None:
        self.keep_versions = max(int(keep_versions), 1)
        # (map, version) -> reads using it now: retention does not delete it (``hold``)
        self._holds: dict[tuple[str, int], int] = {}
        # called with (map, version) once retention deleted a version (``derived`` forgets it)
        self.pruned: list[Callable[[str, int], None]] = []
        self.root = Path(data).expanduser() / ROOT
        self.maps = self.root / "maps"
        self.jobs = self.root / "jobs"
        self.work = self.root / "work"
        self.trash = self.root / "trash"
        self._lock = threading.RLock()
        self._commit_locks: dict[str, threading.RLock] = {}  # per map (``_commit_lock``)
        # called with (map, version) once a version is committed (``derived`` derives the grid,
        # points and locate inputs from it); a listener's failure never fails the commit
        self.listeners: list[Callable[[str, int], None]] = []
        # called with (map, first, last) before a commit replaces the mapper map that versions
        # first..last were committed with (``derived`` derives them from it first)
        self.replacing: list[Callable[[str, int, int], None]] = []
        # called with (map, version, record, mapper map, cameras, JSON files) before a version is
        # published
        # (``derived`` derives it there, so a read never derives: high_level_spec.md §8); a
        # listener's failure never fails the commit (the read then derives it)
        self.preparing: list[Callable[[str, int, dict[str, Any], Path, dict[str, Any],
                                       dict[str, Any]], None]] \
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

    def snapshot(self, name: str, dest: Path, holds: contextlib.ExitStack | None = None
                 ) -> dict[str, Any] | None:
        """The latest committed version of a map (None before any) and its mapper map cloned
        to ``dest``, both read under the commit lock: one version, even while a job commits
        the next (a dry run reads it while a job runs). ``holds`` holds that version until it
        closes (map.md §2 Holds)."""
        with self._lock:
            n = self.latest(name)
            if not n:
                return None
            if holds is not None:
                holds.enter_context(self.hold(name, n))
            v = self.version(name, n)
            if self.mapper_dir(name).exists():
                clone_tree(self.mapper_dir(name), dest)
            return v

    @contextlib.contextmanager
    def _commit_lock(self, name: str) -> Iterator[None]:
        """The lock serialising the commits of one map (a job's, a revert's)."""
        with self._lock:
            lock = self._commit_locks.setdefault(name, threading.RLock())
        with lock:
            yield

    def commit(self, name: str, record: dict[str, Any], mapper: Path | None,
               files: dict[str, Any] | None = None,
               cameras: dict[str, dict[str, Any]] | None = None,
               masks: dict[str, tuple[str, bytes]] | None = None,
               no_mapper: bool = False) -> int:
        """Commit a new version (``_commit``), serialised with the map's other commits."""
        with self._commit_lock(name):
            return self._commit(name, record, mapper, files, cameras, masks, no_mapper)

    def _commit(self, name: str, record: dict[str, Any], mapper: Path | None,
                files: dict[str, Any] | None, cameras: dict[str, dict[str, Any]] | None,
                masks: dict[str, tuple[str, bytes]] | None, no_mapper: bool) -> int:
        """Commit a new version: ``record`` (its ``version``, ``created_at`` and ``cameras``
        set here), the version's JSON ``files`` (``versions/<n>.<file name>``: the objects of an
        anchored map), the job's new ``cameras`` and their ignore ``masks`` (creating the map)
        and the mapper map ``mapper`` moved into place. Returns the version.

        Before the mapper map is replaced, ``replacing`` listeners get the versions the current
        one serves (``mapper_version`` to the latest): what is derived from the mapper map of a
        version is derived from that version's, never from a later one. The replaced map is
        kept under ``mappers/`` (a revert clones it back). A version's ``cameras`` are the map's
        camera registry at its commit, a revert's too (map.md §2 Revert: camera ids are never
        freed). When a revert's target had no mapper map, ``no_mapper``: the version then has
        none."""
        if mapper is not None or no_mapper:
            self._replacing(name)
        self._prepare(name, record, mapper if not no_mapper else self.work / "no-mapper",
                      cameras or {}, files or {})
        with self._lock:
            meta = self.add_cameras(name, cameras or {}, masks or {})
            n = int(meta.get("latest_version") or 0) + 1
            now = time.time()
            record = {**record, "version": n, "created_at": utc(now),
                      "cameras": dict(meta["cameras"])}
            if "mappers" not in meta:  # the history starts (an earlier map: from its latest)
                known = self.mapper_dir(name).exists()
                first = (self.mapper_version(name) or n - 1) if known else 0
                meta["mappers"] = [first] if known else []
                meta["mappers_since"] = first if known else 1
            if mapper is not None or no_mapper:
                current = self.mapper_dir(name)
                if current.exists():
                    dest = self.dir(name) / "mappers" / str(self.mapper_version(name) or n - 1)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    current.rename(dest)  # one map per first version
                if mapper is not None:
                    mapper.rename(current)
                meta["mapper_version"] = n
                # the first versions of the mapper maps in order; negative: from it, none
                meta["mappers"] = [*meta["mappers"], n if mapper is not None else -n]
            for fname, data in (files or {}).items():
                atomic_write_json(self.version_file(name, n, fname), data)
            atomic_write_json(self.dir(name) / "versions" / f"{n}.json", record)
            meta["latest_version"] = n
            meta["created_at"] = meta.get("created_at") or utc(now)
            meta["index"] = {**meta.get("index", {}), str(n): _info(record)}
            self._write_meta(name, meta)
        shutil.rmtree(self.trash, ignore_errors=True)
        self.trash.mkdir(parents=True, exist_ok=True)
        self.committed(name, n)
        self.prune(name)
        return n

    # -- versions (O2) -------------------------------------------------------------------------

    def kept(self, name: str) -> list[int]:
        """The kept versions of a map, ascending."""
        d = self.dir(name) / "versions"
        return sorted(int(p.stem) for p in d.glob("*.json") if p.stem.isdigit())

    def versions(self, name: str) -> list[dict[str, Any]]:
        """The kept versions' ``VersionInfo`` (api.md §5), ascending."""
        meta = self.require(name)
        index = meta.get("index", {})
        out = []
        for n in self.kept(name):
            info = index.get(str(n))
            if info is None:  # a version committed before the index was kept
                try:
                    info = _info(self.version(name, n))
                except ApiError:
                    continue  # deleted meanwhile
            out.append(info)
        return out

    def mapper_for(self, name: str, n: int) -> Path | None:
        """The mapper map version ``n`` was committed with (None: it had none);
        ``version_gone`` when it is no longer kept (replaced before versions kept theirs)."""
        meta = self.require(name)
        gone = ApiError("version_gone", f"version {n} of map {name!r} is no longer kept: its "
                        "mapper map is gone", {"map": name, "version": n,
                                               "latest_version": meta["latest_version"]})
        if "mappers" not in meta:  # no commit kept the history yet: the current map only
            current = self.mapper_version(name)
            if not self.mapper_dir(name).exists():
                return None
            if current > n:  # 0: unknown, as before (the current map)
                raise gone
            return self.mapper_dir(name)
        if n < int(meta.get("mappers_since") or 1):
            raise gone
        firsts = [k for k in meta["mappers"] if abs(k) <= n]
        k = max(firsts, key=abs) if firsts else -1
        if k < 0:
            return None
        path = self.mapper_dir(name) if k == self.mapper_version(name) and k == max(
            abs(j) for j in meta["mappers"]) else self.dir(name) / "mappers" / str(k)
        if not path.exists():
            raise gone
        return path

    @contextlib.contextmanager
    def hold(self, name: str, n: int) -> Iterator[None]:
        """Version ``n`` of a map is used by a read or a job: retention keeps it until it ends
        (map.md §2 Holds)."""
        with self._lock:
            self._holds[(name, n)] = self._holds.get((name, n), 0) + 1
        try:
            yield
        finally:
            with self._lock:
                left = self._holds.pop((name, n)) - 1
                if left:
                    self._holds[(name, n)] = left

    @contextlib.contextmanager
    def hold_latest(self, name: str, required: bool = False) -> Iterator[int | None]:
        """The latest version of a map (None before any; ``required``: ``map_not_found``), held
        as ``hold`` does: reading it and taking the hold are one step, so no commit prunes it in
        between (``--keep-versions 1``)."""
        with self._lock, contextlib.ExitStack() as stack:
            n = self.latest(name)
            if not n and required:
                self.require(name)
            if n:
                stack.enter_context(self.hold(name, n))
            held = stack.pop_all()
        with held:
            yield n

    def revert(self, name: str, to_version: int) -> int:
        """Commit a new version whose content equals ``to_version`` (map.md §2 Revert): its
        record (frames and poses, the similarity, scale, floor, objects, label vocabulary), its
        own files and a clone of the mapper map it was committed with, so updates continue from
        it in the same map frame (never re-gauged). The camera registry is not rolled back, and
        the clone numbers new objects past every id the map has given (``object_ids``). The
        derived reads are the target's (``Derived.prepare``). The caller keeps jobs of the map
        from starting meanwhile (``Jobs.holding``); two reverts of a map are serialised."""
        with self._commit_lock(name):
            return self._revert(name, to_version)

    def _revert(self, name: str, to_version: int) -> int:
        meta = self.require(name)
        latest = int(meta["latest_version"])
        if to_version < 1 or to_version > latest:
            raise invalid("to_version", f"version {to_version} of map {name!r} does not exist "
                          f"(latest is {latest})", latest_version=latest)
        with self.hold(name, to_version):
            try:
                src = self.version(name, to_version)
            except ApiError as exc:
                raise ApiError(exc.code, exc.message, {**exc.details, "field": "to_version"}
                               ) from exc
            source = self.mapper_for(name, to_version)
            # every object and surface id the map has given: the current map's counts, the
            # target's, and the marks of an earlier revert (the ids of the versions it reverted
            # away)
            current = self.mapper_dir(name)
            mark = self.id_marks(name).max(
                object_ids.next_ids(current if current.exists() else None)).max(
                object_ids.next_ids(source))
            with self._lock:
                fresh = self.meta(name) or meta
                fresh["next_object_id"], fresh["next_surface_id"] = mark.objects, mark.surfaces
                self._write_meta(name, fresh)
            mapper = None
            if source is not None:
                mapper = self.work / f"revert-{uuid.uuid4().hex}"
                clone_tree(source, mapper)
                object_ids.reserve(mapper, mark)
            files = {}
            prefix = f"{to_version}."
            for p in sorted((self.dir(name) / "versions").glob(f"{to_version}.*.json")):
                fname = p.name[len(prefix):]
                if fname != KEPT_DEPTH:  # read by the frames' added_version, not copied
                    files[fname] = json.loads(p.read_text())
            record = {k: v for k, v in src.items()
                      if k not in ("version", "created_at", "cameras")}
            record.update(kind="revert", job_id=None, reverted_to=to_version)
            try:
                return self._commit(name, record, mapper, files, None, None,
                                    no_mapper=source is None)
            finally:
                if mapper is not None:
                    shutil.rmtree(mapper, ignore_errors=True)

    def id_marks(self, name: str) -> object_ids.Ids:
        """The object and surface id high-water marks a revert recorded (map.md §2 Revert)."""
        meta = self.meta(name) or {}
        return object_ids.Ids(int(meta.get("next_object_id") or 1),
                              int(meta.get("next_surface_id") or 1))

    def seed_ids(self, name: str, mapper: Path, anchored: bool) -> bool:
        """Before a job builds a new mapper map in ``mapper`` for a map that has versions but
        no stored mapper map (a revert to a version from before mapper maps were kept): seed it
        with the recorded high-water marks, so ids never repeat (map.md §2 Revert). False when
        there is nothing to seed (no mark recorded)."""
        marks = self.id_marks(name)
        if marks == object_ids.Ids():
            return False
        object_ids.seed(mapper, marks, anchored)
        return True

    def prune(self, name: str) -> None:
        """Retention (map.md §2): delete the versions older than the last ``keep_versions``,
        except those a read holds, and the mapper maps no kept version uses."""
        gone = []
        with self._lock:
            meta = self.meta(name)
            if meta is None or not meta.get("latest_version"):
                return
            kept = self.kept(name)
            last = kept[-self.keep_versions:]
            old = [n for n in kept if n not in last and (name, n) not in self._holds]
            if not old:
                return
            stay = [n for n in kept if n not in old]
            records = {}
            for n in stay:
                with contextlib.suppress(ApiError, OSError, ValueError):
                    records[n] = self.version(name, n)
            # a kept version's frames read the kept depth of the version that added them
            readers = {int(f["added_version"]) for r in records.values()
                       for f in r.get("frames", []) if f.get("content") and f.get("added_version")}
            vdir = self.dir(name) / "versions"
            for n in old:
                (vdir / f"{n}.json").unlink(missing_ok=True)
                for p in vdir.glob(f"{n}.*.json"):
                    if not (p.name == f"{n}.{KEPT_DEPTH}" and n in readers):
                        p.unlink(missing_ok=True)
                d = self.dir(name) / "derived" / str(n)
                if d.exists():
                    d.rename(self.trash / f"{name}-{n}-{uuid.uuid4().hex}")
                gone.append(n)
            meta["index"] = {k: v for k, v in meta.get("index", {}).items()
                             if int(k) not in old}
            # the mapper maps no kept version uses: their range holds no kept version
            firsts = sorted(abs(k) for k in meta.get("mappers", []))
            for i, k in enumerate(firsts):
                end = firsts[i + 1] if i + 1 < len(firsts) else latest_plus(meta)
                path = self.dir(name) / "mappers" / str(k)
                if path.exists() and not any(k <= n < end for n in stay):
                    path.rename(self.trash / f"{name}-m{k}-{uuid.uuid4().hex}")
            self._write_meta(name, meta)
        shutil.rmtree(self.trash, ignore_errors=True)
        self.trash.mkdir(parents=True, exist_ok=True)
        for n in gone:
            for listener in self.pruned:
                try:
                    listener(name, n)
                except Exception:
                    from oh_my_slam.core.log import get_logger

                    get_logger("oh_my_slam.service.store").exception(
                        "pruned listener failed for %s version %d", name, n)

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
                 cameras: dict[str, dict[str, Any]], files: dict[str, Any]) -> None:
        """The ``preparing`` listeners on the version about to be committed: its number (the
        map's commits are serialised), its record as it will read (``version``, ``cameras``),
        the mapper map it is committed with (the current one when none), the map's cameras and
        the version's JSON files."""
        n = int((self.meta(name) or {}).get("latest_version") or 0) + 1
        cams = {**self.cameras(name), **cameras}
        full = {**record, "version": n, "cameras": cams}
        for listener in self.preparing:
            try:
                listener(name, n, full, mapper or self.mapper_dir(name), cams, files)
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
