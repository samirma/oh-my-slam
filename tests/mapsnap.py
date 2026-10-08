"""Read-only checks for map folders: a snapshot of every entry, hidden ones included, and a hash
of the visible files.

``full_tree_hash`` skips hidden entries (``.lock``, ``.staging``), so a reader that rolled a
committed update forward, or discarded a staging folder, would pass it unnoticed; ``snapshot``
compares kind, size, mtime and content of every entry instead."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from oh_my_slam.mapping import store

Snapshot = dict[str, tuple[bool, int, int, str]]


def snapshot(root: Path) -> Snapshot:
    """Every entry under ``root``, hidden ones included: kind, size, mtime and content hash."""
    out = {}
    for p in sorted(Path(root).rglob("*")):
        st = p.stat()
        digest = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else ""
        out[str(p.relative_to(root))] = (p.is_dir(), st.st_size, st.st_mtime_ns, digest)
    return out


def full_tree_hash(root: Path) -> str:
    """Hash of every visible file (paths + contents); hidden entries are ignored."""
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


def with_committed_overlay(src: Path, dst: Path) -> Path:
    """A copy of the map ``src`` at ``dst`` left as an update killed right after its commit point:
    ``.staging/`` holds a committed copy of ``map.json`` (and its ``COMMIT`` marker) that the next
    update would roll forward. Read-only readers see it through the overlay and must leave it."""
    shutil.copytree(src, dst)
    staging = dst / store.STAGING
    staging.mkdir()
    shutil.copy2(dst / store.MAP_JSON, staging / store.MAP_JSON)
    (staging / store.COMMIT).write_text(json.dumps({"files": [store.MAP_JSON], "delete": []}))
    assert store.MapReader(dst)._overlay == {store.MAP_JSON}
    return dst
