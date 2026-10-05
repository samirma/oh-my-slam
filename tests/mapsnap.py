"""Read-only checks for map folders: a snapshot of every entry, hidden ones included.

``store.full_tree_hash`` skips hidden entries (``.lock``, ``.staging``), so a reader that rolled a
committed update forward, or discarded a staging folder, would pass it unnoticed. These helpers
compare kind, size, mtime and content of every entry instead."""

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
