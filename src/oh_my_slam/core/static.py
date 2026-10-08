"""The files of a page's ``static`` folder, served confined to it: by ``view.sh`` (``viewer.routes``)
and by ``server.sh`` (``web.app``, which also serves the viewer's modules and libraries)."""

from __future__ import annotations

import mimetypes
from pathlib import Path

# The media type of what the pages load, by suffix; any other is guessed from the file's name.
TYPES = {".js": "text/javascript", ".css": "text/css", ".html": "text/html",
         ".json": "application/json", ".png": "image/png", ".svg": "image/svg+xml",
         ".txt": "text/plain", ".md": "text/plain"}


def find_static(root: Path, rel: str) -> tuple[Path, str] | None:
    """The file ``rel`` of the folder ``root``, its symlinks resolved, and its media type; None
    for anything that is no file inside ``root`` (``..``, an absolute path, a symlink that leads
    out, an embedded NUL byte, a folder)."""
    try:
        target = (root / rel).resolve()
    except (ValueError, OSError):  # e.g. an embedded NUL byte
        return None
    if root.resolve() not in target.parents or not target.is_file():
        return None
    media = TYPES.get(target.suffix) or mimetypes.guess_type(target.name)[0]
    return target, media or "application/octet-stream"
