"""The viewer's routes, independent of any HTTP framework: ``ViewerRoutes(bundle).handle(method,
path, query)`` returns a :class:`Response`. ``view.sh`` serves them with the standard library
(``viewer.server``). The page uses relative URLs only.

Routes (GET/HEAD only; anything else is 405):

* ``/`` — the page; ``/static/…`` — its scripts, styles and vendored libraries.
* ``/api/meta`` — JSON: mode, title, display transform, the cameras of the scene JSON (pose
  ``T``, centre ``position`` in the scene frame, intrinsics, image name), the point-cloud controls
  (from ``core.cloud_attrs``) and their defaults.
* ``/api/scene`` — the OpenLABEL scene JSON; ``/api/catalog`` — the catalogue rows (JSON) and
  ``/api/segmented.png`` — the segmented image (``view.sh -i`` only: 404 for a map).
* ``/api/cloud?key=value&…`` — the point cloud derived with those §2.2 attributes (keys not given
  keep their defaults; ``label`` and ``encoding`` concern PLY files only and are refused). Invalid
  input is a 400 with ``{"error": "<actionable message>"}``. The 200 body is one binary document
  (see :func:`cloud_document`), sent straight from the cloud's arrays (:class:`CloudDocument`):
  no copy of a cloud is ever assembled in memory.
"""

from __future__ import annotations

import json
import struct
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote

import numpy as np

from oh_my_slam.core.errors import UsageError
from oh_my_slam.core.log import get_logger
from oh_my_slam.core.static import find_static
from oh_my_slam.viewer.bundle import DisplayCloud, ViewBundle

log = get_logger("oh_my_slam.viewer")

STATIC = Path(str(resources.files("oh_my_slam.viewer") / "static"))
CLOUD_CACHE = 3  # recent clouds kept (e.g. toggling a control back and forth) …
CLOUD_CACHE_BYTES = 256_000_000  # … whose own arrays hold this many bytes (the latest one always;
# arrays shared with the source, as in a map's complete cloud, cost nothing and do not count)


@dataclass(frozen=True)
class Response:
    """One answer: status, headers (``Content-Type``, ``Content-Length``, ``Cache-Control``) and
    the body as pieces to write in order (read-only views of a cloud's arrays, not copies)."""

    status: int
    headers: tuple[tuple[str, str], ...]
    body: tuple[bytes | memoryview, ...]

    @staticmethod
    def of(status: int, ctype: str, pieces: tuple[bytes | memoryview, ...],
           size: int | None = None) -> Response:
        n = sum(memoryview(p).nbytes for p in pieces) if size is None else size
        return Response(status, (("Content-Type", ctype), ("Content-Length", str(n)),
                                 ("Cache-Control", "no-store")), pieces)

    def tobytes(self) -> bytes:
        return b"".join(self.body)


def _error(status: int, message: str) -> Response:
    return Response.of(status, "application/json", (json.dumps({"error": message}).encode(),))


def _text(status: int, text: bytes) -> Response:
    return Response.of(status, "text/plain", (text,))


@dataclass(frozen=True)
class CloudDocument:
    """One cloud document (see :func:`cloud_document`) as the pieces it is sent in: the
    length-prefixed header, then every buffer (a read-only byte view of the cloud's array, not a
    copy) and its padding."""

    pieces: tuple[bytes | memoryview, ...]
    size: int  # bytes of the whole document
    owned_bytes: int  # memory it keeps beyond the cloud source (see ``DisplayCloud.owned_bytes``)

    def tobytes(self) -> bytes:
        return b"".join(self.pieces)


def cloud_document(dc: DisplayCloud, attrs: str) -> CloudDocument:
    """The binary cloud document of ``dc``, without copying its arrays: ``uint32`` little-endian
    length ``J`` of a UTF-8 JSON header, the header (space-padded so that ``4 + J`` is a multiple
    of 4), then the buffers, each starting on a 4-byte boundary at ``4 + J + offset``. The
    header::

        {"count": n, "total": n0, "voxel": e, "attrs": "color=rgb,…",
         "buffers": [{"name", "type", "size", "offset", "bytes"}, …]}

    ``count`` points are shown out of ``total`` derived (the §2.5 display budget): one per occupied
    voxel of edge ``voxel`` metres; with ``voxel`` 0 all of them, or, above the budget, every finite
    one or one per distinct position. Buffers, little-endian,
    ``size`` components per point: ``position`` float32 x 3 (always), ``color`` uint8 x 3 (sRGB;
    absent for ``color=none``), ``label`` int32 x 1 (object id, 0 = unsegmented), ``normal``
    float32 x 3 (``normals=on``). The document depends only on the cloud and its attributes, so
    identical requests get identical bytes."""
    c = dc.cloud
    parts: list[tuple[str, str, int, Any]] = [("position", "float32", 3, c.xyz)]
    if c.rgb is not None:
        parts.append(("color", "uint8", 3, c.rgb))
    if c.label is not None:
        parts.append(("label", "int32", 1, c.label))
    if c.normals is not None:
        parts.append(("normal", "float32", 3, c.normals))
    buffers, pieces, offset = [], list[bytes | memoryview](), 0
    for name, dtype, size, arr in parts:
        data = np.ascontiguousarray(arr, dtype=np.dtype(dtype).newbyteorder("<")).reshape(-1)
        view = memoryview(data.view(np.uint8)).toreadonly()
        pad = -view.nbytes % 4
        buffers.append({"name": name, "type": dtype, "size": size, "offset": offset,
                        "bytes": view.nbytes})
        pieces += [view, b"\0" * pad] if pad else [view]
        offset += view.nbytes + pad
    header = json.dumps({"count": len(c), "total": dc.total, "voxel": dc.voxel, "attrs": attrs,
                         "buffers": buffers}).encode()
    header += b" " * (-(4 + len(header)) % 4)
    head = struct.pack("<I", len(header)) + header
    return CloudDocument((head, *pieces), len(head) + offset, dc.owned_bytes)


def parse_cloud_payload(body: bytes) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Inverse of :func:`cloud_document` (for tests and tools): the header and the arrays."""
    (n,) = struct.unpack_from("<I", body)
    header = json.loads(body[4:4 + n])
    base = 4 + n
    arrays = {}
    for b in header["buffers"]:
        dtype = np.dtype(b["type"]).newbyteorder("<")
        a = np.frombuffer(body, dtype, b["bytes"] // dtype.itemsize, base + b["offset"])
        arrays[b["name"]] = a.reshape(-1, b["size"]) if b["size"] > 1 else a
    return header, arrays


def static_file(rel: str) -> Response:
    """A file of the page's ``static`` folder (scripts, styles, vendored libraries); 404 for
    anything outside it."""
    found = find_static(STATIC, rel)
    if found is None:
        return _text(404, b"not found")
    target, ctype = found
    return Response.of(200, ctype, (target.read_bytes(),))


class ViewerRoutes:
    """The routes of one bundle. ``handle`` is thread-safe; JSON answers are built once, and the
    latest clouds are kept (``CLOUD_CACHE``, ``CLOUD_CACHE_BYTES``)."""

    def __init__(self, bundle: ViewBundle) -> None:
        self.bundle = bundle
        self._json: dict[str, Callable[[], Any]] = {
            "/api/meta": bundle.meta,
            "/api/scene": lambda: bundle.scene,
        }
        if bundle.catalog is not None:  # an image's (spec §2.5); a map has none: 404
            self._json["/api/catalog"] = lambda: bundle.catalog
        self._cache: dict[str, bytes] = {}
        self._clouds: OrderedDict[str, CloudDocument] = OrderedDict()
        self._lock = threading.Lock()

    def cloud(self, query: str) -> CloudDocument:
        """The cloud document of a ``/api/cloud`` query string. Raises :class:`UsageError` or
        ``ValueError`` (a 400)."""
        attrs = self.bundle.parse_attrs(parse_qsl(query, keep_blank_values=True))
        key = self.bundle.describe(attrs)
        with self._lock:
            if key in self._clouds:
                self._clouds.move_to_end(key)
                return self._clouds[key]
        doc = cloud_document(self.bundle.cloud(attrs), key)
        with self._lock:
            self._clouds[key] = doc
            while len(self._clouds) > 1 and (len(self._clouds) > CLOUD_CACHE or sum(
                    d.owned_bytes for d in self._clouds.values()) > CLOUD_CACHE_BYTES):
                self._clouds.popitem(last=False)
        return doc

    def handle(self, method: str, path: str, query: str = "") -> Response:
        """The answer to ``method path?query``; ``path`` is relative to the mount point (starts
        with ``/``) and may still be percent-encoded. A HEAD answer has the headers of the GET
        one; the caller leaves its body out."""
        if method not in ("GET", "HEAD"):
            r = _text(405, b"read-only")
            return Response(r.status, (*r.headers, ("Allow", "GET, HEAD")), r.body)
        path = unquote(path)
        try:
            if path in ("/", "/index.html"):
                return Response.of(200, "text/html; charset=utf-8",
                                   ((STATIC / "index.html").read_bytes(),))
            if path in self._json:
                with self._lock:
                    if path not in self._cache:
                        self._cache[path] = json.dumps(self._json[path]()).encode()
                return Response.of(200, "application/json", (self._cache[path],))
            if path == "/api/cloud":
                try:
                    doc = self.cloud(query)
                except (UsageError, ValueError) as exc:  # bad attributes, or not derivable
                    return _error(400, str(exc))
                return Response.of(200, "application/octet-stream", doc.pieces, doc.size)
            if path == "/api/segmented.png" and self.bundle.segmented_png is not None:
                return Response.of(200, "image/png", (self.bundle.segmented_png,))
            if path.startswith("/static/"):
                return static_file(path.removeprefix("/static/"))
            if path == "/favicon.ico":
                return Response.of(204, "image/x-icon", (b"",))
            return _text(404, b"not found")
        except Exception as exc:  # keep serving; the page shows the message
            log.warning("viewer: %s failed: %s", path, exc, exc_info=True)
            return _error(500, f"{type(exc).__name__}: {exc}")
