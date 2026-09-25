"""Read-only local web server for the viewer (127.0.0.1, any free port unless ``--port``).

Routes (GET/HEAD only; anything else is 405):

* ``/`` — the page; ``/static/…`` — its scripts, styles and vendored libraries.
* ``/api/meta`` — JSON: mode, title, stats, display transform, the cameras of the scene JSON (pose
  ``T``, centre ``position`` in the scene frame, intrinsics, image name), the point-cloud controls
  (from ``core.cloud_attrs``) and their defaults.
* ``/api/scene`` — the OpenLABEL scene JSON; ``/api/catalog`` — the catalogue rows (JSON);
  ``/api/segmented.png`` — the segmented image (``view.sh -i`` only).
* ``/api/cloud?key=value&…`` — the point cloud derived with those §2.2 attributes (keys not given
  keep their defaults; ``label`` and ``encoding`` concern PLY files only and are refused). Invalid
  input is a 400 with ``{"error": "<actionable message>"}``. The 200 body is one binary document
  (see :func:`cloud_payload`), sent straight from the cloud's arrays (:class:`CloudDocument`):
  no copy of a cloud is ever assembled in memory.
"""

from __future__ import annotations

import json
import mimetypes
import struct
import threading
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote, urlparse

import numpy as np

from oh_my_slam.core.errors import UsageError
from oh_my_slam.core.log import get_logger
from oh_my_slam.viewer.bundle import DisplayCloud, ViewBundle

log = get_logger("oh_my_slam.viewer")

STATIC = Path(str(resources.files("oh_my_slam.viewer") / "static"))
_TYPES = {".js": "text/javascript", ".css": "text/css", ".html": "text/html",
          ".json": "application/json", ".png": "image/png", ".txt": "text/plain",
          ".md": "text/plain"}
CLOUD_CACHE = 3  # recent clouds kept (e.g. toggling a control back and forth) …
CLOUD_CACHE_BYTES = 256_000_000  # … whose own arrays hold this many bytes (the latest one always;
# arrays shared with the source, as in a map's complete cloud, cost nothing and do not count)


@dataclass(frozen=True)
class CloudDocument:
    """One cloud document (see :func:`cloud_payload`) as the pieces it is sent in: the
    length-prefixed header, then every buffer (a read-only byte view of the cloud's array, not a
    copy) and its padding."""

    pieces: tuple[bytes | memoryview, ...]
    size: int  # bytes of the whole document
    owned_bytes: int  # memory it keeps beyond the cloud source (see ``DisplayCloud.owned_bytes``)

    def tobytes(self) -> bytes:
        return b"".join(self.pieces)


def cloud_document(dc: DisplayCloud, attrs: str) -> CloudDocument:
    """The binary cloud document of ``dc`` (see :func:`cloud_payload`), without copying its
    arrays."""
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
    header = json.dumps({"count": len(c), "total": dc.total, "step": dc.step, "attrs": attrs,
                         "seconds": round(dc.seconds, 4), "buffers": buffers}).encode()
    header += b" " * (-(4 + len(header)) % 4)
    head = struct.pack("<I", len(header)) + header
    return CloudDocument((head, *pieces), len(head) + offset,
                         offset if dc.owned_bytes is None else dc.owned_bytes)


def cloud_payload(dc: DisplayCloud, attrs: str) -> bytes:
    """Binary cloud document: ``uint32`` little-endian length ``J`` of a UTF-8 JSON header, the
    header (space-padded so that ``4 + J`` is a multiple of 4), then the buffers, each starting on
    a 4-byte boundary at ``4 + J + offset``. The header::

        {"count": n, "total": n0, "step": k, "attrs": "color=rgb,…", "seconds": s,
         "buffers": [{"name", "type", "size", "offset", "bytes"}, …]}

    ``count`` points are shown out of ``total`` derived (every ``step``-th). Buffers, little-endian,
    ``size`` components per point: ``position`` float32 x 3 (always), ``color`` uint8 x 3 (sRGB;
    absent for ``color=none``), ``label`` int32 x 1 (object id, 0 = unsegmented), ``normal``
    float32 x 3 (``normals=on``). The server sends the same bytes piece by piece
    (:func:`cloud_document`)."""
    return cloud_document(dc, attrs).tobytes()


def parse_cloud_payload(body: bytes) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Inverse of :func:`cloud_payload` (for tests and tools): the header and the arrays."""
    (n,) = struct.unpack_from("<I", body)
    header = json.loads(body[4:4 + n])
    base = 4 + n
    arrays = {}
    for b in header["buffers"]:
        dtype = np.dtype(b["type"]).newbyteorder("<")
        a = np.frombuffer(body, dtype, b["bytes"] // dtype.itemsize, base + b["offset"])
        arrays[b["name"]] = a.reshape(-1, b["size"]) if b["size"] > 1 else a
    return header, arrays


def make_handler(bundle: ViewBundle) -> type[BaseHTTPRequestHandler]:
    payloads: dict[str, tuple[str, Any]] = {
        "/api/meta": ("application/json", lambda: json.dumps(bundle.meta()).encode()),
        "/api/scene": ("application/json", lambda: json.dumps(bundle.scene).encode()),
        "/api/catalog": ("application/json", lambda: json.dumps(bundle.catalog).encode()),
    }
    cache: dict[str, bytes] = {}
    clouds: OrderedDict[str, CloudDocument] = OrderedDict()
    lock = threading.Lock()

    def cloud_doc(query: str) -> CloudDocument:
        attrs = bundle.parse_attrs(parse_qsl(query, keep_blank_values=True))
        key = bundle.describe(attrs)
        with lock:
            if key in clouds:
                clouds.move_to_end(key)
                return clouds[key]
        doc = cloud_document(bundle.cloud(attrs), key)
        with lock:
            clouds[key] = doc
            while len(clouds) > 1 and (len(clouds) > CLOUD_CACHE or sum(
                    d.owned_bytes for d in clouds.values()) > CLOUD_CACHE_BYTES):
                clouds.popitem(last=False)
        return doc

    class Handler(BaseHTTPRequestHandler):
        server_version = "oh-my-slam-viewer"

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
            pass

        def _send(self, code: int, ctype: str, body: bytes) -> None:
            self._send_pieces(code, ctype, len(body), (body,))

        def _send_pieces(self, code: int, ctype: str, size: int,
                         pieces: Iterable[bytes | memoryview]) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                for piece in pieces:
                    self.wfile.write(piece)

        def _error(self, code: int, message: str) -> None:
            self._send(code, "application/json", json.dumps({"error": message}).encode())

        def do_HEAD(self) -> None:
            self.do_GET()

        def do_GET(self) -> None:
            url = urlparse(self.path)
            path = unquote(url.path)
            try:
                if path in ("/", "/index.html"):
                    self._send(200, "text/html; charset=utf-8",
                               (STATIC / "index.html").read_bytes())
                elif path in payloads:
                    ctype, fn = payloads[path]
                    with lock:
                        if path not in cache:
                            cache[path] = fn()
                    self._send(200, ctype, cache[path])
                elif path == "/api/cloud":
                    try:
                        doc = cloud_doc(url.query)
                    except (UsageError, ValueError) as exc:  # bad attributes, or not derivable
                        self._error(400, str(exc))
                        return
                    self._send_pieces(200, "application/octet-stream", doc.size, doc.pieces)
                elif path == "/api/segmented.png" and bundle.segmented_png is not None:
                    self._send(200, "image/png", bundle.segmented_png)
                elif path.startswith("/static/"):
                    rel = Path(path.removeprefix("/static/"))
                    target = (STATIC / rel).resolve()
                    if STATIC.resolve() not in target.parents or not target.is_file():
                        self._send(404, "text/plain", b"not found")
                        return
                    ctype = _TYPES.get(target.suffix) or mimetypes.guess_type(target.name)[0] \
                        or "application/octet-stream"
                    self._send(200, ctype, target.read_bytes())
                elif path == "/favicon.ico":
                    self._send(204, "image/x-icon", b"")
                else:
                    self._send(404, "text/plain", b"not found")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:  # keep serving; the page shows the message
                log.warning("viewer: %s failed: %s", path, exc, exc_info=True)
                try:
                    self._error(500, f"{type(exc).__name__}: {exc}")
                except OSError:
                    pass

        def do_POST(self) -> None:
            self._send(405, "text/plain", b"read-only")

        do_PUT = do_DELETE = do_PATCH = do_POST

    return Handler


def serve(bundle: ViewBundle, port: int = 0) -> ThreadingHTTPServer:
    """Bind 127.0.0.1 (port 0 = any free port); caller runs ``serve_forever``."""
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(bundle))
    httpd.daemon_threads = True
    return httpd


def url_of(httpd: ThreadingHTTPServer) -> str:
    host, port = httpd.server_address[:2]
    return f"http://{host}:{port}/"
