"""Read-only local web server for the viewer (127.0.0.1, free port unless ``--port``)."""

from __future__ import annotations

import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from oh_my_slam.viewer.bundle import ViewBundle

STATIC = Path(str(resources.files("oh_my_slam.viewer") / "static"))
_TYPES = {".js": "text/javascript", ".css": "text/css", ".html": "text/html",
          ".json": "application/json", ".png": "image/png", ".txt": "text/plain",
          ".md": "text/plain"}


def make_handler(bundle: ViewBundle) -> type[BaseHTTPRequestHandler]:
    payloads: dict[str, tuple[str, Any]] = {
        "/api/meta": ("application/json", lambda: json.dumps(bundle.meta()).encode()),
        "/api/scene": ("application/json", lambda: json.dumps(bundle.scene).encode()),
        "/api/points.bin": ("application/octet-stream", bundle.points_bytes),
        "/api/colors.bin": ("application/octet-stream", bundle.colors_bytes),
        "/api/segments.bin": ("application/octet-stream", bundle.segments_bytes),
        "/api/labels.bin": ("application/octet-stream", bundle.labels_bytes),
        "/api/catalog": ("application/json", lambda: json.dumps(bundle.catalog).encode()),
    }
    cache: dict[str, bytes] = {}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "oh-my-slam-viewer"

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
            pass

        def _send(self, code: int, ctype: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def do_HEAD(self) -> None:
            self.do_GET()

        def do_GET(self) -> None:
            path = unquote(urlparse(self.path).path)
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

        def do_POST(self) -> None:
            self._send(405, "text/plain", b"read-only")

        do_PUT = do_DELETE = do_POST

    return Handler


def serve(bundle: ViewBundle, port: int = 0) -> ThreadingHTTPServer:
    """Bind 127.0.0.1 (port 0 = any free port); caller runs ``serve_forever``."""
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(bundle))
    httpd.daemon_threads = True
    return httpd


def url_of(httpd: ThreadingHTTPServer) -> str:
    host, port = httpd.server_address[:2]
    return f"http://{host}:{port}/"
