"""``view.sh``'s read-only local web server (127.0.0.1, any free port): the standard library's
threading HTTP server answering every request with :class:`~oh_my_slam.viewer.routes.ViewerRoutes`
(the routes, framework-neutral, are documented there)."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from oh_my_slam.viewer.bundle import ViewBundle
from oh_my_slam.viewer.routes import ViewerRoutes


def make_handler(bundle: ViewBundle) -> type[BaseHTTPRequestHandler]:
    routes = ViewerRoutes(bundle)

    class Handler(BaseHTTPRequestHandler):
        server_version = "oh-my-slam-viewer"

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
            pass

        def _answer(self) -> None:
            url = urlparse(self.path)
            r = routes.handle(self.command, url.path, url.query)
            try:
                self.send_response(r.status)
                for k, v in r.headers:
                    self.send_header(k, v)
                self.end_headers()
                if self.command != "HEAD":
                    for piece in r.body:
                        self.wfile.write(piece)
            except (BrokenPipeError, ConnectionResetError):
                pass

        do_GET = do_HEAD = do_POST = do_PUT = do_DELETE = do_PATCH = _answer

    return Handler


def serve(bundle: ViewBundle) -> ThreadingHTTPServer:
    """Bind 127.0.0.1 on any free port (a fixed one may be held by another process); the caller
    runs ``serve_forever``."""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(bundle))
    httpd.daemon_threads = True
    return httpd


def url_of(httpd: ThreadingHTTPServer) -> str:
    host, port = httpd.server_address[:2]
    return f"http://{host}:{port}/"
