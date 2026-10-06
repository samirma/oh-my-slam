"""A record-or-replay proxy of the inference server, so that a command and the same command run as
a ``server.sh`` request get the same inference (http_server.md "Evaluation → Parity": response
bodies byte-identical to the command's result).

The models are not bit-reproducible from one request to the next (GeoCalib's gravity, for one,
differs in the last digits between two runs of the same image), so two runs of one command differ
by themselves. The proxy listens on its own Unix socket, in a runtime folder of its own
(``OH_MY_SLAM_RUNTIME_DIR``, the variable every command and the service's requests read to find the
inference server): the first request of a kind is forwarded to the real server and its response
recorded; every later identical request is answered from the record. A request is identified by
its route, its fields without paths, and the contents of the files it names (``*_path`` /
``*_paths``: the image as sent), so different images never share a response and concurrent
requests (a mapping update's frames) cannot be swapped. Response files (``*_path``: depth maps,
masks) are copied into the store when recorded and back into the request's ``out_dir`` (else a
temporary folder) when replayed. Anything else (``/health``) is forwarded unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socketserver
import tempfile
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import httpx

SOCKET = "srv.sock"  # core.paths.socket_path() inside the runtime folder
STORE = "store"


def _walk(obj: Any, fn: Callable[[str, Any], Any]) -> Any:
    """``obj`` with every value of a key ending in ``_path`` (a string) mapped by ``fn(key, v)``."""
    if isinstance(obj, dict):
        return {k: fn(k, v) if k.endswith("_path") and isinstance(v, str) else _walk(v, fn)
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk(v, fn) for v in obj]
    return obj


def _file_digest(path: str) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def request_key(route: str, request: dict[str, Any]) -> str:
    """Route + the request without its paths + the digest of every input file it names."""
    fields: dict[str, Any] = {}
    files: list[tuple[str, str | None]] = []
    for k, v in sorted(request.items()):
        if k == "out_dir":
            continue
        if k.endswith("_path") and isinstance(v, str):
            files.append((k, _file_digest(v)))
        elif k.endswith("_paths") and isinstance(v, list):
            files += [(f"{k}[{i}]", _file_digest(str(x))) for i, x in enumerate(v)]
        else:
            fields[k] = v
    text = json.dumps([route, fields, files], sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


class Store:
    """Recorded responses by request key (in memory; their files in ``folder``)."""

    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.responses: dict[str, Any] = {}
        self.lock = threading.Lock()
        self.recorded = self.replayed = self.forwarded = 0

    def keep(self, key: str, response: dict[str, Any]) -> None:
        def copy(_k: str, path: str) -> str:
            src = Path(path)
            if not src.is_file():
                return path
            fd, name = tempfile.mkstemp(dir=self.folder, prefix="f", suffix=src.suffix)
            os.close(fd)
            shutil.copyfile(src, name)
            return "@" + Path(name).name

        stored = _walk(response, copy)
        with self.lock:
            self.responses.setdefault(key, stored)  # the first response stands
            self.recorded += 1

    def answer(self, key: str, out_dir: str | None) -> dict[str, Any] | None:
        with self.lock:
            stored = self.responses.get(key)
            if stored is None:
                return None
            self.replayed += 1
        dest = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="oms-proxy-"))
        dest.mkdir(parents=True, exist_ok=True)

        def restore(_k: str, value: str) -> str:
            if not value.startswith("@"):
                return value
            fd, name = tempfile.mkstemp(dir=dest, suffix=Path(value).suffix)
            os.close(fd)
            shutil.copyfile(self.folder / value[1:], name)
            return name

        out: dict[str, Any] = _walk(stored, restore)
        return out


class _Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


class InferenceProxy:
    """The proxy, serving in a thread: ``runtime`` is the folder to give the commands as
    ``OH_MY_SLAM_RUNTIME_DIR``; ``upstream`` the real server's socket."""

    def __init__(self, runtime: Path, upstream: Path, timeout: float = 900.0) -> None:
        self.runtime, self.upstream = Path(runtime), Path(upstream)
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.socket = self.runtime / SOCKET
        self.store = Store(self.runtime / STORE)
        self.client = httpx.Client(transport=httpx.HTTPTransport(uds=str(self.upstream)),
                                   base_url="http://oh-my-slam", timeout=timeout)
        self.server: _Server | None = None
        self.thread: threading.Thread | None = None

    def env(self) -> dict[str, str]:
        return {"OH_MY_SLAM_RUNTIME_DIR": str(self.runtime)}

    def forward(self, method: str, path: str, body: bytes | None
                ) -> tuple[int, str, bytes]:
        try:
            r = self.client.request(method, path, content=body,
                                    headers={"Content-Type": "application/json"} if body else None)
        except httpx.HTTPError as exc:  # the real server is down: the commands see it down too
            msg = json.dumps({"error": "unavailable", "detail": f"{type(exc).__name__}"})
            return 503, "application/json", msg.encode()
        with self.store.lock:
            self.store.forwarded += 1
        return r.status_code, r.headers.get("content-type", "application/json"), r.content

    def handle(self, method: str, path: str, body: bytes | None) -> tuple[int, str, bytes]:
        if method != "POST" or not body:
            return self.forward(method, path, body)
        try:
            request = json.loads(body)
        except ValueError:
            return self.forward(method, path, body)
        if not isinstance(request, dict):
            return self.forward(method, path, body)
        key = request_key(path, request)
        recorded = self.store.answer(key, request.get("out_dir"))
        if recorded is not None:
            return 200, "application/json", json.dumps(recorded).encode()
        status, ctype, content = self.forward(method, path, body)
        if status == 200:
            try:
                self.store.keep(key, json.loads(content))
            except ValueError:
                pass
        return status, ctype, content

    def start(self) -> InferenceProxy:
        self.socket.unlink(missing_ok=True)
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                pass  # silent; also the only caller of address_string (no AF_UNIX peer address)

            def _serve(self) -> None:
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else None
                status, ctype, content = proxy.handle(self.command, self.path, body)
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

            do_GET = do_POST = do_PUT = do_DELETE = _serve

        self.server = _Server(str(self.socket), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True,
                                       name="inference-proxy")
        self.thread.start()
        return self

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        self.client.close()
        self.socket.unlink(missing_ok=True)

    def stats(self) -> dict[str, int]:
        return {"recorded": self.store.recorded, "replayed": self.store.replayed,
                "forwarded": self.store.forwarded}

    def __enter__(self) -> InferenceProxy:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


def short_runtime() -> Path:
    """A fresh runtime folder whose socket path fits ``AF_UNIX`` (104 bytes on macOS)."""
    path = Path(tempfile.mkdtemp(prefix="oms-eval-", dir="/tmp"))
    assert len(str(path / SOCKET).encode()) < 100
    return path

