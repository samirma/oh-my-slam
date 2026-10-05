"""The HTTP routes of ``server.sh`` (spec §2.6 "API"), on Starlette (served by uvicorn).

Everything under ``/api/`` is described by ``/api/openapi.json`` (``web.openapi``). The viewer's
data is served in-process by the viewer's own code (``viewer.routes.ViewerRoutes``):
``/api/maps/<name>/viewer/…`` over a workspace map's read-only bundle, ``/api/jobs/<id>/viewer/…``
over the bundle a job saved (``viewer.bundle.load_bundle``); ``/viewer/map/<name>/`` and
``/viewer/job/<id>/`` are the same routes as stable page URLs (the page's own URLs are relative).
``/`` is the web application (``web/static``: plain ES modules built only on the public API);
``/static/…`` serves its files, ``/static/viewer/…`` the viewer's own modules and vendored
libraries (which the 3D scene viewer reuses), and ``/static/openlabel_json_schema.json`` the
vendored scene schema that the browser validates scene documents against.

Every request must name this machine in ``Host`` (no DNS rebinding); a state-changing request must
come from no foreign ``Origin`` (scheme, host and port: this service's own) and carry a content
type a cross-site page cannot send without a CORS preflight, which this service never grants:
``application/json``, or for an upload any type but the CORS-safelisted ``text/plain``,
``application/x-www-form-urlencoded`` and ``multipart/form-data``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import mimetypes
import os
import shutil
import socket
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect, Request
from starlette.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import Route

from oh_my_slam.commands import spec
from oh_my_slam.core.errors import HTTP_STATUS, ExitCode, OhMySlamError, ServerUnavailableError
from oh_my_slam.version import __version__
from oh_my_slam.web import openapi
from oh_my_slam.web.jobs import TERMINAL, JobError, Runner
from oh_my_slam.web.operations import (
    OUT_DIR,
    Operation,
    error_body,
    inference_of,
    operations,
    prepare,
    problem,
)
from oh_my_slam.web.workspace import NotFoundError, Workspace

START_COMMAND = "./start_inference_server.sh"
SSE_POLL_S = 0.2
SSE_HEARTBEAT_S = 15.0
MAX_UPLOAD_BYTES = 8 << 30  # 8 GiB: room for a long phone video
MIN_FREE_BYTES = 1 << 30  # an upload never leaves less than this free on the workspace's disk
UPLOAD_CHECK_BYTES = 64 << 20  # free space is re-checked as an undeclared upload grows
HOSTS_REFRESH_S = 30.0  # an unknown Host re-reads the machine's addresses at most this often
SAFELISTED = frozenset({"", "text/plain", "application/x-www-form-urlencoded",
                        "multipart/form-data"})  # what a cross-site form sends without a preflight
VIEWER_CACHE = 2  # loaded viewer bundles kept per kind (maps, jobs)
DISPLAY_DIR = "display"  # a job's PLY files as the viewer draws them (cloud documents)
_MEDIA = {".json": "application/json", ".ply": "application/octet-stream", ".png": "image/png",
          ".csv": "text/csv", ".md": "text/markdown", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
Json = dict[str, Any]
WEB_STATIC = Path(str(resources.files("oh_my_slam.web") / "static"))
VIEWER_STATIC = Path(str(resources.files("oh_my_slam.viewer") / "static"))
SCHEMA_FILE = Path(str(resources.files("oh_my_slam.schema") / "openlabel_json_schema.json"))
_STATIC_TYPES = {".js": "text/javascript", ".css": "text/css", ".html": "text/html",
                 ".json": "application/json", ".png": "image/png", ".svg": "image/svg+xml",
                 ".txt": "text/plain"}


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message, "http_status": status}},
                        status)


def inference_health() -> Json:
    """The inference server's health, or why it is down and the command that starts it."""
    from oh_my_slam.client.client import InferenceClient

    try:
        h = InferenceClient().health(timeout=1.0)
    except ServerUnavailableError as exc:
        return {"status": "down", "message": str(exc), "start_command": START_COMMAND}
    return {"status": h.status, "health": h.model_dump(),
            "start_command": None if h.status == "ready" else START_COMMAND}


def inference_problem() -> spec.Problem | None:
    """The commands' own check of the inference server (exit 3 when it is down or its models
    failed); a server still loading is fine — the command waits for it itself."""
    from oh_my_slam.client.client import InferenceClient

    client = InferenceClient()
    try:
        if client.health(timeout=1.0).status in ("ready", "loading"):
            return None
        client.require_ready(wait_loading_s=0.0)
    except OhMySlamError as exc:
        return problem((), str(exc), exc.exit_code, "inference_server")
    return None


def machine_hosts() -> set[str]:
    """The names and addresses this machine answers to (``Host`` / ``Origin`` check)."""
    names = {"localhost", "127.0.0.1", "::1"}
    host = socket.gethostname().lower()
    names |= {host, host.split(".")[0], f"{host.split('.')[0]}.local"}
    try:
        names.add(socket.getfqdn().lower())
    except OSError:
        pass
    try:
        import psutil

        for addrs in psutil.net_if_addrs().values():
            names |= {a.address.split("%")[0].lower() for a in addrs
                      if a.family in (socket.AF_INET, socket.AF_INET6)}
    except Exception:  # the loopback names still work
        pass
    return names


def _hostname(value: str) -> str:
    """The host part of a ``Host`` header (``[::1]:80`` → ``::1``)."""
    value = value.strip().lower()
    if value.startswith("["):
        return value[1:].split("]")[0]
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


class _LRU:
    def __init__(self, size: int) -> None:
        self.size = size
        self.items: OrderedDict[Any, Any] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: Any, make: Callable[[], Any]) -> Any:
        """The cached value of ``key``, else ``make()`` (built under the lock: one at a time)."""
        with self.lock:
            if key in self.items:
                self.items.move_to_end(key)
                return self.items[key]
            value = make()
            self.items[key] = value
            while len(self.items) > self.size:
                self.items.popitem(last=False)
            return value


@dataclass
class Service:
    """What the routes share: the workspace, the job runner and the service's own facts."""

    workspace: Workspace
    runner: Runner
    url: str = ""
    started_at: float = field(default_factory=time.time)
    inference_check: Callable[[], spec.Problem | None] = inference_problem
    inference_health: Callable[[], Json] = inference_health
    ops: dict[str, Operation] = field(default_factory=operations)
    extra_hosts: set[str] = field(default_factory=set)  # e.g. the test client's
    max_upload_bytes: int = MAX_UPLOAD_BYTES
    min_free_bytes: int = MIN_FREE_BYTES
    _hosts: set[str] | None = None
    _hosts_at: float = 0.0
    _map_views: _LRU = field(default_factory=lambda: _LRU(VIEWER_CACHE))
    _job_views: _LRU = field(default_factory=lambda: _LRU(VIEWER_CACHE))
    _building: dict[Path, Any] = field(default_factory=dict)  # display_cloud builds in flight
    _building_lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if self.runner.reevaluate is None:  # a conditional job's inference need, at its start
            self.runner.reevaluate = lambda job: inference_of(self.ops[job.operation], job.actual)

    @property
    def port(self) -> int | None:
        return urlsplit(self.url).port if self.url else None

    def knows_host(self, host: str) -> bool:
        """Whether ``host`` names this machine; an unknown one re-reads its addresses (a new
        network), at most every ``HOSTS_REFRESH_S``."""
        now = time.monotonic()
        if self._hosts is None or (host not in self._hosts
                                   and now - self._hosts_at > HOSTS_REFRESH_S):
            self._hosts = machine_hosts() | {h.lower() for h in self.extra_hosts}
            self._hosts_at = now
        return host in self._hosts

    def same_origin(self, origin: str, host_header: str) -> bool:
        """Whether ``origin`` is this service: the request's own ``Host``, or one of the machine's
        names on the service's port, over http."""
        parts = urlsplit(origin)
        if parts.scheme != "http" or not parts.hostname:
            return False
        if parts.netloc.lower() == host_header.strip().lower():
            return True
        return self.knows_host(parts.hostname.lower()) and parts.port == self.port

    def health(self) -> Json:
        return {
            "status": "ok",
            "service": {"version": __version__, "url": self.url, "pid": os.getpid(),
                        "workspace": self.workspace.root.name, "data": str(self.workspace.root),
                        "started_at": self.started_at, "jobs": self.runner.counts(),
                        "max_upload_bytes": self.max_upload_bytes},
            "inference": self.inference_health(),
        }

    # -- submission --------------------------------------------------------------------------------

    def submit(self, op: Operation, params: Any, resubmitted_from: str | None = None,
               viewer: bool = False) -> JSONResponse:
        jid = self.runner.new_id()
        prep = prepare(op, params, self.workspace, self.workspace.job_dir(jid), viewer)
        if not prep.problems and prep.inference:
            p = self.inference_check()
            if p is not None:
                prep.problems.append(p)
        if prep.problems:
            status, body = error_body(prep.problems)
            return JSONResponse(body, status)
        try:
            job = self.runner.submit(op, dict(params), prep, jid, resubmitted_from)
        except JobError as exc:
            return _error(exc.status, exc.code, str(exc))
        return JSONResponse(job.public(), 202, headers={"Location": f"/api/jobs/{job.id}"})

    def validate(self, op: Operation, params: Any, viewer: bool = False) -> Json:
        prep = prepare(op, params, self.workspace, self.workspace.job_dir("validate"), viewer)
        if not prep.problems and prep.inference:
            p = self.inference_check()
            if p is not None:
                prep.problems.append(p)
        return {"valid": not prep.problems, "command": prep.command,
                "inference": prep.inference if not prep.problems else None,
                "problems": [p.describe() for p in prep.problems],
                "by_parameter": spec.by_parameter(prep.problems)}

    # -- viewers -----------------------------------------------------------------------------------

    def map_routes(self, name: str) -> Any:
        """The viewer's routes over a map's read-only bundle, rebuilt when the map changed."""
        root = self.workspace.map_dir(name)
        stamp = (root / "map.json").stat().st_mtime_ns

        def make() -> Any:
            from oh_my_slam.viewer.bundle import map_bundle
            from oh_my_slam.viewer.routes import ViewerRoutes

            return ViewerRoutes(map_bundle(root))

        return self._map_views.get((name, stamp), make)

    def job_routes(self, jid: str) -> Any:
        """The viewer's routes over the bundle a job saved (a map's: that map's routes)."""
        from oh_my_slam.viewer.bundle import load_bundle, saved_map

        folder = self.runner.viewer_dir(jid)
        if folder is None:
            raise NotFoundError(f"job {jid} has no viewer")
        m = saved_map(folder)
        if m is not None:
            return self.map_routes(m.name)

        def make() -> Any:
            from oh_my_slam.viewer.routes import ViewerRoutes

            return ViewerRoutes(load_bundle(folder))

        return self._job_views.get(jid, make)

    def display_cloud(self, jid: str, path: Path) -> Path:
        """A PLY file of a job as the viewer draws it (the 3D scene viewer): the viewer's cloud
        document, within the display budget (spec §2.5), with the file's header comments.

        The document is kept as a file in the job's own folder (``display/``, deleted with the
        job), never in memory, and built once per file version: concurrent requests for the same
        file wait for the one build (its future), others build in parallel."""
        from concurrent.futures import Future

        from oh_my_slam.viewer import bundle

        st = path.stat()
        key = f"{path.name}|{st.st_mtime_ns}|{st.st_size}|{bundle.DISPLAY_POINT_BUDGET}"
        target = self.workspace.job_dir(jid) / DISPLAY_DIR / (
            hashlib.sha256(f"{path}|{key}".encode()).hexdigest()[:32] + ".cloud")
        if target.is_file():
            return target
        with self._building_lock:
            fut = self._building.get(target)
            mine = fut is None
            if mine:
                fut = self._building[target] = Future()
        assert fut is not None
        if not mine:
            return fut.result()
        try:
            from oh_my_slam.viewer.routes import cloud_document

            dc, comments = bundle.ply_display(path)
            attrs = next((c.removeprefix("attributes ") for c in comments
                          if c.startswith("attributes ")), "")
            doc = cloud_document(dc, attrs, {"comments": comments})
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}.part")
            with part.open("wb") as f:
                for piece in doc.pieces:
                    f.write(piece)
            part.replace(target)
            fut.set_result(target)
            return target
        except BaseException as exc:
            fut.set_exception(exc)
            raise
        finally:
            with self._building_lock:
                self._building.pop(target, None)


def _viewer_response(r: Any, method: str) -> Response:
    """A ``viewer.routes.Response`` as a Starlette response (its pieces streamed in order)."""
    headers = dict(r.headers)
    if method == "HEAD":
        return Response(b"", r.status, headers=headers)

    async def body() -> AsyncIterator[bytes | memoryview]:
        for piece in r.body:
            yield piece

    return StreamingResponse(body(), r.status, headers=headers)


_STATE_CHANGING = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def guard(app: Any, service: Service) -> Callable[..., Awaitable[None]]:
    """``Host``, ``Origin`` and content-type checks (CSRF and DNS rebinding) around ``app``."""

    async def asgi(scope: Json, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        reason, status = None, 403
        host = headers.get("host", "")
        if not service.knows_host(_hostname(host)):
            reason = "the Host header does not name this machine"
        elif scope["method"] in _STATE_CHANGING:
            origin = headers.get("origin")
            ctype = headers.get("content-type", "").split(";")[0].strip().lower()
            if origin is not None and not service.same_origin(origin, host):
                reason = f"requests from {origin} are not accepted"
            elif scope["method"] == "DELETE":
                pass
            elif scope["path"] == "/api/uploads":
                if ctype in SAFELISTED:
                    reason, status = "send the file with its own media type (e.g. image/jpeg " \
                        "or application/octet-stream)", 415
            elif ctype != "application/json":
                reason, status = "send the request body as application/json", 415
        if reason is None:
            await app(scope, receive, send)
            return
        await _error(status, "forbidden" if status == 403 else "unsupported_media_type",
                     reason)(scope, receive, send)

    return asgi


def create_app(service: Service) -> Callable[..., Awaitable[None]]:
    ws, runner = service.workspace, service.runner

    def op_of(request: Request) -> Operation:
        op = service.ops.get(request.path_params["op"])
        if op is None:
            raise NotFoundError(f"no operation {request.path_params['op']}; see /api/openapi.json")
        return op

    def wants_viewer(request: Request) -> bool:
        return request.query_params.get("viewer", "").lower() in ("1", "true", "yes")

    async def body_json(request: Request) -> Any:
        raw = await request.body()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except ValueError:
            return None  # refused by prepare as not an object

    async def index(request: Request) -> Response:
        return FileResponse(WEB_STATIC / "index.html", media_type="text/html; charset=utf-8",
                            headers={"Cache-Control": "no-cache"})

    async def static(request: Request) -> Response:
        """The web application's files; ``viewer/…`` the viewer's (its modules and vendored
        libraries); ``openlabel_json_schema.json`` the scene schema."""
        rel = request.path_params["path"]
        if rel == SCHEMA_FILE.name:
            return FileResponse(SCHEMA_FILE, media_type="application/json")
        root = WEB_STATIC
        if rel.startswith("viewer/"):
            root, rel = VIEWER_STATIC, rel.removeprefix("viewer/")
        try:
            target = (root / rel).resolve()
        except (ValueError, OSError):  # e.g. an embedded NUL byte
            target = root
        if root.resolve() not in target.parents or not target.is_file():
            raise NotFoundError(f"no file {request.path_params['path']}")
        media = _STATIC_TYPES.get(target.suffix) or mimetypes.guess_type(target.name)[0]
        return FileResponse(target, media_type=media, headers={"Cache-Control": "no-cache"})

    async def health(request: Request) -> Response:
        return JSONResponse(await run_in_threadpool(service.health))

    async def openapi_doc(request: Request) -> Response:
        return JSONResponse(openapi.document(service.ops))

    async def describe(request: Request) -> Response:
        return JSONResponse(spec.describe())

    async def submit(request: Request) -> Response:
        op = op_of(request)
        params = await body_json(request)
        return await run_in_threadpool(service.submit, op, params, None, wants_viewer(request))

    async def validate(request: Request) -> Response:
        op = op_of(request)
        params = await body_json(request)
        return JSONResponse(await run_in_threadpool(service.validate, op, params,
                                                    wants_viewer(request)))

    # -- uploads -----------------------------------------------------------------------------------

    def too_large(size: int) -> Response | None:
        if size > service.max_upload_bytes:
            return _error(413, "too_large", f"an upload may hold at most "
                          f"{service.max_upload_bytes / 2**30:g} GiB")
        if shutil.disk_usage(ws.uploads).free - size < service.min_free_bytes:
            return _error(413, "insufficient_storage", f"not enough free space in {ws.root} for "
                          "this upload; free some space or use a path inside the workspace")
        return None

    async def upload(request: Request) -> Response:
        try:
            declared = int(request.headers.get("content-length", "0"))
        except ValueError:
            declared = 0
        refused = too_large(declared)
        if refused is not None:
            return refused
        name = request.query_params.get("name", "")
        uid, target = ws.new_upload(name)
        part = target.with_name(f".{target.name}.part")
        size = 0
        try:
            with part.open("wb") as f:
                async for chunk in request.stream():
                    size += len(chunk)
                    step = UPLOAD_CHECK_BYTES
                    checked = size > service.max_upload_bytes or (
                        size > declared and size // step != (size - len(chunk)) // step)
                    if checked and (refused := too_large(size)) is not None:
                        ws.delete_upload(uid)
                        return refused
                    f.write(chunk)
            part.replace(target)
        except BaseException:  # interrupted (client gone, service stopping): deleted at once
            ws.delete_upload(uid)
            raise
        return JSONResponse(ws.upload(uid).describe(ws.root), 201)

    async def uploads(request: Request) -> Response:
        return JSONResponse([u.describe(ws.root) for u in ws.list_uploads()])

    async def delete_upload(request: Request) -> Response:
        uid = request.path_params["id"]
        ws.upload(uid)
        if any(uid in j.uploads and j.state not in TERMINAL for j in runner.all_jobs()):
            return _error(409, "upload_in_use", f"upload {uid} is the input of a queued or "
                          "running job; cancel the job instead")
        ws.delete_upload(uid)
        return Response(status_code=204)

    # -- maps --------------------------------------------------------------------------------------

    async def maps(request: Request) -> Response:
        return JSONResponse(await run_in_threadpool(ws.list_maps))

    async def map_detail(request: Request) -> Response:
        return JSONResponse(await run_in_threadpool(ws.map_summary, request.path_params["name"],
                                                    True))

    async def map_file(request: Request) -> Response:
        p = ws.map_file(request.path_params["name"], request.path_params["path"])
        return FileResponse(p, media_type=_MEDIA.get(p.suffix.lower()))

    # -- jobs --------------------------------------------------------------------------------------

    async def jobs(request: Request) -> Response:
        return JSONResponse([j.public() for j in runner.all_jobs()])

    async def job(request: Request) -> Response:
        return JSONResponse(runner.get(request.path_params["id"]).public())

    async def cancel(request: Request) -> Response:
        return JSONResponse(runner.cancel(request.path_params["id"]).public())

    async def resubmit(request: Request) -> Response:
        old = runner.get(request.path_params["id"])
        override = await body_json(request)
        if not isinstance(override, dict):
            return _error(400, "usage", "the request body must be a JSON object of parameters")
        op = service.ops[old.operation]
        params = {**old.params, **override}
        return await run_in_threadpool(service.submit, op, params, old.id,
                                       old.saves_viewer and not op.browser)

    def out_dir(jid: str) -> Path:
        return ws.job_dir(runner.get(jid).id) / OUT_DIR

    def media_of(jid: str, path: Path) -> str | None:
        op = service.ops.get(runner.get(jid).operation)
        for o in op.mode.outputs if op else ():
            if o.name == path.name:
                return o.describe()["media_type"]
        return _MEDIA.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0]

    async def result(request: Request) -> Response:
        j = runner.get(request.path_params["id"])
        if j.state != "succeeded" or j.result_name is None:
            return _error(404, "not_found", f"job {j.id} has no result ({j.state})")
        p = out_dir(j.id) / j.result_name
        media = media_of(j.id, p) if j.result_format is None else spec.Output(
            "result", "stdout", j.result_format, "").describe()["media_type"]
        return FileResponse(p, media_type=media, filename=j.result_name)

    async def files(request: Request) -> Response:
        jid = request.path_params["id"]
        root = out_dir(jid)
        out = [{"path": str(p.relative_to(root)), "size": p.stat().st_size,
                "media_type": media_of(jid, p), "url": f"/api/jobs/{jid}/files/"
                f"{p.relative_to(root)}"}
               for p in sorted(root.rglob("*")) if p.is_file() and not p.name.startswith(".")]
        return JSONResponse(out)

    async def job_file(request: Request) -> Response:
        from oh_my_slam.web.workspace import inside

        jid = request.path_params["id"]
        p = inside(out_dir(jid), request.path_params["path"])
        if not p.is_file():
            raise NotFoundError(f"no file {request.path_params['path']} in job {jid}")
        return FileResponse(p, media_type=media_of(jid, p), filename=p.name)

    async def display_cloud(request: Request) -> Response:
        """A job's PLY (``?file=<path>``, else its result) as the viewer draws it."""
        from oh_my_slam.web.workspace import inside

        j = runner.get(request.path_params["id"])
        rel = request.query_params.get("file")
        if rel is None:
            if j.state != "succeeded" or j.result_name is None:
                return _error(404, "not_found", f"job {j.id} has no result ({j.state})")
            rel = j.result_name
        p = inside(out_dir(j.id), rel)
        if not p.is_file():
            raise NotFoundError(f"no file {rel} in job {j.id}")
        try:
            doc = await run_in_threadpool(service.display_cloud, j.id, p)
        except (ValueError, KeyError, UnicodeDecodeError) as exc:
            return _error(400, "usage", f"{rel} is not a PLY file the viewer can draw: {exc}")
        return FileResponse(doc, media_type="application/octet-stream")

    async def display_transform(request: Request) -> Response:
        """The viewer's display transform of a scene: identity for map coordinates; for a single
        image's camera frame view.sh's upright transform, with the estimated ``up=x,y,z`` when the
        scene states one. The frame is decided by the viewer's one rule (``is_camera_frame``),
        from a scene JSON's coordinate-system types (``cs_types=a,b``) or a PLY's header
        ``comment``s; ``camera=true`` states it outright."""
        from oh_my_slam.viewer.bundle import display_transform as transform
        from oh_my_slam.viewer.bundle import is_camera_frame

        q = request.query_params
        up: list[float] | None = None
        if q.get("up"):
            try:
                up = [float(x) for x in q["up"].split(",")]
            except ValueError:
                up = []
            if len(up) != 3:
                return _error(400, "usage", "up must be x,y,z: three numbers")
        cs_types = None if "cs_types" not in q else [t for t in q["cs_types"].split(",") if t]
        camera = q.get("camera", "").lower() in ("1", "true", "yes") or \
            await run_in_threadpool(is_camera_frame, q.getlist("comment"), cs_types)
        matrix = await run_in_threadpool(transform, camera, up)
        return JSONResponse({"camera_frame": camera, "display_transform": matrix})

    async def log(request: Request) -> Response:
        p = ws.job_dir(runner.get(request.path_params["id"]).id) / "stderr.log"
        return PlainTextResponse(p.read_text("utf-8", "replace") if p.is_file() else "")

    async def timings(request: Request) -> Response:
        return JSONResponse(runner.timings(request.path_params["id"]))

    async def events(request: Request) -> Response:
        jid = request.path_params.get("id")
        if jid is not None:
            runner.get(jid)

        async def stream() -> AsyncIterator[str]:
            seen, beat = 0, time.monotonic()
            yield "retry: 1000\n\n"
            while not runner.stopping and not await request.is_disconnected():
                seen, changed = runner.changed_since(seen, jid)
                for j in changed:
                    yield f"event: job\nid: {seen}\ndata: {json.dumps(j)}\n\n"
                if jid is not None and runner.get(jid).state in TERMINAL and not changed:
                    break
                if time.monotonic() - beat > SSE_HEARTBEAT_S:
                    beat = time.monotonic()
                    yield ": keep-alive\n\n"
                await asyncio.sleep(SSE_POLL_S)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store"})

    # -- viewer ------------------------------------------------------------------------------------

    def viewer(kind: str, key: str, routes: Callable[[str], Any]) -> Callable[..., Any]:
        async def endpoint(request: Request) -> Response:
            ident = request.path_params[key]
            if "path" not in request.path_params:  # the page needs its trailing slash
                return RedirectResponse(f"{request.url.path}/")
            if kind == "job":
                runner.get(ident)
            r = await run_in_threadpool(routes(ident).handle, request.method,
                                        "/" + request.path_params["path"], request.url.query)
            return _viewer_response(r, request.method)

        return endpoint

    map_viewer = viewer("map", "name", service.map_routes)
    job_viewer = viewer("job", "id", service.job_routes)

    routes = [
        Route("/", index),
        Route("/static/{path:path}", static),
        Route("/api/health", health),
        Route("/api/openapi.json", openapi_doc),
        Route("/api/operations", describe),
        Route("/api/ops/{op}", submit, methods=["POST"]),
        Route("/api/ops/{op}/validate", validate, methods=["POST"]),
        Route("/api/uploads", upload, methods=["POST"]),
        Route("/api/uploads", uploads, methods=["GET"]),
        Route("/api/uploads/{id}", delete_upload, methods=["DELETE"]),
        Route("/api/maps", maps),
        Route("/api/maps/{name}", map_detail),
        Route("/api/maps/{name}/files/{path:path}", map_file),
        Route("/api/maps/{name}/viewer", map_viewer),
        Route("/api/maps/{name}/viewer/{path:path}", map_viewer),
        Route("/api/jobs", jobs),
        Route("/api/jobs/events", events),
        Route("/api/jobs/{id}", job),
        Route("/api/jobs/{id}/events", events),
        Route("/api/jobs/{id}/cancel", cancel, methods=["POST"]),
        Route("/api/jobs/{id}/resubmit", resubmit, methods=["POST"]),
        Route("/api/jobs/{id}/result", result),
        Route("/api/jobs/{id}/files", files),
        Route("/api/jobs/{id}/files/{path:path}", job_file),
        Route("/api/jobs/{id}/display-cloud", display_cloud),
        Route("/api/display-transform", display_transform),
        Route("/api/jobs/{id}/log", log),
        Route("/api/jobs/{id}/timings", timings),
        Route("/api/jobs/{id}/viewer", job_viewer),
        Route("/api/jobs/{id}/viewer/{path:path}", job_viewer),
        Route("/viewer/map/{name}", map_viewer),
        Route("/viewer/map/{name}/{path:path}", map_viewer),
        Route("/viewer/job/{id}", job_viewer),
        Route("/viewer/job/{id}/{path:path}", job_viewer),
    ]

    async def not_found(request: Request, exc: Exception) -> Response:
        return _error(404, "not_found", str(exc))

    async def job_error(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, JobError)
        return _error(exc.status, exc.code, str(exc))

    async def command_error(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, OhMySlamError)
        code = ExitCode(exc.exit_code)
        return _error(HTTP_STATUS[code], code.name.lower(), str(exc))

    async def disconnect(request: Request, exc: Exception) -> Response:
        return Response(status_code=499)

    app = Starlette(routes=routes, exception_handlers={
        NotFoundError: not_found, JobError: job_error, OhMySlamError: command_error,
        ClientDisconnect: disconnect})
    return guard(app, service)
