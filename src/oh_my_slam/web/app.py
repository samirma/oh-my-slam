"""The HTTP routes of ``server.sh`` (spec §2.6 "API"), on Starlette (served by uvicorn).

Everything under ``/api/`` is described by ``/api/openapi.json`` (``web.openapi``): the service's
health, uploads (create and discard), the workspace's maps (as a list and one by one), and per
operation ``POST /api/ops/<op>`` — which runs the command within the request and answers when it
ends (``web.runner``) — and ``POST /api/ops/<op>/validate``. ``/`` is the web application
(``web/static``: plain ES modules built only on the public API); ``/static/…`` serves its files.

An operation's answer is the command's: on success its stdout, byte for byte, in the result's media
type, with the command's per-stage timings in a ``Server-Timing`` header (its own stage names, in
milliseconds, and ``total``); otherwise the command's message and the code of its exit status, by
the generic exit status → HTTP status rule.

Every request must name this machine in ``Host`` (no DNS rebinding); a state-changing request must
come from no foreign ``Origin`` (scheme, host and port: this service's own) and carry a content
type a cross-site page cannot send without a CORS preflight, which this service never grants:
``application/json``, or for an upload any type but the CORS-safelisted ``text/plain``,
``application/x-www-form-urlencoded`` and ``multipart/form-data``.
"""

from __future__ import annotations

import json
import mimetypes
import os
import shutil
import socket
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect, Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

from oh_my_slam.commands import spec
from oh_my_slam.core.errors import (
    HTTP_STATUS,
    ExitCode,
    OhMySlamError,
    ServerUnavailableError,
    error_code,
    http_status,
)
from oh_my_slam.version import __version__
from oh_my_slam.web import openapi
from oh_my_slam.web.operations import (
    Operation,
    Prepared,
    error_body,
    media_of,
    operations,
    prepare,
    problem,
)
from oh_my_slam.web.runner import STDOUT, STOPPING, TIMINGS, Outcome, Run, RunError, Runner
from oh_my_slam.web.workspace import NotFoundError, Workspace

START_COMMAND = "./start_inference_server.sh"
MAX_UPLOAD_BYTES = 8 << 30  # 8 GiB: room for a long phone video
MIN_FREE_BYTES = 1 << 30  # an upload never leaves less than this free on the workspace's disk
UPLOAD_CHECK_BYTES = 64 << 20  # free space is re-checked as an undeclared upload grows
HOSTS_REFRESH_S = 30.0  # an unknown Host re-reads the machine's addresses at most this often
SAFELISTED = frozenset({"", "text/plain", "application/x-www-form-urlencoded",
                        "multipart/form-data"})  # what a cross-site form sends without a preflight
CLIENT_GONE = 499  # what an interrupted request is logged with (nobody reads it)
Json = dict[str, Any]
WEB_STATIC = Path(str(resources.files("oh_my_slam.web") / "static"))
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


def timing_header(record: Json) -> str:
    """A command's timing record (``core.timing``) as a ``Server-Timing`` header value: each stage
    under its own name, then ``total``, in milliseconds."""
    stages = [*record["stages_s"].items(), ("total", record["total_s"])]
    return ", ".join(f"{name};dur={float(s) * 1000:.1f}" for name, s in stages)


def server_timing(folder: Path) -> str | None:
    """The ``Server-Timing`` value of the timing record a request's command wrote, if any."""
    try:
        return timing_header(json.loads((folder / TIMINGS).read_text()))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def failure(outcome: Outcome) -> tuple[int, Json]:
    """HTTP status and body of a request whose command did not succeed: its message and the code
    of its exit status, by the generic rule; a service that stopped under it says so."""
    if outcome.interrupted == STOPPING:
        return 503, {"error": {"code": STOPPING, "message": "the service stopped: the command was "
                               "interrupted, as Ctrl-C would (a map update leaves the map as it "
                               "was); send the request again once the service runs",
                               "http_status": 503}}
    status = http_status(outcome.code)
    return status, {"error": {"code": error_code(outcome.code), "exit_code": outcome.code,
                              "message": outcome.message or "", "http_status": status}}


class _ResultResponse(FileResponse):
    """The command's stdout, sent once: the request's folder is deleted when the response ends,
    whether it was sent in full or not (the service keeps no results)."""

    def __init__(self, folder: Path, media_type: str, headers: dict[str, str]) -> None:
        super().__init__(folder / STDOUT, media_type=media_type, headers=headers)
        self.folder = folder

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await run_in_threadpool(shutil.rmtree, self.folder, True)


@dataclass
class Service:
    """What the routes share: the workspace, the request runner and the service's own facts."""

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

    def health(self, requests: dict[str, int], in_progress: list[Json]) -> Json:
        return {
            "status": "ok",
            "service": {"version": __version__, "url": self.url, "pid": os.getpid(),
                        "workspace": self.workspace.root.name, "data": str(self.workspace.root),
                        "started_at": self.started_at, "requests": requests,
                        "in_progress": in_progress},
            "inference": self.inference_health(),
        }

    def prepare(self, op: Operation, params: Any) -> Prepared:
        """The request checked as the command checks it, then — when it will use the inference
        server — the commands' own check of that server."""
        prep = prepare(op, params, self.workspace)
        if not prep.problems and prep.inference:
            p = self.inference_check()
            if p is not None:
                prep.problems.append(p)
        return prep

    def validate(self, op: Operation, params: Any) -> Json:
        prep = self.prepare(op, params)
        return {"valid": not prep.problems, "command": prep.command,
                "inference": prep.inference if not prep.problems else None,
                "problems": [p.describe() for p in prep.problems],
                "by_parameter": spec.by_parameter(prep.problems)}


async def disconnected(request: Request) -> None:
    """Return once the client has gone (its request body was read in full before)."""
    while (await request.receive())["type"] != "http.disconnect":
        pass


def answer(outcome: Outcome, prep: Prepared, folder: Path) -> Response:
    """An operation's response: the command's stdout and its timings on success, else its
    error."""
    if outcome.code == 0:
        timing = server_timing(folder)
        return _ResultResponse(folder, media_of(prep.result_format),
                               {"Server-Timing": timing} if timing else {})
    shutil.rmtree(folder, ignore_errors=True)
    status, body = failure(outcome)
    return JSONResponse(body, status)


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
        """The web application's files."""
        rel = request.path_params["path"]
        root = WEB_STATIC
        try:
            target = (root / rel).resolve()
        except (ValueError, OSError):  # e.g. an embedded NUL byte
            target = root
        if root.resolve() not in target.parents or not target.is_file():
            raise NotFoundError(f"no file {request.path_params['path']}")
        media = _STATIC_TYPES.get(target.suffix) or mimetypes.guess_type(target.name)[0]
        return FileResponse(target, media_type=media, headers={"Cache-Control": "no-cache"})

    async def health(request: Request) -> Response:
        return JSONResponse(await run_in_threadpool(service.health, runner.counts(),
                                                    runner.in_progress()))

    async def openapi_doc(request: Request) -> Response:
        return JSONResponse(openapi.document(service.ops))

    async def run_op(request: Request) -> Response:
        """Validate, wait for the request's turn, run the command, answer when it ends."""
        ticket, arrived = runner.ticket(), time.time()
        op = op_of(request)
        params = await body_json(request)
        if runner.stopping:
            return _error(503, STOPPING, "the service is stopping; send the request again once "
                          "it runs")
        prep = await run_in_threadpool(service.prepare, op, params)
        if prep.problems:  # refused: the uploads it was given are consumed all the same
            runner.discard(prep.uploads)
            status, body = error_body(prep.problems)
            return JSONResponse(body, status)
        run = Run(ticket, op.program.prog, op.module, prep.argv, prep.inference, prep.writes,
                  prep.uploads, op.id, prep.command, arrived)
        try:
            runner.admit(run)
        except RunError as exc:
            if exc.code != "upload_in_use":
                runner.discard(prep.uploads)
            return _error(exc.status, exc.code, str(exc))
        outcome = await runner.wait(run, disconnected(request))
        folder = ws.request_dir(run.id)
        if outcome.interrupted is not None and outcome.interrupted != STOPPING:
            shutil.rmtree(folder, ignore_errors=True)
            return Response(status_code=CLIENT_GONE)  # the client left: nobody reads it
        return answer(outcome, prep, folder)

    async def validate(request: Request) -> Response:
        op = op_of(request)
        params = await body_json(request)
        return JSONResponse(await run_in_threadpool(service.validate, op, params))

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

    async def delete_upload(request: Request) -> Response:
        uid = request.path_params["id"]
        ws.upload(uid)
        if runner.in_use(uid):
            return _error(409, "upload_in_use", f"upload {uid} is the input of a request in "
                          "progress; it is deleted when that request ends")
        ws.delete_upload(uid)
        return Response(status_code=204)

    # -- maps --------------------------------------------------------------------------------------

    async def maps(request: Request) -> Response:
        return JSONResponse(await run_in_threadpool(ws.list_maps))

    async def map_detail(request: Request) -> Response:
        return JSONResponse(await run_in_threadpool(ws.map_summary, request.path_params["name"],
                                                    True))

    routes = [
        Route("/", index),
        Route("/static/{path:path}", static),
        Route("/api/health", health),
        Route("/api/openapi.json", openapi_doc),
        Route("/api/ops/{op}", run_op, methods=["POST"]),
        Route("/api/ops/{op}/validate", validate, methods=["POST"]),
        Route("/api/uploads", upload, methods=["POST"]),
        Route("/api/uploads/{id}", delete_upload, methods=["DELETE"]),
        Route("/api/maps", maps),
        Route("/api/maps/{name}", map_detail),
    ]

    async def not_found(request: Request, exc: Exception) -> Response:
        return _error(404, "not_found", str(exc))

    async def command_error(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, OhMySlamError)
        code = ExitCode(exc.exit_code)
        return _error(HTTP_STATUS[code], code.name.lower(), str(exc))

    async def disconnect(request: Request, exc: Exception) -> Response:
        return Response(status_code=CLIENT_GONE)

    app = Starlette(routes=routes, exception_handlers={
        NotFoundError: not_found, OhMySlamError: command_error, ClientDisconnect: disconnect})
    return guard(app, service)
