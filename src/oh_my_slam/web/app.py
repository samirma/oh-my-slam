"""The HTTP routes of ``server.sh`` (spec §2.6 "API"), on Starlette (served by uvicorn).

Everything under ``/api/`` is described by ``/api/openapi.json`` (``web.openapi``): the service's
health, uploads (create and discard), the workspace's maps (as a list and one by one), and per
operation ``POST /api/ops/<op>`` — which runs the command within the request and answers when it
ends (``web.runner``) — and ``POST /api/ops/<op>/validate``. ``/`` is the web application
(``web/static``: plain ES modules built only on the public API); ``/static/…`` serves its files,
and ``/static/viewer/lib/…`` and ``/static/viewer/vendor/…`` the §2.5 viewer's own ES modules and
vendored libraries, from the viewer's package (not copied), with which the application draws a
point-cloud result.

An operation's answer is the command's: on success its stdout, byte for byte, in the result's media
type, with the command's per-stage timings in a ``Server-Timing`` header (its own stage names, in
milliseconds, and ``total``); otherwise the command's message and the code of its exit status, by
the generic exit status → HTTP status rule.

Every request must name this machine in ``Host`` (no DNS rebinding); a state-changing request must
come from no foreign ``Origin`` (scheme, host and port: this service's own) and carry a content
type a cross-site page cannot send without a CORS preflight, which this service never grants:
``application/json``, or for a ``POST`` upload any type but the CORS-safelisted ``text/plain``,
``application/x-www-form-urlencoded`` and ``multipart/form-data``. A ``PUT`` upload (``curl -T``:
the raw file, often with no type) always needs a preflight, so it takes any type but a form's.

The service's own refusals — the errors that are no command's — are one table (:func:`refusals`:
code, HTTP status and when), which the OpenAPI document exports (``x-oms.refusals``) and the agent
skill states. Every error, an unknown route or method included, has the same JSON shape; anything
that crashes — a check, a route — is the internal error (exit 1: 500 ``internal``), told as a
command tells it, and the uploads its request names are consumed all the same.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.requests import ClientDisconnect, Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

from oh_my_slam.commands import spec
from oh_my_slam.core.constants import START_INFERENCE_SERVER
from oh_my_slam.core.errors import (
    HTTP_STATUS,
    ExitCode,
    OhMySlamError,
    ServerUnavailableError,
    error_code,
    http_status,
    internal_message,
)
from oh_my_slam.core.static import find_static
from oh_my_slam.version import __version__
from oh_my_slam.web import openapi
from oh_my_slam.web.operations import (
    Operation,
    Prepared,
    error_body,
    internal,
    media_of,
    operations,
    prepare,
    problem,
)
from oh_my_slam.web.runner import STDOUT, STOPPING, TIMINGS, Outcome, Run, RunError, Runner
from oh_my_slam.web.workspace import NotFoundError, Workspace

START_COMMAND = START_INFERENCE_SERVER  # what the health names while the server is not ready
MAX_UPLOAD_BYTES = 8 << 30  # 8 GiB: room for a long phone video
MIN_FREE_BYTES = 1 << 30  # an upload never leaves less than this free on the workspace's disk
UPLOAD_CHECK_BYTES = 64 << 20  # free space is re-checked as an undeclared upload grows
HOSTS_REFRESH_S = 30.0  # an unknown Host re-reads the machine's addresses at most this often
FORMS = frozenset({"application/x-www-form-urlencoded", "multipart/form-data"})  # never a raw file
SAFELISTED = FORMS | {"", "text/plain"}  # what a cross-site form sends without a preflight
CLIENT_GONE = 499  # what an interrupted request is logged with (nobody reads it)
Json = dict[str, Any]
WEB_STATIC = Path(str(resources.files("oh_my_slam.web") / "static"))
# the viewer's modules and vendored libraries (spec §2.6 "Image": a point-cloud result is drawn with
# the §2.5 viewer's own rendering); its page (index.html, app.js, style.css) is view.sh's alone
VIEWER_STATIC = Path(str(resources.files("oh_my_slam.viewer") / "static"))
VIEWER_PARTS = ("lib", "vendor")


@dataclass(frozen=True)
class Refusal:
    """One of the service's own refusals: its code, HTTP status and when it is answered."""

    code: str
    http_status: int
    when: str


def _gib(n: int) -> str:
    return f"{n / 2**30:g} GiB"


def refusals() -> dict[str, Refusal]:
    """The service's own refusals, by code (the commands' errors are those of the exit-code
    table). Read when called, so the texts follow the limits."""
    return {r.code: r for r in (
        Refusal("not_found", 404, "no such route, map, upload or operation"),
        Refusal("method_not_allowed", 405, "a method the route does not take"),
        Refusal("forbidden", 403, "a `Host` that does not name the service's machine, or a "
                                  "foreign `Origin`"),
        Refusal("unsupported_media_type", 415, "a `POST` whose body is not `application/json` "
                                               "(an operation) or that has no type of its own (an "
                                               "upload), or an upload sent as a form (`curl -F`)"),
        Refusal("too_large", 413, f"an upload over {_gib(MAX_UPLOAD_BYTES)}"),
        Refusal("insufficient_storage", 413, "an upload that would leave less than "
                                             f"{_gib(MIN_FREE_BYTES)} free on the workspace's "
                                             "disk"),
        Refusal("upload_in_use", 409, "an upload that another request in progress was given"),
        Refusal(STOPPING, 503, "a request that arrives while the service stops"),
    )}


def _error(status: int, code: str, message: str,
           headers: Mapping[str, str] | None = None) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message, "http_status": status}},
                        status, headers)


def refuse(code: str, message: str, headers: Mapping[str, str] | None = None) -> JSONResponse:
    """The answer of one of the service's own refusals (:func:`refusals`)."""
    return _error(refusals()[code].http_status, code, message, headers)


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
    """The names and addresses this machine answers to (``Host`` / ``Origin`` check), with
    ``0.0.0.0``, the address the service binds, which its listening line prints."""
    names = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
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
    of its exit status, by the generic rule — also when the service's stop interrupted it (exit
    130: 499 ``interrupted``, as for a client that left)."""
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
            self._hosts = machine_hosts()
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
        server — the commands' own check of that server. A check that crashes refuses it with
        the command's internal error (exit 1); the uploads it names are known all the same."""
        prep = Prepared()
        try:
            prepare(op, params, self.workspace, prep)
            if not prep.problems and prep.inference:
                p = self.inference_check()
                if p is not None:
                    prep.problems.append(p)
        except Exception as exc:
            prep.problems.append(internal(exc))
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
UPLOADS = "/api/uploads"  # POST (the web application) or PUT (curl -T) the raw file


def guard(app: Any, service: Service) -> Callable[..., Awaitable[None]]:
    """``Host``, ``Origin`` and content-type checks (CSRF and DNS rebinding) around ``app``."""

    async def asgi(scope: Json, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        reason, code = None, "forbidden"
        host = headers.get("host", "")
        method = scope["method"]
        if not service.knows_host(_hostname(host)):
            reason = "the Host header does not name this machine"
        elif method in _STATE_CHANGING:
            origin = headers.get("origin")
            ctype = headers.get("content-type", "").split(";")[0].strip().lower()
            if origin is not None and not service.same_origin(origin, host):
                reason = f"requests from {origin} are not accepted"
            elif method == "DELETE":
                pass
            elif scope["path"] == UPLOADS:
                if ctype in (SAFELISTED if method == "POST" else FORMS):
                    reason, code = "send the raw file as the body (curl -T <file>), not a " \
                        "form; a POST needs the file's own media type (e.g. image/jpeg or " \
                        "application/octet-stream)", "unsupported_media_type"
            elif ctype != "application/json":
                reason, code = "send the request body as application/json (curl -H " \
                    "'Content-Type: application/json' -d …, or curl --json …)", \
                    "unsupported_media_type"
        if reason is None:
            await app(scope, receive, send)
            return
        await refuse(code, reason)(scope, receive, send)

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
        """The web application's files; ``viewer/lib/…`` and ``viewer/vendor/…`` the viewer's."""
        rel = request.path_params["path"]
        root = WEB_STATIC
        if rel.startswith("viewer/"):
            part, _, rel = rel.removeprefix("viewer/").partition("/")
            if part not in VIEWER_PARTS:
                raise NotFoundError(f"no file {request.path_params['path']}")
            root = VIEWER_STATIC / part
        found = find_static(root, rel)
        if found is None:
            raise NotFoundError(f"no file {request.path_params['path']}")
        target, media = found
        return FileResponse(target, media_type=media, headers={"Cache-Control": "no-cache"})

    async def health(request: Request) -> Response:
        return JSONResponse(await run_in_threadpool(service.health, runner.counts(),
                                                    runner.in_progress()))

    async def openapi_doc(request: Request) -> Response:
        return JSONResponse(openapi.document(service.ops))

    async def run_op(request: Request) -> Response:
        """Validate, wait for the request's turn, run the command, answer when it ends."""
        op = op_of(request)
        params = await body_json(request)
        ticket, arrived = runner.ticket(), time.time()  # its place in the arrival order
        try:
            if runner.stopping:
                return refuse(STOPPING, "the service is stopping; send the request again once it "
                              "runs")
            prep = await run_in_threadpool(service.prepare, op, params)
            if prep.problems:  # refused: the uploads it was given are consumed all the same
                runner.discard(prep.uploads)
                status, body = error_body(prep.problems)
                return JSONResponse(body, status)
            run = Run(ticket, op.program.prog, op.module, prep.argv, prep.inference, prep.writes,
                      prep.uploads, op.id, prep.command, arrived)
            try:
                runner.admit(run)
            except RunError as exc:  # its uploads are consumed, but those another request uses
                runner.discard(prep.uploads)
                return refuse(exc.code, str(exc))
        finally:
            runner.arrived(ticket)  # validated: the requests after it no longer wait for it
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
            return refuse("too_large", f"an upload may hold at most "
                          f"{_gib(service.max_upload_bytes)}")
        if shutil.disk_usage(ws.uploads).free - size < service.min_free_bytes:
            return refuse("insufficient_storage", f"not enough free space in {ws.root} for this "
                          "upload; free some space or use a path inside the workspace")
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
            return refuse("upload_in_use", f"upload {uid} is the input of a request in progress; "
                          "it is deleted when that request ends")
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
        Route(UPLOADS, upload, methods=["POST", "PUT"]),
        Route("/api/uploads/{id}", delete_upload, methods=["DELETE"]),
        Route("/api/maps", maps),
        Route("/api/maps/{name}", map_detail),
    ]

    async def not_found(request: Request, exc: Exception) -> Response:
        return refuse("not_found", str(exc))

    async def no_route(request: Request, exc: Exception) -> Response:
        """No route has this path (404), or it does not take this method (405, with ``Allow``):
        the error shape of every other answer."""
        assert isinstance(exc, HTTPException)
        what = f"{request.method} {request.url.path}"
        if exc.status_code == 405:
            return refuse("method_not_allowed", f"{what}: the route takes "
                          f"{(exc.headers or {}).get('Allow', '')}; see /api/openapi.json",
                          exc.headers)
        return refuse("not_found", f"no route {what}; see /api/openapi.json")

    async def command_error(request: Request, exc: Exception) -> Response:
        """A command's error, by the generic exit status → HTTP status rule; anything else that
        fails (a corrupt ``map.json`` a map's route reads) is the internal error, told as a
        command tells it."""
        if isinstance(exc, OhMySlamError):
            code, message = ExitCode(exc.exit_code), str(exc)
        else:
            code, message = ExitCode.INTERNAL, internal_message(exc)
        return _error(HTTP_STATUS[code], error_code(code), message)

    async def disconnect(request: Request, exc: Exception) -> Response:
        return Response(status_code=CLIENT_GONE)

    app = Starlette(routes=routes, exception_handlers={
        NotFoundError: not_found, OhMySlamError: command_error, ClientDisconnect: disconnect,
        HTTPException: no_route, Exception: command_error})
    return guard(app, service)
