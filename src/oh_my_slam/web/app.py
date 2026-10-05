"""The HTTP routes of ``server.sh`` (spec §2.6 "API"), on Starlette (served by uvicorn).

Everything under ``/api/`` is described by ``/api/openapi.json`` (``web.openapi``). The viewer's
data is served by the viewer's own code: ``/viewer/map/<name>/`` mounts ``viewer.routes.
ViewerRoutes`` over the map's read-only bundle in this process, and ``/viewer/job/<id>/`` is the
viewer a ``view.sh`` job opened (its own process, proxied). ``/`` is the web application's
placeholder page.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import threading
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
from oh_my_slam.web.operations import OUT_DIR, Operation, error_body, operations, prepare, problem
from oh_my_slam.web.workspace import NotFoundError, Workspace

START_COMMAND = "./start_inference_server.sh"
SSE_POLL_S = 0.2
SSE_HEARTBEAT_S = 15.0
_MEDIA = {".json": "application/json", ".ply": "application/octet-stream", ".png": "image/png",
          ".csv": "text/csv", ".md": "text/markdown", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message, "http_status": status}},
                        status)


def inference_health() -> dict[str, Any]:
    """The inference server's health, or why it is down and the command that starts it."""
    from oh_my_slam.client.client import InferenceClient

    try:
        h = InferenceClient().health(timeout=1.0)
    except ServerUnavailableError as exc:
        return {"status": "down", "message": str(exc), "start_command": START_COMMAND}
    return {"status": h.status, "health": h.model_dump(),
            "start_command": None if h.status == "ready" else START_COMMAND}


def inference_problem() -> spec.Problem | None:
    """The commands' own check of the inference server (exit 3 when it is down)."""
    from oh_my_slam.client.client import InferenceClient

    try:
        InferenceClient().require_ready(wait_loading_s=0.0)
    except OhMySlamError as exc:
        return problem((), str(exc), exc.exit_code, "inference_server")
    return None


@dataclass
class Service:
    """What the routes share: the workspace, the job runner and the service's own facts."""

    workspace: Workspace
    runner: Runner
    url: str = ""
    started_at: float = field(default_factory=time.time)
    inference_check: Callable[[], spec.Problem | None] = inference_problem
    inference_health: Callable[[], dict[str, Any]] = inference_health
    ops: dict[str, Operation] = field(default_factory=operations)
    _map_views: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    _map_lock: threading.Lock = field(default_factory=threading.Lock)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "service": {"version": __version__, "url": self.url, "pid": os.getpid(),
                        "workspace": self.workspace.root.name, "data": str(self.workspace.root),
                        "started_at": self.started_at, "jobs": self.runner.counts()},
            "inference": self.inference_health(),
        }

    # -- submission --------------------------------------------------------------------------------

    def submit(self, op: Operation, params: Any, resubmitted_from: str | None = None
               ) -> JSONResponse:
        jid = self.runner.new_id()
        prep = prepare(op, params, self.workspace, self.workspace.job_dir(jid))
        if not prep.problems and op.mode.inference == "required":
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

    def validate(self, op: Operation, params: Any) -> dict[str, Any]:
        prep = prepare(op, params, self.workspace, self.workspace.job_dir("validate"))
        if op.mode.inference == "required":
            p = self.inference_check()
            if p is not None:
                prep.problems.append(p)
        return {"valid": not prep.problems, "command": prep.command,
                "problems": [p.describe() for p in prep.problems],
                "by_parameter": spec.by_parameter(prep.problems)}

    # -- viewer of a map ---------------------------------------------------------------------------

    def map_routes(self, name: str) -> Any:
        """The viewer's routes over a map's read-only bundle, rebuilt when the map changed."""
        root = self.workspace.map_dir(name)
        stamp = (root / "map.json").stat().st_mtime_ns
        with self._map_lock:
            cached = self._map_views.get(name)
            if cached is not None and cached[0] == stamp:
                return cached[1]
            from oh_my_slam.viewer.bundle import map_bundle
            from oh_my_slam.viewer.routes import ViewerRoutes

            routes = ViewerRoutes(map_bundle(root))
            self._map_views[name] = (stamp, routes)
            return routes


def _viewer_response(r: Any, method: str) -> Response:
    """A ``viewer.routes.Response`` as a Starlette response (its pieces streamed in order)."""
    headers = dict(r.headers)
    if method == "HEAD":
        return Response(b"", r.status, headers=headers)

    async def body() -> AsyncIterator[bytes | memoryview]:
        for piece in r.body:
            yield piece

    return StreamingResponse(body(), r.status, headers=headers)


def create_app(service: Service) -> Starlette:
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
        return Response(PLACEHOLDER.format(url=service.url), media_type="text/html")

    async def health(request: Request) -> Response:
        return JSONResponse(await run_in_threadpool(service.health))

    async def openapi_doc(request: Request) -> Response:
        return JSONResponse(openapi.document(service.ops))

    async def describe(request: Request) -> Response:
        return JSONResponse(spec.describe())

    async def submit(request: Request) -> Response:
        op = op_of(request)
        params = await body_json(request)
        return await run_in_threadpool(service.submit, op, params)

    async def validate(request: Request) -> Response:
        op = op_of(request)
        params = await body_json(request)
        return JSONResponse(await run_in_threadpool(service.validate, op, params))

    # -- uploads -----------------------------------------------------------------------------------

    async def upload(request: Request) -> Response:
        name = request.query_params.get("name", "")
        uid, target = ws.new_upload(name)
        part = target.with_name(f".{target.name}.part")
        size = 0
        try:
            with part.open("wb") as f:
                async for chunk in request.stream():
                    f.write(chunk)
                    size += len(chunk)
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
        return await run_in_threadpool(service.submit, op, params, old.id)

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
        return FileResponse(p, media_type=media_of(j.id, p) if j.result_format is None else
                            spec.Output("result", "stdout", j.result_format, "").describe()[
                                "media_type"], filename=j.result_name)

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

    async def map_viewer(request: Request) -> Response:
        name, rest = request.path_params["name"], request.path_params.get("path", "")
        if "path" not in request.path_params:
            return RedirectResponse(f"/viewer/map/{name}/")
        routes = await run_in_threadpool(service.map_routes, name)
        r = await run_in_threadpool(routes.handle, request.method, "/" + rest,
                                    request.url.query)
        return _viewer_response(r, request.method)

    async def job_viewer(request: Request) -> Response:
        jid, rest = request.path_params["id"], request.path_params.get("path", "")
        j = runner.get(jid)
        if "path" not in request.path_params:
            return RedirectResponse(f"/viewer/job/{jid}/")
        base = runner.viewer_url(jid)
        if base is None:
            return _error(410 if j.state in TERMINAL else 409, "viewer_closed",
                          f"job {jid} has no open viewer ({j.state}); re-submit the job to "
                          "open it again")
        return await _proxy(request, base + rest)

    routes = [
        Route("/", index),
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
        Route("/api/jobs", jobs),
        Route("/api/jobs/events", events),
        Route("/api/jobs/{id}", job),
        Route("/api/jobs/{id}/events", events),
        Route("/api/jobs/{id}/cancel", cancel, methods=["POST"]),
        Route("/api/jobs/{id}/resubmit", resubmit, methods=["POST"]),
        Route("/api/jobs/{id}/result", result),
        Route("/api/jobs/{id}/files", files),
        Route("/api/jobs/{id}/files/{path:path}", job_file),
        Route("/api/jobs/{id}/log", log),
        Route("/api/jobs/{id}/timings", timings),
        Route("/viewer/map/{name}", map_viewer, methods=["GET", "HEAD"]),
        Route("/viewer/map/{name}/{path:path}", map_viewer, methods=["GET", "HEAD", "POST"]),
        Route("/viewer/job/{id}", job_viewer, methods=["GET", "HEAD"]),
        Route("/viewer/job/{id}/{path:path}", job_viewer, methods=["GET", "HEAD", "POST"]),
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

    return Starlette(routes=routes, exception_handlers={
        NotFoundError: not_found, JobError: job_error, OhMySlamError: command_error,
        ClientDisconnect: disconnect})


async def _proxy(request: Request, url: str) -> Response:
    """Forward a viewer request to the job's own viewer process (local, read-only)."""
    import httpx

    client = httpx.AsyncClient(timeout=None)
    try:
        upstream = await client.send(client.build_request(
            request.method, url, params=request.url.query or None), stream=True)
    except httpx.HTTPError as exc:
        await client.aclose()
        return _error(502, "viewer_unavailable", f"the job's viewer does not answer ({exc})")
    keep = {"content-type", "content-length", "cache-control", "allow"}
    headers = {k: v for k, v in upstream.headers.items() if k.lower() in keep}

    async def body() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    if request.method == "HEAD":
        await upstream.aclose()
        await client.aclose()
        return Response(b"", upstream.status_code, headers=headers)
    return StreamingResponse(body(), upstream.status_code, headers=headers)


PLACEHOLDER = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width, initial-scale=1"><title>oh-my-slam</title>
<style>:root{{color-scheme:light dark}}body{{font:16px/1.5 system-ui,sans-serif;margin:2rem;
max-width:48rem}}</style></head>
<body><h1>oh-my-slam</h1>
<p>The web service is running. Its API is described at
<a href="/api/openapi.json">/api/openapi.json</a>; health at <a href="/api/health">/api/health</a>,
jobs at <a href="/api/jobs">/api/jobs</a>, maps at <a href="/api/maps">/api/maps</a>.</p>
</body></html>
"""
