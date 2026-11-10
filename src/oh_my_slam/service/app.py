"""The Starlette application of ``server.sh`` (api.md): routes from ``routes.ENDPOINTS``, request
bodies validated against ``api.schema.json``, typed error bodies, and a ``Server-Timing`` header
on every response."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from email.parser import BytesHeaderParser
from typing import Any, Protocol

import numpy as np
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from oh_my_slam.core.log import get_logger
from oh_my_slam.service import nav, openapi, schema
from oh_my_slam.service import objects as objects_read
from oh_my_slam.service.cameras import decode_mask, decode_png16, decode_rgb, same_camera
from oh_my_slam.service.derived import Derived
from oh_my_slam.service.errors import ApiError, invalid
from oh_my_slam.service.jobs import Jobs
from oh_my_slam.service.routes import ENDPOINTS
from oh_my_slam.service.schema import SCHEMA_VERSION
from oh_my_slam.service.store import Store, check_name
from oh_my_slam.service.timings import Timings
from oh_my_slam.service.update import FrameWork, UpdateWork

log = get_logger("oh_my_slam.service.app")

CAPABILITIES = ["O1", "O3-jobs", "O4", "O5", "O6", "O8", "O9", "O10"]  # the groups this service
# implements (high_level_spec.md §12)
UPLOAD_CAP_BYTES = 512 * 1024 * 1024  # one request's body
FREE_DISK_FACTOR = 3  # an upload needs this many times its size free on the data disk
ROTATION_TOLERANCE = 1e-3


class InferenceState(Protocol):
    def state(self) -> dict[str, Any]: ...


@dataclass
class Service:
    store: Store
    jobs: Jobs
    inference: InferenceState
    # ``inference_unavailable`` unless the inference server is ready (``inference.inference_ready``)
    require_inference: Callable[[], None]
    timings: Timings = field(default_factory=Timings)
    version: str = "unknown"
    upload_cap: int = UPLOAD_CAP_BYTES
    stopping: bool = False
    started_at: float = field(default_factory=time.time)
    # the dry run of an update request (``Updater.dry_run``), None: not served
    dry_run: Callable[[Any, dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None
    derived: Derived | None = None  # the grid, points and locate inputs of each version
    ids: nav.Ids | None = None


def body(data: dict[str, Any], status: int = 200, headers: dict[str, str] | None = None
         ) -> JSONResponse:
    return JSONResponse({"schema": SCHEMA_VERSION, **data}, status_code=status, headers=headers)


def error_response(exc: ApiError) -> JSONResponse:
    return JSONResponse({"schema": SCHEMA_VERSION, "error": exc.body()}, status_code=exc.status)


# -- Server-Timing -------------------------------------------------------------------------------


class ServerTiming:
    """Adds ``Server-Timing`` to every response: the stages a handler recorded in
    ``scope["oms_timings"]`` (name -> seconds) and the total."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":  # lifespan
            await self.app(scope, receive, send)
            return
        t0 = time.perf_counter()
        stages: dict[str, float] = {}
        scope["oms_timings"] = stages

        async def timed(message: Message) -> None:
            if message["type"] == "http.response.start":
                parts = [f"{k};dur={v * 1000:.1f}" for k, v in stages.items()]
                parts.append(f"total;dur={(time.perf_counter() - t0) * 1000:.1f}")
                message.setdefault("headers", []).append(
                    (b"server-timing", ", ".join(parts).encode()))
            await send(message)

        await self.app(scope, receive, timed)


def _stage(request: Request, name: str, t0: float) -> None:
    request.scope.setdefault("oms_timings", {})[name] = time.perf_counter() - t0


# -- request bodies ------------------------------------------------------------------------------


async def read_body(request: Request, cap: int) -> bytes:
    """The body, ``payload_too_large`` beyond ``cap`` bytes."""
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > cap:
        raise ApiError("payload_too_large", f"the request is {int(declared)} bytes; the cap is "
                       f"{cap} bytes — send fewer frames per update", {"cap_bytes": cap})
    chunks, total = [], 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > cap:
            raise ApiError("payload_too_large", f"the request exceeds the cap of {cap} bytes — "
                           "send fewer frames per update", {"cap_bytes": cap})
        chunks.append(chunk)
    return b"".join(chunks)


def parse_multipart(content_type: str, data: bytes) -> dict[str, bytes]:
    """The parts of a ``multipart/form-data`` body by name."""
    header = BytesHeaderParser().parsebytes(b"Content-Type: " + content_type.encode("latin-1")
                                            + b"\r\n\r\n")
    boundary = header.get_param("boundary")
    if header.get_content_type() != "multipart/form-data" or not isinstance(boundary, str):
        raise invalid("body", "the body must be multipart/form-data with a boundary: a part "
                      "named 'request' (the JSON body) and one part per file")
    delim = b"\r\n--" + boundary.encode("latin-1")
    sections = (b"\r\n" + data).split(delim)
    if len(sections) < 2 or not sections[-1].startswith(b"--"):
        raise invalid("body", "malformed multipart body (no closing boundary)")
    parts: dict[str, bytes] = {}
    for sec in sections[1:-1]:
        head, sep, payload = sec.partition(b"\r\n\r\n")
        if not sep:
            raise invalid("body", "malformed multipart part (no header end)")
        h = BytesHeaderParser().parsebytes(head.lstrip(b"\r\n") + b"\r\n\r\n")
        name = h.get_param("name", header="content-disposition")
        if not isinstance(name, str) or not name:
            raise invalid("body", "a multipart part has no name")
        if name in parts:
            raise invalid(name, f"the part {name!r} is sent twice")
        parts[name] = payload
    return parts


def _rotation_ok(T: list[list[float]]) -> bool:
    M = np.asarray(T, np.float64)
    R = M[:3, :3]
    return (bool(np.allclose(M[3], [0, 0, 0, 1], atol=ROTATION_TOLERANCE))
            and bool(np.allclose(R.T @ R, np.eye(3), atol=ROTATION_TOLERANCE))
            and float(np.linalg.det(R)) > 0 and bool(np.isfinite(M).all()))


def update_work(store: Store, name: str, req: dict[str, Any], parts: dict[str, bytes]
                ) -> tuple[UpdateWork, dict[str, dict[str, Any]], dict[str, tuple[str, bytes]]]:
    """The job of a valid ``UpdateRequest`` and the cameras (with ignore masks) it adds to the
    map; ``invalid_request`` or ``camera_conflict`` otherwise."""
    known = store.cameras(name)
    new: dict[str, dict[str, Any]] = {}
    masks: dict[str, tuple[str, bytes]] = {}

    def part(field: str, part_name: str) -> bytes:
        if part_name not in parts:
            raise invalid(field, f"{field}: no multipart part named {part_name!r}")
        return parts[part_name]

    for cid, cam in (req.get("cameras") or {}).items():
        field = f"cameras.{cid}"
        digest = None
        if "ignore_mask" in cam:
            data = part(f"{field}.ignore_mask", cam["ignore_mask"])
            mask = decode_mask(data, f"{field}.ignore_mask")
            if mask.shape != (cam["height"], cam["width"]):
                raise invalid(f"{field}.ignore_mask", f"{field}.ignore_mask is "
                              f"{mask.shape[1]}x{mask.shape[0]}, the camera "
                              f"{cam['width']}x{cam['height']}")
            digest = hashlib.sha256(data).hexdigest()
        if cid in known:
            if not same_camera(known[cid], cam) or (
                    digest is not None and digest != store.mask_digest(name, cid)):
                raise ApiError("camera_conflict", f"camera {cid!r} of map {name!r} is already "
                               "defined with other values: omit it, or use a new camera id",
                               {"field": field, "camera_id": cid, "known": known[cid]})
            continue
        if digest is not None:
            masks[cid] = (digest, parts[cam["ignore_mask"]])
        new[cid] = cam
    cameras = {**known, **new}
    seen: set[str] = set()
    frames = []
    for i, f in enumerate(req["frames"]):
        field = f"frames[{i}]"
        if f["name"] in seen:
            raise invalid(f"{field}.name", f"{field}.name {f['name']!r} is not unique in the "
                          "request")
        seen.add(f["name"])
        cam = cameras.get(f["camera_id"])
        if cam is None:
            raise invalid(f"{field}.camera_id", f"{field}.camera_id {f['camera_id']!r} is "
                          "neither in the request's cameras nor known to the map")
        image = part(f"{field}.image", f["image"])
        rgb = decode_rgb(image, f"{field}.image")
        if rgb.shape[:2] != (cam["height"], cam["width"]):
            raise invalid(f"{field}.image", f"{field}.image is {rgb.shape[1]}x{rgb.shape[0]}; "
                          f"camera {f['camera_id']!r} is {cam['width']}x{cam['height']}")
        depth = None
        if "depth" in f:
            depth = part(f"{field}.depth", f["depth"])
            d = decode_png16(depth, f"{field}.depth")
            if d.shape != (cam["height"], cam["width"]):
                raise invalid(f"{field}.depth", f"{field}.depth is {d.shape[1]}x{d.shape[0]}; "
                              f"camera {f['camera_id']!r} is {cam['width']}x{cam['height']}")
        frame_mask = None
        if "ignore_mask" in f:
            frame_mask = part(f"{field}.ignore_mask", f["ignore_mask"])
            m = decode_mask(frame_mask, f"{field}.ignore_mask")
            if m.shape != (cam["height"], cam["width"]):
                raise invalid(f"{field}.ignore_mask", f"{field}.ignore_mask is "
                              f"{m.shape[1]}x{m.shape[0]}; camera {f['camera_id']!r} is "
                              f"{cam['width']}x{cam['height']}")
        prior = f.get("prior")
        if prior is not None and "T_map_cam" in prior and not _rotation_ok(prior["T_map_cam"]):
            raise invalid(f"{field}.prior.T_map_cam", f"{field}.prior.T_map_cam is not a rigid "
                          "transform (orthonormal rotation, last row 0 0 0 1)")
        frames.append(FrameWork(i, f["name"], f["camera_id"], image, depth, prior,
                                f.get("height_above_floor_m"), f.get("height_sigma_m"), frame_mask))
    return UpdateWork(name, frames, req.get("accept_if")), new, masks


# -- the API -------------------------------------------------------------------------------------


class Api:
    def __init__(self, service: Service) -> None:
        self.s = service

    async def health(self, request: Request) -> Response:
        inference = await run_in_threadpool(self.s.inference.state)
        running = self.s.jobs.running
        queued = self.s.jobs.queued()
        now = time.time()
        oldest = None
        if running:
            oldest = now - self.s.jobs.get(running).created
        elif queued:
            oldest = now - queued[0].created
        return body({
            "version": self.s.version, "service": "stopping" if self.s.stopping else "ready",
            "inference": inference, "capabilities": CAPABILITIES,
            "timings": self.s.timings.body(),
            "lanes": {"interactive": {"state": "busy" if running else "idle",
                                      "queued": len(queued),
                                      "oldest_s": None if oldest is None else round(oldest, 3)},
                      "bulk": {"state": "idle", "queued": 0, "oldest_s": None}}})

    async def openapi(self, request: Request) -> Response:
        return JSONResponse(openapi.build(self.s.version))

    async def post_update(self, request: Request) -> Response:
        name = check_name(request.path_params["name"])
        await run_in_threadpool(self.s.require_inference)
        dry = request.query_params.get("dry_run") not in (None, "0", "false")
        if dry and self.s.dry_run is None:
            raise invalid("dry_run", "dry runs (?dry_run=1) are not served by this service")
        t0 = time.perf_counter()
        data = await read_body(request, self.s.upload_cap)
        free = shutil.disk_usage(self.s.store.root).free
        if free < FREE_DISK_FACTOR * len(data):
            raise ApiError("payload_too_large", f"only {free} bytes are free on the data disk "
                           "for this upload — free disk space or send fewer frames",
                           {"free_bytes": free})
        parts = parse_multipart(request.headers.get("content-type", ""), data)
        _stage(request, "upload", t0)
        t0 = time.perf_counter()
        if "request" not in parts:
            raise invalid("request", "no multipart part named 'request' (the JSON body)")
        try:
            req = json.loads(parts["request"])
        except (ValueError, UnicodeDecodeError) as exc:
            raise invalid("request", f"the 'request' part is not JSON ({exc})") from exc
        schema.validate("UpdateRequest", req)
        work, cameras, masks = await run_in_threadpool(update_work, self.s.store, name, req,
                                                       parts)
        _stage(request, "validate", t0)
        if dry:
            assert self.s.dry_run is not None
            t0 = time.perf_counter()
            out = await run_in_threadpool(self.s.dry_run, work, cameras, masks)
            _stage(request, "dry_run", t0)
            return body(out)
        if cameras or masks or self.s.store.meta(name) is None:
            self.s.store.add_cameras(name, cameras, masks)
        job = self.s.jobs.submit(name, work, len(work.frames))
        return JSONResponse(job.queued_view, 202, {"Location": f"/api/jobs/{job.id}"})

    async def get_job(self, request: Request) -> Response:
        raw = request.query_params.get("wait_s")
        wait = 0.0
        if raw is not None:
            try:
                wait = float(raw)
            except ValueError:
                wait = math.nan
            if not 0 <= wait <= 30:
                raise invalid("wait_s", "wait_s must be a number of seconds between 0 and 30")
        job_id = request.path_params["id"]
        job = await run_in_threadpool(self.s.jobs.wait, job_id, wait) if wait else \
            self.s.jobs.get(job_id)
        return JSONResponse(self.s.jobs.view(job))

    def _version(self, request: Request) -> tuple[str, dict[str, Any], dict[str, Any]]:
        name = check_name(request.path_params["name"])
        raw = request.query_params.get("version")
        n = None
        if raw is not None:
            if not raw.isdigit() or int(raw) < 1:
                raise invalid("version", "version must be a positive integer")
            n = int(raw)
        meta = self.s.store.require(name)
        return name, meta, self.s.store.version(name, n)

    async def get_map(self, request: Request) -> Response:
        name, meta, v = self._version(request)
        job = self.s.jobs.active_for(name)
        return body({
            "map": name, "map_version": v["version"], "latest_version": meta["latest_version"],
            "created_at": meta["created_at"], "frames": len(v["frames"]),
            "provisional_frames": sum(f["pose_state"] == "provisional" for f in v["frames"]),
            "objects": v["objects"], "cameras": meta["cameras"], "bounds_m": v["bounds_m"],
            "floor_z_m": v["floor_z_m"], "running_job": None if job is None else job.id,
            "scale_source": v["scale_source"], "scale_sigma_rel": v["scale_sigma_rel"]})

    async def get_frames(self, request: Request) -> Response:
        name, _, v = self._version(request)
        keys = ("name", "camera_id", "T_map_cam", "sigma_pos_m", "sigma_rot_deg", "pose_state",
                "added_version")
        return body({"map": name, "map_version": v["version"],
                     "scale_source": v["scale_source"], "scale_sigma_rel": v["scale_sigma_rel"],
                     "frames": [{k: f[k] for k in keys} for f in v["frames"]]})

    def _derived(self) -> Derived:
        assert self.s.derived is not None, "the service derives versions (main.serve)"
        return self.s.derived

    def _ids(self) -> nav.Ids:
        if self.s.ids is None:
            self.s.ids = nav.Ids(self.s.store)
        return self.s.ids

    async def get_grid(self, request: Request) -> Response:
        t0 = time.perf_counter()
        name, _, v = self._version(request)
        q = request.query_params
        cell = nav.number(q, "cell_m", nav.DEFAULT_CELL_M, nav.MIN_CELL_M, nav.MAX_CELL_M)
        z_min = nav.number(q, "z_min", nav.DEFAULT_Z_BAND[0])
        z_max = nav.number(q, "z_max", nav.DEFAULT_Z_BAND[1])
        foot = nav.number(q, "footprint_radius_m", None, 0.0)
        assert cell is not None and z_min is not None and z_max is not None
        if z_max <= z_min:
            raise invalid("z_max", f"z_max ({z_max}) must be above z_min ({z_min})")
        dv = await run_in_threadpool(self._derived().get, name, v["version"])
        _stage(request, "derive", t0)
        t1 = time.perf_counter()
        answer = await run_in_threadpool(nav.grid, dv, self._ids(), cell, z_min, z_max, foot)
        _stage(request, "grid", t1)
        self.s.timings.add("grid_read", time.perf_counter() - t0)
        return body(answer.body)

    async def get_objects(self, request: Request) -> Response:
        t0 = time.perf_counter()
        name, _, v = self._version(request)
        derived = self._derived()
        dv = await run_in_threadpool(derived.get, name, v["version"])
        out = await run_in_threadpool(objects_read.body, self.s.store, derived, self._ids(), dv)
        _stage(request, "objects", t0)
        self.s.timings.add("objects_read", time.perf_counter() - t0)
        return body(out)

    async def get_points(self, request: Request) -> Response:
        name, _, v = self._version(request)
        q = request.query_params
        lo, hi = nav.vec3(q, "min"), nav.vec3(q, "max")
        if (hi < lo).any():
            raise invalid("max", "max must be at least min on every axis")
        voxel = nav.number(q, "voxel_m", nav.DEFAULT_VOXEL_M, 0.001, 1.0)
        cap = nav.number(q, "max_points", nav.DEFAULT_MAX_POINTS, 1, nav.MAX_POINTS)
        assert voxel is not None and cap is not None
        if cap != int(cap):
            raise invalid("max_points", "max_points must be an integer")
        dv = await run_in_threadpool(self._derived().get, name, v["version"])
        return body(await run_in_threadpool(nav.points, dv, lo, hi, voxel, int(cap)))

    async def post_locate(self, request: Request) -> Response:
        name = check_name(request.path_params["name"])
        meta = self.s.store.require(name)
        t0 = time.perf_counter()
        data = await read_body(request, self.s.upload_cap)
        parts = parse_multipart(request.headers.get("content-type", ""), data)
        _stage(request, "upload", t0)
        t0 = time.perf_counter()
        if "request" not in parts:
            raise invalid("request", "no multipart part named 'request' (the JSON body)")
        try:
            req = json.loads(parts["request"])
        except (ValueError, UnicodeDecodeError) as exc:
            raise invalid("request", f"the 'request' part is not JSON ({exc})") from exc
        schema.validate("LocateRequest", req)
        frames = await run_in_threadpool(nav.locate_frames, self.s.store, name, req, parts)
        v = self.s.store.version(name, req.get("version") or int(meta["latest_version"]))
        dv = await run_in_threadpool(self._derived().get, name, v["version"])
        _stage(request, "prepare", t0)
        t0 = time.perf_counter()
        results = await run_in_threadpool(nav.locate, self._derived(), dv, frames)
        dt = time.perf_counter() - t0
        _stage(request, "locate", t0)
        self.s.timings.add("locate_per_frame", dt / len(frames), len(frames))
        located = [r for r in results if r["status"] == "located"]
        if req.get("record") and located:
            nav.record_pose(self.s.store, name, v["version"], located[-1])
        return body({"map": name, "map_version": v["version"],
                     "scale_source": v["scale_source"], "scale_sigma_rel": v["scale_sigma_rel"],
                     "frames": results})

    async def get_pose(self, request: Request) -> Response:
        name = check_name(request.path_params["name"])
        self.s.store.require(name)
        return body({"map": name, **nav.recorded_pose(self.s.store, name)})

    async def delete_map(self, request: Request) -> Response:
        name = check_name(request.path_params["name"])
        job = self.s.jobs.active_for(name)
        if job is not None:
            raise ApiError("map_busy", f"job {job.id} of map {name!r} is {job.state}; delete the "
                           "map once it has ended", {"job_id": job.id, "state": job.state})
        if not await run_in_threadpool(self.s.store.delete, name):
            raise ApiError("map_not_found", f"no map named {name!r}", {"map": name})
        return Response(status_code=204)


def _wrap(handler: Callable[[Request], Awaitable[Response]]
          ) -> Callable[[Request], Awaitable[Response]]:
    async def run(request: Request) -> Response:
        try:
            return await handler(request)
        except ApiError as exc:
            return error_response(exc)
        except Exception as exc:
            log.exception("%s %s failed", request.method, request.url.path)
            return error_response(ApiError("internal", f"internal error: {type(exc).__name__}: "
                                           f"{exc}", {"stage": handler.__name__}))
    return run


async def _http_error(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, HTTPException)
    what = "no such endpoint" if exc.status_code == 404 else f"HTTP {exc.status_code}"
    if exc.status_code == 405:
        what = f"{request.method} is not allowed on {request.url.path}"
    return error_response(ApiError("invalid_request", f"{what}: see /api/openapi.json",
                                   {"field": "path", "path": request.url.path},
                                   status=exc.status_code))


def create_app(service: Service) -> ASGIApp:
    api = Api(service)
    if service.derived is not None:  # each version's objects answer, computed once derived
        derived = service.derived
        derived.after.append(lambda dv: objects_read.body(service.store, derived, api._ids(), dv))
    routes = [Route(e.path, _wrap(getattr(api, e.handler)), methods=[e.method])
              for e in ENDPOINTS]
    app = Starlette(routes=routes, exception_handlers={HTTPException: _http_error})
    return ServerTiming(app)
