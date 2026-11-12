"""The Starlette application of ``server.sh`` (api.md): routes from ``routes.ENDPOINTS``, request
bodies validated against ``api.schema.json``, typed error bodies, and a ``Server-Timing`` header
on every response."""

from __future__ import annotations

import contextlib
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
from oh_my_slam.service import birdseye as birdseye_read
from oh_my_slam.service import nav, openapi, schema
from oh_my_slam.service import objects as objects_read
from oh_my_slam.service.cameras import decode_mask, decode_png16, decode_rgb, same_camera
from oh_my_slam.service.derived import Derived, DerivedVersion
from oh_my_slam.service.errors import ApiError, invalid
from oh_my_slam.service.jobs import Jobs
from oh_my_slam.service.prior_first import camera_only, classes
from oh_my_slam.service.routes import ENDPOINTS
from oh_my_slam.service.schema import SCHEMA_VERSION
from oh_my_slam.service.store import Store, check_name
from oh_my_slam.service.timings import Timings
from oh_my_slam.service.update import FrameWork, UpdateWork

log = get_logger("oh_my_slam.service.app")

CAPABILITIES = ["O1", "O2", "O3-jobs", "O4", "O5", "O6", "O8", "O9", "O10"]  # the groups this service
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
    dry_run: Callable[[Any], dict[str, Any]] | None = None
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


def _usable(image: bytes, cam: dict[str, Any]) -> bool:
    """Whether a frame's image decodes as RGB at its camera's size."""
    try:
        rgb = decode_rgb(image, "image")
    except ApiError:
        return False
    return bool(rgb.shape[:2] == (cam["height"], cam["width"]))


def update_work(store: Store, name: str, req: dict[str, Any], parts: dict[str, bytes],
                pending: dict[str, dict[str, Any]] | None = None) -> UpdateWork:
    """The job of a valid ``UpdateRequest``, with the cameras (and ignore masks) it adds to the
    map, recorded when the job commits; ``invalid_request`` or ``camera_conflict`` otherwise. A
    frame may name a camera that an earlier job of the map, still queued or running, adds
    (``pending``): the job fails if that one does not commit it."""
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
    cameras = {**(pending or {}), **known, **new}
    seen: set[str] = set()
    frames = []
    for i, f in enumerate(req["frames"]):
        field = f"frames[{i}]"
        # a repeated name is the frame's input status, not an error (api.md §3-§4)
        reject = "duplicate" if f["name"] in seen else None
        seen.add(f["name"])
        cam = cameras.get(f["camera_id"])
        if cam is None:
            raise invalid(f"{field}.camera_id", f"{field}.camera_id {f['camera_id']!r} is "
                          "neither in the request's cameras nor known to the map")
        image = part(f"{field}.image", f["image"])
        if reject is None and not _usable(image, cam):
            reject = "unusable_image"  # it does not decode, or is not its camera's size
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
                                f.get("height_above_floor_m"), f.get("height_sigma_m"), frame_mask,
                                reject))
    return UpdateWork(name, frames, req.get("accept_if"), new, masks,
                      [str(lab).strip() for lab in req.get("labels") or []])


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
            # no lanes: one lane, the service's single job worker (inference.md)
            "lanes": {"worker": {"state": "busy" if running else "idle",
                                 "queued": len(queued),
                                 "oldest_s": None if oldest is None else round(oldest, 3)}}})

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
        try:
            schema.validate("UpdateRequest", req)
        except ApiError as exc:
            # a label is refused as the field labels (api.md §3), never as a frame status
            if str(exc.details.get("field", "")).startswith("labels"):
                raise invalid("labels", exc.message, schema="UpdateRequest") from exc
            raise
        pending: dict[str, dict[str, Any]] = {}
        for j in self.s.jobs.active_jobs(name):
            pending.update(getattr(j.work, "cameras", None) or {})
        work = await run_in_threadpool(update_work, self.s.store, name, req, parts,
                                       None if dry else pending)
        _stage(request, "validate", t0)
        if dry:
            assert self.s.dry_run is not None
            t0 = time.perf_counter()
            out = await run_in_threadpool(self.s.dry_run, work)
            _stage(request, "dry_run", t0)
            return body(out)
        latest = self.s.store.latest(name)
        keyframes = len(self.s.store.version(name, latest)["frames"]) if latest else 0
        job = self.s.jobs.submit(name, work, len(work.frames), camera_only(work), classes(work),
                                 keyframes)
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
        self.s.store.require(name)
        n = self._hold(request, name, n)
        return name, self.s.store.require(name), self.s.store.version(name, n)

    def _hold(self, request: Request, name: str, n: int | None) -> int:
        """Retention keeps version ``n`` (None: the latest, read and held in one step) until the
        request ends (map.md §2 Holds, ``_wrap``); returns it."""
        stack = request.scope.setdefault("oms_holds", contextlib.ExitStack())
        if n is not None:
            stack.enter_context(self.s.store.hold(name, n))
            return n
        latest = stack.enter_context(self.s.store.hold_latest(name, required=True))
        assert latest is not None  # else map_not_found
        return int(latest)

    async def get_map(self, request: Request) -> Response:
        name, meta, v = self._version(request)
        return body(self._summary(name, meta, v))

    def _summary(self, name: str, meta: dict[str, Any], v: dict[str, Any]) -> dict[str, Any]:
        """A version's ``MapSummary`` (without ``schema``)."""
        job = self.s.jobs.active_for(name)
        return ({
            "map": name, "map_version": v["version"], "latest_version": meta["latest_version"],
            "created_at": meta["created_at"], "frames": len(v["frames"]),
            "provisional_frames": sum(f["pose_state"] == "provisional" for f in v["frames"]),
            "objects": v["objects"], "cameras": v.get("cameras", meta["cameras"]),
            "bounds_m": v["bounds_m"],
            "floor_z_m": v["floor_z_m"], "running_job": None if job is None else job.id,
            # the version's label vocabulary (oms/2.3, map.md §5), without the defaults
            "labels": list(v.get("labels") or []),
            "scale_source": v["scale_source"], "scale_sigma_rel": v["scale_sigma_rel"]})

    async def get_versions(self, request: Request) -> Response:
        name = check_name(request.path_params["name"])
        if "version" in request.query_params:
            raise invalid("version", "GET …/versions takes no ?version: it lists every kept "
                          "version, and latest_version names the newest")
        meta = self.s.store.require(name)
        versions = await run_in_threadpool(self.s.store.versions, name)
        return body({"map": name, "latest_version": meta["latest_version"],
                     "versions": versions})

    async def post_revert(self, request: Request) -> Response:
        name = check_name(request.path_params["name"])
        self.s.store.require(name)
        data = await read_body(request, self.s.upload_cap)
        try:
            req = json.loads(data)
        except (ValueError, UnicodeDecodeError) as exc:
            raise invalid("body", f"the body is not JSON ({exc})") from exc
        schema.validate("RevertRequest", req)
        with self.s.jobs.holding(name):  # map_busy while a job registers or fuses
            n = await run_in_threadpool(self.s.store.revert, name, int(req["to_version"]))
        self._hold(request, name, n)
        return body(self._summary(name, self.s.store.require(name),
                                  self.s.store.version(name, n)))

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
        # the height_m layer's ceiling (map.md §6): as asked, else the band's z_max when the
        # request gives one, else the default
        cap = nav.number(q, "height_cap_m", z_max if "z_max" in q else nav.HEIGHT_CAP_M)
        assert cell is not None and z_min is not None and z_max is not None and cap is not None
        if z_max <= z_min:
            raise invalid("z_max", f"z_max ({z_max}) must be above z_min ({z_min})")
        if cap <= 0:
            raise invalid("height_cap_m", f"height_cap_m ({cap}) must be above the floor (> 0)")
        dv = await run_in_threadpool(self._derived().get, name, v["version"])
        _stage(request, "derive", t0)
        t1 = time.perf_counter()
        answer = await run_in_threadpool(nav.grid_answer, dv, self._ids(), cell, z_min, z_max,
                                         foot, cap)
        out = body(answer.body)
        _stage(request, "grid", t1)
        # what the client waits for: the whole read, the body serialised, a cold read too
        self.s.timings.add("grid_read", time.perf_counter() - t0)
        return out

    async def get_objects(self, request: Request) -> Response:
        t0 = time.perf_counter()
        name, _, v = self._version(request)
        derived = self._derived()
        dv = await run_in_threadpool(derived.get, name, v["version"])
        got = await run_in_threadpool(objects_read.body, self.s.store, derived, self._ids(), dv)
        out = body(got)
        _stage(request, "objects", t0)
        self.s.timings.add("objects_read", time.perf_counter() - t0)  # as the client waits
        return out

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

    async def _birdseye(self, request: Request) -> tuple[bytes, dict[str, Any]]:
        t0 = time.perf_counter()
        name, _, v = self._version(request)
        derived = self._derived()
        dv = await run_in_threadpool(derived.get, name, v["version"])
        objects = await run_in_threadpool(objects_read.body, self.s.store, derived, self._ids(),
                                          dv)
        out = await run_in_threadpool(birdseye_read.read, dv, objects)
        _stage(request, "birdseye", t0)
        return out

    async def get_birdseye_png(self, request: Request) -> Response:
        png, _ = await self._birdseye(request)
        return Response(png, media_type="image/png")

    async def get_birdseye(self, request: Request) -> Response:
        _, meta = await self._birdseye(request)
        return body(meta)

    async def post_locate(self, request: Request) -> Response:
        name = check_name(request.path_params["name"])
        self.s.store.require(name)
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
        # the version it answers from, held from now until it ends (map.md §2 Holds)
        n = self._hold(request, name, req.get("version"))
        frames = await run_in_threadpool(nav.locate_frames, self.s.store, name, req, parts)
        v = self.s.store.version(name, n)
        dv = await run_in_threadpool(self._derived().get, name, v["version"])
        _stage(request, "prepare", t0)
        t0 = time.perf_counter()
        results = await run_in_threadpool(nav.locate, self._derived(), dv, frames)
        dt = time.perf_counter() - t0
        _stage(request, "locate", t0)
        self.s.timings.add("locate_per_frame", dt / len(frames), len(frames))
        located = [r for r in results if r["status"] == "located"]
        if req.get("record") and located:
            nav.record_pose(self.s.store, name, v["version"], located[-1], v["scale_source"],
                            v["scale_sigma_rel"])
        return body({"map": name, "map_version": v["version"],
                     "scale_source": v["scale_source"], "scale_sigma_rel": v["scale_sigma_rel"],
                     "frames": results})

    async def get_pose(self, request: Request) -> Response:
        name = check_name(request.path_params["name"])
        if "version" in request.query_params:
            raise invalid("version", "GET …/pose takes no ?version: the recorded pose lives "
                          "outside the versions, and its map_version is the version it was "
                          "located in")
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
        finally:
            holds = request.scope.pop("oms_holds", None)
            if holds is not None:  # the versions the read used (``Api._hold``)
                holds.close()
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
    if service.derived is not None:
        derived = service.derived

        def reads(dv: DerivedVersion, files: dict[str, Any]) -> None:
            """A version's default grid (5 cm) and objects answers, before it is published."""
            nav.grid_answer(dv, api._ids(), nav.DEFAULT_CELL_M, *nav.DEFAULT_Z_BAND, None,
                            nav.HEIGHT_CAP_M)
            objects_read.body(service.store, derived, api._ids(), dv, files)

        derived.prepared.append(reads)
    routes = [Route(e.path, _wrap(getattr(api, e.handler)), methods=[e.method])
              for e in ENDPOINTS]
    app = Starlette(routes=routes, exception_handlers={HTTPException: _http_error})
    return ServerTiming(app)
