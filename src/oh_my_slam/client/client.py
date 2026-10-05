"""HTTP-over-Unix-socket client for the inference server.

Any failure to reach the server becomes :class:`ServerUnavailableError` (exit 3, with the hint to
run ``./start_inference_server.sh``). Queue-full responses (503) are retried with back-off.
Responses can be recorded and replayed (``client.replay``, set by the environment).
"""

from __future__ import annotations

import time
from contextlib import ExitStack
from pathlib import Path
from typing import TypeVar

import httpx
import numpy as np
from pydantic import BaseModel

from oh_my_slam.client import protocol as p
from oh_my_slam.client import replay
from oh_my_slam.client.images import request_image
from oh_my_slam.core import paths, timing
from oh_my_slam.core.errors import (
    InferenceError,
    InputError,
    ServerBusyError,
    ServerModelsFailedError,
    ServerUnavailableError,
)
from oh_my_slam.core.images import upright_size
from oh_my_slam.core.log import get_logger

M = TypeVar("M", bound=BaseModel)

HEALTH_TIMEOUT_S = 0.5
REQUEST_TIMEOUT_S = 900.0
BUSY_RETRY_S = 300.0
LOADING_WAIT_S = 180.0

log = get_logger("oh_my_slam.client")


class InferenceClient:
    def __init__(self, socket_path: Path | None = None, timeout: float = REQUEST_TIMEOUT_S) -> None:
        self.socket_path = Path(socket_path) if socket_path else paths.socket_path()
        self.timeout = timeout
        self._client: httpx.Client | None = None

    # -- plumbing ----------------------------------------------------------------------------------

    def _http(self) -> httpx.Client:
        if self._client is None:
            transport = httpx.HTTPTransport(uds=str(self.socket_path))
            self._client = httpx.Client(
                transport=transport, base_url="http://oh-my-slam", timeout=self.timeout
            )
        return self._client

    def clone(self) -> InferenceClient:
        """A separate client (own connection pool) for use from another thread."""
        return InferenceClient(self.socket_path, self.timeout)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> InferenceClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- health ------------------------------------------------------------------------------------

    def health(self, timeout: float = HEALTH_TIMEOUT_S) -> p.Health:
        if not self.socket_path.exists():
            raise ServerUnavailableError("no socket")
        try:
            r = self._http().get(p.ROUTE_HEALTH, timeout=timeout)
        except httpx.HTTPError as exc:
            raise ServerUnavailableError(type(exc).__name__) from exc
        if r.status_code != 200:
            raise ServerUnavailableError(f"health returned HTTP {r.status_code}")
        return p.Health.model_validate(r.json())

    def require_ready(self, wait_loading_s: float = LOADING_WAIT_S) -> p.Health:
        """Fail fast if the server is down; wait (bounded) while it is still loading."""
        h = self.health()
        deadline = time.monotonic() + wait_loading_s
        announced = False
        while h.status == "loading" and time.monotonic() < deadline:
            if not announced:
                log.info("inference server is still loading models; waiting")
                announced = True
            time.sleep(1.0)
            h = self.health()
        if h.status == "ready":
            return h
        if h.status == "error":
            failed = "; ".join(f"{m.name}: {m.error}" for m in h.models.values() if m.error)
            raise ServerModelsFailedError(failed or "no detail", str(paths.server_log()))
        raise ServerUnavailableError(f"server status '{h.status}'")

    # -- requests ----------------------------------------------------------------------------------

    def _post(self, route: str, req: BaseModel, model: type[M]) -> M:
        t0 = time.perf_counter()
        data = req.model_dump()
        if replay.replaying() is not None:  # answered from a recording, else forwarded
            recorded = replay.replay(route, data, data.get("out_dir"))
            if recorded is not None:
                return model.model_validate(recorded)
        deadline = time.monotonic() + BUSY_RETRY_S
        delay = 0.2
        while True:
            if not self.socket_path.exists():
                raise ServerUnavailableError("no socket")
            try:
                r = self._http().post(route, json=data)
            except httpx.TimeoutException as exc:
                raise InferenceError(f"{route} timed out after {self.timeout:.0f} s") from exc
            except httpx.HTTPError as exc:
                raise ServerUnavailableError(type(exc).__name__) from exc
            if r.status_code == 200:
                raw = r.json()
                replay.record(route, data, raw)
                out = model.model_validate(raw)
                t = getattr(out, "timings", None)
                timing.record_request(route, time.perf_counter() - t0,
                                      getattr(t, "queue_s", 0.0), getattr(t, "compute_s", 0.0))
                return out
            body = _error_body(r)
            if r.status_code == 503:
                if time.monotonic() > deadline:
                    raise ServerBusyError(f"{route}: server busy ({body})")
                time.sleep(delay)
                delay = min(delay * 1.5, 2.0)
                continue
            if r.status_code == 400:
                raise InputError(f"{route}: {body}")
            raise InferenceError(f"{route} failed: {body}")

    # The image of a request is sent at the size the server reads it (``client.images``); what
    # the server computes from the size of the file it reads is converted here, with the server's
    # own arithmetic, so every response is the one the original image gives.

    def geometry(self, req: p.GeometryRequest) -> p.GeometryResponse:
        with request_image(Path(req.image_path), req.max_side) as sent:
            res = self._post(p.ROUTE_GEOMETRY, req.model_copy(update={"image_path": str(sent.path)}),
                             p.GeometryResponse)
        if sent.downscaled:
            res.orig_width, res.orig_height = upright_size(Path(req.image_path))
        return res

    def gravity(self, req: p.GravityRequest) -> p.GravityResponse:
        with request_image(Path(req.image_path), p.GRAVITY_SIDE) as sent:
            sx = sent.scale[0]
            res = self._post(p.ROUTE_GRAVITY, p.GravityRequest(
                image_path=str(sent.path),
                focal_px=None if req.focal_px is None or not sent.downscaled
                else req.focal_px * sx), p.GravityResponse)
        if sent.downscaled:  # the server's focal lengths are in pixels of the file it read
            res.focal_px = res.focal_px / sx
            res.focal_unc_px = res.focal_unc_px / sx
        return res

    def segment_image(self, req: p.SegmentRequest) -> p.SegmentResponse:
        with request_image(Path(req.image_path), req.max_side) as sent:
            return self._post(p.ROUTE_SEGMENT,
                              req.model_copy(update={"image_path": str(sent.path)}),
                              p.SegmentResponse)

    def multiview(self, req: p.MultiviewRequest) -> p.MultiviewResponse:
        with ExitStack() as stack:
            sent = [stack.enter_context(request_image(Path(i), p.MULTIVIEW_SIDE))
                    for i in req.image_paths]
            K = req.intrinsics
            if K is not None:  # the server scales them to the file it reads, in float32
                K = [k if k is None or not s.downscaled else
                     _scaled_intrinsics(k, s.scale) for k, s in zip(K, sent, strict=True)]
            return self._post(p.ROUTE_MULTIVIEW, req.model_copy(update={
                "image_paths": [str(s.path) for s in sent], "intrinsics": K}),
                p.MultiviewResponse)


def _scaled_intrinsics(K: list[list[float]], scale: tuple[float, float]) -> list[list[float]]:
    """A full-resolution 3x3 ``K`` in pixels of the downscaled file, as the multi-view server
    scales it for the image it reads (float32 rows), so that it reads it unchanged."""
    k = np.asarray(K, dtype=np.float32).copy()
    k[0] *= scale[0]
    k[1] *= scale[1]
    return [[float(v) for v in row] for row in k]


def _error_body(r: httpx.Response) -> str:
    try:
        data = r.json()
        return f"{data.get('error')}: {data.get('detail')}" if data.get("detail") else str(
            data.get("error")
        )
    except ValueError:
        return r.text[:200]


_shared: InferenceClient | None = None


def connect(require: bool = True) -> InferenceClient:
    """Process-wide client; checks the server first (exit 3 path) when ``require``."""
    global _shared
    if _shared is None:
        _shared = InferenceClient()
    if require and replay.replaying() is None:  # a replay asks the server only for what it
        _shared.require_ready()  # does not hold (and fails then, exit 3, if it is down)
    return _shared
