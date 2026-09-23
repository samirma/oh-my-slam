"""HTTP-over-Unix-socket client for the inference server.

Any failure to reach the server becomes :class:`ServerUnavailableError` (exit 3, with the hint to
run ``./start_inference_server.sh``). Queue-full responses (503) are retried with back-off.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TypeVar

import httpx
from pydantic import BaseModel

from oh_my_slam.client import protocol as p
from oh_my_slam.core import paths, timing
from oh_my_slam.core.errors import (
    InferenceError,
    InputError,
    ServerBusyError,
    ServerUnavailableError,
)
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
        if h.status in ("ready", "degraded"):
            return h
        failed = {k: m.error for k, m in h.models.items() if m.error}
        raise ServerUnavailableError(f"server status '{h.status}' {failed or ''}".strip())

    # -- requests ----------------------------------------------------------------------------------

    def _post(self, route: str, req: BaseModel, model: type[M]) -> M:
        t0 = time.perf_counter()
        deadline = time.monotonic() + BUSY_RETRY_S
        delay = 0.2
        while True:
            if not self.socket_path.exists():
                raise ServerUnavailableError("no socket")
            try:
                r = self._http().post(route, json=req.model_dump())
            except httpx.TimeoutException as exc:
                raise InferenceError(f"{route} timed out after {self.timeout:.0f} s") from exc
            except httpx.HTTPError as exc:
                raise ServerUnavailableError(type(exc).__name__) from exc
            if r.status_code == 200:
                out = model.model_validate(r.json())
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

    def geometry(self, req: p.GeometryRequest) -> p.GeometryResponse:
        return self._post(p.ROUTE_GEOMETRY, req, p.GeometryResponse)

    def gravity(self, req: p.GravityRequest) -> p.GravityResponse:
        return self._post(p.ROUTE_GRAVITY, req, p.GravityResponse)

    def segment_image(self, req: p.SegmentRequest) -> p.SegmentResponse:
        return self._post(p.ROUTE_SEGMENT, req, p.SegmentResponse)

    def multiview(self, req: p.MultiviewRequest) -> p.MultiviewResponse:
        return self._post(p.ROUTE_MULTIVIEW, req, p.MultiviewResponse)


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
    if require:
        _shared.require_ready()
    return _shared
