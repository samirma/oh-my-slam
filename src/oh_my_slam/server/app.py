"""FastAPI application: a non-blocking ``/health`` and one POST route per model.

Every model call is handed to the single :class:`GpuWorker`; the HTTP event loop only awaits
futures, so ``/health`` never waits on the device.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from oh_my_slam.client import protocol as p
from oh_my_slam.core.errors import InputError
from oh_my_slam.server.gpu_worker import GpuWorker, QueueFullError, WorkerStoppedError
from oh_my_slam.server.models import Registry
from oh_my_slam.version import PROTOCOL_VERSION


@dataclass
class ServerState:
    registry: Registry
    worker: GpuWorker
    started_at: float = field(default_factory=time.time)
    loading: bool = True
    stopping: bool = False
    versions: dict[str, str] = field(default_factory=dict)

    def status(self) -> str:
        if self.stopping:
            return "stopping"
        if self.loading:
            return "loading"
        return self.registry.status()

    def health(self) -> p.Health:
        reg = self.registry
        models = {
            key: p.ModelStatus(
                name=getattr(a, "name", key),
                loaded=reg.state[key].loaded,
                error=reg.state[key].error,
                detail=reg.state[key].detail,
            )
            for key, a in reg.adapters.items()
        }
        return p.Health(
            status=self.status(),  # type: ignore[arg-type]
            models=models,
            device=reg.device,
            precision=reg.precision,
            versions=self.versions,
            pid=os.getpid(),
            protocol=PROTOCOL_VERSION,
            queue_depth=self.worker.depth,
            queue_limit=self.worker.max_queue,
            uptime_s=time.time() - self.started_at,
        )


def _error(status: int, error: str, detail: str | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content=p.ErrorBody(error=error, detail=detail).model_dump())


def create_app(state: ServerState) -> FastAPI:
    app = FastAPI(title="oh-my-slam inference server", docs_url=None, redoc_url=None)

    async def dispatch(key: str, call: Callable[[Any], dict[str, Any]], model: type) -> Any:
        status = state.status()
        if status in ("loading", "stopping"):
            return _error(503, f"server {status}")
        adapter = state.registry.get(key)
        if adapter is None:
            err = state.registry.state.get(key)
            return _error(500, f"model '{key}' unavailable", err.error if err else None)
        try:
            fut = state.worker.submit(call, adapter)
        except QueueFullError as exc:
            return _error(503, "busy", str(exc))
        except WorkerStoppedError:
            return _error(503, "server stopping")
        try:
            result = await asyncio.wrap_future(fut)
        except (InputError, FileNotFoundError, ValueError) as exc:
            return _error(400, type(exc).__name__, str(exc))
        except Exception as exc:
            return _error(500, type(exc).__name__, str(exc))
        return model.model_validate(result)

    @app.get(p.ROUTE_HEALTH, response_model=p.Health)
    def health() -> p.Health:
        return state.health()

    @app.post(p.ROUTE_GEOMETRY, response_model=p.GeometryResponse)
    async def geometry(req: p.GeometryRequest) -> Any:
        return await dispatch("geometry", lambda a: a.run(req), p.GeometryResponse)

    @app.post(p.ROUTE_GRAVITY, response_model=p.GravityResponse)
    async def gravity(req: p.GravityRequest) -> Any:
        return await dispatch("gravity", lambda a: a.run(req), p.GravityResponse)

    @app.post(p.ROUTE_SEGMENT, response_model=p.SegmentResponse)
    async def segment(req: p.SegmentRequest) -> Any:
        return await dispatch("segment_yoloe", lambda a: a.run(req), p.SegmentResponse)

    @app.post(p.ROUTE_MULTIVIEW, response_model=p.MultiviewResponse)
    async def multiview(req: p.MultiviewRequest) -> Any:
        return await dispatch("multiview", lambda a: a.run(req), p.MultiviewResponse)

    return app
