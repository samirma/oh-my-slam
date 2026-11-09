"""FastAPI application: a non-blocking ``/health`` and one POST route per model.

Every model call is handed to the single :class:`GpuWorker`; the HTTP event loop only awaits
futures, so ``/health`` never waits on the device. The part of a call that needs no device
(``CpuFinish``: GeoCalib's least-squares refinement) then runs on a CPU thread (``cpu_pool``). After every job the worker calls
:meth:`ServerState.after_job` on its thread, which frees and measures the device memory that
``/health`` reports (``memory``, ``last_oom`` and the ``degraded`` status).
"""

from __future__ import annotations

import asyncio
import gc
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from oh_my_slam.client import protocol as p
from oh_my_slam.core.errors import InputError
from oh_my_slam.server.gpu_worker import CpuFinish, GpuWorker, QueueFullError, WorkerStoppedError
from oh_my_slam.server.models import Registry
from oh_my_slam.version import PROTOCOL_VERSION

GIB = float(1 << 30)
# Memory still in use after a job, as a fraction of the cap, from which the server is degraded
# (a request then has under a tenth of the cap for its own tensors), and from which a successful
# job also releases the allocator's cache.
NEAR_LIMIT = 0.9
# CPU threads for the device-free part of the calls (``CpuFinish``): the mapper keeps two frames
# in flight
CPU_THREADS = 2


def is_out_of_memory(exc: BaseException) -> bool:
    """MPS's "MPS backend out of memory", torch's OutOfMemoryError, or a MemoryError."""
    return isinstance(exc, MemoryError) or "out of memory" in str(exc).lower()


def device_memory(torch: Any, limit: float) -> p.DeviceMemory:
    """The MPS memory now (GPU thread only), against the cap ``limit`` in bytes."""
    return p.DeviceMemory(allocated_gib=torch.mps.current_allocated_memory() / GIB,
                          driver_gib=torch.mps.driver_allocated_memory() / GIB,
                          limit_gib=limit / GIB)


@dataclass
class ServerState:
    registry: Registry
    worker: GpuWorker
    started_at: float = field(default_factory=time.time)
    loading: bool = True
    stopping: bool = False
    versions: dict[str, str] = field(default_factory=dict)
    memory_limit: float = 0.0  # the MPS cap in bytes (set when the models load on MPS)
    memory: p.DeviceMemory | None = None
    last_oom: str | None = None
    last_job_oom: bool = False
    cpu_pool: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(
        CPU_THREADS, thread_name_prefix="cpu-finish"))

    def status(self) -> str:
        if self.stopping:
            return "stopping"
        if self.loading:
            return "loading"
        status = self.registry.status()
        if status == "ready" and (self.last_job_oom or self._near_limit(self.memory)):
            return "degraded"
        return status

    @staticmethod
    def _near_limit(mem: p.DeviceMemory | None) -> bool:
        return mem is not None and mem.allocated_gib >= NEAR_LIMIT * mem.limit_gib

    def after_job(self, exc: BaseException | None) -> None:
        """On the GPU thread after every job: records an out-of-memory failure, and on MPS frees
        what the job left (Python's cycles and the allocator's cache) after a failure or when the
        memory in use is near the cap, then measures the memory."""
        self.last_job_oom = exc is not None and is_out_of_memory(exc)
        if self.last_job_oom:
            self.last_oom = f"{type(exc).__name__}: {exc}"
        if self.registry.device != "mps" or not self.memory_limit:
            return
        import torch

        mem = device_memory(torch, self.memory_limit)
        if exc is not None or self._near_limit(mem):
            gc.collect()
            torch.mps.empty_cache()
            mem = device_memory(torch, self.memory_limit)
        self.memory = mem

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
            memory=self.memory,
            last_oom=self.last_oom,
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
            if isinstance(result, CpuFinish):
                result = await _finish(result)
        except (InputError, FileNotFoundError, ValueError) as exc:
            return _error(400, type(exc).__name__, str(exc))
        except Exception as exc:
            return _error(500, p.OUT_OF_MEMORY if is_out_of_memory(exc) else type(exc).__name__,
                          str(exc))
        return model.model_validate(result)

    async def _finish(part: CpuFinish) -> dict[str, Any]:
        """``part``'s response, computed on a CPU thread; its time adds to the device part's."""
        t0 = time.perf_counter()
        out = await asyncio.get_running_loop().run_in_executor(state.cpu_pool, part.finish)
        out["timings"] = {"queue_s": part.queue_s,
                          "compute_s": part.compute_s + time.perf_counter() - t0}
        return out

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
