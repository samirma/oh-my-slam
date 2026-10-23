"""Device memory in the inference server: after every job, on the device thread, the server frees
what a failed job left and measures the memory; ``/health`` reports it, and ``degraded`` (not
``ready``) while the last job ran out of memory or the memory in use stays near the cap."""

from __future__ import annotations

import threading
from typing import Any

from fastapi.testclient import TestClient

from oh_my_slam.client import protocol as p
from oh_my_slam.server import main as server_main
from oh_my_slam.server.app import ServerState, create_app, is_out_of_memory
from oh_my_slam.server.gpu_worker import GpuWorker
from tests.fakes.fake_torch import installed, make_torch
from tests.fakes.stub_models import stub_registry

GIB = 1 << 30
OOM = ("MPS backend out of memory (MPS allocated: 9.24 GiB, other allocations: 43.45 MiB, "
       "max allowed: 9.32 GiB). Tried to allocate 61.00 MiB on shared pool.")


def test_after_job_runs_on_the_device_thread_with_the_jobs_outcome() -> None:
    """Before the future resolves, with the job's exception (None on success); a hook that fails
    is logged and the job's own outcome still reaches the caller."""
    w = GpuWorker()
    seen: list[tuple[str | None, int]] = []
    w.after_job = lambda exc: seen.append((None if exc is None else str(exc),
                                           threading.get_ident()))
    w.start()
    try:
        assert w.submit(lambda: 1).result(timeout=5) == 1

        def boom() -> None:
            raise RuntimeError("bad")

        assert str(w.submit(boom).exception(timeout=5)) == "bad"
        assert seen == [(None, w.thread_ident), ("bad", w.thread_ident)]

        def broken(exc: BaseException | None) -> None:
            raise ValueError("hook")

        w.after_job = broken
        assert w.submit(lambda: 2).result(timeout=5) == 2
    finally:
        w.stop()


def test_out_of_memory_errors_are_recognised() -> None:
    assert is_out_of_memory(RuntimeError(OOM))
    assert is_out_of_memory(MemoryError())
    assert is_out_of_memory(RuntimeError("CUDA out of memory"))
    assert not is_out_of_memory(RuntimeError("shape mismatch"))


def _mps_state(torch: Any, limit: float = 10 * GIB) -> ServerState:
    reg = stub_registry()
    reg.device = "mps"
    for st in reg.state.values():
        st.loaded = True
    state = ServerState(registry=reg, worker=GpuWorker(), loading=False)
    state.memory_limit = limit
    return state


def test_a_job_that_ran_out_of_memory_degrades_the_server_until_one_succeeds() -> None:
    torch = make_torch(mps_available=True)
    torch.memory.update(allocated=9.5 * GIB, driver=9.8 * GIB, freed_to=8 * GIB)
    state = _mps_state(torch)
    with installed({"torch": torch}):
        state.after_job(RuntimeError(OOM))
        # the failure freed the allocator's cache and was measured after that
        assert torch.calls == [("empty_cache",)]
        assert state.memory == p.DeviceMemory(allocated_gib=8.0, driver_gib=8.0, limit_gib=10.0)
        assert state.status() == "degraded"
        h = state.health()
        assert h.status == "degraded" and h.last_oom == f"RuntimeError: {OOM}"
        assert h.memory_note() == f"8.00 of 10.00 GiB in use; last error: RuntimeError: {OOM}"
        state.after_job(None)  # memory back under the cap: ready, keeping the last error
        assert torch.calls == [("empty_cache",)]  # a success below the threshold keeps the cache
        assert state.status() == "ready" and state.health().last_oom == f"RuntimeError: {OOM}"


def test_memory_near_the_cap_after_a_job_degrades_the_server() -> None:
    """At 90 % of the cap or more, a successful job also frees the cache; what stays in use that
    high leaves a request too little, so the server is degraded, without an error to report."""
    torch = make_torch(mps_available=True)
    torch.memory.update(allocated=9.5 * GIB, driver=9.9 * GIB)  # nothing to free
    state = _mps_state(torch)
    with installed({"torch": torch}):
        state.after_job(None)
    assert torch.calls == [("empty_cache",)] and state.status() == "degraded"
    assert state.health().memory_note() == "9.50 of 10.00 GiB in use"
    state.loading = True  # the other statuses come first
    assert state.status() == "loading"


def test_a_failure_other_than_memory_frees_but_does_not_degrade() -> None:
    torch = make_torch(mps_available=True)
    torch.memory.update(allocated=8 * GIB, driver=8.5 * GIB)
    state = _mps_state(torch)
    with installed({"torch": torch}):
        state.after_job(ValueError("bad image"))
    assert torch.calls == [("empty_cache",)] and state.status() == "ready"
    assert state.last_oom is None and state.health().memory_note() == "8.00 of 10.00 GiB in use"


def test_on_the_cpu_nothing_is_measured_but_out_of_memory_still_degrades() -> None:
    state = ServerState(registry=stub_registry(), worker=GpuWorker(), loading=False)
    state.registry.state = {k: type(v)(loaded=True) for k, v in state.registry.state.items()}
    with installed({"torch": None}):  # never imported
        state.after_job(MemoryError("no room"))
    assert state.memory is None and state.status() == "degraded"
    assert state.health().memory_note() == "last error: MemoryError: no room"
    assert p.Health(status="degraded").memory_note() == "no detail"


def test_loading_on_mps_sets_the_cap_the_memory_is_measured_against() -> None:
    torch = make_torch(mps_available=True)
    state = ServerState(registry=stub_registry(), worker=GpuWorker())
    with installed({"torch": torch}):
        server_main._load_models(state, None)
    assert state.memory_limit == 16 * GIB * server_main.MEMORY_FRACTION


def test_health_reports_the_memory_over_http() -> None:
    torch = make_torch(mps_available=True)
    torch.memory.update(allocated=8 * GIB, driver=8.25 * GIB)
    state = _mps_state(torch)
    with installed({"torch": torch}):
        state.after_job(None)
    h = TestClient(create_app(state)).get(p.ROUTE_HEALTH).json()
    assert h["status"] == "ready" and h["last_oom"] is None
    assert h["memory"] == {"allocated_gib": 8.0, "driver_gib": 8.25, "limit_gib": 10.0}

