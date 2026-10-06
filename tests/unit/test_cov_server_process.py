"""The inference server process (spec §2.1) beyond the happy path: its entry point refusing a
second instance, model loading on the device thread, the lock / state / socket bookkeeping, the
single device thread at its limits, and the HTTP answers when the device thread or a model fails."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from oh_my_slam.client import protocol as p
from oh_my_slam.core import paths
from oh_my_slam.server import gpu_worker, lifecycle
from oh_my_slam.server import main as server_main
from oh_my_slam.server.app import ServerState, create_app
from oh_my_slam.server.gpu_worker import GpuWorker, WorkerStoppedError
from oh_my_slam.server.models import Registry
from oh_my_slam.version import __version__
from tests.fakes.fake_torch import installed, make_torch
from tests.fakes.stub_models import stub_registry

REPO = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------------------------------------
# entry point and model loading


def test_the_entry_point_refuses_to_start_while_another_server_holds_the_lock() -> None:
    """``python -m oh_my_slam.server.main`` (what start_inference_server.sh launches) exits 1 with
    the reason when another server holds the single-instance lock, loading no model."""
    lock = lifecycle.ServerLock()
    lock.acquire()
    try:
        res = subprocess.run([sys.executable, "-m", "oh_my_slam.server.main"], cwd=REPO,
                             capture_output=True, text=True, timeout=120)
    finally:
        lock.release()
    assert res.returncode == 1 and "another server holds" in res.stderr
    assert res.stdout == "" and not paths.state_file().exists()


def test_versions_list_the_installed_model_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    from importlib import metadata

    monkeypatch.setattr(server_main, "_VERSION_PACKAGES", ("numpy", "no-such-package-oms"))
    assert server_main._versions() == {"oh-my-slam": __version__,
                                       "numpy": metadata.version("numpy")}


class _Recorder(Registry):
    def __init__(self) -> None:
        super().__init__()
        self.loaded_on: list[str] = []

    def load_all(self, device: str, log: Any = None) -> None:
        self.loaded_on.append(device)


def _state(registry: Registry) -> ServerState:
    return ServerState(registry=registry, worker=GpuWorker())


def test_models_load_on_mps_with_a_memory_cap() -> None:
    torch = make_torch(mps_available=True)
    state = _state(_Recorder())
    with installed({"torch": torch}):
        server_main._load_models(state, None)  # the preferred device: MPS here
    assert state.registry.loaded_on == ["mps"] and not state.loading
    assert torch.calls == [("memory_fraction", server_main.MEMORY_FRACTION)]


def test_models_load_on_the_device_asked_for_without_a_cap_on_the_cpu() -> None:
    torch = make_torch(mps_available=True)
    state = _state(_Recorder())
    with installed({"torch": torch}):
        server_main._load_models(state, "cpu")
    assert state.registry.loaded_on == ["cpu"] and torch.calls == []


def test_a_failed_device_selection_still_ends_the_loading_state() -> None:
    """Without torch the server cannot pick a device: it stops saying ``loading`` (health then
    reports ``error``, every model unloaded) instead of loading forever."""
    reg = stub_registry()
    state = _state(reg)
    with installed({"torch": None}), pytest.raises(ImportError):
        server_main._load_models(state, None)
    assert not state.loading and state.status() == "error"


# ------------------------------------------------------------------------------------------------
# lock, state file and socket


def test_releasing_the_lock_twice_is_harmless(tmp_path: Path) -> None:
    lock = lifecycle.ServerLock(tmp_path / "server.lock")
    lock.acquire()
    assert lifecycle.ServerLock.is_held(tmp_path / "server.lock")
    lock.release()
    lock.release()
    assert not lifecycle.ServerLock.is_held(tmp_path / "server.lock")
    assert not lifecycle.ServerLock.is_held(tmp_path / "missing.lock")


def test_a_listening_socket_is_live_and_a_missing_one_is_not() -> None:
    sock_path = Path("/tmp") / f"oms-live-{os.getpid()}.sock"
    assert not lifecycle.socket_is_live(sock_path)
    s = lifecycle.bind_socket(sock_path)
    try:
        assert oct(sock_path.stat().st_mode & 0o777) == "0o600"
        assert lifecycle.socket_is_live(sock_path)
        assert not lifecycle.cleanup_stale_socket(sock_path)  # in use: kept
    finally:
        s.close()
        sock_path.unlink(missing_ok=True)


def test_clearing_keeps_the_state_file_of_another_server(tmp_path: Path) -> None:
    sock = tmp_path / "srv.sock"
    sock.touch()
    lifecycle.write_state(sock, "9.9", None)
    state = lifecycle.read_state()
    assert state is not None and state["pid"] == os.getpid() and state["log"] is None
    state["pid"] = os.getpid() + 100_000  # another server's (it started since)
    paths.state_file().write_text(json.dumps(state))
    lifecycle.clear_state(sock)
    assert not sock.exists() and lifecycle.read_state() == state  # only our own state is removed
    paths.state_file().unlink()


def test_a_process_we_may_not_signal_is_alive() -> None:
    assert lifecycle.pid_alive(1)  # launchd: running, owned by root (EPERM)
    assert lifecycle.pid_alive(os.getpid())


# ------------------------------------------------------------------------------------------------
# the device thread at its limits


def test_stop_returns_when_the_queue_stays_full() -> None:
    w = GpuWorker(max_queue=1)
    gate = threading.Event()
    w.start()
    w.submit(gate.wait, 5)
    _until(lambda: w.depth == 1 and w._queue.qsize() == 0)
    queued = w.submit(lambda: "ran")  # the queue is full now
    t0 = time.monotonic()
    w.stop(timeout=0.2)  # cannot even queue the stop marker: gives up after the timeout
    assert time.monotonic() - t0 < 2.0
    gate.set()
    assert queued.result(timeout=5) == "ran"
    w._queue.put(gpu_worker._STOP)  # end the thread for good
    w._thread.join(5)
    assert not w._thread.is_alive()


def test_a_cancelled_job_never_runs() -> None:
    w = GpuWorker(max_queue=4)
    gate = threading.Event()
    w.start()
    w.submit(gate.wait, 5)
    ran: list[int] = []
    job = w.submit(ran.append, 1)
    assert job.cancel()  # still waiting: cancelled
    after = w.submit(lambda: "after")
    gate.set()
    assert after.result(timeout=5) == "after" and ran == []
    w.stop()


def test_jobs_queued_behind_the_stop_marker_fail() -> None:
    """A job that got past ``submit``'s check while the worker was being stopped lands behind the
    stop marker: it fails with WorkerStoppedError instead of waiting forever."""
    w = GpuWorker(max_queue=8)
    gate = threading.Event()
    w.start()
    w.submit(gate.wait, 5)
    _until(lambda: w._queue.qsize() == 0 and w.depth == 1)
    stopper = threading.Thread(target=w.stop, kwargs={"timeout": 5})
    stopper.start()
    _until(lambda: w._queue.qsize() == 1)  # the stop marker is queued
    late: Future[Any] = Future()
    w._queue.put_nowait((late, lambda: "late", (), {}, time.perf_counter()))
    w._queue.put_nowait(gpu_worker._STOP)  # a second stop() racing the first
    gate.set()
    stopper.join(5)
    with pytest.raises(WorkerStoppedError):
        late.result(timeout=5)
    with pytest.raises(WorkerStoppedError):
        w.submit(lambda: None)


def _until(cond: Any, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.005)


# ------------------------------------------------------------------------------------------------
# HTTP answers when the device thread or a model fails


def _ready_state() -> ServerState:
    reg = stub_registry()
    worker = GpuWorker()
    worker.start()
    worker.call(reg.load_all, "cpu")
    state = ServerState(registry=reg, worker=worker)
    state.loading = False
    return state


def test_a_stopped_device_thread_answers_503_stopping() -> None:
    state = _ready_state()
    state.worker.stop()
    client = TestClient(create_app(state))
    r = client.post(p.ROUTE_GRAVITY, json=p.GravityRequest(image_path="x.jpg").model_dump())
    assert r.status_code == 503 and r.json() == {"error": "server stopping", "detail": None}


def test_a_model_failure_answers_500_with_its_type_and_message(tmp_path: Path) -> None:
    state = _ready_state()

    class Broken:
        key, name = "multiview", "broken"

        def run(self, req: p.MultiviewRequest) -> dict[str, Any]:
            raise RuntimeError("MPS backend out of memory")

    state.registry.adapters["multiview"] = Broken()
    client = TestClient(create_app(state))
    r = client.post(p.ROUTE_MULTIVIEW, json=p.MultiviewRequest(
        image_paths=["a.jpg"], out_dir=str(tmp_path)).model_dump())
    assert r.status_code == 500
    assert r.json() == {"error": "RuntimeError", "detail": "MPS backend out of memory"}
    # the server keeps answering
    assert client.get(p.ROUTE_HEALTH).json()["status"] == "ready"
    state.worker.stop()

