"""The single device thread (CLAUDE.md: two threads touching MPS abort the process): every job
runs on it, one at a time, within its queue limit; every model operation — load, warm-up, run —
reaches it through the app and through the server process's ``serve``; and a failed job frees
its locals on it before its exception leaves it."""

from __future__ import annotations

import logging
import os
import shutil
import signal
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from oh_my_slam.client import protocol as p
from oh_my_slam.client.client import InferenceClient
from oh_my_slam.core import paths
from oh_my_slam.core.errors import ServerUnavailableError
from oh_my_slam.server import main as server_main
from oh_my_slam.server.app import ServerState, create_app
from oh_my_slam.server.gpu_worker import GpuWorker, QueueFullError, WorkerStoppedError
from oh_my_slam.server.models import Registry
from tests.fakes.stub_models import stub_adapters


def test_all_jobs_run_on_one_thread() -> None:
    w = GpuWorker(max_queue=64)
    w.start()
    idents = [w.submit(threading.get_ident) for _ in range(50)]
    seen = {f.result(timeout=5) for f in idents}
    assert seen == {w.thread_ident}
    assert w.thread_ident != threading.get_ident()
    w.stop()


def test_concurrent_submitters_are_serialised() -> None:
    w = GpuWorker(max_queue=256)
    w.start()
    active = 0
    peak = 0
    lock = threading.Lock()

    def job() -> int:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.001)
        with lock:
            active -= 1
        return 1

    futures = []
    flock = threading.Lock()

    def client() -> None:
        for _ in range(20):
            f = w.submit(job)
            with flock:
                futures.append(f)

    threads = [threading.Thread(target=client) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(f.result(timeout=10) for f in futures) == 160
    assert peak == 1
    w.stop()


def test_queue_limit_and_errors() -> None:
    w = GpuWorker(max_queue=2)
    gate = threading.Event()
    w.start()
    first = w.submit(gate.wait, 5)
    time.sleep(0.05)  # first job is running, queue empty
    w.submit(lambda: 1)
    w.submit(lambda: 2)
    assert w.depth == 3
    with pytest.raises(QueueFullError):
        w.submit(lambda: 3)
    gate.set()
    assert first.result(timeout=5) is True

    def boom() -> None:
        raise ValueError("bad")

    with pytest.raises(ValueError):
        w.submit(boom).result(timeout=5)
    res = w.submit(lambda: {"timings": None, "x": 1}).result(timeout=5)
    assert res["timings"]["compute_s"] >= 0 and res["timings"]["queue_s"] >= 0
    w.stop()
    with pytest.raises(WorkerStoppedError):
        w.submit(lambda: 1)


def test_stop_fails_pending_jobs() -> None:
    w = GpuWorker(max_queue=10)
    gate = threading.Event()
    w.start()
    w.submit(gate.wait, 2)
    time.sleep(0.05)
    pending = [w.submit(lambda: 1) for _ in range(3)]
    t = threading.Thread(target=w.stop, kwargs={"timeout": 5})
    t.start()
    time.sleep(0.05)
    gate.set()
    t.join()
    for f in pending:
        # either they ran before the stop marker or were failed afterwards
        try:
            f.result(timeout=2)
        except WorkerStoppedError:
            pass


# --- every model operation on the device thread --------------------------------------------------


Seen = list[tuple[str, str, int]]  # (adapter, operation, thread ident)


class _Recording:
    """A stub adapter whose load, warm-up and run record the thread they run on."""

    def __init__(self, adapter: Any, seen: Seen) -> None:
        self._adapter, self._seen = adapter, seen
        self.key, self.name, self.precision = adapter.key, adapter.name, adapter.precision

    def _record(self, what: str) -> None:
        self._seen.append((self.key, what, threading.get_ident()))

    def load(self, device: str) -> None:
        self._record("load")
        self._adapter.load(device)

    def warmup(self) -> None:
        self._record("warmup")
        self._adapter.warmup()

    def run(self, req: Any) -> dict[str, Any]:
        self._record("run")
        return self._adapter.run(req)  # type: ignore[no-any-return]


def _recording_registry(seen: Seen) -> Registry:
    reg = Registry()
    for a in stub_adapters():
        reg.add(_Recording(a, seen))
    return reg


def _post_every_route(post: Callable[..., Any], tmp_path: Path) -> list[int]:
    """One request to each model route; their HTTP statuses."""
    image = tmp_path / "img.png"
    Image.fromarray(np.full((60, 80, 3), 90, np.uint8)).save(image)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    bodies = [(p.ROUTE_GEOMETRY, p.GeometryRequest(image_path=str(image), out_dir=str(out))),
              (p.ROUTE_GRAVITY, p.GravityRequest(image_path=str(image))),
              (p.ROUTE_SEGMENT, p.SegmentRequest(image_path=str(image), labels=["chair"])),
              (p.ROUTE_MULTIVIEW, p.MultiviewRequest(image_paths=[str(image)] * 2,
                                                     out_dir=str(out)))]
    return [post(route, json=body.model_dump()).status_code for route, body in bodies]


def _assert_on_the_device_thread(seen: Seen, worker: GpuWorker) -> None:
    every = {(a.key, what) for a in stub_adapters() for what in ("load", "warmup", "run")}
    assert {(key, what) for key, what, _ in seen} == every
    assert {ident for *_, ident in seen} == {worker.thread_ident}
    assert worker.thread_ident != threading.get_ident()


def test_the_app_runs_every_model_operation_on_the_device_thread(tmp_path: Path) -> None:
    """Loading as the server process does it (``_load_models`` on the worker), then one request
    per route through the HTTP app: each load, warm-up and run happens on the worker's thread."""
    seen: Seen = []
    worker = GpuWorker()
    worker.start()
    state = ServerState(registry=_recording_registry(seen), worker=worker)
    try:
        worker.submit(server_main._load_models, state, "cpu").result(timeout=30)
        assert _post_every_route(TestClient(create_app(state)).post, tmp_path) == [200] * 4
    finally:
        worker.stop()
    _assert_on_the_device_thread(seen, worker)


def test_serve_runs_every_model_operation_on_its_device_thread(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The server process's own ``serve`` (lock, socket, uvicorn, worker), in this process with
    the recording adapters: requests over its Unix socket, then SIGTERM, its normal stop."""
    runtime = Path(tempfile.mkdtemp(prefix="oms-", dir="/tmp"))  # short: the socket path fits
    monkeypatch.setenv("OH_MY_SLAM_RUNTIME_DIR", str(runtime))
    monkeypatch.setattr(paths, "weights_dir", lambda: tmp_path)  # serve() changes into it
    monkeypatch.chdir(tmp_path)  # and the cwd is restored after the test
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: None)  # pytest's handlers stay
    workers: list[GpuWorker] = []

    class Worker(GpuWorker):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            workers.append(self)

    monkeypatch.setattr(server_main, "GpuWorker", Worker)
    original = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    seen: Seen = []
    codes: list[int] = []

    def drive() -> None:
        client = InferenceClient()
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                try:
                    if client.health(timeout=1.0).status == "ready":
                        break
                except ServerUnavailableError:
                    pass
                time.sleep(0.05)
            with httpx.Client(transport=httpx.HTTPTransport(uds=str(paths.socket_path())),
                              base_url="http://oh-my-slam", timeout=60) as http:
                codes.extend(_post_every_route(http.post, tmp_path))
        finally:
            client.close()
            if signal.getsignal(signal.SIGTERM) is not original[signal.SIGTERM]:
                os.kill(os.getpid(), signal.SIGTERM)  # serve()'s handlers are in place

    driver = threading.Thread(target=drive)
    driver.start()
    try:
        assert server_main.serve(lambda: _recording_registry(seen), device="cpu") == 0
    finally:
        driver.join(60)
        for sig, handler in original.items():
            signal.signal(sig, handler)
        shutil.rmtree(runtime, ignore_errors=True)
    assert codes == [200] * 4
    (worker,) = workers
    _assert_on_the_device_thread(seen, worker)


def test_a_failed_job_frees_its_locals_on_the_device_thread() -> None:
    """The frames of a failed job's traceback, and of the exceptions it was raised from or while
    handling, are cleared on the worker: what they held (device tensors, in the server) is freed
    there, before the exception reaches the event loop, which still gets its type and message."""
    freed: list[tuple[str, int]] = []

    class Tensor:
        def __init__(self, name: str) -> None:
            self.name = name

        def __del__(self) -> None:
            freed.append((self.name, threading.get_ident()))

    def forward() -> None:
        t = Tensor("cause")  # a local of the frame that failed first
        raise MemoryError(t.name)

    def job() -> None:
        t = Tensor("raised")  # a local of the failing frame
        try:
            forward()
        except MemoryError as exc:
            raise RuntimeError(f"MPS backend out of memory ({t.name})") from exc

    w = GpuWorker()
    w.start()
    try:
        exc = w.submit(job).exception(timeout=5)
        assert isinstance(exc, RuntimeError) and str(exc) == "MPS backend out of memory (raised)"
        assert isinstance(exc.__cause__, MemoryError)
        assert sorted(freed) == [("cause", w.thread_ident), ("raised", w.thread_ident)]
    finally:
        w.stop()
