from __future__ import annotations

import threading
import time

import pytest

from oh_my_slam.server.gpu_worker import GpuWorker, QueueFullError, WorkerStoppedError


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
        w.call(boom)
    res = w.call(lambda: {"timings": None, "x": 1})
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
