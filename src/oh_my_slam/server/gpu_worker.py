"""The single device thread.

PyTorch's MPS backend aborts the process when two threads touch the device (pytorch#197805), so
every model operation — loading, ``.to()``, forward passes, read-back and cache clearing — runs on
this one thread. Requests queue up to ``max_queue``; beyond that :class:`QueueFullError` (HTTP 503).
A job that fails has its traceback's frames (and those of the exceptions chained to it) cleared
here before its exception leaves the thread, so the device tensors they held are freed on this
thread, not on the event loop's.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any, TypeVar

T = TypeVar("T")

_STOP = object()


def _clear_frames(exc: BaseException) -> None:
    """Free the locals of every frame of ``exc``'s traceback, and of the exceptions it was raised
    from or while handling, on the calling thread."""
    todo: list[BaseException | None] = [exc]
    seen: set[int] = set()
    while todo:
        e = todo.pop()
        if e is not None and id(e) not in seen:
            seen.add(id(e))
            traceback.clear_frames(e.__traceback__)
            todo += [e.__cause__, e.__context__]


class QueueFullError(RuntimeError):
    pass


class WorkerStoppedError(RuntimeError):
    pass


class GpuWorker:
    def __init__(self, max_queue: int = 8, name: str = "gpu-worker") -> None:
        self.max_queue = max_queue
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max_queue)
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._stopped = threading.Event()
        self._busy = threading.Event()
        self.thread_ident: int | None = None

    def start(self) -> None:
        self._thread.start()

    @property
    def depth(self) -> int:
        """Jobs waiting plus the one running."""
        return self._queue.qsize() + (1 if self._busy.is_set() else 0)

    def submit(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> Future[T]:
        if self._stopped.is_set():
            raise WorkerStoppedError("worker stopped")
        fut: Future[T] = Future()
        try:
            self._queue.put_nowait((fut, fn, args, kwargs, time.perf_counter()))
        except queue.Full as exc:
            raise QueueFullError(f"queue full ({self.max_queue} jobs)") from exc
        return fut

    def stop(self, timeout: float = 10.0) -> None:
        self._stopped.set()
        try:
            self._queue.put(_STOP, timeout=timeout)
        except queue.Full:
            pass
        self._thread.join(timeout)

    def _run(self) -> None:
        self.thread_ident = threading.get_ident()
        while True:
            item = self._queue.get()
            if item is _STOP:
                break
            fut, fn, args, kwargs, enqueued = item
            if not fut.set_running_or_notify_cancel():
                continue
            self._busy.set()
            try:
                started = time.perf_counter()
                result = fn(*args, **kwargs)
                if isinstance(result, dict) and "timings" in result:
                    result["timings"] = {
                        "queue_s": started - enqueued,
                        "compute_s": time.perf_counter() - started,
                    }
                fut.set_result(result)
            except BaseException as exc:
                _clear_frames(exc)
                fut.set_exception(exc)
            finally:
                self._busy.clear()
        # fail anything left
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is not _STOP:
                item[0].set_exception(WorkerStoppedError("worker stopped"))
