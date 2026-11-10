"""The single device thread.

PyTorch's MPS backend aborts the process when two threads touch the device (pytorch#197805), so
every model operation — loading, ``.to()``, forward passes, read-back and cache clearing — runs on
this one thread. Requests queue up to ``max_queue``; beyond that :class:`QueueFullError` (HTTP 503).
The queue is a priority queue (inference.md, single worker): a job submitted with
``INTERACTIVE`` priority (locate, dry runs, frame analysis) runs before every ``BULK`` one (a
job's fusion) still waiting, and jobs of one priority run in submission order.
A job that fails has its traceback's frames (and those of the exceptions chained to it) cleared
here before its exception leaves the thread, so the device tensors they held are freed on this
thread, not on the event loop's. ``after_job``, when set, then runs on this thread after every
job with the job's exception (None on success), before its future is resolved: the server frees
and measures device memory there.

A job whose remainder needs no device returns a :class:`CpuFinish`: the worker records its queue
and device time in it, and the server runs ``finish`` on a CPU thread (``app.ServerState``), so
the next job's device work does not wait for it.
"""

from __future__ import annotations

import itertools
import logging
import queue
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")

log = logging.getLogger(__name__)

_STOP = object()
INTERACTIVE = 0  # priorities: lower runs first
BULK = 1
_LAST = 2  # the stop marker: after everything queued


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


@dataclass
class CpuFinish:
    """What remains of a job once its device work is done: ``finish`` computes the response from
    CPU data only (it must never touch the device: it runs on another thread). ``queue_s`` and
    ``compute_s`` are the device part's, set by the worker."""

    finish: Callable[[], dict[str, Any]]
    queue_s: float = 0.0
    compute_s: float = 0.0


class QueueFullError(RuntimeError):
    pass


class WorkerStoppedError(RuntimeError):
    pass


class GpuWorker:
    def __init__(self, max_queue: int = 8, name: str = "gpu-worker") -> None:
        self.max_queue = max_queue
        self._queue: queue.PriorityQueue[Any] = queue.PriorityQueue(maxsize=max_queue)
        self._seq = itertools.count()  # submission order within a priority
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._stopped = threading.Event()
        self._busy = threading.Event()
        self.thread_ident: int | None = None
        self.after_job: Callable[[BaseException | None], None] | None = None

    def start(self) -> None:
        self._thread.start()

    @property
    def depth(self) -> int:
        """Jobs waiting plus the one running."""
        return self._queue.qsize() + (1 if self._busy.is_set() else 0)

    def submit(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> Future[T]:
        return self.submit_at(BULK, fn, *args, **kwargs)

    def submit_at(self, priority: int, fn: Callable[..., T], *args: Any, **kwargs: Any
                  ) -> Future[T]:
        """``submit`` with a priority (``INTERACTIVE`` before ``BULK``)."""
        if self._stopped.is_set():
            raise WorkerStoppedError("worker stopped")
        fut: Future[T] = Future()
        try:
            self._queue.put_nowait((priority, next(self._seq),
                                    (fut, fn, args, kwargs, time.perf_counter())))
        except queue.Full as exc:
            raise QueueFullError(f"queue full ({self.max_queue} jobs)") from exc
        return fut

    def stop(self, timeout: float = 10.0) -> None:
        self._stopped.set()
        try:
            self._queue.put((_LAST, next(self._seq), _STOP), timeout=timeout)
        except queue.Full:
            pass
        self._thread.join(timeout)

    def _after(self, exc: BaseException | None) -> None:
        if self.after_job is not None:
            try:
                self.after_job(exc)
            except Exception:
                log.exception("after_job failed")

    def _run(self) -> None:
        self.thread_ident = threading.get_ident()
        while True:
            item = self._queue.get()[2]
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
                elif isinstance(result, CpuFinish):
                    result.queue_s = started - enqueued
                    result.compute_s = time.perf_counter() - started
            except BaseException as exc:
                _clear_frames(exc)
                self._after(exc)
                fut.set_exception(exc)
            else:
                self._after(None)
                fut.set_result(result)
            finally:
                self._busy.clear()
        # fail anything left
        while True:
            try:
                item = self._queue.get_nowait()[2]
            except queue.Empty:
                break
            if item is not _STOP:
                item[0].set_exception(WorkerStoppedError("worker stopped"))
