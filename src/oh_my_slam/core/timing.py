"""Per-stage wall-clock timings, inference-server request times and peak memory of one command.

Instrumentation only — nothing here writes to stdout. A command opens a collector with
:func:`collect`; code then marks

* **stages** (:func:`stage`): the sequential steps of the command. Stages may nest; a nested
  stage's time is taken out of its parent, so the stages are exclusive and add up to the total.
* **parts** (:func:`part`): sub-operations that repeat per keyframe/object and may run
  concurrently (e.g. geometry, gravity and segmentation of two keyframes in flight); summed
  wall time and call count.
* **requests** (:func:`record_request`, called by the inference client): per endpoint, the
  client-observed wall time and the server's queue and compute time.

Peak memory per stage: while a collection is open a daemon thread samples this process's resident
set every ``MEMORY_SAMPLE_S`` seconds (a few microseconds per reading) and raises the peak of
every stage running at that moment; the resident set is also read when a stage starts and ends.
When the process's lifetime high-water mark (``ru_maxrss``) rose while a stage ran, that mark is
the stage's exact peak; otherwise the sampled maximum is (a stage shorter than the sampling period
keeps its boundary readings). Peaks include nested stages — unlike times they do not add up — and
cover this process only: the evaluator adds child processes (COLMAP) and the inference server by
sampling them from outside during each stage's time windows (``t0_unix`` + ``stage_windows``,
wall-clock seconds).

Outside :func:`collect` every call is a no-op. ``mapper.sh update``, ``reconstruct.sh`` and
``segment.sh`` log a one-line summary on stderr and, when ``OH_MY_SLAM_TIMINGS=<path>`` is set,
write the full record there as JSON.

Live progress (spec §2.6 Jobs: a running job reports the command's own stage and its progress
where the command knows its size): while a collection is open, every stage start and end, every
:func:`count`, every finished :func:`part` and every :func:`progress` tick is an event (a
JSON-serialisable dict) passed to the listeners registered with :func:`listen`. When
``OH_MY_SLAM_PROGRESS=<path>`` is set, :func:`collect` appends each event to that file as one JSON
line (``/dev/fd/<n>`` reaches a pipe a job runner passed in). stdout and the human stderr output
are the same whether progress is on or off. The ``finish`` event carries the outcome (``ok``,
``exit_code`` and its ``code``, ``core.errors``).

This is how the web service (spec §2.6) follows a job: it runs each job as a subprocess
(``python -m oh_my_slam.cli.<command>``) with ``OH_MY_SLAM_PROGRESS`` set, never in its own
process, so the collector and the listeners here are per process and need no isolation between
jobs.

Stage names are the members of :class:`Stage`; ``oh_my_slam.commands.spec`` lists the stages of each
command.
"""

from __future__ import annotations

import json
import os
import resource
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Any

from oh_my_slam.core.errors import ExitCode, OhMySlamError, error_code

ENV_PATH = "OH_MY_SLAM_TIMINGS"
ENV_PROGRESS = "OH_MY_SLAM_PROGRESS"
MEMORY_SAMPLE_S = 0.05  # resident-set sampling period while a collection is open


class Stage(StrEnum):
    """Every timing stage name a command records (``tests/unit/test_commands.py`` checks that
    each ``stage("…")`` in the package is one of them)."""

    CONNECT = "connect"
    SETUP = "setup"
    INGEST = "ingest"
    INFERENCE = "inference"
    SEGMENT = "segment"
    SFM = "sfm"
    FEATURES_MATCHING = "features_matching"
    POSE_REFINEMENT = "pose_refinement"
    FOCAL_RERUN = "focal_rerun"
    MAP_FRAME = "map_frame"
    DEPTH_ALIGNMENT = "depth_alignment"
    PERSIST_FRAMES = "persist_frames"
    VALIDITY = "validity"
    OBJECTS = "objects"
    CLOUD = "cloud"
    POSE = "pose"
    EXPORT = "export"
    ARTIFACTS = "artifacts"
    WRITE = "write"
    COMMIT = "commit"


Listener = Callable[[dict[str, Any]], None]
_listeners: list[Listener] = []
_listeners_lock = threading.Lock()


@contextmanager
def listen(fn: Listener) -> Iterator[None]:
    """Pass every progress event to ``fn`` while the block runs (from any thread)."""
    with _listeners_lock:
        _listeners.append(fn)
    try:
        yield
    finally:
        with _listeners_lock:
            _listeners.remove(fn)


def _emit(event: str, **fields: Any) -> None:
    with _listeners_lock:
        targets = list(_listeners)
    if not targets:
        return
    record = {"event": event, **fields}
    for fn in targets:
        try:
            fn(record)
        except Exception:  # instrumentation never fails a command
            pass


@contextmanager
def _progress_file() -> Iterator[None]:
    """The ``OH_MY_SLAM_PROGRESS`` sink: one JSON line per event, appended to that path."""
    target = os.environ.get(ENV_PROGRESS)
    if not target:
        yield
        return
    try:
        fd = os.open(target, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    except OSError:  # instrumentation never fails a command
        yield
        return
    lock = threading.Lock()

    def write(record: dict[str, Any]) -> None:
        line = (json.dumps(record, default=str) + "\n").encode()
        with lock:
            os.write(fd, line)

    try:
        with listen(write):
            yield
    finally:
        os.close(fd)


class Timings:
    """Thread-safe collector for one command run; ``sample_every`` (seconds; None: stage
    boundaries only) is the resident-set sampling period of :meth:`start_sampling`."""

    def __init__(self, sample_every: float | None = MEMORY_SAMPLE_S) -> None:
        self._lock = threading.Lock()
        self._local = threading.local()
        self._t0 = time.perf_counter()
        self._t0_unix = time.time()
        self.stages: dict[str, float] = {}
        self.parts: dict[str, dict[str, float]] = {}
        self.requests: dict[str, dict[str, float]] = {}
        self.counts: dict[str, Any] = {}
        self.stage_peak_mb: dict[str, float] = {}  # peak resident set while the stage ran
        self.windows: list[tuple[str, float, float]] = []  # (stage, start, end) after t0_unix
        self._active: dict[str, int] = {}  # stages running now (any thread) → nesting depth
        self._sample_every = sample_every
        self._stop = threading.Event()
        self._sampler: threading.Thread | None = None

    # -- memory ------------------------------------------------------------------------------------

    def start_sampling(self) -> None:
        if self._sample_every is None or self._sampler is not None:
            return
        self._stop.clear()
        self._sampler = threading.Thread(target=self._sample_loop, name="timing-rss", daemon=True)
        self._sampler.start()

    def stop_sampling(self) -> None:
        if self._sampler is not None:
            self._stop.set()
            self._sampler.join()
            self._sampler = None

    def _sample_loop(self) -> None:
        every = self._sample_every or MEMORY_SAMPLE_S
        while not self._stop.wait(every):
            rss = rss_mb()
            with self._lock:
                self._observe(rss)

    def _observe(self, rss: float) -> None:
        """Raise the peak of every running stage to ``rss`` (the caller holds the lock)."""
        for name in self._active:
            if rss > self.stage_peak_mb.get(name, 0.0):
                self.stage_peak_mb[name] = rss

    # -- recording ---------------------------------------------------------------------------------

    def _stack(self) -> list[str]:
        st: list[str] | None = getattr(self._local, "stack", None)
        if st is None:
            st = self._local.stack = []
        return st

    @contextmanager
    def stage(self, stage: Stage) -> Iterator[None]:
        name = str(stage)  # recorded as its plain name
        stack = self._stack()
        parent = stack[-1] if stack else None
        stack.append(name)
        high0, rss = max_rss_mb(), rss_mb()
        with self._lock:
            self._active[name] = self._active.get(name, 0) + 1
            self._observe(rss)
        w0 = time.time()
        t0 = time.perf_counter()
        _emit("stage_start", stage=name, t=round(t0 - self._t0, 3))
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            w1 = time.time()
            _emit("stage_end", stage=name, t=round(t0 + dt - self._t0, 3),
                  seconds=round(dt, 3))
            stack.pop()
            high1, rss = max_rss_mb(), rss_mb()
            with self._lock:
                self.stages[name] = self.stages.get(name, 0.0) + dt
                if parent is not None:  # exclusive times: the parent adds dt when it ends
                    self.stages[parent] = self.stages.get(parent, 0.0) - dt
                self._observe(rss)
                if high1 > high0:  # the process's lifetime peak was reached during this stage
                    self.stage_peak_mb[name] = max(self.stage_peak_mb.get(name, 0.0), high1)
                depth = self._active.pop(name) - 1
                if depth:
                    self._active[name] = depth
                self.windows.append((name, w0 - self._t0_unix, w1 - self._t0_unix))

    @contextmanager
    def part(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add_part(name, time.perf_counter() - t0)

    def add_part(self, name: str, seconds: float) -> None:
        with self._lock:
            p = self.parts.setdefault(name, {"count": 0, "seconds": 0.0})
            p["count"] += 1
            p["seconds"] += seconds
            n = int(p["count"])
        _emit("part", part=name, count=n, seconds=round(seconds, 3))

    def request(self, endpoint: str, wall_s: float, queue_s: float, compute_s: float) -> None:
        with self._lock:
            r = self.requests.setdefault(
                endpoint, {"count": 0, "wall_s": 0.0, "queue_s": 0.0, "compute_s": 0.0})
            r["count"] += 1
            r["wall_s"] += wall_s
            r["queue_s"] += queue_s
            r["compute_s"] += compute_s

    def count(self, **values: Any) -> None:
        with self._lock:
            self.counts.update(values)
        _emit("count", **values)

    def progress(self, done: int, total: int) -> None:
        """``done`` of ``total`` items of the innermost running stage (this thread's)."""
        stack = self._stack()
        _emit("progress", stage=stack[-1] if stack else None, done=int(done), total=int(total))

    # -- reporting ---------------------------------------------------------------------------------

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_s": round(self.elapsed, 3),
                "stages_s": {k: round(v, 3) for k, v in self.stages.items()},
                "parts": {k: {"count": int(v["count"]), "seconds": round(v["seconds"], 3)}
                          for k, v in self.parts.items()},
                "server": {k: {"count": int(v["count"]), "wall_s": round(v["wall_s"], 3),
                               "queue_s": round(v["queue_s"], 3),
                               "compute_s": round(v["compute_s"], 3)}
                           for k, v in self.requests.items()},
                "counts": dict(self.counts),
                "peak_rss_mb": peak_rss_mb(),
                "stages_peak_rss_mb": {k: round(self.stage_peak_mb.get(k, 0.0), 1)
                                       for k in self.stages},
                "t0_unix": round(self._t0_unix, 3),
                "stage_windows": [[n, round(a, 3), round(b, 3)]
                                  for n, a, b in sorted(self.windows, key=lambda w: w[1])],
            }

    def summary(self) -> str:
        return summary_line(self.to_dict())


def summary_line(d: dict[str, Any]) -> str:
    """One stderr line: total, stages (largest first) and server compute per endpoint."""
    stages = sorted(d["stages_s"].items(), key=lambda kv: -kv[1])
    fields = [f"timings: total {d['total_s']:.1f} s"]
    if stages:
        fields.append(", ".join(f"{k} {v:.2f}" for k, v in stages))
    if d["server"]:
        fields.append("server compute: " + ", ".join(
            f"{k} {v['count']}× {v['compute_s']:.2f}" for k, v in d["server"].items()))
    fields.append(f"peak RSS {d['peak_rss_mb']['self']:.0f} MB")
    return " | ".join(fields)


_MAXRSS_UNIT = 1.0 if os.uname().sysname == "Darwin" else 1024.0  # ru_maxrss: bytes / KiB


def max_rss_mb(who: int = resource.RUSAGE_SELF) -> float:
    """Lifetime high-water mark of the resident set in MB (``RUSAGE_CHILDREN``: of the largest
    waited-for child)."""
    return resource.getrusage(who).ru_maxrss * _MAXRSS_UNIT / 1e6


_process: Any = None


def rss_mb() -> float:
    """Current resident set of this process in MB (0 if it cannot be read)."""
    global _process
    try:
        if _process is None:
            import psutil  # on first use (the sampler thread usually gets here first)

            _process = psutil.Process()
        return float(_process.memory_info().rss) / 1e6
    except Exception:  # instrumentation never fails a command
        return 0.0


def peak_rss_mb() -> dict[str, float]:
    """Peak resident set of this process and of its largest waited-for child (e.g. COLMAP)."""
    return {"self": round(max_rss_mb(), 1),
            "children": round(max_rss_mb(resource.RUSAGE_CHILDREN), 1)}


_current: Timings | None = None


@contextmanager
def collect(sample_every: float | None = MEMORY_SAMPLE_S) -> Iterator[Timings]:
    """Make a fresh collector current for the enclosed block (restores the previous one); its
    resident-set sampler runs while the block does."""
    global _current
    with _progress_file():
        prev = _current
        t = _current = Timings(sample_every)
        t.start_sampling()
        _emit("begin")
        code = ExitCode.OK
        try:
            yield t
        except OhMySlamError as exc:
            code = exc.exit_code
            raise
        except KeyboardInterrupt:
            code = ExitCode.INTERRUPTED
            raise
        except BaseException:
            code = ExitCode.INTERNAL
            raise
        finally:
            t.stop_sampling()
            _current = prev
            _emit("finish", total_s=round(t.elapsed, 3), ok=code is ExitCode.OK,
                  exit_code=int(code), code=error_code(code))


def current() -> Timings | None:
    return _current


@contextmanager
def stage(name: Stage) -> Iterator[None]:
    t = _current
    if t is None:
        yield
        return
    with t.stage(name):
        yield


@contextmanager
def part(name: str) -> Iterator[None]:
    t = _current
    if t is None:
        yield
        return
    with t.part(name):
        yield


def record_request(route: str, wall_s: float, queue_s: float, compute_s: float) -> None:
    t = _current
    if t is not None:
        t.request(route.rstrip("/").rsplit("/", 1)[-1], wall_s, queue_s, compute_s)


def count(**values: Any) -> None:
    t = _current
    if t is not None:
        t.count(**values)


def progress(done: int, total: int) -> None:
    """Report ``done`` of ``total`` items of the running stage (a progress event only)."""
    t = _current
    if t is not None:
        t.progress(done, total)


def report(t: Timings | dict[str, Any], logger: Any, **extra: Any) -> dict[str, Any]:
    """Log the summary line on stderr; write the full record to ``$OH_MY_SLAM_TIMINGS`` if set."""
    record = {**extra, **(t.to_dict() if isinstance(t, Timings) else t)}
    logger.info(summary_line(record))
    target = os.environ.get(ENV_PATH)
    if target:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(record, indent=1, default=float) + "\n")
        tmp.replace(path)
    return record
