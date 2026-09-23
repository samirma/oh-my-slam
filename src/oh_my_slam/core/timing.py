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

Outside :func:`collect` every call is a no-op. ``mapper.sh update``, ``reconstruct.sh`` and
``segment.sh -i`` log a one-line summary on stderr and, when ``OH_MY_SLAM_TIMINGS=<path>`` is
set, write the full record there as JSON.
"""

from __future__ import annotations

import json
import os
import resource
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ENV_PATH = "OH_MY_SLAM_TIMINGS"


class Timings:
    """Thread-safe collector for one command run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._local = threading.local()
        self._t0 = time.perf_counter()
        self.stages: dict[str, float] = {}
        self.parts: dict[str, dict[str, float]] = {}
        self.requests: dict[str, dict[str, float]] = {}
        self.counts: dict[str, Any] = {}

    # -- recording ---------------------------------------------------------------------------------

    def _stack(self) -> list[str]:
        st: list[str] | None = getattr(self._local, "stack", None)
        if st is None:
            st = self._local.stack = []
        return st

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        stack = self._stack()
        parent = stack[-1] if stack else None
        stack.append(name)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            stack.pop()
            with self._lock:
                self.stages[name] = self.stages.get(name, 0.0) + dt
                if parent is not None:  # exclusive times: the parent adds dt when it ends
                    self.stages[parent] = self.stages.get(parent, 0.0) - dt

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


def peak_rss_mb() -> dict[str, float]:
    """Peak resident set of this process and of its largest waited-for child (COLMAP, OpenMVS).

    ``ru_maxrss`` is in bytes on macOS (kilobytes on Linux)."""
    unit = 1.0 if os.uname().sysname == "Darwin" else 1024.0
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit / 1e6
    kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * unit / 1e6
    return {"self": round(own, 1), "children": round(kids, 1)}


_current: Timings | None = None


@contextmanager
def collect() -> Iterator[Timings]:
    """Make a fresh collector current for the enclosed block (restores the previous one)."""
    global _current
    prev = _current
    _current = Timings()
    try:
        yield _current
    finally:
        _current = prev


def current() -> Timings | None:
    return _current


@contextmanager
def stage(name: str) -> Iterator[None]:
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
