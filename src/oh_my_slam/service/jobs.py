"""Update jobs (map.md §3, minimal until O3): created at once, run one at a time by a background
worker, never cancelled by a client disconnect.

States: ``queued`` → ``registering`` → ``committed``, or ``rejected`` (the quality gate or a
hard limit) or ``failed`` (an error). A job commits one version. Each job is kept as
``<data>/oms/jobs/<id>.json``, rewritten at every change, for ``JOB_KEEP_S`` after it ends; one
that was running when the service stopped is ``failed`` when it starts again."""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from oh_my_slam.core.atomic import atomic_write_json
from oh_my_slam.core.log import get_logger
from oh_my_slam.service.errors import ApiError
from oh_my_slam.service.schema import SCHEMA_VERSION
from oh_my_slam.service.store import utc

log = get_logger("oh_my_slam.service.jobs")

JOB_KEEP_S = 3600.0
MAX_WAIT_S = 30.0
ACTIVE = ("queued", "registering", "fusing")


@dataclass
class Job:
    id: str
    map: str
    created: float
    state: str = "queued"
    progress: float = 0.0
    eta_s: float | None = None
    base_version: int | None = None
    result_version: int | None = None
    inputs: list[dict[str, Any]] | None = None
    quality: dict[str, Any] | None = None
    scale_source: str | None = None
    scale_sigma_rel: float | None = None
    timings_s: dict[str, float] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    ended: float | None = None
    frames: int = 0  # frames sent
    started: float | None = None
    # the request, kept while the job runs (not part of the body)
    work: Any = None
    # the job as it was queued: the 202 answer, even when the worker has already started it
    queued_view: dict[str, Any] | None = None

    @property
    def active(self) -> bool:
        return self.state in ACTIVE

    def body(self) -> dict[str, Any]:
        """The ``Job`` body."""
        return {
            "schema": SCHEMA_VERSION, "job_id": self.id, "map": self.map, "state": self.state,
            "progress": self.progress, "eta_s": self.eta_s, "created_at": utc(self.created),
            "base_version": self.base_version, "provisional_version": None,
            "result_version": self.result_version, "latest_version": self.result_version,
            "ended_at": None if self.ended is None else utc(self.ended), "inputs": self.inputs,
            "quality": self.quality, "scale_source": self.scale_source,
            "scale_sigma_rel": self.scale_sigma_rel, "timings_s": self.timings_s,
            "error": self.error,
        }

    def record(self) -> dict[str, Any]:
        return {**self.body(), "created": self.created, "ended": self.ended,
                "frames": self.frames}

    @staticmethod
    def from_record(d: dict[str, Any]) -> Job:
        return Job(d["job_id"], d["map"], float(d["created"]), d["state"], float(d["progress"]),
                   d["eta_s"], d.get("base_version"), d["result_version"], d["inputs"],
                   d["quality"], d.get("scale_source"), d.get("scale_sigma_rel"),
                   d.get("timings_s", {}), d["error"], d.get("ended"), int(d.get("frames", 0)))


Runner = Callable[[Job, Callable[[], None]], None]


class Jobs:
    """The jobs of a service and its worker. ``runner(job, changed)`` runs a job, updating its
    fields and calling ``changed()`` after each state change; it ends the job in a final state
    (an exception fails it as ``internal``)."""

    def __init__(self, folder: Path, runner: Runner,
                 predict: Callable[[int], float] = lambda frames: 0.0) -> None:
        self.folder = folder
        self.runner = runner
        self.predict = predict  # seconds an update of so many frames takes
        self._jobs: dict[str, Job] = {}
        self._cond = threading.Condition()
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self.running: str | None = None

    # -- persistence -----------------------------------------------------------------------------

    def load(self) -> None:
        """The kept jobs of an earlier run: those it left running are failed."""
        now = time.time()
        for p in sorted(self.folder.glob("*.json")):
            try:
                job = Job.from_record(json.loads(p.read_text()))
            except (OSError, ValueError, KeyError, TypeError):
                p.unlink(missing_ok=True)
                continue
            if job.active:
                job.state, job.ended = "failed", now
                job.error = ApiError("internal", "the service stopped while this job ran; "
                                     "send the update again", {"stage": "service_stop"}).body()
                self._save(job)
            if job.ended is not None and now - job.ended > JOB_KEEP_S:
                p.unlink(missing_ok=True)
                continue
            self._jobs[job.id] = job

    def _save(self, job: Job) -> None:
        atomic_write_json(self.folder / f"{job.id}.json", job.record())

    def _expire(self) -> None:
        now = time.time()
        for jid in [j.id for j in self._jobs.values()
                    if j.ended is not None and now - j.ended > JOB_KEEP_S]:
            del self._jobs[jid]
            (self.folder / f"{jid}.json").unlink(missing_ok=True)

    # -- jobs ------------------------------------------------------------------------------------

    def submit(self, map_name: str, work: Any, frames: int = 0) -> Job:
        job = Job(uuid.uuid4().hex[:16], map_name, time.time(), work=work, frames=frames)
        with self._cond:
            self._expire()
            self._jobs[job.id] = job
            job.eta_s = self._eta(job)
            self._save(job)
            job.queued_view = self.view(job)
        self._queue.put(job.id)
        return job

    def get(self, job_id: str) -> Job:
        with self._cond:
            job = self._jobs.get(job_id)
        if job is None:
            raise ApiError("job_not_found", f"no job {job_id!r} (jobs are kept for an hour "
                           "after they end)", {"job_id": job_id})
        return job

    def _remaining(self, job: Job, now: float) -> float:
        own = self.predict(job.frames)
        return max(0.0, own - (now - job.started)) if job.started is not None else own

    def _eta(self, job: Job) -> float:
        """Seconds until ``job`` ends: the jobs ahead of it and its own predicted duration."""
        if not job.active:
            return 0.0
        now = time.time()
        if job.state != "queued":
            return round(self._remaining(job, now), 1)
        ahead = sum(self._remaining(j, now) for j in self._jobs.values()
                    if j.active and j is not job and (j.state != "queued" or j.created < job.created))
        return round(ahead + self._remaining(job, now), 1)

    def view(self, job: Job) -> dict[str, Any]:
        """The ``Job`` body with its current ``eta_s``."""
        with self._cond:
            job.eta_s = self._eta(job)
            return job.body()

    def changed(self, job: Job) -> None:
        with self._cond:
            if not job.active and job.ended is None:
                job.ended = time.time()
            job.eta_s = self._eta(job)
            self._save(job)
            self._cond.notify_all()

    def wait(self, job_id: str, wait_s: float) -> Job:
        """The job once its state differs from the current one, or after ``wait_s``."""
        job = self.get(job_id)
        state = job.state
        deadline = time.monotonic() + min(max(wait_s, 0.0), MAX_WAIT_S)
        with self._cond:
            while job.state == state and (left := deadline - time.monotonic()) > 0:
                self._cond.wait(left)
        return job

    def active_for(self, map_name: str) -> Job | None:
        """The map's running job, else its oldest queued one."""
        with self._cond:
            mine = sorted((j for j in self._jobs.values() if j.map == map_name and j.active),
                          key=lambda j: (j.state == "queued", j.created))
        return mine[0] if mine else None

    def queued(self) -> list[Job]:
        with self._cond:
            return sorted((j for j in self._jobs.values() if j.state == "queued"),
                          key=lambda j: j.created)

    # -- worker ----------------------------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._work, name="oms-jobs", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        """Stop the worker after the job it runs (queued jobs stay queued: they fail at the next
        start)."""
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout)

    def _work(self) -> None:
        while (jid := self._queue.get()) is not None:
            job = self._jobs.get(jid)
            if job is None or job.state != "queued":
                continue
            self.running = jid
            job.started = time.time()
            try:
                self.runner(job, lambda j=job: self.changed(j))  # type: ignore[misc]
            except Exception as exc:  # a bug: the job fails, the worker goes on
                log.exception("job %s failed", jid)
                job.state = "failed"
                job.error = ApiError("internal", f"internal error: {type(exc).__name__}: {exc}",
                                     {"stage": "job"}).body()
            finally:
                job.work = None
                self.running = None
                if job.active:  # a runner that returned without ending the job
                    job.state = "failed"
                    job.error = ApiError("internal", "the job ended without a result",
                                         {"stage": "job"}).body()
                self.changed(job)
