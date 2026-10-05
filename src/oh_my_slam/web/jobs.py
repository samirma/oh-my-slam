"""Jobs (spec §2.6 "Jobs"): each runs the command's own Python entry point
(``python -m oh_my_slam.cli.<command>``, what ``scripts/_common.sh`` execs) as a subprocess in its
own process group, so the web process never loads the pipeline, a model or torch.

* **States** are those of ``core.errors.job_state``: ``queued`` → ``running`` → ``succeeded`` |
  ``failed`` | ``cancelled``; a failure carries the command's own message (its ``<prog>: error:``
  stderr line) and the code of its exit status.
* **Progress**: the command appends its timing events to ``OH_MY_SLAM_PROGRESS`` (a per-job
  ``progress.jsonl``); the runner follows that file, so the stage is the command's own stage name
  and ``done``/``total`` come from the command where it knows its size.
* **Timings and logs**: ``OH_MY_SLAM_TIMINGS`` (``timings.json``) and every stderr line
  (``stderr.log``), in the command's own form.
* **Order**: jobs that use (or may use) the inference server run one at a time, in submission
  order; the others start at once. Two jobs never write the same map at once.
* **Cancel**: SIGINT to the job's process group — Ctrl-C in a terminal — so a cancelled map update
  is the command's own interrupted, uncommitted transaction.
* **Results**: the command writes its result with ``-o`` and its artefacts with ``-d`` into the
  job's ``out/`` folder, so they are the command's bytes. A command whose output is the browser
  (``view.sh``) succeeds once its viewer listens; that viewer process stays open (the latest
  ``MAX_VIEWERS``) and the service proxies it.
* **Persistence**: ``jobs/<id>/job.json`` per job; the list survives a restart, and a job that was
  queued or running when the service stopped is ``cancelled``.
* **Uploads** a job consumes are deleted when it ends, whatever its state.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from oh_my_slam.core.atomic import atomic_write_json
from oh_my_slam.core.errors import error_code, http_status, job_state
from oh_my_slam.core.timing import ENV_PATH, ENV_PROGRESS
from oh_my_slam.web.operations import OUT_DIR, Operation, Prepared
from oh_my_slam.web.workspace import Workspace

TERMINAL = frozenset({"succeeded", "failed", "cancelled"})
MAX_VIEWERS = 4  # viewer processes kept open (the oldest closes first)
LOG_TAIL = 40  # stderr lines kept in the job record (all of them in stderr.log)
POLL_S = 0.1
_LISTENING = re.compile(r"listening on (http://\S+/)")


@dataclass
class Job:
    id: str
    operation: str  # Operation.id
    label: str  # the command as typed (segment.sh -i)
    params: dict[str, Any]  # as submitted
    command: list[str]  # the command line, with workspace paths
    argv: list[str]  # what the subprocess runs after ``python -m <module>``
    module: str
    prog: str
    inference: bool
    browser: bool
    uploads: list[str] = field(default_factory=list)
    writes: str | None = None
    result_name: str | None = None
    result_format: str | None = None
    state: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    ended_at: float | None = None
    stage: str | None = None
    progress: dict[str, Any] | None = None
    stages: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, Any] = field(default_factory=dict)
    exit_code: int | None = None
    error: dict[str, Any] | None = None
    viewer: str | None = None
    log_tail: list[str] = field(default_factory=list)
    resubmitted_from: str | None = None
    cancel_requested: bool = False
    version: int = 0

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("argv", "module", "version"):
            d.pop(k)
        d["result"] = None
        if self.state == "succeeded" and self.result_name is not None:
            d["result"] = {"name": self.result_name, "format": self.result_format,
                           "url": f"/api/jobs/{self.id}/result"}
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Job:
        known = {f.name for f in fields(Job)}
        return Job(**{k: v for k, v in d.items() if k in known})


class JobError(Exception):
    """A job request the runner refuses (``status``: the HTTP status)."""

    def __init__(self, status: int, message: str, code: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


def _new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + secrets.token_hex(3)


class Runner:
    def __init__(self, workspace: Workspace, python: str = sys.executable,
                 stop_grace_s: float = 120.0, max_viewers: int = MAX_VIEWERS) -> None:
        self.ws = workspace
        self.python = python
        self.stop_grace_s = stop_grace_s
        self.max_viewers = max_viewers
        self.jobs: dict[str, Job] = {}
        self.version = 0
        self.stopping = False
        self._lock = threading.RLock()
        self._procs: dict[str, subprocess.Popen[bytes]] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._viewers: OrderedDict[str, str] = OrderedDict()  # job id → local viewer URL

    # -- records -----------------------------------------------------------------------------------

    def _touch(self, job: Job, save: bool = False) -> None:
        """Mark ``job`` changed (the SSE streams send it); persist it when ``save``."""
        with self._lock:
            self.version += 1
            job.version = self.version
            if save:
                d = self.ws.job_dir(job.id)
                d.mkdir(parents=True, exist_ok=True)
                atomic_write_json(d / "job.json", asdict(job))

    def load(self) -> None:
        """Read the persisted job list; a job that was queued or running when the service stopped
        is cancelled, and a viewer that was open is closed."""
        for f in sorted(self.ws.jobs.glob("*/job.json")):
            try:
                job = Job.from_dict(json.loads(f.read_text()))
            except (OSError, ValueError, TypeError):
                continue
            changed = False
            if job.state not in TERMINAL:
                job.state, job.ended_at, changed = "cancelled", job.ended_at or time.time(), True
                job.error = {"code": "interrupted", "message": "the service stopped before the "
                             "job finished; re-submit it"}
            if job.viewer is not None:
                job.viewer, changed = None, True
            self.jobs[job.id] = job
            self._touch(job, save=changed)

    def get(self, jid: str) -> Job:
        with self._lock:
            job = self.jobs.get(jid)
        if job is None:
            raise JobError(404, f"no job {jid}", "not_found")
        return job

    def all_jobs(self) -> list[Job]:
        with self._lock:
            return sorted(self.jobs.values(), key=lambda j: (j.created_at, j.id))

    def counts(self) -> dict[str, int]:
        with self._lock:
            states = [j.state for j in self.jobs.values()]
        return {s: states.count(s) for s in ("queued", "running")}

    def changed_since(self, seen: int, jid: str | None = None) -> tuple[int, list[dict[str, Any]]]:
        with self._lock:
            jobs = [j for j in self.jobs.values() if j.version > seen and (jid in (None, j.id))]
            return self.version, [j.public() for j in sorted(jobs, key=lambda j: j.version)]

    def viewer_url(self, jid: str) -> str | None:
        with self._lock:
            return self._viewers.get(jid)

    # -- submission and scheduling -------------------------------------------------------------------

    def submit(self, op: Operation, params: dict[str, Any], prep: Prepared, jid: str,
               resubmitted_from: str | None = None) -> Job:
        with self._lock:
            if self.stopping:
                raise JobError(503, "the service is stopping", "stopping")
            busy = {u: j.id for j in self.jobs.values() if j.state not in TERMINAL
                    for u in j.uploads}
            taken = [u for u in prep.uploads if u in busy]
            if taken:
                raise JobError(409, f"upload {taken[0]} is already the input of job "
                               f"{busy[taken[0]]}; upload the file again", "upload_in_use")
            job = Job(id=jid, operation=op.id, label=op.label, params=params,
                      command=prep.command, argv=prep.argv, module=op.module,
                      prog=op.program.prog, inference=op.uses_inference, browser=op.browser,
                      uploads=list(dict.fromkeys(prep.uploads)), writes=prep.writes,
                      result_name=prep.result, result_format=prep.result_format,
                      resubmitted_from=resubmitted_from)
            self.jobs[jid] = job
            self._touch(job, save=True)
            self._schedule()
            return job

    @staticmethod
    def new_id() -> str:
        return _new_id()

    def _schedule(self) -> None:
        """Start every queued job that may start: inference jobs one at a time in submission
        order (none overtakes an earlier one), the others at once; never two writers of a map."""
        with self._lock:
            if self.stopping:
                return
            running = [j for j in self.jobs.values() if j.state == "running"]
            inference_busy = any(j.inference for j in running)
            writing = {j.writes for j in running if j.writes}
            for job in self.all_jobs():
                if job.state != "queued":
                    continue
                if job.writes and job.writes in writing:
                    inference_busy = inference_busy or job.inference  # keeps the order
                    continue
                if job.inference:
                    if inference_busy:
                        continue
                    inference_busy = True
                if job.writes:
                    writing.add(job.writes)
                self._start(job)

    def _start(self, job: Job) -> None:
        job.state, job.started_at = "running", time.time()
        self._touch(job, save=True)
        t = threading.Thread(target=self._run, args=(job,), name=f"job-{job.id}", daemon=True)
        self._threads[job.id] = t
        t.start()

    # -- one job -----------------------------------------------------------------------------------

    def _run(self, job: Job) -> None:
        d = self.ws.job_dir(job.id)
        (d / OUT_DIR).mkdir(parents=True, exist_ok=True)
        progress = d / "progress.jsonl"
        progress.write_bytes(b"")
        env = {**os.environ, ENV_PROGRESS: str(progress), ENV_PATH: str(d / "timings.json"),
               "PYTHONUNBUFFERED": "1"}
        try:
            with (d / "stdout").open("wb") as out:
                proc = subprocess.Popen([self.python, "-m", job.module, *job.argv],
                                        stdin=subprocess.DEVNULL, stdout=out,
                                        stderr=subprocess.PIPE, cwd=self.ws.root, env=env,
                                        start_new_session=True)
        except OSError as exc:
            self._finish(job, 1, f"could not start the command: {exc}")
            return
        with self._lock:
            self._procs[job.id] = proc
            if job.cancel_requested:
                self._interrupt(proc)
        reader = threading.Thread(target=self._read_stderr, args=(job, proc, d / "stderr.log"),
                                  daemon=True)
        reader.start()
        with progress.open("rb") as events:
            pending = b""
            while True:
                done = proc.poll() is not None
                chunk = events.read()
                if chunk:
                    pending += chunk
                    *lines, pending = pending.split(b"\n")
                    for line in lines:
                        self._event(job, line)
                elif done:
                    break
                else:
                    time.sleep(POLL_S)
        reader.join()
        with self._lock:
            self._procs.pop(job.id, None)
            self._viewers.pop(job.id, None)
        code = proc.returncode
        if job.state in TERMINAL:  # a viewer that closed after it had succeeded
            if job.viewer is not None:
                job.viewer = None
                self._touch(job, save=True)
            return
        self._finish(job, code)

    def _event(self, job: Job, line: bytes) -> None:
        try:
            ev = json.loads(line)
        except ValueError:
            return
        kind = ev.get("event")
        with self._lock:
            if kind == "stage_start":
                job.stage, job.progress = ev.get("stage"), None
            elif kind == "stage_end":
                job.stages.append({"stage": ev.get("stage"), "seconds": ev.get("seconds")})
            elif kind == "progress":
                job.progress = {k: ev.get(k) for k in ("stage", "done", "total")}
            elif kind == "count":
                job.counts.update({k: v for k, v in ev.items() if k != "event"})
            else:
                return
            self._touch(job)

    def _read_stderr(self, job: Job, proc: subprocess.Popen[bytes], path: Path) -> None:
        assert proc.stderr is not None
        with path.open("ab") as log:
            for raw in proc.stderr:
                log.write(raw)
                log.flush()
                line = raw.decode("utf-8", "replace").rstrip("\n")
                with self._lock:
                    job.log_tail = [*job.log_tail[-(LOG_TAIL - 1):], line]
                    self._touch(job)
                m = _LISTENING.search(line) if job.browser else None
                if m and job.state == "running" and line.startswith(f"{job.prog}: "):
                    self._listening(job, m.group(1))

    def _listening(self, job: Job, url: str) -> None:
        """A viewer job's command listens: its result (the viewer) is ready."""
        with self._lock:
            self._viewers[job.id] = url
            job.viewer = f"/viewer/job/{job.id}/"
            job.state, job.ended_at, job.exit_code = "succeeded", time.time(), None
            self._end(job)
            while len(self._viewers) > self.max_viewers:
                oldest, _ = self._viewers.popitem(last=False)
                proc = self._procs.get(oldest)
                if proc is not None:
                    self._interrupt(proc)

    def _finish(self, job: Job, code: int, message: str | None = None) -> None:
        state = job_state(code)
        if job.cancel_requested and state != "succeeded":
            state = "cancelled"
        if job.browser and state == "succeeded":  # exited without ever listening
            state = "cancelled" if job.cancel_requested else "failed"
            message = message or "the viewer stopped before it was ready"
        d = self.ws.job_dir(job.id)
        with self._lock:
            job.state, job.exit_code, job.ended_at = state, code, time.time()
            if state == "failed":
                job.error = {"code": error_code(code), "exit_code": code,
                             "http_status": http_status(code),
                             "message": message or self._message(job, code)}
            elif state == "cancelled":
                job.error = {"code": "interrupted", "exit_code": code,
                             "http_status": http_status(130), "message": "cancelled"}
            if state == "succeeded" and job.result_name is not None \
                    and not (d / OUT_DIR / job.result_name).is_file():
                job.result_name = None
            self._end(job)

    def _end(self, job: Job) -> None:
        """Bookkeeping when a job ends: its uploads go, its record is saved, the queue moves."""
        for uid in job.uploads:
            self.ws.delete_upload(uid)
        self._touch(job, save=True)
        self._schedule()

    @staticmethod
    def _message(job: Job, code: int) -> str:
        """The command's own error line (``<prog>: error: …``), else the last stderr line."""
        for line in reversed(job.log_tail):
            for tag in (f"{job.prog}: error: ", f"{job.prog}: internal error: "):
                if line.startswith(tag):
                    return line[len(tag):]
        if code < 0:
            return f"the command was stopped by signal {-code}"
        return job.log_tail[-1] if job.log_tail else f"the command exited with status {code}"

    def timings(self, jid: str) -> Any:
        p = self.ws.job_dir(self.get(jid).id) / "timings.json"
        return json.loads(p.read_text()) if p.is_file() else None

    # -- cancellation and stop ---------------------------------------------------------------------

    @staticmethod
    def _interrupt(proc: subprocess.Popen[bytes], sig: int = signal.SIGINT) -> None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, sig)

    def cancel(self, jid: str) -> Job:
        with self._lock:
            job = self.get(jid)
            if job.state == "queued":
                job.state, job.ended_at = "cancelled", time.time()
                job.error = {"code": "interrupted", "message": "cancelled before it started"}
                self._end(job)
            elif job.state == "running":
                job.cancel_requested = True
                proc = self._procs.get(jid)
                if proc is not None:
                    self._interrupt(proc)
                self._touch(job, save=True)
            else:
                raise JobError(409, f"job {jid} is {job.state}; only a queued or running job can "
                               "be cancelled", "not_cancellable")
            return job

    def wait(self, jid: str, timeout: float | None = None) -> Job:
        """Block until ``jid`` ends (tests and tools)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.get(jid).state not in TERMINAL:
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f"job {jid} still {self.get(jid).state}")
            time.sleep(0.05)
        return self.get(jid)

    def shutdown(self) -> None:
        """Stop: queued jobs are cancelled, running ones interrupted (SIGINT, as Ctrl-C) and
        waited for — killed only after ``stop_grace_s`` — viewers closed, uploads deleted."""
        with self._lock:
            self.stopping = True
            for job in self.all_jobs():
                if job.state == "queued":
                    job.state, job.ended_at = "cancelled", time.time()
                    job.error = {"code": "interrupted", "message": "the service stopped"}
                    self._end(job)
                elif job.state == "running":
                    job.cancel_requested = True
            procs = dict(self._procs)
            threads = list(self._threads.values())
        for proc in procs.values():
            self._interrupt(proc)
        deadline = time.monotonic() + self.stop_grace_s
        for t in threads:
            t.join(max(0.0, deadline - time.monotonic()))
        with self._lock:
            left = dict(self._procs)
        for proc in left.values():
            self._interrupt(proc, signal.SIGKILL)
        for t in threads:
            t.join(5.0)
        self.ws.clear_uploads()
