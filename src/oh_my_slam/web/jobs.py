"""Jobs (spec §2.6 "Jobs"): each runs the command's own Python entry point
(``python -m oh_my_slam.cli.<command>``, what ``scripts/_common.sh`` execs) as a subprocess in its
own process group, so the web process never loads the pipeline, a model or torch. A job that shows
a viewer has a second step, the viewer's bundle writer (``web.operations``).

* **States** are those of ``core.errors.job_state``: ``queued`` → ``running`` → ``succeeded`` |
  ``failed`` | ``cancelled``; a failure carries the command's own message (its ``<prog>: error:``
  stderr line) and the code of its exit status.
* **Progress**: the command appends its timing events to ``OH_MY_SLAM_PROGRESS`` (a per-job
  ``progress.jsonl``); the runner follows that file, so the stage is the command's own stage name
  and ``done``/``total`` come from the command where it knows its size.
* **Timings and logs**: ``OH_MY_SLAM_TIMINGS`` (``timings.json``) and every stderr line
  (``stderr.log``), in the command's own form.
* **Order**: jobs that use the inference server (``spec.needs_inference``) run one at a time, in
  submission order; the others start at once, at most ``max_parallel`` together. Two jobs never
  write the same map at once.
* **Cancel**: SIGINT to the job's process group — Ctrl-C in a terminal — so a cancelled map update
  is the command's own interrupted, uncommitted transaction; a command that ignores it gets
  SIGTERM after ``cancel_grace_s`` and SIGKILL after as long again (its atomic commit still keeps
  the map whole).
* **Results**: the command writes its result with ``-o`` and its artefacts with ``-d`` into the
  job's ``out/`` folder, so they are the command's bytes; a viewer is saved in ``viewer/``.
* **Persistence**: ``jobs/<id>/job.json`` per job; the list survives a restart, and a job that was
  queued or running when the service stopped is ``cancelled``. The record names the running
  step's process group with its leader's start time, and the leader carries the job id in its
  environment (``ENV_JOB``): a group left behind by a crash is killed at the next start only when
  its leader still matches both, so a recycled process id is never signalled.
* **Uploads** a job consumes are deleted when it ends, whatever its state.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from oh_my_slam.core.atomic import atomic_write_json
from oh_my_slam.core.errors import error_code, http_status, job_state
from oh_my_slam.core.timing import ENV_PATH, ENV_PROGRESS
from oh_my_slam.web.operations import OUT_DIR, VIEWER_DIR, Operation, Prepared
from oh_my_slam.web.workspace import Workspace

TERMINAL = frozenset({"succeeded", "failed", "cancelled"})
LOG_TAIL = 40  # stderr lines kept in the job record (all of them in stderr.log)
POLL_S = 0.1
LOG_TOUCH_S = 0.25  # stderr lines mark a job changed at most this often
CANCEL_GRACE_S = 30.0
ENV_JOB = "OH_MY_SLAM_JOB"  # the job id, in the environment of its steps


def leader_matches(pgid: int, ctime: float | None, jid: str) -> bool:
    """Whether process ``pgid`` is still the leader a job started: same start time, and the job's
    id in its environment. Anything that cannot be checked does not match."""
    if ctime is None:
        return False
    try:
        import psutil

        proc = psutil.Process(pgid)
        return abs(proc.create_time() - ctime) < 1e-3 and proc.environ().get(ENV_JOB) == jid
    except Exception:  # gone, another user's, unreadable
        return False


def _ctime(pid: int) -> float | None:
    try:
        import psutil

        return float(psutil.Process(pid).create_time())
    except Exception:
        return None


def default_parallel() -> int:
    return max(1, (os.cpu_count() or 2) // 2)


@dataclass
class Job:
    id: str
    operation: str  # Operation.id
    label: str  # the command as typed (segment.sh -i)
    params: dict[str, Any]  # as submitted
    command: list[str]  # the command line, with workspace paths
    steps: list[dict[str, Any]]  # web.operations.Step as data
    inference: bool
    conditional: bool = False  # inference depends on what the job reads (re-checked at start)
    actual: dict[str, Any] = field(default_factory=dict)  # the parameters the command gets
    uploads: list[str] = field(default_factory=list)
    writes: str | None = None
    result_name: str | None = None
    result_format: str | None = None
    saves_viewer: bool = False
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
    pgid: int | None = None  # process group of the running step
    leader_ctime: float | None = None  # its leader's start time (psutil create_time)
    version: int = 0

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("steps", "version", "pgid", "leader_ctime", "actual"):
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


def _killpg(pgid: int, sig: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, sig)


class Runner:
    def __init__(self, workspace: Workspace, python: str = sys.executable,
                 stop_grace_s: float = 120.0, cancel_grace_s: float = CANCEL_GRACE_S,
                 max_parallel: int | None = None,
                 reevaluate: Callable[[Job], bool] | None = None) -> None:
        self.ws = workspace
        self.python = python
        self.stop_grace_s = stop_grace_s
        self.cancel_grace_s = cancel_grace_s
        self.max_parallel = max_parallel or default_parallel()
        self.reevaluate = reevaluate  # a conditional job's inference need, when it may start
        self.jobs: dict[str, Job] = {}
        self.version = 0
        self.stopping = False
        self._lock = threading.RLock()
        self._procs: dict[str, subprocess.Popen[bytes]] = {}
        self._threads: dict[str, threading.Thread] = {}

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
        """Read the persisted job list. A job that was queued or running when the service stopped
        is cancelled; a process group it left behind (the service was killed) is killed."""
        for f in sorted(self.ws.jobs.glob("*/job.json")):
            try:
                job = Job.from_dict(json.loads(f.read_text()))
            except (OSError, ValueError, TypeError):
                continue
            changed = False
            if job.state not in TERMINAL:
                if job.pgid is not None and leader_matches(job.pgid, job.leader_ctime, job.id):
                    _killpg(job.pgid, signal.SIGKILL)
                job.state, job.ended_at, changed = "cancelled", job.ended_at or time.time(), True
                job.error = {"code": "interrupted", "message": "the service stopped before the "
                             "job finished; re-submit it"}
            if job.pgid is not None or job.leader_ctime is not None:
                job.pgid, job.leader_ctime, changed = None, None, True
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

    def viewer_dir(self, jid: str) -> Path | None:
        """The saved viewer of a succeeded job."""
        job = self.get(jid)
        d = self.ws.job_dir(job.id) / VIEWER_DIR
        return d if job.viewer is not None and d.is_dir() else None

    # -- submission and scheduling -------------------------------------------------------------------

    @staticmethod
    def new_id() -> str:
        return time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + secrets.token_hex(3)

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
                      command=prep.command, steps=[asdict(s) for s in prep.steps],
                      inference=prep.inference, conditional=prep.conditional,
                      actual=prep.actual, uploads=list(dict.fromkeys(prep.uploads)),
                      writes=prep.writes, result_name=prep.result,
                      result_format=prep.result_format, saves_viewer=prep.viewer,
                      resubmitted_from=resubmitted_from)
            self.jobs[jid] = job
            self._touch(job, save=True)
            self._schedule()
            return job

    def _schedule(self) -> None:
        """Start every queued job that may start: inference jobs one at a time in submission
        order (none overtakes an earlier one), the others at once up to ``max_parallel``; never
        two writers of a map."""
        with self._lock:
            if self.stopping:
                return
            running = [j for j in self.jobs.values() if j.state == "running"]
            inference_busy = any(j.inference for j in running)
            others = sum(not j.inference for j in running)
            writing = {j.writes for j in running if j.writes}
            for job in self.all_jobs():
                if job.state != "queued":
                    continue
                if job.conditional and self.reevaluate is not None:
                    job.inference = self.reevaluate(job)  # what it reads may have changed
                if job.writes and job.writes in writing:
                    inference_busy = inference_busy or job.inference  # keeps the order
                    continue
                if job.inference:
                    if inference_busy:
                        continue
                    inference_busy = True
                elif others >= self.max_parallel:
                    continue
                else:
                    others += 1
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
        try:
            code, message = 0, None
            for step in job.steps:
                if job.cancel_requested:
                    code = -int(signal.SIGINT)
                    break
                code, message = self._step(job, step)
                if code != 0:
                    break
            self._finish(job, code, message)
        finally:
            with self._lock:
                self._threads.pop(job.id, None)

    def _step(self, job: Job, step: dict[str, Any]) -> tuple[int, str | None]:
        d = self.ws.job_dir(job.id)
        (d / OUT_DIR).mkdir(parents=True, exist_ok=True)
        progress = d / "progress.jsonl"
        progress.touch()
        offset = progress.stat().st_size  # this step's events start here
        env = {**os.environ, **step.get("env", {}), ENV_PROGRESS: str(progress),
               ENV_JOB: job.id, "PYTHONUNBUFFERED": "1"}
        env.pop(ENV_PATH, None)
        if step.get("timings", True):
            env[ENV_PATH] = str(d / "timings.json")
        try:
            with (d / "stdout").open("ab") as out:
                proc = subprocess.Popen([self.python, "-m", step["module"], *step["argv"]],
                                        stdin=subprocess.DEVNULL, stdout=out,
                                        stderr=subprocess.PIPE, cwd=self.ws.root, env=env,
                                        start_new_session=True)
        except OSError as exc:
            return 1, f"could not start the command: {exc}"
        with self._lock:
            self._procs[job.id] = proc
            job.pgid, job.leader_ctime = proc.pid, _ctime(proc.pid)
            self._touch(job, save=True)
            if job.cancel_requested:
                self._signal(proc, signal.SIGINT)
        reader = threading.Thread(target=self._read_stderr, args=(job, proc, d / "stderr.log"),
                                  daemon=True)
        reader.start()
        with progress.open("rb") as events:
            events.seek(offset)
            pending = b""
            while True:
                with self._lock:  # reaped under the lock (see _signal)
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
            job.pgid = job.leader_ctime = None
        return proc.returncode, None

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
        last = 0.0
        with path.open("ab") as log:
            for raw in proc.stderr:
                log.write(raw)
                log.flush()
                line = raw.decode("utf-8", "replace").rstrip("\n")
                with self._lock:
                    job.log_tail = [*job.log_tail[-(LOG_TAIL - 1):], line]
                    if time.monotonic() - last >= LOG_TOUCH_S:
                        last = time.monotonic()
                        self._touch(job)

    def _finish(self, job: Job, code: int, message: str | None = None) -> None:
        state = job_state(code)
        if job.cancel_requested and state != "succeeded":
            state = "cancelled"
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
            if state == "succeeded":
                if job.result_name is not None and not (d / OUT_DIR / job.result_name).is_file():
                    job.result_name = None
                if job.saves_viewer and (d / VIEWER_DIR).is_dir():
                    job.viewer = f"/viewer/job/{job.id}/"
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
        progs = {s["prog"] for s in job.steps}
        for line in reversed(job.log_tail):
            for prog in progs:
                for tag in (f"{prog}: error: ", f"{prog}: internal error: "):
                    if line.startswith(tag):
                        return line[len(tag):]
        if code < 0:
            return f"the command was stopped by signal {-code}"
        return job.log_tail[-1] if job.log_tail else f"the command exited with status {code}"

    def timings(self, jid: str) -> Any:
        p = self.ws.job_dir(self.get(jid).id) / "timings.json"
        return json.loads(p.read_text()) if p.is_file() else None

    # -- cancellation and stop ---------------------------------------------------------------------

    def _signal(self, proc: subprocess.Popen[bytes], sig: int) -> None:
        """Signal ``proc``'s process group while it runs (checked under the lock, so a reaped
        process group id is never signalled)."""
        with self._lock:
            if proc.poll() is None:
                _killpg(proc.pid, sig)

    def _escalate(self, proc: subprocess.Popen[bytes]) -> None:
        """A command that ignores SIGINT gets SIGTERM, then SIGKILL, ``cancel_grace_s`` apart."""
        for sig in (signal.SIGTERM, signal.SIGKILL):
            deadline = time.monotonic() + self.cancel_grace_s
            while time.monotonic() < deadline:
                with self._lock:
                    if proc.poll() is not None:
                        return
                time.sleep(POLL_S)
            self._signal(proc, sig)

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
                    self._signal(proc, signal.SIGINT)
                    threading.Thread(target=self._escalate, args=(proc,), daemon=True).start()
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

    def kill_all(self) -> None:
        """SIGKILL every running job's process group (a second stop signal) and record those jobs
        as cancelled, with no process group left to recover, before the process may exit."""
        with self._lock:
            self.stopping = True
            for proc in self._procs.values():
                self._signal(proc, signal.SIGKILL)
            for job in self.jobs.values():
                if job.state == "running":
                    job.cancel_requested = True
                    job.state, job.ended_at = "cancelled", time.time()
                    job.pgid = job.leader_ctime = None
                    job.error = {"code": "interrupted", "message": "killed: the service was "
                                 "stopped twice"}
                    for uid in job.uploads:
                        self.ws.delete_upload(uid)
                    self._touch(job, save=True)

    def shutdown(self) -> None:
        """Stop: queued jobs are cancelled, running ones interrupted (SIGINT, as Ctrl-C) and
        waited for — killed only after ``stop_grace_s`` — and uploads deleted."""
        with self._lock:
            self.stopping = True
            for job in self.all_jobs():
                if job.state == "queued":
                    job.state, job.ended_at = "cancelled", time.time()
                    job.error = {"code": "interrupted", "message": "the service stopped"}
                    self._end(job)
                elif job.state == "running":
                    job.cancel_requested = True
            procs = list(self._procs.values())
            threads = list(self._threads.values())
        for proc in procs:
            self._signal(proc, signal.SIGINT)
        deadline = time.monotonic() + self.stop_grace_s
        for t in threads:
            t.join(max(0.0, deadline - time.monotonic()))
        self.kill_all()
        for t in threads:
            t.join(5.0)
        self.ws.clear_uploads()
