"""The order in which requests run (spec §2.6 "Requests"). There are no jobs: an operation runs
within its own HTTP request, and this module runs the command for it.

* **Process.** A request's command is its Python entry point (``python -m
  oh_my_slam.cli.<command>``, what ``scripts/_common.sh`` execs) run as a subprocess in its own
  process group, so the web process never loads the pipeline, a model or torch. Its stdout — the
  result — goes to a file of the request's folder (``Workspace.request_dir``), its timing record
  there too (``OH_MY_SLAM_TIMINGS``), and its stderr is read for the command's own error line.
* **Order.** Requests that use the inference server run one at a time, in arrival order (each
  takes a ticket once its body has arrived, and none overtakes an earlier one still being
  validated, which may need the inference server too), and wait for their turn with their
  connection open; the others start at once. Two requests never write the same map at once: a writer waits for the
  one before it. The scheduling state lives in the event loop's thread; a running command is
  waited for in a thread of its own, which hands the outcome back to the loop. ``GET
  /api/health`` counts the requests running and waiting, and lists each one (``in_progress``:
  operation, command line as typed, state, times), so a client tells whether its own request
  runs or waits for its turn.
* **Interruption.** A client that disconnects, and a stopping service, interrupt a request: a
  waiting one never starts; a running one gets SIGINT on its process group — Ctrl-C in a terminal
  — so an interrupted map update is the command's own uncommitted transaction. A command that
  ignores it gets SIGTERM after ``interrupt_grace_s`` and SIGKILL after as long again.
* **Uploads.** A request consumes the uploads it names: they are deleted when it ends, whatever its
  outcome, and an upload another request in progress uses is refused (``upload_in_use``).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Awaitable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from oh_my_slam.core.process import default_sigint
from oh_my_slam.core.timing import ENV_PATH
from oh_my_slam.web.workspace import Workspace

POLL_S = 0.1
LOG_TAIL = 40  # stderr lines kept for the command's error message
INTERRUPT_GRACE_S = 30.0
STDOUT = "stdout"  # the command's stdout in the request's folder: the result
TIMINGS = "timings.json"  # its timing record there (OH_MY_SLAM_TIMINGS)
DISCONNECTED = "disconnected"  # why a request was interrupted: its client left …
STOPPING = "stopping"  # … or the service stops


class RunError(Exception):
    """A request the runner refuses (``status``: the HTTP status)."""

    def __init__(self, status: int, message: str, code: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass(frozen=True)
class Outcome:
    """How a request's command ended: its exit status (negative: the signal that stopped it), the
    command's own error message, and why the request was interrupted, if it was."""

    code: int
    message: str | None = None
    interrupted: str | None = None


@dataclass(eq=False)
class Run:
    """One request of an operation, from its arrival to its end."""

    ticket: int  # arrival order
    prog: str  # the command's program, which names its error lines
    module: str
    argv: list[str]
    inference: bool
    writes: str | None = None  # the map folder it writes
    uploads: list[str] = field(default_factory=list)
    operation: str = ""  # the API operation id
    command: list[str] = field(default_factory=list)  # as typed, with workspace paths
    arrived_at: float = field(default_factory=time.time)
    started_at: float | None = None
    id: str = field(default_factory=lambda: secrets.token_hex(8))
    state: str = "waiting"  # waiting → running → ended
    interrupted: str | None = None
    proc: subprocess.Popen[bytes] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)  # guards proc and its reaping
    done: asyncio.Future[Outcome] | None = None


def _killpg(pgid: int, sig: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, sig)


class Runner:
    def __init__(self, workspace: Workspace, python: str = sys.executable,
                 interrupt_grace_s: float = INTERRUPT_GRACE_S) -> None:
        self.ws = workspace
        self.python = python
        self.interrupt_grace_s = interrupt_grace_s
        if signal.getsignal(signal.SIGINT) is signal.SIG_IGN:
            default_sigint()  # an ignored SIGINT would stay ignored in the commands: no interrupt
        self.stopping = False
        self.waiting: list[Run] = []  # by ticket
        self.running: list[Run] = []
        self._tickets = 0
        self._arriving: set[int] = set()  # tickets of requests still being validated
        self._in_use: dict[str, Run] = {}  # upload id → the request that consumes it
        self._threads: dict[str, threading.Thread] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    # -- state (the event loop's thread) -------------------------------------------------------------

    def ticket(self) -> int:
        """A request's place in the arrival order, taken when it arrives; :meth:`arrived` once it
        was validated (admitted or refused)."""
        self._tickets += 1
        self._arriving.add(self._tickets)
        return self._tickets

    def arrived(self, ticket: int) -> None:
        """A request's validation ended: the inference requests after it no longer wait for it."""
        if ticket in self._arriving:
            self._arriving.discard(ticket)
            self._schedule()

    def counts(self) -> dict[str, int]:
        return {"running": len(self.running), "waiting": len(self.waiting)}

    def in_progress(self) -> list[dict[str, Any]]:
        """The requests in progress in arrival order: each one's operation, command line (as
        typed, with workspace paths: what its validation answered), state and times, so a client
        can tell whether its own request runs or waits for its turn."""
        runs = sorted([*self.running, *self.waiting], key=lambda r: r.ticket)
        return [{"operation": r.operation, "command": r.command, "state": r.state,
                 "arrived_at": r.arrived_at, "started_at": r.started_at} for r in runs]

    def in_use(self, uid: str) -> bool:
        return uid in self._in_use

    def discard(self, uploads: list[str]) -> None:
        """A refused request ends: the uploads it named go too, unless another request uses
        them."""
        for uid in uploads:
            if uid not in self._in_use:
                self.ws.delete_upload(uid)

    def admit(self, run: Run) -> None:
        """Queue a valid request (in ticket order) and start what may start. Refused when the
        service stops or another request in progress consumes one of its uploads."""
        if self.stopping:
            raise RunError(503, "the service is stopping; send the request again once it runs",
                           "stopping")
        taken = [u for u in run.uploads if u in self._in_use]
        if taken:
            raise RunError(409, f"upload {taken[0]} is the input of another request in progress; "
                           "upload the file again", "upload_in_use")
        self._loop = asyncio.get_running_loop()
        run.done = self._loop.create_future()
        self._in_use.update(dict.fromkeys(run.uploads, run))
        self._arriving.discard(run.ticket)
        at = next((i for i, r in enumerate(self.waiting) if r.ticket > run.ticket),
                  len(self.waiting))
        self.waiting.insert(at, run)
        self._schedule()

    def _schedule(self) -> None:
        """Start every waiting request that may start: inference requests one at a time in ticket
        order (none overtakes an earlier one, waiting or still being validated), the others at
        once; never two writers of a map."""
        if self.stopping:
            return
        inference_busy = any(r.inference for r in self.running)
        validating = min(self._arriving, default=None)
        writing = {r.writes for r in self.running if r.writes}
        for run in list(self.waiting):
            earlier = validating is not None and validating < run.ticket
            blocked = (run.inference and (inference_busy or earlier)) or (run.writes in writing)
            inference_busy = inference_busy or run.inference  # nothing later overtakes it
            if run.writes:
                writing.add(run.writes)
            if not blocked:
                self._start(run)

    def _start(self, run: Run) -> None:
        self.waiting.remove(run)
        self.running.append(run)
        run.state = "running"
        run.started_at = time.time()
        t = threading.Thread(target=self._execute, args=(run,), name=f"request-{run.id}",
                             daemon=True)
        self._threads[run.id] = t
        t.start()

    def _ended(self, run: Run, outcome: Outcome) -> None:
        """A request's end (in the loop): its uploads are free, the next ones may start."""
        if run.state == "ended":
            return
        if run in self.waiting:
            self.waiting.remove(run)
        if run in self.running:
            self.running.remove(run)
        run.state = "ended"
        for uid in run.uploads:
            if self._in_use.get(uid) is run:
                del self._in_use[uid]
        if run.done is not None and not run.done.done():
            run.done.set_result(outcome)
        self._schedule()

    async def wait(self, run: Run, gone: Awaitable[None]) -> Outcome:
        """The outcome of an admitted request; ``gone`` completes when its client disconnects,
        which interrupts it. The request's task being cancelled (a service that stops before the
        command ended) interrupts it too."""
        assert run.done is not None
        done: asyncio.Future[Outcome] = run.done
        watch: asyncio.Future[None] = asyncio.ensure_future(gone)
        try:
            await asyncio.wait([watch, done], return_when=asyncio.FIRST_COMPLETED)  # type: ignore[type-var]
            if not done.done():
                self.interrupt(run, DISCONNECTED)
            return await asyncio.shield(done)
        except asyncio.CancelledError:
            self.interrupt(run, STOPPING)
            raise
        finally:
            watch.cancel()
            if watch.done() and not watch.cancelled():
                watch.exception()  # retrieved: a failed watch only ends the wait

    def interrupt(self, run: Run, why: str) -> None:
        """Interrupt a request as Ctrl-C would: a waiting one ends at once, a running one gets
        SIGINT (SIGTERM and SIGKILL later if it ignores it)."""
        if run.state == "ended" or run.interrupted is not None:
            return
        run.interrupted = why
        if run.state == "waiting":
            for uid in run.uploads:
                self.ws.delete_upload(uid)
            self._ended(run, Outcome(130, "interrupted before it started", why))
            return
        self._signal(run, signal.SIGINT)
        threading.Thread(target=self._escalate, args=(run,), daemon=True).start()

    def stop(self) -> None:
        """The service stops (in the loop): every request in progress is interrupted."""
        self.stopping = True
        for run in [*self.waiting, *self.running]:
            self.interrupt(run, STOPPING)

    # -- one command (its own thread) ----------------------------------------------------------------

    def _execute(self, run: Run) -> None:
        folder = self.ws.request_dir(run.id)
        try:
            outcome = self._command(run, folder)
        except Exception as exc:  # the request still ends, with the reason
            outcome = Outcome(1, f"the service could not run the command: {exc}", run.interrupted)
        finally:
            for uid in run.uploads:
                self.ws.delete_upload(uid)
        loop = self._loop
        try:
            if loop is None:
                raise RuntimeError("no loop")
            loop.call_soon_threadsafe(self._ended, run, outcome)
        except RuntimeError:  # the loop is closed (the service stopped): nobody waits for it
            shutil.rmtree(folder, ignore_errors=True)
        with contextlib.suppress(Exception):
            self._threads.pop(run.id, None)

    def _command(self, run: Run, folder: Path) -> Outcome:
        folder.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if k != ENV_PATH}
        env.update({ENV_PATH: str(folder / TIMINGS), "PYTHONUNBUFFERED": "1"})
        with run.lock:
            if run.interrupted is not None:  # interrupted before its command started
                return Outcome(130, "interrupted before it started", run.interrupted)
            try:
                with (folder / STDOUT).open("wb") as out:
                    run.proc = subprocess.Popen(
                        [self.python, "-m", run.module, *run.argv], stdin=subprocess.DEVNULL,
                        stdout=out, stderr=subprocess.PIPE, cwd=self.ws.root, env=env,
                        start_new_session=True)
            except OSError as exc:
                return Outcome(1, f"could not start the command: {exc}", run.interrupted)
        proc = run.proc
        assert proc.stderr is not None
        tail: deque[str] = deque(maxlen=LOG_TAIL)
        for raw in proc.stderr:
            tail.append(raw.decode("utf-8", "replace").rstrip("\n"))
        while True:
            with run.lock:  # reaped under the lock (see _signal)
                if proc.poll() is not None:
                    break
            time.sleep(POLL_S)
        code = proc.returncode
        return Outcome(code, None if code == 0 else self._message(run, list(tail), code),
                       run.interrupted)

    @staticmethod
    def _message(run: Run, tail: list[str], code: int) -> str:
        """The command's own error line (``<prog>: error: …``), else its last stderr line."""
        for line in reversed(tail):
            for tag in (f"{run.prog}: error: ", f"{run.prog}: internal error: "):
                if line.startswith(tag):
                    return line[len(tag):]
        if code < 0:
            return f"the command was stopped by signal {-code}"
        return tail[-1] if tail else f"the command exited with status {code}"

    # -- signals -------------------------------------------------------------------------------------

    def _signal(self, run: Run, sig: int) -> None:
        """Signal ``run``'s process group while its command runs (checked under its lock, so a
        reaped process group id is never signalled)."""
        with run.lock:
            if run.proc is not None and run.proc.poll() is None:
                _killpg(run.proc.pid, sig)

    def _escalate(self, run: Run) -> None:
        """A command that ignores SIGINT gets SIGTERM, then SIGKILL, ``interrupt_grace_s``
        apart."""
        for sig in (signal.SIGTERM, signal.SIGKILL):
            deadline = time.monotonic() + self.interrupt_grace_s
            while time.monotonic() < deadline:
                with run.lock:
                    if run.proc is None or run.proc.poll() is not None:
                        return
                time.sleep(POLL_S)
            self._signal(run, sig)

    def kill_all(self) -> None:
        """SIGKILL every running command's process group (a second stop signal)."""
        self.stopping = True
        for run in list(self.running):
            run.interrupted = run.interrupted or STOPPING
            self._signal(run, signal.SIGKILL)

    def shutdown(self) -> None:
        """After the HTTP server stopped: interrupt what still runs, wait for it — killed only
        after twice ``interrupt_grace_s`` and a little more — then delete every upload and what
        the requests left."""
        self.stopping = True
        for run in list(self.running):
            if run.interrupted is None:
                run.interrupted = STOPPING
                self._signal(run, signal.SIGINT)
                threading.Thread(target=self._escalate, args=(run,), daemon=True).start()
        deadline = time.monotonic() + 2 * self.interrupt_grace_s + 5
        for t in list(self._threads.values()):
            t.join(max(0.0, deadline - time.monotonic()))
        self.kill_all()
        for t in list(self._threads.values()):
            t.join(5.0)
        self.ws.clear_uploads()
        self.ws.clear_requests()
