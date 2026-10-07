"""The order in which ``server.sh`` runs requests (spec §2.6 "Requests") at its edges: a service
started with SIGINT ignored still interrupts its commands, a request ends once and never frees an
upload another request consumes, a request whose task is cancelled (the service stopping) is
interrupted, an interrupt is sent once, a command that cannot start or whose request folder cannot
be made ends with the reason, an outcome nobody waits for any more is cleaned up, a command deaf
to SIGINT and SIGTERM is killed, and shutting down interrupts and waits for what still runs."""

from __future__ import annotations

import asyncio
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from oh_my_slam.web.runner import STOPPING, Outcome, Run, Runner
from oh_my_slam.web.workspace import Workspace
from tests.fakes import slow_command
from tests.unit.test_web_api import _repo_importable  # noqa: F401


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    w = Workspace(tmp_path / "data")
    w.create()
    return w


def slow(seconds: float, *extra: str, uploads: list[str] | None = None, ticket: int = 1) -> Run:
    return Run(ticket, "slow.sh", slow_command.MODULE, [f"--seconds={seconds:g}", *extra],
               inference=False, uploads=list(uploads or []))


def until(check: Callable[[], bool], timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


def test_a_service_started_with_sigint_ignored_gives_its_commands_ctrl_c(ws: Workspace) -> None:
    old = signal.signal(signal.SIGINT, signal.SIG_IGN)  # e.g. started as a shell `&` job
    try:
        Runner(ws)
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    finally:
        signal.signal(signal.SIGINT, old)


def test_a_request_ends_once_and_never_frees_an_upload_another_consumes(ws: Workspace) -> None:
    async def scenario() -> None:
        runner = Runner(ws)
        uid, target = ws.new_upload("a.jpg")
        target.write_bytes(b"x")
        owner = slow(60, uploads=[uid])
        runner.admit(owner)  # it runs, and consumes the upload
        stranger = slow(0, uploads=[uid], ticket=2)  # never admitted
        runner._ended(stranger, Outcome(0))
        assert runner.in_use(uid) and stranger.state == "ended"
        runner.interrupt(owner, STOPPING)
        outcome = await asyncio.wait_for(asyncio.shield(owner.done), 30)  # type: ignore[arg-type]
        assert outcome.interrupted == STOPPING and not runner.in_use(uid)
        runner._ended(owner, Outcome(0))  # once ended, ending it again changes nothing
        assert owner.done is not None and owner.done.result() is outcome
        runner.interrupt(owner, "disconnected")  # nor does interrupting it
        assert owner.interrupted == STOPPING
        runner.shutdown()

    asyncio.run(scenario())


def test_an_interrupt_is_sent_once(ws: Workspace) -> None:
    async def scenario() -> None:
        runner = Runner(ws, interrupt_grace_s=0.2)
        run = slow(60, "--ignore-sigint")
        runner.admit(run)
        await asyncio.to_thread(until, lambda: run.proc is not None)
        runner.interrupt(run, "disconnected")
        runner.interrupt(run, STOPPING)  # already interrupted: no second SIGINT, same reason
        outcome = await asyncio.wait_for(asyncio.shield(run.done), 30)  # type: ignore[arg-type]
        assert outcome.interrupted == "disconnected" and outcome.code == -signal.SIGTERM
        runner.shutdown()

    asyncio.run(scenario())


def test_a_request_whose_task_is_cancelled_is_interrupted(ws: Workspace) -> None:
    """A service that stops before a command ended cancels the request's task: the command is
    interrupted as Ctrl-C would, and the task stays cancelled."""
    async def scenario() -> None:
        runner = Runner(ws)
        run = slow(60)
        runner.admit(run)
        await asyncio.to_thread(until, lambda: run.proc is not None)
        never = asyncio.get_running_loop().create_future()
        task = asyncio.ensure_future(runner.wait(run, never))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert run.interrupted == STOPPING
        outcome = await asyncio.wait_for(asyncio.shield(run.done), 30)  # type: ignore[arg-type]
        assert outcome.code == 130 and outcome.interrupted == STOPPING
        runner.shutdown()

    asyncio.run(scenario())


def test_a_request_whose_folder_cannot_be_made_ends_with_the_reason(ws: Workspace) -> None:
    ws.requests.write_text("not a folder")  # a broken workspace

    async def scenario() -> None:
        runner = Runner(ws)
        uid, target = ws.new_upload("a.jpg")
        target.write_bytes(b"x")
        run = slow(0, uploads=[uid])
        runner.admit(run)
        outcome = await asyncio.wait_for(asyncio.shield(run.done), 30)  # type: ignore[arg-type]
        assert outcome.code == 1 and outcome.message is not None
        assert outcome.message.startswith("the service could not run the command: ")
        assert not (ws.uploads / uid).exists()  # its upload goes all the same

    asyncio.run(scenario())


def test_a_command_that_cannot_start_ends_with_the_reason(ws: Workspace) -> None:
    runner = Runner(ws, python=str(ws.root / "no-such-python"))
    run = slow(0)
    outcome = runner._command(run, ws.request_dir(run.id))
    assert outcome.code == 1 and outcome.message is not None
    assert outcome.message.startswith("could not start the command: ") and run.proc is None


def test_a_request_interrupted_before_its_command_started_never_starts_it(ws: Workspace) -> None:
    runner = Runner(ws)
    run = slow(0)
    run.interrupted = STOPPING
    assert runner._command(run, ws.request_dir(run.id)) == Outcome(
        130, "interrupted before it started", STOPPING)
    assert run.proc is None
    runner._signal(run, signal.SIGINT)  # no process: nothing to signal


def test_an_outcome_nobody_waits_for_is_cleaned_up(ws: Workspace) -> None:
    """A command that ends after the service's event loop is gone (or before there was one): its
    request folder is deleted, nobody reads its result."""
    runner = Runner(ws)
    run = slow(0)
    runner._execute(run)  # never admitted: no loop
    assert not ws.request_dir(run.id).exists()
    loop = asyncio.new_event_loop()
    loop.close()
    runner._loop = loop  # the service stopped
    run = slow(0)
    runner._execute(run)
    assert not ws.request_dir(run.id).exists() and run.proc is not None
    assert run.proc.returncode == 0


def test_a_command_deaf_to_sigint_and_sigterm_is_killed(ws: Workspace) -> None:
    runner = Runner(ws, interrupt_grace_s=0.2)
    proc = subprocess.Popen(
        [sys.executable, "-c", "import signal, time\n"
         "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
         "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
         "print('deaf', flush=True)\n"
         "time.sleep(60)"], stdout=subprocess.PIPE, start_new_session=True)
    try:
        assert proc.stdout is not None and proc.stdout.readline() == b"deaf\n"
        run = slow(60)
        run.proc = proc
        t0 = time.monotonic()
        runner._escalate(run)  # SIGTERM after the grace period, SIGKILL after as long again
        assert proc.wait(10) == -signal.SIGKILL and time.monotonic() - t0 >= 0.4
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_shutting_down_interrupts_and_waits_for_what_still_runs(ws: Workspace) -> None:
    runner = Runner(ws, interrupt_grace_s=5)
    run = slow(60)
    runner.waiting.append(run)
    runner._start(run)  # (a command started outside the event loop: nobody waits for it)
    until(lambda: run.proc is not None)
    uid, target = ws.new_upload("left.jpg")
    target.write_bytes(b"x")
    t0 = time.monotonic()
    runner.shutdown()
    assert time.monotonic() - t0 < 5  # it stopped at its SIGINT
    assert run.interrupted == STOPPING and run.proc is not None and run.proc.returncode == 130
    assert runner.stopping and list(ws.uploads.iterdir()) == []
    assert not ws.requests.exists() or list(ws.requests.iterdir()) == []


def test_killing_every_command_marks_the_running_ones_interrupted(ws: Workspace) -> None:
    runner = Runner(ws)
    run = slow(60, "--ignore-sigint")
    runner.waiting.append(run)
    runner._start(run)
    until(lambda: run.proc is not None)
    runner.kill_all()
    assert run.interrupted == STOPPING and runner.stopping
    assert run.proc is not None and run.proc.wait(10) == -signal.SIGKILL
    runner.shutdown()


ESCAPING = """#!{python}
import subprocess, sys, time
# a descendant in a session of its own, holding the command's stderr open for a while
subprocess.Popen([sys.executable, "-c", "import time; time.sleep({linger})"],
                 start_new_session=True)
print("started", flush=True)
time.sleep(60)
"""


def test_shutting_down_gives_up_on_a_command_whose_descendant_keeps_its_stderr(
        ws: Workspace, tmp_path: Path) -> None:
    """A command killed with its process group whose descendant escaped the group still holds
    the stderr the runner reads: shutting down waits for it until its deadline (twice the grace
    period and 5 s), then a bounded while more, and never hangs."""
    command = tmp_path / "escaping-command"
    command.write_text(ESCAPING.format(python=sys.executable, linger=5.8))
    command.chmod(0o755)
    runner = Runner(ws, python=str(command), interrupt_grace_s=0.05)
    run = slow(60)
    runner.waiting.append(run)
    runner._start(run)
    result = ws.request_dir(run.id) / "stdout"
    until(lambda: result.is_file() and result.read_bytes() == b"started\n")  # (its descendant too)
    t0 = time.monotonic()
    runner.shutdown()
    took = time.monotonic() - t0
    assert 5.0 <= took < 15.0, took  # past the deadline, then until the descendant let go
    assert run.proc is not None and run.proc.returncode == -signal.SIGTERM
    assert runner._threads == {}
