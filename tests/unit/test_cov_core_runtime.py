"""The process plumbing every entry point shares (spec §4): one payload on stdout and everything
else on stderr (``core.log``), exceptions mapped to exit codes (``core.process``), and the timing
collector's edges (``core.timing``)."""

from __future__ import annotations

import io
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import pytest

from oh_my_slam.core import log, process, timing
from oh_my_slam.core.errors import ExitCode, InputError, ServerUnavailableError

REPO = Path(__file__).resolve().parents[2]

# --- stdout carries the payload only -------------------------------------------------------------

_CLAIM = r"""
import os, sys
from pathlib import Path
from oh_my_slam.core.log import claim_stdout
out = claim_stdout()
assert claim_stdout() is out  # claiming twice changes nothing
print("a banner from Python")
os.write(1, b"progress from C code on fd 1\n")
claim_stdout(Path(sys.argv[1])).write_json({"to": "file"})  # -o: the file, never stdout
out.write_json({"payload": 1})
"""


def test_stdout_carries_one_payload_and_everything_else_goes_to_stderr(tmp_path: Path) -> None:
    res = subprocess.run([sys.executable, "-c", _CLAIM, str(tmp_path / "o" / "out.json")],
                         cwd=REPO, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    assert res.stdout == '{"payload":1}\n'
    assert "a banner from Python" in res.stderr and "progress from C code" in res.stderr
    assert json.loads((tmp_path / "o" / "out.json").read_text()) == {"to": "file"}


def test_the_payload_writer_takes_exactly_one_target_and_one_payload(tmp_path: Path) -> None:
    for kwargs in ({}, {"stream": io.BytesIO(), "path": tmp_path / "x"}):
        with pytest.raises(ValueError, match="exactly one of stream and path"):
            log.PayloadWriter(**kwargs)  # type: ignore[arg-type]
    stream = io.BytesIO()
    w = log.PayloadWriter(stream)
    w.write_json({"a": "é"})
    assert stream.getvalue() == '{"a":"é"}\n'.encode()
    with pytest.raises(RuntimeError, match="payload already written"):
        w.write_bytes(b"second")


def test_a_consumer_that_closed_the_pipe_is_not_an_error() -> None:
    class ClosedPipe(io.BytesIO):
        def write(self, data: Any) -> int:
            raise BrokenPipeError

    w = log.PayloadWriter(ClosedPipe())  # e.g. ``reconstruct.sh … | head -c 10``
    w.write_bytes(b"ply\n")
    with pytest.raises(RuntimeError, match="payload already written"):  # it counted as written
        w.write_bytes(b"ply\n")


def test_the_whole_payload_reaches_a_non_blocking_stdout() -> None:
    """The real stdout is a raw stream, whose write may take only part of the payload: a full
    non-blocking pipe takes 64 KiB, then nothing (None) until it is read. Every byte arrives (once,
    65 536 of 1 MiB did, and the command exited 0)."""
    r, w = os.pipe()
    os.set_blocking(w, False)
    payload = bytes(range(256)) * 4096  # 1 MiB
    got = bytearray()

    def read() -> None:
        time.sleep(0.2)  # the pipe fills first
        while chunk := os.read(r, 1 << 16):
            got.extend(chunk)

    reader = threading.Thread(target=read)
    reader.start()
    with os.fdopen(w, "wb", buffering=0) as stream:
        log.PayloadWriter(stream).write_bytes(payload)
    reader.join(60)
    os.close(r)
    assert bytes(got) == payload


def test_the_package_logger_writes_to_stderr_without_propagating() -> None:
    root = log.get_logger()
    assert root is logging.getLogger("oh_my_slam") and not root.propagate
    assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)
    assert log.get_logger("oh_my_slam.viewer") is root.getChild("viewer")
    assert log.get_logger("plain") is root.getChild("plain")


# --- exit codes ----------------------------------------------------------------------------------


class _ExitCalledError(Exception):
    def __init__(self, code: int) -> None:
        self.code = code


@pytest.fixture
def exits(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """``run_main`` in this process: ``os._exit`` raises ``_ExitCalledError``, coverage is not saved."""
    saved: list[str] = []

    def fake_exit(code: int) -> None:
        raise _ExitCalledError(code)

    monkeypatch.setattr(process.os, "_exit", fake_exit)
    monkeypatch.setattr(process, "_save_coverage", lambda: saved.append("saved"))
    return saved


def _run(main: Any, argv: list[str] | None = None) -> int:
    with pytest.raises(_ExitCalledError) as e:
        process.run_main("prog.sh", main, argv)
    return e.value.code


@pytest.mark.parametrize(("raised", "code", "message"), [
    (None, 0, ""),
    (InputError("image not found: x.jpg"), 2, "prog.sh: error: image not found: x.jpg\n"),
    (ServerUnavailableError(), 3, "prog.sh: error: inference server is not running"),
    (KeyboardInterrupt(), 130, "prog.sh: interrupted\n"),
    (BrokenPipeError(), 0, ""),  # the reader went away: nothing to report
    (ValueError("bad"), 1, "prog.sh: internal error: ValueError: bad\n"),
])
def test_exceptions_become_exit_codes_with_one_stderr_line(
        exits: list[str], capsys: pytest.CaptureFixture[str], raised: BaseException | None,
        code: int, message: str) -> None:
    def main(args: list[str]) -> int:
        assert args == ["-i", "x.jpg"]
        if raised is not None:
            raise raised
        return 0

    assert _run(main, ["-i", "x.jpg"]) == code
    out, err = capsys.readouterr()
    assert out == "" and err.startswith(message) and (message or err == "")
    assert exits == ["saved"]  # coverage data is saved before os._exit skips atexit


def test_the_arguments_default_to_the_command_line(exits: list[str],
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prog.sh", "--status"])
    assert _run(lambda args: 7 if args == ["--status"] else 0) == 7
    assert int(ExitCode.INTERRUPTED) == 130


def _sigint_blocked_now() -> bool:
    return signal.SIGINT in signal.pthread_sigmask(signal.SIG_BLOCK, [])


def test_a_ctrl_c_held_while_the_process_started_is_its_interrupt(exits: list[str]) -> None:
    """A process started with SIGINT blocked (``sigint_blocked``): ``run_main`` unblocks it inside
    its handler, so a SIGINT that arrived before is a ``KeyboardInterrupt`` there (exit 130), and
    the thread's mask is restored after a ``sigint_blocked`` block."""
    assert not _sigint_blocked_now()
    with process.sigint_blocked():
        assert _sigint_blocked_now()
        # to this thread, where it is blocked, so it stays pending (another thread of the test
        # process that does not block SIGINT would take one sent to the process)
        signal.pthread_kill(threading.get_ident(), signal.SIGINT)
        ran: list[list[str]] = []
        assert _run(lambda args: ran.append(args) or 0, ["x"]) == 130  # before main ran
        assert ran == [] and not _sigint_blocked_now()  # run_main unblocked it
    assert not _sigint_blocked_now()  # restored
    with process.sigint_blocked():  # nothing pending: main runs, SIGINT is unblocked
        assert _run(lambda args: 0) == 0 and not _sigint_blocked_now()


def test_ctrl_c_is_restored_only_on_the_main_thread() -> None:
    old = signal.signal(signal.SIGINT, signal.SIG_IGN)  # e.g. a shell ``&`` job
    try:
        t = threading.Thread(target=process.default_sigint)
        t.start()
        t.join()
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN  # threads cannot set handlers
        process.default_sigint()
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    finally:
        signal.signal(signal.SIGINT, old)


def _coverage_module(current: Any) -> types.ModuleType:
    mod = types.ModuleType("coverage")
    mod.Coverage = types.SimpleNamespace(current=current)  # type: ignore[attr-defined]
    return mod


def test_coverage_is_saved_when_measured_and_never_changes_the_exit(
        monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    cov = types.SimpleNamespace(stop=lambda: events.append("stop"),
                                save=lambda: events.append("save"))
    monkeypatch.setitem(sys.modules, "coverage", _coverage_module(lambda: cov))
    process._save_coverage()
    assert events == ["stop", "save"]
    monkeypatch.setitem(sys.modules, "coverage", _coverage_module(lambda: None))
    process._save_coverage()  # imported, not measuring: nothing to save
    assert events == ["stop", "save"]

    def broken() -> None:
        raise RuntimeError("no data file")

    monkeypatch.setitem(sys.modules, "coverage", _coverage_module(broken))
    process._save_coverage()  # swallowed
    monkeypatch.delitem(sys.modules, "coverage")
    process._save_coverage()  # not imported at all: nothing to do


# --- timing --------------------------------------------------------------------------------------


def test_sampling_starts_once_and_only_when_asked() -> None:
    t = timing.Timings(sample_every=None)
    t.start_sampling()
    assert t._sampler is None  # boundaries only
    t.stop_sampling()  # nothing to stop
    t = timing.Timings(sample_every=0.01)
    t.start_sampling()
    sampler = t._sampler
    t.start_sampling()
    assert t._sampler is sampler and sampler is not None and sampler.is_alive()
    t.stop_sampling()
    assert t._sampler is None and not sampler.is_alive()


def test_a_stage_reentered_stays_active_until_its_outer_run_ends() -> None:
    t = timing.Timings(sample_every=None)
    with t.stage(timing.Stage.OBJECTS):
        with t.stage(timing.Stage.OBJECTS):
            assert t._active == {"objects": 2}
        assert t._active == {"objects": 1}
    assert t._active == {} and t.stages["objects"] >= 0
    assert [w[0] for w in t.windows] == ["objects", "objects"]


def test_the_summary_line_of_a_collector() -> None:
    t = timing.Timings(sample_every=None)
    with t.stage(timing.Stage.SETUP):
        pass
    t.request("geometry", 0.5, 0.1, 0.3)
    line = t.summary()
    assert line.startswith("timings: total ") and "setup " in line
    assert "server compute: geometry 1× 0.30" in line and "peak RSS" in line


def test_an_unreadable_resident_set_reads_as_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    class Gone:
        def memory_info(self) -> Any:
            raise ProcessLookupError("no such process")

    monkeypatch.setattr(timing, "_process", Gone())
    assert timing.rss_mb() == 0.0  # instrumentation never fails a command
