"""Sequential command runs. Each run keeps its stdout and stderr in files, its per-stage timings
(``OH_MY_SLAM_TIMINGS``), its wall time and the peaks of the command's resident set and of the
server's footprint. A command that fails, times out or cannot start becomes a record with the
stderr tail — never an exception."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from oh_my_slam.core.timing import ENV_PATH as TIMINGS_ENV
from oh_my_slam.tools.evaluate.memory import PeakSampler, server_pid

REPO = Path(__file__).resolve().parents[4]
TAIL_LINES = 30
STOP_GRACE_S = 10.0
PROGRESS_WIDTH = 160


@dataclass(frozen=True)
class RunSpec:
    """One command: ``entry`` (a script in the repository root) with ``args``."""

    tag: str  # unique name of the run (file names, report rows)
    group: str  # performance group the run belongs to
    entry: str  # entry point, e.g. "reconstruct.sh"; contracts are reported per entry point
    args: tuple[str, ...] = ()
    stdout: str = "json"  # expected stdout of a successful run: "json" | "ply" | "empty"
    output: Path | None = None  # the ``-o`` file, if any
    output_kind: str | None = None  # its expected payload: "json" | "ply"
    ok_exit: tuple[int, ...] = (0,)
    timeout_s: float = 3600.0


@dataclass
class RunRecord:
    spec: RunSpec
    argv: list[str]
    exit_code: int | None
    wall_s: float
    client_peak_mb: float
    server_peak_gb: float | None
    stdout_path: Path
    stderr_path: Path
    stderr_tail: str
    timings: dict[str, Any] | None = None
    error: str | None = None  # did not start / timed out
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def tag(self) -> str:
        return self.spec.tag

    @property
    def ok(self) -> bool:
        return self.error is None and self.exit_code in self.spec.ok_exit

    def stdout_bytes(self) -> bytes:
        return self.stdout_path.read_bytes() if self.stdout_path.exists() else b""

    def failure(self) -> str:
        """One line: why the run failed, with the last stderr line."""
        why = self.error or f"exit {self.exit_code}"
        last = next((ln for ln in reversed(self.stderr_tail.splitlines()) if ln.strip()), "")
        return f"{self.tag} failed ({why})" + (f": {last.strip()}" if last else "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag, "group": self.spec.group, "entry": self.spec.entry,
            "argv": self.argv, "exit_code": self.exit_code, "ok": self.ok,
            "wall_s": round(self.wall_s, 3), "client_peak_mb": round(self.client_peak_mb, 1),
            "server_peak_gb": None if self.server_peak_gb is None else round(self.server_peak_gb, 2),
            "stdout": str(self.stdout_path), "stdout_bytes": len(self.stdout_bytes()),
            "stderr": str(self.stderr_path), "stderr_tail": self.stderr_tail if not self.ok else "",
            "timings": self.timings, "error": self.error, **self.notes,
        }


def _tail(path: Path, lines: int = TAIL_LINES) -> str:
    try:
        text = path.read_text(errors="replace")
    except FileNotFoundError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


class Live:
    """A started command (``Runner.start``); ``finish`` turns it into a :class:`RunRecord`."""

    def __init__(self, runner: Runner, spec: RunSpec, argv: list[str]) -> None:
        self.runner, self.spec, self.argv = runner, spec, argv
        runs = runner.out / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        self.stdout_path = runs / f"{spec.tag}.stdout"
        self.stderr_path = runs / f"{spec.tag}.stderr.txt"
        self.timings_path = runs / f"{spec.tag}.timings.json"
        self.timings_path.unlink(missing_ok=True)
        self.error: str | None = None
        self.proc: subprocess.Popen[bytes] | None = None
        self.sampler: PeakSampler | None = None
        env = {**runner.env, TIMINGS_ENV: str(self.timings_path)}
        self.t0 = time.perf_counter()
        with self.stdout_path.open("wb") as out, self.stderr_path.open("wb") as err:
            try:
                self.proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out,
                                             stderr=err, env=env, cwd=runner.out,
                                             start_new_session=True)
            except OSError as exc:
                self.error = f"could not start: {exc}"
                return
        self.sampler = PeakSampler(self.proc.pid, server_pid())
        self.sampler.start()

    def stderr_text(self) -> str:
        return self.stderr_path.read_text(errors="replace") if self.stderr_path.exists() else ""

    def poll(self) -> int | None:
        return None if self.proc is None else self.proc.poll()

    def wait(self, timeout: float) -> None:
        if self.proc is None:
            return
        try:
            self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            self.error = f"timed out after {timeout:.0f} s"
            self.stop()

    def stop(self) -> None:
        """Interrupt the command's process group (Ctrl-C), then terminate, then kill."""
        if self.proc is None or self.proc.poll() is not None:
            return
        for sig, grace in ((signal.SIGINT, STOP_GRACE_S), (signal.SIGTERM, 5.0),
                           (signal.SIGKILL, 5.0)):
            try:
                os.killpg(self.proc.pid, sig)
            except ProcessLookupError:
                return
            try:
                self.proc.wait(grace)
                return
            except subprocess.TimeoutExpired:
                continue

    def finish(self, error: str | None = None) -> RunRecord:
        wall = time.perf_counter() - self.t0
        if self.sampler is not None:
            self.sampler.sample()
            self.sampler.stop()
        timings = self._timings()
        client = self.sampler.client_peak_mb if self.sampler else 0.0
        if timings:
            client = max(client, float((timings.get("peak_rss_mb") or {}).get("self") or 0.0))
        rec = RunRecord(
            self.spec, self.argv, None if self.proc is None else self.proc.returncode, wall,
            client, self.sampler.server_peak_gb if self.sampler else None, self.stdout_path,
            self.stderr_path, _tail(self.stderr_path), timings, error or self.error)
        self.runner.finished(rec)
        return rec

    def _timings(self) -> dict[str, Any] | None:
        try:
            data = json.loads(self.timings_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None


class Runner:
    """Runs the entry points of ``repo`` one at a time (never two concurrently)."""

    def __init__(self, out: Path, repo: Path = REPO, env: dict[str, str] | None = None) -> None:
        self.out, self.repo = Path(out), Path(repo)
        self.env = dict(os.environ if env is None else env)
        self.env["PYTHONUNBUFFERED"] = "1"
        self.records: list[RunRecord] = []
        self._tags: set[str] = set()

    def start(self, spec: RunSpec) -> Live:
        if spec.tag in self._tags:
            raise ValueError(f"duplicate run tag {spec.tag!r}")
        self._tags.add(spec.tag)
        argv = [str(self.repo / spec.entry), *spec.args]
        line = f"{spec.entry} {' '.join(Path(a).name if '/' in a else a for a in spec.args)}"
        print(f"[{len(self._tags)}] {spec.tag}: {line[:PROGRESS_WIDTH]}"
              f"{'…' if len(line) > PROGRESS_WIDTH else ''}", file=sys.stderr, flush=True)
        return Live(self, spec, argv)

    def run(self, spec: RunSpec) -> RunRecord:
        live = self.start(spec)
        try:
            live.wait(spec.timeout_s)
        except BaseException:  # e.g. Ctrl-C: never leave the command running
            live.stop()
            raise
        return live.finish()

    def finished(self, rec: RunRecord) -> None:
        self.records.append(rec)
        server = "" if rec.server_peak_gb is None else f", server {rec.server_peak_gb:.1f} GB"
        status = "ok" if rec.ok else "FAILED"
        print(f"    {status}: exit {rec.exit_code} in {rec.wall_s:.1f} s "
              f"(client {rec.client_peak_mb:.0f} MB{server})", file=sys.stderr, flush=True)
        if not rec.ok:
            print("    " + rec.failure(), file=sys.stderr, flush=True)
