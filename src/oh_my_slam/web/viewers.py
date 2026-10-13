"""The maps' viewers (spec §2.6 "Operations": start or stop a map's viewer). A viewer is ``view.sh``
on the map (its map mode, the one that opens a persisted map read-only and never needs the
inference server), run as ``view.sh`` runs: its Python entry point (``python -m
oh_my_slam.cli.view``, what ``scripts/_common.sh`` execs) as a subprocess in its own process group,
with the command line built from its shared definition (``commands.spec.VIEW``) and no browser.
Its URL is the one of its ``listening on`` line: ``view.sh`` binds 127.0.0.1, so the page is served
on this machine only, and the service never proxies it.

At most one viewer runs per map: starting one that runs answers its URL again. A viewer that
ended by itself counts as stopped. Stopping the service stops every viewer it started
(:meth:`Viewers.stop_all`; a second stop signal kills them, :meth:`Viewers.kill_all`)."""

from __future__ import annotations

import re
import signal
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from oh_my_slam.commands import spec
from oh_my_slam.commands.spec import Kind
from oh_my_slam.core.errors import ExitCode
from oh_my_slam.web.runner import LOG_TAIL, STOPPING, RunError, command_message, killpg
from oh_my_slam.web.workspace import Workspace

PROGRAM = spec.VIEW
COMMAND = PROGRAM.command()
MAP = next(o for o in COMMAND.options if o.kind is Kind.MAP)  # -m
MODE = next(m for m in COMMAND.modes if m.selector == MAP.name)  # the map's: read-only
NO_BROWSER = COMMAND.option("no_browser")
MODULE = f"oh_my_slam.cli.{PROGRAM.prog.removesuffix('.sh')}"
LISTENING = re.compile(rf"^{re.escape(PROGRAM.prog)}: listening on (http://127\.0\.0\.1:\d+/)$")
START_TIMEOUT_S = 120.0  # opening a large map's cloud takes a while before it listens
STOP_TIMEOUT_S = 10.0  # SIGTERM is view.sh's normal stop; SIGKILL after this


class ViewerError(Exception):
    """A viewer that ended (or never listened) before it served: ``code`` is its exit status, the
    message its own error line (``view.sh: error: …``), else its last stderr line."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(eq=False)
class Viewer:
    proc: subprocess.Popen[bytes]
    url: str | None = None
    tail: deque[str] = field(default_factory=lambda: deque(maxlen=LOG_TAIL))
    listening: threading.Event = field(default_factory=threading.Event)  # its URL, or its end

    def running(self) -> bool:
        return self.proc.poll() is None


def command(map_dir: Path) -> list[str]:
    """``view.sh``'s command line on the map folder ``map_dir`` (no browser)."""
    return spec.argv_of(COMMAND, MODE, {MAP.name: str(map_dir), NO_BROWSER.name: True})


class Viewers:
    def __init__(self, workspace: Workspace, python: str = sys.executable,
                 start_timeout_s: float = START_TIMEOUT_S) -> None:
        self.ws = workspace
        self.python = python
        self.start_timeout_s = start_timeout_s
        self.stopping = False
        self._viewers: dict[str, Viewer] = {}  # map name → its viewer
        self._locks: dict[str, threading.Lock] = {}  # map name → the lock of its start and stop
        self._guard = threading.Lock()

    def _lock(self, name: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(name, threading.Lock())

    def state(self, name: str) -> dict[str, Any]:
        """Whether the map's viewer runs, and its URL while it does."""
        v = self._viewers.get(name)
        if v is None or not v.running():
            return {"running": False, "url": None}
        return {"running": True, "url": v.url}

    def start(self, name: str, map_dir: Path) -> str:
        """Start the viewer of the map ``name`` (folder ``map_dir``), or find the one that runs:
        its URL once it listens. :class:`ViewerError` when it ends first or does not listen within
        ``start_timeout_s``; refused while the service stops."""
        with self._lock(name):
            v = self._viewers.get(name)
            if v is not None and v.running():
                assert v.url is not None
                return v.url
            if self.stopping:
                raise RunError("the service is stopping; start the viewer again once it runs",
                               STOPPING)
            proc = subprocess.Popen(
                [self.python, "-m", MODULE, *command(map_dir)], stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, cwd=self.ws.root,
                start_new_session=True)
            v = Viewer(proc)
            threading.Thread(target=self._read, args=(v,), name=f"viewer-{name}",
                             daemon=True).start()
            if not v.listening.wait(self.start_timeout_s):
                self._end(proc, signal.SIGKILL)
                raise ViewerError(int(ExitCode.INTERNAL),
                                  f"the viewer of map {name} did not start within "
                                  f"{self.start_timeout_s:g} s")
            if v.url is None:  # it ended before it listened
                code = proc.wait()
                raise ViewerError(code, command_message(PROGRAM.prog, list(v.tail), code,
                                                        "viewer"))
            self._viewers[name] = v
            return v.url

    @staticmethod
    def _read(v: Viewer) -> None:
        """The viewer's stderr until it ends: its URL, and before it the lines that say why it did
        not start; nothing is kept once it listens."""
        assert v.proc.stderr is not None
        for raw in v.proc.stderr:
            if v.url is None:
                line = raw.decode("utf-8", "replace").rstrip("\n")
                m = LISTENING.match(line)
                if m:
                    v.url = m.group(1)
                    v.listening.set()
                else:
                    v.tail.append(line)
        v.listening.set()

    def stop(self, name: str) -> None:
        """Stop the map's viewer, if one runs (one being started is stopped once it started)."""
        with self._lock(name):
            v = self._viewers.pop(name, None)
            if v is not None:
                self._end(v.proc, signal.SIGTERM)

    @staticmethod
    def _end(proc: subprocess.Popen[bytes], sig: int) -> None:
        """End a viewer's process group with ``sig`` (SIGKILL if it outlives ``STOP_TIMEOUT_S``)
        and reap it."""
        if proc.poll() is None:
            killpg(proc.pid, sig)
            try:
                proc.wait(STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                killpg(proc.pid, signal.SIGKILL)
                proc.wait()

    def stop_all(self) -> None:
        """The service stops: no viewer starts any more, and every one it started is stopped
        (each after a start in progress has ended)."""
        self.stopping = True
        with self._guard:
            names = list(self._locks)
        for name in names:
            self.stop(name)

    def kill_all(self) -> None:
        """A second stop signal: SIGKILL every viewer's process group, waiting for none."""
        self.stopping = True
        for v in list(self._viewers.values()):
            if v.running():
                killpg(v.proc.pid, signal.SIGKILL)
