"""``server.sh`` as a process (api.md §1): one stderr line once listening, ``--status``, one
service per data folder, ``--stop``, and the inference server started with it (a no-op command
here: the tests never start the real one, nor bind a fixed port)."""

from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from oh_my_slam.service import main as sm
from oh_my_slam.service.store import Store

REPO = Path(__file__).resolve().parents[2]
LINE = re.compile(r"^server\.sh: listening on http://127\.0\.0\.1:(\d+)/$")


def env(tmp_path: Path) -> dict[str, str]:
    flag = tmp_path / "inference.calls"
    return {**os.environ, "OH_MY_SLAM_INFERENCE_CMD":
            f"{sys.executable} -c \"import sys; open('{flag}', 'a').write(' '.join(sys.argv[1:]) + ';')\""}


def sh(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(REPO / "server.sh"), "--data", str(tmp_path / "data"), *args],
                          capture_output=True, text=True, timeout=120, env=env(tmp_path))


@pytest.fixture
def served(tmp_path: Path) -> Iterator[tuple[subprocess.Popen[str], int]]:
    proc = subprocess.Popen([str(REPO / "server.sh"), "--data", str(tmp_path / "data"),
                             "--host", "127.0.0.1", "--port", "0"], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=env(tmp_path))
    assert proc.stderr is not None
    line = proc.stderr.readline().rstrip("\n")
    m = LINE.match(line)
    assert m, line
    yield proc, int(m.group(1))
    sh(tmp_path, "--stop")
    proc.wait(30)


def test_lifecycle(tmp_path: Path, served: tuple[subprocess.Popen[str], int]) -> None:
    proc, port = served
    r = httpx.get(f"http://127.0.0.1:{port}/api/health", timeout=10)
    assert r.status_code == 200 and "server-timing" in r.headers
    assert r.json()["inference"]["state"] in ("starting", "down")
    st = sh(tmp_path, "--status")
    assert st.returncode == 0 and json.loads(st.stdout)["service"] == "ready"
    second = sh(tmp_path, "--port", "0")
    assert second.returncode == 1 and f"port {port}" in second.stderr
    # the inference server was not running: the service started it, and --stop stops it
    deadline = time.monotonic() + 10
    while not (tmp_path / "inference.calls").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    stop = sh(tmp_path, "--stop")
    assert stop.returncode == 0 and "stopped (pid" in stop.stderr
    assert "the inference server was not running" in stop.stderr  # the no-op ended at once
    assert proc.wait(30) == 0 and proc.stdout is not None and proc.stdout.read() == ""
    assert (tmp_path / "inference.calls").read_text() == ";"
    assert sh(tmp_path, "--status").returncode == 3
    assert "not running" in sh(tmp_path, "--stop").stderr
    assert (tmp_path / "data" / "oms" / "service.log").exists()


def test_bad_ports(tmp_path: Path) -> None:
    assert sh(tmp_path, "--port", "70000").returncode == 2
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        r = sh(tmp_path, "--host", "127.0.0.1", "--port", str(busy.getsockname()[1]))
    assert r.returncode == 1 and "cannot bind" in r.stderr


def test_status_of_a_service_that_does_not_answer(tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch,
                                                  capsys: pytest.CaptureFixture[str]) -> None:
    from oh_my_slam.server.lifecycle import ServerLock

    store = Store(tmp_path / "data")
    store.create()
    lock = ServerLock(store.root / sm.LOCK)
    lock.acquire()
    try:
        (store.root / sm.STATE).write_text("{")
        assert sm.status(store) == 3
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        (store.root / sm.STATE).write_text(json.dumps({"port": port, "pid": os.getpid()}))
        assert sm.status(store) == 3 and "does not answer" in capsys.readouterr().err
        (store.root / sm.STATE).write_text("[]")
        assert sm._state(store) is None
    finally:
        lock.release()


def test_stop_kills_a_service_that_ignores_sigterm(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    store = Store(tmp_path / "data")
    store.create()
    child = subprocess.Popen([sys.executable, "-c", (
        "import signal, sys, time\n"
        "from oh_my_slam.server.lifecycle import ServerLock\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"ServerLock(Path({str(store.root / sm.LOCK)!r})).acquire()\n"
        "print('held', flush=True)\n"
        "time.sleep(60)\n")], stdout=subprocess.PIPE, text=True)
    assert child.stdout is not None and child.stdout.readline() == "held\n"
    (store.root / sm.STATE).write_text(json.dumps({"pid": child.pid, "port": 1}))
    monkeypatch.setattr(sm, "STOP_TIMEOUT_S", 0.3)
    assert sm.stop(store) == 0
    assert child.wait(10) == -signal.SIGKILL


def test_git_version_falls_back_to_the_package_version(monkeypatch: pytest.MonkeyPatch) -> None:
    assert sm.git_version()
    def broken(*a: Any, **k: Any) -> Any:
        raise OSError("no git")

    monkeypatch.setattr(subprocess, "run", broken)
    assert sm.git_version() == sm.__version__


def test_main_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["server.sh", "--port", "-1"])
    with pytest.raises(SystemExit) as e:
        sm.entry()
    assert e.value.code == 2
