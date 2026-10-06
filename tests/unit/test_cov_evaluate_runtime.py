"""The evaluator's process side, offline (spec §4, §5): stopping a command that ignores Ctrl-C or
has gone, Ctrl-C during a run, the server's physical footprint and its samples, the Playwright
browser and the render probe (fake Playwright objects), and the inference proxy's records,
replays and pass-throughs when the upstream server is down or answers something else."""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import types
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from oh_my_slam.tools.evaluate import memory
from oh_my_slam.tools.evaluate import proxy as px
from oh_my_slam.tools.evaluate import runner as rn
from oh_my_slam.tools.evaluate.runner import Live, Runner, RunSpec
from oh_my_slam.tools.evaluate.viewer import (
    PAGE_ERROR_JS,
    SETTLED_JS,
    BrowserProbe,
    edge_browser,
)
from tests.unit.test_evaluate_contracts import scene_bytes
from tests.unit.test_evaluate_runner import script, view_runner

# -- stopping commands --------------------------------------------------------------------------------


def test_a_command_that_ignores_ctrl_c_is_terminated(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rn, "STOP_GRACE_S", 0.3)
    repo = tmp_path / "repo"
    repo.mkdir()
    script(repo, "view.sh", "trap '' INT\necho ignoring Ctrl-C >&2\nsleep 30")
    live = Runner(tmp_path / "out", repo).start(RunSpec("v", "g", "view.sh"))
    deadline = time.monotonic() + 30
    while "ignoring" not in live.stderr_text() and time.monotonic() < deadline:
        time.sleep(0.02)
    live.wait(0.1)
    rec = live.finish()
    assert rec.error == "timed out after 0 s" and rec.exit_code == -signal.SIGTERM
    assert not rec.ok


class Stuck:
    """A process that outlives every signal (uninterruptible I/O, say)."""

    pid = 4242

    def __init__(self) -> None:
        self.waits: list[float] = []

    def poll(self) -> None:
        return None

    def wait(self, timeout: float) -> int:
        self.waits.append(timeout)
        raise subprocess.TimeoutExpired("x", timeout)


def stuck_live() -> tuple[Live, Stuck]:
    live = Live.__new__(Live)
    proc = Stuck()
    live.proc = proc  # type: ignore[assignment]
    return live, proc


def test_stopping_gives_up_after_kill_and_never_hangs(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(rn.os, "killpg", lambda pid, sig: sent.append((pid, sig)))
    live, proc = stuck_live()
    live.stop()
    assert sent == [(4242, signal.SIGINT), (4242, signal.SIGTERM), (4242, signal.SIGKILL)]
    assert proc.waits == [rn.STOP_GRACE_S, 5.0, 5.0]


def test_stopping_a_command_whose_process_group_is_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    def gone(pid: int, sig: int) -> None:
        raise ProcessLookupError(pid)

    monkeypatch.setattr(rn.os, "killpg", gone)
    live, proc = stuck_live()
    live.stop()
    assert proc.waits == []  # nothing left to wait for


def test_ctrl_c_during_a_run_stops_the_command(tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    script(repo, "mapper.sh", "sleep 30")
    runner = Runner(tmp_path / "out", repo)
    started: list[Live] = []
    start = runner.start

    def tracked(spec: RunSpec) -> Live:
        started.append(start(spec))
        return started[-1]

    def interrupted(self: Live, timeout: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "start", tracked)
    monkeypatch.setattr(Live, "wait", interrupted)
    with pytest.raises(KeyboardInterrupt):
        runner.run(RunSpec("m", "g", "mapper.sh"))
    (live,) = started
    assert live.proc is not None and live.proc.poll() is not None  # not left running
    assert runner.records == []


def test_a_run_whose_stderr_file_is_gone_has_no_tail(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    script(repo, "segment.sh", 'echo "segment.sh: error: down" >&2\nexit 3')
    live = Runner(tmp_path / "out", repo).start(RunSpec("s", "g", "segment.sh"))
    live.wait(30)
    live.stderr_path.unlink()
    rec = live.finish()
    assert rec.stderr_tail == "" and rec.failure() == "s failed (exit 3)"


# -- memory --------------------------------------------------------------------------------------------


class FakeLibproc:
    def __init__(self, status: int) -> None:
        self.status = status
        self.calls: list[tuple[int, int]] = []

    def proc_pid_rusage(self, pid: int, flavor: int, ref: Any) -> int:
        self.calls.append((pid, flavor))
        info = ref._obj  # the structure passed by reference
        info.ri_phys_footprint = 2_500_000_000
        info.ri_lifetime_max_phys_footprint = 4_000_000_000
        return self.status


def test_the_physical_footprint_of_a_process(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeLibproc(0)
    monkeypatch.setattr(memory, "_libproc", fake)
    assert memory.phys_footprint_gb(77) == (2.5, 4.0)
    assert fake.calls == [(77, 4)]  # RUSAGE_INFO_V4
    monkeypatch.setattr(memory, "_libproc", FakeLibproc(-1))  # e.g. no such process
    assert memory.phys_footprint_gb(77) is None
    monkeypatch.setattr(memory, "_libproc", None)  # not macOS
    assert memory.phys_footprint_gb(77) is None


@pytest.mark.skipif(sys.platform != "darwin", reason="libproc is macOS only")
def test_the_physical_footprint_of_this_process_on_macos() -> None:
    fp = memory.phys_footprint_gb(os.getpid())
    assert fp is not None and 0 < fp[0] <= fp[1]  # (current, lifetime peak) in GB
    assert memory.phys_footprint_gb(2**22 + 12345) is None  # no such process


def test_the_sampler_keeps_the_servers_peak(monkeypatch: pytest.MonkeyPatch) -> None:
    footprints = iter([(1.5, 2.0), (0.5, 2.0)])
    monkeypatch.setattr(memory, "phys_footprint_gb", lambda pid: next(footprints))
    sampler = memory.PeakSampler(os.getpid(), server=os.getpid())
    sampler.sample()
    sampler.sample()
    assert sampler.server_peak_gb == 1.5
    assert [s[2] for s in sampler.samples] == [1.5, 0.5] and sampler.client_peak_mb > 0


# -- the browser -----------------------------------------------------------------------------------------


class FakeBrowser:
    def __init__(self, page_factory: Callable[[], FakePage] | None = None) -> None:
        self.closed = False
        self.pages: list[FakePage] = []
        self.page_factory = page_factory or FakePage

    def new_page(self, viewport: dict[str, int]) -> FakePage:
        self.pages.append(self.page_factory())
        return self.pages[-1]

    def close(self) -> None:
        self.closed = True


class Message:
    def __init__(self, type_: str, text: str) -> None:
        self.type, self.text = type_, text


class FakePage:
    """A page that loads, logs to its console, and settles as ``failure`` says."""

    def __init__(self, failure: str | None = None, goto_error: str | None = None) -> None:
        self.failure, self.goto_error = failure, goto_error
        self.handlers: dict[str, list[Callable[[Any], Any]]] = {}
        self.calls: list[tuple[Any, ...]] = []
        self.closed = False

    def on(self, event: str, handler: Callable[[Any], Any]) -> None:
        self.handlers.setdefault(event, []).append(handler)

    def goto(self, url: str, timeout: float) -> None:
        self.calls.append(("goto", url))
        if self.goto_error is not None:
            raise RuntimeError(self.goto_error)
        for h in self.handlers.get("console", []):
            h(Message("log", "loading"))
            h(Message("error", "WebGL: context lost"))
        for h in self.handlers.get("pageerror", []):
            h(ValueError("cloud: bad header"))

    def wait_for_function(self, js: str, **kw: Any) -> None:
        self.calls.append(("wait", js, kw.get("arg")))

    def evaluate(self, js: str) -> Any:
        self.calls.append(("evaluate", js))
        return self.failure

    def close(self) -> None:
        self.closed = True


def fake_playwright(monkeypatch: pytest.MonkeyPatch, working: set[str]) -> list[str]:
    tried: list[str] = []

    class Chromium:
        def launch(self, channel: str, headless: bool) -> FakeBrowser:
            tried.append(channel)
            assert headless
            if channel not in working:
                raise RuntimeError(f"Executable doesn't exist for {channel}\nInstall it")
            return FakeBrowser()

    @contextmanager
    def sync_playwright() -> Iterator[Any]:
        yield types.SimpleNamespace(chromium=Chromium())

    module = types.ModuleType("playwright.sync_api")
    module.sync_playwright = sync_playwright  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "playwright.sync_api", module)
    return tried


def test_the_browser_is_edge_else_chrome(monkeypatch: pytest.MonkeyPatch) -> None:
    tried = fake_playwright(monkeypatch, {"chrome"})
    with edge_browser() as browser:
        assert isinstance(browser, FakeBrowser) and not browser.closed
    assert browser.closed and tried == ["msedge", "chrome"]
    tried = fake_playwright(monkeypatch, set())
    with pytest.raises(RuntimeError) as err, edge_browser():
        pass
    assert str(err.value) == ("no Edge/Chrome for Playwright: msedge: Executable doesn't exist "
                              "for msedge; chrome: Executable doesn't exist for chrome")


def test_the_render_time_and_console_of_a_view_in_a_browser(tmp_path: Path) -> None:
    browser = FakeBrowser()

    @contextmanager
    def launch() -> Iterator[FakeBrowser]:
        yield browser

    seen = BrowserProbe(launch).measure(view_runner(tmp_path), RunSpec(
        "v", "view_image", "view.sh", ("--no-browser",), "empty", ok_exit=(0, 130), timeout_s=60))
    assert seen.error is None and seen.render_s is not None and seen.render_s > 0
    assert seen.console_errors == ["WebGL: context lost", "cloud: bad header"]
    assert seen.scene == scene_bytes() and seen.cloud is not None and seen.record.ok
    (page,) = browser.pages
    assert page.closed and page.calls[0] == ("goto", seen.url)
    assert page.calls[1:] == [("wait", SETTLED_JS, None), ("evaluate", PAGE_ERROR_JS)]
    assert seen.record.notes["render_s"] == seen.render_s


@pytest.mark.parametrize(("page", "error"), [
    (FakePage(failure="cloud: 500"), "the page failed to load: cloud: 500"),
    (FakePage(goto_error="Timeout 120000ms exceeded.\n=== logs ==="),
     "page did not signal rendered: Timeout 120000ms exceeded."),
])
def test_a_page_that_fails_or_never_settles_has_no_render_time(page: FakePage, error: str) -> None:
    console: list[str] = []
    render_s, why = BrowserProbe._render(FakeBrowser(lambda: page), "http://127.0.0.1:1/", 0.0,
                                         console)
    assert render_s is None and why == error and page.closed


def closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_a_viewer_that_serves_nothing_has_no_scene(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    script(repo, "view.sh", f"""import signal, sys, time
signal.signal(signal.SIGINT, signal.default_int_handler)
print("view.sh: listening on http://127.0.0.1:{closed_port()}/", file=sys.stderr, flush=True)
try:
    time.sleep(30)
except KeyboardInterrupt:
    pass""", shebang=f"#!{sys.executable}")
    seen = BrowserProbe(None).measure(Runner(tmp_path / "out", repo), RunSpec(
        "v", "view_image", "view.sh", (), "empty", ok_exit=(0, 130), timeout_s=60))
    assert seen.url is not None and seen.scene is None and seen.cloud is None
    assert seen.error == "browser: no browser available" and seen.record.ok


# -- the inference proxy -----------------------------------------------------------------------------------


@pytest.fixture
def short() -> Iterator[Path]:
    """A folder whose socket paths fit AF_UNIX (pytest's tmp_path is too deep on macOS)."""
    root = Path(tempfile.mkdtemp(prefix="oms-t-", dir="/tmp"))
    yield root
    shutil.rmtree(root, ignore_errors=True)


def test_recorded_files_nested_in_lists_are_restored_and_others_kept(tmp_path: Path) -> None:
    store = px.Store(tmp_path / "store")
    mask = tmp_path / "mask.png"
    mask.write_bytes(b"mask")
    store.keep("k", {"masks": [{"mask_path": str(mask)}, {"mask_path": "/gone/mask.png"}],
                     "count_path": 3, "n": 2})
    assert store.answer("other", None) is None
    got = store.answer("k", None)  # no out_dir: a temporary folder
    assert got is not None and got["n"] == 2 and got["count_path"] == 3
    restored = Path(got["masks"][0]["mask_path"])
    assert restored.read_bytes() == b"mask" and restored.parent != mask.parent
    assert got["masks"][1]["mask_path"] == "/gone/mask.png"  # never a file: passed through
    assert (store.recorded, store.replayed) == (1, 1)


def test_a_missing_input_file_is_keyed_as_missing(tmp_path: Path) -> None:
    key = px.request_key
    assert key("/r", {"image_path": str(tmp_path / "a.jpg")}) == \
        key("/r", {"image_path": str(tmp_path / "b.jpg")})
    (tmp_path / "a.jpg").write_bytes(b"A")
    assert key("/r", {"image_path": str(tmp_path / "a.jpg")}) != \
        key("/r", {"image_path": str(tmp_path / "b.jpg")})


def test_with_the_server_down_the_proxy_answers_unavailable(short: Path) -> None:
    proxy = px.InferenceProxy(short / "rt", short / "nothing.sock", timeout=5.0)
    try:
        status, ctype, body = proxy.handle("GET", "/health", None)
        assert (status, ctype) == (503, "application/json")
        assert json.loads(body) == {"error": "unavailable", "detail": "ConnectError"}
        status, _, _ = proxy.handle("POST", "/v1/geometry", b'{"side": 512}')
        assert status == 503
        assert proxy.stats() == {"recorded": 0, "replayed": 0, "forwarded": 0}
    finally:
        proxy.stop()  # never started: nothing to shut down
    assert not proxy.socket.exists()


def test_only_json_object_requests_answered_with_json_are_recorded(short: Path) -> None:
    proxy = px.InferenceProxy(short / "rt", short / "up.sock")
    calls: list[tuple[str, str, bytes | None]] = []
    answer = (200, "text/plain", b"plain")

    def forward(method: str, path: str, body: bytes | None) -> tuple[int, str, bytes]:
        calls.append((method, path, body))
        return answer

    proxy.forward = forward  # type: ignore[method-assign]
    try:
        assert proxy.handle("POST", "/v1/x", b"not json") == answer  # passed through
        assert proxy.handle("POST", "/v1/x", b"[1, 2]") == answer
        assert proxy.handle("POST", "/v1/x", b'{"k": 1}') == answer  # not JSON: not kept
        assert proxy.handle("POST", "/v1/x", b'{"k": 1}') == answer
        assert len(calls) == 4 and proxy.store.recorded == 0
        answer = (500, "application/json", b'{"error": "internal"}')
        assert proxy.handle("POST", "/v1/x", b'{"k": 2}') == answer  # a failure: not kept
        answer = (200, "application/json", b'{"n": 1}')
        proxy.handle("POST", "/v1/x", b'{"k": 2}')
        assert proxy.handle("POST", "/v1/x", b'{"k": 2}') == (200, "application/json",
                                                             b'{"n": 1}')
        assert len(calls) == 6 and proxy.stats()["replayed"] == 1
    finally:
        proxy.stop()


def test_the_proxy_stops_once(short: Path) -> None:
    proxy = px.InferenceProxy(short / "rt", short / "up.sock").start()
    assert proxy.socket.exists()
    proxy.stop()
    proxy.stop()
    assert proxy.server is None and not proxy.socket.exists()
