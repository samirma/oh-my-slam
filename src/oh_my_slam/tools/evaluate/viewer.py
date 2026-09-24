"""``view.sh``: time from process start until the page has rendered (Playwright, Edge channel as
in ``tests/browser``), and the scene the viewer shows, for the contract checks.

The viewer's contract: stdout stays empty; once its socket listens it prints exactly one stderr
line ``view.sh: listening on http://127.0.0.1:<port>/`` (``oh_my_slam.cli.view.URL_LINE``; a
loopback ``http://`` URL in another line is accepted as a fallback); the page sets ``<body
data-rendered="true">`` after the first frame that drew the point cloud, or ``data-error`` on a
load failure; ``/api/scene`` serves the OpenLABEL scene it draws (OBB colours included). The
browser is launched before view.sh starts, so its start-up is not counted."""

from __future__ import annotations

import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

from oh_my_slam.tools.evaluate.runner import Live, Runner, RunRecord, RunSpec

CHANNELS = ("msedge", "chrome")
URL_LINE = re.compile(r"^view\.sh: listening on (http://127\.0\.0\.1:\d+/)$", re.MULTILINE)
_LOCAL_URL = re.compile(r"http://(?:127\.0\.0\.1|localhost|\[::1\]):\d+[^\s'\"<>]*")
SETTLED_JS = ("() => document.body !== null && (document.body.dataset.rendered === 'true' "
              "|| document.body.dataset.error !== undefined)")
PAGE_ERROR_JS = "() => document.body.dataset.error ?? null"
RENDER_TIMEOUT_S = 120.0  # from the URL to the rendered page
SCENE_API = "/api/scene"


def served_url(stderr: str) -> str | None:
    """The URL of view.sh's "listening on" line, else the first local http URL on stderr."""
    m = URL_LINE.search(stderr)
    if m is not None:
        return m.group(1)
    m = _LOCAL_URL.search(stderr)
    return None if m is None else m.group(0).rstrip(").,;:!]>")


@contextmanager
def edge_browser() -> Iterator[Any]:
    """A headless Chromium browser (Microsoft Edge, else Chrome) through Playwright."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        errors = []
        for channel in CHANNELS:
            try:
                browser = p.chromium.launch(channel=channel, headless=True)
                break
            except Exception as exc:
                errors.append(f"{channel}: {exc}".splitlines()[0])
        else:
            raise RuntimeError("no Edge/Chrome for Playwright: " + "; ".join(errors))
        try:
            yield browser
        finally:
            browser.close()


@dataclass
class ViewOutcome:
    record: RunRecord
    url: str | None
    render_s: float | None
    error: str | None  # why render_s is missing
    console_errors: list[str]
    scene: bytes | None  # what /api/scene served


def _fetch(url: str, path: str) -> bytes | None:
    try:
        with urllib.request.urlopen(urljoin(url, path), timeout=60) as resp:
            data: bytes = resp.read()
            return data
    except (urllib.error.URLError, OSError):
        return None


class BrowserProbe:
    """Measures one ``view.sh`` run; ``launch`` opens a browser (None: no browser available)."""

    def __init__(self, launch: Callable[[], AbstractContextManager[Any]] | None = edge_browser,
                 poll_s: float = 0.05) -> None:
        self.launch, self.poll_s = launch, poll_s

    def measure(self, runner: Runner, spec: RunSpec) -> ViewOutcome:
        with ExitStack() as stack:
            browser, error = None, None
            try:
                if self.launch is None:
                    raise RuntimeError("no browser available")
                browser = stack.enter_context(self.launch())
            except Exception as exc:
                error = f"browser: {exc}"
            live = runner.start(spec)
            try:
                url = self._wait_url(live, live.t0 + spec.timeout_s)
                render_s, console = None, list[str]()
                if url is not None and browser is not None:
                    render_s, error = self._render(browser, url, live.t0, console)
                scene = None if url is None else _fetch(url, SCENE_API)
            finally:
                live.stop()
            rec = live.finish(error=None if url else "no URL on stderr")
            rec.notes.update(url=url, render_s=render_s)
            return ViewOutcome(rec, url, render_s, error if url else rec.failure(), console, scene)

    def _wait_url(self, live: Live, deadline: float) -> str | None:
        while live.poll() is None and time.perf_counter() < deadline:
            url = served_url(live.stderr_text())
            if url is not None:
                return url
            time.sleep(self.poll_s)
        return served_url(live.stderr_text())

    @staticmethod
    def _render(browser: Any, url: str, t0: float, console: list[str]
                ) -> tuple[float | None, str | None]:
        """(seconds from ``t0`` until the page signalled it has rendered, error)."""
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.on("console", lambda m: console.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: console.append(str(e)))
        ms = RENDER_TIMEOUT_S * 1000
        try:
            page.goto(url, timeout=ms)
            page.wait_for_function(SETTLED_JS, timeout=ms)
            render_s = time.perf_counter() - t0
            failure = page.evaluate(PAGE_ERROR_JS)
            if failure is not None:
                return None, f"the page failed to load: {failure}"
            return render_s, None
        except Exception as exc:
            return None, f"page did not signal rendered: {str(exc).splitlines()[0]}"
        finally:
            page.close()
