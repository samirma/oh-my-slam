"""The server.sh service for the web application's browser tests: the real Starlette app and
request runner on a free loopback port (uvicorn in a thread), over a scratch workspace; and a
browser tab on it that records the console errors and failed responses."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from oh_my_slam.web.app import Service, create_app
from oh_my_slam.web.runner import Runner
from oh_my_slam.web.workspace import Workspace

AXE = Path(__file__).parent / "vendor" / "axe-core" / "axe.min.js"
REQUEST_TIMEOUT_MS = 180_000


@contextmanager
def running_service(data: Path, **service_kw: Any) -> Iterator[tuple[Service, str]]:
    """Serve a workspace at ``data`` on 127.0.0.1 and a free port; yields (service, base URL)."""
    import uvicorn

    ws = Workspace(data)
    ws.create()
    runner = Runner(ws, interrupt_grace_s=10)
    service = Service(ws, runner, **service_kw)
    config = uvicorn.Config(create_app(service), host="127.0.0.1", port=0, log_config=None,
                            log_level="warning", access_log=False, lifespan="off")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("the service did not start")
        time.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    service.url = f"http://127.0.0.1:{port}/"
    try:
        yield service, service.url
    finally:
        runner.shutdown()
        server.should_exit = True
        thread.join(timeout=30)


def axe_violations(page: Any) -> list[str]:
    """Run the vendored axe-core on the page with the WCAG 2.0 / 2.1 A and AA rules; every
    violation, whatever its impact, one line each."""
    if not page.evaluate("() => !!window.axe"):
        page.add_script_tag(path=str(AXE))
    result = page.evaluate("""async () => {
        const r = await axe.run(document, {
            runOnly: {type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']},
        });
        return r.violations.map(v => ({id: v.id, impact: v.impact, help: v.help,
            nodes: v.nodes.slice(0, 5).map(n => n.target.join(' ') + ': ' + (n.failureSummary || ''))}));
    }""")
    return [f"{v['id']} ({v['impact']}): {v['help']} — {v['nodes']}" for v in result]


class Tab:
    """A browser page on the web application, with the console errors and failed responses it
    logged (``errors``)."""

    def __init__(self, browser: Any, base: str, width: int = 1280, height: int = 900,
                 scheme: str = "light") -> None:
        self.base, self.errors = base, []
        self.ctx = browser.new_context(viewport={"width": width, "height": height},
                                       color_scheme=scheme, accept_downloads=True)
        self.pg = self.ctx.new_page()
        self.pg.on("console", lambda m: self.errors.append(m.text) if m.type == "error" else None)
        self.pg.on("pageerror", lambda e: self.errors.append(str(e)))
        self.pg.on("response", lambda r: self.errors.append(f"{r.status} {r.request.method} {r.url}")
                   if r.status >= 400 else None)

    def go(self, route: str) -> Any:
        self.pg.goto(self.base + route)
        self.pg.wait_for_selector("body[data-ready=true]", timeout=30000)
        return self.pg

    def js(self, expr: str, arg: Any = None) -> Any:
        return self.pg.evaluate(expr, arg)

    def a11y(self) -> None:
        bad = axe_violations(self.pg)
        assert not bad, "\n".join(bad)

    def wait_request(self, state: str = "done", timeout: int = REQUEST_TIMEOUT_MS) -> None:
        """Wait for this page's request to reach ``state`` (done, failed, interrupted, running,
        waiting)."""
        self.pg.wait_for_selector(f"[data-testid=request][data-state={state}]", timeout=timeout)

    def close(self) -> None:
        self.ctx.close()


class Answer:
    """What the service answered a POST /api/ops/<op> of the page: ``status``, ``headers``,
    ``body`` (the bytes as sent)."""

    status: int
    headers: dict[str, str]
    body: bytes


@contextmanager
def operation_response(pg: Any, op: str) -> Iterator[Answer]:
    """Record the answer to the page's POST /api/ops/<op> requests made inside the block (the
    request goes through the test, which hands the page the service's answer unchanged; the
    browser's own record of a body the page read as a Blob is empty). The answer is filled in once
    it arrived, e.g. after ``Tab.wait_request``."""
    answer = Answer()

    def handle(route: Any) -> None:
        if route.request.method != "POST":
            route.continue_()
            return
        res = route.fetch(timeout=0)
        answer.status, answer.headers, answer.body = res.status, res.headers, res.body()
        route.fulfill(response=res)

    pattern = f"**/api/ops/{op}"
    pg.route(pattern, handle)
    try:
        yield answer
    finally:
        pass  # the route stays until the page closes: the answer may arrive after the block
