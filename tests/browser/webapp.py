"""The server.sh service for the web application's browser tests: the real Starlette app and job
runner on a free loopback port (uvicorn in a thread), over a scratch workspace."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from oh_my_slam.web.app import Service, create_app
from oh_my_slam.web.jobs import Runner
from oh_my_slam.web.workspace import Workspace

AXE = Path(__file__).parent / "vendor" / "axe-core" / "axe.min.js"


@contextmanager
def running_service(data: Path, **service_kw: Any) -> Iterator[tuple[Service, str]]:
    """Serve a workspace at ``data`` on 127.0.0.1 and a free port; yields (service, base URL)."""
    import uvicorn

    ws = Workspace(data)
    ws.create()
    runner = Runner(ws, stop_grace_s=30, cancel_grace_s=10)
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
    """Run the vendored axe-core on the page and its frames (the embedded viewer included), with
    the WCAG 2.0 / 2.1 A and AA rules; every violation, whatever its impact, one line each."""
    for frame in page.frames:
        if frame.url.startswith("http") and not frame.evaluate("() => !!window.axe"):
            frame.add_script_tag(path=str(AXE))
    result = page.evaluate("""async () => {
        const r = await axe.run(document, {
            runOnly: {type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']},
            iframes: true,
            preload: false,  // its CSSOM preload resolves a frame's @import against the page: a false 404
        });
        return r.violations.map(v => ({id: v.id, impact: v.impact, help: v.help,
            nodes: v.nodes.slice(0, 5).map(n => n.target.join(' ') + ': ' + (n.failureSummary || ''))}));
    }""")
    return [f"{v['id']} ({v['impact']}): {v['help']} — {v['nodes']}" for v in result]
