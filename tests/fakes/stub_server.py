"""A real inference-server process running the deterministic stub models of ``stub_models``.

``python -m tests.fakes.stub_server`` runs the shipped server (lock, socket, worker thread, HTTP
app) with the stub adapters on the CPU. ``start_stub_server`` launches it in the background the way
``start_inference_server.sh`` launches the real one and waits until it is ready; the shipped
``start_inference_server.sh --status`` / ``--stop`` then manage it like any server.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from oh_my_slam.client.client import InferenceClient
from oh_my_slam.core import paths
from oh_my_slam.core.errors import ServerUnavailableError

REPO = Path(__file__).resolve().parents[2]
SERVER_CMD = [sys.executable, "-m", "tests.fakes.stub_server"]


def stub_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO), env.get("PYTHONPATH")]))
    env.update(extra or {})
    return env


def start_stub_server(env: dict[str, str] | None = None, timeout: float = 60.0) -> int:
    """Start the stub server detached (its log in the runtime dir); return its pid once ready."""
    with paths.server_log().open("ab") as logf:
        proc = subprocess.Popen(SERVER_CMD, cwd=REPO, stdin=subprocess.DEVNULL, stdout=logf,
                                stderr=logf, start_new_session=True, env=stub_env(env))
    client = InferenceClient()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"stub server exited with {proc.returncode}: "
                               f"{paths.server_log().read_text()[-2000:]}")
        try:
            if client.health(timeout=1.0).status == "ready":
                # reap it whenever it stops, so it never lingers as a zombie of the test process
                threading.Thread(target=proc.wait, daemon=True).start()
                return proc.pid
        except ServerUnavailableError:
            pass
        time.sleep(0.1)
    proc.terminate()
    raise RuntimeError("stub server not ready in time")


def main() -> int:
    from oh_my_slam.server.main import serve
    from tests.fakes.stub_models import stub_registry

    return serve(stub_registry, device="cpu")


if __name__ == "__main__":
    sys.exit(main())
