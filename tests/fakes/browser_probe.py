"""Run a command module with ``webbrowser.open`` replaced by a probe (view.sh / server.sh tests).

    python browser_probe.py <record.jsonl> <module> [args...]

The probe connects to the URL it is asked to open, as a browser would at that moment, and
appends ``{"url": ..., "connected": true|false, "error": ...}`` to ``record.jsonl``; no browser
is started. A test then checks that the browser is opened only once the service accepts
connections, and never with ``--no-browser`` (no record)."""

from __future__ import annotations

import json
import runpy
import socket
import sys
import webbrowser
from pathlib import Path
from urllib.parse import urlparse


def main() -> None:
    record, module, *args = sys.argv[1:]

    def probe(url: str, *_a: object, **_k: object) -> bool:
        u = urlparse(url)
        entry: dict[str, object] = {"url": url}
        try:
            with socket.create_connection((u.hostname, u.port), timeout=2.0):
                entry["connected"] = True
        except OSError as exc:
            entry.update(connected=False, error=str(exc))
        with Path(record).open("a") as f:
            f.write(json.dumps(entry) + "\n")
        return True

    webbrowser.open = probe  # type: ignore[assignment]
    sys.argv = [module, *args]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
