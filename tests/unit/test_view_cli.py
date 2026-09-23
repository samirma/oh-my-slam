"""view.sh: -i needs the server (exit 3 fast when it is down); -m does not."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]


def test_view_image_needs_server(tmp_path: Path) -> None:
    img = tmp_path / "x.jpg"
    Image.fromarray(np.zeros((20, 30, 3), np.uint8)).save(img)
    t0 = time.monotonic()
    res = subprocess.run([str(REPO / "view.sh"), "-i", str(img), "--no-browser"],
                         capture_output=True, timeout=60, env=os.environ.copy())
    assert res.returncode == 3 and time.monotonic() - t0 < 2.0
    assert b"./start_inference_server.sh" in res.stderr and res.stdout == b""


def test_view_map_serves_without_server(tmp_path: Path) -> None:
    """A minimal map folder is served read-only; the URL goes to stderr, stdout stays empty."""
    from oh_my_slam.mapping import store

    root = tmp_path / "m"
    with store.MapTransaction(root) as tx:
        tx.write_json(store.FRAMES_JSON, {"frames": []})
        tx.commit({"update_count": 1})
    before = store.full_tree_hash(root)
    proc = subprocess.Popen([str(REPO / "view.sh"), "-m", str(root), "--no-browser"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=os.environ.copy())
    try:
        assert proc.stderr is not None
        line = proc.stderr.readline().decode()
        assert "http://127.0.0.1:" in line
        import urllib.request

        url = line[line.index("http://"):].split()[0]
        with urllib.request.urlopen(url + "api/meta", timeout=10) as r:
            assert b'"mode": "map"' in r.read()
    finally:
        proc.send_signal(signal.SIGINT)
        out, _ = proc.communicate(timeout=20)
    assert out == b""
    assert store.full_tree_hash(root) == before
