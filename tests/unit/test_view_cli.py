"""view.sh end to end: -i needs the server (exit 3 fast when it is down) and, once served, keeps
answering control changes with the server stopped; -m needs no server and never modifies the map.
The URL is the one ``URL_LINE`` of stderr; stdout stays empty."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from oh_my_slam.cli.view import URL_LINE
from oh_my_slam.viewer.server import parse_cloud_payload

REPO = Path(__file__).resolve().parents[2]


def sh(script: str, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run([str(REPO / script), *args], capture_output=True, timeout=120,
                          env=os.environ.copy())


def start_view(*args: str) -> tuple[subprocess.Popen[bytes], str, list[str]]:
    """Start view.sh; returns the process, its URL and the stderr lines up to the URL line."""
    proc = subprocess.Popen([str(REPO / "view.sh"), *args, "--no-browser"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy())
    assert proc.stderr is not None
    lines: list[str] = []
    while True:
        line = proc.stderr.readline().decode()
        if not line:
            proc.kill()
            raise AssertionError(f"view.sh exited without a URL: {lines}")
        lines.append(line.rstrip("\n"))
        m = URL_LINE.match(lines[-1])
        if m:
            return proc, m.group(1), lines


def stop_view(proc: subprocess.Popen[bytes]) -> bytes:
    proc.send_signal(signal.SIGINT)
    out, _ = proc.communicate(timeout=20)
    assert proc.returncode == 0
    return out


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


@pytest.fixture(scope="module")
def image(tmp_path_factory: pytest.TempPathFactory) -> Path:
    p = tmp_path_factory.mktemp("img") / "photo.jpg"
    rng = np.random.default_rng(5)
    Image.fromarray(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)).save(p, quality=95)
    return p


def test_view_image_needs_server(image: Path) -> None:
    t0 = time.monotonic()
    res = sh("view.sh", "-i", str(image), "--no-browser")
    assert res.returncode == 3 and time.monotonic() - t0 < 2.0
    assert b"./start_inference_server.sh" in res.stderr and res.stdout == b""


def test_view_image_controls_work_without_the_server(image: Path) -> None:
    """Inference runs once at start-up: with the (stub) server stopped afterwards, every control
    change is still answered from the data in memory."""
    res = sh("start_inference_server.sh", "--stub")
    assert res.returncode == 0, res.stderr.decode()
    try:
        proc, url, _ = start_view("-i", str(image))
    finally:
        sh("start_inference_server.sh", "--stop")
    try:
        meta = json.loads(fetch(url + "api/meta"))
        assert meta["mode"] == "image" and len(meta["cameras"]) == 1
        for query in ("", "stride=2", "voxel=0.05&normals=on", "color=segment", "edge=0"):
            h, arrays = parse_cloud_payload(fetch(f"{url}api/cloud?{query}"))
            assert h["count"] > 0 and "position" in arrays
        assert b"importmap" in fetch(url)
    finally:
        out = stop_view(proc)
    assert out == b""


def test_view_map_serves_without_server(tmp_path: Path) -> None:
    """A minimal map folder is served read-only; the URL goes to stderr, stdout stays empty."""
    from oh_my_slam.mapping import store

    root = tmp_path / "m"
    with store.MapTransaction(root) as tx:
        tx.write_json(store.FRAMES_JSON, {"frames": []})
        tx.commit({"update_count": 1})
    before = store.full_tree_hash(root)
    proc, url, lines = start_view("-m", str(root))
    try:
        assert all(line.startswith("view.sh: ") for line in lines)
        assert json.loads(fetch(url + "api/meta"))["mode"] == "map"
        head, arrays = parse_cloud_payload(fetch(url + "api/cloud?voxel=0.1&normals=on"))
        assert head["count"] == 0 and arrays["position"].shape == (0, 3)
    finally:
        out = stop_view(proc)
    assert out == b""
    assert store.full_tree_hash(root) == before
