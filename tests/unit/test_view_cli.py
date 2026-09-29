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


def _ignore_sigint() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # what a shell `&` job starts with


def start_view(*args: str, background_job: bool = False
               ) -> tuple[subprocess.Popen[bytes], str, list[str]]:
    """Start view.sh; returns the process, its URL and the stderr lines up to the URL line.
    ``background_job`` starts it with SIGINT ignored, as a shell ``&`` job would."""
    proc = subprocess.Popen([str(REPO / "view.sh"), *args, "--no-browser"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy(),
                            preexec_fn=_ignore_sigint if background_job else None)
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


def stop_view(proc: subprocess.Popen[bytes], sig: signal.Signals = signal.SIGINT) -> bytes:
    """Stop view.sh with ``sig``; it must exit 0 within 5 s. A server that does not is killed, so
    a failing test never leaves one running."""
    try:
        proc.send_signal(sig)
        out, _ = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise AssertionError(f"view.sh did not stop on {sig.name} within 5 s") from None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
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


def minimal_map(root: Path) -> Path:
    from oh_my_slam.mapping import store

    with store.MapTransaction(root) as tx:
        tx.write_json(store.FRAMES_JSON, {"frames": []})
        tx.commit({"update_count": 1})
    return root


def test_view_map_serves_without_server(tmp_path: Path) -> None:
    """A minimal map folder is served read-only; the URL goes to stderr, stdout stays empty."""
    from oh_my_slam.mapping import store

    root = minimal_map(tmp_path / "m")
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


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_view_stops_on_signal_even_as_a_background_job(tmp_path: Path, sig: signal.Signals
                                                       ) -> None:
    """Ctrl-C and SIGTERM are the normal stop (exit 0, stdout empty) — also for a process that
    starts with SIGINT ignored, as every shell `&` job does."""
    proc, _url, _ = start_view("-m", str(minimal_map(tmp_path / "m")), background_job=True)
    t0 = time.monotonic()
    assert stop_view(proc, sig) == b""
    assert time.monotonic() - t0 < 5.0


def test_view_usage_errors(tmp_path: Path, image: Path) -> None:
    """Neither -i nor -m, both, and a missing image are usage errors (2); an existing folder that
    is not a map is exit 4. None of them starts a server, writes to stdout or needs one."""
    plain = tmp_path / "not_a_map"
    plain.mkdir()
    (plain / "a.txt").write_text("x")
    map_dir = minimal_map(tmp_path / "m")
    cases = {
        "neither": ([], 2),
        "both": (["-i", str(image), "-m", str(map_dir)], 2),
        "missing image": (["-i", str(tmp_path / "nope.jpg")], 2),
        "not a map": (["-m", str(plain)], 4),
        "missing map folder": (["-m", str(tmp_path / "nope")], 4),
    }
    for name, (args, code) in cases.items():
        res = sh("view.sh", *args, "--no-browser")
        assert res.returncode == code, (name, res.stderr)
        assert res.stdout == b"" and b"view.sh" in res.stderr, name
    assert [p.name for p in plain.iterdir()] == ["a.txt"]
