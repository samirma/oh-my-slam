"""stdout carries exactly one JSON document or one PLY for every command (AC21), and the
inference commands fail fast with exit 3 when the server is down (AC2). Runs the real shell
entry points against a server process with stub models."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from oh_my_slam.core.ply import parse_ply

REPO = Path(__file__).resolve().parents[2]


def sh(script: str, *args: str, timeout: float = 120) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run([str(REPO / script), *args], capture_output=True, timeout=timeout,
                          env=os.environ.copy())


@pytest.fixture(scope="module")
def image(tmp_path_factory: pytest.TempPathFactory) -> Path:
    p = tmp_path_factory.mktemp("img") / "photo.jpg"
    rng = np.random.default_rng(3)
    Image.fromarray(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)).save(p, quality=95)
    return p


@pytest.fixture(scope="module")
def stub_server() -> Iterator[None]:
    res = sh("start_inference_server.sh", "--stub")
    assert res.returncode == 0, res.stderr.decode()
    yield
    sh("start_inference_server.sh", "--stop")


def assert_one_json(out: bytes) -> dict:
    text = out.decode("utf-8")
    doc = json.loads(text)  # fails on banners or a second document
    assert text.strip().startswith("{") and text.endswith("\n")
    return doc


def assert_one_ply(out: bytes) -> None:
    cloud = parse_ply(out)
    header_end = out.find(b"end_header\n") + len(b"end_header\n")
    per = 15 + (4 if cloud.label is not None else 0)
    assert len(out) == header_end + per * len(cloud)


def test_down_server_fails_fast(image: Path) -> None:
    for script, args in [
        ("reconstruct.sh", ["-i", str(image)]),
        ("reconstruct.sh", ["-i", str(image), "-f", "ply"]),
        ("segment.sh", ["-i", str(image)]),
    ]:
        t0 = time.monotonic()
        res = sh(script, *args)
        assert res.returncode == 3, (script, res.stderr)
        assert time.monotonic() - t0 < 2.0
        assert res.stdout == b""
        assert b"./start_inference_server.sh" in res.stderr


def test_reconstruct_and_segment_stdout(stub_server: None, image: Path, tmp_path: Path) -> None:
    res = sh("reconstruct.sh", "-i", str(image))
    assert res.returncode == 0, res.stderr.decode()
    assert_one_json(res.stdout)
    res = sh("reconstruct.sh", "-i", str(image), "-f", "ply")
    assert res.returncode == 0
    assert_one_ply(res.stdout)
    res = sh("segment.sh", "-i", str(image))
    assert res.returncode == 0
    doc = assert_one_json(res.stdout)
    assert doc["openlabel"]["objects"]
    res = sh("segment.sh", "-i", str(image), "-f", "ply", "-o", str(tmp_path / "o"))
    assert res.returncode == 0
    assert_one_ply(res.stdout)
    assert len(list((tmp_path / "o").iterdir())) == 5


def test_timings_go_to_stderr_and_the_env_file_only(stub_server: None, image: Path,
                                                   tmp_path: Path) -> None:
    """Instrumentation (R44): a summary line on stderr, the full record in $OH_MY_SLAM_TIMINGS;
    stdout is still exactly one payload and the same document as without the variable."""
    plain = sh("reconstruct.sh", "-i", str(image))
    for script, args, stages in [
        ("reconstruct.sh", ["-i", str(image)],
         {"connect", "inference", "segment", "export", "write"}),
        ("reconstruct.sh", ["-i", str(image), "-f", "ply"], {"connect", "inference", "export"}),
        ("segment.sh", ["-i", str(image), "-o", str(tmp_path / "o")],
         {"inference", "segment", "export", "artifacts", "write"}),
    ]:
        target = tmp_path / f"{script}-{len(args)}.json"
        res = subprocess.run([str(REPO / script), *args], capture_output=True, timeout=120,
                             env={**os.environ, "OH_MY_SLAM_TIMINGS": str(target)})
        assert res.returncode == 0, res.stderr.decode()
        if "ply" in args:
            assert_one_ply(res.stdout)
        else:
            doc = assert_one_json(res.stdout)
            if script == "reconstruct.sh":
                assert doc["openlabel"]["objects"] == json.loads(plain.stdout)["openlabel"][
                    "objects"]
        assert b"timings: total" in res.stderr and b"timings" not in res.stdout
        rec = json.loads(target.read_text())
        assert stages <= set(rec["stages_s"]), rec["stages_s"]
        assert rec["server"]["geometry"]["count"] == 1
        assert {"geometry"} <= set(rec["parts"])
        if "ply" not in args:
            assert rec["server"]["segment"]["count"] == 1
            assert rec["server"]["gravity"]["count"] == 1
            assert {"gravity", "segmentation", "lift"} <= set(rec["parts"])
        assert rec["peak_rss_mb"]["self"] > 0


def test_usage_errors_exit_2(image: Path, tmp_path: Path) -> None:
    for script, args in [
        ("reconstruct.sh", []),
        ("reconstruct.sh", ["-i", str(image), "-f", "xyz"]),
        ("segment.sh", ["-i", str(image), "-m", str(tmp_path)]),
        ("segment.sh", ["-m", str(tmp_path), "--labels", "chair"]),
        ("segment.sh", []),
    ]:
        res = sh(script, *args)
        assert res.returncode == 2, (script, args, res.stderr)
        assert res.stdout == b""
