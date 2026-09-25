"""stdout carries exactly one JSON document or one PLY for every command (AC21) — or nothing
with ``-o <file>`` — and the inference commands fail fast with exit 3 when the server is down (AC2)
while option errors (e.g. a bad ``-p``) exit 2 before the server is contacted. Runs the real shell
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
    if out.startswith(b"ply\nformat ascii 1.0\n"):
        assert out[header_end:].count(b"\n") == len(cloud) and out.endswith(b"\n")
        return
    per = 12 + (3 if cloud.rgb is not None else 0) + (4 if cloud.label is not None else 0) \
        + (12 if cloud.normals is not None else 0)
    assert len(out) == header_end + per * len(cloud)


def test_down_server_fails_fast(image: Path, tmp_path: Path) -> None:
    for script, args in [
        ("reconstruct.sh", ["-i", str(image)]),
        ("reconstruct.sh", ["-i", str(image), "-f", "ply", "-p", "voxel=0.01"]),
        ("segment.sh", ["-i", str(image)]),
        ("segment.sh", ["-i", str(image), "-o", str(tmp_path / "x.json")]),
        ("mapper.sh", ["update", "-i", str(image), "-m", str(tmp_path / "m")]),
        ("mapper.sh", ["update", "-i", str(image), "-m", str(tmp_path / "m"), "-f", "ply",
                       "-p", "voxel=0.05"]),
    ]:
        t0 = time.monotonic()
        res = sh(script, *args)
        assert res.returncode == 3, (script, res.stderr)
        assert time.monotonic() - t0 < 2.0
        assert res.stdout == b""
        assert b"./start_inference_server.sh" in res.stderr
    assert not (tmp_path / "x.json").exists() and not (tmp_path / "m").exists()


def test_bad_attributes_exit_2_before_the_server_is_contacted(image: Path, tmp_path: Path
                                                              ) -> None:
    """The server is down here: exit 2 (not 3) shows the options were checked first."""
    for script, args, hint in [
        ("reconstruct.sh", ["-i", str(image), "-f", "ply", "-p", "colour=rgb"],
         b"unknown point-cloud attribute"),
        ("reconstruct.sh", ["-i", str(image), "-p", "voxel=0.1"], b"-f ply"),
        ("segment.sh", ["-i", str(image), "-f", "ply", "-p", "stride=0"], b"stride must be"),
        ("segment.sh", ["-i", str(image), "-f", "ply", "-p", "color=height"], b"fixed to segment"),
        ("segment.sh", ["-i", str(image), "-p", "voxel=0.1"], b"-f ply or -d"),
        ("mapper.sh", ["update", "-i", str(image), "-m", str(tmp_path / "m"), "-f", "ply",
                       "-p", "stride=2"], b"pixel-level attribute"),
        ("mapper.sh", ["update", "-i", str(image), "-m", str(tmp_path / "m"), "-p", "voxel=0.1"],
         b"-f ply"),
    ]:
        res = sh(script, *args)
        assert res.returncode == 2, (script, args, res.stderr)
        assert res.stdout == b"" and hint in res.stderr, res.stderr
        assert b"start_inference_server" not in res.stderr
    assert not (tmp_path / "m").exists()


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
    res = sh("segment.sh", "-i", str(image), "-f", "ply", "-d", str(tmp_path / "o"))
    assert res.returncode == 0
    assert_one_ply(res.stdout)
    assert len(list((tmp_path / "o").iterdir())) == 5
    assert (tmp_path / "o" / "segments.ply").read_bytes() == res.stdout


def test_output_file_leaves_stdout_empty(stub_server: None, image: Path, tmp_path: Path) -> None:
    """``-o <file>`` on the three result-writing commands: the file holds exactly the payload
    stdout would have carried, and stdout stays empty."""
    for script, args, kind in [
        ("reconstruct.sh", ["-i", str(image)], "json"),
        ("reconstruct.sh", ["-i", str(image), "-f", "ply", "-p", "normals=on,encoding=ascii"],
         "ply"),
        ("segment.sh", ["-i", str(image), "-f", "ply", "-p", "label=on"], "ply"),
        ("segment.sh", ["-i", str(image), "-d", str(tmp_path / "art")], "json"),
        ("mapper.sh", ["update", "-i", str(image), "-m", str(tmp_path / "map")], "json"),
        ("mapper.sh", ["update", "-i", str(image), "-m", str(tmp_path / "map2"), "-t", "single",
                       "-f", "ply", "-p", "color=segment,label=on,voxel=0.05"], "ply"),
    ]:
        target = tmp_path / "results" / f"{script}-{len(args)}.{kind}"
        res = sh(script, *args, "-o", str(target))
        assert res.returncode == 0, (script, args, res.stderr.decode())
        assert res.stdout == b"", script
        data = target.read_bytes()
        if kind == "json":
            assert_one_json(data)
        else:
            assert_one_ply(data)
            assert b"comment attributes " in data[:data.find(b"end_header")]
        if script == "segment.sh" and "-d" in args:
            assert (tmp_path / "art" / "segmentation.json").read_bytes() == data
    assert not list((tmp_path / "results").glob(".*.tmp"))  # atomic writes leave no temp files


def test_timings_go_to_stderr_and_the_env_file_only(stub_server: None, image: Path,
                                                   tmp_path: Path) -> None:
    """Instrumentation (R44): a summary line on stderr, the full record in $OH_MY_SLAM_TIMINGS;
    stdout is still exactly one payload and the same document as without the variable."""
    plain = sh("reconstruct.sh", "-i", str(image))
    for script, args, stages in [
        ("reconstruct.sh", ["-i", str(image)],
         {"connect", "inference", "segment", "export", "write"}),
        ("reconstruct.sh", ["-i", str(image), "-f", "ply"], {"connect", "inference", "export"}),
        ("segment.sh", ["-i", str(image), "-d", str(tmp_path / "o")],
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
        ("segment.sh", ["-m", str(tmp_path), "--min-score", "0.3"]),
        ("segment.sh", []),
        ("segment.sh", ["-i", str(image), "-o", str(tmp_path)]),  # -o names a file, not a folder
        ("mapper.sh", ["update", "-i", str(image), "-m", str(tmp_path / "m2"), "-f", "ply",
                       "-p", "min-depth=1"]),
        ("mapper.sh", ["update", "-a", str(image), "-m", str(tmp_path / "m2")]),  # -i, not -a
        ("mapper.sh", ["update", "-i", str(image)]),
    ]:
        res = sh(script, *args)
        assert res.returncode == 2, (script, args, res.stderr)
        assert res.stdout == b""
