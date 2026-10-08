"""stdout carries exactly one JSON document, one PLY or one PNG for every command (AC21) — or
nothing with ``-o <file>`` — and the inference commands fail fast with exit 3 when the server is
down (AC2) while option errors (e.g. a bad ``-p``) exit 2 before the server is contacted. Runs the
real shell entry points against a server process with stub models."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from oh_my_slam.core.constants import UPDATE_EXHAUSTIVE_MAX
from oh_my_slam.core.ply import parse_ply
from tests.fakes.stub_server import start_stub_server, stub_env

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
    start_stub_server()
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


def assert_one_png(out: bytes) -> np.ndarray:
    """Exactly one PNG: its signature first, its IEND chunk last, and pixels that decode."""
    assert out.startswith(b"\x89PNG\r\n\x1a\n") and out.endswith(b"\0\0\0\0IEND\xaeB`\x82")
    with Image.open(io.BytesIO(out)) as im:
        return np.asarray(im)


# The command's own failure path, timed in a process of its own once the interpreter has started
# and loaded the command's module (that start-up is no part of failing fast, and under load it
# alone can take seconds): from ``main(argv)`` to the error ``run_main`` turns into the exit status.
FAILURE_PATH = r"""
import importlib, json, sys, time
from oh_my_slam.core.errors import OhMySlamError
record, module, *argv = sys.argv[1:]
main = importlib.import_module(module).main
t0 = time.perf_counter()
try:
    code = main(argv)
except OhMySlamError as exc:
    code = int(exc.exit_code)
with open(record, "w") as f:
    json.dump({"code": code, "seconds": time.perf_counter() - t0}, f)
"""
FAIL_FAST_S = 2.0  # what failing fast means: no wait for the server (a health timeout, a retry)


def failure_path(script: str, args: list[str], record: Path) -> dict[str, Any]:
    """The exit status and the seconds of ``script``'s own path from its ``main`` to its error."""
    module = f"oh_my_slam.cli.{script.removesuffix('.sh')}"  # what the script execs
    res = subprocess.run([sys.executable, "-c", FAILURE_PATH, str(record), module, *args],
                         capture_output=True, timeout=120, env=os.environ.copy(), cwd=REPO)
    assert res.returncode == 0, res.stderr
    return json.loads(record.read_text())  # type: ignore[no-any-return]


def make_clip(path: Path) -> Path:
    """A short real video (one second of noise frames)."""
    import av

    with av.open(str(path), "w") as out:
        stream = out.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        rng = np.random.default_rng(0)
        for _ in range(10):
            frame = av.VideoFrame.from_ndarray(rng.integers(0, 256, (48, 64, 3), np.uint8),
                                               format="rgb24")
            for pkt in stream.encode(frame):
                out.mux(pkt)
        for pkt in stream.encode():
            out.mux(pkt)
    return path


def test_down_server_fails_fast(image: Path, tmp_path: Path) -> None:
    """Exit 3 with the start command on stderr, nothing on stdout and no file or folder written
    — not even the missing folders of an ``-o`` file, nor a new map holding it; and the command's
    failure path takes no time (``failure_path``)."""
    work = tmp_path / "work"
    work.mkdir()
    clip = make_clip(tmp_path / "walk.mp4")
    for script, args in [
        ("reconstruct.sh", ["-i", str(image)]),
        ("reconstruct.sh", ["-i", str(image), "-f", "ply", "-p", "voxel=0.01"]),
        ("reconstruct.sh", ["-i", str(image), "-f", "depth"]),
        ("reconstruct.sh", ["-i", str(image), "-o", str(work / "new" / "dir" / "x.json")]),
        ("segment.sh", ["-i", str(image)]),
        ("segment.sh", ["-i", str(image), "-o", str(work / "x.json")]),
        ("segment.sh", ["-i", str(image), "-f", "png", "-o", str(work / "new2" / "x.png")]),
        ("segment.sh", ["-i", str(image), "-f", "png", "-d", str(work / "d")]),
        ("mapper.sh", ["update", "-i", str(image), "-m", str(work / "m")]),
        ("mapper.sh", ["update", "-i", str(image), "-m", str(work / "m"), "-f", "ply",
                       "-p", "voxel=0.05"]),
        ("mapper.sh", ["update", "-i", str(image), "-m", str(work / "m"),
                       "-o", str(work / "m" / "sub" / "r.json")]),
        ("mapper.sh", ["update", "-i", str(clip), "-m", str(work / "m")]),  # a video
        ("mapper.sh", ["update", "-i", str(clip), "-m", str(work / "m"), "-fps", "1",
                       "-o", str(work / "new3" / "r.json")]),
    ]:
        res = sh(script, *args)
        assert res.returncode == 3, (script, args, res.stderr)
        assert res.stdout == b""
        assert b"./start_inference_server.sh" in res.stderr
        timed = failure_path(script, args, tmp_path / "failure.json")
        assert timed["code"] == 3 and timed["seconds"] < FAIL_FAST_S, (script, args, timed)
        assert list(work.iterdir()) == [], (script, args)  # no file, no folder, no map


# ``mapper.sh locate`` on a map of more keyframes than are matched exhaustively, through the
# command: the map is a stand-in (its keyframes' features and descriptors), and retrieval needs
# the query's descriptor from the inference server.
LOCATE_LARGE_MAP = r"""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from oh_my_slam.cli import mapper
from oh_my_slam.core.constants import UPDATE_EXHAUSTIVE_MAX
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.mapping import locate, store

root, *argv = sys.argv[1:]
K = Intrinsics(300.0, 300.0, 200.0, 150.0, 400, 300)
frames = [store.FrameRecord(i, f"f{i:06d}", f"frames/f{i:06d}.jpg", "", 1, 400, 300, K,
                            Pose.identity(), 400, 300) for i in range(UPDATE_EXHAUSTIVE_MAX + 1)]
reader = SimpleNamespace(root=Path(root), frames=frames, meta={}, read_json=lambda name: {},
                         descriptor=lambda fr: np.eye(8)[fr.index % 8])
locate.open_map = lambda map_dir: reader
db = {f"{fr.name}.jpg": locate._DbImage(fr.index + 1, 1) for fr in frames}
locate._extract = lambda reader, queries, work: (
    SimpleNamespace(match_pairs=lambda pairs, names: None),
    {**db, **{q.name: locate._DbImage(len(frames) + 1 + q.index, 2) for q in queries}})
sys.argv = ["mapper.sh", *argv]
mapper.entry()
"""


def test_locate_in_a_large_map_needs_the_server(image: Path, tmp_path: Path) -> None:
    """``mapper.sh locate`` in a map of more than ``UPDATE_EXHAUSTIVE_MAX`` keyframes retrieves
    candidate keyframes with the query's descriptor: with the server down, exit 3 and the start
    command on stderr, nothing on stdout."""
    res = subprocess.run([sys.executable, "-c", LOCATE_LARGE_MAP, str(tmp_path),
                          "locate", "-i", str(image), "-m", str(tmp_path)],
                         capture_output=True, timeout=120, env=os.environ.copy(), cwd=REPO)
    assert res.returncode == 3, res.stderr.decode()
    assert res.stdout == b"" and b"./start_inference_server.sh" in res.stderr
    keyframes = f"({UPDATE_EXHAUSTIVE_MAX + 1} keyframes)".encode()
    assert keyframes in res.stderr  # the stand-in map was the one located in


# ``start_inference_server.sh`` with no option, launching the stub server instead of the models
START_WITH_STUB = r"""
from oh_my_slam.cli import server
from tests.fakes.stub_server import SERVER_CMD
server._server_cmd = lambda: SERVER_CMD
server.entry()
"""


def test_the_server_command_writes_stdout_for_status_only() -> None:
    """``start_inference_server.sh``: with the server down, ``--status`` (exit 3) and ``--stop``
    (exit 0) write nothing on stdout, and neither does a start; once it runs, ``--status`` writes
    exactly its health JSON, and ``--stop`` nothing."""
    for args, code in ((["--status"], 3), (["--stop"], 0)):
        res = sh("start_inference_server.sh", *args)
        assert (res.returncode, res.stdout) == (code, b""), (args, res.stderr)
    try:
        res = subprocess.run([sys.executable, "-c", START_WITH_STUB], capture_output=True,
                             timeout=120, env=stub_env(), cwd=REPO)
        assert (res.returncode, res.stdout) == (0, b""), res.stderr.decode()
        assert b"ready after" in res.stderr
        res = sh("start_inference_server.sh", "--status")
        assert res.returncode == 0 and assert_one_json(res.stdout)["status"] == "ready"
    finally:
        res = sh("start_inference_server.sh", "--stop")
    assert (res.returncode, res.stdout) == (0, b"") and b"stopped (pid" in res.stderr


def test_bad_attributes_exit_2_before_the_server_is_contacted(image: Path, tmp_path: Path
                                                              ) -> None:
    """The server is down here: exit 2 (not 3) shows the options and the input image were checked
    first."""
    notes, bad, gif = tmp_path / "notes.txt", tmp_path / "bad.jpg", tmp_path / "real.gif"
    notes.write_text("not an image")
    bad.write_text("not an image")
    Image.new("RGB", (8, 8)).save(gif)  # a readable image of a suffix -i does not take
    for script, args, hint in [
        *[(s, ["-i", str(f), *more], b"unsupported input (not an image)")
          for s, more in (("reconstruct.sh", []), ("segment.sh", []), ("view.sh", ["--no-browser"]))
          for f in (notes, gif)],
        *[(s, ["-i", str(bad), *more], b"cannot read image")
          for s, more in (("reconstruct.sh", ["-f", "ply"]), ("segment.sh", ["-f", "png"]),
                          ("view.sh", ["--no-browser"]))],
        ("reconstruct.sh", ["-i", str(image), "-f", "ply", "-p", "colour=rgb"],
         b"unknown point-cloud attribute"),
        ("reconstruct.sh", ["-i", str(image), "-p", "voxel=0.1"], b"-f ply"),
        ("reconstruct.sh", ["-i", str(image), "-f", "depth", "-p", "voxel=0.1"],
         b"only the PLY output has: use -f ply"),
        ("segment.sh", ["-i", str(image), "-p", "voxel=0.1"], b"unrecognized arguments: -p"),
        ("segment.sh", ["-i", str(image), "-f", "ply"], b"invalid choice: 'ply'"),
        ("mapper.sh", ["update", "-i", str(image), "-m", str(tmp_path / "m"), "-f", "ply",
                       "-p", "stride=2"], b"pixel-level attribute"),
        ("mapper.sh", ["update", "-i", str(image), "-m", str(tmp_path / "m"), "-p", "voxel=0.1"],
         b"-f ply"),
        ("mapper.sh", ["locate", "-i", str(image), "-m", str(tmp_path / "m"), "-f", "ply",
                       "-p", "stride=2"], b"pixel-level attribute"),
        ("mapper.sh", ["locate", "-i", str(image), "-m", str(tmp_path / "m"), "-p", "voxel=0.1"],
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
    res = sh("reconstruct.sh", "-i", str(image), "-f", "depth")
    assert res.returncode == 0, res.stderr.decode()
    depth = assert_one_png(res.stdout)
    assert depth.dtype == np.uint16 and depth.shape == (240, 320)  # the input's pixel size
    res = sh("segment.sh", "-i", str(image))
    assert res.returncode == 0
    doc = assert_one_json(res.stdout)
    assert doc["openlabel"]["objects"]
    res = sh("segment.sh", "-i", str(image), "-f", "png", "-d", str(tmp_path / "o"))
    assert res.returncode == 0
    assert assert_one_png(res.stdout).dtype == np.uint8
    assert len(list((tmp_path / "o").iterdir())) == 4
    assert (tmp_path / "o" / "segmented.png").read_bytes() == res.stdout


def test_output_file_leaves_stdout_empty(stub_server: None, image: Path, tmp_path: Path) -> None:
    """``-o <file>`` on the three result-writing commands: the file holds exactly the payload
    stdout would have carried, and stdout stays empty."""
    for script, args, kind in [
        ("reconstruct.sh", ["-i", str(image)], "json"),
        ("reconstruct.sh", ["-i", str(image), "-f", "ply", "-p", "normals=on,encoding=ascii"],
         "ply"),
        ("reconstruct.sh", ["-i", str(image), "-f", "depth"], "png"),
        ("segment.sh", ["-i", str(image), "-f", "png"], "png"),
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
        elif kind == "png":
            assert_one_png(data)
        else:
            assert_one_ply(data)
            assert b"comment attributes " in data[:data.find(b"end_header")]
        if script == "segment.sh" and "-d" in args:
            assert (tmp_path / "art" / "segmentation.json").read_bytes() == data
    assert not list((tmp_path / "results").glob(".*.tmp"))  # atomic writes leave no temp files


@pytest.mark.skipif(shutil.which("colmap") is None, reason="needs Homebrew colmap")
def test_mapper_update_stdout_on_a_multi_image_map(stub_server: None, tmp_path: Path) -> None:
    """``mapper.sh update`` through the shell on several images (COLMAP features and matching,
    the map's frames registered together): stdout is exactly one JSON document, then, extending
    the map, exactly one PLY; with ``-o`` stdout stays empty. Progress and timings go to stderr."""
    frames = sorted((REPO / "examples" / "ainex-captures").glob("*.jpg"))
    first, second = frames[:4], frames[4:6]
    mdir = tmp_path / "map"
    res = sh("mapper.sh", "update", "-i", *map(str, first), "-m", str(mdir), timeout=600)
    assert res.returncode == 0, res.stderr.decode()
    doc = assert_one_json(res.stdout)
    assert len(doc["openlabel"]["frames"]) >= 2  # a multi-image map
    assert b"timings: total" in res.stderr
    res = sh("mapper.sh", "update", "-i", *map(str, second), "-m", str(mdir), "-f", "ply",
             "-t", "single", timeout=600)
    assert res.returncode == 0, res.stderr.decode()
    assert_one_ply(res.stdout)
    target = tmp_path / "full.json"
    res = sh("mapper.sh", "update", "-i", str(frames[6]), "-m", str(mdir), "-o", str(target),
             timeout=600)
    assert res.returncode == 0, res.stderr.decode()
    assert res.stdout == b""
    assert len(assert_one_json(target.read_bytes())["openlabel"]["frames"]) >= len(
        doc["openlabel"]["frames"])


def test_an_output_file_inside_a_new_map(stub_server: None, image: Path, tmp_path: Path) -> None:
    """``mapper.sh update -m new -o new/sub/r.json``: the ``-o`` folders are created with the
    result, once the new map exists — not before, when they would make the folder a non-empty
    one that is not a map (exit 4)."""
    new = tmp_path / "new"
    target = new / "sub" / "r.json"
    res = sh("mapper.sh", "update", "-i", str(image), "-m", str(new), "-o", str(target))
    assert res.returncode == 0, res.stderr.decode()
    assert res.stdout == b"" and "openlabel" in assert_one_json(target.read_bytes())
    assert (new / "map.json").is_file()


def test_timings_go_to_stderr_and_the_env_file_only(stub_server: None, image: Path,
                                                   tmp_path: Path) -> None:
    """Instrumentation (R44): a summary line on stderr, the full record in $OH_MY_SLAM_TIMINGS;
    stdout is still exactly one payload and the same document as without the variable."""
    plain = sh("reconstruct.sh", "-i", str(image))
    for script, args, stages in [
        ("reconstruct.sh", ["-i", str(image)],
         {"connect", "inference", "segment", "export", "write"}),
        ("reconstruct.sh", ["-i", str(image), "-f", "ply"], {"connect", "inference", "export"}),
        ("reconstruct.sh", ["-i", str(image), "-f", "depth"],
         {"connect", "inference", "export", "write"}),
        ("segment.sh", ["-i", str(image), "-d", str(tmp_path / "o")],
         {"inference", "segment", "export", "artifacts", "write"}),
    ]:
        target = tmp_path / f"{script}-{'-'.join(args[2:4])}.json"
        res = subprocess.run([str(REPO / script), *args], capture_output=True, timeout=120,
                             env={**os.environ, "OH_MY_SLAM_TIMINGS": str(target)})
        assert res.returncode == 0, res.stderr.decode()
        if "ply" in args:
            assert_one_ply(res.stdout)
        elif "depth" in args:
            assert_one_png(res.stdout)
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
        if "depth" in args:  # the depth model alone
            assert set(rec["server"]) == {"geometry"} and "segment" not in rec["stages_s"]
        if "ply" not in args and "depth" not in args:
            assert rec["server"]["segment"]["count"] == 1
            assert rec["server"]["gravity"]["count"] == 1
            assert {"gravity", "segmentation", "lift"} <= set(rec["parts"])
        assert rec["peak_rss_mb"]["self"] > 0


def test_usage_errors_exit_2(image: Path, tmp_path: Path) -> None:
    for script, args in [
        ("reconstruct.sh", []),
        ("reconstruct.sh", ["-i", str(image), "-f", "xyz"]),
        ("reconstruct.sh", ["-i", str(image), "-f", "depth", "-p", "normals=on"]),
        ("segment.sh", ["-i", str(image), "-m", str(tmp_path)]),
        ("segment.sh", ["-m", str(tmp_path), "--min-score", "0.3"]),
        ("segment.sh", ["-i", str(image), "-f", "ply"]),
        ("segment.sh", []),
        ("segment.sh", ["-i", str(image), "-o", str(tmp_path)]),  # -o names a file, not a folder
        ("mapper.sh", ["update", "-i", str(image), "-m", str(tmp_path / "m2"), "-f", "ply",
                       "-p", "min-depth=1"]),
        ("mapper.sh", ["update", "-a", str(image), "-m", str(tmp_path / "m2")]),  # -i, not -a
        ("mapper.sh", ["update", "-i", str(image)]),
        ("mapper.sh", ["locate", "-i", str(image)]),  # -m is required
        ("mapper.sh", ["locate", "-i", str(tmp_path / "walk.mp4"), "-m", str(tmp_path)]),  # video
        ("mapper.sh", ["locate", "-i", str(image), "-m", str(tmp_path / "m3")]),  # no map there
        ("mapper.sh", ["locate", "-i", str(image), "-m", str(tmp_path / "m3"), "-t", "all"]),
        ("mapper.sh", ["locate", "-i", str(image), "-m", str(tmp_path / "m3"), "-fps", "2"]),
    ]:
        res = sh(script, *args)
        assert res.returncode == 2, (script, args, res.stderr)
        assert res.stdout == b""
    assert not (tmp_path / "m3").exists()  # locate never creates the map folder


def test_help_goes_to_stderr() -> None:
    """The help is human-facing: stderr, never stdout (spec §4), for every entry point."""
    for script, args in [("reconstruct.sh", ["-h"]), ("segment.sh", ["-h"]), ("view.sh", ["-h"]),
                         ("start_inference_server.sh", ["-h"]), ("mapper.sh", ["-h"]),
                         ("mapper.sh", ["update", "-h"]), ("mapper.sh", ["locate", "-h"])]:
        res = sh(script, *args)
        assert res.returncode == 0, (script, res.stderr)
        assert res.stdout == b"" and b"usage:" in res.stderr, script


def test_an_undecodable_image_is_an_input_error(stub_server: None, tmp_path: Path) -> None:
    """A file with an image suffix that is not an image is an input error naming it (exit 2), not
    an internal error, for every command that reads images."""
    from oh_my_slam.mapping.api import update
    from tests.fakes.client import FakeClient
    from tests.synth.mapping import add_frames, mapping_room, ring

    bad = tmp_path / "bad.jpg"
    bad.write_text("not an image")
    client = FakeClient()
    keys = add_frames(client, mapping_room(), ring(1), tmp_path / "k", "k")
    update(tmp_path / "map", keys, client=client, progress=lambda m: None)
    for script, *args in (("reconstruct.sh", "-i", str(bad)),
                          ("segment.sh", "-i", str(bad)),
                          ("mapper.sh", "update", "-i", str(bad), "-m", str(tmp_path / "new")),
                          ("mapper.sh", "locate", "-i", str(bad), "-m", str(tmp_path / "map"))):
        res = sh(script, *args)
        assert res.returncode == 2 and res.stdout == b"", (script, res.stderr.decode())
        assert f"cannot read image {bad}".encode() in res.stderr, (script, res.stderr.decode())
