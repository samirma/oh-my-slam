"""The service's real backend (``backend.MapperBackend``, ``mapper_proc``) and its view of the
inference server (``inference``), without a model: the mapper process is replaced by scripts,
the inference client and the reconstruction API by doubles."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from oh_my_slam.client import protocol as p
from oh_my_slam.core.errors import DeviceMemoryError, InputError, ServerUnavailableError
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.service import backend as be
from oh_my_slam.service import inference as inf
from oh_my_slam.service import mapper_proc
from oh_my_slam.service.errors import ApiError

K = Intrinsics(300.0, 300.0, 160.0, 120.0, 320, 240, "given")


# -- the mapper process --------------------------------------------------------------------------


def fake_mapper(tmp_path: Path, body: str) -> tuple[str, ...]:
    """A mapper command: a script given the args file, ``body`` run with ``args`` loaded."""
    script = tmp_path / "fake_mapper.py"
    script.write_text("import json, sys, time\nargs = json.load(open(sys.argv[1]))\n"
                      "def result(d):\n    json.dump(d, open(args['result'], 'w'))\n" + body)
    return (sys.executable, str(script))


@pytest.mark.parametrize(("body", "code", "message"), [
    ("result({'rejected': ['x'], 'objects': 2, 'seconds': 1.5, 'stages': {'cloud': 1.0}})",
     None, None),
    ("result({'error': {'message': 'none placed'}}); sys.exit(5)", "nothing", "none placed"),
    ("result({'error': {'message': 'down'}}); sys.exit(3)", "inference_unavailable", "down"),
    ("result({'error': {'message': 'oom', 'out_of_memory': True}}); sys.exit(1)",
     "out_of_memory", "oom"),
    ("print('trace line'); sys.exit(1)", "internal", "exit code 1"),
])
def test_register_reads_the_mapper_s_result(tmp_path: Path, body: str, code: str | None,
                                            message: str | None) -> None:
    b = be.MapperBackend(fake_mapper(tmp_path, body))
    img = tmp_path / "0000.jpg"
    img.write_bytes(b"x")
    log = tmp_path / "mapper0.log"
    if code is None:
        r = b.register(tmp_path / "mapper", [img], {img: K}, log)
        assert (r.sources_rejected, r.objects, r.seconds, r.stages) == (["x"], 2, 1.5,
                                                                        {"cloud": 1.0})
        args = json.loads(log.with_suffix(".args.json").read_text())
        assert args["intrinsics"][str(img)]["fx"] == 300.0
        return
    with pytest.raises((be.NothingRegisteredError, ApiError)) as e:
        b.register(tmp_path / "mapper", [img], {img: K}, log)
    assert message in str(e.value)
    if code != "nothing":
        assert isinstance(e.value, ApiError) and e.value.code == code
    if code == "internal":
        assert "trace line" in e.value.details["log"]


def test_terminate_ends_a_running_mapper(tmp_path: Path) -> None:
    b = be.MapperBackend(fake_mapper(tmp_path, "time.sleep(30)"))
    b.terminate()  # nothing running
    img = tmp_path / "a.jpg"
    out: list[Any] = []

    def run() -> None:
        try:
            b.register(tmp_path / "m", [img], {}, tmp_path / "log.log")
        except ApiError as exc:
            out.append(exc)

    t = threading.Thread(target=run)
    t.start()
    deadline = time.monotonic() + 10
    while b._proc is None and time.monotonic() < deadline:
        time.sleep(0.01)
    b.terminate()
    t.join(10)
    assert out and out[0].code == "internal"
    assert be._tail(tmp_path / "missing.log") == ""


def test_mapper_proc_runs_the_mapper_with_the_given_intrinsics(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from oh_my_slam.mapping import api

    seen: dict[str, Any] = {}

    def update(map_dir: Path, inputs: list[Path], **kw: Any) -> Any:
        seen.update(kw)
        return api.UpdateResult(["f000003"], ["f000004"], 2.0,
                                {"stages_s": {"sfm": 1.0, "cloud": 0.5}}, objects=2)

    monkeypatch.setattr(api, "update", update)
    m = tmp_path / "map"
    m.mkdir()
    (m / "map.json").write_text(json.dumps({"next_frame_index": 3}))
    args = {"map_dir": str(m), "inputs": ["/a.jpg", "/b.jpg"],
            "intrinsics": {"/a.jpg": K.to_dict()}, "result": str(tmp_path / "r.json")}
    (tmp_path / "args.json").write_text(json.dumps(args))
    assert mapper_proc.main([str(tmp_path / "args.json")]) == 0
    assert json.loads((tmp_path / "r.json").read_text()) == {
        "rejected": ["/b.jpg"], "objects": 2, "seconds": 2.0,
        "stages": {"sfm": 1.0, "cloud": 0.5}}
    assert seen["intrinsics"] == {Path("/a.jpg"): K}

    def fails(*a: Any, **kw: Any) -> Any:
        raise DeviceMemoryError("out of memory")

    monkeypatch.setattr(api, "update", fails)
    args["map_dir"] = str(tmp_path / "new")
    assert mapper_proc.run(args) == 1
    err = json.loads((tmp_path / "r.json").read_text())["error"]
    assert err["out_of_memory"] and err["exit_code"] == 1


def test_mapper_proc_as_a_process(tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    (other / "x.txt").write_text("not a map")
    img = tmp_path / "a.jpg"
    Image.new("RGB", (8, 8)).save(img)
    args = tmp_path / "args.json"
    args.write_text(json.dumps({"map_dir": str(other), "inputs": [str(img)], "intrinsics": {},
                                "result": str(tmp_path / "r.json")}))
    r = subprocess.run([sys.executable, "-m", "oh_my_slam.service.mapper_proc", str(args)],
                       capture_output=True, timeout=120)
    assert r.returncode == 4
    assert "not a map" in json.loads((tmp_path / "r.json").read_text())["error"]["message"]


# -- reading the mapper's map --------------------------------------------------------------------


def test_read_gives_the_internal_frames_floor_bounds_and_depth(tmp_path: Path) -> None:
    from oh_my_slam.core.ply import PointCloud, ply_bytes
    from oh_my_slam.mapping.store import FrameRecord

    m = tmp_path / "map"
    (m / "per_frame" / "f000000").mkdir(parents=True)
    recs = [FrameRecord(i, f"f{i:06d}", f"frames/f{i:06d}.jpg", f"/in/{i}.jpg", 1, 320, 240,
                        K, Pose(np.eye(3), np.array([i, 0.0, 1.0])), 32, 24,
                        stats={"observations": 50.0}, low_confidence=bool(i))
            for i in range(2)]
    (m / "frames.json").write_text(json.dumps({"frames": [r.to_dict() for r in recs]}))
    (m / "map.json").write_text(json.dumps({"floor_z": -1.0, "updates": [
        {"notes": {"restarted": {}}}]}))
    np.save(m / "per_frame" / "f000000" / "depth.npy", np.full((24, 32), 2.0, np.float16))
    b = be.MapperBackend()
    r = b.read(m)
    assert [f.name for f in r.frames] == ["f000000", "f000001"] and r.frames[1].low_confidence
    assert r.frames[0].source == "/in/0.jpg" and r.frames[0].T[0, 3] == 0.0
    assert r.floor_z == -1.0 and r.rebuilt and r.bounds is None
    d = r.depth("f000000")
    assert d is not None and float(d.mean()) == 2.0
    assert r.depth("f000001") is None and r.depth("nope") is None
    (m / "cloud.ply").write_bytes(ply_bytes(PointCloud(np.array([[0, 0, 0], [1, 2, 3]],
                                                                np.float32))))
    (m / "map.json").write_text(json.dumps({}))
    r = b.read(m)
    assert r.bounds == ([0, 0, 0], [1, 2, 3]) and r.floor_z is None and not r.rebuilt
    (m / "cloud.ply").write_bytes(ply_bytes(PointCloud(np.zeros((0, 3), np.float32))))
    assert b.read(m).bounds is None


def test_read_gives_keyframe_images_the_cloud_and_objects(tmp_path: Path) -> None:
    from oh_my_slam.core.ply import PointCloud, ply_bytes
    from oh_my_slam.mapping.store import FrameRecord

    m = tmp_path / "map"
    m.mkdir()
    rec = FrameRecord(0, "f000000", "frames/f000000.jpg", "/in/0.jpg", 1, 320, 240, K,
                      Pose(np.eye(3), np.zeros(3)), 32, 24)
    (m / "frames.json").write_text(json.dumps({"frames": [rec.to_dict()]}))
    (m / "map.json").write_text(json.dumps({}))
    b = be.MapperBackend()
    r = b.read(m)
    assert r.image("f000000") == m / "frames" / "f000000.jpg" and r.image("nope") is None
    assert r.cloud() is None and r.objects() == []
    (m / "cloud.ply").write_bytes(ply_bytes(PointCloud(np.array([[0, 0, 0], [1, 2, 3]],
                                                                np.float32))))
    c = b.read(m).cloud()
    assert c is not None and c.rgb.shape == (2, 3) and c.object_id.tolist() == [0, 0]
    rgb = np.array([[1, 2, 3], [4, 5, 6]], np.uint8)
    (m / "cloud.ply").write_bytes(ply_bytes(PointCloud(np.zeros((2, 3), np.float32), rgb)))
    np.save(m / "cloud_objects.npy", np.array([-1, 7], np.int32))
    c = b.read(m).cloud()
    assert c is not None and c.rgb.tolist() == rgb.tolist() and c.object_id.tolist() == [0, 7]
    np.save(m / "cloud_objects.npy", np.array([7], np.int32))  # stale: of another cloud
    c = b.read(m).cloud()
    assert c is not None and c.object_id.tolist() == [0, 0]
    (m / "objects.json").write_text(json.dumps({"objects": [
        {"id": 3, "label": "door", "obb": {"center": [1, 2, 3], "R": np.eye(3).tolist(),
                                           "size": [1, 0.1, 2]}},
        {"id": 4, "label": "ghost"}]}))
    assert b.read(m).objects() == [{"id": 3, "label": "door", "center": [1, 2, 3],
                                    "R": np.eye(3).tolist(), "size": [1, 0.1, 2]}]


# -- the depth model -----------------------------------------------------------------------------


class _Client:
    def close(self) -> None:
        pass

    def clone(self) -> _Client:
        return _Client()


def test_median_depth_asks_the_depth_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch
                                           ) -> None:
    from oh_my_slam.reconstruction import api as rec

    seen: dict[str, Any] = {}

    def reconstruct(path: Path, **kw: Any) -> Any:
        seen.update(kw)
        return type("F", (), {"depth": np.array([[1.0, 3.0], [5.0, 9.0]], np.float32),
                              "valid": np.array([[True, True], [True, False]])})()

    monkeypatch.setattr(rec, "connect_server", lambda: _Client())
    monkeypatch.setattr(rec, "reconstruct_image", reconstruct)
    assert be.MapperBackend().median_depth(tmp_path / "a.jpg", K) == 3.0
    assert seen["intrinsics"] == K and seen["max_side"] == 512
    for exc, code in ((ServerUnavailableError("down"), "inference_unavailable"),
                      (DeviceMemoryError("oom"), "out_of_memory"), (InputError("bad"), "internal")):
        def boom(e: Exception = exc) -> Any:
            raise e

        monkeypatch.setattr(rec, "connect_server", boom)
        with pytest.raises(ApiError) as err:
            be.MapperBackend().median_depth(tmp_path / "a.jpg", K)
        assert err.value.code == code


def test_infer_returns_the_monocular_depth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch
                                           ) -> None:
    from oh_my_slam.reconstruction import api as rec

    seen: dict[str, Any] = {}
    grav = type("G", (), {"up_cam": (0.0, -1.0, 0.0)})()

    def reconstruct(path: Path, **kw: Any) -> Any:
        seen.update(kw)
        return type("F", (), {"depth": np.array([[1.0, 3.0]], np.float32),
                              "valid": np.array([[True, False]]), "K_grid": K,
                              "descriptor": np.ones(4) if kw["want_gravity"] else None,
                              "gravity": grav if kw["want_gravity"] else None})()

    monkeypatch.setattr(rec, "connect_server", lambda: _Client())
    monkeypatch.setattr(rec, "reconstruct_image", reconstruct)
    m = be.MapperBackend().infer(tmp_path / "a.jpg", K, gravity=True)
    assert m.depth.tolist() == [[1.0, 0.0]] and m.up_cam is not None and m.descriptor is not None
    assert seen["max_side"] == be.MONO_GRID_SIDE and seen["want_descriptor"]
    d, Kg, desc, up = be.MapperBackend().infer(tmp_path / "a.jpg", K).as_tuple()
    assert desc is None and up is None and Kg == K

    def boom() -> Any:
        raise ServerUnavailableError("down")

    monkeypatch.setattr(rec, "connect_server", boom)
    with pytest.raises(ApiError) as err:
        be.MapperBackend().infer(tmp_path / "a.jpg", K)
    assert err.value.code == "inference_unavailable"


# -- the inference server ------------------------------------------------------------------------


def health(status: str, memory: bool = True, error: str | None = None) -> p.Health:
    return p.Health(status=status, device="mps",
                    memory=p.DeviceMemory(allocated_gib=2.0, driver_gib=3.0, limit_gib=10.0)
                    if memory else None,
                    models={"depth": p.ModelStatus(name="depth", error=error)})


def test_inference_ready_and_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    answers: list[Any] = []

    def ask(self: Any, timeout: float = 1.0) -> p.Health:
        a = answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a

    monkeypatch.setattr(inf.InferenceClient, "health", ask)
    answers[:] = [health("ready")]
    inf.inference_ready()
    for a, state in ((ServerUnavailableError("no socket"), "down"), (health("loading"), "starting")):
        answers[:] = [a]
        with pytest.raises(ApiError) as e:
            inf.inference_ready()
        assert e.value.code == "inference_unavailable" and e.value.details["state"] == state
    answers[:] = [health("ready")]
    be.MapperBackend().require_inference()
    i = inf.Inference()
    answers[:] = [health("ready"), health("degraded"), health("error", False, "no weights"),
                  health("error", False), ServerUnavailableError("no socket")]
    s = i.state()
    assert s == {"state": "ready", "device": "mps", "memory_used_gb": 2.147,
                 "memory_cap_gb": 10.737, "error": None}
    assert i.state()["error"].startswith("2.00 of 10.00 GiB")
    assert i.state()["error"] == "depth: no weights"
    assert i.state() == {"state": "down", "device": "mps", "memory_used_gb": None,
                         "memory_cap_gb": None, "error": "models failed to load"}
    assert i.state()["state"] == "down" and not i.started


def test_the_service_starts_and_stops_the_inference_server(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    flag = tmp_path / "ran"
    monkeypatch.setenv("OH_MY_SLAM_INFERENCE_CMD",
                       f"{sys.executable} -c \"import sys, time; open('{flag}', 'a')"
                       f".write(' '.join(sys.argv[1:]) + ';'); time.sleep(1)\"")
    monkeypatch.setattr(inf.ServerLock, "is_held", staticmethod(lambda path=None: False))
    i = inf.Inference()
    i.ensure_started()
    assert i.started and i.state()["state"] == "starting"
    i.launcher.wait(10)  # type: ignore[union-attr]
    assert i.state()["state"] == "down"
    assert flag.read_text() == ";"
    monkeypatch.delenv("OH_MY_SLAM_INFERENCE_CMD")
    assert inf.inference_cmd()[1:] == ["-m", "oh_my_slam.server.main"]
    # already answering, or starting (its process holds the lock): nothing is launched
    j = inf.Inference()
    monkeypatch.setattr(inf.InferenceClient, "health", lambda self, timeout=1.0: health("ready"))
    j.ensure_started()
    assert not j.started

    def down(self: Any, timeout: float = 1.0) -> p.Health:
        raise ServerUnavailableError("no socket")

    monkeypatch.setattr(inf.InferenceClient, "health", down)
    monkeypatch.setattr(inf.ServerLock, "is_held", staticmethod(lambda path=None: True))
    j.ensure_started()
    assert not j.started and j.state()["state"] == "starting"


def _holder(script: str) -> subprocess.Popen[str]:
    """A process holding the inference server's lock as ``script`` does, once it prints."""
    child = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    assert child.stdout is not None and child.stdout.readline() == "held\n"
    return child


HOLD = ("import signal, time\n"
        "from oh_my_slam.server.lifecycle import ServerLock\n"
        "{extra}"
        "ServerLock().acquire()\n"
        "print('held', flush=True)\n"
        "time.sleep(60)\n")


def test_stop_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    import signal

    from oh_my_slam.core import paths

    assert inf.stop_inference() == "the inference server was not running"
    child = _holder(HOLD.format(extra=""))
    assert inf.stop_inference() == f"stopped the inference server (pid {child.pid})"
    assert child.wait(10) == -signal.SIGTERM
    monkeypatch.setattr(inf, "STOP_TIMEOUT_S", 0.3)
    child = _holder(HOLD.format(extra="signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"))
    assert inf.stop_inference() == f"stopped the inference server (pid {child.pid}, killed)"
    assert child.wait(10) == -signal.SIGKILL
    child = _holder("import fcntl, os, time\n"
                    f"fd = os.open({str(paths.lock_file())!r}, os.O_RDWR | os.O_CREAT)\n"
                    "os.ftruncate(fd, 0)\n"
                    "fcntl.flock(fd, fcntl.LOCK_EX)\n"
                    "print('held', flush=True)\n"
                    "time.sleep(60)\n")
    try:
        assert "without its pid" in inf.stop_inference()
    finally:
        child.kill()
        child.wait(10)


def test_read_of_an_anchored_map_without_cloud(tmp_path: Path) -> None:
    from oh_my_slam.mapping.anchored import AnchoredMap

    am = AnchoredMap(tmp_path / "mapper")
    am.root.mkdir()
    am.save()
    m = be.MapperBackend().read(am.root)
    assert m.frames == [] and m.cloud() is None and m.points_z is None and m.bounds is None
    assert m.depth("k000000") is None and m.image("k000000") is None and m.objects() == []


def test_detect_asks_the_detector_on_a_grid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch
                                            ) -> None:
    from oh_my_slam.reconstruction import api as rec
    from oh_my_slam.segmentation import api as seg

    seen: dict[str, Any] = {}

    def detect_on_grid(path: Path, grid: tuple[int, int], client: Any, **kw: Any) -> list[Any]:
        seen["grid"] = grid
        seen.update(kw)
        return ["d"]

    monkeypatch.setattr(rec, "connect_server", lambda: _Client())
    monkeypatch.setattr(seg, "detect_on_grid", detect_on_grid)
    assert be.MapperBackend().detect(tmp_path / "a.jpg", (64, 48)) == ["d"]
    assert seen["grid"] == (64, 48) and seen["min_score"] == be.MAP_MIN_SCORE

    def boom() -> Any:
        raise ServerUnavailableError("down")

    monkeypatch.setattr(rec, "connect_server", boom)
    with pytest.raises(ApiError) as err:
        be.MapperBackend().detect(tmp_path / "a.jpg", (64, 48))
    assert err.value.code == "inference_unavailable"
