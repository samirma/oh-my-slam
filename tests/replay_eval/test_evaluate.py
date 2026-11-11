"""The reference evaluator (tools/evaluate/evaluate.py) against the fake service: the captures,
splits, metrics, contracts, baseline, recorded request sets and the local service start."""

from __future__ import annotations

import json
import stat
import struct
import sys
from pathlib import Path
from typing import Any

import cv2
import evaluate as ev
import numpy as np
import pytest
import rpl_targets as rt
from rpl_api import Api, Log
from rpl_validate import Validator

from .fake_server import Config, Fake

NAMES = [f"2026_{i:02d}.jpg" for i in range(6)]


def pose(i: int) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)  # looking along +x
    T[:3, 3] = [1.0 + 0.3 * i, 2.0, 1.0]
    return T


def jpeg(w: int = 64, h: int = 48) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), 128, np.uint8))
    assert ok
    return bytes(buf)


def exif_jpeg(f35: int, big_endian: bool = False) -> bytes:
    """A JPEG with an APP1 Exif segment holding FocalLengthIn35mmFilm."""
    e = ">" if big_endian else "<"
    head = (b"MM" if big_endian else b"II") + struct.pack(e + "HI", 42, 8)
    ifd0 = struct.pack(e + "H", 1) + struct.pack(e + "HHII", 0x8769, 4, 1, 26) + \
        struct.pack(e + "I", 0)
    exif = struct.pack(e + "H", 1) + struct.pack(e + "HHI", 0xA405, 3, 1) + \
        struct.pack(e + "HH", f35, 0) + struct.pack(e + "I", 0)
    tiff = head + ifd0 + exif
    seg = b"Exif\x00\x00" + tiff
    data = jpeg()
    return data[:2] + b"\xff\xe1" + struct.pack(">H", len(seg) + 2) + seg + data[2:]


@pytest.fixture(autouse=True)
def private_out(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Never read or write the machine's ~/oh-my-slam-data/evaluations."""
    monkeypatch.setattr(ev, "DEFAULT_OUT", tmp_path / "evals")


@pytest.fixture
def examples(tmp_path: Path) -> Path:
    root = tmp_path / "examples"
    (root / "office_sequence").mkdir(parents=True)
    for n in NAMES:
        (root / "office_sequence" / n).write_bytes(jpeg())
    (root / "office_sequence" / ".DS_Store").write_bytes(b"x")
    (root / "ground_truth").mkdir()
    (root / "ground_truth" / "office_sequence.json").write_text(json.dumps(
        {"kind": "map_update", "absent": [{"label": "cup"}], "splits": [[3, 3]]}))
    (root / "ainex-captures").mkdir()
    (root / "ainex-captures" / "001_bootstrap_level.jpg").write_bytes(b"not an image")
    return root


def config(**kw: Any) -> Config:
    return Config(truth={n: pose(i) for i, n in enumerate(NAMES)}, **kw)


def test_reference_run_metrics_contracts_and_baseline(examples: Path, tmp_path: Path) -> None:
    out, base = tmp_path / "out", tmp_path / "baseline.json"
    with Fake(config(eta=0.1)) as fake:
        code = ev.main(["--url", fake.url, "--out", str(out), "--examples", str(examples),
                        "--captures", "office_sequence,ainex-captures,missing",
                        "--splits", "single,halves,truth,per_frame,interleaved",
                        "--baseline", str(base), "--set-baseline", "--wait-s", "0.01"])
    res = json.loads((out / "result.json").read_text())
    assert (out / "summary.md").read_text().startswith("# Reference evaluation")
    assert code == 1  # the fake keeps the cup: update.later_wins fails; ainex is unreadable
    m = res["metrics"]
    assert set(m) == set(rt.Targets.load().metrics)
    assert m["update.later_wins"]["value"]["cup_struck"] == 0.0
    assert m["update.later_wins"]["value"]["ids_kept"] == 1.0
    assert m["update.later_wins"]["status"] == "fail"
    assert m["locate.recall"]["value"] == 1.0 and m["locate.recall"]["basis"]
    for mid in ("update.readable_latency", "fusion.time_per_frame", "reads.latency",
                "locate.latency_idle", "frame.shift", "grid.id_stability", "reads.atomic",
                "objects.id_stability", "objects.label_agreement", "objects.centre_error",
                "features.region_set", "grid.camera_cells_free", "jobs.eta_accuracy",
                "interactive.latency_under_load", "revert.latency"):
        assert m[mid]["status"] != "not_measured", (mid, m[mid])
    assert m["grid.id_stability"]["value"] == 1.0
    assert m["objects.label_agreement"]["value"] == 1.0
    assert m["locate.median_pos_error"]["status"] == "not_measured"
    assert "replay_eval" in m["locate.median_pos_error"]["reason"]
    assert m["fusion.intermediate_interval"]["reason"].startswith("needs O3")
    c = res["contracts"]
    assert c["contract.schema_valid"]["violations"] == 0
    assert c["contract.server_timing"]["violations"] == 0
    assert c["contract.job_survives_disconnect"]["violations"] == 0
    assert c["contract.birdseye_footprints"]["violations"] == 0
    assert c["contract.reads_atomic"]["violations"] == 0
    assert any(e.startswith("ainex-captures") for e in res["errors"])
    assert set(res["runs"]["office_sequence"]["splits"]) == {
        "single", "halves", "truth", "per_frame", "interleaved"}
    assert json.loads(base.read_text())["metrics"] == m
    # a second run compares with the stored baseline
    with Fake(config(eta=0.1)) as fake:
        ev.main(["--url", fake.url, "--out", str(out), "--examples", str(examples),
                 "--captures", "office_sequence", "--splits", "single,halves",
                 "--baseline", str(base), "--keep"])
    res = json.loads((out / "result.json").read_text())
    assert res["baseline"] == str(base)
    assert res["metrics"]["update.readable_latency"]["baseline"]["regressed"] in (True, False)
    assert "Baseline" in (out / "summary.md").read_text()


def test_reference_run_without_capabilities(examples: Path, tmp_path: Path) -> None:
    with Fake(config(capabilities=["O1", "O4"], eta=None)) as fake:
        with pytest.raises(SystemExit, match="unknown split bogus"):
            ev.main(["--url", fake.url, "--examples", str(examples), "--splits", "single,bogus"])
        ev.main(["--url", fake.url, "--out", str(tmp_path / "o"), "--examples", str(examples),
                 "--captures", "office_sequence", "--splits", "single"])
    res = json.loads((tmp_path / "o" / "result.json").read_text())
    m = res["metrics"]
    assert m["locate.recall"]["status"] == "not_measured"
    assert "no samples" in m["locate.recall"]["reason"]
    assert m["update.readable_latency"]["status"] != "not_measured"


def test_request_sets_recorded_and_replayed(tmp_path: Path) -> None:
    rec = tmp_path / "sets" / "e041-metric"
    cam = {"fx": 50.0, "fy": 50.0, "cx": 32.0, "cy": 24.0, "width": 64, "height": 48}
    body = {"cameras": {"cam": cam}, "frames": [
        {"name": n, "image": f"f{i}", "camera_id": "cam", "prior": {"trust": "none"}}
        for i, n in enumerate(NAMES[:2])]}
    parts = [(f"f{i}", n, jpeg(), "image/jpeg") for i, n in enumerate(NAMES[:2])]
    with Fake(config()) as fake:
        api = Api(fake.url, Validator(), Log(record_dir=rec))
        job, _ = api.call("POST", "/api/maps/m/updates", "POST updates", "Job", body=body,
                          parts=parts, ok=(202,))
        while job["state"] not in ev.TERMINAL:
            job, _ = api.call("GET", f"/api/jobs/{job['job_id']}", "GET job", "Job",
                              params={"wait_s": 1})
        api.call("POST", "/api/maps/m/locate", "POST locate", "Locate",
                 body={**body, "record": False}, parts=parts)
        lines = (rec / "requests.jsonl").read_text().splitlines()
        assert len(lines) >= 3 and (rec / "files").is_dir()
        with (rec / "requests.jsonl").open("a") as fh:
            fh.write(json.dumps({"method": "GET", "path": "/api/maps/gone", "route": "GET map",
                                 "schema": "MapSummary", "params": None, "body": None,
                                 "parts": None, "ok": [200]}) + "\n")
        out = tmp_path / "o"
        ev.main(["--url", fake.url, "--out", str(out), "--captures", "",
                 "--requests", str(tmp_path / "sets")])
        res = json.loads((out / "result.json").read_text())
        (r,) = res["requests"]
        assert r["set"] == "e041-metric" and r["requests"] == len(lines) + 1
        assert set(r["routes"]) == {"POST updates", "POST locate"}
        assert r["errors"] and "map_not_found" in r["errors"][0]
        assert res["metrics"]["update.readable_latency"]["n"] >= 1
        assert res["metrics"]["locate.latency_idle"]["n"] == 1
        ev.main(["--url", fake.url, "--out", str(out), "--captures", "", "--requests", str(rec)])
        assert json.loads((out / "result.json").read_text())["requests"][0]["requests"] == \
            len(lines) + 1


def test_local_service_on_a_free_port(tmp_path: Path) -> None:
    script = tmp_path / "server.sh"
    script.write_text(f"""#!/bin/sh
if [ "$1" = "--stop" ]; then echo stopped >> "{tmp_path}/stops"; exit 0; fi
exec "{sys.executable}" -c '
import http.server, json, sys
port = int(sys.argv[sys.argv.index("--port") + 1])
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.end_headers(); self.wfile.write(b"{{}}")
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
' "$@"
""")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    with ev.local_service(script, timeout_s=30) as url:
        assert url.startswith("http://127.0.0.1:")
        assert ":9091" not in url
    assert (tmp_path / "stops").read_text().strip() == "stopped"
    dead = tmp_path / "dead.sh"
    dead.write_text("#!/bin/sh\nexit 3\n")
    dead.chmod(dead.stat().st_mode | stat.S_IEXEC)
    with pytest.raises(SystemExit, match="exited with 3"), ev.local_service(dead, timeout_s=30):
        pass
    slow = tmp_path / "slow.sh"
    slow.write_text("#!/bin/sh\nexec sleep 30\n")
    slow.chmod(slow.stat().st_mode | stat.S_IEXEC)
    with pytest.raises(SystemExit, match="no service"), ev.local_service(slow, timeout_s=0.5):
        pass


def test_main_starts_a_local_service(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                     examples: Path) -> None:
    with Fake(config()) as fake:
        import contextlib

        @contextlib.contextmanager
        def svc(_: Path) -> Any:
            yield fake.url

        monkeypatch.setattr(ev, "local_service", svc)
        (tmp_path / "evals").mkdir()
        (tmp_path / "evals" / ev.BASELINE).write_text("[]")  # not a result: not compared
        assert ev.main(["--examples", str(examples), "--captures", "office_sequence",
                        "--splits", "single"]) in (0, 1)
    (run,) = [p for p in (tmp_path / "evals").iterdir() if p.is_dir()]
    res = json.loads((run / "result.json").read_text())
    assert any("not a result.json" in e for e in res["errors"])


def test_images_cameras_and_splits(examples: Path) -> None:
    assert ev.jpeg_info(jpeg(64, 48)) == (64, 48, None)
    assert ev.jpeg_info(exif_jpeg(28)) == (64, 48, 28.0)
    assert ev.jpeg_info(exif_jpeg(26, big_endian=True))[2] == 26.0
    _, png = cv2.imencode(".png", np.zeros((5, 7, 3), np.uint8))
    assert ev.jpeg_info(bytes(png)) == (7, 5, None)
    for bad in (b"GIF89a", b"\xff\xd8\x00\x00", b"\xff\xd8\xff\xd9\x00\x02"):
        with pytest.raises(ValueError):
            ev.jpeg_info(bad)
    assert ev.exif_focal35(b"xx") is None
    assert ev.exif_focal35(b"II" + struct.pack("<HI", 42, 8) + struct.pack("<H", 0)) is None
    assert ev.exif_focal35(b"II" + struct.pack("<HI", 42, 100)) is None
    caps = {c.key: c for c in ev.default_captures(examples)}
    office = caps["office_sequence"]
    assert office.absent == ["cup"] and office.splits_from_truth == [[3, 3]]
    assert [p.name for p in office.frames()] == NAMES
    cam = ev.camera_of(office, exif_jpeg(28), {})
    assert cam["fx"] == pytest.approx(28 / 36 * 64)
    assert ev.camera_of(caps["ainex-captures"], jpeg(640, 480), {})["fx"] == 270.0
    wide = ev.camera_of(caps["camera"], jpeg(1920, 1080), {})
    assert wide["fx"] == pytest.approx(960 / np.tan(np.radians(55)))
    assert ev.camera_of(office, b"", {"office_sequence": {"fx": 1}}) == {"fx": 1}
    assert ev.default_captures(examples / "none")[0].absent == []
    assert ev.split(5, "single") == [[0, 1, 2, 3, 4]]
    assert ev.split(5, "halves") == [[0, 1], [2, 3, 4]] and ev.split(1, "halves") == [[0]]
    assert ev.split(3, "per_frame") == [[0], [1], [2]]
    assert ev.split(5, "interleaved") == [[0, 3], [1, 4], [2]]
    assert ev.split(2, "interleaved") == [[0], [1]]
    assert ev.split(6, "truth", [4, 4]) == [[0, 1, 2, 3], [4, 5]]
    with pytest.raises(ValueError):
        ev.split(3, "bogus")


def test_measures() -> None:
    s = ev.Samples()
    a = {"frames": [{"name": "a", "T_map_cam": np.eye(4).tolist(), "pose_state": "fused"},
                    {"name": "p", "T_map_cam": np.eye(4).tolist(), "pose_state": "kept_at_prior"},
                    {"name": "gone", "T_map_cam": np.eye(4).tolist()}]}
    moved = np.eye(4)
    moved[0, 3] = 0.05
    b = {"frames": [{"name": "a", "T_map_cam": moved.tolist()},
                    {"name": "p", "T_map_cam": moved.tolist()}]}
    ev.frame_shift(a, b, {"translation_m": 0.01, "rotation_deg": 0.0}, s)
    assert s.values["frame.shift#pos_m"] == [pytest.approx(0.04)]
    obj = {"id": 1, "label": "cup", "centre_m": [0, 0, 0], "state": "confirmed", "region": None}
    ev.object_ids({"objects": [obj]}, {"objects": [{**obj, "id": 2}]}, s)
    ev.object_ids({"objects": [obj]}, {"objects": [{**obj, "centre_m": [5, 0, 0]}]}, s)
    assert s.values["objects.id_stability"] == [0.0]
    meta = {"width": 10, "height": 10, "pixel_to_map": [[1, 0, 0], [0, 1, 0]]}
    assert ev.birdseye_agrees(meta, {"objects": [obj]})
    assert not ev.birdseye_agrees(meta, {"objects": [{**obj, "centre_m": [50, 0, 0]}]})
    assert not ev.birdseye_agrees({**meta, "pixel_to_map": [[0, 0, 0], [0, 0, 0]]},
                                  {"objects": []})
    ev.later_wins([{"objects": [obj]}], ["cup"], s)  # one version: nothing to compare
    other = {**obj, "id": 3, "label": "book"}
    ev.later_wins([{"objects": [obj, other]}, {"objects": [{**other, "centre_m": [0.1, 0, 0]}]}],
                  ["cup"], s)
    assert s.values["update.later_wins#cup_struck"] == [1.0]
    assert s.values["update.later_wins#box_centre_delta_m"] == [pytest.approx(0.1)]
    few = {"frames": a["frames"][:2]}
    assert ev.align(few, few) is None
    line = {"frames": [{"name": str(i), "T_map_cam": np.eye(4).tolist()} for i in range(3)]}
    assert ev.align(line, line) is None  # degenerate: every centre at one point
    ev.compare_runs({"frames": few, "objects": None}, {"frames": few, "objects": None}, s)
    tri = {"frames": [{"name": str(i), "T_map_cam": t.tolist()} for i, t in
                      enumerate([pose(0), pose(1), np.eye(4)])]}
    ev.compare_runs({"frames": tri, "objects": {"objects": []}},
                    {"frames": tri, "objects": {"objects": [obj]}}, s)
    assert "objects.label_agreement" not in s.values
    grid = {"cell_m": 1.0, "regions": [{"polygon_m": [[0, 0], [2, 0], [2, 2]]}],
            "frontiers": [{"centre_m": [1, 1], "region": None}]}
    ev.region_set(grid, None, s)
    assert s.values["features.region_set"] == [0.0]
    assert ev.fmt(None) == "-" and ev.fmt({"a": 1.0}) == "a 1"


def test_targets_file_and_rules(tmp_path: Path) -> None:
    t = rt.Targets.load()
    appendix = {
        "locate.recall", "locate.median_pos_error", "locate.ambiguity_flagged",
        "locate.sigma_right_median", "locate.nearest_view", "locate.sigma_containment",
        "locate.sigma_tightness", "locate.sigma_tightness_bound", "update.sigma_tightness",
        "update.sigma_tightness_bound", "update.sigma_containment",
        "update.sigma_containment_kept", "scale.error_metric",
        "scale.error_camera_height", "scale.error_by_direction", "frame.shift", "update.readable_latency",
        "fusion.time_per_frame", "reads.atomic", "jobs.eta_accuracy",
        "fusion.intermediate_interval", "inputs.reason_agreement", "revert.latency",
        "gate.warping_rejected", "gate.others_accepted", "grid.floor_plan_agreement",
        "passages.door_recall", "passages.geometry", "grid.floor_observed",
        "grid.camera_cells_free", "grid.floor_false_obstacles", "grid.unconfirmed_occupancy",
        "grid.id_stability",
        "features.region_set", "grid.band_occupied", "grid.floor_height", "reads.latency",
        "objects.support_height", "objects.relations", "objects.centre_error",
        "objects.label_agreement", "objects.id_stability", "analysis.range_scan_error",
        "interactive.latency_under_load", "locate.latency_idle", "update.later_wins"}
    assert set(t.metrics) == appendix  # exactly the appendix ids, no gate parameters
    assert t.param("passages.geometry", "centre_tolerance_m", 0) == 0.10
    assert t.param("nothing", "x", 7.0) == 7.0
    for bad in ([], {"parts": {}}, {"parts": {"a": {"parts": {}}}}, {"op": "==", "value": 1},
                {"op": "<=", "value": "x"}, {"op": "between", "value": 1},
                {"op": "between", "value": [2, 1]}, {"op": "between", "value": [0, "x"]}):
        with pytest.raises(ValueError):
            rt.Targets({"m": bad})  # type: ignore[dict-item]
    assert rt.reduce([1, 2], "sum") == 3 and rt.reduce([1, 3], "median") == 2
    assert rt.meets(1, "<", 2) and rt.meets(3, ">", 2) and not rt.meets(3, "<=", 2)
    parts = {"parts": {"a": {"op": "<=", "value": 1}, "b": {"op": ">=", "value": 1}}}
    assert rt.judge(None, None) == "no_samples" and rt.judge(1.0, None) == "no_target"
    assert rt.judge(None, {"info": True}) == "no_samples"
    assert rt.judge({"a": 0.5, "b": None}, parts) == "no_samples"
    assert rt.judge(1.0, parts) == "no_samples"
    assert rt.judge({"a": 0.5}, {"op": "<=", "value": 1}) == "no_samples"
    tg = rt.Targets({"lat": {"op": "<=", "value": 2, "tolerance_abs": 0.5},
                     "share": {"op": ">=", "value": 0.9}, "c": parts},
                    {"tolerance_rel": 0.1})
    res = {"lat": {"value": 2.0}, "share": {"value": 0.5}, "c": {"value": {"a": 2.0, "b": 2.0}},
           "x": {"value": 1.0}, "mixed": {"value": {"a": 1}}}
    base = {"lat": {"value": 1.0}, "share": {"value": 0.9}, "c": {"value": {"a": 0.5, "b": 2.0}},
            "mixed": {"value": 1.0}}
    cmp = rt.compare(res, base, tg)
    assert cmp["lat"]["regressed"] and cmp["share"]["regressed"] and cmp["c"]["regressed"]
    assert not cmp["mixed"]["regressed"] and "x" not in cmp
    assert not rt.worse("lat", float("nan"), 1.0, tg.get("lat"))
    band = {"op": "between", "value": [0.3, 1.5]}
    assert rt.meets(0.3, "between", [0.3, 1.5]) and not rt.meets(1.6, "between", [0.3, 1.5])
    assert rt.judge(1.0, band) == "pass" and rt.judge(0.2, band) == "fail"
    assert rt.worse("b", 2.0, 1.6, band) and not rt.worse("b", 1.0, 1.6, band)
    assert rt.worse("b", 0.1, 0.2, band)
