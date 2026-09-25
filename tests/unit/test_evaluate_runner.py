"""The evaluator's command runner and plan with fake entry points: records, stderr tails and
timings; failures and timeouts become failed metrics, never a crash; view.sh URL and API handling
(the rendering itself needs a browser: ``-m browser``)."""

from __future__ import annotations

import json
import os
import stat
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from oh_my_slam.core.ply import PointCloud
from oh_my_slam.tools.evaluate.memory import stage_peaks
from oh_my_slam.tools.evaluate.metrics import load_targets
from oh_my_slam.tools.evaluate.names import captures_in
from oh_my_slam.tools.evaluate.report import build_result, write_report
from oh_my_slam.tools.evaluate.runner import Runner, RunSpec
from oh_my_slam.tools.evaluate.suite import EXAMPLES, Evaluation, expected_ids
from oh_my_slam.tools.evaluate.viewer import BrowserProbe, ViewOutcome, served_url
from oh_my_slam.viewer.bundle import DisplayCloud
from oh_my_slam.viewer.server import cloud_payload
from tests.unit.test_evaluate_contracts import labelled_cloud, scene_bytes

ENTRY_POINTS = ("start_inference_server.sh", "reconstruct.sh", "mapper.sh", "segment.sh",
                "view.sh")


def script(folder: Path, name: str, body: str, shebang: str = "#!/bin/bash") -> Path:
    p = folder / name
    p.write_text(f"{shebang}\n{body}\n")
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p


def test_a_successful_run_keeps_stdout_stderr_and_timings(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    script(repo, "reconstruct.sh", """echo '{"openlabel": {}}'
echo "progress 50 %" >&2
printf '{"stages_s": {"inference": 0.5}, "peak_rss_mb": {"self": 123.0}}' > "$OH_MY_SLAM_TIMINGS"
""")
    runner = Runner(tmp_path / "out", repo)
    rec = runner.run(RunSpec("recon", "reconstruct_json", "reconstruct.sh", ("-i", "x.jpg")))
    assert rec.ok and rec.exit_code == 0 and rec.error is None
    assert rec.stdout_bytes() == b'{"openlabel": {}}\n'
    assert rec.stderr_tail == "progress 50 %"
    assert rec.timings == {"stages_s": {"inference": 0.5}, "peak_rss_mb": {"self": 123.0}}
    assert rec.client_peak_mb >= 123.0 and rec.wall_s > 0
    assert rec.argv == [str(repo / "reconstruct.sh"), "-i", "x.jpg"]
    assert runner.records == [rec]
    with pytest.raises(ValueError, match="duplicate"):
        runner.run(RunSpec("recon", "g", "reconstruct.sh"))


STAGED = """import logging, time
import numpy as np
from oh_my_slam.core import timing
with timing.collect() as tm:
    with timing.stage("load"):
        time.sleep(0.5)
    with timing.stage("inference"):
        a = np.ones(40_000_000)  # 320 MB, touched
        time.sleep(0.8)
        del a
    with timing.stage("write"):
        print('{"openlabel": {}}')
timing.report(tm, logging.getLogger("fake"))
"""


def test_per_stage_peak_memory_of_a_run(tmp_path: Path) -> None:
    """A command instrumented with core.timing: the runner attributes its process-tree samples
    to the stages through the recorded windows, and keeps the command's own stage peaks."""
    repo = tmp_path / "repo"
    repo.mkdir()
    script(repo, "reconstruct.sh", STAGED, shebang=f"#!{sys.executable}")
    rec = Runner(tmp_path / "out", repo).run(RunSpec("r", "reconstruct_json", "reconstruct.sh"))
    assert rec.ok and rec.stages is not None
    assert list(rec.stages) == ["load", "inference", "write"]  # in the order they ran
    load, inference = rec.stages["load"], rec.stages["inference"]
    assert inference["s"] == pytest.approx(0.8, abs=0.3)
    assert load["client_peak_mb"] and inference["client_peak_mb"]
    assert inference["client_peak_mb"] >= load["client_peak_mb"] + 250
    assert inference["client_peak_mb"] <= rec.client_peak_mb + 1
    assert inference["server_peak_gb"] is None  # no server in the offline suite
    assert rec.to_dict()["stages"] == rec.stages


def test_stage_peaks_attribute_samples_by_window() -> None:
    timings = {"stages_s": {"b": 0.5, "a": 1.0, "c": 0.01, "d": 0.2},
               "stages_peak_rss_mb": {"a": 300.0, "b": 120.0, "c": 50.0},
               "t0_unix": 1000.0,
               "stage_windows": [["a", 0.0, 1.0], ["b", 1.0, 1.5], ["c", 1.52, 1.53]]}
    samples = [(1000.1, 250.0, 11.0), (1000.5, 400.0, 12.0), (1001.2, 200.0, 11.5),
               (1001.6, 150.0, 13.0)]
    got = stage_peaks(timings, samples)
    assert got is not None and list(got) == ["a", "b", "c", "d"]  # by start; unwindowed last
    assert got["a"] == {"s": 1.0, "client_peak_mb": 400.0, "server_peak_gb": 12.0}
    assert got["b"] == {"s": 0.5, "client_peak_mb": 200.0, "server_peak_gb": 11.5}
    # shorter than the sampling period: the samples around it
    assert got["c"] == {"s": 0.01, "client_peak_mb": 200.0, "server_peak_gb": 13.0}
    assert got["d"] == {"s": 0.2, "client_peak_mb": None, "server_peak_gb": None}
    assert stage_peaks(None, samples) is None and stage_peaks({"stages_s": {}}, []) is None


def test_failures_become_records(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    script(repo, "segment.sh", 'echo "loading" >&2\necho "segment.sh: error: server down" >&2\n'
                               "exit 3")
    (repo / "mapper.sh").write_text("not executable")
    script(repo, "view.sh", "sleep 30")
    runner = Runner(tmp_path / "out", repo)
    failed = runner.run(RunSpec("seg", "g", "segment.sh"))
    assert not failed.ok and failed.exit_code == 3
    assert failed.failure() == "seg failed (exit 3): segment.sh: error: server down"
    assert failed.to_dict()["stderr_tail"].startswith("loading")
    missing = runner.run(RunSpec("map", "g", "mapper.sh"))
    assert not missing.ok and missing.exit_code is None
    assert (missing.error or "").startswith("could not start")
    t0 = time.monotonic()
    slow = runner.run(RunSpec("view", "g", "view.sh", timeout_s=0.5))
    assert time.monotonic() - t0 < 10
    assert not slow.ok and slow.error == "timed out after 0 s"
    assert runner.run(RunSpec("status", "g", "segment.sh", ok_exit=(0, 3))).ok


def fake_repo(tmp_path: Path, **bodies: str) -> Path:
    """The five entry points; each fails with "boom" unless a body is given."""
    repo = tmp_path / "repo"
    repo.mkdir()
    for name in ENTRY_POINTS:
        key = name.removesuffix(".sh")
        script(repo, name, bodies.get(key, f'echo "{name}: error: boom" >&2\nexit 1'))
    return repo


def test_payload_contracts_of_single_runs(tmp_path: Path) -> None:
    payload = tmp_path / "scene.json"
    payload.write_bytes(scene_bytes())
    repo = fake_repo(tmp_path, reconstruct=f"""out=""
while [ $# -gt 0 ]; do [ "$1" = "-o" ] && out="$2"; shift; done
[ -n "$out" ] && cp {payload} "$out" && exit 0
[ -n "$BANNER" ] && echo "model loaded"
cat {payload}""")
    ev = Evaluation(tmp_path / "out", Runner(tmp_path / "out", repo), BrowserProbe(None))
    doc = ev.scene(ev.run("json", "g", "reconstruct.sh", "-i", "x.jpg"))
    assert doc is not None and ev.contracts.checks[("colour", "reconstruct")]["json"] == []
    target = tmp_path / "out" / "r.json"
    rec = ev.run("file", "g", "reconstruct.sh", "-o", target, stdout="empty", output=target)
    assert ev.scene(rec) is not None
    ev.runner.env["BANNER"] = "1"
    ev.run("banner", "g", "reconstruct.sh")
    ev.run("failed", "g", "segment.sh")
    checks = ev.contracts.checks[("stdout", "reconstruct")]
    assert checks["json"] == [] and checks["file"] == []
    assert checks["banner"] and "does not start with a JSON object" in checks["banner"][0]
    assert ev.contracts.checks[("openlabel", "reconstruct")] == {"json": [], "file": []}
    assert ev.contracts.checks[("stdout", "segment")]["failed"] == []  # failed, but silent


def test_frames_are_segmented_one_by_one(tmp_path: Path) -> None:
    payload = tmp_path / "scene.json"
    payload.write_bytes(scene_bytes())
    repo = fake_repo(tmp_path, segment=f'case "$2" in *002_*) echo boom >&2; exit 1;; esac\n'
                                       f"cat {payload}")
    ev = Evaluation(tmp_path / "out", Runner(tmp_path / "out", repo), BrowserProbe(None))
    ev.frames(captures_in(EXAMPLES / "ainex-captures")[:3])
    m = ev.metrics.items["seg.frames.with_detections_fraction"]
    assert m.value == 1.0 and m.detail == {"segmented": 2, "frames": 3}
    assert sorted(ev.images) == ["ainex-captures/001_bootstrap_level.jpg",
                                 "ainex-captures/003_bootstrap_side2_level.jpg"]
    rows = ev.details["segmentation.frames"]
    assert rows[0]["objects"] == 4 and "boom" in rows[1]["error"]


def test_every_command_failing_yields_failed_metrics_not_a_crash(tmp_path: Path) -> None:
    repo = fake_repo(tmp_path, start_inference_server='[ "$1" = "--status" ] && exit 3\n'
                                                      'echo "server: error: boom" >&2\nexit 1')
    out = tmp_path / "out"
    ev = Evaluation(out, Runner(out, repo), BrowserProbe(None), examples=EXAMPLES)
    ev.run_all()
    targets = load_targets(EXAMPLES / "targets.json")
    ev.metrics.judge(targets, None)
    assert "errors" not in ev.details  # every step handled the failures
    assert set(expected_ids()) <= set(ev.metrics.items)
    assert not any(k.startswith("gt.") for k in ev.metrics.items)
    for mid in expected_ids():
        m = ev.metrics.items[mid]
        if mid.startswith("contract.stdout."):  # failing commands kept stdout empty
            assert m.passed and m.detail["checked"] > 0, mid
            continue
        assert m.passed is False, mid
        assert m.error or mid == "contract.exit_codes", mid
    assert "boom" in (ev.metrics.items["perf.reconstruct_json.wall_s"].error or "")
    assert "boom" in (ev.metrics.items["perf.server.cold_start_s"].error or "")
    tags = [r.tag for r in ev.runner.records]
    assert tags[:3] == ["server_status_initial", "server_stop", "server_cold_start"]
    assert tags[-1] == "server_stop_final"  # the server was down at the start
    assert sum(t.startswith("segment_frame_") for t in tags) == 79
    assert sum(t.startswith("mapper_split_") for t in tags) == 3
    failed = [r for r in ev.runner.records if not r.ok]
    assert ev.metrics.items["contract.exit_codes"].value == len(failed) == len(tags) - 1
    result = build_result(ev.metrics, ev.runner.records, ev.details, started="s", finished="f",
                          duration_s=1.0, env={}, targets=Path("t"), baseline={"status": "missing"})
    _, md = write_report(out, result)
    assert "**Result: FAIL**" in md.read_text()


VIEW = """import http.server, os, sys
SCENE = open(os.environ["FAKE_SCENE"], "rb").read()
CLOUD = open(os.environ["FAKE_CLOUD"], "rb").read()
PAGE = os.environ.get("FAKE_PAGE", "").encode()
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = {"/api/scene": SCENE, "/api/cloud?color=segment": CLOUD, "/": PAGE,
                "/favicon.ico": b""}.get(self.path)
        self.send_response(200 if body is not None else 404)
        self.end_headers()
        self.wfile.write(body or b"")
    def log_message(self, *args):
        pass
print("[oh-my-slam] reconstructing x.jpg (see https://example.org/docs)", file=sys.stderr,
      flush=True)
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
print(f"view.sh: listening on http://127.0.0.1:{srv.server_address[1]}/", file=sys.stderr,
      flush=True)
try:
    srv.serve_forever(0.05)
except KeyboardInterrupt:
    pass
"""
RENDERS = 'setTimeout(() => { document.body.dataset.rendered = "true"; }, 200);'
FAILS = 'setTimeout(() => { document.body.dataset.error = "cloud: 500"; }, 100);'


def view_runner(tmp_path: Path, page_script: str = RENDERS, cloud: PointCloud | None = None
                ) -> Runner:
    repo = tmp_path / "repo"
    repo.mkdir()
    script(repo, "view.sh", VIEW, shebang=f"#!{sys.executable}")
    (tmp_path / "scene.json").write_bytes(scene_bytes())
    cloud = labelled_cloud() if cloud is None else cloud
    (tmp_path / "cloud.bin").write_bytes(
        cloud_payload(DisplayCloud(cloud, len(cloud), 1, 0.0), "color=segment"))
    env = {**os.environ, "FAKE_SCENE": str(tmp_path / "scene.json"),
           "FAKE_CLOUD": str(tmp_path / "cloud.bin"),
           "FAKE_PAGE": f"<!doctype html><html><body><script>{page_script}</script></body></html>"}
    return Runner(tmp_path / "out", repo, env)


def test_view_url_and_scene_without_a_browser(tmp_path: Path) -> None:
    runner = view_runner(tmp_path)
    ev = Evaluation(tmp_path / "out", runner, BrowserProbe(None))
    other = json.loads(scene_bytes())
    other["openlabel"]["objects"]["7"]["type"] = "table"
    ev.view("view_image", ("image", "segment.sh -i", other), "-i", "x.jpg")
    (rec,) = runner.records
    assert rec.ok and rec.exit_code == 0  # stopped with Ctrl-C after the measurement
    assert rec.argv[1:] == ["-i", "x.jpg", "--no-browser"]
    assert rec.notes["url"].startswith("http://127.0.0.1:") and rec.notes["render_s"] is None
    assert rec.notes["render_error"] == "browser: no browser available"
    assert ev.contracts.checks[("stdout", "view")] == {"view_image": []}
    assert ev.contracts.checks[("openlabel", "view")] == {"view_image": []}
    # the scene's OBB colours and the served color=segment cloud keep the colour contract
    assert ev.contracts.checks[("colour", "view")] == {"view_image": [],
                                                       "view_image/cloud color=segment": []}
    same = ev.contracts.checks[("same_objects", "image")]["view.sh vs segment.sh -i"]
    assert same and same[0].startswith("1 objects differ (ids [7])")
    assert ("console_errors", "view") not in ev.contracts.checks


def test_view_cloud_breaking_the_colour_contract(tmp_path: Path) -> None:
    cloud = labelled_cloud()
    assert cloud.rgb is not None and cloud.label is not None
    cloud.rgb[np.flatnonzero(cloud.label == 7)[:3]] = (10, 200, 10)  # blended / foreign colour
    cloud.label[:2] = 99  # an object the scene does not have
    ev = Evaluation(tmp_path / "out", view_runner(tmp_path, cloud=cloud), BrowserProbe(None))
    ev.view("view_map", ("map", "mapper.sh -t full", None), "-m", "m")
    problems = ev.contracts.checks[("colour", "view")]["view_map/cloud color=segment"]
    assert any("not in their object's colour" in p for p in problems)
    assert any("labels not in the scene: [99]" in p for p in problems)


def probe_fake_view(tmp_path: Path, page_script: str) -> ViewOutcome:
    pytest.importorskip("playwright.sync_api")
    runner = view_runner(tmp_path, page_script)
    try:
        seen = BrowserProbe().measure(runner, RunSpec("v", "view_image", "view.sh",
                                                      ("--no-browser",), "empty",
                                                      ok_exit=(0, 130), timeout_s=60))
    except RuntimeError as exc:
        pytest.skip(str(exc))
    if seen.error and seen.error.startswith("browser:"):
        pytest.skip(seen.error)
    return seen


@pytest.mark.browser
def test_view_render_time_in_a_browser(tmp_path: Path) -> None:
    seen = probe_fake_view(tmp_path, RENDERS)
    assert seen.error is None and seen.render_s is not None and 0.2 < seen.render_s < 60
    assert seen.url and seen.url.startswith("http://127.0.0.1:")  # not the docs URL before it
    assert seen.scene == scene_bytes()
    assert seen.cloud == (tmp_path / "cloud.bin").read_bytes()  # fetched after rendering
    assert seen.console_errors == [] and seen.record.ok


@pytest.mark.browser
def test_view_load_failure_in_a_browser(tmp_path: Path) -> None:
    seen = probe_fake_view(tmp_path, FAILS)
    assert seen.render_s is None and seen.error == "the page failed to load: cloud: 500"
    assert seen.console_errors == []  # the page was opened, so its console was watched


@pytest.mark.parametrize(("stderr", "url"), [
    ("[oh-my-slam] see https://example.org\nview.sh: listening on http://127.0.0.1:53211/",
     "http://127.0.0.1:53211/"),  # the contract's line wins over earlier URLs
    ("view.sh: serving map 'm' at http://127.0.0.1:53211/ (Ctrl-C to stop)",
     "http://127.0.0.1:53211/"),
    ("[oh-my-slam] loading…\nOpen http://localhost:8123 in a browser.", "http://localhost:8123"),
    ("URL: http://127.0.0.1:5000/?token=abc, press Ctrl-C", "http://127.0.0.1:5000/?token=abc"),
    ("serving at <http://[::1]:4000/>", "http://[::1]:4000/"),
    ("docs: https://example.org:443/view and http://example.org:80/", None),  # not local
    ("still loading", None),
])
def test_served_url_is_parsed_tolerantly(stderr: str, url: str | None) -> None:
    assert served_url(stderr) == url
