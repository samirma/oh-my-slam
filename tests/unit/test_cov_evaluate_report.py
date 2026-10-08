"""The evaluator's single command and its report, offline (spec §5: one command, a machine-readable
result and a human-readable summary stored outside the repository, targets and the baseline as
data): a whole run with a stand-in evaluation (judged, written, compared with and stored as the
baseline), the default result folder, the module entry point, the machine description, and every
section of the summary."""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from oh_my_slam.tools.evaluate import __main__ as cli
from oh_my_slam.tools.evaluate import report
from oh_my_slam.tools.evaluate.metrics import Metrics
from oh_my_slam.tools.evaluate.report import environment, summary_md, write_report
from oh_my_slam.tools.evaluate.runner import Runner
from oh_my_slam.tools.evaluate.viewer import BrowserProbe
from tests.unit.test_evaluate_report import judged, result_of, targets_file

# -- the command -----------------------------------------------------------------------------------


class StandIn:
    """The evaluation with its runs replaced: ``run_all`` records the next value of
    ``perf.a.wall_s``."""

    made: ClassVar[list[StandIn]] = []
    values: ClassVar[list[float]] = []
    extra: ClassVar[dict[str, float]] = {}  # more metrics, recorded by every run

    def __init__(self, out: Path, runner: Runner, probe: BrowserProbe, street2: Path) -> None:
        self.out, self.runner, self.probe, self.street2 = out, runner, probe, street2
        self.metrics, self.details = Metrics(), {"runs": "none"}
        StandIn.made.append(self)

    def run_all(self) -> None:
        self.metrics.add("perf.a.wall_s", StandIn.values.pop(0))
        for k, v in StandIn.extra.items():
            self.metrics.add(k, v)


@pytest.fixture
def stand_in(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> type[StandIn]:
    StandIn.made, StandIn.values, StandIn.extra = [], [], {}
    monkeypatch.setattr(cli, "Evaluation", StandIn)
    monkeypatch.setattr(cli, "environment", lambda repo: {"commit": "abc", "dirty": False})
    monkeypatch.setattr(cli, "DATA", tmp_path / "data")
    return StandIn


def test_a_run_is_judged_written_and_can_become_the_baseline(
        tmp_path: Path, stand_in: type[StandIn], capsys: pytest.CaptureFixture[str]) -> None:
    stand_in.values = [2.5, 3.5]
    baseline = tmp_path / "baseline.json"
    args = ["--targets", str(targets_file(tmp_path)), "--baseline", str(baseline),
            "--street2", str(tmp_path / "street2.mp4")]
    first = tmp_path / "run1"
    assert cli.main(["--out", str(first), *args, "--set-baseline"]) == 0
    out, err = capsys.readouterr()
    assert out == ""  # spec §4: the path of summary.md goes to stderr, stdout stays empty
    assert f"evaluate: results in {first}" in err
    assert ("1 passed, 0 failed, 0 without a target, baseline not compared — "
            f"{first / 'summary.md'}") in err
    assert f"stored {first / 'result.json'} as the baseline {baseline}" in err
    result = json.loads((first / "result.json").read_text())
    assert result["baseline"]["status"] == "missing" and result["details"] == {"runs": "none"}
    assert result["environment"] == {"commit": "abc", "dirty": False}
    assert baseline.read_bytes() == (first / "result.json").read_bytes()
    ev = stand_in.made[0]
    assert ev.out == first and ev.runner.out == first and ev.street2 == tmp_path / "street2.mp4"
    assert ev.probe.launch is not None  # a browser is asked for
    # the next run fails its target and regresses against the stored baseline, which stays
    # (with a metric the baseline has no value for: reported as not compared)
    second = tmp_path / "run2"
    stand_in.extra = {"pose.b.fraction": 0.95}
    assert cli.main(["--out", str(second), *args]) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert ("1 passed, 1 failed, 0 without a target, 1 regressions (1 metrics without a baseline "
            f"value) — {second / 'summary.md'}") in err
    result = json.loads((second / "result.json").read_text())
    assert result["baseline"]["status"] == "compared"
    assert result["baseline"]["commit"] == "abc"
    assert result["baseline"]["not_compared"] == ["pose.b.fraction"]
    metric, new = result["metrics"]
    assert metric["passed"] is False and metric["regression"] and metric["baseline"] == 2.5
    assert new["baseline"] is None and result["summary"]["without_baseline"] == 1
    text = (second / "summary.md").read_text()
    assert "**Result: FAIL**" in text and "## Not compared with the baseline" in text
    assert baseline.read_bytes() == (first / "result.json").read_bytes()


def test_results_go_to_a_utc_folder_of_the_data_folder_by_default(
        tmp_path: Path, stand_in: type[StandIn], capsys: pytest.CaptureFixture[str]) -> None:
    stand_in.values = [1.0]
    assert cli.main(["--targets", str(targets_file(tmp_path))]) == 0
    (folder,) = (tmp_path / "data").iterdir()
    assert re.fullmatch(r"\d{8}T\d{6}Z", folder.name)
    assert (folder / "result.json").is_file() and (folder / "summary.md").is_file()
    result = json.loads((folder / "result.json").read_text())
    assert result["baseline"] == {"path": str(tmp_path / "data" / "baseline.json"),
                                  "status": "missing"}
    capsys.readouterr()


def test_the_module_runs_as_a_command(tmp_path: Path) -> None:
    bad = targets_file(tmp_path, {"metrics": {"x": {"op": "<", "value": 1}}})
    res = subprocess.run([sys.executable, "-m", "oh_my_slam.tools.evaluate", "--targets",
                          str(bad)], capture_output=True, text=True, timeout=120)
    assert res.returncode == 2 and res.stdout == ""
    assert "needs op in ('<=', '>=') and a numeric value" in res.stderr


# -- the machine ------------------------------------------------------------------------------------


def test_the_environment_names_the_commit_and_the_machine(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    processor = platform.processor()
    real_run = subprocess.run
    answers: dict[str, tuple[int, str]] = {}
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kw: Any) -> Any:
        key = argv[3] if argv[0] == "git" else argv[0]
        if key not in answers:
            return real_run(argv, **kw)
        calls.append(argv)
        code, out = answers[key]
        return subprocess.CompletedProcess(argv, code, out, "")

    monkeypatch.setattr(report.subprocess, "run", fake_run)
    answers.update({"rev-parse": (0, "abc123\n"), "status": (0, " M src/x.py\n"),
                    "sysctl": (0, "Apple M4 Max\n")})
    env = environment(tmp_path)
    assert (env["commit"], env["dirty"], env["cpu"]) == ("abc123", True, "Apple M4 Max")
    assert ["git", "-C", str(tmp_path), "rev-parse", "HEAD"] in calls
    assert env["memory_gb"] > 0 and env["python"] == platform.python_version()
    answers.update({"status": (0, "")})
    assert environment(tmp_path)["dirty"] is False
    # not a git checkout, and no CPU name from sysctl
    answers.update({"rev-parse": (128, ""), "status": (128, ""), "sysctl": (1, "")})
    env = environment(tmp_path)
    assert (env["commit"], env["dirty"], env["cpu"]) == (None, None, processor)


# -- the report -------------------------------------------------------------------------------------


def test_values_json_cannot_hold_are_written_readably(tmp_path: Path) -> None:
    result = result_of(Metrics(), [], {"status": "missing"},
                       {"numpy": np.float32(1.5), "ids": {3}, "frozen": frozenset({4}),
                        "where": Path("/x/y")})
    res, _ = write_report(tmp_path, result)
    details = json.loads(res.read_text())["details"]
    assert details == {"numpy": 1.5, "ids": [3], "frozen": [4], "where": "/x/y"}


def test_a_failed_metric_that_also_regressed_says_both(tmp_path: Path) -> None:
    m = judged(tmp_path, {"perf.a.wall_s": 3.5}, {"perf.a.wall_s": 2.0})
    text = summary_md(result_of(m, [], {"path": "b.json", "status": "compared"}))
    assert ("| perf.a.wall_s | 3.5 | <= 3 s | 3.5 s exceeds the limit 3 s; worse than the "
            "baseline 2 s by 1.5 (tolerance 0.2) |") in text


def row(mid: str, value: float | None, passed: bool | None, target: dict[str, Any] | None,
        baseline: float | None = None, regression: bool = False) -> dict[str, Any]:
    return {"id": mid, "value": value, "passed": passed, "error": None, "target": target,
            "baseline": baseline, "regression": regression, "detail": None}


def test_a_result_with_incomplete_rows_still_renders(tmp_path: Path) -> None:
    """``summary.md`` is rendered from ``result.json`` alone; rows written by hand or by another
    version of the evaluator (no target, no baseline value, no tolerance) render without a reason
    rather than fail."""
    result = result_of(Metrics(), [], {"path": "b.json", "status": "compared"})
    limit = {"op": "<=", "value": 3.0, "unit": "s"}
    result["metrics"] = [row("seg.untargeted", 1.0, False, None),
                         row("seg.no_tolerance", 2.0, True, limit, 1.0, True),
                         row("seg.no_baseline", 2.0, True, limit, None, True)]
    text = summary_md(result)
    assert "| seg.untargeted | 1 | — |  |" in text
    assert "| seg.no_tolerance | 2 | 1 | <= 3 s | worse than the baseline 1 s by 1 |" in text
    assert "| seg.no_baseline | 2 | — | <= 3 s |  |" in text


def test_every_detail_section_of_the_summary(tmp_path: Path) -> None:
    details: dict[str, Any] = {
        "poses.single": [{"capture": "005_bootstrap_left015_up.jpg", "commanded_yaw_deg": 15.0,
                          "yaw_deg": 14.2, "yaw_err_deg": -0.8, "pitch_delta_deg": 9.5,
                          "pitch_ok": True}],
        "map.stability": [{"single_id": 1, "split_id": 7, "single_label": "sofa",
                           "split_label": "couch", "iou": 0.81, "centre_delta_m": 0.05,
                           "extent_delta_rel": 0.1}],
        "map.split.pairs": [{"pair": "001.jpg~053.jpg", "gap": 52, "angle_deg": 3.1,
                             "median_pct": 1.2, "p90_pct": 4.5}],
        "poses.locate": [{"capture": "011_left_060_level.jpg", "commanded_yaw_deg": 60.0,
                          "yaw_deg": 58.0, "yaw_err_deg": 2.0, "vs_mapped_rot_deg": 1.0,
                          "vs_mapped_m": 0.1}],
        "poses.camera_locate": [{"capture": "img_016_p06_up.jpg", "located": True,
                                 "yaw_deg": 50.0, "yaw_err_deg": 2.0, "vs_mapped_rot_deg": 2.1,
                                 "vs_mapped_m": 0.02}],
        "poses.camera_split": [{"capture": "img_010_p04_up.jpg", "pan": 4, "tilt": "up",
                                "registered": True, "yaw_deg": 21.4, "pitch_delta_deg": 12.0,
                                "pitch_ok": True},
                               {"capture": "img_011_p04_mid.jpg", "pan": 4, "tilt": "mid",
                                "registered": False}],
        "segmentation.frames": [{"image": "001_bootstrap_level.jpg", "objects": 2,
                                 "labels": {"sofa": 2}, "min_score": 0.6, "median_score": 0.7}],
        "segmentation.camera_frames": [{"image": "img_007_p03_down.jpg", "error": "boom"}],
        "map.camera_stability": [{"single_id": 4, "split_id": 4, "single_label": "desk",
                                  "split_label": "desk", "iou": 0.62, "centre_delta_m": 0.04,
                                  "extent_delta_rel": 0.12}],
        "map.camera_single.pairs": [{"pair": "img_007_p03_down.jpg~img_008_p03_mid.jpg",
                                     "gap": 1, "angle_deg": 14.8, "median_pct": 2.2,
                                     "p90_pct": 6.1}],
        "map_update": {"splits": {"split_4_4_5": {"ids_broken": [
            {"id": 3, "label": "monitor", "published_by_update": 1, "update": 3,
             "now": None}]}}},
        "errors": ["server.sh: RuntimeError: boom"],
    }
    text = summary_md(result_of(Metrics(), [], {"status": "missing"}, details))
    assert "| 005_bootstrap_left015_up.jpg | 15 | 14.2 | -0.8 | 9.5 | yes |" in text
    assert "## Objects: one update vs split" in text
    assert "| 1 | 7 | sofa / couch | 0.81 | 0.05 | 0.1 |" in text
    assert "## Least consistent overlapping keyframe pairs — split map" in text
    assert "| 001.jpg~053.jpg | 52 | 3.1 | 1.2 | 4.5 |" in text
    assert "## Poses — held-out captures located by mapper.sh locate" in text
    assert "| 011_left_060_level.jpg | 60 | 58 | 2 | 1 | 0.1 |" in text
    assert "## Poses — held-out captures located by mapper.sh locate — camera" in text
    assert "| img_016_p06_up.jpg | — | 50 | 2 | 2.1 | 0.02 |" in text
    assert "## Map update: split_4_4_5, published ids that did not persist" in text
    assert "| 3 | monitor | 1 | 3 | — |" in text
    assert "split_4_4_5, objects that never changed" not in text  # no stability rows
    assert "## Evaluator errors\n\n* server.sh: RuntimeError: boom" in text
    # examples/camera: its own pose table (no commanded yaw), frames, objects and pairs
    assert "## Poses — camera_split map\n\nYaw relative to the first registered capture" in text
    assert "| img_010_p04_up.jpg | 4 | up | 21.4 | 12 | yes |" in text
    assert "| img_011_p04_mid.jpg | 4 | mid | — | — | — |" in text
    assert "## Poses — camera_single map" not in text
    assert "| 001_bootstrap_level.jpg | 2 | sofa 2 | 0.6 | 0.7 |" in text
    assert "| img_007_p03_down.jpg | — | boom | — | — |" in text
    assert "## Objects: one update vs split — camera" in text
    assert "| 4 | 4 | desk / desk | 0.62 | 0.04 | 0.12 |" in text
    assert "## Least consistent overlapping keyframe pairs — camera_single map" in text
    assert "| img_007_p03_down.jpg~img_008_p03_mid.jpg | 1 | 14.8 | 2.2 | 6.1 |" in text
