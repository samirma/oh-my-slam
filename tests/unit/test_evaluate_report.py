"""Evaluator data files: targets (pass/fail), the baseline (regressions), ground-truth discovery,
and the written result.json / summary.md."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from oh_my_slam.schema import openlabel as ol
from oh_my_slam.tools.evaluate import groundtruth as gt
from oh_my_slam.tools.evaluate.__main__ import load_baseline, main
from oh_my_slam.tools.evaluate.metrics import (
    Metrics,
    TargetsError,
    baseline_values,
    load_targets,
    target_for,
)
from oh_my_slam.tools.evaluate.performance import perf_metrics
from oh_my_slam.tools.evaluate.report import build_result, summary_md, write_report
from oh_my_slam.tools.evaluate.runner import RunRecord, RunSpec
from oh_my_slam.tools.evaluate.scene import DocObject
from oh_my_slam.tools.evaluate.suite import EXAMPLES, expected_ids
from tests.unit.test_evaluate_metrics import cam

TARGETS = {
    "defaults": {"tolerance_rel": 0.1},
    "metrics": {
        "perf.a.wall_s": {"op": "<=", "value": 3.0, "unit": "s"},
        "pose.b.fraction": {"op": ">=", "value": 0.9, "tolerance_abs": 0.05},
        "contract.c.x": {"op": "<=", "value": 0},
    },
}


def targets_file(tmp_path: Path, doc: dict | None = None) -> Path:
    p = tmp_path / "targets.json"
    p.write_text(json.dumps(doc or TARGETS))
    return p


def test_shipped_targets_cover_every_metric() -> None:
    targets = load_targets(EXAMPLES / "targets.json")
    gt_ids = {"gt.objects.recall", "gt.objects.precision", "gt.objects.obb_iou_median",
              "gt.poses.yaw_err_median_deg", "gt.poses.yaw_err_max_deg",
              "gt.poses.pitch_err_median_deg"}
    assert gt_ids <= set(targets)
    # no stale targets: besides those, only per-stage targets (explicit, or patterns)
    # (a split of the office sequence is named after its sizes: its targets are patterns too)
    named = {k for k in targets if "*" not in k and ".stage." not in k}
    patterns = {k for k in targets if "*" in k}
    expected = set(expected_ids())
    assert named == {k for k in expected if target_for(targets, k) is targets.get(k)} | gt_ids
    assert all(target_for(targets, k) is not None for k in expected)
    assert all(k.startswith(("perf.", "map_update.split_*.")) for k in patterns)
    # every performance group's stages have a target, whatever stage a command adds
    for mid in expected_ids():
        parts = mid.split(".")
        if parts[0] == "perf" and parts[1] != "server":
            for key in ("s", "client_peak_mb", "server_peak_gb"):
                assert target_for(targets, f"perf.{parts[1]}.stage.new_stage.{key}") is not None


def test_pattern_targets_cover_metrics_named_after_data(tmp_path: Path) -> None:
    """A key with ``*`` targets every metric it matches; a metric's own target, then the most
    specific pattern, win."""
    doc = {"metrics": {
        "perf.*.stage.*.s": {"op": "<=", "value": 60, "unit": "s"},
        "perf.mapper_single.stage.*.s": {"op": "<=", "value": 30, "unit": "s"},
        "perf.mapper_single.stage.sfm.s": {"op": "<=", "value": 9, "unit": "s"},
    }}
    t = load_targets(targets_file(tmp_path, doc))
    assert target_for(t, "perf.mapper_single.stage.sfm.s").value == 9  # its own
    assert target_for(t, "perf.mapper_single.stage.cloud.s").value == 30  # the group's pattern
    assert target_for(t, "perf.segment_map.stage.export.s").value == 60  # the generic pattern
    assert target_for(t, "perf.segment_map.wall_s") is None
    m = Metrics()
    m.add("perf.mapper_single.stage.cloud.s", 31.0)
    m.add("perf.segment_map.stage.export.s", 1.0)
    m.add("perf.segment_map.wall_s", 1.0)
    m.judge(t, None)
    assert m.items["perf.mapper_single.stage.cloud.s"].passed is False
    assert m.items["perf.segment_map.stage.export.s"].passed is True
    assert m.items["perf.segment_map.wall_s"].passed is None  # untargeted


def test_targets_file_is_validated(tmp_path: Path) -> None:
    t = load_targets(targets_file(tmp_path))
    assert t["perf.a.wall_s"].tolerance_rel == 0.1  # default
    assert (t["pose.b.fraction"].tolerance_abs, t["pose.b.fraction"].tolerance_rel) == (0.05, 0.0)
    with pytest.raises(TargetsError):
        load_targets(targets_file(tmp_path, {"metrics": {"x": {"op": "<", "value": 1}}}))
    with pytest.raises(TargetsError):
        load_targets(targets_file(tmp_path, {"metrics": {"x": {"op": "<=", "value": "1"}}}))
    with pytest.raises(TargetsError):
        load_targets(tmp_path / "missing.json")


def judged(tmp_path: Path, values: dict[str, float | None],
           baseline: dict[str, float] | None = None) -> Metrics:
    m = Metrics()
    for k, v in values.items():
        m.add(k, v, error=None if v is not None else "reconstruct_json failed (exit 3): down")
    m.judge(load_targets(targets_file(tmp_path)), baseline)
    return m


def test_pass_fail_and_missing_values(tmp_path: Path) -> None:
    m = judged(tmp_path, {"perf.a.wall_s": 2.0, "pose.b.fraction": 0.8, "contract.c.x": None,
                          "seg.untargeted": 5.0})
    got = {k: x.passed for k, x in m.items.items()}
    assert got == {"perf.a.wall_s": True, "pose.b.fraction": False, "contract.c.x": False,
                   "seg.untargeted": None}
    assert "exit 3" in (m.items["contract.c.x"].error or "")


def test_regressions_against_a_stored_baseline(tmp_path: Path) -> None:
    base_result = {"metrics": [{"id": "perf.a.wall_s", "value": 2.0},
                               {"id": "pose.b.fraction", "value": 1.0},
                               {"id": "contract.c.x", "value": 0}]}
    (tmp_path / "baseline.json").write_text(json.dumps(base_result))
    doc, about = load_baseline(tmp_path / "baseline.json")
    assert about["status"] == "compared" and doc is not None
    base = baseline_values(doc)
    # 2.19 s is within 10 % of 2.0 s; 0.96 within 0.05 of 1.0; one contract violation regresses
    m = judged(tmp_path, {"perf.a.wall_s": 2.19, "pose.b.fraction": 0.96, "contract.c.x": 1},
               base)
    assert {k: x.regression for k, x in m.items.items()} == {
        "perf.a.wall_s": False, "pose.b.fraction": False, "contract.c.x": True}
    m = judged(tmp_path, {"perf.a.wall_s": 2.3, "pose.b.fraction": 0.94, "contract.c.x": 0}, base)
    assert {k: x.regression for k, x in m.items.items()} == {
        "perf.a.wall_s": True, "pose.b.fraction": True, "contract.c.x": False}
    assert m.items["perf.a.wall_s"].passed  # a regression still within target passes
    m = judged(tmp_path, {"perf.a.wall_s": 1.0, "pose.b.fraction": 1.0}, base)  # improvements
    assert not any(x.regression for x in m.items.values())


def test_renamed_metrics_of_stored_runs_keep_their_history() -> None:
    old = {"metrics": [{"id": "seg.map.recall", "value": 0.99, "detail": None, "error": None},
                       {"id": "seg.map.precision", "value": 0.88}]}
    assert baseline_values(old) == {"seg.map_consistency.map_objects_detected": 0.99,
                                    "seg.map_consistency.detections_in_map": 0.88}


def test_missing_or_broken_baseline_is_reported(tmp_path: Path) -> None:
    assert load_baseline(tmp_path / "none.json") == (
        None, {"path": str(tmp_path / "none.json"), "status": "missing"})
    (tmp_path / "bad.json").write_text("{")
    doc, about = load_baseline(tmp_path / "bad.json")
    assert doc is None and about["status"].startswith("unreadable")


# -- ground truth --------------------------------------------------------------------------------------


def write_gt(folder: Path) -> None:
    folder.mkdir(parents=True)
    cub = ol.cuboid_val(np.array([0.0, 0.5, 4.0]), np.eye(3), np.array([0.5, 0.5, 0.9]))
    (folder / "restaurant.json").write_text(json.dumps({
        "kind": "objects", "image": "restaurant.jpg",
        "objects": [{"label": "chair", "cuboid": cub},
                    {"label": "person", "cuboid": ol.cuboid_val(
                        np.array([2.0, 0.0, 5.0]), np.eye(3), np.array([0.5, 0.4, 1.7]))}]}))
    sub = folder / "ainex"
    sub.mkdir()
    (sub / "frame1.json").write_text(json.dumps({
        "kind": "objects", "image": "ainex-captures/001_bootstrap_level.jpg",
        "objects": [{"label": "sofa"}, {"label": "plant"}]}))
    (sub / "poses.json").write_text(json.dumps({"kind": "poses", "frames": {
        "001_bootstrap_level.jpg": {"yaw_deg": 100.0, "pitch_deg": 0.0},
        "004_bootstrap_left015_level.jpg": {"yaw_deg": 115.0},
        "005_bootstrap_left015_up.jpg": {"pitch_deg": 12.0}}}))
    (folder / "broken.json").write_text("{")
    (folder / "other.json").write_text(json.dumps({"kind": "depth", "image": "x.jpg"}))
    (folder / "unrun.json").write_text(json.dumps({"kind": "objects", "image": "new.jpg",
                                                   "objects": []}))
    (folder / "README.md").write_text("# format")


def test_ground_truth_is_discovered_without_code_changes(tmp_path: Path) -> None:
    write_gt(tmp_path / "ground_truth")
    files, skipped = gt.discover(tmp_path / "ground_truth")
    assert sorted(f.path.name for f in files) == ["frame1.json", "poses.json", "restaurant.json",
                                                  "unrun.json"]
    assert sorted(Path(s["file"]).name for s in skipped) == ["broken.json", "other.json"]
    assert gt.discover(tmp_path / "absent") == ([], [])


def test_ground_truth_metrics(tmp_path: Path) -> None:
    write_gt(tmp_path / "ground_truth")
    files, skipped = gt.discover(tmp_path / "ground_truth")
    chair = DocObject(4, "chair", 0.9, None, None, tuple(ol.cuboid_val(
        np.array([0.05, 0.5, 4.0]), np.eye(3), np.array([0.5, 0.5, 0.9]))))
    table = DocObject(5, "dining table", 0.8, None, None, tuple(ol.cuboid_val(
        np.array([2.0, 0.0, 5.0]), np.eye(3), np.array([0.5, 0.4, 1.7]))))  # the person's box
    evaluated = {"restaurant.jpg": [chair, table],
                 "ainex-captures/001_bootstrap_level.jpg": [
                     DocObject(1, "couch", 0.7, None, None, None)]}
    m = Metrics()
    gt.object_metrics(m, [f for f in files if f.kind == "objects"], evaluated, skipped)
    # restaurant: chair found (box + label), person's box labelled "dining table" → not found;
    # frame 001: sofa ~ couch
    assert m.items["gt.objects.recall"].value == pytest.approx(2 / 4)
    assert m.items["gt.objects.precision"].value == pytest.approx(2 / 3)
    # boxes 0.5 m wide, 5 cm apart: IoU 0.45 / 0.55 (Monte-Carlo estimate)
    assert m.items["gt.objects.obb_iou_median"].value == pytest.approx(0.818, abs=0.03)
    assert any("new.jpg was not evaluated" in s["reason"] for s in skipped)

    poses = {"001_bootstrap_level.jpg": cam(-20.0, 0.0),
             "004_bootstrap_left015_level.jpg": cam(-4.0, 0.0),
             "005_bootstrap_left015_up.jpg": cam(-5.0, 10.0)}
    gt.pose_metrics(m, [f for f in files if f.kind == "poses"], poses)
    assert m.items["gt.poses.yaw_err_median_deg"].value == pytest.approx(0.5, abs=1e-6)
    assert m.items["gt.poses.yaw_err_max_deg"].value == pytest.approx(0.5, abs=1e-6)
    assert m.items["gt.poses.pitch_err_median_deg"].value == pytest.approx(1.0, abs=1e-6)


def test_no_ground_truth_adds_no_metrics() -> None:
    m = Metrics()
    gt.object_metrics(m, [], {}, [])
    gt.pose_metrics(m, [], None)
    assert m.items == {}


# -- report --------------------------------------------------------------------------------------------


def record(tmp_path: Path, tag: str, code: int, stderr: str, group: str = "reconstruct_json",
           stages: dict | None = None) -> RunRecord:
    out, err = tmp_path / f"{tag}.stdout", tmp_path / f"{tag}.stderr.txt"
    out.write_bytes(b"")
    err.write_text(stderr)
    return RunRecord(RunSpec(tag, group, "reconstruct.sh", ("-i", "x.jpg")),
                     ["reconstruct.sh", "-i", "x.jpg"], code, 1.25, 812.0, 11.2, out, err, stderr,
                     {"stages_s": {"inference": 0.9, "export": 0.1}}, stages=stages)


def result_of(m: Metrics, runs: list[RunRecord], baseline: dict,
              details: dict | None = None) -> dict:
    return build_result(m, runs, details or {}, started="2026-09-24T10:00:00+00:00",
                        finished="2026-09-24T10:30:00+00:00", duration_s=1800.0,
                        env={"commit": "abc", "dirty": False}, targets=Path("targets.json"),
                        baseline=baseline)


def test_result_and_summary_are_written(tmp_path: Path) -> None:
    m = judged(tmp_path, {"perf.a.wall_s": 2.4, "pose.b.fraction": None, "contract.c.x": 0},
               {"perf.a.wall_s": 2.0})
    runs = [record(tmp_path, "reconstruct_json", 0, "timings: total 1.2 s"),
            record(tmp_path, "segment_image", 3, "segment.sh: error: server down\nhint: start it")]
    result = result_of(m, runs, {"path": "b.json", "status": "compared"},
                       {"poses.single": [{"capture": "001.jpg", "commanded_yaw_deg": 0.0,
                                          "registered": False}]})
    res, md = write_report(tmp_path / "out", result)
    doc = json.loads(res.read_text())
    assert doc["summary"] == {"metrics": 3, "passed": 2, "failed": 1, "untargeted": 0,
                              "baseline": "compared", "regressions": 1}
    by_id = {x["id"]: x for x in doc["metrics"]}
    assert by_id["perf.a.wall_s"]["baseline"] == 2.0 and by_id["perf.a.wall_s"]["regression"]
    assert by_id["perf.a.wall_s"]["target"]["tolerance_rel"] == 0.1
    assert [r["ok"] for r in doc["runs"]] == [True, False]
    assert doc["runs"][1]["stderr_tail"].endswith("hint: start it")
    text = md.read_text()
    assert "**Result: FAIL**" in text and "1 regressions against the baseline" in text
    # the why column: the error of a metric without a value, the baseline delta of a regression
    assert "| pose.b.fraction | — | >= 0.9 | reconstruct_json failed (exit 3): down |" in text
    assert ("| perf.a.wall_s | 2.4 | 2 | <= 3 s | worse than the baseline 2 s by 0.4 "
            "(tolerance 0.2) |") in text
    assert "segment.sh: error: server down" in text  # stderr tail of the failed run
    assert "| inference 0.90, export 0.10 |" in text
    assert "## Poses — single map" in text
    assert "No ground-truth annotations" in text


def test_why_a_metric_misses_its_target(tmp_path: Path) -> None:
    m = judged(tmp_path, {"perf.a.wall_s": 3.5, "pose.b.fraction": 0.8, "contract.c.x": 2})
    text = summary_md(result_of(m, [], {"path": "b.json", "status": "missing"}))
    assert "| perf.a.wall_s | 3.5 | <= 3 s | 3.5 s exceeds the limit 3 s |" in text
    assert "| pose.b.fraction | 0.8 | >= 0.9 | 0.8 is below the minimum 0.9 |" in text
    assert "| contract.c.x | 2 | <= 0 | 2 exceeds the limit 0 |" in text


@pytest.mark.parametrize("status", ["missing", "unreadable: Expecting value"])
def test_without_a_baseline_nothing_is_compared(tmp_path: Path, status: str) -> None:
    m = judged(tmp_path, {"perf.a.wall_s": 2.0, "pose.b.fraction": 0.95, "contract.c.x": 0})
    result = result_of(m, [], {"path": "b.json", "status": status})
    word = status.split(":")[0]
    assert result["summary"]["regressions"] is None and result["summary"]["baseline"] == word
    text = summary_md(result)
    assert f"baseline {word} — not compared" in text
    assert "regressions against the baseline" not in text
    assert "**Result: PASS**" in text


def test_per_stage_time_and_memory(tmp_path: Path) -> None:
    def st(s: float, mb: float, gb: float | None) -> dict:
        return {"s": s, "client_peak_mb": mb, "server_peak_gb": gb}

    runs = [record(tmp_path, "recon", 0, "", stages={"inference": st(0.9, 500.0, 11.3),
                                                      "export": st(0.1, 510.0, None)}),
            *(record(tmp_path, f"mapper_split_{k}", 0, "", "mapper_split",
                     {"sfm": st(20.0 + k, 2000.0 * k, 14.0 + k)}) for k in (1, 2))]
    m = Metrics()
    perf_metrics(m, runs)
    stages = m.items["perf.reconstruct_json.wall_s"].detail["stages"]
    assert stages == {"inference": st(0.9, 500.0, 11.3), "export": st(0.1, 510.0, None)}
    # per mapping update: the slowest update's stage time, the peak memory over the updates
    assert m.items["perf.mapper_split.wall_s"].detail["stages"] == {"sfm": st(22.0, 4000.0, 16.0)}
    assert m.items["perf.mapper_split.per_update_wall_s"].value == max(r.wall_s for r in runs[1:])
    # every stage is a metric of its own (time, client and server memory)
    assert m.items["perf.reconstruct_json.stage.inference.s"].value == 0.9
    assert m.items["perf.reconstruct_json.stage.inference.client_peak_mb"].value == 500.0
    assert m.items["perf.mapper_split.stage.sfm.server_peak_gb"].value == 16.0
    export = m.items["perf.reconstruct_json.stage.export.server_peak_gb"]
    assert export.value is None and "not sampled" in (export.error or "")
    text = summary_md(result_of(m, runs, {"status": "missing"}))
    assert "## Per-stage time and peak memory" in text
    assert "| reconstruct_json | inference | 0.9 | 500 | 11.3 |" in text
    assert "| reconstruct_json | export | 0.1 | 510 | — |" in text
    # a few runs (the split map's updates): one row per update
    assert "| mapper_split_1 | sfm | 21 | 2000 | 15 |" in text
    assert "| mapper_split_2 | sfm | 22 | 4000 | 16 |" in text


def test_cli_usage_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = targets_file(tmp_path, {"metrics": {"x": {"op": "==", "value": 1}}})
    assert main(["--targets", str(bad), "--out", str(tmp_path / "o")]) == 2
    assert main(["--out", str(EXAMPLES / "results")]) == 2  # inside the repository
    assert not (tmp_path / "o").exists()
    err = capsys.readouterr().err
    assert "outside the repository" in err


def test_a_stored_run_is_judged_again_with_new_targets(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Targets are data: ``--rejudge`` judges a stored run again without running anything."""
    m = judged(tmp_path, {"perf.a.wall_s": 2.4, "pose.b.fraction": 0.95, "contract.c.x": 0})
    run = tmp_path / "run"
    write_report(run, result_of(m, [], {"status": "missing"}))
    strict = targets_file(tmp_path, {"metrics": {"perf.a.wall_s": {"op": "<=", "value": 2.0},
                                                 "pose.b.fraction": {"op": ">=", "value": 0.9}}})
    assert main(["--rejudge", str(run), "--targets", str(strict),
                 "--baseline", str(tmp_path / "none.json")]) == 1
    result = json.loads((run / "result.json").read_text())
    by_id = {x["id"]: x for x in result["metrics"]}
    assert by_id["perf.a.wall_s"]["passed"] is False and by_id["pose.b.fraction"]["passed"]
    assert by_id["contract.c.x"]["passed"] is None  # no target any more
    assert result["judged"] and result["summary"]["failed"] == 1
    assert "judged again" in (run / "summary.md").read_text()
    assert main(["--rejudge", str(tmp_path / "nothing")]) == 2
    capsys.readouterr()
