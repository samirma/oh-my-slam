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
)
from oh_my_slam.tools.evaluate.report import build_result, write_report
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
    assert set(expected_ids()) <= set(targets)
    gt_ids = {"gt.objects.recall", "gt.objects.precision", "gt.objects.obb_iou_median",
              "gt.poses.yaw_err_median_deg", "gt.poses.yaw_err_max_deg",
              "gt.poses.pitch_err_median_deg"}
    assert gt_ids <= set(targets)
    assert set(targets) == set(expected_ids()) | gt_ids  # no stale targets


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


def record(tmp_path: Path, tag: str, code: int, stderr: str) -> RunRecord:
    out, err = tmp_path / f"{tag}.stdout", tmp_path / f"{tag}.stderr.txt"
    out.write_bytes(b"")
    err.write_text(stderr)
    return RunRecord(RunSpec(tag, "reconstruct_json", "reconstruct.sh", ("-i", "x.jpg")),
                     ["reconstruct.sh", "-i", "x.jpg"], code, 1.25, 812.0, 11.2, out, err, stderr,
                     {"stages_s": {"inference": 0.9, "export": 0.1}})


def test_result_and_summary_are_written(tmp_path: Path) -> None:
    m = judged(tmp_path, {"perf.a.wall_s": 2.4, "pose.b.fraction": None, "contract.c.x": 0},
               {"perf.a.wall_s": 2.0})
    runs = [record(tmp_path, "reconstruct_json", 0, "timings: total 1.2 s"),
            record(tmp_path, "segment_image", 3, "segment.sh: error: server down\nhint: start it")]
    result = build_result(m, runs, {"poses.single": [{"capture": "001.jpg",
                                                      "commanded_yaw_deg": 0.0,
                                                      "registered": False}]},
                          started="2026-09-24T10:00:00+00:00", finished="2026-09-24T10:30:00+00:00",
                          duration_s=1800.0, env={"commit": "abc", "dirty": False},
                          targets=Path("targets.json"),
                          baseline={"path": "b.json", "status": "compared"})
    res, md = write_report(tmp_path / "out", result)
    doc = json.loads(res.read_text())
    assert doc["summary"] == {"metrics": 3, "passed": 2, "failed": 1, "untargeted": 0,
                              "regressions": 1}
    by_id = {x["id"]: x for x in doc["metrics"]}
    assert by_id["perf.a.wall_s"]["baseline"] == 2.0 and by_id["perf.a.wall_s"]["regression"]
    assert [r["ok"] for r in doc["runs"]] == [True, False]
    assert doc["runs"][1]["stderr_tail"].endswith("hint: start it")
    text = md.read_text()
    assert "**Result: FAIL**" in text
    assert "## Failed metrics" in text and "pose.b.fraction" in text
    assert "## Regressions against the baseline" in text
    assert "segment.sh: error: server down" in text  # stderr tail of the failed run
    assert "| inference 0.90, export 0.10 |" in text
    assert "## Poses — single map" in text


def test_cli_usage_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = targets_file(tmp_path, {"metrics": {"x": {"op": "==", "value": 1}}})
    assert main(["--targets", str(bad), "--out", str(tmp_path / "o")]) == 2
    assert main(["--out", str(EXAMPLES / "results")]) == 2  # inside the repository
    assert main(["--splits", "2", "--out", str(tmp_path / "o")]) == 2
    assert not (tmp_path / "o").exists()
    err = capsys.readouterr().err
    assert "outside the repository" in err and "--splits" in err
