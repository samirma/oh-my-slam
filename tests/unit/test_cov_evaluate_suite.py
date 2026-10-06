"""The evaluation plan's paths that the all-failing run does not take, offline with fake entry
points (spec §5): the inference server's cold start, footprint and restore; a run whose output is
not JSON; a viewer that rendered but served no scene; the ainex maps built in one update and
split, with the held-out captures located between the updates, then judged (poses, depth
agreement, object stability); the split's middle updates publishing ids through their points'
labels; the reference-map runs failing; office splits that cannot be built; and a section that
raises."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oh_my_slam.core.ply import PointCloud, ply_bytes
from oh_my_slam.core.types import Pose
from oh_my_slam.segmentation.colors import UNSEGMENTED, segment_colors
from oh_my_slam.tools.evaluate import mapupdate
from oh_my_slam.tools.evaluate import suite as st
from oh_my_slam.tools.evaluate.names import Capture, captures_in
from oh_my_slam.tools.evaluate.runner import Runner, RunSpec
from oh_my_slam.tools.evaluate.scene import DocObject
from oh_my_slam.tools.evaluate.suite import EXAMPLES, Evaluation, _pose_lines, office_splits
from oh_my_slam.tools.evaluate.viewer import BrowserProbe, ViewOutcome
from tests.unit.test_evaluate_locate import frame
from tests.unit.test_evaluate_mapupdate import (
    ANNOTATION,
    IMAGES,
    early_map,
    final_map,
    office_examples,
    office_repo,
)
from tests.unit.test_evaluate_metrics import box, cam, record, sphere_depth, write_map
from tests.unit.test_evaluate_runner import fake_repo


def evaluation(tmp_path: Path, examples: Path = EXAMPLES, **bodies: str) -> Evaluation:
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "out"
    return Evaluation(out, Runner(out, fake_repo(tmp_path, **bodies)), BrowserProbe(None),
                      examples=examples)


# -- the plan's inputs ---------------------------------------------------------------------------------


def test_the_office_splits_come_from_the_annotation_and_the_sequence(tmp_path: Path) -> None:
    assert office_splits(tmp_path) == []  # no annotation
    (tmp_path / "ground_truth").mkdir()
    (tmp_path / "ground_truth" / "office.json").write_text(json.dumps(
        {**ANNOTATION, "splits": [[4, 2]]}))
    assert office_splits(tmp_path) == []  # annotated, but the sequence is not there
    (tmp_path / "office_sequence").mkdir()
    for name in IMAGES:
        (tmp_path / "office_sequence" / name).write_bytes(b"")
    assert office_splits(tmp_path) == ["split_4_2"]
    assert set(mapupdate.metric_ids(["split_4_2"])) <= set(st.expected_ids(tmp_path))


def test_a_pose_header_line_that_is_not_one() -> None:
    ply = ply_bytes(PointCloud(np.zeros((1, 3), np.float32)),
                    comments=["located_0", "located_1 {not json"])
    assert _pose_lines(ply, 2) == ["not a pose line: located_0",
                                   "not a pose line: located_1 {not json"]


# -- running and checking one command ----------------------------------------------------------------


def test_a_successful_run_whose_output_is_not_json(tmp_path: Path) -> None:
    ev = evaluation(tmp_path, reconstruct='echo "model loaded"')
    rec = ev.run("recon", "g", "reconstruct.sh")
    assert rec.ok and ev.scene(rec) is None
    (problem,) = ev.contracts.checks[("openlabel", "reconstruct")]["recon"]
    assert problem.startswith("not JSON")
    assert ("colour", "reconstruct") not in ev.contracts.checks


class Rendered:
    """A probe whose page rendered, served ``scene`` from /api/scene and no cloud."""

    def __init__(self, scene: bytes | None) -> None:
        self.scene = scene

    def measure(self, runner: Runner, spec: RunSpec) -> ViewOutcome:
        live = runner.start(spec)
        live.wait(30)
        return ViewOutcome(live.finish(), "http://127.0.0.1:1/", 1.25, None,
                           ["WebGL warning"], self.scene)


def test_a_rendered_view_whose_scene_is_not_json(tmp_path: Path) -> None:
    ev = evaluation(tmp_path, view="exit 0")
    ev.probe = Rendered(b"<html>")  # type: ignore[assignment]
    ev.view("view_image", ("image", "segment.sh -i", None), "-i", "x.jpg")
    (rec,) = ev.runner.records
    assert "render_error" not in rec.notes
    assert ev.details["viewer.view_image"] == {"url": "http://127.0.0.1:1/", "render_s": 1.25,
                                               "error": None, "console_errors": ["WebGL warning"]}
    assert ev.contracts.checks[("openlabel", "view")]["view_image"][0].startswith("not JSON")
    assert ("colour", "view") not in ev.contracts.checks


# -- the inference server ------------------------------------------------------------------------------

SERVER = """case "$1" in --status) echo '{"status": "ready", "device": "mps"}';; esac
exit 0"""


def test_the_cold_start_footprint_and_restore(tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(st, "server_pid", lambda: 4242)
    monkeypatch.setattr(st, "phys_footprint_gb", lambda pid: (9.5, 12.0))
    ev = evaluation(tmp_path, start_inference_server=SERVER)
    ev.server_start()
    m = ev.metrics.items
    assert ev.was_running
    cold = next(r for r in ev.runner.records if r.tag == "server_cold_start")
    assert m["perf.server.cold_start_s"].value == cold.wall_s
    assert m["perf.server.resident_gb"].value == 9.5
    assert m["perf.server.resident_gb"].detail == {"pid": 4242}
    assert ev.details["server_health"] == {"status": "ready", "device": "mps"}
    ev.server_restore()  # it was running and still is: left alone
    assert m["perf.server.peak_gb"].value == 12.0
    assert [r.tag for r in ev.runner.records] == ["server_status_initial", "server_stop",
                                                  "server_cold_start", "server_status"]
    monkeypatch.setattr(st, "server_pid", lambda: None)
    ev = evaluation(tmp_path / "2", start_inference_server=SERVER)
    ev.server_start()
    resident = ev.metrics.items["perf.server.resident_gb"]
    assert resident.value is None and resident.error == "server footprint unavailable"
    ev.server_restore()  # it was running at the start and is down now: restarted
    peak = ev.metrics.items["perf.server.peak_gb"]
    assert peak.error == "the server is not running at the end of the evaluation"
    assert ev.runner.records[-1].tag == "server_restart"


# -- the ainex maps ------------------------------------------------------------------------------------

CAPTURES = captures_in(EXAMPLES / "ainex-captures")[:6]  # 001-006: three updates of two
TILT = {"level": 0.0, "up": 10.0, "down": -10.0}
POSES = {c.name: cam(c.yaw_deg, TILT[c.tilt]) for c in CAPTURES}  # every camera at the origin
OBJECTS = [box(1, "chair", (2.0, 0.0, 0.45)), box(2, "sofa", (0.0, 3.0, 0.4), (2.0, 0.9, 0.8))]


def scene(captures: list[Capture], objects: list[DocObject],
          located: list[Capture] | None = None) -> dict[str, Any]:
    """A map's (or a locate's) scene: its frames' sources and poses, and its objects."""
    frames = {str(k): frame(POSES[c.name], f"/in/{c.name}") for k, c in enumerate(captures)}
    frames.update({str(len(captures) + k): frame(POSES[c.name], f"/q/{c.name}", located=True)
                   for k, c in enumerate(located or [])})
    objs = {str(o.id): {"name": f"{o.label} {o.id}", "type": o.label, "object_data": {
        "cuboid": [{"name": "obb", "val": list(o.cuboid or ())}]}} for o in objects}
    return {"openlabel": {"metadata": {}, "frames": frames, "objects": objs}}


def middle_ply(kind: str) -> bytes:
    if kind == "broken":
        return b"ply\nnot a header\n"
    labels = np.array([0, 5, 5], np.int32)
    xyz = np.zeros((3, 3), np.float32)
    if kind == "unlabelled":
        return ply_bytes(PointCloud(xyz, np.array([UNSEGMENTED] * 3, np.uint8)))
    return ply_bytes(PointCloud(xyz, segment_colors(labels), labels))


MAPPER = """cmd="$1"; shift; out=""; dir=""; mode=""; fmt=""
while [ $# -gt 0 ]; do
  case "$1" in -o) out="$2"; shift;; -m) dir="$2"; shift;; -t) mode="$2"; shift;;
    -f) fmt="$2"; shift;; esac
  shift
done
echo "$cmd $mode $fmt" >> "$DOCS/log"
if [ "$cmd" = locate ]; then
  [ -n "$LOCATE_FAIL" ] && { echo "mapper.sh: error: nothing matched" >&2; exit 1; }
  cat "$DOCS/located.json"; exit 0
fi
mkdir -p "$dir"; cp -R "$DOCS/map/." "$dir/"
[ -n "$out" ] && { mkdir -p "$(dirname "$out")"; cp "$DOCS/first.json" "$out"; exit 0; }
[ "$fmt" = ply ] && { cat "$DOCS/middle.ply"; exit 0; }
cat "$DOCS/full.json"
"""


def ainex(tmp_path: Path, middle: str = "labelled") -> Evaluation:
    docs = tmp_path / "docs"
    write_map(docs / "map", [record(i, c.name, POSES[c.name]) for i, c in enumerate(CAPTURES)],
              {f"f{i:06d}": sphere_depth() for i in range(len(CAPTURES))})
    held = CAPTURES[2:4]
    (docs / "full.json").write_text(json.dumps(scene(CAPTURES, OBJECTS)))
    (docs / "first.json").write_text(json.dumps(scene(CAPTURES[:2], OBJECTS[:1])))
    (docs / "located.json").write_text(json.dumps(scene([], [], located=held)))
    (docs / "middle.ply").write_bytes(middle_ply(middle))
    ev = evaluation(tmp_path, mapper=MAPPER)
    ev.runner.env["DOCS"] = str(docs)
    return ev


def test_the_maps_are_built_located_and_judged(tmp_path: Path) -> None:
    ev = ainex(tmp_path)
    single, split = ev.build_maps(CAPTURES)
    assert single is not None and split is not None
    assert [r.tag for r in ev.runner.records] == ["mapper_single", "mapper_split_1",
                                                  "locate_held_out", "mapper_split_2",
                                                  "mapper_split_3"]
    assert (tmp_path / "docs" / "log").read_text().splitlines() == [
        "update  ", "update  ", "locate  ", "update single ply", "update full "]
    # the ids the earlier updates published: the first's objects, the middle's point labels
    assert ev.published == {1, 5}
    checks = ev.contracts.checks
    assert checks[("readonly", "map")] == {"mapper.sh locate (held out)": []}
    assert checks[("openlabel", "mapper")]["locate_held_out/located frames"] == []
    assert checks[("colour", "mapper")]["mapper_split_2"] == []
    assert [c.name for c in ev.held_out["captures"]] == [c.name for c in CAPTURES[2:4]]
    ev.map_metrics(CAPTURES, single, split)
    v = {k: x.value for k, x in ev.metrics.items.items()}
    for mp in ("single", "split"):
        assert v[f"pose.{mp}.registered_fraction"] == 1.0
        assert v[f"pose.{mp}.yaw_err_max_deg"] == pytest.approx(0.0, abs=1e-6)
        assert v[f"pose.{mp}.pitch_direction_fraction"] == 1.0  # 005 up, 006 down of 004
        assert v[f"pose.{mp}.same_heading_yaw_diff_max_deg"] is None  # no such pair in 001-006
        assert v[f"map.{mp}.frame_agreement_median_pct"] is None
        assert v[f"map.{mp}.frame_agreement_pairs_max_pct"] == pytest.approx(0.0, abs=0.2)
    assert v["map.stability.matched_fraction"] == 1.0 and v["map.stability.id_agreement"] == 1.0
    assert v["pose.locate.located_fraction"] == 1.0
    assert v["pose.locate.yaw_err_max_deg"] == pytest.approx(0.0, abs=1e-6)
    rows = ev.details["poses.locate"]
    assert [r["capture"] for r in rows] == [c.name for c in CAPTURES[2:4]]
    assert all(r["vs_mapped_rot_deg"] == pytest.approx(0.0, abs=1e-3) for r in rows)
    assert ev.single_poses is not None and set(ev.single_poses) == set(POSES)
    assert len(ev.details["map.single.pairs"]) > 0 and ev.details["map.stability"]


@pytest.mark.parametrize(("middle", "published", "problem"), [
    ("unlabelled", {1}, "no label property and no object list to check the colours against"),
    ("broken", {1}, "ValueError: "),
])
def test_a_middle_update_without_point_labels_publishes_nothing(
        tmp_path: Path, middle: str, published: set[int], problem: str) -> None:
    ev = ainex(tmp_path, middle)
    ev.build_maps(CAPTURES)
    assert ev.published == published
    (found,) = ev.contracts.checks[("colour", "mapper")]["mapper_split_2"]
    assert found.startswith(problem)


def test_a_failed_held_out_locate_fails_its_metrics(tmp_path: Path) -> None:
    ev = ainex(tmp_path)
    ev.runner.env["LOCATE_FAIL"] = "1"
    single, split = ev.build_maps(CAPTURES)
    assert single is not None and split is not None  # the split map is still built
    assert ev.held_out["located"] is None
    assert "locate_held_out/located frames" not in ev.contracts.checks[("openlabel", "mapper")]
    assert ev.contracts.checks[("readonly", "map")] == {"mapper.sh locate (held out)": []}
    ev.held_out_metrics(split)
    for k in ("located_fraction", "yaw_err_median_deg", "yaw_err_max_deg"):
        assert ev.metrics.items[f"pose.locate.{k}"].error == "mapper.sh locate failed"


def test_the_reference_map_runs_failing_leave_nothing_to_compare(tmp_path: Path) -> None:
    ev = evaluation(tmp_path)  # every entry point fails
    ref = tmp_path / "ref"
    ref.mkdir()
    ev.locate_reference(scene(CAPTURES, OBJECTS), ref)
    assert [(r.tag, r.ok) for r in ev.runner.records] == [
        ("locate_single", False), ("locate_full", False), ("locate_ply", False)]
    assert ("openlabel", "mapper") not in ev.contracts.checks
    assert ("same_objects", "map") not in ev.contracts.checks
    assert ev.contracts.checks[("stdout", "mapper")] == {
        "locate_single": [], "locate_full": [], "locate_ply": []}  # failed, but silent


def test_held_out_metrics_without_the_split_map_compare_nothing_later(tmp_path: Path) -> None:
    ev = evaluation(tmp_path)
    ev.held_out = {"located": {CAPTURES[2].name: POSES[CAPTURES[2].name]},
                   "reference": Pose.identity(), "captures": CAPTURES[2:3]}
    ev.held_out_metrics(None)
    (row,) = ev.details["poses.locate"]
    assert row["located"] and "vs_mapped_rot_deg" not in row


# -- the office sequence --------------------------------------------------------------------------------


def test_an_annotation_naming_no_image_of_the_sequence_maps_nothing(tmp_path: Path) -> None:
    root = office_examples(tmp_path)
    (root / "ground_truth" / "office.json").write_text(json.dumps(
        {**ANNOTATION, "absent": [{"label": "cup", "seen_in": {"other.jpg": [0, 0, 1, 1]}}]}))
    ev = evaluation(tmp_path, root)
    ev.map_update()
    assert ev.runner.records == []
    for k in mapupdate.metric_ids():
        assert ev.metrics.items[k].error == f"no annotated image is in {root / 'office_sequence'}"
    assert ev.details["map_update"]["before_images"] == []


def test_a_split_that_does_not_cover_the_sequence_fails_alone(tmp_path: Path) -> None:
    root = office_examples(tmp_path)
    (root / "ground_truth" / "office.json").write_text(json.dumps(
        {**ANNOTATION, "splits": [[3, 3], [2, 2]]}))
    out = tmp_path / "out"
    ev = Evaluation(out, Runner(out, office_repo(tmp_path, final_map(False), early_map())),
                    BrowserProbe(None), examples=root)
    ev.map_update()
    m = ev.metrics.items
    for k in mapupdate.split_metric_ids("split_2_2"):
        assert m[k].error == "split 2+2 does not cover the 6 images of the sequence"
    assert m["map_update.split_3_3.absent_fraction"].value == 1.0  # the other split is judged
    assert [r.tag for r in ev.runner.records] == [
        "mapper_office", "mapper_office_split_3_3_1", "mapper_office_split_3_3_2"]


# -- sections ----------------------------------------------------------------------------------------------


def test_an_error_ends_only_its_section(tmp_path: Path,
                                        capsys: pytest.CaptureFixture[str]) -> None:
    ev = evaluation(tmp_path)

    def boom(what: str) -> None:
        raise RuntimeError(f"no {what}")

    assert ev.section("maps", boom, "map") is None
    assert ev.section("sum", lambda a, b: a + b, 1, 2) == 3
    assert ev.details["errors"] == ["maps: RuntimeError: no map"]
    err = capsys.readouterr().err
    assert "== maps" in err and "Traceback" in err and "== sum" in err
