"""The evaluator's data paths on synthetic inputs, offline (spec §4, §5): payloads that break the
stdout contract in the rarer ways, malformed ground-truth annotations, accuracy without a usable
reference, keyframe pairs that cannot be compared, map views without intrinsics or shared
captures, performance groups with failed or many runs, and frames without a pose."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oh_my_slam.core.ply import PointCloud, ply_bytes
from oh_my_slam.mapping import store
from oh_my_slam.segmentation.catalog import catalog_csv
from oh_my_slam.tools.evaluate import groundtruth as gt
from oh_my_slam.tools.evaluate import mapupdate as mu
from oh_my_slam.tools.evaluate.contracts import (
    catalog_csv_problems,
    payload_problems,
    scene_colour_problems,
)
from oh_my_slam.tools.evaluate.locate import held_out_metrics
from oh_my_slam.tools.evaluate.mapquality import agreement_metrics, split_alignment
from oh_my_slam.tools.evaluate.metrics import Metrics
from oh_my_slam.tools.evaluate.names import captures_in, parse_capture, same_heading_pairs
from oh_my_slam.tools.evaluate.performance import PER_RUN_DETAIL_MAX, perf_metrics
from oh_my_slam.tools.evaluate.runner import RunRecord, RunSpec
from oh_my_slam.tools.evaluate.scene import DocObject, doc_objects, frame_poses
from tests.unit.test_evaluate_contracts import objects, scene_doc
from tests.unit.test_evaluate_mapupdate import (
    CAMERA,
    IMAGES,
    early_map,
    final_map,
    judged,
    view,
)
from tests.unit.test_evaluate_mapupdate import (
    scene_doc as map_scene,
)
from tests.unit.test_evaluate_metrics import SEQUENCE, K, cam, record, write_map
from tests.unit.test_view_cli import minimal_map

# -- stdout purity and the colour contract ---------------------------------------------------------


def test_a_json_payload_must_be_utf8() -> None:
    assert payload_problems(b'{"label": "caf\xe9"}', "json")[0].startswith("not UTF-8")


def test_an_ascii_ply_whose_rows_do_not_parse() -> None:
    head = (b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\n"
            b"property float z\nend_header\n")
    assert payload_problems(head + b"1 2 3\n", "ply") == []
    assert payload_problems(head + b"a b c\n", "ply")[0].startswith("ASCII body does not parse")


def test_an_object_id_must_be_positive() -> None:
    assert scene_colour_problems([DocObject(0, "chair", 0.9, None, None, None)]) == [
        "object id 0 is not positive"]


def test_catalog_csv_rows_must_be_the_scenes_objects() -> None:
    objs = doc_objects(scene_doc())  # ids 1, 2, 7, 23
    assert catalog_csv_problems(catalog_csv(objects()), objs[:2]) == [
        "catalog.csv ids differ from the scene: [7, 23]"]


# -- ground truth ------------------------------------------------------------------------------------

ABSENT = [{"label": "cup", "seen_in": {"a.jpg": [0.1, 0.1, 0.4, 0.4]}}]


@pytest.mark.parametrize(("doc", "reason"), [
    ({"kind": "objects", "image": "a.jpg"}, "needs 'image' and an 'objects' list"),
    ({"kind": "objects", "image": "a.jpg", "objects": [{"label": "cup", "cuboid": [0] * 9}]},
     "optional 10-value 'cuboid'"),
    ({"kind": "objects", "image": "a.jpg", "objects": [{"cuboid": [0] * 10}]}, "needs a 'label'"),
    ({"kind": "poses"}, "needs a 'frames' object"),
    ({"kind": "poses", "frames": {"001_bootstrap_level.jpg": 3.0}}, "needs a 'frames' object"),
    ({"kind": "map_update", "sequence": "office_sequence", "absent": [{"label": "cup"}]},
     "needs a 'label' and 'seen_in'"),
    ({"kind": "map_update", "sequence": "office_sequence", "absent": ABSENT,
      "stable": "monitor"}, "'stable' is a list of labels"),
])
def test_malformed_annotations_are_skipped_with_the_reason(tmp_path: Path, doc: dict[str, Any],
                                                           reason: str) -> None:
    (tmp_path / "a.json").write_text(json.dumps(doc))
    files, skipped = gt.discover(tmp_path)
    assert files == [] and len(skipped) == 1 and reason in skipped[0]["reason"]


def test_the_map_update_plan_reads_only_map_update_files(tmp_path: Path) -> None:
    (tmp_path / "objects.json").write_text(json.dumps(
        {"kind": "objects", "image": "restaurant.jpg", "objects": [{"label": "cup"}]}))
    (tmp_path / "update.json").write_text(json.dumps(
        {"kind": "map_update", "sequence": "office_sequence", "absent": ABSENT}))
    files, skipped = gt.discover(tmp_path)
    assert [f.kind for f in files] == ["objects", "map_update"]
    plan = gt.map_update_plan(files, skipped)
    assert plan is not None and plan.files == [tmp_path / "update.json"] and skipped == []


def test_objects_annotated_without_boxes_are_paired_by_label(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(json.dumps({"kind": "objects", "image": "restaurant.jpg",
                                                 "objects": [{"label": "chair"},
                                                             {"label": "sofa"}]}))
    files, skipped = gt.discover(tmp_path)
    m = Metrics()
    found = [DocObject(1, "chair", 0.9, None, None, None), DocObject(2, "cup", 0.8, None, None,
                                                                     None),
             DocObject(3, "couch", 0.7, None, None, None)]
    gt.object_metrics(m, files, {"restaurant.jpg": found}, skipped)
    assert m.items["gt.objects.recall"].value == 1.0  # sofa ~ couch
    assert m.items["gt.objects.precision"].value == pytest.approx(2 / 3)
    assert m.items["gt.objects.recall"].detail == {
        "restaurant.jpg": {"truth": 2, "detections": 3, "paired": 2}}
    assert "gt.objects.obb_iou_median" not in m.items  # no box to compare


def poses_file(tmp_path: Path) -> list[gt.GroundTruth]:
    (tmp_path / "poses.json").write_text(json.dumps({"kind": "poses", "frames": {
        "001_bootstrap_level.jpg": {"yaw_deg": 10.0},
        "005_bootstrap_left015_up.jpg": {"pitch_deg": 12.0}}}))
    return gt.discover(tmp_path)[0]


def test_annotated_poses_without_a_map_or_a_registered_capture_fail(tmp_path: Path) -> None:
    files = poses_file(tmp_path)
    ids = ["gt.poses.yaw_err_median_deg", "gt.poses.yaw_err_max_deg",
           "gt.poses.pitch_err_median_deg"]
    m = Metrics()
    gt.pose_metrics(m, files, None)
    assert [m.items[k].error for k in ids] == ["the one-update map was not built"] * 3
    m = Metrics()
    gt.pose_metrics(m, files, {"002_bootstrap_side1_level.jpg": cam(0.0)})
    assert [m.items[k].error for k in ids] == [
        "no annotated capture is registered in the map"] * 3
    assert all(m.items[k].value is None for k in ids)


def test_only_the_annotated_angles_are_measured(tmp_path: Path) -> None:
    (tmp_path / "yaw" / "poses.json").parent.mkdir()
    (tmp_path / "yaw" / "poses.json").write_text(json.dumps({"kind": "poses", "frames": {
        "001_bootstrap_level.jpg": {"yaw_deg": 10.0}}}))
    m = Metrics()
    gt.pose_metrics(m, gt.discover(tmp_path / "yaw")[0], {"001_bootstrap_level.jpg": cam(-5.0)})
    assert set(m.items) == {"gt.poses.yaw_err_median_deg", "gt.poses.yaw_err_max_deg"}
    assert m.items["gt.poses.yaw_err_max_deg"].value == pytest.approx(0.0, abs=1e-9)  # offset
    (tmp_path / "pitch" / "poses.json").parent.mkdir()
    (tmp_path / "pitch" / "poses.json").write_text(json.dumps({"kind": "poses", "frames": {
        "005_bootstrap_left015_up.jpg": {"pitch_deg": 12.0}}}))
    m = Metrics()
    gt.pose_metrics(m, gt.discover(tmp_path / "pitch")[0],
                    {"005_bootstrap_left015_up.jpg": cam(15.0, 10.0)})
    assert set(m.items) == {"gt.poses.pitch_err_median_deg"}
    assert m.items["gt.poses.pitch_err_median_deg"].value == pytest.approx(2.0)


# -- pose accuracy, map quality ---------------------------------------------------------------------


def test_located_captures_without_the_maps_reference_frame_have_no_yaw() -> None:
    held = [parse_capture("011_left_060_level.jpg")]
    m = Metrics()
    rows = held_out_metrics(m, "pose.locate", {held[0].name: cam(40.0)}, None, held, None)
    assert m.items["pose.locate.located_fraction"].value == 1.0
    assert rows == [{"capture": held[0].name, "commanded_yaw_deg": 60.0}]
    for k in ("yaw_err_median_deg", "yaw_err_max_deg"):
        x = m.items[f"pose.locate.{k}"]
        assert x.value is None and x.error == "the map has no pose of capture 001"


def test_a_keyframe_pair_without_enough_shared_pixels_is_not_compared(tmp_path: Path) -> None:
    caps = captures_in(SEQUENCE)[:2]
    recs = [record(0, caps[0].name, cam(0.0)), record(1, caps[1].name, cam(20.0))]
    write_map(tmp_path, recs, {"f000001": np.zeros((K.height, K.width))})  # no valid depth
    m = Metrics()
    assert agreement_metrics(m, "map.t", tmp_path, caps) == []
    pairs = m.items["map.t.frame_agreement_pairs_median_pct"]
    assert pairs.value is None
    assert pairs.error == "no overlapping keyframe pair to compare (1 candidates)"
    assert "no same-heading pair" in (m.items["map.t.frame_agreement_median_pct"].error or "")


def test_maps_sharing_no_capture_cannot_be_aligned() -> None:
    with pytest.raises(ValueError, match="share no registered capture"):
        split_alignment({"a.jpg": cam(0.0)}, {"b.jpg": cam(0.0)})


def test_same_heading_pairs_need_both_captures_and_one_heading() -> None:
    assert same_heading_pairs([parse_capture("001_bootstrap_level.jpg"),
                               parse_capture("026_left_210_level.jpg")]) == []
    with pytest.raises(ValueError, match="do not share a commanded heading"):
        same_heading_pairs([parse_capture("001_bootstrap_level.jpg"),
                            parse_capture("053_left_030_level.jpg")])


# -- map update -------------------------------------------------------------------------------------


def test_the_map_cloud_is_read_from_the_map(tmp_path: Path) -> None:
    root = minimal_map(tmp_path / "m")
    assert mu.map_points(root) is None  # a map without a cloud
    xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    (root / store.CLOUD_PLY).write_bytes(ply_bytes(PointCloud(xyz.astype(np.float32))))
    got = mu.map_points(root)
    assert got is not None and got.dtype == np.float64 and np.allclose(got, xyz)
    assert mu.map_points(tmp_path / "not-a-map") is None


def test_the_camera_comes_from_the_first_stream_with_intrinsics(tmp_path: Path) -> None:
    doc = map_scene(final_map(False), "final")
    streams = doc["openlabel"]["streams"]
    doc["openlabel"]["streams"] = {"camera_0": {"type": "camera"}, **streams}
    v = mu.MapView.of(doc, tmp_path)
    assert v.camera == CAMERA and set(v.poses) == set(IMAGES) and v.cloud is None
    doc["openlabel"]["streams"] = {"camera_0": {"type": "camera"}}
    assert mu.MapView.of(doc, tmp_path).camera is None


def test_an_unregistered_image_of_the_absent_object_is_named_in_the_hole_test(
        tmp_path: Path) -> None:
    registered = [n for n in IMAGES if n != IMAGES[2]]
    final = view(final_map(False).objects, registered)
    m, _ = judged(tmp_path, final, early_map())
    holes = m.items["map_update.hole_fraction"]
    assert holes.detail[f"cup in {IMAGES[2]}"] == "not registered"
    assert holes.value == 0.0 and holes.detail[f"cup in {IMAGES[0]}"]["cells"] == 64


def test_published_ids_persist_without_boxes_or_shared_captures(tmp_path: Path) -> None:
    """Objects without a box keep their id on label alone; two updates that share no registered
    capture are compared where they stand (no alignment)."""
    p = gt.MapUpdatePlan("office_sequence", [gt.Absent("cup", {IMAGES[0]: (0.4, 0.4, 0.6, 0.6)})],
                         [], [])
    bare = [DocObject(1, "monitor", 0.9, None, None, None, frozenset({0}))]
    m = Metrics()
    first = mu.MapView(bare, {IMAGES[0]: cam(0.0)}, {0: IMAGES[0]}, CAMERA)
    later = mu.MapView(bare, {IMAGES[5]: cam(90.0)}, {0: IMAGES[5]}, CAMERA)
    assert mu.ids_persistent(m, "a", p, [first, later]) == []
    assert m.items["a"].value == 1.0 and m.items["a"].detail["checks"] == 1
    boxed = final_map(False).objects[:1]  # the monitor, with its box
    m = Metrics()
    assert mu.ids_persistent(m, "b", p, [mu.MapView(boxed, first.poses, first.sources, CAMERA),
                                         mu.MapView(boxed, later.poses, later.sources,
                                                    CAMERA)]) == []
    assert m.items["b"].value == 1.0


# -- performance ------------------------------------------------------------------------------------


def run(tmp_path: Path, tag: str, group: str, wall: float, ok: bool = True,
        notes: dict[str, Any] | None = None) -> RunRecord:
    out = tmp_path / f"{tag}.stdout"
    out.write_bytes(b"")
    return RunRecord(RunSpec(tag, group, "x.sh"), ["x.sh"], 0 if ok else 1, wall, 100.0 + wall,
                     None, out, out, "" if ok else "x.sh: error: boom", notes=notes or {})


def test_a_per_frame_group_takes_the_median_of_the_frames_that_ran(tmp_path: Path) -> None:
    runs = [run(tmp_path, "f1", "segment_frames", 1.0), run(tmp_path, "f2", "segment_frames", 2.5,
                                                             ok=False),
            run(tmp_path, "f3", "segment_frames", 3.0)]
    m = Metrics()
    perf_metrics(m, runs)
    wall = m.items["perf.segment_frames.wall_s"]
    assert wall.value == 2.0 and wall.detail["failed"] == ["f2"]
    assert wall.detail["per_run"]["f2"] == {"wall_s": None, "stages": None}
    assert m.items["perf.segment_frames.client_peak_mb"].value == 103.0  # the frames that ran


def test_a_large_group_reports_its_slowest_run_instead_of_every_run(tmp_path: Path) -> None:
    n = PER_RUN_DETAIL_MAX + 2
    runs = [run(tmp_path, f"f{k}", "segment_frames", 1.0 + k) for k in range(n)]
    m = Metrics()
    perf_metrics(m, runs)
    detail = m.items["perf.segment_frames.wall_s"].detail
    assert "per_run" not in detail and detail["max_wall_s"] == float(n) and detail["runs"] == n


def test_a_failed_update_fails_the_split_maps_time(tmp_path: Path) -> None:
    runs = [run(tmp_path, f"mapper_split_{k}", "mapper_split", 10.0 + k, ok=k != 2)
            for k in (1, 2, 3)]
    m = Metrics()
    perf_metrics(m, runs)
    for k in ("wall_s", "per_update_wall_s"):
        x = m.items[f"perf.mapper_split.{k}"]
        assert x.value is None and x.error == "mapper_split_2 failed (exit 1): x.sh: error: boom"
    assert m.items["perf.mapper_split.wall_s"].detail["runs"] == 3  # the runs stay in the detail
    assert m.items["perf.mapper_split.client_peak_mb"].value == 113.0


def test_a_view_that_never_rendered_has_no_render_time(tmp_path: Path) -> None:
    rec = run(tmp_path, "view_image", "view_image", 4.0,
              notes={"render_s": None, "render_error": "the page failed to load: cloud: 500"})
    m = Metrics()
    perf_metrics(m, [rec])
    x = m.items["perf.view_image.render_s"]
    assert x.value is None and x.error == "the page failed to load: cloud: 500"
    assert m.items["perf.view_image.client_peak_mb"].value == 104.0


# -- scenes -----------------------------------------------------------------------------------------


def test_frames_without_a_pose_in_the_target_frame_are_left_out() -> None:
    T = {"matrix4x4": np.eye(4).reshape(-1).tolist()}

    def tr(dst: str) -> dict[str, Any]:
        return {"src": "camera", "dst": dst, "transform_src_to_dst": T}

    doc = {"openlabel": {"frames": {
        "0": {"frame_properties": {"transforms": {"a": tr("odom"), "b": tr("map")}}},
        "1": {"frame_properties": {"transforms": {"a": tr("odom")}}},
        "2": {}}}}
    assert list(frame_poses(doc)) == [0]
    assert list(frame_poses(doc, "odom")) == [0, 1]
