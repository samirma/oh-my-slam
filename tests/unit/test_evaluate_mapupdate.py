"""The Map update evaluation (spec §5) on synthetic maps: the annotation kind, the remnant test
(label + projection of the box into the images that showed the object), the control map, the
stability of the objects that never changed, and the plan with fake entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oh_my_slam.core.types import Pose
from oh_my_slam.schema import openlabel as ol
from oh_my_slam.segmentation.obb import OBB
from oh_my_slam.tools.evaluate import groundtruth as gt
from oh_my_slam.tools.evaluate.mapupdate import (
    REGION_COVERAGE,
    MapView,
    map_update_metrics,
    metric_ids,
    region_coverage,
    remnants,
    sequence_images,
)
from oh_my_slam.tools.evaluate.metrics import Metrics, load_targets
from oh_my_slam.tools.evaluate.report import build_result, summary_md
from oh_my_slam.tools.evaluate.runner import Runner
from oh_my_slam.tools.evaluate.scene import DocObject
from oh_my_slam.tools.evaluate.suite import EXAMPLES, Evaluation, expected_ids
from oh_my_slam.tools.evaluate.viewer import BrowserProbe
from tests.unit.test_evaluate_metrics import cam
from tests.unit.test_evaluate_runner import fake_repo, script

IMAGES = [f"img{k:02d}.jpg" for k in range(6)]  # the cup is in img00..img02
CAMERA = (500.0, 500.0, 320.0, 240.0, 640, 480)
CUP_REGION = [0.43, 0.40, 0.57, 0.60]  # a 0.4 m box 3 m ahead covers ~60 % of it
ANNOTATION = {"kind": "map_update", "sequence": "office_sequence",
              "absent": [{"label": "cup", "seen_in": {n: CUP_REGION for n in IMAGES[:3]}}]}


def obj(oid: int, label: str, centre: tuple[float, float, float],
        size: tuple[float, float, float] = (0.4, 0.4, 0.4), frames: tuple[int, ...] = (0, 1, 2),
        detected_as: tuple[str, ...] = ()) -> DocObject:
    return DocObject(oid, label, 0.9, None, None,
                     tuple(ol.cuboid_val(np.array(centre), np.eye(3), np.array(size))),
                     frozenset(frames), detected_as)


def view(objects: list[DocObject], images: list[str], yaw: float = 0.0) -> MapView:
    """A map whose cameras all stand at the origin looking along +x (``yaw`` turns them)."""
    poses = {n: cam(yaw) for n in images}
    return MapView(objects, poses, dict(enumerate(images)), CAMERA)


# what never changes: a monitor and a keyboard; the cup 3 m ahead of every early camera
MONITOR = (3.0, 1.5, 0.3)
KEYBOARD = (2.5, -1.2, -0.4)
CUP = (3.0, 0.0, 0.0)


def early_map(with_cup: bool = True, label: str = "cup") -> MapView:
    objs = [obj(1, "monitor", MONITOR, (0.6, 0.2, 0.4)), obj(2, "keyboard", KEYBOARD)]
    return view(objs + ([obj(3, label, CUP)] if with_cup else []), IMAGES[:3])


def final_map(with_cup: bool, label: str = "cup", cup_at: tuple[float, float, float] = CUP,
              **kwargs: Any) -> MapView:
    objs = [obj(1, "monitor", MONITOR, (0.6, 0.2, 0.4), frames=(0, 1, 2, 5)),
            obj(2, "keyboard", KEYBOARD, frames=(0, 5)), obj(4, "backpack", (-2.0, 1.0, 0.0),
                                                              frames=(4, 5))]
    if with_cup:
        objs.append(obj(3, label, cup_at, **kwargs))
    return view(objs, IMAGES)


def plan(tmp_path: Path, **extra: Any) -> gt.MapUpdatePlan:
    (tmp_path / "gt").mkdir(exist_ok=True)
    (tmp_path / "gt" / "a.json").write_text(json.dumps({**ANNOTATION, **extra}))
    files, skipped = gt.discover(tmp_path / "gt")
    assert skipped == []
    found = gt.map_update_plan(files, skipped)
    assert found is not None
    return found


def judged(tmp_path: Path, final: MapView, before: MapView, extended: MapView | None = None,
           **extra: Any) -> tuple[Metrics, Any]:
    """``final``: the whole sequence in one update; ``before``: the extended map after its first
    update; ``extended``: after its second (default: the same as ``final``)."""
    m = Metrics()
    details = map_update_metrics(m, plan(tmp_path, **extra), IMAGES, final, before,
                                 final if extended is None else extended)
    m.judge(load_targets(EXAMPLES / "targets.json"), None)
    return m, details


# -- the annotation kind -------------------------------------------------------------------------------


def test_map_update_files_are_discovered_and_merged(tmp_path: Path) -> None:
    folder = tmp_path / "ground_truth"
    folder.mkdir()
    (folder / "a.json").write_text(json.dumps({**ANNOTATION, "stable": ["monitor"]}))
    (folder / "b.json").write_text(json.dumps({
        "kind": "map_update", "sequence": "office_sequence",
        "absent": [{"label": "book", "seen_in": {IMAGES[4]: [0.1, 0.1, 0.2, 0.2]}}],
        "stable": ["monitor", "keyboard"]}))
    (folder / "other.json").write_text(json.dumps({**ANNOTATION, "sequence": "ainex-captures"}))
    bad = {"kind": "map_update", "sequence": "office_sequence"}
    (folder / "bad.json").write_text(json.dumps(bad))
    (folder / "bad2.json").write_text(json.dumps({**ANNOTATION, "absent": [
        {"label": "cup", "seen_in": {"x.jpg": [0.5, 0.1, 0.4, 0.2]}}]}))  # x0 > x1
    files, skipped = gt.discover(folder)
    assert sorted(f.path.name for f in files) == ["a.json", "b.json", "other.json"]
    assert sorted(Path(s["file"]).name for s in skipped) == ["bad.json", "bad2.json"]
    merged = gt.map_update_plan(files, skipped)
    assert merged is not None
    assert [a.label for a in merged.absent] == ["cup", "book"]
    assert merged.stable == ["monitor", "keyboard"]
    assert [p.name for p in merged.files] == ["a.json", "b.json"]
    assert any("ainex-captures is not evaluated" in s["reason"] for s in skipped)
    # the early part ends at the last image that shows any absent object
    assert merged.before_images(IMAGES) == IMAGES[:5]
    assert gt.map_update_plan([], []) is None


def test_the_shipped_annotation_is_valid_and_names_images_of_the_sequence() -> None:
    files, skipped = gt.discover(EXAMPLES / "ground_truth")
    assert skipped == []
    found = gt.map_update_plan(files, skipped)
    assert found is not None and [a.label for a in found.absent] == ["cup"]
    images = sequence_images(EXAMPLES / found.sequence)
    assert len(images) == 13
    assert all(n in images for a in found.absent for n in a.seen_in)
    assert found.before_images(images) == images[:4]  # the cup is gone from the fifth image on


def test_every_map_update_metric_has_a_target() -> None:
    targets = load_targets(EXAMPLES / "targets.json")
    assert set(metric_ids()) <= set(targets) and set(metric_ids()) <= set(expected_ids())
    assert targets["map_update.absent_fraction"].op == ">="
    assert targets["map_update.absent_fraction"].value == 1.0


# -- the remnant test ----------------------------------------------------------------------------------


def test_region_coverage_of_a_projected_box() -> None:
    box = OBB(np.array(CUP), np.eye(3), np.array([0.4, 0.4, 0.4]))
    pose = cam(0.0)
    region = (0.43, 0.40, 0.57, 0.60)
    # the box's near face (2.8 m) spans 500 * 0.4 / 2.8 = 71 px of the region's 90 x 96 px
    assert region_coverage(box, pose, CAMERA, region) == pytest.approx(0.59, abs=0.02)
    elsewhere = OBB(np.array([3.0, 1.0, 0.0]), np.eye(3), np.array([0.4, 0.4, 0.4]))
    assert region_coverage(elsewhere, pose, CAMERA, region) == 0.0
    assert region_coverage(box, cam(180.0), CAMERA, region) is None  # behind the camera
    assert region_coverage(box, cam(0.0, t=(2.9, 0.0, 0.0)), CAMERA, region) is None  # inside


def test_a_cup_still_in_the_final_map_is_a_remnant_with_the_image_it_covers() -> None:
    item = gt.Absent("cup", {IMAGES[0]: (0.43, 0.40, 0.57, 0.60)})
    rows = remnants(final_map(True), item)
    assert [(r["id"], r["label"], r["localised"], r["image"]) for r in rows] == [
        (3, "cup", True, IMAGES[0])]
    assert rows[0]["coverage"] >= REGION_COVERAGE and rows[0]["frames"] == IMAGES[:3]
    assert remnants(final_map(False), item) == []


def test_remnants_follow_the_map_poses_not_the_map_origin() -> None:
    """The whole final map (cameras and objects) turned about the origin: same result."""
    item = gt.Absent("cup", {IMAGES[0]: (0.43, 0.40, 0.57, 0.60)})
    yaw = np.array([[np.cos(np.radians(37)), -np.sin(np.radians(37)), 0],
                    [np.sin(np.radians(37)), np.cos(np.radians(37)), 0], [0, 0, 1.0]])
    base = final_map(True)
    turned = MapView(
        [DocObject(o.id, o.label, o.score, None, None,
                   tuple(ol.cuboid_val(yaw @ np.array(o.cuboid[:3]), yaw, np.array(o.cuboid[7:]))),
                   o.frames, o.labels) for o in base.objects],
        {n: Pose(yaw @ p.R, yaw @ p.t) for n, p in base.poses.items()}, base.sources, CAMERA)
    assert [r["id"] for r in remnants(turned, item)] == [3]
    assert remnants(view([obj(3, "cup", (3.0, 1.5, 0.0))], IMAGES), item) == []  # elsewhere


def test_a_compatible_or_detected_as_label_counts_and_other_labels_do_not() -> None:
    item = gt.Absent("cup", {IMAGES[0]: (0.43, 0.40, 0.57, 0.60)})
    assert [r["id"] for r in remnants(final_map(True, label="mug"), item)] == [3]
    assert [r["id"] for r in remnants(final_map(True, label="vase",
                                                detected_as=("bottle", "cup")), item)] == [3]
    assert remnants(final_map(True, label="vase"), item) == []  # something else on that spot
    # a "cup" whose box stands elsewhere is not the absent cup
    assert remnants(final_map(True, cup_at=(3.0, 1.5, 0.0)), item) == []


def test_without_a_registered_image_the_label_alone_decides() -> None:
    item = gt.Absent("cup", {"unregistered.jpg": (0.43, 0.40, 0.57, 0.60)})
    rows = remnants(final_map(True, cup_at=(3.0, 1.5, 0.0)), item)
    assert [(r["id"], r["localised"]) for r in rows] == [(3, False)]


# -- the metrics ---------------------------------------------------------------------------------------


def test_a_map_without_the_cup_that_kept_the_rest_passes(tmp_path: Path) -> None:
    m, details = judged(tmp_path, final_map(False), early_map())
    assert {k: v.value for k, v in m.items.items() if not k.startswith("map_update.stability")} == {
        "map_update.absent_fraction": 1.0, "map_update.incremental.absent_fraction": 1.0,
        "map_update.before_present_fraction": 1.0}
    s = {k.split(".")[-1]: v.value for k, v in m.items.items() if ".stability." in k}
    assert s["matched_fraction"] == 1.0 and s["label_agreement"] == 1.0
    assert s["id_agreement"] == 1.0 and s["centre_delta_median_m"] == pytest.approx(0.0)
    assert s["obb_iou_median"] == pytest.approx(1.0)
    assert set(m.items) == set(metric_ids())
    assert all(v.passed for v in m.items.values())
    # the backpack (only seen late) and the cup are not compared: two objects in the stability rows
    assert {(r["single_id"], r["split_id"]) for r in details["stability"]} == {(1, 1), (2, 2)}
    assert details["before_images"] == IMAGES[:3]


def test_a_map_that_still_has_the_cup_fails_with_the_object_to_look_at(tmp_path: Path) -> None:
    m, details = judged(tmp_path, final_map(True), early_map())
    absent = m.items["map_update.absent_fraction"]
    assert absent.value == 0.0 and absent.passed is False
    (left,) = absent.detail["present_in_the_map"]
    assert left["label"] == "cup" and left["objects"][0]["id"] == 3
    assert left["objects"][0]["image"] == IMAGES[0]
    assert m.items["map_update.incremental.absent_fraction"].passed is False
    assert m.items["map_update.before_present_fraction"].passed is True
    # the stale cup is left out of the stability comparison: the unchanged objects still pass
    assert m.items["map_update.stability.id_agreement"].passed is True
    assert details["absent"][0]["final"][0]["id"] == 3


def test_the_extended_map_is_judged_apart_from_the_one_update_map(tmp_path: Path) -> None:
    """The update that adds the later images has to drop the cup too, not only the map that
    saw the whole sequence at once."""
    m, details = judged(tmp_path, final_map(False), early_map(), extended=final_map(True))
    assert m.items["map_update.absent_fraction"].passed is True
    assert m.items["map_update.incremental.absent_fraction"].value == 0.0
    assert details["absent"][0]["final"] == [] and details["absent"][0]["incremental"][0]["id"] == 3


def test_an_empty_map_does_not_count_as_a_map_without_the_cup(tmp_path: Path) -> None:
    empty = view([], IMAGES)
    m, _ = judged(tmp_path, final_map(False), early_map(), extended=empty)
    assert m.items["map_update.absent_fraction"].passed is True
    absent = m.items["map_update.incremental.absent_fraction"]
    assert absent.value is None and absent.passed is False and "no objects at all" in absent.error
    assert m.items["map_update.stability.matched_fraction"].passed is False  # nothing kept
    m, _ = judged(tmp_path, empty, early_map())
    assert m.items["map_update.absent_fraction"].passed is False


def test_a_control_map_without_the_cup_makes_the_absence_meaningless(tmp_path: Path) -> None:
    m, _ = judged(tmp_path, final_map(False), early_map(with_cup=False))
    assert m.items["map_update.absent_fraction"].passed is True
    control = m.items["map_update.before_present_fraction"]
    assert control.passed is False and control.detail == {"missing_before": ["cup"]}


def test_unchanged_objects_that_lost_their_id_label_or_box_are_flagged(tmp_path: Path) -> None:
    final = final_map(False)
    changed = [obj(9, "tv", (MONITOR[0] + 0.2, MONITOR[1], MONITOR[2]), (0.6, 0.2, 0.4),
                   frames=(0, 1, 2)),  # new id, new label, moved by 0.2 m
               obj(2, "keyboard", (KEYBOARD[0] + 0.2, KEYBOARD[1], KEYBOARD[2]),
                   frames=(0, 5))]  # the box moved by 0.2 m
    final = MapView(changed, final.poses, final.sources, CAMERA)
    m, details = judged(tmp_path, final, early_map(with_cup=False))
    v = {k.split(".")[-1]: x for k, x in m.items.items() if ".stability." in k}
    assert v["id_agreement"].value < 1.0 and v["id_agreement"].passed is False
    assert v["centre_delta_median_m"].value > 0.1 and v["centre_delta_median_m"].passed is False
    assert {(r["single_id"], r["split_id"]) for r in details["stability"]} == {(9, 1), (2, 2)}


def test_stable_labels_restrict_the_comparison(tmp_path: Path) -> None:
    m, details = judged(tmp_path, final_map(False), early_map(), stable=["monitor"])
    assert {(r["single_id"], r["split_id"]) for r in details["stability"]} == {(1, 1)}
    assert m.items["map_update.stability.matched_fraction"].value == 1.0


def test_the_two_updates_of_one_map_are_compared_in_their_common_frame(tmp_path: Path) -> None:
    other = view([obj(1, "monitor", MONITOR), obj(2, "keyboard", KEYBOARD)],
                 [f"other{k}.jpg" for k in range(3)])  # registers other images, same frame
    m, _ = judged(tmp_path, final_map(False), other)
    assert m.items["map_update.absent_fraction"].value == 1.0
    assert m.items["map_update.stability.id_agreement"].value == 1.0  # one map frame: no alignment


# -- the plan with fake entry points ---------------------------------------------------------------------


def scene_doc(v: MapView, name: str) -> dict[str, Any]:
    """The map's ``-t full`` scene as the evaluator reads it (frames carry their source)."""
    frames = {}
    for key, image in v.sources.items():
        T = v.poses[image].matrix()
        frames[str(key)] = {"frame_properties": {
            "source": image, "transforms": {"camera_1_to_map": {
                "src": "camera_1", "dst": "map",
                "transform_src_to_dst": {"matrix4x4": T.reshape(-1).tolist()}}}}}
    fx, fy, cx, cy, w, h = CAMERA
    streams = {"camera_1": {"type": "camera", "stream_properties": {"intrinsics_pinhole": {
        "width_px": w, "height_px": h, "camera_matrix": [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]}}}}
    objects = {}
    for o in v.objects:
        data: dict[str, Any] = {"cuboid": [{"name": "obb", "val": list(o.cuboid or ())}]}
        if o.labels:
            data["vec"] = [{"name": "detected_as", "val": list(o.labels)}]
        objects[str(o.id)] = {"name": f"{o.label} {o.id}", "type": o.label, "object_data": data,
                              "frame_intervals": [{"frame_start": f, "frame_end": f}
                                                  for f in sorted(o.frames)]}
    return {"openlabel": {"metadata": {"name": name}, "streams": streams, "frames": frames,
                          "objects": objects}}


def office_examples(tmp_path: Path) -> Path:
    """An examples folder with a sequence of empty images and the annotation."""
    root = tmp_path / "examples"
    (root / "office_sequence").mkdir(parents=True)
    for n in IMAGES:
        (root / "office_sequence" / n).write_bytes(b"")
    (root / "office_sequence" / ".DS_Store").write_bytes(b"")
    (root / "ground_truth").mkdir()
    (root / "ground_truth" / "office.json").write_text(json.dumps(ANNOTATION))
    return root


def office_repo(tmp_path: Path, final: MapView, early: MapView) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "final.json").write_text(json.dumps(scene_doc(final, "final")))
    (docs / "early.json").write_text(json.dumps(scene_doc(early, "early")))
    repo = fake_repo(tmp_path)
    script(repo, "mapper.sh", f"""echo "$@" >> "{tmp_path}/mapper.log"
case "$*" in *img03*) cat "{docs}/final.json";; *office_extended*) cat "{docs}/early.json";;
*) cat "{docs}/final.json";; esac""")
    return repo


def run_plan(tmp_path: Path, final: MapView, early: MapView) -> Evaluation:
    out = tmp_path / "out"
    ev = Evaluation(out, Runner(out, office_repo(tmp_path, final, early)), BrowserProbe(None),
                    examples=office_examples(tmp_path))
    ev.map_update()
    ev.metrics.judge(load_targets(EXAMPLES / "targets.json"), None)
    return ev


def test_the_plan_maps_the_sequence_whole_and_extended_then_judges(tmp_path: Path) -> None:
    ev = run_plan(tmp_path, final_map(False), early_map())
    log = (tmp_path / "mapper.log").read_text().splitlines()
    root = tmp_path / "examples" / "office_sequence"
    out = tmp_path / "out" / "maps"
    assert log[0] == (f"update -i {' '.join(str(root / n) for n in IMAGES)} "
                      f"-m {out / 'office'}")  # one update, the sorted files
    assert log[1] == (f"update -i {' '.join(str(root / n) for n in IMAGES[:3])} "
                      f"-m {out / 'office_extended'}")  # the early images, then the rest
    assert log[2] == (f"update -i {' '.join(str(root / n) for n in IMAGES[3:])} "
                      f"-m {out / 'office_extended'}")
    assert set(ev.metrics.items) == set(metric_ids())
    assert all(m.passed for m in ev.metrics.items.values())
    assert ev.details["map_update"]["before_images"] == IMAGES[:3]
    assert [r.spec.tag for r in ev.runner.records] == [
        "mapper_office", "mapper_office_early", "mapper_office_rest"]


def test_the_plan_reports_the_cup_left_in_the_map(tmp_path: Path) -> None:
    ev = run_plan(tmp_path, final_map(True), early_map())
    assert ev.metrics.items["map_update.absent_fraction"].passed is False
    result = build_result(ev.metrics, ev.runner.records, ev.details, started="s", finished="f",
                          duration_s=1.0, env={}, targets=EXAMPLES / "targets.json",
                          baseline={"status": "missing", "path": "b"})
    text = summary_md(result)
    assert "## Map update\n" in text and "map_update.absent_fraction | 0 | >= 1" in text
    assert "Map update: objects of the absent labels in the map of the whole sequence" in text
    assert "Map update: objects of the absent labels in the extended map (second update)" in text
    assert "| cup | 3 | cup |" in text and IMAGES[0] in text
    assert "Map update: objects that never changed" in text


def test_without_an_annotation_or_maps_the_metrics_fail_with_the_reason(tmp_path: Path) -> None:
    out = tmp_path / "out"
    root = office_examples(tmp_path)
    (root / "ground_truth" / "office.json").unlink()
    repo = fake_repo(tmp_path)
    ev = Evaluation(out, Runner(out, repo), BrowserProbe(None), examples=root)
    ev.map_update()
    assert ev.runner.records == []  # nothing is mapped without an annotation
    m = ev.metrics.items["map_update.absent_fraction"]
    assert m.value is None and "no 'map_update' file" in (m.error or "")
    (root / "ground_truth" / "office.json").write_text(json.dumps(ANNOTATION))
    ev = Evaluation(out / "2", Runner(out / "2", repo), BrowserProbe(None), examples=root)
    ev.map_update()  # the mapper fails
    assert [r.ok for r in ev.runner.records] == [False, False]  # no second update after a failure
    assert "was not built" in (ev.metrics.items["map_update.stability.id_agreement"].error or "")
