"""``mapper.sh locate`` in the evaluator, offline: located cameras read from the scene, held-out
pose accuracy against the commanded headings, the PLY pose header, the read-only digest (hidden
entries included), the reference-map runs with a fake mapper, and the street2 video run."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oh_my_slam.core.ply import PointCloud, ply_bytes
from oh_my_slam.core.types import Pose
from oh_my_slam.tools.evaluate.contracts import tree_digest
from oh_my_slam.tools.evaluate.locate import (
    held_out_metrics,
    located_poses,
    located_problems,
    rotation_deg,
)
from oh_my_slam.tools.evaluate.metrics import Metrics, load_targets
from oh_my_slam.tools.evaluate.names import parse_capture
from oh_my_slam.tools.evaluate.runner import Runner
from oh_my_slam.tools.evaluate.suite import EXAMPLES, Evaluation, _pose_lines
from oh_my_slam.tools.evaluate.viewer import BrowserProbe
from tests.unit.test_evaluate_metrics import cam
from tests.unit.test_evaluate_runner import fake_repo, script


def frame(T: Pose, image: str | None = None, located: bool = False) -> dict[str, Any]:
    props: dict[str, Any] = {"transforms": {"c_to_map": {
        "src": "c", "dst": "map", "transform_src_to_dst": {"matrix4x4":
                                                          T.matrix().reshape(-1).tolist()}}}}
    if located:
        props.update(located=True, image=image)
    else:
        props["source"] = image
    return {"frame_properties": props}


def locate_doc(located: dict[str, Pose], mapped: dict[str, Pose] | None = None) -> dict[str, Any]:
    frames = {str(k): frame(p, f"/x/{n}") for k, (n, p) in enumerate((mapped or {}).items())}
    base = len(frames)
    frames.update({str(base + k): frame(p, f"/q/{n}", located=True)
                   for k, (n, p) in enumerate(located.items())})
    return {"openlabel": {"metadata": {}, "frames": frames, "objects": {}}}


HELD = [parse_capture(n) for n in ("011_left_060_level.jpg", "014_left_090_level.jpg",
                                   "017_left_120_level.jpg")]


def test_located_cameras_are_read_and_checked() -> None:
    doc = locate_doc({HELD[0].name: cam(60.0)}, {"001_bootstrap_level.jpg": cam(0.0)})
    poses = located_poses(doc)
    assert list(poses) == [HELD[0].name]  # the map's own frame is not a located one
    assert located_problems(doc, 1) == []
    assert located_problems(doc, 0) == ["1 located frames for 0 input images"]
    del doc["openlabel"]["frames"]["1"]["frame_properties"]["image"]
    assert located_problems(doc, 1) == ["located frame 1 has no image or pose"]


def test_held_out_yaw_against_the_commanded_headings() -> None:
    m = Metrics()
    ref = cam(-20.0)  # the map's own frame 001 (the map's x axis is anywhere)
    located = {HELD[0].name: cam(40.0), HELD[1].name: cam(76.0)}  # 60 and 96 deg left of 001
    later = {HELD[0].name: cam(41.0, t=(0.1, 0, 0))}
    rows = held_out_metrics(m, "pose.locate", located, ref, HELD, later)
    m.judge(load_targets(EXAMPLES / "targets.json"), None)
    v = {k.removeprefix("pose.locate."): x for k, x in m.items.items()}
    assert v["located_fraction"].value == pytest.approx(2 / 3)
    assert v["located_fraction"].passed is False  # one held-out capture was not located
    assert v["yaw_err_median_deg"].value == pytest.approx(3.0)
    assert v["yaw_err_max_deg"].value == pytest.approx(6.0) and v["yaw_err_max_deg"].passed
    assert rows[0]["vs_mapped_rot_deg"] == pytest.approx(1.0, abs=0.01)
    assert rows[0]["vs_mapped_m"] == pytest.approx(0.1)
    assert rows[2] == {"capture": HELD[2].name, "commanded_yaw_deg": 120.0, "located": False}
    m = Metrics()
    held_out_metrics(m, "pose.locate", None, ref, HELD, None)
    assert all("locate failed" in (x.error or "") for x in m.items.values())
    assert rotation_deg(cam(10.0), cam(10.0)) == pytest.approx(0.0, abs=1e-6)


def test_the_ply_pose_header() -> None:
    T = {"matrix4x4": np.eye(4).reshape(-1).tolist()}
    good = ply_bytes(PointCloud(np.zeros((1, 3), np.float32)), comments=[
        f"located_0 {json.dumps({'image': 'a.jpg', 'located': True, 'transform_src_to_dst': T})}",
        f"located_1 {json.dumps({'image': 'b.jpg', 'located': False})}"])
    assert _pose_lines(good, 2) == []
    assert _pose_lines(good, 3) == ["2 located_ header lines for 3 input images"]
    bad = ply_bytes(PointCloud(np.zeros((1, 3), np.float32)), comments=[
        f"located_0 {json.dumps({'located': True})}"])
    assert _pose_lines(bad, 1) == ["located_0: located but no transform"]


def test_the_read_only_digest_sees_hidden_entries(tmp_path: Path) -> None:
    root = tmp_path / "map"
    (root / ".staging").mkdir(parents=True)
    (root / "map.json").write_text("{}")
    (root / ".lock").write_bytes(b"")
    before = tree_digest(root)
    assert tree_digest(root) == before
    for p in root.rglob("*"):  # reading changes nothing
        if p.is_file():
            p.read_bytes()
    assert tree_digest(root) == before
    (root / ".staging" / "part").write_text("x")  # a hidden file appears
    assert tree_digest(root) != before
    before = tree_digest(root)
    t = time.time() + 5
    os.utime(root / ".lock", (t, t))  # touched, same contents
    assert tree_digest(root) != before


def test_locate_on_the_reference_map_with_a_fake_mapper(tmp_path: Path) -> None:
    """The three locate runs (-t single, -t full, -f ply -o) and their contracts."""
    obj = {"1": {"name": "chair 1", "type": "chair", "object_data": {
        "text": [{"name": "color_hex", "val": "#000000"}]}}}
    single_doc = {"openlabel": {"metadata": {}, "objects": obj}}
    full = locate_doc({"001_bootstrap_level.jpg": cam(0.0)})
    full["openlabel"]["objects"] = obj
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "single.json").write_text(json.dumps(locate_doc({"001_bootstrap_level.jpg": cam(0.0)})))
    (docs / "full.json").write_text(json.dumps(full))
    (docs / "out.ply").write_bytes(ply_bytes(PointCloud(np.zeros((1, 3), np.float32)), comments=[
        'located_0 {"located": false}']))
    repo = fake_repo(tmp_path)
    script(repo, "mapper.sh", f'''echo "$@" >> "{tmp_path}/log"
case "$*" in *-f\\ ply*) cp "{docs}/out.ply" "${{@: -1}}";; *-t\\ full*) cat "{docs}/full.json";;
*) cat "{docs}/single.json";; esac''')
    out = tmp_path / "out"
    ev = Evaluation(out, Runner(out, repo), BrowserProbe(None), examples=EXAMPLES)
    ref = tmp_path / "ref"
    ref.mkdir()
    ev.locate_reference(single_doc, ref)
    assert [r.spec.tag for r in ev.runner.records] == ["locate_single", "locate_full", "locate_ply"]
    assert all(r.ok for r in ev.runner.records)
    checks = ev.contracts.checks
    assert checks[("stdout", "mapper")]["locate_ply/pose header lines"] == [
        "1 located_ header lines for 3 input images"]
    assert checks[("same_objects", "map")]["mapper.sh locate -t full vs update -t full"] == []
    assert checks[("openlabel", "mapper")]["locate_single/located frames"] == []
    assert "located frames for 3" not in str(checks[("openlabel", "mapper")])
    assert (tmp_path / "log").read_text().splitlines()[0].startswith("locate -i ")


def test_street2_is_mapped_from_where_it_is(tmp_path: Path) -> None:
    timings = json.dumps({"stages_s": {"inference": 1.0}, "counts": {
        "keyframes_sampled": 10, "keyframes_registered": 9}})
    repo = fake_repo(tmp_path, mapper=f'''echo "$@" > "{tmp_path}/args"
printf '%s' '{timings}' > "$OH_MY_SLAM_TIMINGS"
echo '{{"openlabel": {{}}}}' ''')
    video = tmp_path / "videos" / "street2.mp4"
    out = tmp_path / "out"
    ev = Evaluation(out, Runner(out, repo), BrowserProbe(None), examples=EXAMPLES, street2=video)
    ev.street2_video()  # missing: the metric fails and names the option
    m = ev.metrics.items["pose.street2.registered_fraction"]
    assert m.value is None and "--street2" in (m.error or "") and ev.runner.records == []
    video.parent.mkdir()
    video.write_bytes(b"")
    ev = Evaluation(out, Runner(out, repo), BrowserProbe(None), examples=EXAMPLES, street2=video)
    ev.street2_video()
    m = ev.metrics.items["pose.street2.registered_fraction"]
    assert m.value == 0.9 and m.detail == {"sampled": 10, "registered": 9}
    assert (tmp_path / "args").read_text().split()[:3] == ["update", "-i", str(video)]
    assert ev.runner.records[0].spec.group == "mapper_street2"
