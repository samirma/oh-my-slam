"""``mapper.sh locate`` in the evaluator, offline: located cameras read from the scene, held-out
pose accuracy against the commanded headings (or, for a pan-tilt sequence, the later map's), the
PLY pose header, the read-only digest (hidden entries included), the runs on each sequence's
one-update map with a fake mapper, and the street2 video run."""

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
from oh_my_slam.tools.evaluate import service as sv
from oh_my_slam.tools.evaluate.contracts import tree_digest
from oh_my_slam.tools.evaluate.locate import (
    held_out_metrics,
    last_known_problems,
    located_poses,
    located_problems,
    rotation_deg,
)
from oh_my_slam.tools.evaluate.metrics import Metrics, load_targets
from oh_my_slam.tools.evaluate.names import parse_capture, parse_pan_capture
from oh_my_slam.tools.evaluate.runner import Runner
from oh_my_slam.tools.evaluate.suite import (
    AINEX,
    CAMERA,
    EXAMPLES,
    Evaluation,
    _pose_lines,
    locate_picks,
)
from oh_my_slam.tools.evaluate.viewer import BrowserProbe
from tests.unit.test_evaluate_metrics import cam
from tests.unit.test_evaluate_runner import fake_repo, script

# the runs of mapper.sh locate on a sequence's one-update map (suite.locate_reference): with three
# of its captures, then without images (the last known camera); a bird's-eye image after the JSON
# it is checked against
RUNS = ["locate_single", "locate_full", "locate_png_full", "locate_ply", "locate_last_known",
        "locate_last_known_png", "locate_last_known_full", "locate_last_known_png_obb"]


def eye_png(doc: dict[str, Any]) -> bytes:
    """A bird's-eye image that agrees with ``doc``: 1 cm per pixel around the origin, each camera
    of ``doc`` a white pixel where it is, the cloud layer alone (no OBB to check)."""
    import io

    from PIL import Image, PngImagePlugin

    from oh_my_slam.mapping.birdseye import BACKGROUND, CAMERA_FILL
    from oh_my_slam.tools.evaluate.contracts import doc_cameras

    rgb = np.zeros((200, 200, 3), np.uint8)
    rgb[:] = BACKGROUND
    cams = {}
    for key, (_, t) in doc_cameras(doc).items():
        rgb[int(100 - 100 * t[1]), int(100 + 100 * t[0])] = CAMERA_FILL
        cams[key] = {"transform_src_to_dst": {"quaternion": [0, 0, 0, 1], "translation": t.tolist()}}
    info = PngImagePlugin.PngInfo()
    for k, v in {"pixel_to_map": [[0.01, 0, -1], [0, -0.01, 1]],
                 "map_to_pixel": [[100, 0, 100], [0, -100, 100]], "cameras": cams}.items():
        info.add_text(k, json.dumps(v))
    info.add_text("layers", "cloud")
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


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


def test_held_out_pan_captures_against_the_later_maps_headings() -> None:
    """examples/camera records no step angle: a located capture's yaw (relative to the first
    capture's in the map it was located in) is judged against the yaw the split map gives it once
    an update has added it (relative to the first capture's there)."""
    first = "img_009_p03_floor.jpg"
    held = [parse_pan_capture(n) for n in ("img_021_p06_high.jpg", "img_022_p06_mid.jpg",
                                           "img_023_p06_low.jpg")]
    m = Metrics()
    ref = cam(30.0)  # the first map's pose of img_009
    located = {held[0].name: cam(80.0), held[1].name: cam(77.0)}  # 50 and 47 deg left of it
    later = {first: cam(-10.0), held[0].name: cam(38.0), held[1].name: cam(39.0)}  # 48 and 49
    rows = held_out_metrics(m, "pose.camera_locate", located, ref, held, later, first)
    v = {k.removeprefix("pose.camera_locate."): x.value for k, x in m.items.items()}
    assert v["located_fraction"] == pytest.approx(2 / 3)
    assert v["yaw_err_median_deg"] == pytest.approx(2.0) and v["yaw_err_max_deg"] == 2.0
    assert rows[0]["yaw_err_deg"] == 2.0 and "commanded_yaw_deg" not in rows[0]
    assert rows[2] == {"capture": held[2].name, "located": False}
    # without the later map (or its pose of the first capture) there is no heading to judge by
    m = Metrics()
    rows = held_out_metrics(m, "pose.camera_locate", located, ref, held, None, first)
    assert "yaw_err_deg" not in rows[0] and rows[0]["yaw_deg"] == 50.0
    assert m.items["pose.camera_locate.yaw_err_median_deg"].error == (
        "no capture was located with a heading to judge it by")


def test_three_captures_of_each_sequence_are_located_in_its_map() -> None:
    """The first, the one halfway and the one three quarters through: 001, 040 and 060 of
    ainex-captures (the reference map's captures before), and as many of a shorter sequence."""
    ainex = AINEX.read(EXAMPLES / AINEX.folder)
    assert [c.index for c in locate_picks(ainex)] == [1, 40, 60]
    assert [c.name for c in locate_picks(CAMERA.read(EXAMPLES / CAMERA.folder))] == [
        "img_009_p03_floor.jpg", "img_027_p07_mid.jpg", "img_036_p09_high.jpg"]
    assert locate_picks([7]) == [7] and locate_picks([]) == []


def test_the_last_known_camera_is_the_keyframe_added_last() -> None:
    """Without images: one camera marked last_known, naming the keyframe of the latest update
    with the highest frame key, at that keyframe's pose."""
    def kf(T: Pose, name: str, update: int) -> dict[str, Any]:
        f = frame(T, f"/x/{name}")
        f["frame_properties"].update(keyframe=name, update_id=update)
        return f

    map_doc = {"openlabel": {"frames": {"0": kf(cam(0.0), "f000000", 1),
                                        "1": kf(cam(30.0), "f000001", 2),
                                        "2": kf(cam(60.0), "f000002", 1)}}}

    def doc(name: str, T: Pose, *more: dict[str, Any]) -> dict[str, Any]:
        f = frame(T, "frames/x.jpg")
        f["frame_properties"].update(last_known=True, keyframe=name)
        return {"openlabel": {"frames": {"3": f, **{str(4 + k): m for k, m in enumerate(more)}}}}

    assert last_known_problems(doc("f000001", cam(30.0)), map_doc) == []
    assert last_known_problems(doc("f000002", cam(60.0)), map_doc) == [
        "the last known camera names keyframe f000002; the keyframe added last is f000001"]
    assert last_known_problems(doc("f000001", cam(31.0)), map_doc) == [
        "the last known camera's pose is not that of keyframe f000001"]
    moved = cam(30.0, t=(0.0, 0.0, 0.01))
    assert last_known_problems(doc("f000001", moved), map_doc) == [
        "the last known camera's pose is not that of keyframe f000001"]
    no_pose = doc("f000001", cam(30.0))
    no_pose["openlabel"]["frames"]["3"]["frame_properties"].pop("transforms")
    assert last_known_problems(no_pose, map_doc) == [
        "the last known camera's pose is not that of keyframe f000001"]
    twice = doc("f000001", cam(30.0), doc("f000001", cam(30.0))["openlabel"]["frames"]["3"])
    assert last_known_problems(twice, map_doc) == ["2 frames marked last_known, not one"]
    assert last_known_problems({"openlabel": {}}, map_doc) == ["0 frames marked last_known, not one"]
    assert last_known_problems(doc("f000001", cam(30.0)), {"openlabel": {"frames": {}}}) == [
        "the map has no keyframe"]


def test_the_ply_pose_header() -> None:
    """One line per located image (the others are not in the result), each marked located with
    its pose."""
    T = {"matrix4x4": np.eye(4).reshape(-1).tolist()}

    def ply(*lines: str) -> bytes:
        return ply_bytes(PointCloud(np.zeros((1, 3), np.float32)), comments=list(lines))

    good = ply(f"located_1 {json.dumps({'image': 'b.jpg', 'located': True, 'transform_src_to_dst': T})}")
    assert _pose_lines(good, 2) == []  # one of two images located
    assert _pose_lines(good, 0) == ["1 located_ header lines for 0 input images"]
    assert _pose_lines(ply(), 2) == ["0 located_ header lines for 2 input images"]
    bad = ply(f"located_0 {json.dumps({'located': True})}",
              f"located_1 {json.dumps({'located': False, 'transform_src_to_dst': T})}")
    assert _pose_lines(bad, 2) == ["located_0: not marked located with a transform",
                                   "located_1: not marked located with a transform"]


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
    (docs / "eye.png").write_bytes(eye_png(full))  # a bird's-eye image that agrees with full
    repo = fake_repo(tmp_path)
    script(repo, "mapper.sh", f'''echo "$@" >> "{tmp_path}/log"
case "$*" in *-f\\ ply*) cp "{docs}/out.ply" "${{@: -1}}";; *-f\\ png*) cat "{docs}/eye.png";;
*-t\\ full*) cat "{docs}/full.json";; *) cat "{docs}/single.json";; esac''')
    out = tmp_path / "out"
    ev = Evaluation(out, Runner(out, repo), BrowserProbe(None), examples=EXAMPLES)
    ref = tmp_path / "ref"
    ref.mkdir()
    ev.locate_reference(AINEX, single_doc, ref)
    assert [r.spec.tag for r in ev.runner.records] == RUNS
    assert [r.spec.group for r in ev.runner.records] == ["locate", *RUNS[1:]]
    assert all(r.ok for r in ev.runner.records)
    checks = ev.contracts.checks
    assert checks[("stdout", "mapper")]["locate_ply/pose header lines"] == [
        "located_0: not marked located with a transform"]
    # each bird's-eye image against the JSON result of the same call
    eye = checks[("birdseye", "mapper")]
    assert eye["locate_png_full"] == []  # the image agrees with the -t full JSON
    assert eye["locate_last_known_png_obb"] == []  # the same -t full JSON (the fake's)
    assert eye["locate_last_known_png"] == []  # the -t single JSON: the same camera
    # without images: the last known camera, judged against the map (none here)
    assert "NotAMapError" in checks[("openlabel", "mapper")][
        "locate_last_known/last known camera"][0]
    assert checks[("stdout", "mapper")]["locate_png_full"] == []  # one PNG on stdout
    assert checks[("same_objects", "map")]["mapper.sh locate -t full vs update -t full"] == []
    assert checks[("openlabel", "mapper")]["locate_single/located frames"] == []
    assert "located frames for 3" not in str(checks[("openlabel", "mapper")])
    assert (tmp_path / "log").read_text().splitlines()[0].startswith("locate -i ")
    # examples/camera: the same runs on its own map, under its names
    ev.locate_reference(CAMERA, single_doc, ref)
    camera = ev.runner.records[len(RUNS):]
    assert [(r.spec.tag, r.spec.group) for r in camera] == [
        (f"camera_{t}", f"camera_{g}") for t, g in zip(RUNS, ["locate", *RUNS[1:]], strict=True)]
    assert camera[0].argv[3:6] == [str(EXAMPLES / "camera" / n) for n in (
        "img_009_p03_floor.jpg", "img_027_p07_mid.jpg", "img_036_p09_high.jpg")]
    assert checks[("same_objects", "map")][
        "mapper.sh locate -t full vs update -t full (camera)"] == []
    assert checks[("stdout", "mapper")]["camera_locate_ply/pose header lines"] == [
        "located_0: not marked located with a transform"]
    assert (out / "outputs" / "camera_locate.ply").is_file()


@pytest.mark.parametrize("writes", [False, True])
def test_the_reference_map_stays_read_only(tmp_path: Path, writes: bool) -> None:
    """view.sh -m and mapper.sh locate on the reference map must leave its folder as it was,
    hidden entries included (contract.readonly.map)."""
    doc = locate_doc({"001_bootstrap_level.jpg": cam(0.0)})
    (tmp_path / "doc.json").write_text(json.dumps(doc))
    touch = 'touch "${@: -1}/.lock"' if writes else "true"
    repo = fake_repo(tmp_path, mapper=f'''{touch}; cat "{tmp_path}/doc.json"''')
    out = tmp_path / "out"
    single = out / "maps" / "single"
    single.mkdir(parents=True)
    (single / "map.json").write_text("{}")
    ev = Evaluation(out, Runner(out, repo), BrowserProbe(None), examples=EXAMPLES)
    ev.map_commands(AINEX, doc)
    assert [r.spec.tag for r in ev.runner.records] == ["view_map", *RUNS]
    assert ev.contracts.checks[("readonly", "map")] == {
        "view.sh -m, mapper.sh locate": ["the map folder changed"] if writes else []}
    ev = Evaluation(tmp_path / "none", Runner(tmp_path / "none", repo), BrowserProbe(None),
                    examples=EXAMPLES)
    ev.map_commands(AINEX, None)  # no reference map: view.sh -m runs (fails), nothing to compare
    assert [r.spec.tag for r in ev.runner.records] == ["view_map"]
    assert ("readonly", "map") not in ev.contracts.checks


@pytest.mark.parametrize("writes", [False, True])
def test_the_camera_map_stays_read_only_under_view_and_locate(tmp_path: Path,
                                                               writes: bool) -> None:
    """examples/camera's one-update map gets view.sh -m and mapper.sh locate, as the ainex one
    (spec §5: the same uses), which must leave its folder as it was."""
    touch = 'touch "${@: -2:1}/.lock"' if writes else "true"
    repo = fake_repo(tmp_path, view=f"{touch}; exit 1")
    out = tmp_path / "out"
    camera = out / "maps" / "camera_single"
    camera.mkdir(parents=True)
    (camera / "map.json").write_text("{}")
    ev = Evaluation(out, Runner(out, repo), BrowserProbe(None), examples=EXAMPLES)
    ev.map_commands(CAMERA, locate_doc({}))
    assert [(r.spec.tag, r.spec.group) for r in ev.runner.records] == [
        ("view_camera_map", "view_camera_map"), *((f"camera_{t}", f"camera_{g}") for t, g in zip(
            RUNS, ["locate", *RUNS[1:]], strict=True))]
    assert ev.runner.records[0].argv[1:3] == ["-m", str(camera)]
    assert ev.contracts.checks[("readonly", "map")] == {
        "view.sh -m, mapper.sh locate (camera)": ["the map folder changed"] if writes else []}


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
    # into server.sh's workspace (a hard link of the video, not a copy), as the service would
    ws = sv.workspace_root(out)
    placed = ws / "inputs" / "street2" / "street2.mp4"
    assert placed.stat().st_ino == video.stat().st_ino
    assert (tmp_path / "args").read_text().split() == ["update", "-i", str(placed),
                                                       f"-m={ws / 'maps' / 'street2'}"]
    assert ev.runner.records[0].spec.group == "mapper_street2"
    assert ev.street2_ran is None  # not through the recording proxy: no server.sh parity case
    assert [i["name"] for i in ev.parity_inputs()] == [
        "restaurant.jpg", "ainex-captures", "camera", "office_sequence"]
    # through the proxy, the run is the command's result its server.sh request is compared with
    ev = Evaluation(out, Runner(out, repo), BrowserProbe(None), examples=EXAMPLES, street2=video,
                    proxy_env={"OH_MY_SLAM_RUNTIME_DIR": str(tmp_path / "rt")})
    ev.street2_video()
    assert ev.street2_ran is not None and ev.street2_ran.record.ok
    assert ev.street2_ran.argv == ("update", "-i", str(placed), f"-m={ws / 'maps' / 'street2'}")
    assert ev.runner.records[0].spec.env == (("OH_MY_SLAM_RUNTIME_DIR", str(tmp_path / "rt")),)
    last = ev.parity_inputs()[-1]
    assert last == {"name": "street2.mp4", "sequence": [placed], "ran": ev.street2_ran,
                    "writes": "street2"}


def test_the_parity_inputs_are_every_reference_input(tmp_path: Path) -> None:
    """server.sh's reference inputs: restaurant.jpg, and each example sequence with its first
    image, its first images to map, two captures to locate and its one-update map once built."""
    out = tmp_path / "out"
    for name in ("single", "camera_single", "office"):
        (out / "maps" / name).mkdir(parents=True)
        (out / "maps" / name / "map.json").write_text("{}")
    ev = Evaluation(out, Runner(out, fake_repo(tmp_path)), BrowserProbe(None), examples=EXAMPLES)
    restaurant, ainex, camera, office = ev.parity_inputs()
    assert restaurant == {"name": "restaurant.jpg", "image": EXAMPLES / "restaurant.jpg"}
    folder = EXAMPLES / "ainex-captures"
    assert ainex["image"] == folder / "001_bootstrap_level.jpg"
    assert [p.name[:3] for p in ainex["images"]] == ["040", "060"]
    assert [p.name[:3] for p in ainex["sequence"]] == ["001", "002", "003"]
    assert ainex["map"] == out / "maps" / "single"
    assert [p.name for p in camera["images"]] == ["img_027_p07_mid.jpg", "img_036_p09_high.jpg"]
    assert camera["map"] == out / "maps" / "camera_single"
    assert len(office["sequence"]) == 3 and office["map"] == out / "maps" / "office"
    (out / "maps" / "office" / "map.json").unlink()  # not built: no map to locate in
    assert ev.parity_inputs()[3]["map"] is None
    bad = tmp_path / "examples"
    (bad / "camera").mkdir(parents=True)
    (bad / "camera" / "not_a_capture.jpg").write_bytes(b"")
    ev = Evaluation(out, ev.runner, BrowserProbe(None), examples=bad)
    names = {i["name"]: i for i in ev.parity_inputs()}
    assert names["camera"]["image"] is None and names["ainex-captures"]["sequence"] == []
    assert names["office_sequence"]["sequence"] == []
