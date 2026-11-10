"""The episode readers: sets, the replay plan, cameras, truth poses, verdicts and --dump-layout."""

from __future__ import annotations

import json
import math
import shutil
from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rpl_episode as epi
from rpl_layout import LayoutError, load_layout, merge

from . import synth

LAYOUT = load_layout(None)


def load(root: Path, layout: dict[str, Any] = LAYOUT) -> epi.Episode:
    return epi.load_episode(root, layout)


def check_truth(ep: epi.Episode, syn: synth.Synth, tol: float = 1e-6) -> None:
    for f in ep.all_frames():
        assert np.allclose(f.truth[:3, 3], syn.truth[f.name][:3, 3], atol=tol), f.name
        assert np.allclose(f.truth[:3, :3], syn.truth[f.name][:3, :3], atol=tol), f.name


def test_base_episode(tmp_path: Path) -> None:
    syn = synth.build(tmp_path)
    ep = load(syn.root)
    assert (ep.robot, ep.kind, ep.camera_name, ep.scene) == (
        "rosmaster_x3_plus", "base", "camera_color_optical_frame", "procthor:7")
    assert ep.floor_z == 0.0 and ep.worktop_z == pytest.approx(0.776)
    chain = synth.files_of(syn, "r1/chain1")
    assert [(e.op, e.set_key, e.files) for e in ep.events] == [
        ("update", "r1/start", None), ("update", "r1/scan1", None),
        ("update", "r1/chain1", chain[:4]), ("locate", "r1/chain1", chain[4:]),
        ("locate", "r1/loc", None), ("update", "r1/wall", None), ("update", "r2/start", None)]
    assert ep.cameras == {"cam0": {"fx": 655.8, "fy": 655.8, "cx": 319.5, "cy": 239.5,
                                   "width": 640, "height": 480}}
    assert {f.truth_source for f in ep.all_frames()} == {"odometry"}
    # file-name times are UTC; the base body's quat is turned 90° from its heading and its
    # odometry over-counts by 8 %: the start pose and the times come from the odometry fit
    check_truth(ep, syn)
    assert all(f.height_m == pytest.approx(0.5) for f in ep.all_frames())
    assert np.allclose(ep.start_poses["r1"], syn.start["r1"], atol=1e-9)
    assert np.allclose(ep.start_poses["r2"], syn.start["r2"], atol=1e-9)  # yaw 180°
    assert ep.odom_scale == {"r1": pytest.approx(1 / synth.ODOM_SCALE),
                             "r2": pytest.approx(1 / synth.ODOM_SCALE)}
    chain = ep.sets["r1/chain1"].frames
    assert chain[1].travel_m - chain[0].travel_m == pytest.approx(0.4 * synth.ODOM_SCALE)
    assert ep.truth_objects(LAYOUT) == [] and ep.passed is True


def test_humanoid_and_arm(tmp_path: Path) -> None:
    hum = synth.build(tmp_path, "e017", "ainex", image="png")
    ep = load(hum.root)
    assert ep.kind == "humanoid" and ep.sets["r1/start"].variant == "humanoid"
    assert ep.sets["r1/start"].frames[1].file.endswith("mp1.png")
    assert {f.truth_source for f in ep.all_frames()} == {"stamp"}
    check_truth(ep, hum)
    sets = [synth.SetSpec("r1", "pp", [(0, 0, 0), (0.3, 0, 10), (0.6, 0.3, 20), (0.2, 0.4, 0)]),
            synth.SetSpec("r1", "look", [(0.1, 0, 12)])]
    arm = synth.build(tmp_path, "e044", "rebot_b601_rs", sets=sets, scene="ithor:1",
                      objects_lines=True)
    # the robot line's pos and quat belong to another body: the start pose is bodies.base_link;
    # the truth is the simulator's camera (a few mm off the kinematics), oriented by gripper_end
    # through the fixed mount rotation estimated from the frames
    ep = load(arm.root)
    assert np.allclose(ep.start_poses["r1"], arm.start["r1"], atol=1e-9)
    check_truth(ep, arm)
    assert ep.mount_rot_spread_deg == pytest.approx(0.0, abs=1e-5)
    vt = epi._views_vs_truth(ep)
    assert vt["pos_m"] == pytest.approx(float(np.linalg.norm(synth.ARM_CAM_ERROR)))
    assert vt["rot_deg"] == pytest.approx(0.0, abs=1e-4)
    # without base_link, the fallback refines the line's pose: shift search, then ICP
    truth = arm.root / "truth.jsonl"
    lines = [json.loads(x) for x in truth.read_text().splitlines() if x.strip()]

    def write(yaw_deg: float) -> None:
        out = []
        for x in lines:
            if x["kind"] == "robot":
                x = {**x, "bodies": {}, "quat": synth.quat_wxyz(synth.rot_z(math.radians(yaw_deg)))}
            out.append(x)
        truth.write_text("\n".join(json.dumps(x) for x in out))

    write(4.0)  # a yaw error too: one ICP iteration does not settle
    one = load(arm.root, merge(LAYOUT, {"refine_iterations": 1}))
    assert one.start_poses["r1"].shape == (4, 4)
    write(0.0)
    ep = load(arm.root)
    assert ep.kind == "arm" and ep.sets["r1/pp"].variant == "arm"
    assert ep.sets["r1/pp"].frames[0].file.endswith("-approach.jpg")
    assert [e.op for e in ep.events] == ["update", "locate"]
    assert all(f.height_m is None for f in ep.all_frames())
    assert {f.truth_source for f in ep.all_frames()} == {"views position"}
    for f in ep.all_frames():  # the fallback has no gripper_end: positions only
        assert np.allclose(f.truth[:3, 3], arm.truth[f.name][:3, 3], atol=1e-6)
    assert ep.mount_rot_spread_deg is None
    assert np.allclose(ep.start_poses["r1"][:3, :3], arm.start["r1"][:3, :3], atol=1e-6)
    assert ep.odom_scale == {"r1": 1.0}
    # read from meta.sim.hello (the arm's real nesting), with the placement
    assert ep.worktop_z == pytest.approx(1.10001) and ep.floor_z == 0.0
    assert ep.placement == "worktop" and ep.scene == "ithor:1"
    objs = {o.label: o for o in ep.truth_objects(LAYOUT)}
    assert objs["apple"].support_z == pytest.approx(1.10001) and objs["box"].support_z is None
    assert epi.dump_layout(arm.root, LAYOUT)["truth"]["meta"]["hello"] == [
        "engine", "floor_z", "rtf", "scene", "starting", "worktop"]


def test_track_matching() -> None:
    t = np.array([0.0, 1.0, 2.0])
    cam = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0.0]])
    R = np.stack([synth.rot_z(0.0), synth.rot_z(0.0), synth.rot_z(math.pi / 2)])
    tr = epi.Track(t, cam, R, cam, [{}, {}, {}])
    assert (tr.index(-5), tr.index(9), tr.index(1.4), tr.index(1.6)) == (0, 2, 1, 2)
    p, tm = tr.match(np.array([1.5, 1, 0]), 10, 20)  # empty window: the window's middle
    assert tm == 15 and np.allclose(p, [2, 0, 0])
    p, tm = tr.match(np.array([9, 9, 9]), 0.5, 1.5)  # one line in the window
    assert tm == 1.0 and np.allclose(p, [1, 0, 0])
    p, tm = tr.match(np.array([1.25, 5, 0]), -1, 3)
    assert tm == 1.0 and np.allclose(p, [1, 0, 0])
    T0 = np.eye(4)
    assert tr.match_odometry(T0, (2.0, 0.0, math.pi / 2), -1, 3, 0.5) == 2.0
    assert tr.match_odometry(T0, (2.0, 0.0, 0.0), -1, 1.5, 0.5) == 1.0
    assert tr.match_odometry(T0, (0.0, 0.0, 0.0), 5, 7, 0.5) == 6.0


def test_plan_order_holdout_and_untimed_sets(tmp_path: Path) -> None:
    syn = synth.build(tmp_path)
    odd = syn.root / "agent_files" / "runs" / "r1" / "zz_odd"
    odd.mkdir(parents=True)
    for name in ("frame.jpg", "extra.jpg", "nodate-20261399-990000.jpg"):
        (odd / name).write_bytes(synth.jpeg(640, 480))
    pose = synth.camera_body("rosmaster_x3_plus", 0, 0, 0).tolist()
    (odd / "views.json").write_text(json.dumps({
        "frame.jpg": {"pose": pose}, "gone.jpg": {"pose": pose},
        "nodate-20261399-990000.jpg": {"pose": pose}}))
    empty = syn.root / "agent_files" / "runs" / "r2" / "aaa_empty"
    empty.mkdir(parents=True)
    (empty / "views.json").write_text("{}")
    lay = merge(LAYOUT, {"replay": {"holdout_every": 2}})
    ep = load(syn.root, lay)
    keys = [(e.op, e.set_key) for e in ep.events]
    # zz_odd has no time: it keeps its folder place after r1/wall (the last r1 set)
    assert keys.index(("update", "r1/zz_odd")) == keys.index(("locate", "r1/wall")) + 1
    assert ("update", "r2/aaa_empty") not in keys
    f = ep.sets["r1/zz_odd"].frames[0]
    assert f.truth_source == "views" and f.name_time is None
    assert ep.sets["r1/zz_odd"].variant == "dict"
    assert any("'gone.jpg' has no image" in w for w in ep.warnings)
    assert any("'extra.jpg' is not in views.json" in w for w in ep.warnings)
    start = [e for e in ep.events if e.set_key == "r1/start"]
    assert [e.op for e in start] == ["update", "locate"] and len(start[1].files or []) == 2
    no_hold = merge(LAYOUT, {"replay": {"holdout_every": 0}})
    assert all(e.op == "update" or e.set_key == "r1/loc" for e in load(syn.root, no_hold).events)
    # a run with no time at all: the first truth line is its start pose
    only = tmp_path / "only"
    synth.build(only)
    for p in (only / "e041" / "agent_files" / "runs" / "r2" / "start").iterdir():
        if p.suffix == ".jpg":
            p.rename(p.with_name(p.name.split("-")[0] + p.name.split("-")[-1]))
    views = only / "e041" / "agent_files" / "runs" / "r2" / "start" / "views.json"
    data = json.loads(views.read_text())
    views.write_text(json.dumps({k.split("-")[0] + k.split("-")[-1]: v for k, v in data.items()}))
    ep = load(only / "e041")
    assert any("run r2: no frame time" in w for w in ep.warnings)


def test_rotation_without_link(tmp_path: Path) -> None:
    syn = synth.build(tmp_path)
    lay = merge(LAYOUT, {"rotation_body": {"base": "no_such_link"}})
    check_truth(load(syn.root, lay), syn)  # the views rotation composed with the start pose
    none = merge(LAYOUT, {"rotation_body": {"base": ""}})  # no link for the kind at all
    check_truth(load(syn.root, none), syn)


def test_camera_table_scaling_exif_and_errors(tmp_path: Path) -> None:
    hum = synth.build(tmp_path, "e017", "ainex")
    folder = hum.root / "agent_files" / "runs" / "r1" / "start"
    for p in folder.glob("*.jpg"):
        p.write_bytes(synth.jpeg(320, 240))
    ep = load(hum.root)
    small = [c for c in ep.cameras.values() if c["width"] == 320]
    assert small == [{"fx": 135.0, "fy": 135.0, "cx": 159.5, "cy": 119.5, "width": 320,
                      "height": 240}]
    assert any("differs from the intrinsics table" in w for w in ep.warnings)
    base = synth.build(tmp_path, "e041")
    p = next((base.root / "agent_files" / "runs" / "r1" / "start").glob("*.jpg"))
    p.write_bytes(synth.jpeg(800, 600, synth.tiff_exif(synth.focal_entries(500.0))))
    ep = load(base.root)
    big = [c for c in ep.cameras.values() if c["width"] == 800]
    assert big == [{"fx": 500.0, "fy": 500.0, "cx": 399.5, "cy": 299.5, "width": 800,
                    "height": 600}]
    no_table = merge(LAYOUT, {"robots": {"rosmaster_x3_plus": {"intrinsics": "exif"}}})
    p.write_bytes(synth.jpeg(640, 480))
    with pytest.raises(LayoutError, match="no EXIF"):
        load(base.root, no_table)
    p.write_bytes(b"junk")
    with pytest.raises(LayoutError, match="not a JPEG"):
        load(base.root)


@pytest.mark.parametrize("damage, match", [
    ("no_runs", "no folder agent_files/runs"),
    ("no_views", "no <run>/<set>/views.json"),
    ("views_list", "keyed by image file name"),
    ("bad_jsonl", "not JSON"),
    ("no_meta", "no line with kind = 'meta'"),
    ("unknown_robot", "not in the layout's robots table"),
    ("no_robot", "no line with kind = 'robot'"),
    ("no_cam", "missing key 'camera_color_optical_frame'"),
])
def test_layout_errors(tmp_path: Path, damage: str, match: str) -> None:
    syn = synth.build(tmp_path)
    root = syn.root
    runs = root / "agent_files" / "runs"
    truth = root / "truth.jsonl"
    lines = [json.loads(x) for x in truth.read_text().splitlines() if x.strip()]
    if damage == "no_runs":
        shutil.rmtree(runs)
    elif damage == "no_views":
        for v in runs.glob("*/*/views.json"):
            v.unlink()
    elif damage == "views_list":
        (runs / "r1" / "start" / "views.json").write_text("[1]")
    elif damage == "bad_jsonl":
        truth.write_text(truth.read_text() + "{oops\n")
    elif damage == "no_meta":
        truth.write_text("\n".join(json.dumps(x) for x in lines if x["kind"] != "meta"))
    elif damage == "unknown_robot":
        lines[0]["robot"] = "other_bot"
        truth.write_text("\n".join(json.dumps(x) for x in lines))
    elif damage == "no_robot":
        truth.write_text("\n".join(json.dumps(x) for x in lines if x["kind"] != "robot"))
    else:
        for x in lines:
            if x["kind"] == "robot":
                del x["cams"]["camera_color_optical_frame"]
        truth.write_text("\n".join(json.dumps(x) for x in lines))
    with pytest.raises(LayoutError, match=match):
        load(root)


def test_verdicts(tmp_path: Path) -> None:
    syn = synth.build(tmp_path)
    v = syn.root / "verdict.md"
    assert epi.read_verdict(syn.root, LAYOUT) is True
    v.write_text("# Verdict\nThe run FAILED: it passed the door but timed out.\n")
    assert epi.read_verdict(syn.root, LAYOUT) is False
    v.write_text("unclear\n")
    assert epi.read_verdict(syn.root, LAYOUT) is None
    v.unlink()
    assert epi.read_verdict(syn.root, LAYOUT) is None
    for value, want in ((True, True), (False, False), ("success", True), ("FAIL", False)):
        (syn.root / "meta.json").write_text(json.dumps({"verdict": value}))
        assert epi.read_verdict(syn.root, LAYOUT) is want
    (syn.root / "meta.json").write_text("[1]")
    assert epi.read_verdict(syn.root, LAYOUT) is None
    (syn.root / "meta.json").unlink()
    assert epi.read_verdict(syn.root, LAYOUT) is None


def test_dump_layout(tmp_path: Path) -> None:
    syn = synth.build(tmp_path, meta_verdict="pass")
    root = syn.root
    full = epi.dump_layout(root, LAYOUT)
    assert full["load"]["events"][2] == "update r1/chain1 (4)"
    # the views over-count by 8 %; the rotations agree
    assert full["load"]["views_vs_truth_m"]["pos_m"] > 0.05
    assert full["load"]["views_vs_truth_m"]["rot_deg"] == pytest.approx(0, abs=1e-4)
    assert full["load"]["odometry_scale"]["r1"] == pytest.approx(1 / synth.ODOM_SCALE)
    start = next(s for s in full["sets"] if s["set"] == "r1/start")
    assert start["value"] == ["base", "pose"] and start["name_time_parsed"] == 4
    assert full["requests_log"] == {"lines": 6, "json_lines": 6, "keys": [
        "command", "id", "is_error", "result", "subagent"], "used": False}
    log = root / "oms" / "agent_requests.jsonl"
    log.write_text(log.read_text() + "not json /home/u/secret\n")
    assert epi.dump_layout(root, LAYOUT)["requests_log"]["json_lines"] == 6
    assert "/home/u" not in json.dumps(full)  # never a raw command line
    assert full["meta_json_keys"] == ["robot", "verdict"] and full["verdict_passed"] is True
    assert full["truth"]["robot"]["bodies"] == ["base_link", "camera_link"]
    (root / "truth.jsonl").unlink()
    (root / "oms" / "agent_requests.jsonl").unlink()
    (root / "meta.json").unlink()
    shutil.rmtree(root / "robot_control")
    runs = root / "agent_files" / "runs" / "r1"
    (runs / "start" / "views.json").write_text("{}")
    (runs / "scan1" / "views.json").write_text(json.dumps({"missing.jpg": [[1]]}))
    views = json.loads((runs / "chain1" / "views.json").read_text())
    (runs / "chain1" / next(iter(views))).write_bytes(b"junk")
    d = epi.dump_layout(root, LAYOUT)
    assert d["truth"].startswith("missing") and d["requests_log"] is None
    assert d["meta_json_keys"] is None and d["robot_control"] == []
    by_set = {s["set"]: s for s in d["sets"]}
    assert "frames" not in by_set["r1/start"]
    assert by_set["r1/scan1"]["value"] == "list of 1" and "size" not in by_set["r1/scan1"]
    assert "image_error" in by_set["r1/chain1"]
    assert "error" in d["load"]


def test_views_vs_truth_without_truth(tmp_path: Path) -> None:
    syn = synth.build(tmp_path)
    ep = load(syn.root)
    for f in ep.all_frames():
        f.truth = None
    assert epi._views_vs_truth(ep) == {"pos_m": None, "rot_deg": None}


def test_name_times_are_utc() -> None:
    from datetime import datetime
    want = datetime(2026, 10, 26, 8, 52, 0, tzinfo=UTC).timestamp()
    assert epi.parse_name_time("c1-20261026-085200.jpg", LAYOUT) == want
    assert epi.parse_name_time("start-20261026-085200mp9.jpg", LAYOUT) == want
    assert epi.parse_name_time("pp-20261026-085200-pregrasp.jpg", LAYOUT) == want
    assert epi.parse_name_time("sheet.jpg", LAYOUT) is None
    assert epi.parse_name_time("x-20261399-990000.jpg", LAYOUT) is None


def test_odometry_fit_fallbacks(tmp_path: Path) -> None:
    # turning in place only: too little spread for the fit, the base quat gives the start pose
    turn = [synth.SetSpec("r1", "scan", [(0, 0, 0), (0, 0, 30), (0, 0, 60)])]
    syn = synth.build(tmp_path, sets=turn)
    ep = load(syn.root)
    assert ep.odom_scale == {"r1": 1.0}
    yaw = math.degrees(math.atan2(ep.start_poses["r1"][1, 0], ep.start_poses["r1"][0, 0]))
    assert yaw == pytest.approx(math.degrees(synth.MODEL_YAW))  # the body's own convention
    assert epi.fit_odometry(ep, ep.sets["r1/scan"].frames[:1], LAYOUT) is None
    for f in ep.all_frames():  # odometry that never moved while the truth did
        f.odom = (0.0, 0.0, 0.0)
    moved = merge(LAYOUT, {"odometry_fit_min_spread_m": 0.0})
    assert epi.fit_odometry(ep, ep.all_frames(), moved) is None


def test_hello_paths() -> None:
    keys = LAYOUT["truth"]
    sim = {"sim": {"hello": {"floor_z": 1.0}}, "hello": {"floor_z": 2.0}}
    assert epi.hello_of(sim, keys) == {"floor_z": 1.0}  # meta.sim.hello first
    assert epi.hello_of({"sim": 3, "hello": {"floor_z": 2.0}}, keys) == {"floor_z": 2.0}
    flat = {"floor_z": 4.0}
    assert epi.hello_of(flat, keys) is flat  # neither: the meta line itself
    assert epi.hello_of({"h": {"x": 1}}, {**keys, "meta_hello": "h"}) == {"x": 1}
