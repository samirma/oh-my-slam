"""The episode readers: sets, request order, cameras, truth poses and --dump-layout."""

from __future__ import annotations

import json
import shutil
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


def test_base_episode(tmp_path: Path) -> None:
    syn = synth.build(tmp_path)
    ep = load(syn.root)
    assert (ep.robot, ep.kind, ep.camera_name, ep.scene) == (
        "rosmaster_x3_plus", "base", "camera_color_optical_frame", "procthor:7")
    assert ep.floor_z == 0.0 and ep.worktop_z == pytest.approx(0.7)
    assert [(e.op, e.set_key) for e in ep.events] == [
        ("update", "r1/start"), ("update", "r1/scan1"), ("update", "r1/chain1"),
        ("locate", "r1/loc"), ("update", "r1/wall"), ("update", "r2/start")]
    assert ep.cameras == {"cam0": {"fx": synth.FX_EXIF, "fy": synth.FX_EXIF, "cx": 319.5,
                                   "cy": 239.5, "width": 640, "height": 480}}
    for f in ep.all_frames():
        assert f.truth_source == "file-name time"
        assert np.allclose(f.truth, syn.truth[f.name])
        assert f.height_m == pytest.approx(0.5)
    assert np.allclose(ep.start_poses["r2"], syn.start["r2"])
    assert ep.truth_objects(LAYOUT) == []
    first = ep.sets["r1/start"].frames[0]
    assert first.nn == 1 and first.time == first.name_time


def test_humanoid_stamp_and_arm(tmp_path: Path) -> None:
    hum = synth.build(tmp_path, "e017", "ainex", image="png")
    ep = load(hum.root)
    assert ep.kind == "humanoid" and ep.sets["r1/start"].variant == "humanoid"
    assert {f.truth_source for f in ep.all_frames()} == {"stamp"}
    for f in ep.all_frames():
        assert np.allclose(f.truth, hum.truth[f.name])
    sets = [synth.SetSpec("r1", "look", [(0, 0, 0), (0.3, 0, 10)])]
    arm = synth.build(tmp_path, "e044", "rebot_b601_rs", sets=sets, scene="ithor:1",
                      objects_lines=True)
    ep = load(arm.root)
    assert ep.kind == "arm" and ep.sets["r1/look"].variant == "arm"
    assert all(f.height_m is None for f in ep.all_frames())
    for f in ep.all_frames():
        assert np.allclose(f.truth, arm.truth[f.name])
    objs = {o.label: o for o in ep.truth_objects(LAYOUT)}
    assert objs["apple"].support_z == pytest.approx(0.7) and objs["box"].support_z is None


def test_track_matching() -> None:
    t = np.array([0.0, 1.0, 2.0])
    cam = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0.0]])
    tr = epi.Track(t, cam, np.stack([np.eye(3)] * 3), cam, [{}, {}, {}])
    assert (tr.index(-5), tr.index(9), tr.index(1.4), tr.index(1.6)) == (0, 2, 1, 2)
    p, tm = tr.match(np.array([1.5, 1, 0]), 10, 20)  # empty window: the window's middle
    assert tm == 15 and np.allclose(p, [2, 0, 0])
    p, tm = tr.match(np.array([9, 9, 9]), 0.5, 1.5)  # one line in the window
    assert tm == 1.0 and np.allclose(p, [1, 0, 0])
    p, tm = tr.match(np.array([1.25, 5, 0]), -1, 3)
    assert tm == pytest.approx(1.25) and np.allclose(p, [1.25, 0, 0])


def test_requests_variants(tmp_path: Path) -> None:
    syn = synth.build(tmp_path)
    loc = synth.files_of(syn, "r1/loc")
    reqs = [
        {"cmd": "mapper.sh update", "images": synth.files_of(syn, "r1/start")[:2]},  # a subset
        {"cmd": "mapper.sh update", "sets": ["scan1"]},  # a bare name inside a list
        {"cmd": "mapper.sh update", "dir": "runs/r1/scan1"},  # the same set again
        {"cmd": "mapper.sh update", "set": "start", "run": "r2"},  # name shared by two runs
        {"cmd": "mapper.sh locate", "images": loc},
        {"cmd": "mapper.sh update", "path": "nothing/here"},
        {"cmd": "segment.sh", "args": ["start"]},
        7,
    ]
    (syn.root / "oms" / "agent_requests.jsonl").write_text(
        "\n".join(json.dumps(r) for r in reqs))
    ep = load(syn.root)
    ev = [(e.op, e.set_key, e.files) for e in ep.events]
    assert ev == [("update", "r1/start", synth.files_of(syn, "r1/start")[:2]),
                  ("update", "r1/scan1", None), ("update", "r2/start", None),
                  ("locate", "r1/loc", None)]
    assert [f.file for f in ep.frames(ep.events[0])] == synth.files_of(syn, "r1/start")[:2]
    assert any("sent again" in w for w in ep.warnings)
    assert any("names no known set" in w for w in ep.warnings)
    no_dedupe = merge(LAYOUT, {"requests": {"dedupe_updates": False}})
    assert len(load(syn.root, no_dedupe).events) == 5
    (syn.root / "oms" / "agent_requests.jsonl").unlink()
    ep = load(syn.root)
    assert [e.set_key for e in ep.events][:2] == ["r1/start", "r1/scan1"]
    assert any("is missing" in w for w in ep.warnings)


def test_untimed_frames_and_warnings(tmp_path: Path) -> None:
    syn = synth.build(tmp_path)
    odd = syn.root / "agent_files" / "runs" / "r3" / "odd"
    odd.mkdir(parents=True)
    (odd / "frame.jpg").write_bytes(synth.jpeg(640, 480, synth.tiff_exif(
        synth.focal_entries(400.0))))
    (odd / "extra.jpg").write_bytes(synth.jpeg(640, 480))
    pose = synth.camera_body("rosmaster_x3_plus", 0, 0, 0).tolist()
    (odd / "views.json").write_text(json.dumps({"frame.jpg": {"pose": pose},
                                                "gone.jpg": {"pose": pose}}))
    lay = merge(LAYOUT, {"rotation_body": {"base": "base_link"}})
    ep = load(syn.root, lay)
    f = ep.sets["r3/odd"].frames[0]
    assert f.truth_source == "views" and f.name_time is None
    assert ep.sets["r3/odd"].variant == "dict"
    assert any("no frame time" in w for w in ep.warnings)
    assert any("'gone.jpg' has no image" in w for w in ep.warnings)
    assert any("'extra.jpg' is not in views.json" in w for w in ep.warnings)
    g = ep.sets["r1/start"].frames[0]
    assert np.allclose(g.truth[:3, :3], np.eye(3))  # the base_link body's orientation


def test_camera_table_scaling_and_exif_errors(tmp_path: Path) -> None:
    hum = synth.build(tmp_path, "e017", "ainex")
    folder = hum.root / "agent_files" / "runs" / "r1" / "start"
    for p in folder.glob("*.jpg"):
        p.write_bytes(synth.jpeg(320, 240))
    ep = load(hum.root)
    cams = list(ep.cameras.values())
    assert {"fx": 135.0, "cx": 159.5, "width": 320}.items() <= cams[-1].items() or \
        {"fx": 135.0, "cx": 159.5, "width": 320}.items() <= cams[0].items()
    assert any("differs from the intrinsics table" in w for w in ep.warnings)
    base = synth.build(tmp_path, "e041")
    p = next((base.root / "agent_files" / "runs" / "r1" / "start").glob("*.jpg"))
    p.write_bytes(synth.jpeg(640, 480))
    with pytest.raises(LayoutError, match="no EXIF"):
        load(base.root)
    p.write_bytes(b"junk")
    with pytest.raises(LayoutError, match="not a JPEG"):
        load(base.root)


@pytest.mark.parametrize("damage, match", [
    ("no_runs", "no folder agent_files/runs"),
    ("no_views", "no <run>/<set>/views.json"),
    ("views_list", "keyed by image file name"),
    ("bad_jsonl", "not JSON"),
    ("no_meta", "no line with kind = 'meta'"),
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
    elif damage == "no_robot":
        truth.write_text("\n".join(json.dumps(x) for x in lines if x["kind"] != "robot"))
    else:
        for x in lines:
            if x["kind"] == "robot":
                del x["cams"]["camera_color_optical_frame"]
        truth.write_text("\n".join(json.dumps(x) for x in lines))
    with pytest.raises(LayoutError, match=match):
        load(root)


def test_dump_layout_degraded(tmp_path: Path) -> None:
    syn = synth.build(tmp_path)
    root = syn.root
    full = epi.dump_layout(root, LAYOUT)
    assert full["load"]["events"][2] == "update r1/chain1"
    assert full["sets"][0]["value"] == ["base", "pose"]
    (root / "truth.jsonl").unlink()
    (root / "oms" / "agent_requests.jsonl").unlink()
    (root / "robot_control" / "rosmaster_x3_plus" / "calibration.md").unlink()
    shutil.rmtree(root / "robot_control")
    runs = root / "agent_files" / "runs" / "r1"
    (runs / "start" / "views.json").write_text("{}")
    (runs / "scan1" / "views.json").write_text(json.dumps({"missing.jpg": [[1]]}))
    bad = next((runs / "chain1").glob("*.jpg"))
    views = json.loads((runs / "chain1" / "views.json").read_text())
    first = next(iter(views))
    (runs / "chain1" / first).write_bytes(b"junk")
    d = epi.dump_layout(root, LAYOUT)
    assert d["truth"].startswith("missing") and d["requests"].startswith("missing")
    assert d["robot_control"] == []
    by_set = {s["set"]: s for s in d["sets"]}
    assert "frames" not in by_set["r1/start"]
    assert by_set["r1/scan1"]["value"] == "list of 1" and "size" not in by_set["r1/scan1"]
    assert "image_error" in by_set["r1/chain1"]
    assert "error" in d["load"]
    assert bad.exists()


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
    assert epi.dump_layout(syn.root, LAYOUT)["verdict_passed"] is None
