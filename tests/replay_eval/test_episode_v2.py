"""The skill-v2 layout reader, on fixtures shaped like the conductor's e051."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import replay_eval
import rpl_episode as epi
import rpl_episode_v2 as v2
from rpl_layout import LayoutError, load_layout, merge

from . import synth, synth_v2
from .fake_server import Config, Fake, Sim

LAYOUT = load_layout(None)


def run_dir(s: synth_v2.SynthV2) -> Path:
    return s.root / "agent_files" / "runs" / "r1"


def test_reader_e051_shape(tmp_path: Path) -> None:
    s = synth_v2.build_v2(tmp_path)
    assert epi.is_v2(s.root, LAYOUT)
    ep = epi.load_episode(s.root, LAYOUT)
    assert ep.version == 2 and ep.kind == "base"
    assert ep.cameras == {"astra": {"fx": 655.8, "fy": 655.8, "cx": 319.5, "cy": 239.5,
                                    "width": 640, "height": 480}}  # calibrate_camera.json
    assert ep.meta["camera_sources"]["astra"].endswith("calibrate_camera.json")
    look, probe, loc = s.names["look"], s.names["probe"], s.names["loc"]
    ev = [(e.op, [Path(f).stem for f in e.files or []], e.accept_if, e.job_id, e.measured_s)
          for e in ep.events]
    assert ev == [("update", look, {"max_gauge_shift_m": 0.05}, "2dab2e4523d4486a", 4.958),
                  ("update", probe, {"max_gauge_shift_m": 0.05, "max_rejected_share": 0.5},
                   "77aa", 6.1),
                  ("locate", loc, None, None, None)]  # close001 is in no job and not loc
    assert any("'ghost-20261110-000000' has no sidecar" in w for w in ep.warnings)
    for f in ep.all_frames():
        stem = Path(f.file).stem
        assert f.truth_source == "stamp" and f.camera_id == "astra"
        assert np.allclose(f.truth, s.truth[stem], atol=1e-9), stem
        assert f.depth_path is not None and f.depth_path.name == f"{stem}.depth.png"
        assert f.height_m == 0.5 and f.height_sigma_m is None
    close = next(f for f in ep.all_frames() if f.file.startswith("close"))
    assert close.prior is None and not close.has_pose
    assert ep.odom_scale["r1"] == pytest.approx(1 / synth.ODOM_SCALE)  # biased priors
    d = epi.dump_layout(s.root, LAYOUT)
    r = d["runs_v2"][0]
    assert d["layout_version"] == 2 and r["skipped"] == ["_select.json"]
    assert r["by_prefix"] == {"close": 1, "loc": 2, "look": 4, "probe": 4}
    assert r["intrinsics"] == {"astra": "calibrate_camera.json"} and r["with_depth"] == 11
    assert r["jobs"]["n"] == 2 and r["log"] == {"lines": 3, "keys": [
        "by", "code", "command", "script", "seconds", "status", "t"]}
    assert "/home/u" not in json.dumps(d["runs_v2"])
    assert d["load"]["frames"] == 11


def test_intrinsics_order(tmp_path: Path) -> None:
    info = synth_v2.build_v2(tmp_path / "a", camera="camera_info")
    cam = epi.load_episode(info.root, LAYOUT).cameras["astra"]
    assert cam == {"fx": 600.0, "fy": 601.0, "cx": 320.0, "cy": 240.0, "width": 640,
                   "height": 480, "distortion": {"model": "opencv",
                                                 "coefficients": [0.1, -0.05, 0.0, 0.0, 0.0]}}
    # calibrate_camera.json of another size: camera_info wins
    cal = synth_v2.build_v2(tmp_path / "b")
    (run_dir(cal) / "calibrate_camera.json").write_text(json.dumps(
        {"intrinsics": {**synth_v2.E051_CAMERA, "width_px": 1280, "height_px": 720}}))
    assert epi.load_episode(cal.root, LAYOUT).cameras["astra"]["fx"] == 600.0
    none = synth_v2.build_v2(tmp_path / "c", camera="none")
    with pytest.raises(LayoutError, match=r"camera 'astra' \(640x480\): no intrinsics: .*"
                                          r"calibrate_camera.json is missing.*camera_info_astra"
                                          r".json is missing, no .*robot_control/<robot>/calibration"
                                          r"\.json lists 'astra' with intrinsics, and no "
                                          r"--camera"):
        epi.load_episode(none.root, LAYOUT)
    over = merge(LAYOUT, {"camera_override": {"fx": 500.0, "fy": 500.0, "cx": 320.0,
                                              "cy": 240.0, "width": 640, "height": 480}})
    ep = epi.load_episode(none.root, over)
    assert ep.cameras["astra"]["fx"] == 500.0 and ep.meta["camera_sources"]["astra"] == "--camera"
    assert "error:" in epi.dump_layout(none.root, LAYOUT)["runs_v2"][0]["intrinsics"]["astra"]
    zero = run_dir(info) / "camera_info_astra.json"
    zero.write_text(json.dumps({"msg": {"K": [1, 0, 2, 0, 3, 4, 0, 0, 1], "width": 640,
                                        "height": 480, "D": [0, 0, 0, 0, 0],
                                        "distortion_model": "plumb_bob"}}))
    assert "distortion" not in v2.camera_from_info(zero)  # type: ignore[operator]
    zero.write_text(json.dumps({"msg": {"K": [1, 2], "width": 1, "height": 1}}))
    with pytest.raises(LayoutError, match="key 'K' must hold 9 numbers"):
        v2.camera_from_info(zero)
    (run_dir(cal) / "calibrate_camera.json").write_text(json.dumps({"intrinsics": {"fx_px": 1}}))
    with pytest.raises(LayoutError, match=r"calibrate_camera\.json intrinsics: missing key 'fy_px'"):
        epi.load_episode(cal.root, LAYOUT)


def test_sidecar_and_jobs_errors(tmp_path: Path) -> None:
    s = synth_v2.build_v2(tmp_path)
    frames = run_dir(s) / "frames"
    side = frames / f"{s.names['loc'][0]}.json"
    data = json.loads(side.read_text())

    def write(d: Any) -> None:
        side.write_text(json.dumps(d))

    write({**data, "depth": None, "prior": {**data["prior"], "sigma_rot_deg": None}})
    f = next(f for f in epi.load_episode(s.root, LAYOUT).all_frames() if f.file == data["image"])
    assert f.depth_path is None and f.prior["sigma_rot_deg"] is None  # type: ignore[index]
    write({**data, "depth": "gone.depth.png", "name": None})
    ep = epi.load_episode(s.root, LAYOUT)
    f = next(f for f in ep.all_frames() if f.file == data["image"])
    assert f.depth_path is None and Path(f.file).stem == side.stem
    write({**data, "image": "nope.png"})
    with pytest.raises(LayoutError, match=r"key 'image' names 'nope.png'"):
        epi.load_episode(s.root, LAYOUT)
    write({**data, "stamp": None})
    with pytest.raises(LayoutError, match=r"loc001-.*\.json: key 'stamp' is missing or null"):
        epi.load_episode(s.root, LAYOUT)
    write([1])
    with pytest.raises(LayoutError, match="expected a JSON object"):
        epi.load_episode(s.root, LAYOUT)
    (frames / data["image"]).write_bytes(b"junk")
    write(data)
    with pytest.raises(LayoutError, match="not a JPEG or PNG"):
        epi.load_episode(s.root, LAYOUT)
    (frames / data["image"]).write_bytes(synth.png(640, 480))
    jobs = run_dir(s) / "oms" / "jobs.json"
    jobs.write_text(json.dumps({"jobs": 3}))
    with pytest.raises(LayoutError, match="key 'jobs' must be a list"):
        epi.load_episode(s.root, LAYOUT)
    js = [{"frames": ["nobody"], "submitted_at": "not a time"},
          {"frames": s.names["look"], "submitted_at": None}]
    jobs.write_text(json.dumps(js))  # a bare list; a job with no known frame
    ep = epi.load_episode(s.root, LAYOUT)
    assert [e.op for e in ep.events] == ["update", "locate"]
    assert ep.events[0].measured_s is None and ep.events[0].job_id is None
    jobs.unlink()
    ep = epi.load_episode(s.root, LAYOUT)
    assert [e.op for e in ep.events] == ["locate"] and any("nothing to update" in w
                                                         for w in ep.warnings)
    assert epi.dump_layout(s.root, LAYOUT)["runs_v2"][0]["jobs"]["found"] is False
    assert v2.parse_time(5) == 5.0 and v2.parse_time("2026-11-10T15:59:23Z") == pytest.approx(
        1794326363.0) and v2.parse_time("x") is None and v2.parse_time([1]) is None


def test_runs_cameras_truth_fallbacks(tmp_path: Path) -> None:
    s = synth_v2.build_v2(tmp_path)
    # a second run with another camera calibration under the same id
    src, dst = run_dir(s), run_dir(s).parent / "r2"
    dst.mkdir()
    (dst / "frames").mkdir()
    one = s.names["look"][0]
    for suffix in (".png", ".depth.png", ".json"):
        (dst / "frames" / f"{one}{suffix}").write_bytes((src / "frames" / f"{one}{suffix}")
                                                        .read_bytes())
    (dst / "calibrate_camera.json").write_text(json.dumps(
        {"intrinsics": {**synth_v2.E051_CAMERA, "fx_px": 700.0}}))
    ep = epi.load_episode(s.root, LAYOUT)
    assert ep.cameras["astra@r2"]["fx"] == 700.0 and ep.sets["r2/frames"].frames[0].camera_id \
        == "astra@r2"
    assert any("differs from an earlier run's" in w for w in ep.warnings)
    # no truth link and a frame without a prior: identity, with a warning; one prior only:
    # the start pose comes from the base quat; a start link wins when present
    lay = merge(LAYOUT, {"rotation_body": {"base": ""}})
    ep = epi.load_episode(s.root, lay)
    assert any("identity orientation" in w for w in ep.warnings)
    assert np.allclose(ep.sets["r2/frames"].frames[0].truth[:3, 3], s.truth[one][:3, 3])
    lay = merge(LAYOUT, {"start_link": {"base": "base_link"}})
    ep = epi.load_episode(s.root, lay)
    assert np.allclose(ep.start_poses["r1"], synth.make_T(np.eye(3), [0.8, 1.45, 0.0]))
    empty = tmp_path / "empty"
    synth_v2.build_v2(empty, "e052")
    for p in (empty / "e052" / "agent_files" / "runs" / "r1" / "frames").glob("*"):
        if p.suffix == ".json" and not p.name.startswith("_"):
            d = json.loads(p.read_text())
            p.write_text(json.dumps({**d, "prior": {"trust": "none", "T_map_cam": None}}))
    ep = epi.load_episode(empty / "e052", LAYOUT)
    assert "r1" not in ep.odom_scale  # no priors to fit: the base pose
    assert v2.start_from_priors(ep, [], LAYOUT) is None
    arm_like = epi.load_episode(empty / "e052", merge(LAYOUT, {"mount_fit": ["base"]}))
    assert arm_like.mount_rot_spread_deg is None  # nothing believed to fit a mount from
    frames = list(ep.sets["r1/frames"].frames)
    look = [f for f in frames if f.file.startswith("look")]
    for f in look:  # the look sweep turns in place: too little spread to fit
        f.prior = {"trust": "metric"}
        f.pose = synth.camera_body("rosmaster_x3_plus", 0, 0, 0)
    assert v2.start_from_priors(ep, look, LAYOUT) is None
    for f in frames:  # priors that never moved while the truth did
        f.prior = {"trust": "metric"}
        f.pose = np.eye(4)
    assert v2.start_from_priors(ep, frames, LAYOUT) is None
    meta = empty / "e052" / "truth.jsonl"
    lines = meta.read_text().splitlines()
    meta.write_text("\n".join([json.dumps({**json.loads(lines[0]), "robot": "x"})] + lines[1:]))
    with pytest.raises(LayoutError, match="robot 'x' is not in the layout's robots table"):
        epi.load_episode(empty / "e052", LAYOUT)
    meta.write_text("\n".join(lines[1:]))
    with pytest.raises(LayoutError, match="no line with kind = 'meta'"):
        epi.load_episode(empty / "e052", LAYOUT)
    (empty / "e052" / "agent_files" / "runs" / "r1" / "log.jsonl").unlink()
    assert "log" not in epi.dump_layout(empty / "e052", LAYOUT)["runs_v2"][0]


def test_v2_replay_end_to_end(tmp_path: Path) -> None:
    root = tmp_path / "eps"
    s = synth_v2.build_v2(root)
    truth = {s.api[n]: T for n, T in s.truth.items()}
    cfg = Config(truth=truth, metric_sim=Sim(1.0, 0.0, tuple(s.start[:3, 3])))
    with Fake(cfg) as fake:
        out = tmp_path / "r.json"
        assert replay_eval.main(["--server", fake.url, "--episodes", str(root), "--out",
                                 str(out), "--probe-interval-s", "0.02", "--wait-s", "1",
                                 "--priors", "metric"]) == 0
        assert fake.accept_ifs == [{"max_gauge_shift_m": 0.05},
                                   {"max_gauge_shift_m": 0.05, "max_rejected_share": 0.5}]
        assert len(fake.depths) == 8  # every update frame's depth part
        assert {p["trust"] for p in fake.locate_priors} >= {"metric"}
        sent = [p for p in fake.priors]
        assert {p["sigma_pos_m"] for p in sent} == {0.0372}  # the sidecar's sigmas
    res = json.loads(out.read_text())
    run = res["episodes"]["e051"]["runs"]["metric"]
    assert [j["recorded_s"] for j in run["jobs"]] == [4.958, 6.1]
    assert run["jobs"][0]["recorded_job_id"] == "2dab2e4523d4486a"
    m = run["metrics"]
    assert m["c8.sweep_yaw_error_deg"]["value"] == pytest.approx(0.0, abs=1e-6)
    assert m["c8.sweep_yaw_span_ratio"]["value"] == pytest.approx(1.0)
    sw = run["info"]["sweeps"]
    assert len(sw) == 1 and sw[0]["keyframes"] == 4 and sw[0]["span_truth_deg"] == \
        pytest.approx(75.0)
    with Fake(cfg) as fake:
        assert replay_eval.main(["--server", fake.url, "--episodes", str(root), "--out",
                                 str(out), "--priors", "none", "--no-depth", "--no-probes",
                                 "--no-height",
                                 "--camera", '{"fx": 1, "fy": 1, "cx": 1, "cy": 1, '
                                 '"width": 640, "height": 480}']) == 0
        assert fake.depths == [] and {p["trust"] for p in fake.priors} == {"none"}
        assert fake.heights == []


def test_v2_frame_entry_partial_prior(tmp_path: Path) -> None:
    from rpl_api import Log
    from rpl_replay import Options, Replayer

    s = synth_v2.build_v2(tmp_path)
    ep = epi.load_episode(s.root, LAYOUT)
    f = ep.sets["r1/frames"].frames[0]
    f.prior = {"trust": "drifting", "sigma_pos_m": None}

    class Stub:
        log = Log()

    r = Replayer(Stub(), ep, "metric", ["O1"], "m", Options())  # type: ignore[arg-type]
    e = r.frame_entry(f, "f0")
    assert e["prior"] == {"trust": "drifting", "T_map_cam": f.pose.tolist()}
    assert e["height_above_floor_m"] == 0.5 and e["height_sigma_m"] == 0.02


def test_robot_control_calibration(tmp_path: Path) -> None:
    """Fallback 3: the episode's robot_control/<robot>/calibration.json that lists the
    sidecar's camera_id, whatever its layout; --dump-layout shows its keys."""
    s = synth_v2.build_v2(tmp_path, camera="none")
    rc = s.root / "robot_control"
    def entry(v: Any, unit: str = "px") -> dict[str, Any]:
        return {"value": v, "unit": unit, "source": "camera_info", "run_id": "20261110-155923",
                "date": "2026-11-10", "confirmed": True}

    def calibration(robot: str, cam: str, fx: float, w: int, h: int) -> dict[str, Any]:
        """The harness layout of e051's robot_control/<robot>/calibration.json."""
        vals = {f"camera.{cam}.{k}": entry(v) for k, v in (
            ("fx_px", fx), ("fy_px", fx), ("cx_px", (w - 1) / 2), ("cy_px", (h - 1) / 2),
            ("width_px", w), ("height_px", h))}
        vals[f"camera.{cam}.T_body_cam"] = entry(
            [[0, 0, 1, -0.043645], [-1, 0, 0, 0], [0, -1, 0, 0.49555], [0, 0, 0, 1]], "m")
        vals.update({"base.kind": entry("mecanum", ""), "base.half_width_m": entry(0.12, "m"),
                     "base.odom_scale.x": entry(1.01, ""), "base.axis.yaw.sign": entry(1, ""),
                     "scan.T_body_scan": entry([[1, 0, 0, 0]] * 4, "m")})
        return {"robot_id": robot, "values": vals}

    (rc / "re_bot_arm_rs").mkdir(parents=True)
    (rc / "re_bot_arm_rs" / "calibration.json").write_text(json.dumps(
        calibration("rebot_b601_rs", "rs", 337.2, 640, 360)))
    (rc / "yahboomcar_x3plus").mkdir()
    (rc / "yahboomcar_x3plus" / "calibration.json").write_text(json.dumps(
        calibration("rosmaster_x3_plus", "astra", 655.8, 640, 480)))
    ep = epi.load_episode(s.root, LAYOUT)
    assert ep.cameras["astra"] == {"fx": 655.8, "fy": 655.8, "cx": 319.5, "cy": 239.5,
                                   "width": 640, "height": 480}
    assert ep.meta["camera_sources"]["astra"].endswith("yahboomcar_x3plus/calibration.json")
    r = epi.dump_layout(s.root, LAYOUT)["runs_v2"][0]
    assert r["intrinsics"] == {"astra": "calibration.json"}
    tree = r["robot_control"]["yahboomcar_x3plus"]
    assert tree["robot_id"] == "str" and tree["values"]["camera.astra.fx_px"] == {
        "value": "float", "unit": "str", "source": "str", "run_id": "str", "date": "str",
        "confirmed": "bool"}
    # the calibration of another size: not used
    (rc / "yahboomcar_x3plus" / "calibration.json").write_text(json.dumps(
        calibration("rosmaster_x3_plus", "astra", 655.8, 320, 240)))
    with pytest.raises(LayoutError, match="lists 'astra' with intrinsics"):
        epi.load_episode(s.root, LAYOUT)


def test_calibration_parsing_variants() -> None:
    block = v2._block_for
    assert block({"a": [{"id": "astra", "x": 1}]}, "astra") == {"id": "astra", "x": 1}
    assert block({"astra": {"K": 1}}, "astra") == {"K": 1}
    assert block([1, "astra"], "astra") is None and block(3, "astra") is None
    assert v2._intrinsics({"K": [[1, 0, 2], [0, 3, 4], [0, 0, 1]]}) == {
        "fx": 1.0, "fy": 3.0, "cx": 2.0, "cy": 4.0}
    assert v2._intrinsics({"nested": {"fx": 1, "fy": 2, "ppx": 3, "ppy": 4}}) == {
        "fx": 1.0, "fy": 2.0, "cx": 3.0, "cy": 4.0}
    assert v2._intrinsics({"K": [1, 2], "other": 5}) is None
    assert v2._intrinsics({"camera_matrix": {"data": [5, 0, 6, 0, 7, 8, 0, 0, 1]}}) == {
        "fx": 5.0, "fy": 7.0, "cx": 6.0, "cy": 8.0}
    assert v2._size({"image_width": 8, "image_height": 6}) == (8, 6)
    assert v2._size({"inner": {"size": [4, 3]}}) == (4, 3)
    assert v2._size({"width": 8, "size": [1]}) is None
    assert v2._from_values({"values": {"camera.astra.fx_px": 1}}, "astra") is None
    assert v2._from_values({"values": [1]}, "astra") is None and v2._from_values(3, "a") is None
    k, size = v2._from_values({"values": {f"camera.a.{n}": 1 for n in  # type: ignore[misc]
                                          ("fx_px", "fy_px", "cx_px", "cy_px")}}, "a")
    assert size is None and k["fx"] == 1.0
    assert v2.key_tree({"a": []}) == {"a": []} and v2.key_tree({"a": 1}, 1) == {"a": "..."}
    assert v2._intrinsics({"a": {"x": 1}, "b": {"fx": 1, "fy": 1, "cx": 1, "cy": 1}}) == {
        "fx": 1.0, "fy": 1.0, "cx": 1.0, "cy": 1.0}
    assert v2._size({"a": {"x": 1}, "b": {"width": 2, "height": 1}}) == (2, 1)
    no_k = Path(tempfile.mkdtemp())
    (no_k / "bot").mkdir()
    (no_k / "bot" / "calibration.json").write_text(json.dumps({"astra": {"model": "x"}}))
    assert v2.camera_from_robot_control(no_k, "astra", (640, 480)) is None
    (no_k / "bot" / "calibration.json").write_text(json.dumps({"astra": {"fx": 1, "fy": 1,
                                                                         "cx": 1, "cy": 1}}))
    cam, src = v2.camera_from_robot_control(no_k, "astra", (640, 480))  # type: ignore[misc]
    assert cam["width"] == 640 and src.endswith("calibration.json")  # no size: the frame's
    (no_k / "abot").mkdir()  # listed first, another size: the next robot's is used
    (no_k / "abot" / "calibration.json").write_text(json.dumps(
        {"astra": {"fx": 9, "fy": 9, "cx": 9, "cy": 9, "size": [320, 240]}}))
    cam, src = v2.camera_from_robot_control(no_k, "astra", (640, 480))  # type: ignore[misc]
    assert cam["fx"] == 1.0 and "/bot/" in src
