"""End to end: the evaluator's command line against the fake service on synthetic episodes."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import replay_eval

from . import synth
from .fake_server import ALL_CAPS, Config, Fake, Sim

FAST = ["--probe-interval-s", "0.02", "--wait-s", "1", "--range-scans", "3"]


def name(syn: synth.Synth, key: str, k: int) -> str:
    return syn.names[key][k]


def base_config(syn: synth.Synth, **kw: Any) -> Config:
    T1 = syn.start["r1"]
    return Config(truth=syn.truth, metric_sim=Sim(1.0, 0.0, tuple(T1[:3, 3])), **kw)


@pytest.fixture
def base_episode(tmp_path: Path) -> tuple[Path, synth.Synth]:
    root = tmp_path / "episodes"
    syn = synth.build(root)
    (tmp_path / "house.json").write_text(json.dumps(synth.house()))
    return root, syn


def run(fake: Fake, root: Path, out: Path, *extra: str, rc: int = 0) -> dict[str, Any]:
    code = replay_eval.main(["--server", fake.url, "--episodes", str(root), "--out", str(out),
                             *FAST, *extra])
    assert code == rc
    return json.loads(out.read_text())


def test_base_episode_all_capabilities(base_episode: tuple[Path, synth.Synth],
                                       tmp_path: Path) -> None:
    root, syn = base_episode
    cfg = base_config(
        syn,
        reject={name(syn, "r1/wall", 1): "too_close", name(syn, "r1/wall", 2): "plain",
                name(syn, "r2/start", 0): "no_overlap"},
        locate_offset={name(syn, "r1/loc", 2): 0.5}, ambiguous={name(syn, "r1/loc", 2)},
        locate_reject={name(syn, "r1/loc", 3)})
    with Fake(cfg) as fake:
        res = run(fake, root, tmp_path / "out" / "r.json", "--house",
                  str(tmp_path / "house.json"), "--episode", "e041")
        assert fake.accept_ifs and all(a == {"max_residual_m": 0.10, "max_gauge_shift_m": 0.05,
                                             "max_rejected_share": 0.5}
                                       for a in fake.accept_ifs)
        priors = [p for p in fake.priors if p["trust"] != "none"]
        # the base's odometry priors are drifting, sigma 10 % of the distance travelled
        assert priors and {p["trust"] for p in priors} == {"drifting"}
        sig = [p["sigma_pos_m"] for p in priors]
        assert min(sig) == 0.02 and sig[1] == pytest.approx(0.1 * 0.4 * synth.ODOM_SCALE)
        assert max(sig) > 0.5  # the whole r1 path, over-counted
        # locate frames carry the episode's prior in the prior replay, none in the other
        trusts = {p["trust"] for p in fake.locate_priors}
        assert trusts == {"drifting", "none"}
    ov = res["overall"]
    assert ov["c6.other_updates_accepted"]["value"] == 1.0
    assert ov["c1.prior_residual_m"]["status"] == "info"
    assert ov["c12.locate_first_s"]["status"] == "info"
    # the per-object section: map labels "cup" and "dining table" agree with Mug and DiningTable
    objs = res["episodes"]["e041"]["runs"]["metric"]["info"]["objects"]
    rows = {r["name"]: r for r in objs["truth"]}
    mug = rows["Mug|1|2"]
    assert mug["matched"]["label"] == "cup" and mug["label_agreed"] is True
    assert mug["canonical"] == "cup" and mug["centre_error_m"] == pytest.approx(0, abs=1e-6)
    assert rows["DiningTable|1|1"]["label_agreed"] is True
    assert [u["label"] for u in objs["unmatched_map"]] == ["Chair"]
    assert res["label_rule"]["examples"]["Apple_1"] == "apple"
    assert ov["c10.label_agreement"]["value"] == 1.0
    # nearest_view turns are judged only in the prior replay: turn_deg is null without a prior
    runs = res["episodes"]["e041"]["runs"]
    assert runs["metric"]["metrics"]["c8.nearest_view_turn"]["value"] == 1.0
    assert runs["none"]["metrics"]["c8.nearest_view_turn"]["status"] == "no_samples"
    assert ov["c1.prior_residual_m"]["value"] == pytest.approx(0.01)
    assert res["episodes"]["e041"]["passed"] is True
    assert res["health"]["start"]["capabilities"] == ALL_CAPS
    assert res["client"]["jsonschema"] is True
    e = res["episodes"]["e041"]
    assert set(e["runs"]) == {"metric", "none"}
    assert e["floor_plan"] is True
    for mid in ("c8.located_near_keyframe", "c8.ambiguity_on_wrong", "c8.ambiguity_rule",
                "c8.nearest_view_turn", "c4.reason_agreement", "c4.inputs_contract",
                "c3.reads_one_version", "contract.schema_valid", "contract.server_timing",
                "c9.camera_cells_free", "c9.regions_set", "c9.band_as_asked",
                "c9.doors_found", "c9.passage_geometry", "c10.relations_agreement",
                "c5.revert_contract"):
        assert ov[mid]["value"] == 1.0, (mid, ov[mid])
    assert ov["c8.ambiguity_null_on_right"]["value"] == 1.0
    assert ov["c8.pos_error_m"]["status"] == "info"
    assert ov["c2.scale_error_priors"]["value"] == pytest.approx(0.0, abs=1e-9)
    assert ov["c2.scale_error_height"]["value"] == pytest.approx(0.05, abs=1e-9)
    assert ov["c10.centre_error_m"]["value"] == pytest.approx(0.0, abs=1e-9)
    assert ov["c11.range_rel_error"]["value"] == pytest.approx(0.0, abs=1e-9)
    assert ov["c9.grid_agreement"]["value"] > 0.95
    assert ov["c9.floor_height_error_m"]["value"] == pytest.approx(0.0, abs=1e-9)
    assert "c6.warping_rejected" not in ov  # not a warping episode
    # latencies of the fake (milliseconds, noisy ratios) are not judged here
    fails = {k: m for k, m in ov.items() if m["status"] == "fail" and m["unit"] != "s"
             and not k.endswith("_busy_ratio")}
    assert not fails, fails
    run_m = e["runs"]["metric"]
    assert run_m["info"]["alignment"]["scale"] == pytest.approx(1.0)
    assert run_m["jobs"][0]["state"] == "committed"
    assert "POST updates" in run_m["calls"]


def test_capabilities_missing_and_gate(base_episode: tuple[Path, synth.Synth],
                                       tmp_path: Path) -> None:
    root, syn = base_episode
    (root / "e041").rename(root / "e034")  # a warping episode
    cfg = base_config(syn, capabilities=["O1", "O4", "O9"],
                      gate_reject={name(syn, "r1/chain1", 0)}, no_server_timing=True)
    with Fake(cfg) as fake:
        res = run(fake, root, tmp_path / "r.json", "--priors", "metric", "--keep",
                  "--no-jsonschema", "--accept-if", "{}", "--passing", "e034",
                  "--base-priors", "metric")
        assert any(m == "DELETE" for m, _ in fake.requests)
        assert {p["trust"] for p in fake.priors} == {"metric"}
        assert fake.accept_ifs and all(a == {} for a in fake.accept_ifs)
    ov = res["overall"]
    assert ov["c6.warping_rejected"]["value"] == 1.0
    assert "c6.other_updates_accepted" not in ov  # a warping episode
    assert ov["c8.located_near_keyframe"] == {**ov["c8.located_near_keyframe"],
                                             "status": "not_measurable",
                                             "missing": "capability O5"}
    assert ov["c9.grid_agreement"]["missing"] == "capability O6"
    assert ov["contract.server_timing"]["value"] == 0.0
    assert res["client"]["jsonschema"] is False
    assert res["episodes"]["e034"]["warnings"]  # no --house
    jobs = res["episodes"]["e034"]["runs"]["metric"]["jobs"]
    assert [j["state"] for j in jobs].count("rejected") == 1


def test_humanoid_and_arm(tmp_path: Path) -> None:
    root = tmp_path / "eps"
    hum = synth.build(root, "e017", "ainex", image="png")
    arm_sets = [synth.SetSpec("r1", "pp", [(0, 0, 0), (0.3, 0, 10), (0.6, 0.3, 20)]),
                synth.SetSpec("r1", "loc", [(0.1, 0, 12)])]
    arm = synth.build(root, "e044", "rebot_b601_rs", sets=arm_sets, scene="ithor:1",
                      objects_lines=True)
    truth = {**hum.truth, **arm.truth}
    houses = tmp_path / "houses"
    houses.mkdir()
    (houses / "7.json").write_text(json.dumps(synth.house()))
    cfg = Config(truth=truth, metric_sim=Sim(1.0, 0.0, tuple(arm.start["r1"][:3, 3])),
                 none_sim=Sim(0.9, -20.0, (1.0, 1.0, 0.0)))
    with Fake(cfg) as fake:
        res = run(fake, root, tmp_path / "r.json", "--house", str(houses))
    h, a = res["episodes"]["e017"], res["episodes"]["e044"]
    assert list(h["runs"]) == ["none"] and list(a["runs"]) == ["metric"]
    assert h["floor_plan"] is True and a["floor_plan"] is False
    hm = h["runs"]["none"]["metrics"]
    assert hm["c2.scale_error_height"]["value"] == pytest.approx(0.1, abs=1e-6)
    am = a["runs"]["metric"]["metrics"]
    assert am["c9.grid_agreement"]["status"] == "not_measurable"
    assert am["c9.grid_agreement"]["missing"].startswith("floor plan")
    assert am["c2.scale_error_priors"]["value"] == pytest.approx(0.0, abs=1e-9)


def test_errors_are_reported_per_episode(base_episode: tuple[Path, synth.Synth],
                                         tmp_path: Path) -> None:
    root, syn = base_episode
    bad = root / "e099"
    bad.mkdir()
    (bad / "truth.jsonl").write_text('{"kind": "meta", "robot": "unknown_bot"}\n')
    houses = tmp_path / "houses"
    houses.mkdir()
    cfg = base_config(syn, fail_route="/grid", busy_once=True)
    with Fake(cfg) as fake:
        res = run(fake, root, tmp_path / "r.json", "--house", str(houses), "--priors", "none",
                  "--no-probes", "--no-height", rc=1)
    assert "unknown_bot" in res["episodes"]["e099"]["error"]
    run_ = res["episodes"]["e041"]["runs"]["none"]
    assert "internal" in run_["error"]
    assert any("no house JSON" in w for w in res["episodes"]["e041"]["warnings"])
    assert "e099: error" in replay_eval.summary_table(res)
    assert "e041 none: error" in replay_eval.summary_table(res)


def test_retry_timeout_and_monocular(base_episode: tuple[Path, synth.Synth],
                                     tmp_path: Path) -> None:
    root, syn = base_episode
    cfg = base_config(syn, busy_once=True, capabilities=["O1", "O2", "O4"], eta=None)
    with Fake(cfg) as fake:
        res = run(fake, root, tmp_path / "r.json", "--priors", "none", "--no-height",
                  "--retry-wait-s", "0", "--targets", str(write(tmp_path / "t.json", {"targets": {
                      "c3.readable_s": {"op": "<=", "value": 1e-9}, "junk": 1}})))
    ov = res["overall"]
    assert ov["c2.scale_error_monocular"]["status"] == "info"
    assert ov["c3.readable_s"]["status"] == "fail"
    assert ov["c5.revert_s"]["status"] == "pass"
    cfg = base_config(syn, durations=(0.0, 0.05, 30.0))
    with Fake(cfg) as fake:
        res = run(fake, root, tmp_path / "r2.json", "--priors", "metric", "--job-timeout-s",
                  "0.1", "--wait-s", "0.05", "--episode", "e041")
    jobs = res["episodes"]["e041"]["runs"]["metric"]["jobs"]
    assert jobs[0]["state"] == "cancelled" and "timed out" in jobs[0]["error"]


def write(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data))
    return path


def test_no_episode_ran_and_cli_checks(base_episode: tuple[Path, synth.Synth], tmp_path: Path,
                                       capsys: pytest.CaptureFixture[str]) -> None:
    root, syn = base_episode
    with pytest.raises(SystemExit, match="required"):
        replay_eval.main(["--episodes", str(root)])
    with pytest.raises(SystemExit, match="no episode folder e777"):
        replay_eval.main(["--episodes", str(root), "--episode", "e777"])
    with pytest.raises(SystemExit, match="no e<NNN>"):
        replay_eval.main(["--episodes", str(tmp_path / "house.json").replace("house.json", "")
                          + "out_none" if False else str(syn.root / "agent_files")])
    empty = tmp_path / "empty"
    (empty / "e001").mkdir(parents=True)
    with Fake(base_config(syn)) as fake:
        rc = replay_eval.main(["--server", fake.url, "--episodes", str(empty), "--out",
                               str(tmp_path / "x.json")])
    assert rc == 1
    assert replay_eval.fmt(None) == "-" and replay_eval.fmt(3) == "3"


def test_dump_layout(base_episode: tuple[Path, synth.Synth],
                     capsys: pytest.CaptureFixture[str]) -> None:
    root, _ = base_episode
    assert replay_eval.main(["--episodes", str(root), "--dump-layout"]) == 0
    dump = json.loads(capsys.readouterr().out)
    d = dump[0]
    assert d["load"]["robot"] == "rosmaster_x3_plus"
    assert d["load"]["odometry_scale"]["r1"] == pytest.approx(1 / synth.ODOM_SCALE)
    assert d["truth"]["robot"]["cams"] == ["camera_color_optical_frame", "usb_cam"]
    assert d["sets"][0]["exif_fx_fy"] == [synth.FX_EXIF, synth.FX_EXIF]
    assert d["requests_log"]["used"] is False


def test_script_entry_point(base_episode: tuple[Path, synth.Synth],
                            monkeypatch: pytest.MonkeyPatch,
                            capsys: pytest.CaptureFixture[str]) -> None:
    root, _ = base_episode
    monkeypatch.setattr(sys, "argv", ["replay_eval.py", "--episodes", str(root),
                                      "--dump-layout"])
    with pytest.raises(SystemExit) as e:
        runpy.run_path(str(Path(replay_eval.__file__)), run_name="__main__")
    assert e.value.code == 0
    assert np.isfinite(len(capsys.readouterr().out))


def test_arm_floor_like_e044(tmp_path: Path) -> None:
    """e044's shape: the hello block at meta.sim.hello, placement "worktop", the arm's map frame
    at its base link at worktop height. A floor_z_m on the worktop (about 0 in the map) is right;
    one on the room floor (about -1.1 in the map) is a 1.1 m error, and the info shows which."""
    root = tmp_path / "eps"
    sets = [synth.SetSpec("r1", "pp", [(0, 0, 0), (0.3, 0, 10), (0.6, 0.3, 20)])]
    arm = synth.build(root, "e044", "rebot_b601_rs", sets=sets, scene="ithor:1",
                      objects_lines=True)
    meta = json.loads((arm.root / "truth.jsonl").read_text().splitlines()[0])
    assert sorted(meta) == ["cameras", "kind", "objects", "placement", "robot", "robot_row",
                            "sim", "t", "t0"]
    worktop = synth.WORKTOP_Z["arm"]
    anchor = Sim(1.0, 0.0, tuple(arm.start["r1"][:3, 3]))  # the base link, at worktop height
    for floor_world, want in ((worktop, 0.0), (0.0, worktop)):
        cfg = Config(truth=arm.truth, metric_sim=anchor, floor_world_z=floor_world)
        with Fake(cfg) as fake:
            res = run(fake, root, tmp_path / f"r{floor_world}.json")
        r = res["episodes"]["e044"]["runs"]["metric"]
        assert r["metrics"]["c9.floor_height_error_m"]["value"] == pytest.approx(want, abs=1e-6)
        assert r["info"]["floor_reference_m"] == pytest.approx(worktop)
        assert r["info"]["floor_z_world_m"] == pytest.approx(floor_world, abs=1e-6)
