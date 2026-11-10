"""Metric paths on hand-made replay records: alignment, latency, reasons, grid, objects, scans."""

from __future__ import annotations

import base64
import dataclasses
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rpl_floorplan as fp
import rpl_metrics as rm
from rpl_api import Call, Log
from rpl_episode import Episode, load_episode
from rpl_layout import load_layout
from rpl_replay import JobObs, LocateObs, Record
from rpl_targets import Targets

from . import synth
from .fake_server import Sim


@pytest.fixture
def world(tmp_path: Path) -> tuple[Episode, synth.Synth, fp.FloorPlan, fp.Raster]:
    syn = synth.build(tmp_path)
    ep = load_episode(syn.root, load_layout(None))
    plan = fp.load_house(synth.house(), ["Bowl"], 0.0)
    return ep, syn, plan, fp.rasterise(plan)


ANCHOR = Sim(1.0, 0.0, (0.8, 1.45, 0.0))


def rows(ep: Episode, names: list[str], sim: Sim = ANCHOR, cam: str = "cam0",
         state: str = "fused") -> list[dict[str, Any]]:
    by = {f.name: f for f in ep.all_frames()}
    return [{"name": n, "camera_id": cam, "T_map_cam": sim.pose_to_map(by[n].truth).tolist(),
             "sigma_pos_m": 0.01, "sigma_rot_deg": 0.1, "pose_state": state,
             "added_version": 1} for n in names]


def job(names: list[str], **kw: Any) -> JobObs:
    j = JobObs("r1/start", names, 0.0, t_posted=0.1, t_readable=0.5, t_end=2.0)
    j.job = {"state": "committed", "provisional_version": None, "result_version": 1,
             "inputs": [{"name": n, "status": "registered", "reason": None} for n in names],
             "quality": None, **kw}
    return j


def record(caps: list[str], **kw: Any) -> Record:
    return Record("e041", "metric", "m", caps, **kw)


def measure(rec: Record, ep: Episode, plan: fp.FloorPlan | None = None,
            raster: fp.Raster | None = None, objs: list[fp.TruthObject] | None = None,
            log: Log | None = None, **p: Any) -> rm.Samples:
    return rm.Measure(rec, ep, plan, raster, objs or [], log or Log(), rm.Params(**p)).run()


def keyframes(syn: synth.Synth) -> list[str]:
    return syn.names["r1/start"] + syn.names["r1/chain1"]


def test_alignment_fallbacks(world: Any) -> None:
    ep, syn, _, _ = world
    scan = syn.names["r1/scan1"]
    start = syn.names["r1/start"]
    same = rows(ep, [start[0], start[3]])
    for r in same:
        r["T_map_cam"] = np.eye(4).tolist()
    rec = record(["O1"], frames={2: {"frames": rows(ep, start[:1])},
                                 3: {"frames": rows(ep, scan[:2])},
                                 4: {"frames": same}})
    rec.jobs = [JobObs("a", [], 0.0), job([], provisional_version=1, result_version=2),
                job([], result_version=3), job([], result_version=4)]
    s = measure(rec, ep)
    assert s.info["alignment"] is None
    assert "scale.error_metric" not in s.values
    assert rm.alignment(rec, {f.name: f for f in ep.all_frames()}, 0.5) is None


def test_scene_depth_and_reasons(world: Any) -> None:
    ep, syn, plan, _ = world
    f = ep.all_frames()[0]
    away = dataclasses.replace(f, truth=synth.make_T(synth.R_BODY_OPT, [-1.0, 1.5, 0.5]))
    away.truth[:3, :3] = synth.rot_z(math.pi) @ synth.R_BODY_OPT  # outside, facing -x
    down = dataclasses.replace(f, truth=synth.make_T(np.diag([1.0, -1.0, -1.0]), [1, 1, 1.0]))
    assert rm.scene_depth(ep, plan, [away]) == 1.0
    assert rm.scene_depth(ep, plan, [down]) == pytest.approx(1 - 0.776)  # above the worktop
    bare = dataclasses.replace(ep, worktop_z=None, floor_z=None)
    assert rm.scene_depth(bare, None, [down]) == 1.0
    p = rm.Params()
    # facing the wall x = 0 from 1 m with an object in between: not plain
    facing = dataclasses.replace(f, truth=synth.make_T(synth.rot_z(math.pi) @ synth.R_BODY_OPT,
                                                       [1.0, 2.5, 0.5]))
    plan.objects.append(fp.TruthObject("o", "Box", np.array([0.5, 2.5, 0.2])))
    assert rm.truth_reason(facing, [], plan, 77.0, p) is None
    plan.objects.pop()
    assert rm.truth_reason(facing, [], plan, 77.0, p) == "plain"
    assert rm.truth_reason(facing, [away], None, 77.0, p) == "no_overlap"
    assert rm.truth_reason(facing, [facing], None, 77.0, p) is None


def test_latency_inputs_and_contracts(world: Any) -> None:
    ep, syn, _, _ = world
    names = syn.names["r1/start"]
    long = job(names)
    long.t_fusing, long.t_end, long.eta_queued = 1.0, 30.0, None
    rejected = job(names, state="rejected")
    rejected.t_readable = None
    other = syn.names["r2/start"][0]
    ins = job([*names, "ghost"], inputs=[
        {"name": names[0], "status": "rejected", "reason": "blur"},
        {"name": "ghost", "status": "rejected", "reason": "no_overlap"},
        {"name": other, "status": "rejected", "reason": "no_overlap"}])
    ins.t_end = 0.1
    ins.t_posted = 0.1  # measured 0: no eta sample
    ins.eta_queued = 1.0
    rec = record(["O3"], frames={1: {"frames": rows(ep, names)}},
                 probe_versions=[(0.5, 1), (5.0, 2), (5.1, 2), (25.0, 3)])
    rec.sent = [*names, other]
    rec.jobs = [long, rejected, ins]
    log = Log([Call("GET x", 200, 0.1, False, None, []), Call("GET y", 200, 0.1, True, "Frames",
                                                              ["bad"])])
    s = measure(rec, ep, log=log, probe_interval_s=0.5)
    assert s.values["fusion.intermediate_interval"] == [pytest.approx(19.5)]
    assert "jobs.eta_accuracy" not in s.values
    assert s.values["update.readable_latency"] == [0.5, 0.5]
    assert s.values["inputs.reason_agreement"] == [1.0]  # r2's first frame: no overlap
    assert s.values["contract.schema_valid"] == [0.0]
    assert s.values["contract.server_timing"] == [0.0, 1.0]
    assert s.info["schema_errors"] == ["GET y: bad"]


def b64(a: np.ndarray) -> dict[str, Any]:
    return {"dtype": str(a.dtype), "shape": list(a.shape),
            "data": base64.b64encode(a.tobytes()).decode()}


def grid(state: np.ndarray, origin: tuple[float, float], cell: float = 0.5,
         **kw: Any) -> dict[str, Any]:
    return {"map_version": 1, "origin_m": list(origin), "cell_m": cell,
            "shape": list(state.shape), "z_band_m": [0.05, 1.8], "floor_z_m": 0.0,
            "layers": {"state": b64(state.astype(np.uint8)),
                       "observations": b64(np.ones(state.shape, np.uint16)),
                       "height_m": b64(np.zeros(state.shape, np.float32)),
                       "floor_observed": b64(np.ones(state.shape, np.uint8))},
            "regions": [], "passages": [], "frontiers": [], **kw}


def test_grid_edge_cases(world: Any) -> None:
    ep, syn, plan, raster = world
    names = syn.names["r1/start"]
    fr = rows(ep, names[:3])
    fr[0]["camera_id"] = "unknown"
    up = np.array(fr[1]["T_map_cam"])
    up[:3, :3] = up[:3, :3] @ np.diag([1.0, -1.0, -1.0])  # looking up and back
    fr[1]["T_map_cam"] = up.tolist()
    fr[2]["T_map_cam"][0][3] = 99.0  # outside the grid
    rec = record(["O6"], frames={1: {"frames": fr}})
    rec.grid = grid(np.ones((4, 4)), (-1.0, -1.0))
    s = measure(rec, ep)  # no alignment: nothing compared with the floor plan
    assert s.values["grid.camera_cells_free"] == [1.0, 1.0, 0.0]
    assert "reads.latency" not in s.values and "grid.floor_height" not in s.values
    # aligned, the door unobserved; then observed with no passage; no floor height
    rec = record(["O6"], frames={1: {"frames": rows(ep, names)}})
    rec.jobs = [job(names)]
    rec.grid = grid(np.zeros((8, 8)), (-1.0, -1.0))
    s = measure(rec, dataclasses.replace(ep, floor_z=None), plan, raster)
    assert "passages.door_recall" not in s.values and "grid.floor_height" not in s.values
    big = np.ones((10, 20))
    rec.grid = grid(big, (-1.0, -1.45))
    s = measure(rec, ep, plan, raster)
    assert s.values["passages.door_recall"] == [0.0]


def test_objects_and_scans_edge_cases(world: Any) -> None:
    ep, syn, plan, _ = world
    names = syn.names["r1/start"]
    obj = {"id": 1, "label": "Sofa", "centre_m": [5.2, 1.05, 0.4], "state": "confirmed",
           "region": None, "supported_by": None, "inside_of": None}
    rec = record(["O8", "O7"], objects={1: {"objects": [obj]}},
                 frames={1: {"frames": rows(ep, names)}})
    s = measure(rec, ep, plan, objs=plan.objects)  # no alignment
    assert "objects.centre_error" not in s.values and "reads.latency" not in s.values
    rec.jobs = [job(names)]
    loose = [fp.TruthObject("s", "Sofa", np.array([6.0, 2.5, 0.4]))]  # no parent, no support
    T = np.array(rows(ep, names[:1])[0]["T_map_cam"])
    T[:3, 3] = [-5.0, 0.0, 0.5]
    T[:3, :3] = synth.rot_z(math.pi) @ synth.R_BODY_OPT  # outside, facing away
    rec.analyses = [("a", T.tolist(), {"frame": "gravity_camera", "range_scan": []}),
                    ("b", T.tolist(), {"frame": "map", "range_scan": [
                        {"bearing_deg": 180.0, "range_m": 1.0, "kind": "obstacle"}]})]
    s = measure(rec, ep, plan, objs=loose)
    assert s.values["objects.centre_error"] == [pytest.approx(0.0, abs=1e-9)]
    assert "objects.relations" not in s.values and "analysis.range_scan_error" not in s.values


def test_locate_without_frames_of_its_version(world: Any) -> None:
    ep, syn, _, _ = world
    n = syn.names["r1/loc"][0]
    lo = LocateObs([n], {"map_version": 9, "frames": [
        {"name": n, "status": "rejected", "reason": "no_overlap", "ambiguity": None,
         "nearest_view": {"name": "elsewhere", "T_map_cam": [], "turn_deg": 0}}]}, 0.1, "plan")
    rec = record(["O5"], locates=[lo])
    s = measure(rec, ep)
    assert "locate.recall" not in s.values and "locate.nearest_view" not in s.values
    assert s.info["locate_tolerance_m"] == 0.02


def test_verdicts_and_overall() -> None:
    t = Targets({"update.readable_latency": {"op": "<=", "value": 2.0},
                 "frame.shift": {"parts": {"pos_m": {"op": "<=", "value": 0.02},
                                           "rot_deg": {"op": "<=", "value": 0.5}}}})
    d = rm.BY_ID["update.readable_latency"]
    assert rm.verdict(d, {d.id: [1.0, 3.0]}, None, t)["status"] == "fail"
    assert rm.verdict(d, {}, None, t)["status"] == "no_samples"
    assert rm.verdict(d, {d.id: [1.0]}, None, Targets({}))["status"] == "no_target"
    info = rm.verdict(d, {d.id: [1.0]}, None, Targets({d.id: {"info": True}}))
    assert info["status"] == "info"
    assert rm.reduce([1, 2, 10], "max") == 10 and rm.reduce([], "share") is None
    shift = rm.BY_ID["frame.shift"]
    v = rm.verdict(shift, {"frame.shift#pos_m": [0.01], "frame.shift#rot_deg": [0.1]}, None, t)
    assert v["status"] == "pass" and v["value"] == {"pos_m": 0.01, "rot_deg": 0.1}
    v = rm.verdict(shift, {"frame.shift#pos_m": [0.03]}, None, t)
    assert v["status"] == "fail" and v["value"]["rot_deg"] is None
    assert rm.verdict(shift, {"frame.shift#pos_m": [0.01]}, None, t)["status"] == "no_samples"
    assert rm.verdict(rm.BY_ID["diag.locate_first_s"], {"diag.locate_first_s": [1.0]}, None,
                      t)["status"] == "info"
    assert rm.verdict(rm.BY_ID["contract.inputs"], {"contract.inputs": [1.0, 0.0]}, None,
                      t)["status"] == "fail"
    na = rm.verdict(d, None, "capability O3", t)
    smp = rm.Samples()
    smp.add("frame.shift#pos_m", 0.01)
    out = rm.overall([(smp, {"update.readable_latency": na,
                             "frame.shift": rm.verdict(shift, smp.values, None, t)})], t)
    assert out["update.readable_latency"]["status"] == "not_measurable"
    assert out["frame.shift"]["value"]["pos_m"] == 0.01
    rule = rm.Params().labels
    assert rule.same("DiningTable", "table") and not rule.same("", "x")


def test_gate_warping_and_clean_updates(world: Any) -> None:
    ep, syn, _, _ = world
    names = syn.names["r1/start"]
    by = {f.name: f for f in ep.all_frames()}

    def priors(slip: float, sigma: float | None = 0.05) -> dict[str, dict[str, Any]]:
        out = {}
        for i, n in enumerate(names):
            T = by[n].truth.copy()
            T[0, 3] += slip * i / (len(names) - 1)
            out[n] = {"trust": "metric", "T_map_cam": T.tolist()}
            if sigma is not None:
                out[n]["sigma_pos_m"] = sigma
        return out

    warping = job(names, state="rejected")
    warping.priors = priors(0.5)
    warped_ok = job(names)
    warped_ok.priors = priors(0.4)
    clean = job(names)
    clean.priors = priors(0.0)
    loose = job(names)  # slips beyond its sigmas, but less than the warping slip
    loose.priors = priors(0.2, None)
    loose.priors[names[-1]]["sigma_pos_m"] = 0.01
    queued = job(names, state="queued")
    rec = record(["O9"])
    rec.jobs = [warping, warped_ok, clean, loose, queued, job(names)]
    s = measure(rec, ep, passing=("e041",))
    assert s.values["gate.warping_rejected"] == [1.0, 0.0]
    assert s.values["gate.others_accepted"] == [1.0, 1.0]  # clean, and the one without priors
    assert "gate.others_accepted" not in measure(rec, ep, passing=("e999",)).values


def test_locate_sigmas_and_update_containment(world: Any) -> None:
    ep, syn, _, _ = world
    kf = keyframes(syn)
    by = {f.name: f for f in ep.all_frames()}
    rec = record(["O5"], frames={1: {"frames": rows(ep, kf)}})
    j = job(kf[:2])
    T = ANCHOR.pose_to_map(by[kf[1]].truth)
    T[0, 3] += 0.2
    j.job["inputs"] = [
        {"name": kf[0], "status": "registered", "sigma_pos_m": 0.01, "sigma_rot_deg": 0.1,
         "T_map_cam": ANCHOR.pose_to_map(by[kf[0]].truth).tolist()},
        {"name": kf[1], "status": "kept_at_prior", "sigma_pos_m": 0.05, "sigma_rot_deg": 1.0,
         "T_map_cam": T.tolist()},
        {"name": kf[2], "status": "rejected", "T_map_cam": None, "sigma_pos_m": None},
        {"name": "unknown", "status": "registered", "T_map_cam": T.tolist(),
         "sigma_pos_m": 0.1, "sigma_rot_deg": 1.0}]
    rec.jobs = [j]
    far = ANCHOR.pose_to_map(by[kf[3]].truth)
    far[0, 3] += 0.3
    lo = LocateObs([kf[2], kf[3]], {"map_version": 1, "frames": [
        {"name": kf[2], "status": "located", "T_map_cam": ANCHOR.pose_to_map(
            by[kf[2]].truth).tolist(), "sigma_pos_m": 0.01, "sigma_rot_deg": 0.5,
         "ambiguity": None, "rivals": []},
        {"name": kf[3], "status": "located", "T_map_cam": far.tolist(), "sigma_pos_m": 0.0,
         "sigma_rot_deg": 0.0, "ambiguity": {"pos_m": 0.3, "rot_deg": 0.0},
         "rivals": [{"pos_m": 0.3, "rot_deg": 0.0, "support": 0.5}]}]}, 0.1, "plan")
    rec.locates = [lo]
    s = measure(rec, ep)
    assert s.values["update.sigma_containment"] == [1.0, 0.0]
    assert s.values["locate.sigma_right_median"] == [pytest.approx(0.01)]
    assert s.values["locate.sigma_containment"] == [1.0, 0.0]
    assert s.values["locate.sigma_tightness#pos"] == [pytest.approx(0.0, abs=1e-6)]
    assert s.values["locate.ambiguity_flagged"] == [1.0]


def test_grid_id_stability(world: Any) -> None:
    ep, _, _, _ = world
    reg = {"id": 1, "area_m2": 4.0, "polygon_m": [[0, 0], [2, 0], [2, 2], [0, 2]],
           "coverage_share": 1.0}
    pas = {"id": 7, "centre_m": [1.0, 2.0], "width_m": 0.9}
    g1 = {"cell_m": 0.05, "regions": [reg], "passages": [pas]}
    g2 = {"cell_m": 0.05, "regions": [{**reg, "id": 2}, {**reg, "id": 3, "area_m2": 9.0}],
          "passages": [pas, {**pas, "id": 8, "centre_m": [5.0, 5.0]}]}
    rec = record(["O6"], grids={1: g1, 2: g2})
    assert measure(rec, ep).values["grid.id_stability"] == [0.0, 1.0]


def test_other_updates_accepted(world: Any) -> None:
    ep, syn, _, _ = world
    names = syn.names["r1/start"]
    failed = JobObs("x", names, 0.0, error="boom")  # never answered: no state
    rec = record(["O9"])
    rec.jobs = [job(names), job(names, state="rejected"), failed]
    assert measure(rec, ep).values["gate.others_accepted"] == [1.0, 0.0]
    assert "gate.others_accepted" not in measure(rec, ep, passing=("e999",)).values
    assert measure(rec, ep, passing=("e041",)).values["gate.others_accepted"] == [1.0, 0.0]


def test_floor_reference_and_floor_observed_by_rule(world: Any) -> None:
    ep, syn, plan, raster = world
    names = syn.names["r1/start"]
    rec = record(["O6"], frames={1: {"frames": rows(ep, names)}})
    rec.jobs = [job(names)]
    rec.grid = grid(np.ones((10, 20)), (-1.0, -1.45), floor_z_m=0.2)
    base = measure(rec, ep)  # a floor robot: judged against the floor (0.0)
    assert base.values["grid.floor_height"] == [pytest.approx(0.2)]
    arm = dataclasses.replace(ep, kind="arm", worktop_z=0.2)  # an arm: against the worktop
    assert measure(rec, arm).values["grid.floor_height"] == [pytest.approx(0.0)]
    placed = dataclasses.replace(ep, placement="worktop", worktop_z=0.2)  # any kind on a worktop
    assert measure(rec, placed).values["grid.floor_height"] == [pytest.approx(0.0)]
    bare = dataclasses.replace(ep, kind="arm", worktop_z=None, floor_z=None)
    assert "grid.floor_height" not in measure(rec, bare).values
    # floor_observed 2 (free by rule: a camera stood there) is not an observation of the floor
    fo = np.ones((10, 20), np.uint8)
    fo[:, :10] = 2
    rec.grid["layers"]["floor_observed"] = b64(fo)
    full = len(measure(rec, ep, plan, raster).values["grid.floor_observed"])
    fo[:, :10] = 1
    rec.grid["layers"]["floor_observed"] = b64(fo)
    assert len(measure(rec, ep, plan, raster).values["grid.floor_observed"]) > full


def test_nearest_view_needs_a_prior_and_o3_jobs(world: Any) -> None:
    ep, syn, _, _ = world
    n = syn.names["r1/loc"][3]
    kf = syn.names["r1/start"][0]
    body = {"map_version": 1, "frames": [
        {"name": n, "status": "rejected", "reason": "no_overlap", "ambiguity": None,
         "nearest_view": {"name": kf, "T_map_cam": [], "turn_deg": None}}]}
    rec = record(["O5", "O3-jobs"], frames={1: {"frames": rows(ep, [kf])}},
                 locates=[LocateObs([n], body, 0.1, "plan", with_prior=True),
                          LocateObs([n], body, 0.1, "plan", with_prior=False)],
                 idle_locate_s=[0.5, 0.1], first_locate_s=0.5)
    j = job([kf])
    j.eta_queued = 1.9
    rec.jobs = [j]
    s = measure(rec, ep)
    assert s.values["locate.nearest_view"] == [0.0]  # with a prior but no turn: a miss
    assert s.values["diag.locate_first_s"] == [0.5]
    assert s.values["jobs.eta_accuracy"] == [1.0]  # O3-jobs serves eta_s
    assert rm.missing_for(rm.BY_ID["fusion.intermediate_interval"], {"O3-jobs"}, None) == \
        "capability O3"
    assert rm.missing_for(rm.BY_ID["jobs.eta_accuracy"], {"O3-jobs"}, None) is None


def test_floor_height_goes_through_the_alignment(world: Any) -> None:
    """The floor's map z is compared after the map -> world similarity. Cameras at one height
    along a line leave a 3-D rotation's roll free; the gravity-aligned (yaw) fit does not."""
    ep, syn, _, _ = world
    names = syn.names["r1/start"] + syn.names["r1/chain1"]
    sim = Sim(1.05, 30.0, (1.0, 2.0, 0.3))  # world = 1.05 R map + t
    fr = rows(ep, names, sim)
    for k, r in enumerate(fr):  # registration noise: 2 cm sideways and up together
        T = np.array(r["T_map_cam"])
        T[1, 3] += 0.02 if k % 2 else -0.02
        T[2, 3] += 0.02 if k % 2 else -0.02
        r["T_map_cam"] = T.tolist()
    rec = record(["O6"], frames={1: {"frames": fr}})
    rec.jobs = [job(names)]
    floor_map = float(sim.to_map([0.0, 0.0, 0.0])[2])  # the true floor in the map frame
    rec.grid = grid(np.ones((4, 4)), (-1.0, -1.0), floor_z_m=floor_map)
    yaw = measure(rec, ep)
    assert yaw.values["grid.floor_height"][0] < 0.02
    assert yaw.info["floor_reference_m"] == 0.0
    full = measure(rec, ep, align_full_rotation=True)
    assert full.values["grid.floor_height"][0] > 0.1  # the roll tips the floor


def test_camera_cells_use_map_poses_and_rows_y(world: Any) -> None:
    """A grid that is free only under the cameras, in a map frame that is not the world's, on a
    non-square grid with a non-zero origin: right only with map-frame poses, rows = y, cols = x
    and origin_m as the corner of cell [0, 0]."""
    ep, syn, _, _ = world
    names = syn.names["r1/start"][:2]
    sim = Sim(1.05, 30.0, (1.0, 2.0, 0.0))
    fr = rows(ep, names, sim)
    for r in fr:  # looking up: no blind ring, only the stand cell is required free
        T = np.array(r["T_map_cam"])
        T[:3, :3] = T[:3, :3] @ np.diag([1.0, -1.0, -1.0])
        r["T_map_cam"] = T.tolist()
    origin, cell, shape = (-3.0, -2.5), 0.1, (40, 70)  # rows (y) x cols (x)
    state = np.full(shape, 2, np.uint8)
    for r in fr:
        x, y = r["T_map_cam"][0][3], r["T_map_cam"][1][3]
        state[math.floor((y - origin[1]) / cell), math.floor((x - origin[0]) / cell)] = 1
    rec = record(["O6"], frames={1: {"frames": fr}})
    rec.grid = grid(state, origin, cell)
    s = measure(rec, ep)
    assert s.values["grid.camera_cells_free"] == [1.0, 1.0]
    assert s.values["diag.camera_stand_free"] == [1.0, 1.0]
    # the same cameras with world poses, or the grid read as rows = x, would miss
    rec.frames = {1: {"frames": rows(ep, names)}}
    assert measure(rec, ep).values["diag.camera_stand_free"] == [0.0, 0.0]
    rec.frames = {1: {"frames": fr}}
    rec.grid = grid(np.ascontiguousarray(state.T), (origin[1], origin[0]), cell)
    assert 1.0 not in measure(rec, ep).values.get("diag.camera_stand_free", [])


def test_object_report_rows(world: Any) -> None:
    ep, syn, plan, _ = world
    names = syn.names["r1/start"]
    base = {"state": "confirmed", "region": None, "supported_by": None, "inside_of": None,
            "size_m": [0.1, 0.1, 0.1], "mean_rgb": [1, 2, 3], "detected_as": ["x"]}
    objs = [  # map frame = world - (0.8, 1.45, 0)
        {**base, "id": 1, "label": "orange", "centre_m": [1.2, 1.05, 0.85]},  # at the Mug
        {**base, "id": 2, "label": "sofa", "centre_m": [5.2, 1.05, 0.4],
         "supported_by": {"kind": "surface", "id": 1, "verified_version": 1}},
        {**base, "id": 3, "label": "sofa", "centre_m": [5.25, 1.05, 0.4]},  # a duplicate
        {**base, "id": 4, "label": "lamp", "centre_m": [-5.0, -5.0, 0.4]},
    ]
    rec = record(["O8"], frames={1: {"frames": rows(ep, names)}},
                 objects={1: {"objects": objs, "surfaces": [
                     {"id": 1, "height_m": 0.03, "polygon_m": [[0, 0], [1, 0], [1, 1]],
                      "area_m2": 1.0}]}})
    rec.jobs = [job(names)]
    truth = [o for o in plan.objects if o.label in ("Mug", "Sofa", "Apple")]
    truth.append(fp.TruthObject("far", "Vase", np.array([20.0, 20.0, 0.0]),
                                size=np.array([0.1, 0.2, 0.3])))
    rep = measure(rec, ep, objs=truth).info["objects"]
    rows_ = {r["name"]: r for r in rep["truth"]}
    mug = rows_["Mug|1|2"]
    assert mug["matched"]["id"] == 1 and mug["label_agreed"] is False  # nearest, other label
    sofa = rows_["Sofa|2|1"]
    assert sofa["matched"]["id"] == 2 and sofa["label_agreed"] is True
    assert sofa["support_height_error_m"] == pytest.approx(0.03)
    assert rows_["far"]["matched"] is None and rows_["far"]["size_world"] == [0.1, 0.2, 0.3]
    assert {u["id"] for u in rep["unmatched_map"]} == {3, 4}


@pytest.mark.parametrize("origin, err, xy, z_key", [("base", 0.0, 0.0, "bottom_z_error_m"),
                                                    ("centre", 0.1, 0.0, "centre_z_error_m")])
def test_object_origin(world: Any, origin: str, err: float, xy: float, z_key: str) -> None:
    """A truth position at the object's base (MuJoCo body origin of a staged object) is compared
    with the box's centre xy and bottom z; a centre position with the box centre."""
    ep, syn, _, _ = world
    names = syn.names["r1/start"]
    obj = {"id": 1, "label": "apple", "centre_m": [0.4, 0.05, 0.8], "size_m": [0.1, 0.1, 0.2],
           "state": "confirmed", "region": None, "supported_by": None, "inside_of": None}
    rec = record(["O8"], frames={1: {"frames": rows(ep, names)}},
                 objects={1: {"objects": [obj]}})
    rec.jobs = [job(names)]  # map = world - (0.8, 1.45, 0)
    truth = [fp.TruthObject("apple_1", "apple", np.array([1.2, 1.5, 0.7]), origin=origin)]
    s = measure(rec, ep, objs=truth)
    assert s.values["objects.centre_error"] == [pytest.approx(err)]
    row = s.info["objects"]["truth"][0]
    assert row["origin"] == origin and row["centre_error_m"] == pytest.approx(err)
    assert row["xy_error_m"] == pytest.approx(xy) and row[z_key] == pytest.approx(err)


def test_sweeps_split_by_stand_and_flat(world: Any) -> None:
    ep, syn, _, _ = world
    names = syn.names["r1/start"] + syn.names["r1/chain1"]
    rec = record(["O1"], frames={1: {"frames": rows(ep, names)}})
    rec.jobs = [job(names)]
    # start frames 0.4 m apart: one frame per stand, no sweep
    s = measure(rec, ep, sweep_prefixes=("start",))
    assert s.info["sweeps"] == [] and "diag.sweep_yaw_error_deg" not in s.values
    # chain frames as one stand that never turns: errors, but no span ratio
    s = measure(rec, ep, sweep_prefixes=("chain",), sweep_stand_m=5.0)
    assert s.values["diag.sweep_yaw_error_deg"] == [pytest.approx(0.0, abs=1e-6)] * 5
    assert s.info["sweeps"][0]["ratio"] is None and "diag.sweep_yaw_span_ratio" not in s.values


def test_sweep_span_over_count(world: Any) -> None:
    """A registration that over-turns a sweep by 10 % shows as a span ratio of 1.1."""
    ep, syn, _, _ = world
    names = syn.names["r1/scan1"] + syn.names["r1/start"][:1]
    fr = rows(ep, names)
    for r in fr[:4]:
        T = np.array(r["T_map_cam"])
        yaw = math.atan2(T[1, 2], T[0, 2])
        T[:3, :3] = synth.rot_z(0.1 * yaw) @ T[:3, :3]
        r["T_map_cam"] = T.tolist()
    rec = record(["O1"], frames={1: {"frames": fr}})
    rec.jobs = [job(names)]
    s = measure(rec, ep, sweep_prefixes=("scan",))
    assert s.values["diag.sweep_yaw_span_ratio"] == [pytest.approx(1.1, abs=0.02)]
