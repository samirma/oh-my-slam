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
    assert "c2.scale_error_priors" not in s.values
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
    assert s.values["c3.fusing_version_gap_s"] == [pytest.approx(19.5)]
    assert "c3.eta_within_30pct" not in s.values
    assert s.values["c3.readable_s"] == [0.5, 0.5]
    assert s.values["c4.reason_agreement"] == [1.0]  # r2's first frame: no overlap
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
    assert s.values["c9.camera_cells_free"] == [1.0, 1.0, 0.0]
    assert "c9.grid_read_s" not in s.values and "c9.floor_height_error_m" not in s.values
    # aligned, the door unobserved; then observed with no passage; no floor height
    rec = record(["O6"], frames={1: {"frames": rows(ep, names)}})
    rec.jobs = [job(names)]
    rec.grid = grid(np.zeros((8, 8)), (-1.0, -1.0))
    s = measure(rec, dataclasses.replace(ep, floor_z=None), plan, raster)
    assert "c9.doors_found" not in s.values and "c9.floor_height_error_m" not in s.values
    big = np.ones((10, 20))
    rec.grid = grid(big, (-1.0, -1.45))
    s = measure(rec, ep, plan, raster)
    assert s.values["c9.doors_found"] == [0.0]


def test_objects_and_scans_edge_cases(world: Any) -> None:
    ep, syn, plan, _ = world
    names = syn.names["r1/start"]
    obj = {"id": 1, "label": "Sofa", "centre_m": [5.2, 1.05, 0.4], "state": "confirmed",
           "region": None, "supported_by": None, "inside_of": None}
    rec = record(["O8", "O7"], objects={1: {"objects": [obj]}},
                 frames={1: {"frames": rows(ep, names)}})
    s = measure(rec, ep, plan, objs=plan.objects)  # no alignment
    assert "c10.centre_error_m" not in s.values and "c10.objects_read_s" not in s.values
    rec.jobs = [job(names)]
    loose = [fp.TruthObject("s", "Sofa", np.array([6.0, 2.5, 0.4]))]  # no parent, no support
    T = np.array(rows(ep, names[:1])[0]["T_map_cam"])
    T[:3, 3] = [-5.0, 0.0, 0.5]
    T[:3, :3] = synth.rot_z(math.pi) @ synth.R_BODY_OPT  # outside, facing away
    rec.analyses = [("a", T.tolist(), {"frame": "gravity_camera", "range_scan": []}),
                    ("b", T.tolist(), {"frame": "map", "range_scan": [
                        {"bearing_deg": 180.0, "range_m": 1.0, "kind": "obstacle"}]})]
    s = measure(rec, ep, plan, objs=loose)
    assert s.values["c10.centre_error_m"] == [pytest.approx(0.0, abs=1e-9)]
    assert "c10.relations_agreement" not in s.values and "c11.range_rel_error" not in s.values


def test_locate_without_frames_of_its_version(world: Any) -> None:
    ep, syn, _, _ = world
    n = syn.names["r1/loc"][0]
    lo = LocateObs([n], {"map_version": 9, "frames": [
        {"name": n, "status": "rejected", "reason": "no_overlap", "ambiguity": None,
         "nearest_view": {"name": "elsewhere", "T_map_cam": [], "turn_deg": 0}}]}, 0.1, "plan")
    rec = record(["O5"], locates=[lo])
    s = measure(rec, ep)
    assert "c8.located_near_keyframe" not in s.values and "c8.nearest_view_turn" not in s.values
    assert s.info["locate_tolerance_m"] == 0.02


def test_verdicts_and_overall() -> None:
    d = rm.BY_ID["c3.readable_s"]
    assert rm.verdict(d, [1.0, 3.0], None, {})["status"] == "fail"
    assert rm.verdict(d, [], None, {})["status"] == "no_samples"
    v = rm.verdict(d, [1.0], None, {"c3.readable_s": {"op": None, "value": None}})
    assert v["status"] == "info" and v["target"] is None
    assert rm.reduce([1, 2, 10], "max") == 10 and rm.reduce([], "share") is None
    na = rm.verdict(d, None, "capability O3", {})
    smp = rm.Samples()
    assert rm.overall([(smp, {"c3.readable_s": na})], {})["c3.readable_s"]["status"] == \
        "not_measurable"
    rule = rm.Params().labels
    assert rule.same("DiningTable", "table") and not rule.same("", "x")


def test_other_updates_accepted(world: Any) -> None:
    ep, syn, _, _ = world
    names = syn.names["r1/start"]
    failed = JobObs("x", names, 0.0, error="boom")  # never answered: no state
    rec = record(["O9"])
    rec.jobs = [job(names), job(names, state="rejected"), failed]
    assert measure(rec, ep).values["c6.other_updates_accepted"] == [1.0, 0.0]
    assert "c6.other_updates_accepted" not in measure(rec, ep, passing=("e999",)).values
    assert measure(rec, ep, passing=("e041",)).values["c6.other_updates_accepted"] == [1.0, 0.0]


def test_floor_reference_and_floor_observed_by_rule(world: Any) -> None:
    ep, syn, plan, raster = world
    names = syn.names["r1/start"]
    rec = record(["O6"], frames={1: {"frames": rows(ep, names)}})
    rec.jobs = [job(names)]
    rec.grid = grid(np.ones((10, 20)), (-1.0, -1.45), floor_z_m=0.2)
    base = measure(rec, ep)  # a floor robot: judged against the floor (0.0)
    assert base.values["c9.floor_height_error_m"] == [pytest.approx(0.2)]
    arm = dataclasses.replace(ep, kind="arm", worktop_z=0.2)  # an arm: against the worktop
    assert measure(rec, arm).values["c9.floor_height_error_m"] == [pytest.approx(0.0)]
    placed = dataclasses.replace(ep, placement="worktop", worktop_z=0.2)  # any kind on a worktop
    assert measure(rec, placed).values["c9.floor_height_error_m"] == [pytest.approx(0.0)]
    bare = dataclasses.replace(ep, kind="arm", worktop_z=None, floor_z=None)
    assert "c9.floor_height_error_m" not in measure(rec, bare).values
    # floor_observed 2 (free by rule: a camera stood there) is not an observation of the floor
    fo = np.ones((10, 20), np.uint8)
    fo[:, :10] = 2
    rec.grid["layers"]["floor_observed"] = b64(fo)
    full = len(measure(rec, ep, plan, raster).values["c9.floor_observed"])
    fo[:, :10] = 1
    rec.grid["layers"]["floor_observed"] = b64(fo)
    assert len(measure(rec, ep, plan, raster).values["c9.floor_observed"]) > full


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
    assert s.values["c8.nearest_view_turn"] == [0.0]  # with a prior but no turn: a miss
    assert s.values["c12.locate_first_s"] == [0.5]
    assert s.values["c3.eta_within_30pct"] == [1.0]  # O3-jobs serves eta_s
    assert rm.missing_for(rm.BY_ID["c3.fusing_version_gap_s"], {"O3-jobs"}, None) == \
        "capability O3"
    assert rm.missing_for(rm.BY_ID["c3.eta_within_30pct"], {"O3-jobs"}, None) is None


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
    assert yaw.values["c9.floor_height_error_m"][0] < 0.02
    assert yaw.info["floor_reference_m"] == 0.0
    full = measure(rec, ep, align_full_rotation=True)
    assert full.values["c9.floor_height_error_m"][0] > 0.1  # the roll tips the floor


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
    assert s.values["c9.camera_cells_free"] == [1.0, 1.0]
    assert s.values["c9.camera_stand_free"] == [1.0, 1.0]
    # the same cameras with world poses, or the grid read as rows = x, would miss
    rec.frames = {1: {"frames": rows(ep, names)}}
    assert measure(rec, ep).values["c9.camera_stand_free"] == [0.0, 0.0]
    rec.frames = {1: {"frames": fr}}
    rec.grid = grid(np.ascontiguousarray(state.T), (origin[1], origin[0]), cell)
    assert 1.0 not in measure(rec, ep).values.get("c9.camera_stand_free", [])


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
