"""The targets of specs/evaluation.md, measured on one replay record against the simulator truth.

Every metric is a list of samples reduced to one value (a share of passing samples, a median, a
p90 or a maximum) and compared with its target. Samples of all episodes are pooled for the
overall value. A metric whose API group the service does not serve (``/api/health``
``capabilities``) is ``not_measurable`` and names the missing group.
"""

from __future__ import annotations

import base64
import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rpl_api import Log
from rpl_episode import Episode, Frame
from rpl_floorplan import (
    BORDER,
    DOOR,
    FLOOR,
    OUTSIDE,
    WALL,
    FloorPlan,
    Raster,
    TruthObject,
    poly_distance,
    ray_cast,
)
from rpl_geometry import Arr, Sim3, angle_deg, inv_T, mat4, optical_yaw_deg, umeyama, wrap_deg
from rpl_labels import LabelRule
from rpl_replay import JobObs, Record
from rpl_targets import Targets, judge, reduce


@dataclass(frozen=True)
class Def:
    """A metric of specs/evaluation.md (``kind`` metric: its threshold is the examples/targets.json
    entry under ``id``), a diagnostic (``diag``: information only) or a contract (zero
    violations: the share of compliant answers must be 1)."""

    id: str
    contract: str
    description: str
    reduce: str  # share | median | p90 | max | sum
    unit: str
    needs: tuple[str, ...] = ()
    floorplan: bool = False
    parts: tuple[tuple[str, str, str], ...] = ()  # (part, reduce, unit) of a compound metric
    kind: str = "metric"

    def sample_keys(self) -> list[str]:
        """The sample keys: ``id``, or ``id#part`` per part."""
        return [f"{self.id}#{p}" for p, _, _ in self.parts] if self.parts else [self.id]


def diag(id: str, contract: str, description: str, reduce: str, unit: str,
         needs: tuple[str, ...] = ()) -> Def:
    return Def(f"diag.{id}", contract, description, reduce, unit, needs, kind="diag")


DEFS: list[Def] = [
    Def("locate.recall", "C8", "Frames within 0.3 m and 15° of a keyframe that are located",
        "share", "share", ("O5",)),
    Def("locate.median_pos_error", "C8", "Median position error of located frames over "
        "max(abs_m, scene_depth_share x the scene depth) (targets.json parameters)", "median",
        "ratio", ("O5",)),
    Def("locate.ambiguity_flagged", "C8", "Located frames whose pose error exceeds 0.10 m or 5° "
        "that carry a non-null ambiguity", "share", "share", ("O5",)),
    Def("locate.sigma_right_median", "C8", "Median sigma_pos_m (world metres) of located frames "
        "within 0.05 m of the truth with rivals empty", "median", "m", ("O5",)),
    Def("locate.nearest_view", "C8", "no_overlap frames sent with a prior whose nearest_view turn "
        "points toward the keyframe's view (within 45°)", "share", "share", ("O5",)),
    Def("locate.sigma_containment", "C8", "Located frames with position error <= 3 sigma_pos_m "
        "and rotation error <= 3 sigma_rot_deg", "share", "share", ("O5",)),
    Def("locate.sigma_tightness", "C8", "Median error / sigma of located frames, position and "
        "rotation", "median", "ratio", ("O5",),
        parts=(("pos", "median", "ratio"), ("rot", "median", "ratio"))),
    Def("update.sigma_containment", "C1, C4", "Non-rejected inputs (registered and "
        "kept_at_prior) whose committed pose is within 3 of its sigmas of the truth", "share",
        "share"),
    Def("scale.error_metric", "C2", "Scale error with metric priors", "max", "ratio"),
    Def("scale.error_camera_height", "C2", "Scale error with camera height only", "max",
        "ratio"),
    Def("scale.error_by_direction", "C2", "Worst relative scale error over four horizontal "
        "directions (0, 45, 90, 135 deg) of the confirmed keyframes' truth positions (error), "
        "and its verification (within): that error over 3 x the stated scale_sigma_rel, and "
        "with a camera height and no priors |floor_z_m| over 0.02 m (<= 1 passes)", "max",
        "ratio", parts=(("error", "max", "ratio"), ("within", "max", "ratio"))),
    Def("frame.shift", "C1", "Shift of stored keyframes between consecutive versions beyond "
        "gauge_shift (medians)", "median", "", parts=(("pos_m", "median", "m"),
                                                         ("rot_deg", "median", "deg"))),
    Def("update.readable_latency", "C3", "Upload to frame readable in a version (p90 over jobs)",
        "p90", "s"),
    Def("fusion.time_per_frame", "C3", "Fusion time per new frame (p90 over jobs)", "p90", "s"),
    Def("reads.atomic", "C3", "Reads during an update answered from one version", "share",
        "share"),
    Def("jobs.eta_accuracy", "C3", "eta_s of a queued job, and predicted_s of a dry run, within "
        "time_tolerance_rel of the measured time", "share", "share", ("O3|O3-jobs",)),
    Def("fusion.intermediate_interval", "C3", "Longest gap between versions while fusion lasts "
        "longer than the interval", "max", "s", ("O3",)),
    Def("inputs.reason_agreement", "C4", "Rejected inputs whose reason agrees with the truth "
        "(bare wall, too close, no overlap)", "share", "share"),
    Def("revert.latency", "C5", "Revert latency", "max", "s", ("O2",)),
    Def("gate.warping_rejected", "C6", "Warping updates (priors slipping >= 0.3 m against the "
        "truth after their first frame) rejected at the gate parameters", "share", "share",
        ("O9",)),
    Def("gate.others_accepted", "C6", "Clean updates (priors within their sigmas, or none) of the "
        "passing episodes accepted at the gate parameters", "share", "share", ("O9",)),
    Def("grid.floor_plan_agreement", "C9", "Observed grid cells agreeing with the floor plan "
        "(free in rooms, occupied at walls)", "share", "share", ("O6",), True),
    Def("passages.door_recall", "C9", "Observed doors of the floor plan found as passages within "
        "0.3 m", "share", "share", ("O6",), True),
    Def("passages.geometry", "C9", "Found doors with centre_m and width_m within the targets.json "
        "tolerances of the door gap", "share", "share", ("O6",), True),
    Def("grid.floor_observed", "C9", "Free cells with floor_observed where the floor plan has "
        "floor", "share", "share", ("O6",), True),
    Def("grid.camera_cells_free", "C9", "Keyframe camera cells and their near blind ring free",
        "share", "share", ("O6",)),
    Def("grid.floor_false_obstacles", "C9", "Occupied cells, and present objects lower than "
        "0.06 m, on observed open floor of the floor plan in a keyframe's horizontal field of "
        "view, away from every truth object (count)", "sum", "count", ("O6",), True),
    Def("grid.unconfirmed_occupancy", "C9, C10", "Occupied cells, and floor-standing objects, "
        "on open floor of the floor plan inside the view of kept_at_prior frames only and of no "
        "confirmed frame (count)", "sum", "count", ("O6",), True),
    Def("grid.id_stability", "C9", "Regions and passages keeping their id between consecutive "
        "versions where their geometry did not change", "share", "share", ("O6",)),
    Def("features.region_set", "C9", "Frontiers bordering a region, and objects in or on one, "
        "with region set", "share", "share", ("O6",)),
    Def("grid.band_occupied", "C9", "Observed wall cells (obstacles across the band) marked "
        "occupied", "share", "share", ("O6",), True),
    Def("grid.floor_height", "C9", "Floor height error against the surface the robot stands on",
        "max", "m", ("O6",)),
    Def("reads.latency", "C9, C10", "GET grid (5 cm cells) and GET objects on a committed "
        "version, each", "max", "s", ("O6|O8",)),
    Def("objects.support_height", "C10", "Support height error of objects on a surface (median)",
        "median", "m", ("O8",)),
    Def("objects.relations", "C10", "supported_by / inside_of agreeing with the truth's "
        "relations", "share", "share", ("O8",)),
    Def("objects.centre_error", "C10", "Object centre error (median, same-label match within "
        "1 m; box centre, or centre xy and bottom z for base-origin truth)", "median", "m",
        ("O8",)),
    Def("objects.label_agreement", "C10", "Objects whose label agrees with the nearest truth "
        "object", "share", "share", ("O8",)),
    Def("objects.id_stability", "C10", "Truth objects keeping their id between consecutive "
        "versions", "share", "share", ("O8",)),
    Def("analysis.range_scan_error", "C11", "Range-scan error against the floor-plan walls "
        "(median, relative)", "median", "ratio", ("O7",), True),
    Def("interactive.latency_under_load", "C12", "Locate and analyse while an update runs: "
        "ratio of medians to idle, and absolute p90", "max", "",
        ("O5|O7",), parts=(("ratio", "max", "ratio"), ("abs_s", "p90", "s"))),
    Def("locate.latency_idle", "C12", "Locate while idle, per frame of the request (p90)", "p90",
        "s", ("O5",)),
    diag("locate_pos_error_m", "C8", "Median position error of located frames", "median", "m",
         ("O5",)),
    diag("ambiguity_null_on_right", "C8", "ambiguity null on located frames within 0.10 m and 5° "
         "of the truth", "share", "share", ("O5",)),
    diag("ambiguity_rule", "C8", "Non-null ambiguity is a rival farther than 0.10 m or 5°",
         "share", "share", ("O5",)),
    diag("locate_first_s", "C12", "The replay's first idle locate, per frame (warm-up)", "max",
         "s", ("O5",)),
    diag("sweep_yaw_error_deg", "C8", "Keyframe yaw error in look sweeps after the "
         "gravity-aligned fit (median)", "median", "deg"),
    diag("sweep_yaw_span_ratio", "C8", "Map over truth yaw span of each look sweep's keyframes "
         "(median; > 1 over-counts the turn)", "median", "ratio"),
    diag("scale_error_monocular", "C2", "Scale error without priors or heights", "max", "ratio"),
    diag("prior_residual_m", "C1", "Inputs' position residual to their priors (median)",
         "median", "m"),
    diag("prior_residual_deg", "C1", "Inputs' rotation residual to their priors (median)",
         "median", "deg"),
    diag("camera_stand_free", "C9", "Keyframe camera cells alone free", "share", "share",
         ("O6",)),
    Def("contract.inputs", "contracts", "Jobs whose inputs[] has one entry per frame, in request "
        "order", "share", "share", kind="contract"),
    Def("contract.revert", "contracts", "Revert commits a new version", "share", "share",
        ("O2",), kind="contract"),
    Def("contract.band_as_asked", "contracts", "Grid z_band_m equal to the requested band",
        "share", "share", ("O6",), kind="contract"),
    Def("contract.schema_valid", "contracts", "Responses valid against api.schema.json", "share",
        "share", kind="contract"),
    Def("contract.server_timing", "contracts", "Responses carrying Server-Timing", "share",
        "share", kind="contract"),
]
BY_ID = {d.id: d for d in DEFS}
CONTRACT_TARGET = {"op": ">=", "value": 1.0}  # zero violations (evaluation.md "Contracts")
WARPING_SLIP_M = 0.3  # evaluation.md "Gate parameters": a warping update's slip
CLEAN_SIGMAS = 3.0  # ... and a clean one's priors stay within this many sigmas of the truth
DIRECTIONS_DEG = (0.0, 45.0, 90.0, 135.0)  # scale.error_by_direction: the directions measured
SCALE_SIGMAS = 3.0  # ... within this many stated scale sigmas
FLOOR_ZERO_M = 0.02  # ... and with a camera height floor_z_m this near 0
VIEW_RAYS = 64  # grid.unconfirmed_occupancy: rays per camera bounding its view by the walls


@dataclass
class Params:
    """Thresholds of the truth-derived checks (README "Metrics")."""

    align_min_spread_m: float = 0.5
    align_full_rotation: bool = False  # a 3-D rotation instead of a yaw (see alignment)
    match_m: float = 1.0
    too_close_m: float = 0.35
    plain_m: float = 1.5
    overlap_m: float = 1.5
    turn_tol_deg: float = 45.0
    door_m: float = 0.3
    door_centre_tol_m: float = 0.10  # targets.json passages.geometry parameters
    door_width_tol_m: float = 0.10
    locate_abs_m: float = 0.02  # targets.json locate.median_pos_error parameters
    locate_depth_share: float = 0.01
    eta_tolerance: float = 0.3  # targets.json jobs.eta_accuracy time_tolerance_rel
    default_sigma_m: float = 0.20  # a prior sent without sigma (api.md §3, drifting)
    low_object_m: float = 0.06  # grid.floor_false_obstacles: a floor-standing object lower
    probe_interval_s: float = 0.5
    passing: tuple[str, ...] | None = None  # None: from each episode's verdict.md
    worktop_kinds: tuple[str, ...] = ("arm",)  # floor_z_m is judged against the worktop
    worktop_placements: tuple[str, ...] = ("worktop",)  # ... as for meta.placement "worktop"
    labels: LabelRule = field(default_factory=LabelRule)
    sweep_prefixes: tuple[str, ...] = ("look",)  # a sweep: consecutive frames named so ...
    sweep_stand_m: float = 0.25  # ... whose truth positions stay this close to the first
    heights: bool = True


@dataclass
class Samples:
    values: dict[str, list[float]] = field(default_factory=dict)
    missing: dict[str, str] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)

    def add(self, mid: str, value: float | bool) -> None:
        self.values.setdefault(mid, []).append(float(value))


def target_of(d: Def, targets: Targets) -> dict[str, Any] | None:
    if d.kind == "contract":
        return CONTRACT_TARGET
    if d.kind == "diag":
        return {"info": True}
    return targets.get(d.id)


def verdict(d: Def, values: dict[str, list[float]] | None, missing: str | None,
            targets: Targets) -> dict[str, Any]:
    t = target_of(d, targets)
    out: dict[str, Any] = {"contract": d.contract, "description": d.description,
                           "unit": d.unit, "target": t}
    if missing is not None:
        return {**out, "value": None, "n": 0, "status": "not_measurable", "missing": missing}
    vals = values or {}
    value: float | dict[str, float | None] | None
    if d.parts:
        value = {p: reduce(vals.get(f"{d.id}#{p}", []), how) for p, how, _ in d.parts}
        n = sum(len(vals.get(k, [])) for k in d.sample_keys())
    else:
        value = reduce(vals.get(d.id, []), d.reduce)
        n = len(vals.get(d.id, []))
    out.update(value=value, n=n, status=judge(value, t))
    if not d.parts and n and d.reduce not in ("share", "sum"):
        a = np.asarray(vals[d.id])
        out["detail"] = {"median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
                         "max": float(a.max())}
    return out


# ---------------------------------------------------------------------------------------------
def frame_pose(r: dict[str, Any]) -> Arr:
    return mat4(r["T_map_cam"])


def alignment(rec: Record, by_name: dict[str, Frame], min_spread: float,
              full_rotation: bool = False) -> tuple[Sim3, int, float] | None:
    """The map -> world similarity fitted on the keyframes of the first version whose truth
    camera positions span ``min_spread`` (the map frame is fixed by the first update, map.md §1,
    so every later version is in that frame).

    Both frames are gravity-aligned with z up, so the rotation is a yaw (scale, yaw and
    translation: z' = s z + t_z). A full 3-D rotation (``full_rotation``) is ill-posed for camera
    positions at one height along a line: its roll about that line is free, which moves every
    point below the cameras, such as the floor, by up to twice their height."""
    for job in rec.jobs:
        for key in ("provisional_version", "result_version"):
            v = job.job.get(key)
            if v is None or int(v) not in rec.frames:
                continue
            rows = [r for r in rec.frames[int(v)]["frames"] if r["name"] in by_name
                    and by_name[r["name"]].truth is not None]
            if len(rows) < 2:
                continue
            src = np.array([frame_pose(r)[:3, 3] for r in rows])
            dst = np.array([by_name[r["name"]].truth[:3, 3] for r in rows])  # type: ignore
            if np.linalg.norm(dst.max(axis=0) - dst.min(axis=0)) < min_spread:
                continue
            try:
                sim, rms = umeyama(src, dst, min_ratio=1e-3 if full_rotation else math.inf)
            except ValueError:
                continue
            return sim, int(v), rms
    return None


def hfov_deg(cam: dict[str, Any]) -> float:
    return math.degrees(2 * math.atan(cam["width"] / 2.0 / cam["fx"]))


def scene_depth(ep: Episode, plan: FloorPlan | None, frames: list[Frame]) -> float:
    """Median depth the cameras look across: the wall distance along their heading in a floor
    plan, else their height above the worktop or the floor; 1 m without either."""
    d: list[float] = []
    for f in frames:
        T = f.truth
        assert T is not None
        fwd = T[:3, 2]
        if plan is not None and plan.walls and math.hypot(fwd[0], fwd[1]) > 0.5:
            r = ray_cast(plan, T[:2, 3], optical_yaw_deg(T[:3, :3]))
            if r is not None:
                d.append(r)
        elif ep.worktop_z is not None or ep.floor_z is not None:
            base = ep.worktop_z if ep.worktop_z is not None else ep.floor_z
            d.append(float(T[2, 3]) - float(base))  # type: ignore[arg-type]
    return float(np.median(d)) if d else 1.0


def truth_reason(f: Frame, earlier: list[Frame], plan: FloorPlan | None, hfov: float,
                 p: Params) -> str | None:
    """too_close, plain or no_overlap from the floor plan and the truth poses, or None."""
    T = f.truth
    assert T is not None
    fwd = T[:3, 2]
    yaw = optical_yaw_deg(T[:3, :3])
    if plan is not None and plan.walls and math.hypot(fwd[0], fwd[1]) > 0.5:
        ahead = ray_cast(plan, T[:2, 3], yaw)
        if ahead is not None and ahead < p.too_close_m:
            return "too_close"
        bearings = np.linspace(yaw - hfov / 2, yaw + hfov / 2, 9)
        walls = [ray_cast(plan, T[:2, 3], float(b)) for b in bearings]
        if all(w is not None and w < p.plain_m for w in walls):
            seen = False
            for o in plan.objects:
                v = o.centre[:2] - T[:2, 3]
                b = math.degrees(math.atan2(v[1], v[0]))
                if abs(wrap_deg(b - yaw)) < hfov / 2:
                    wall = ray_cast(plan, T[:2, 3], b)
                    seen = seen or wall is None or float(np.linalg.norm(v)) < wall
            if not seen:
                return "plain"
    if earlier and not any(
            np.linalg.norm(e.truth[:3, 3] - T[:3, 3]) < p.overlap_m  # type: ignore[index]
            and math.degrees(math.acos(max(-1.0, min(1.0, float(e.truth[:3, 2] @ fwd)))))  # type: ignore[index]
            < hfov for e in earlier):
        return "no_overlap"
    return None


def layer(grid: dict[str, Any], name: str) -> Arr:
    b = grid["layers"][name]
    dt = np.dtype(b["dtype"]).newbyteorder("<")
    return np.frombuffer(base64.b64decode(b["data"]), dtype=dt).reshape(b["shape"])


class Measure:
    def __init__(self, rec: Record, ep: Episode, plan: FloorPlan | None, raster: Raster | None,
                 truth_objects: list[TruthObject], log: Log, params: Params) -> None:
        self.rec, self.ep, self.plan, self.raster, self.log, self.p = rec, ep, plan, raster, \
            log, params
        self.s = Samples()
        self.by_name = {f.name: f for f in ep.all_frames()}
        self.caps = set(rec.capabilities)
        self.al = alignment(rec, self.by_name, params.align_min_spread_m,
                            params.align_full_rotation)
        latest = max(rec.frames) if rec.frames else None
        self.keyframes = [r for r in rec.frames[latest]["frames"] if r["name"] in self.by_name] \
            if latest is not None else []
        self.truth_objects = truth_objects

    def floor_reference(self) -> float | None:
        """The lowest large support plane in view: the worktop when the robot is placed on one
        (meta.placement) or is of a kind that works on one (the arm), else the floor."""
        on_worktop = self.ep.placement in self.p.worktop_placements or \
            self.ep.kind in self.p.worktop_kinds
        if on_worktop and self.ep.worktop_z is not None:
            return self.ep.worktop_z
        return self.ep.floor_z

    def passed(self) -> bool:
        """A passing episode: named in --passing, else by its verdict.md."""
        if self.p.passing is not None:
            return self.ep.id in self.p.passing
        return self.ep.passed is True

    def run(self) -> Samples:
        s = self.s
        if self.al is not None:
            sim, v, rms = self.al
            s.info["alignment"] = {"version": v, "scale": sim.s, "yaw_deg": sim.yaw_deg,
                                   "translation": sim.t.tolist(), "rms_m": rms}
        else:
            s.info["alignment"] = None
        self.latency()
        self.sweeps()
        self.inputs()
        self.update_sigmas()
        self.scale()
        self.stability()
        if "O5" in self.caps:
            self.locate()
        if "O6" in self.caps and self.rec.grid is not None:
            self.grid()
        if "O6" in self.caps:
            self.grid_ids()
        if "O8" in self.caps and self.rec.objects:
            self.objects()
        if "O7" in self.caps:
            self.analyse()
        if "O9" in self.caps:
            self.gate()
        if "O2" in self.caps:
            if self.rec.revert_s is not None:
                s.add("revert.latency", self.rec.revert_s)
            if self.rec.revert_ok is not None:
                s.add("contract.revert", self.rec.revert_ok)
        for c in self.log.calls:
            if c.schema is not None:
                s.add("contract.schema_valid", not c.schema_errors)
            s.add("contract.server_timing", c.server_timing)
        s.info["schema_errors"] = [f"{c.route}: {e}" for c in self.log.calls
                                   for e in c.schema_errors][:50]
        return s

    # ---- always -----------------------------------------------------------------------------
    def latency(self) -> None:
        s = self.s
        for j in self.rec.jobs:
            if j.t_readable is not None:
                s.add("update.readable_latency", j.t_readable - j.t_start)
            if j.job.get("state") != "committed" or j.t_end is None:
                continue
            n = max(1, sum(1 for i in j.job.get("inputs") or [] if i["status"] != "rejected"))
            start = j.t_fusing if j.t_fusing is not None else j.t_start
            s.add("fusion.time_per_frame", (j.t_end - start) / n)
            if has(self.caps, "O3|O3-jobs"):
                measured = j.t_end - j.t_posted
                if j.eta_queued is not None and measured > 0:
                    s.add("jobs.eta_accuracy",
                          abs(j.eta_queued - measured) <= self.p.eta_tolerance * measured)
                if j.dry is not None:
                    pred = j.dry["predicted_s"]
                    total = float(pred["registration"]) + float(pred["fusion"])
                    m = j.t_end - j.t_start
                    s.add("jobs.eta_accuracy", abs(total - m) <= self.p.eta_tolerance * m)
            if "O3" in self.caps and j.t_fusing is not None and j.t_end - j.t_fusing > 10.0:
                seen_at = sorted(self.rec.probe_versions + j.versions_seen)
                v0 = max([v for t, v in seen_at if t <= j.t_fusing], default=-1)
                times, seen = [], v0
                for t, v in seen_at:
                    if j.t_fusing < t <= j.t_end and v > seen:
                        times.append(t)
                        seen = v
                gaps = np.diff([j.t_fusing, *times, j.t_end])
                s.add("fusion.intermediate_interval",
                      max(0.0, float(gaps.max()) - self.p.probe_interval_s))
        for ok in self.rec.consistency:
            s.add("reads.atomic", ok)

    def sweeps(self) -> None:
        """Per sweep (consecutive look frames from one stand, in time order): each keyframe's
        map yaw, after the gravity-aligned fit, against truth; and the map yaw span (sum of the
        turns between consecutive keyframes) over the truth span."""
        if self.al is None or not self.keyframes:
            return
        sim = self.al[0]
        kf = {r["name"]: frame_pose(r) for r in self.keyframes}
        ordered = sorted(self.ep.all_frames(),
                         key=lambda f: (f.time if f.time is not None else math.inf, f.name))
        groups: list[list[Frame]] = []
        prev_in = False
        for f in ordered:
            look = Path(f.file).stem.startswith(self.p.sweep_prefixes)
            assert f.truth is not None
            if look and prev_in and np.linalg.norm(
                    f.truth[:3, 3] - groups[-1][0].truth[:3, 3]) <= self.p.sweep_stand_m:  # type: ignore[index]
                groups[-1].append(f)
            elif look:
                groups.append([f])
            prev_in = look
        report = []
        for g in groups:
            rows = [(optical_yaw_deg(sim.R @ kf[f.name][:3, :3]),
                     optical_yaw_deg(f.truth[:3, :3])) for f in g if f.name in kf]  # type: ignore[index]
            if len(rows) < 2:
                continue
            errs = [abs(wrap_deg(m - t)) for m, t in rows]
            span_m = sum(wrap_deg(b[0] - a[0]) for a, b in itertools.pairwise(rows))
            span_t = sum(wrap_deg(b[1] - a[1]) for a, b in itertools.pairwise(rows))
            ratio = span_m / span_t if abs(span_t) >= 1.0 else None
            for e in errs:
                self.s.add("diag.sweep_yaw_error_deg", e)
            if ratio is not None:
                self.s.add("diag.sweep_yaw_span_ratio", ratio)
            report.append({"first": g[0].name, "frames": len(g), "keyframes": len(rows),
                           "median_deg": float(np.median(errs)), "max_deg": float(max(errs)),
                           "span_map_deg": span_m, "span_truth_deg": span_t, "ratio": ratio})
        self.s.info["sweeps"] = report

    def inputs(self) -> None:
        s = self.s
        order = {n: i for i, n in enumerate(self.rec.sent)}
        for j in self.rec.jobs:
            inputs = j.job.get("inputs")
            if inputs is None:
                continue
            s.add("contract.inputs", [i["name"] for i in inputs] == j.names)
            for i in inputs:
                for key, mid in (("prior_residual_m", "diag.prior_residual_m"),
                                 ("prior_residual_deg", "diag.prior_residual_deg")):
                    if i.get(key) is not None:
                        s.add(mid, i[key])
                if i["status"] != "rejected" or i["name"] not in order:
                    continue
                f = self.by_name[i["name"]]
                earlier = [self.by_name[n] for n in self.rec.sent[:order[i["name"]]]]
                truth = truth_reason(f, earlier, self.plan,
                                     hfov_deg(self.ep.cameras[f.camera_id]), self.p)
                reason = i["reason"]
                walls = self.plan is not None and bool(self.plan.walls)
                judged = {"no_overlap", "plain", "too_close"} if walls else {"no_overlap"}
                if reason in judged or truth in judged or (walls and reason == "few_features"):
                    s.add("inputs.reason_agreement", reason == truth or
                          (reason == "few_features" and truth == "plain"))

    def slip(self, j: JobObs) -> tuple[float, bool] | None:
        """A job's prior slip against the truth after its first frame (the largest position
        offset of its later priors once its first prior is put on its truth pose), and whether
        every prior stays within ``CLEAN_SIGMAS`` of its position sigma (a clean update,
        evaluation.md "Gate parameters"); None without two priors with truth."""
        rows = [(self.by_name[n], j.priors[n]) for n in j.names
                if n in j.priors and n in self.by_name and self.by_name[n].truth is not None]
        if len(rows) < 2:
            return None
        f0, p0 = rows[0]
        D = f0.truth @ inv_T(mat4(p0["T_map_cam"]))  # type: ignore[operator]
        worst, within = 0.0, True
        for f, p in rows[1:]:
            d = float(np.linalg.norm((D @ mat4(p["T_map_cam"]))[:3, 3] - f.truth[:3, 3]))  # type: ignore[index]
            worst = max(worst, d)
            within = within and d <= CLEAN_SIGMAS * float(p.get("sigma_pos_m",
                                                                 self.p.default_sigma_m))
        return worst, within

    def gate(self) -> None:
        """C6 at the gate parameters (``rpl_replay.GATE``): a warping update (priors slipping by
        WARPING_SLIP_M or more) must be rejected; a clean one (priors within CLEAN_SIGMAS of
        their sigmas, or no priors) of a passing episode accepted. Updates neither warping nor
        clean are left out."""
        for j in self.rec.jobs:
            if j.job.get("state") not in ("committed", "rejected"):
                continue
            sl = self.slip(j)
            if sl is not None and sl[0] >= WARPING_SLIP_M:
                self.s.add("gate.warping_rejected", j.job["state"] == "rejected")
            elif (sl is None or sl[1]) and self.passed():
                self.s.add("gate.others_accepted", j.job["state"] == "committed")

    def update_sigmas(self) -> None:
        """update.sigma_containment: each non-rejected input's committed pose against its truth
        (after the map -> world alignment), within 3 of its sigmas (a kept_at_prior input's
        sigmas are its prior's)."""
        if self.al is None:
            return
        sim = self.al[0]
        for j in self.rec.jobs:
            for i in j.job.get("inputs") or []:
                if i["status"] == "rejected" or i.get("T_map_cam") is None \
                        or i.get("sigma_pos_m") is None or i["name"] not in self.by_name:
                    continue
                T = self.by_name[i["name"]].truth
                if T is None:
                    continue
                W = sim.pose(mat4(i["T_map_cam"]))
                e_pos = float(np.linalg.norm(W[:3, 3] - T[:3, 3]))
                e_rot = angle_deg(W[:3, :3], T[:3, :3])
                self.s.add("update.sigma_containment",
                           e_pos <= 3 * sim.s * float(i["sigma_pos_m"])
                           and e_rot <= 3 * float(i["sigma_rot_deg"]))

    def scale(self) -> None:
        if self.al is None:
            return
        err = abs(self.al[0].s - 1.0)
        if self.rec.mode == "metric":
            self.s.add("scale.error_metric", err)
        elif self.p.heights and any(self.by_name[n].height_m is not None
                                    for n in self.rec.sent):
            self.s.add("scale.error_camera_height", err)
        else:
            self.s.add("diag.scale_error_monocular", err)
        self.scale_by_direction()

    def scale_by_direction(self) -> None:
        """scale.error_by_direction (map.md §1): the linear map from the confirmed keyframes'
        horizontal map positions to their truth positions (after the yaw of the alignment, no
        scale), its relative scale error along each of DIRECTIONS_DEG over SCALE_SIGMAS of the
        stated scale_sigma_rel (the map's metres are its stated scale); with keyframes along a
        line, only along it. With camera heights and no priors, |floor_z_m| over FLOOR_ZERO_M too
        (the floor is z 0, map.md §1). Samples: the worst error (``error``) and the worst
        ratio (``within``, <= 1 passes)."""
        if self.al is None or not self.rec.frames:
            return
        latest = self.rec.frames[max(self.rec.frames)]
        sigma = latest.get("scale_sigma_rel")
        rows = [r for r in latest["frames"] if r.get("pose_state", "fused") != "kept_at_prior"
                and r["name"] in self.by_name and self.by_name[r["name"]].truth is not None]
        if sigma is None or len(rows) < 2:
            return
        M = np.array([frame_pose(r)[:2, 3] for r in rows])
        W = np.array([self.by_name[r["name"]].truth[:2, 3] for r in rows])  # type: ignore[index]
        Mc, Wc = M - M.mean(axis=0), W - W.mean(axis=0)
        _, sv, Vt = np.linalg.svd(Mc, full_matrices=False)
        if sv[0] < 1e-6:
            return
        allowed = SCALE_SIGMAS * max(float(sigma), 1e-6)
        if sv[-1] / math.sqrt(len(rows)) < self.p.align_min_spread_m / 4.0:
            u = Vt[0]  # along a line: its direction only
            x = Mc @ u
            a = (x @ Wc) / float(x @ x)
            errors = [abs(float(np.linalg.norm(a)) - 1.0)]
        else:
            A, *_ = np.linalg.lstsq(Mc, Wc, rcond=None)
            errors = []
            for deg in DIRECTIONS_DEG:
                u = np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])
                errors.append(abs(float(np.linalg.norm(u @ A)) - 1.0))
        ratios = [e / allowed for e in errors]
        heights = self.p.heights and self.rec.mode != "metric" and any(
            self.by_name[n].height_m is not None for n in self.rec.sent)
        if heights and self.rec.grid is not None:
            ratios.append(abs(float(self.rec.grid["floor_z_m"])) / FLOOR_ZERO_M)
        self.s.info["scale_by_direction"] = ratios
        self.s.add("scale.error_by_direction#error", max(errors))
        self.s.add("scale.error_by_direction#within", max(ratios))

    def stability(self) -> None:
        gauge: dict[int, dict[str, Any]] = {}
        for j in self.rec.jobs:
            g = (j.job.get("quality") or {}).get("gauge_shift")
            for key in ("provisional_version", "result_version"):
                if j.job.get(key) is not None and g:
                    gauge[int(j.job[key])] = g
        scale = self.al[0].s if self.al is not None else 1.0
        versions = sorted(self.rec.frames)
        for a, b in itertools.pairwise(versions):
            after = {r["name"]: r for r in self.rec.frames[b]["frames"]}
            g = gauge.get(b, {"translation_m": 0.0, "rotation_deg": 0.0})
            for r in self.rec.frames[a]["frames"]:
                if r.get("pose_state", "fused") != "fused" or r["name"] not in after:
                    continue
                Ta, Tb = frame_pose(r), frame_pose(after[r["name"]])
                d = float(np.linalg.norm(Tb[:3, 3] - Ta[:3, 3]))
                self.s.add("frame.shift#pos_m",
                           max(0.0, d - float(g["translation_m"])) * scale)
                self.s.add("frame.shift#rot_deg", max(0.0, angle_deg(Ta[:3, :3], Tb[:3, :3])
                                                   - float(g["rotation_deg"])))

    # ---- O5 ---------------------------------------------------------------------------------
    def locate(self) -> None:
        s, rec = self.s, self.rec
        for x in rec.idle_locate_s:
            s.add("locate.latency_idle", x)
        if rec.first_locate_s is not None:
            s.add("diag.locate_first_s", rec.first_locate_s)
        for x in rec.busy_locate_s:
            s.add("interactive.latency_under_load#abs_s", x)
        if rec.idle_locate_s and rec.busy_locate_s:
            s.add("interactive.latency_under_load#ratio",
                  float(np.median(rec.busy_locate_s)) / float(np.median(rec.idle_locate_s)))
        kf_frames = [self.by_name[r["name"]] for r in self.keyframes]
        depth = scene_depth(self.ep, self.plan, kf_frames) if kf_frames else 0.0
        tol = max(self.p.locate_abs_m, self.p.locate_depth_share * depth)
        s.info["locate_tolerance_m"] = tol
        for lo in rec.locates:
            kfs = [self.by_name[r["name"]] for r in
                   rec.frames.get(int(lo.response["map_version"]), {"frames": []})["frames"]
                   if r["name"] in self.by_name]
            for r in lo.response["frames"]:
                f = self.by_name[r["name"]]
                T = f.truth
                assert T is not None
                near = any(k.name != f.name and np.linalg.norm(k.truth[:3, 3] - T[:3, 3]) <= 0.3  # type: ignore[index]
                           and angle_deg(k.truth[:3, :3], T[:3, :3]) <= 15.0  # type: ignore[index]
                           for k in kfs)
                located = r["status"] == "located"
                if near:
                    s.add("locate.recall", located)
                amb = r.get("ambiguity")
                if amb is not None:
                    s.add("diag.ambiguity_rule", amb["pos_m"] > 0.10 or amb["rot_deg"] > 5.0)
                if located and self.al is not None:
                    W = self.al[0].pose(mat4(r["T_map_cam"]))
                    e_pos = float(np.linalg.norm(W[:3, 3] - T[:3, 3]))
                    e_rot = angle_deg(W[:3, :3], T[:3, :3])
                    s.add("diag.locate_pos_error_m", e_pos)
                    s.add("locate.median_pos_error", e_pos / tol)
                    if e_pos > 0.10 or e_rot > 5.0:
                        s.add("locate.ambiguity_flagged", amb is not None)
                    sp, sr = r.get("sigma_pos_m"), r.get("sigma_rot_deg")
                    if sp is not None and sr is not None:
                        sp_w = self.al[0].s * float(sp)
                        if e_pos <= 0.05 and not r.get("rivals"):
                            s.add("locate.sigma_right_median", sp_w)
                        s.add("locate.sigma_containment",
                              e_pos <= 3 * sp_w and e_rot <= 3 * float(sr))
                        if sp_w > 0:
                            s.add("locate.sigma_tightness#pos", e_pos / sp_w)
                        if float(sr) > 0:
                            s.add("locate.sigma_tightness#rot", e_rot / float(sr))
                    if e_pos <= 0.10 and e_rot <= 5.0:
                        s.add("diag.ambiguity_null_on_right", amb is None)
                nv = r.get("nearest_view")
                # judged only for frames sent with a prior: turn_deg is null without one
                if r["status"] == "rejected" and r.get("reason") == "no_overlap" and nv \
                        and nv["name"] in self.by_name and lo.with_prior:
                    kt = self.by_name[nv["name"]].truth
                    assert kt is not None
                    want = wrap_deg(optical_yaw_deg(kt[:3, :3]) - optical_yaw_deg(T[:3, :3]))
                    turn = nv.get("turn_deg")
                    s.add("locate.nearest_view", turn is not None and
                          abs(wrap_deg(float(turn) - want)) <= self.p.turn_tol_deg)

    # ---- O6 ---------------------------------------------------------------------------------
    def grid(self) -> None:
        s, g = self.s, self.rec.grid
        assert g is not None
        if self.rec.grid_s is not None:
            s.add("reads.latency", self.rec.grid_s)
        s.add("contract.band_as_asked", bool(np.allclose(g["z_band_m"], self.rec.grid_band,
                                                   atol=1e-6)))
        state = layer(g, "state")
        obs = layer(g, "observations")
        floor_obs = layer(g, "floor_observed")
        rows, cols = state.shape
        cell = float(g["cell_m"])
        ox, oy = (float(v) for v in g["origin_m"])
        fz = float(g["floor_z_m"])
        frames = self.rec.frames.get(int(g["map_version"])) or self.rec.frames[max(self.rec.frames)]
        cx = ox + (np.arange(cols) + 0.5) * cell
        cy = oy + (np.arange(rows) + 0.5) * cell
        gx, gy = np.meshgrid(cx, cy)
        sim = self.al[0] if self.al is not None else None
        floor_plan = self.plan is not None and self.raster is not None and self.plan.has_rooms
        W = cls = None
        if sim is not None and floor_plan:
            assert self.raster is not None
            P = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, fz)])
            W = sim.points(P)[:, :2]
            cls = self.raster.classify(W)
        for r in frames["frames"]:
            T = frame_pose(r)
            cam = self.ep.cameras.get(r["camera_id"])
            p = T[:3, 3]
            radius = 0.0
            if cam is not None:
                d = T[:3, :3] @ np.array([0.0, (cam["height"] - 1 - cam["cy"]) / cam["fy"], 1.0])
                if d[2] < -1e-6 and (fz - p[2]) / d[2] > 0:
                    radius = float((fz - p[2]) / d[2] * math.hypot(d[0], d[1]))
            c, rr = math.floor((p[0] - ox) / cell), math.floor((p[1] - oy) / cell)
            if not (0 <= c < cols and 0 <= rr < rows):
                s.add("grid.camera_cells_free", False)
                s.add("diag.camera_stand_free", False)
                continue
            s.add("diag.camera_stand_free", bool(state[rr, c] == 1))
            disc = np.hypot(gx - p[0], gy - p[1]) <= max(0.0, radius - cell)
            if cls is not None:  # floor-plan walls stay occupied
                disc &= ~np.isin(cls.reshape(rows, cols), (WALL, BORDER, OUTSIDE))
            disc[rr, c] = True
            s.add("grid.camera_cells_free", bool((state[disc] == 1).all()))
        polys = [np.asarray(reg["polygon_m"], dtype=float) for reg in g["regions"]]
        for fr in g["frontiers"]:
            c2 = np.asarray([fr["centre_m"]], dtype=float)
            if any(float(poly_distance(c2, poly)[0]) <= 2 * cell for poly in polys):
                s.add("features.region_set", fr["region"] is not None)
        if "O8" in self.caps and self.rec.objects:
            for o in self.rec.objects[max(self.rec.objects)]["objects"]:
                c2 = np.asarray([o["centre_m"][:2]], dtype=float)
                if o.get("state") != "struck" and any(
                        float(poly_distance(c2, poly)[0]) <= cell for poly in polys):
                    s.add("features.region_set", o["region"] is not None)
        if sim is None:
            return
        ref = self.floor_reference()
        if ref is not None and frames["frames"]:
            mean = np.mean([frame_pose(r)[:2, 3] for r in frames["frames"]], axis=0)
            # floor_z_m is a map-frame z: through the same similarity as the cameras
            s.add("grid.floor_height",
                  abs(float(sim.point([mean[0], mean[1], fz])[2]) - ref))
            s.info["floor_z_world_m"] = float(sim.point([mean[0], mean[1], fz])[2])
            s.info["floor_reference_m"] = ref
        if W is None or cls is None or self.plan is None:
            return
        st, ob, fo = state.ravel(), obs.ravel(), floor_obs.ravel()
        floorish = (cls == FLOOR) | (cls == DOOR)
        seen = st != 0
        for ok in np.concatenate([st[seen & floorish] == 1, st[seen & (cls == WALL)] == 2]):
            s.add("grid.floor_plan_agreement", bool(ok))
        for ok in fo[(st == 1) & floorish & (fo != 2)] == 1:  # 2: free by rule, not seen
            s.add("grid.floor_observed", bool(ok))
        for ok in st[(cls == WALL) & (ob > 0)] == 2:
            s.add("grid.band_occupied", bool(ok))
        s.add("grid.floor_false_obstacles",
              self.false_obstacles(W, cls, st == 2, ob > 0, frames["frames"], fz))
        s.add("grid.unconfirmed_occupancy",
              self.unconfirmed(W, cls, st == 2, frames["frames"], fz))
        passages = [(sim.point([*pg["centre_m"], fz])[:2], sim.s * float(pg["width_m"]))
                    for pg in g["passages"]]
        for door in self.plan.doors:
            near = np.linalg.norm(W - door.centre, axis=1) <= self.p.door_m
            if not (near & seen).any():
                continue
            found = [(float(np.linalg.norm(c - door.centre)), w) for c, w in passages
                     if np.linalg.norm(c - door.centre) <= self.p.door_m]
            s.add("passages.door_recall", bool(found))
            if found:
                dist, width = min(found)
                s.add("passages.geometry", dist <= self.p.door_centre_tol_m
                      and abs(width - door.width) <= self.p.door_width_tol_m)

    def false_obstacles(self, W: Arr, cls: Arr, occupied: Arr, observed: Arr,
                        frames: list[dict[str, Any]], fz: float) -> int:
        """Occupied cells, and present objects lower than ``low_object_m``, on open floor: floor
        of the floor plan, more than one cell from every truth object's footprint, observed, in
        some keyframe camera's horizontal field of view (the band's frustum, projected)."""
        assert self.al is not None
        sim = self.al[0]
        cell = float(self.rec.grid["cell_m"]) * sim.s  # type: ignore[index]
        open_floor = (cls == FLOOR) & observed
        for t in self.truth_objects:
            half = (np.asarray(t.size[:2], dtype=float) / 2.0) if t.size is not None \
                else np.zeros(2)
            near = np.all(np.abs(W - t.centre[:2]) <= half + cell, axis=1)
            open_floor &= ~near
        seen = np.zeros(len(W), dtype=bool)
        for r in frames:
            cam = self.ep.cameras.get(r["camera_id"])
            if cam is None:
                continue
            T = sim.pose(frame_pose(r))
            v = W - T[:2, 3]
            b = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
            d = (b - optical_yaw_deg(T[:3, :3]) + 180.0) % 360.0 - 180.0
            seen |= np.abs(d) <= hfov_deg(cam) / 2.0
        count = int((occupied & open_floor & seen).sum())
        if "O8" in self.caps and self.rec.objects:
            final = self.rec.objects[max(self.rec.objects)]
            for o in final["objects"]:
                if o.get("state") == "struck" or float(o["size_m"][2]) >= self.p.low_object_m:
                    continue
                c = o["centre_m"]
                if abs(float(c[2]) - float(o["size_m"][2]) / 2.0 - fz) > self.p.low_object_m:
                    continue  # not floor-standing
                w = sim.point(c)[:2]
                i = int(np.argmin(np.linalg.norm(W - w, axis=1)))
                count += int(bool(open_floor[i] and seen[i]))
        return count

    def in_view(self, W: Arr, r: dict[str, Any]) -> Arr:
        """The world cells ``W`` (x, y) inside a frame's view: in its horizontal field of view
        and nearer than the floor plan's wall along their bearing (``VIEW_RAYS`` rays)."""
        assert self.al is not None and self.plan is not None
        cam = self.ep.cameras.get(r["camera_id"])
        if cam is None:
            return np.zeros(len(W), dtype=bool)
        T = self.al[0].pose(frame_pose(r))
        yaw, half = optical_yaw_deg(T[:3, :3]), hfov_deg(cam) / 2.0
        v = W - T[:2, 3]
        d = (np.degrees(np.arctan2(v[:, 1], v[:, 0])) - yaw + 180.0) % 360.0 - 180.0
        bearings = np.linspace(-half, half, VIEW_RAYS)
        walls = np.array([ray_cast(self.plan, T[:2, 3], yaw + float(b)) or math.inf
                          for b in bearings])
        reach = walls[np.clip(np.round((d + half) / (2 * half) * (VIEW_RAYS - 1)).astype(int),
                              0, VIEW_RAYS - 1)]
        out: Arr = (np.abs(d) <= half) & (np.linalg.norm(v, axis=1) <= reach)
        return out

    def unconfirmed(self, W: Arr, cls: Arr, occupied: Arr, frames: list[dict[str, Any]],
                    fz: float) -> int:
        """grid.unconfirmed_occupancy (map.md §2, §6): occupied cells, and floor-standing
        objects, on open floor of the floor plan (more than one cell from every truth object)
        inside the view of kept_at_prior frames only, and of no image-confirmed frame."""
        assert self.al is not None
        sim = self.al[0]
        cell = float(self.rec.grid["cell_m"]) * sim.s  # type: ignore[index]
        open_floor = cls == FLOOR
        for t in self.truth_objects:
            half = (np.asarray(t.size[:2], dtype=float) / 2.0) if t.size is not None \
                else np.zeros(2)
            open_floor &= ~np.all(np.abs(W - t.centre[:2]) <= half + cell, axis=1)
        kept = np.zeros(len(W), dtype=bool)
        confirmed = np.zeros(len(W), dtype=bool)
        for r in frames:
            if r.get("pose_state", "fused") == "kept_at_prior":
                kept |= self.in_view(W, r)
            else:
                confirmed |= self.in_view(W, r)
        only = open_floor & kept & ~confirmed
        count = int((occupied & only).sum())
        if "O8" in self.caps and self.rec.objects:
            for o in self.rec.objects[max(self.rec.objects)]["objects"]:
                c = o["centre_m"]
                if o.get("state") == "struck" or abs(
                        float(c[2]) - float(o["size_m"][2]) / 2.0 - fz) > self.p.low_object_m:
                    continue  # not floor-standing
                i = int(np.argmin(np.linalg.norm(W - sim.point(c)[:2], axis=1)))
                count += int(bool(only[i]))
        return count

    def grid_ids(self) -> None:
        """grid.id_stability over consecutive versions' grids (``grid_id_matches``)."""
        for a, b in itertools.pairwise(sorted(self.rec.grids)):
            for ok in grid_id_matches(self.rec.grids[a], self.rec.grids[b]):
                self.s.add("grid.id_stability", ok)

    # ---- O8 ---------------------------------------------------------------------------------
    def compared(self, o: dict[str, Any], t: TruthObject) -> Arr:
        """The world point of map object ``o`` that truth ``t``'s position stands for: the box
        centre, or for a base-origin truth the centre xy at the box bottom."""
        assert self.al is not None
        c = [float(v) for v in o["centre_m"]]
        if t.origin == "base":
            c[2] -= float(o["size_m"][2]) / 2.0
        return self.al[0].point(c)

    def error(self, o: dict[str, Any], t: TruthObject) -> float:
        return float(np.linalg.norm(self.compared(o, t) - t.centre))

    def match(self, objs: list[dict[str, Any]], same: bool) -> dict[int, TruthObject]:
        """Map object id -> nearest truth object within match_m (with the same label if
        ``same``)."""
        out: dict[int, TruthObject] = {}
        for o in objs:
            cands = [(self.error(o, t), i, t)
                     for i, t in enumerate(self.truth_objects)
                     if not same or self.p.labels.same(o["label"], t.label)]
            cands = [c for c in cands if c[0] <= self.p.match_m]
            if cands:
                out[int(o["id"])] = min(cands, key=lambda c: (c[0], c[1]))[2]
        return out

    def objects(self) -> None:
        s = self.s
        if self.rec.objects_s is not None:
            s.add("reads.latency", self.rec.objects_s)
        if self.al is None or not self.truth_objects:
            return
        sim = self.al[0]
        final = self.rec.objects[max(self.rec.objects)]
        live = [o for o in final["objects"] if o.get("state") != "struck"]
        nearest = self.match(live, same=False)
        for o in live:
            if int(o["id"]) in nearest:
                s.add("objects.label_agreement",
                      self.p.labels.same(o["label"], nearest[int(o["id"])].label))
        matched = self.match(live, same=True)
        for o in live:
            t = matched.get(int(o["id"]))
            if t is not None:
                s.add("objects.centre_error", self.error(o, t))
        surfaces = {int(x["id"]): x for x in final.get("surfaces", [])}
        by_truth = {t.id: oid for oid, t in matched.items()}
        for o in live:
            t = matched.get(int(o["id"]))
            if t is None:
                continue
            sup, ins = o.get("supported_by"), o.get("inside_of")
            if sup and sup["kind"] == "surface" and int(sup["id"]) in surfaces \
                    and t.support_z is not None:
                c = o["centre_m"]
                z = sim.point([c[0], c[1], surfaces[int(sup["id"])]["height_m"]])[2]
                s.add("objects.support_height", abs(float(z) - t.support_z))
            if t.parent is not None:
                want = by_truth.get(t.parent)
                if t.inside:
                    ok = bool(ins) and int(ins["id"]) == want
                else:
                    ok = bool(sup) and (sup["kind"] == "surface" or int(sup["id"]) == want)
                s.add("objects.relations", ok)
            elif t.support_z is not None:
                s.add("objects.relations",
                      bool(sup) and sup["kind"] == "surface" and not ins)
        s.info["objects"] = self.object_report(live, nearest, matched, surfaces)
        versions = sorted(self.rec.objects)
        for a, b in itertools.pairwise(versions):
            ma = {t.id: oid for oid, t in self.match(self.rec.objects[a]["objects"], True).items()}
            mb = {t.id: oid for oid, t in self.match(self.rec.objects[b]["objects"], True).items()}
            for tid in sorted(set(ma) & set(mb)):
                s.add("objects.id_stability", ma[tid] == mb[tid])

    def object_report(self, live: list[dict[str, Any]], nearest: dict[int, TruthObject],
                      matched: dict[int, TruthObject],
                      surfaces: dict[int, dict[str, Any]]) -> dict[str, Any]:
        """Per truth object: the map object matched to it (same label within match_m, else the
        nearest within match_m), the centre error, the label agreement and the support-height
        error; then the map objects matched to no truth object. No images, no paths."""
        assert self.al is not None
        sim, rule = self.al[0], self.p.labels
        by_id = {int(o["id"]): o for o in live}

        def world(o: dict[str, Any]) -> list[float]:
            return [round(float(v), 4) for v in sim.point(o["centre_m"])]

        def best(pairs: dict[int, TruthObject], t: TruthObject) -> int | None:
            ids = [oid for oid, tt in pairs.items() if tt is t]
            return min(ids, key=lambda i: self.error(by_id[i], t), default=None)

        truth_rows, used = [], set()
        for t in self.truth_objects:
            oid = best(matched, t)
            if oid is None:
                oid = best(nearest, t)
            row: dict[str, Any] = {
                "name": t.id, "label": t.label, "canonical": rule.canon(t.label),
                "centre_world": [round(float(v), 4) for v in t.centre],
                "size_world": None if t.size is None else [float(v) for v in t.size],
                "origin": t.origin, "matched": None, "centre_error_m": None,
                "xy_error_m": None, "bottom_z_error_m": None, "centre_z_error_m": None,
                "label_agreed": None,
                "support_height_error_m": None}
            if oid is not None:
                o = by_id[oid]
                used.add(oid)
                row["matched"] = {
                    "id": oid, "label": o["label"], "canonical": rule.canon(o["label"]),
                    "detected_as": o.get("detected_as"), "centre_world": world(o),
                    "size": o.get("size_m"), "mean_rgb": o.get("mean_rgb"),
                    "state": o.get("state"), "supported_by": o.get("supported_by"),
                    "inside_of": o.get("inside_of")}
                d = self.compared(o, t) - t.centre
                row["centre_error_m"] = float(np.linalg.norm(d))
                row["xy_error_m"] = float(np.linalg.norm(d[:2]))
                row["bottom_z_error_m" if t.origin == "base" else "centre_z_error_m"] = \
                    abs(float(d[2]))
                row["label_agreed"] = rule.same(o["label"], t.label)
                sup = o.get("supported_by")
                if sup and sup["kind"] == "surface" and int(sup["id"]) in surfaces \
                        and t.support_z is not None:
                    c = o["centre_m"]
                    z = sim.point([c[0], c[1], surfaces[int(sup["id"])]["height_m"]])[2]
                    row["support_height_error_m"] = abs(float(z) - t.support_z)
            truth_rows.append(row)
        unmatched = [{"id": int(o["id"]), "label": o["label"], "canonical": rule.canon(o["label"]),
                      "centre_world": world(o)} for o in live if int(o["id"]) not in used]
        return {"truth": truth_rows, "unmatched_map": unmatched}

    # ---- O7 ---------------------------------------------------------------------------------
    def analyse(self) -> None:
        s, rec = self.s, self.rec
        for x in rec.busy_analyse_s:
            s.add("interactive.latency_under_load#abs_s", x)
        if rec.idle_analyse_s and rec.busy_analyse_s:
            s.add("interactive.latency_under_load#ratio",
                  float(np.median(rec.busy_analyse_s)) / float(np.median(rec.idle_analyse_s)))
        if self.al is None or self.plan is None or not self.plan.walls:
            return
        sim = self.al[0]
        for _, T, body in rec.analyses:
            if body.get("frame") != "map":
                continue
            origin = sim.point(mat4(T)[:3, 3])[:2]
            for beam in body.get("range_scan") or []:
                if beam["kind"] != "obstacle" or beam["range_m"] is None:
                    continue
                truth = ray_cast(self.plan, origin, float(beam["bearing_deg"]) + sim.yaw_deg)
                if truth:
                    s.add("analysis.range_scan_error",
                          abs(sim.s * float(beam["range_m"]) - truth) / truth)


def grid_id_matches(ga: dict[str, Any], gb: dict[str, Any]) -> list[bool]:
    """Per region (and passage) of grid ``ga`` whose geometry grid ``gb`` keeps (a region of the
    same area within 2 % and centroid within one cell; a passage of the same centre and width
    within one cell): whether it kept its id."""
    out = []
    cell = float(ga["cell_m"])
    for ra in ga["regions"]:
        ca = np.mean(np.asarray(ra["polygon_m"], dtype=float), axis=0)
        same = [rb for rb in gb["regions"]
                if abs(float(rb["area_m2"]) - float(ra["area_m2"])) <= 0.02 * float(ra["area_m2"])
                and np.linalg.norm(np.mean(np.asarray(rb["polygon_m"], dtype=float), axis=0)
                                   - ca) <= cell]
        if same:
            out.append(any(rb["id"] == ra["id"] for rb in same))
    for pa in ga["passages"]:
        same = [pb for pb in gb["passages"]
                if np.linalg.norm(np.asarray(pb["centre_m"]) - np.asarray(pa["centre_m"])) <= cell
                and abs(float(pb["width_m"]) - float(pa["width_m"])) <= cell]
        if same:
            out.append(any(pb["id"] == pa["id"] for pb in same))
    return out


def has(caps: set[str], need: str) -> bool:
    """``need`` is one group or alternatives joined by ``|`` (e.g. ``O3|O3-jobs``)."""
    return any(c in caps for c in need.split("|"))


def missing_for(d: Def, caps: set[str], plan: FloorPlan | None) -> str | None:
    lack = [c for c in d.needs if not has(caps, c)]
    if lack:
        return "capability " + ", ".join(lack)
    if d.floorplan and (plan is None or not plan.has_rooms):
        return "floor plan (house JSON with rooms)"
    return None


def applies(d: Def, mode: str, episode: str, params: Params) -> bool:
    if d.id in ("scale.error_metric", "diag.prior_residual_m", "diag.prior_residual_deg"):
        return mode == "metric"
    if d.id in ("scale.error_camera_height", "diag.scale_error_monocular"):
        return mode == "none"
    return True


def run_metrics(samples: Samples, caps: set[str], plan: FloorPlan | None, mode: str,
                episode: str, params: Params, targets: Targets) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for d in DEFS:
        if not applies(d, mode, episode, params):
            continue
        miss = missing_for(d, caps, plan)
        out[d.id] = verdict(d, samples.values, miss, targets)
    return out


def overall(runs: list[tuple[Samples, dict[str, dict[str, Any]]]],
            targets: Targets) -> dict[str, dict[str, Any]]:
    """Each metric over the pooled samples of the runs where it was measurable."""
    out: dict[str, dict[str, Any]] = {}
    for d in DEFS:
        present = [(smp, m[d.id]) for smp, m in runs if d.id in m]
        if not present:
            continue
        measurable = [smp for smp, r in present if r["status"] != "not_measurable"]
        if not measurable:
            out[d.id] = verdict(d, None, present[0][1]["missing"], targets)
            continue
        pooled: dict[str, list[float]] = {}
        for k in d.sample_keys():
            pooled[k] = [v for smp in measurable for v in smp.values.get(k, [])]
        out[d.id] = verdict(d, pooled, None, targets)
    return out
