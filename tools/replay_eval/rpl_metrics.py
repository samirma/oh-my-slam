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
from rpl_geometry import Arr, Sim3, angle_deg, mat4, optical_yaw_deg, umeyama, wrap_deg
from rpl_labels import LabelRule
from rpl_replay import Record


@dataclass(frozen=True)
class Def:
    id: str
    contract: str
    description: str
    op: str | None  # '<=' or '>=', None for information only
    target: float | None
    reduce: str  # share | median | p90 | max
    unit: str
    needs: tuple[str, ...] = ()
    floorplan: bool = False


DEFS: list[Def] = [
    Def("c8.located_near_keyframe", "C8", "Frames within 0.3 m and 15° of a keyframe that are "
        "located", ">=", 0.9, "share", "share", ("O5",)),
    Def("c8.pos_error_over_tolerance", "C8", "Median position error of located frames over "
        "max(2 cm, 1 % of the scene depth)", "<=", 1.0, "median", "ratio", ("O5",)),
    Def("c8.pos_error_m", "C8", "Median position error of located frames", None, None,
        "median", "m", ("O5",)),
    Def("c8.ambiguity_on_wrong", "C8", "ambiguity non-null on located frames whose pose error "
        "exceeds 0.3 m or 15°", ">=", 0.8, "share", "share", ("O5",)),
    Def("c8.ambiguity_null_on_right", "C8", "ambiguity null on located frames within 0.10 m "
        "and 5° of the truth", ">=", 0.9, "share", "share", ("O5",)),
    Def("c8.ambiguity_rule", "C8", "Non-null ambiguity is a rival farther than 0.10 m or 5° "
        "(api.md §6)", ">=", 1.0, "share", "share", ("O5",)),
    Def("c8.nearest_view_turn", "C8", "no_overlap frames sent with a prior whose nearest_view "
        "turn points toward "
        "the keyframe's view (within 45°)", ">=", 0.8, "share", "share", ("O5",)),
    Def("c8.sweep_yaw_error_deg", "C8", "Keyframe yaw error in look sweeps after the "
        "gravity-aligned fit (median; is a locate yaw bias from registration?)", None, None,
        "median", "deg"),
    Def("c8.sweep_yaw_span_ratio", "C8", "Map over truth yaw span of each look sweep's "
        "keyframes (median; > 1 over-counts the turn)", None, None, "median", "ratio"),
    Def("c2.scale_error_priors", "C2", "Scale error with metric priors", "<=", 0.03, "max",
        "ratio"),
    Def("c2.scale_error_height", "C2", "Scale error with camera height only", "<=", 0.10,
        "max", "ratio"),
    Def("c2.scale_error_monocular", "C2", "Scale error without priors or heights", None, None,
        "max", "ratio"),
    Def("c1.shift_pos_m", "C1", "Map-frame shift of stored keyframes between consecutive "
        "versions beyond gauge_shift (median)", "<=", 0.02, "median", "m"),
    Def("c1.shift_rot_deg", "C1", "Rotation of stored keyframes between consecutive versions "
        "beyond gauge_shift (median)", "<=", 0.5, "median", "deg"),
    Def("c1.prior_residual_m", "C1", "Registered inputs' position residual to their priors "
        "(median; biased base odometry priors)", None, None, "median", "m"),
    Def("c1.prior_residual_deg", "C1", "Registered inputs' rotation residual to their priors "
        "(median)", None, None, "median", "deg"),
    Def("c3.readable_s", "C3", "Upload to frame readable in a version (p90 over jobs)", "<=",
        2.0, "p90", "s"),
    Def("c3.fusion_s_per_frame", "C3", "Fusion time per new frame (p90 over jobs)", "<=", 3.0,
        "p90", "s"),
    Def("c3.reads_one_version", "C3", "Reads during an update answered from one version", ">=",
        1.0, "share", "share"),
    Def("c3.eta_within_30pct", "C3", "eta_s of a queued job, and predicted_s of a dry run, "
        "within ±30 % of the measured time", ">=", 0.8, "share", "share", ("O3|O3-jobs",)),
    Def("c3.fusing_version_gap_s", "C3", "Longest gap between versions while fusion lasts "
        "longer than 10 s", "<=", 10.0, "max", "s", ("O3",)),
    Def("c4.reason_agreement", "C4", "Rejected inputs whose reason agrees with the truth (bare "
        "wall, too close, no overlap)", ">=", 0.8, "share", "share"),
    Def("c4.inputs_contract", "C4", "Jobs whose inputs[] has one entry per frame, in request "
        "order", ">=", 1.0, "share", "share"),
    Def("c5.revert_s", "C5", "Revert latency", "<=", 2.0, "max", "s", ("O2",)),
    Def("c5.revert_contract", "C5", "Revert commits a new version", ">=", 1.0, "share", "share",
        ("O2",)),
    Def("c6.warping_rejected", "C6", "Warping episodes with an update rejected by accept_if "
        "at the default thresholds", ">=", 1.0, "share", "share", ("O9",)),
    Def("c6.other_updates_accepted", "C6", "Updates of the passing episodes accepted at the "
        "skill's accept_if thresholds", ">=", 0.9, "share", "share", ("O9",)),
    Def("c9.grid_agreement", "C9", "Observed grid cells agreeing with the floor plan (free in "
        "rooms, occupied at walls)", ">=", 0.9, "share", "share", ("O6",), True),
    Def("c9.doors_found", "C9", "Observed doors of the floor plan found as passages within "
        "0.3 m", ">=", 0.8, "share", "share", ("O6",), True),
    Def("c9.passage_geometry", "C9", "Found doors with centre_m within 0.10 m and width_m "
        "within 0.10 m of the door gap", ">=", 0.8, "share", "share", ("O6",), True),
    Def("c9.floor_observed", "C9", "Free cells with floor_observed where the floor plan has "
        "floor", ">=", 0.9, "share", "share", ("O6",), True),
    Def("c9.camera_cells_free", "C9", "Keyframe camera cells and their near blind ring free",
        ">=", 1.0, "share", "share", ("O6",)),
    Def("c9.camera_stand_free", "C9", "Keyframe camera cells alone free (diagnoses "
        "c9.camera_cells_free: stand vs blind ring)", None, None, "share", "share", ("O6",)),
    Def("c9.regions_set", "C9", "Frontiers bordering a region, and objects in or on one, with "
        "region set", ">=", 1.0, "share", "share", ("O6",)),
    Def("c9.band_occupied", "C9", "Observed wall cells (obstacles across the band) marked "
        "occupied", ">=", 0.95, "share", "share", ("O6",), True),
    Def("c9.band_as_asked", "C9", "Grid z_band_m equal to the requested band", ">=", 1.0,
        "share", "share", ("O6",)),
    Def("c9.floor_height_error_m", "C9", "Floor height error against the lowest large support "
        "plane in view (the floor; the worktop for an arm)", "<=", 0.02, "max", "m",
        ("O6",)),
    Def("c9.grid_read_s", "C9", "GET grid (5 cm cells) on a committed version", "<=", 0.5,
        "max", "s", ("O6",)),
    Def("c10.objects_read_s", "C10", "GET objects on a committed version", "<=", 0.5, "max",
        "s", ("O8",)),
    Def("c10.centre_error_m", "C10", "Object position error: box centre, or centre xy and "
        "bottom z for base-origin truth (median, same-label match within "
        "1 m)", "<=", 0.10, "median", "m", ("O8",)),
    Def("c10.label_agreement", "C10", "Objects whose label agrees with the nearest truth "
        "object", ">=", 0.8, "share", "share", ("O8",)),
    Def("c10.id_stability", "C10", "Truth objects keeping their id between consecutive "
        "versions", ">=", 0.8, "share", "share", ("O8",)),
    Def("c10.support_height_error_m", "C10", "Support height of objects on a surface "
        "(median)", "<=", 0.02, "median", "m", ("O8",)),
    Def("c10.relations_agreement", "C10", "supported_by / inside_of agreeing with the truth's "
        "relations", ">=", 0.9, "share", "share", ("O8",)),
    Def("c11.range_rel_error", "C11", "Range-scan error against the floor-plan walls (median, "
        "relative)", "<=", 0.05, "median", "ratio", ("O7",), True),
    Def("c12.locate_first_s", "C12", "The replay's first idle locate, per frame (warm-up)",
        None, None, "max", "s", ("O5",)),
    Def("c12.locate_idle_s", "C12", "Locate while idle, per frame of the request (p90)", "<=",
        2.0, "p90",
        "s", ("O5",)),
    Def("c12.locate_busy_s", "C12", "Locate of one frame while an update runs (p90)", "<=",
        3.0, "p90", "s", ("O5",)),
    Def("c12.locate_busy_ratio", "C12", "Locate latency while an update runs over idle "
        "(medians)", "<=", 2.0, "max", "ratio", ("O5",)),
    Def("c12.analyse_busy_s", "C12", "Analyse while an update runs (p90)", "<=", 3.0, "p90",
        "s", ("O7",)),
    Def("c12.analyse_busy_ratio", "C12", "Analyse latency while an update runs over idle "
        "(medians)", "<=", 2.0, "max", "ratio", ("O7",)),
    Def("contract.schema_valid", "contracts", "Responses valid against api.schema.json", ">=",
        1.0, "share", "share"),
    Def("contract.server_timing", "contracts", "Responses carrying Server-Timing", ">=", 1.0,
        "share", "share"),
]
BY_ID = {d.id: d for d in DEFS}


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
    probe_interval_s: float = 0.5
    warping: tuple[str, ...] = ("e034", "e040")
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


def reduce(values: list[float], how: str) -> float | None:
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    if how in ("share", "median"):
        return float(a.mean() if how == "share" else np.median(a))
    return float(np.percentile(a, 90) if how == "p90" else a.max())


def verdict(d: Def, values: list[float] | None, missing: str | None,
            targets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    t = targets.get(d.id, {"op": d.op, "value": d.target})
    out: dict[str, Any] = {"contract": d.contract, "description": d.description,
                           "unit": d.unit, "target": t if t["op"] else None}
    if missing is not None:
        return {**out, "value": None, "n": 0, "status": "not_measurable", "missing": missing}
    value = reduce(values or [], d.reduce)
    out.update(value=value, n=len(values or []))
    if value is None:
        out["status"] = "no_samples"
    elif not t["op"]:
        out["status"] = "info"
    else:
        ok = value <= t["value"] if t["op"] == "<=" else value >= t["value"]
        out["status"] = "pass" if ok else "fail"
    if values and d.reduce != "share":
        a = np.asarray(values)
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
        self.scale()
        self.stability()
        if "O5" in self.caps:
            self.locate()
        if "O6" in self.caps and self.rec.grid is not None:
            self.grid()
        if "O8" in self.caps and self.rec.objects:
            self.objects()
        if "O7" in self.caps:
            self.analyse()
        if "O9" in self.caps and self.ep.id in self.p.warping:
            s.add("c6.warping_rejected", any(j.job.get("state") == "rejected"
                                             for j in self.rec.jobs))
        elif "O9" in self.caps and self.passed():
            for j in self.rec.jobs:
                if j.job.get("state"):
                    s.add("c6.other_updates_accepted", j.job["state"] != "rejected")
        if "O2" in self.caps:
            if self.rec.revert_s is not None:
                s.add("c5.revert_s", self.rec.revert_s)
            if self.rec.revert_ok is not None:
                s.add("c5.revert_contract", self.rec.revert_ok)
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
                s.add("c3.readable_s", j.t_readable - j.t_start)
            if j.job.get("state") != "committed" or j.t_end is None:
                continue
            n = max(1, sum(1 for i in j.job.get("inputs") or [] if i["status"] != "rejected"))
            start = j.t_fusing if j.t_fusing is not None else j.t_start
            s.add("c3.fusion_s_per_frame", (j.t_end - start) / n)
            if has(self.caps, "O3|O3-jobs"):
                measured = j.t_end - j.t_posted
                if j.eta_queued is not None and measured > 0:
                    s.add("c3.eta_within_30pct", abs(j.eta_queued - measured) <= 0.3 * measured)
                if j.dry is not None:
                    pred = j.dry["predicted_s"]
                    total = float(pred["registration"]) + float(pred["fusion"])
                    m = j.t_end - j.t_start
                    s.add("c3.eta_within_30pct", abs(total - m) <= 0.3 * m)
            if "O3" in self.caps and j.t_fusing is not None and j.t_end - j.t_fusing > 10.0:
                seen_at = sorted(self.rec.probe_versions + j.versions_seen)
                v0 = max([v for t, v in seen_at if t <= j.t_fusing], default=-1)
                times, seen = [], v0
                for t, v in seen_at:
                    if j.t_fusing < t <= j.t_end and v > seen:
                        times.append(t)
                        seen = v
                gaps = np.diff([j.t_fusing, *times, j.t_end])
                s.add("c3.fusing_version_gap_s",
                      max(0.0, float(gaps.max()) - self.p.probe_interval_s))
        for ok in self.rec.consistency:
            s.add("c3.reads_one_version", ok)

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
                self.s.add("c8.sweep_yaw_error_deg", e)
            if ratio is not None:
                self.s.add("c8.sweep_yaw_span_ratio", ratio)
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
            s.add("c4.inputs_contract", [i["name"] for i in inputs] == j.names)
            for i in inputs:
                for key, mid in (("prior_residual_m", "c1.prior_residual_m"),
                                 ("prior_residual_deg", "c1.prior_residual_deg")):
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
                    s.add("c4.reason_agreement", reason == truth or
                          (reason == "few_features" and truth == "plain"))

    def scale(self) -> None:
        if self.al is None:
            return
        err = abs(self.al[0].s - 1.0)
        if self.rec.mode == "metric":
            self.s.add("c2.scale_error_priors", err)
        elif self.p.heights and any(self.by_name[n].height_m is not None
                                    for n in self.rec.sent):
            self.s.add("c2.scale_error_height", err)
        else:
            self.s.add("c2.scale_error_monocular", err)

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
                self.s.add("c1.shift_pos_m",
                           max(0.0, d - float(g["translation_m"])) * scale)
                self.s.add("c1.shift_rot_deg", max(0.0, angle_deg(Ta[:3, :3], Tb[:3, :3])
                                                   - float(g["rotation_deg"])))

    # ---- O5 ---------------------------------------------------------------------------------
    def locate(self) -> None:
        s, rec = self.s, self.rec
        for x in rec.idle_locate_s:
            s.add("c12.locate_idle_s", x)
        if rec.first_locate_s is not None:
            s.add("c12.locate_first_s", rec.first_locate_s)
        for x in rec.busy_locate_s:
            s.add("c12.locate_busy_s", x)
        if rec.idle_locate_s and rec.busy_locate_s:
            s.add("c12.locate_busy_ratio",
                  float(np.median(rec.busy_locate_s)) / float(np.median(rec.idle_locate_s)))
        kf_frames = [self.by_name[r["name"]] for r in self.keyframes]
        tol = max(0.02, 0.01 * scene_depth(self.ep, self.plan, kf_frames)) if kf_frames else 0.02
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
                    s.add("c8.located_near_keyframe", located)
                amb = r.get("ambiguity")
                if amb is not None:
                    s.add("c8.ambiguity_rule", amb["pos_m"] > 0.10 or amb["rot_deg"] > 5.0)
                if located and self.al is not None:
                    W = self.al[0].pose(mat4(r["T_map_cam"]))
                    e_pos = float(np.linalg.norm(W[:3, 3] - T[:3, 3]))
                    e_rot = angle_deg(W[:3, :3], T[:3, :3])
                    s.add("c8.pos_error_m", e_pos)
                    s.add("c8.pos_error_over_tolerance", e_pos / tol)
                    if e_pos > 0.3 or e_rot > 15.0:
                        s.add("c8.ambiguity_on_wrong", amb is not None)
                    if e_pos <= 0.10 and e_rot <= 5.0:
                        s.add("c8.ambiguity_null_on_right", amb is None)
                nv = r.get("nearest_view")
                # judged only for frames sent with a prior: turn_deg is null without one
                if r["status"] == "rejected" and r.get("reason") == "no_overlap" and nv \
                        and nv["name"] in self.by_name and lo.with_prior:
                    kt = self.by_name[nv["name"]].truth
                    assert kt is not None
                    want = wrap_deg(optical_yaw_deg(kt[:3, :3]) - optical_yaw_deg(T[:3, :3]))
                    turn = nv.get("turn_deg")
                    s.add("c8.nearest_view_turn", turn is not None and
                          abs(wrap_deg(float(turn) - want)) <= self.p.turn_tol_deg)

    # ---- O6 ---------------------------------------------------------------------------------
    def grid(self) -> None:
        s, g = self.s, self.rec.grid
        assert g is not None
        if self.rec.grid_s is not None:
            s.add("c9.grid_read_s", self.rec.grid_s)
        s.add("c9.band_as_asked", bool(np.allclose(g["z_band_m"], self.rec.grid_band,
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
                s.add("c9.camera_cells_free", False)
                s.add("c9.camera_stand_free", False)
                continue
            s.add("c9.camera_stand_free", bool(state[rr, c] == 1))
            disc = np.hypot(gx - p[0], gy - p[1]) <= max(0.0, radius - cell)
            if cls is not None:  # floor-plan walls stay occupied
                disc &= ~np.isin(cls.reshape(rows, cols), (WALL, BORDER, OUTSIDE))
            disc[rr, c] = True
            s.add("c9.camera_cells_free", bool((state[disc] == 1).all()))
        polys = [np.asarray(reg["polygon_m"], dtype=float) for reg in g["regions"]]
        for fr in g["frontiers"]:
            c2 = np.asarray([fr["centre_m"]], dtype=float)
            if any(float(poly_distance(c2, poly)[0]) <= 2 * cell for poly in polys):
                s.add("c9.regions_set", fr["region"] is not None)
        if "O8" in self.caps and self.rec.objects:
            for o in self.rec.objects[max(self.rec.objects)]["objects"]:
                c2 = np.asarray([o["centre_m"][:2]], dtype=float)
                if o.get("state") != "struck" and any(
                        float(poly_distance(c2, poly)[0]) <= cell for poly in polys):
                    s.add("c9.regions_set", o["region"] is not None)
        if sim is None:
            return
        ref = self.floor_reference()
        if ref is not None and frames["frames"]:
            mean = np.mean([frame_pose(r)[:2, 3] for r in frames["frames"]], axis=0)
            # floor_z_m is a map-frame z: through the same similarity as the cameras
            s.add("c9.floor_height_error_m",
                  abs(float(sim.point([mean[0], mean[1], fz])[2]) - ref))
            s.info["floor_z_world_m"] = float(sim.point([mean[0], mean[1], fz])[2])
            s.info["floor_reference_m"] = ref
        if W is None or cls is None or self.plan is None:
            return
        st, ob, fo = state.ravel(), obs.ravel(), floor_obs.ravel()
        floorish = (cls == FLOOR) | (cls == DOOR)
        seen = st != 0
        for ok in np.concatenate([st[seen & floorish] == 1, st[seen & (cls == WALL)] == 2]):
            s.add("c9.grid_agreement", bool(ok))
        for ok in fo[(st == 1) & floorish & (fo != 2)] == 1:  # 2: free by rule, not seen
            s.add("c9.floor_observed", bool(ok))
        for ok in st[(cls == WALL) & (ob > 0)] == 2:
            s.add("c9.band_occupied", bool(ok))
        passages = [(sim.point([*pg["centre_m"], fz])[:2], sim.s * float(pg["width_m"]))
                    for pg in g["passages"]]
        for door in self.plan.doors:
            near = np.linalg.norm(W - door.centre, axis=1) <= self.p.door_m
            if not (near & seen).any():
                continue
            found = [(float(np.linalg.norm(c - door.centre)), w) for c, w in passages
                     if np.linalg.norm(c - door.centre) <= self.p.door_m]
            s.add("c9.doors_found", bool(found))
            if found:
                dist, width = min(found)
                s.add("c9.passage_geometry", dist <= 0.10 and abs(width - door.width) <= 0.10)

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
            s.add("c10.objects_read_s", self.rec.objects_s)
        if self.al is None or not self.truth_objects:
            return
        sim = self.al[0]
        final = self.rec.objects[max(self.rec.objects)]
        live = [o for o in final["objects"] if o.get("state") != "struck"]
        nearest = self.match(live, same=False)
        for o in live:
            if int(o["id"]) in nearest:
                s.add("c10.label_agreement",
                      self.p.labels.same(o["label"], nearest[int(o["id"])].label))
        matched = self.match(live, same=True)
        for o in live:
            t = matched.get(int(o["id"]))
            if t is not None:
                s.add("c10.centre_error_m", self.error(o, t))
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
                s.add("c10.support_height_error_m", abs(float(z) - t.support_z))
            if t.parent is not None:
                want = by_truth.get(t.parent)
                if t.inside:
                    ok = bool(ins) and int(ins["id"]) == want
                else:
                    ok = bool(sup) and (sup["kind"] == "surface" or int(sup["id"]) == want)
                s.add("c10.relations_agreement", ok)
            elif t.support_z is not None:
                s.add("c10.relations_agreement",
                      bool(sup) and sup["kind"] == "surface" and not ins)
        s.info["objects"] = self.object_report(live, nearest, matched, surfaces)
        versions = sorted(self.rec.objects)
        for a, b in itertools.pairwise(versions):
            ma = {t.id: oid for oid, t in self.match(self.rec.objects[a]["objects"], True).items()}
            mb = {t.id: oid for oid, t in self.match(self.rec.objects[b]["objects"], True).items()}
            for tid in sorted(set(ma) & set(mb)):
                s.add("c10.id_stability", ma[tid] == mb[tid])

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
            s.add("c12.analyse_busy_s", x)
        if rec.idle_analyse_s and rec.busy_analyse_s:
            s.add("c12.analyse_busy_ratio",
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
                    s.add("c11.range_rel_error",
                          abs(sim.s * float(beam["range_m"]) - truth) / truth)


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
    if d.id in ("c2.scale_error_priors", "c1.prior_residual_m", "c1.prior_residual_deg"):
        return mode == "metric"
    if d.id in ("c2.scale_error_height", "c2.scale_error_monocular"):
        return mode == "none"
    if d.id == "c6.warping_rejected":
        return episode in params.warping
    if d.id == "c6.other_updates_accepted":
        return episode not in params.warping
    return True


def run_metrics(samples: Samples, caps: set[str], plan: FloorPlan | None, mode: str,
                episode: str, params: Params,
                targets: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for d in DEFS:
        if not applies(d, mode, episode, params):
            continue
        miss = missing_for(d, caps, plan)
        out[d.id] = verdict(d, samples.values.get(d.id, []), miss, targets)
    return out


def overall(runs: list[tuple[Samples, dict[str, dict[str, Any]]]],
            targets: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
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
        pooled = [v for smp in measurable for v in smp.values.get(d.id, [])]
        out[d.id] = verdict(d, pooled, None, targets)
    return out
