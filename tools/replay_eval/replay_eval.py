#!/usr/bin/env python3
"""Replay evaluator of oh-my-slam (specs/evaluation.md): replays recorded robot-guide episodes
against a running service's public HTTP API and measures the targets against the simulator truth.

Standalone: Python 3.11+ standard library, numpy and requests (jsonschema optional). It imports
nothing from the service. See README.md beside this file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from rpl_api import Api, ApiError, Log
from rpl_episode import Episode, dump_layout, load_episode
from rpl_floorplan import FloorPlan, Raster, load_house, rasterise
from rpl_labels import LabelRule
from rpl_layout import LayoutError, load_layout
from rpl_metrics import Measure, Params, Samples, overall, run_metrics
from rpl_replay import GATE, Options, Record, Replayer
from rpl_targets import DEFAULT_TARGETS, Targets
from rpl_validate import DEFAULT_SCHEMA, Validator

MODES = ("metric", "none")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="replay_eval.py", description=__doc__.splitlines()[0])
    p.add_argument("--server", help="service URL, e.g. http://192.168.0.106:9091")
    p.add_argument("--episodes", type=Path, required=True, help="folder of e<NNN>/ episodes")
    p.add_argument("--house", type=Path, help="ProcTHOR house JSON, or a folder of them named "
                   "after the episodes' scenes")
    p.add_argument("--map-prefix", default="replay-", help="map name prefix")
    p.add_argument("--out", type=Path, help="result JSON file")
    p.add_argument("--episode", action="append", nargs="+", default=[],
                   help="only these episodes (e.g. e041)")
    p.add_argument("--priors", choices=["metric", "none", "both"], default="both",
                   help="both: each robot's default replays (base: metric and none; humanoid: "
                   "none; arm: metric)")
    p.add_argument("--keep", action="store_true", help="keep the maps (default: delete)")
    p.add_argument("--dump-layout", action="store_true",
                   help="print what each episode folder holds and exit")
    p.add_argument("--layout", type=Path, help="JSON override of the layout table")
    p.add_argument("--targets", type=Path, default=DEFAULT_TARGETS,
                   help="targets file (default examples/targets.json)")
    p.add_argument("--record-requests", type=Path,
                   help="write every request as a request set for tools/evaluate --requests")
    p.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    p.add_argument("--no-jsonschema", action="store_true",
                   help="minimal required-keys validation even if jsonschema is installed")
    p.add_argument("--accept-if", default=json.dumps(GATE),
                   help="accept_if JSON sent with every update when O9 is served (default: the "
                   "gate parameters of evaluation.md; {} checks nothing)")
    p.add_argument("--passing", help="comma-separated passing episodes (C6); default: each "
                   "episode's verdict.md")
    p.add_argument("--wait-s", type=float, default=10.0)
    p.add_argument("--probe-interval-s", type=float, default=0.5)
    p.add_argument("--no-probes", action="store_true", help="no reads while a job runs")
    p.add_argument("--base-priors", choices=["drifting", "metric"], default="drifting",
                   help="trust of the wheeled base's odometry priors in the metric replay "
                   "(sigma: 10 %% of the distance travelled)")
    p.add_argument("--camera", help="intrinsics JSON {fx, fy, cx, cy, width, height} for v2 runs "
                   "with neither calibrate_camera.json nor camera_info_<id>.json")
    p.add_argument("--no-depth", action="store_true", help="no v2 depth images in updates")
    p.add_argument("--no-height", action="store_true",
                   help="no height_above_floor_m in the trust-none replay")
    p.add_argument("--range-scans", type=int, default=5)
    p.add_argument("--job-timeout-s", type=float, default=1800.0)
    p.add_argument("--timeout-s", type=float, default=300.0, help="per HTTP request")
    p.add_argument("--retries", type=int, default=3, help="retries of a retryable error")
    p.add_argument("--retry-wait-s", type=float, default=2.0)
    p.add_argument("--align-min-spread-m", type=float, default=0.5)
    p.add_argument("--align-full-rotation", action="store_true",
                   help="fit a 3-D rotation from map to world instead of a yaw")
    return p.parse_args(argv)


def select_episodes(root: Path, wanted: list[str]) -> list[Path]:
    eps = sorted(p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"e\d+", p.name))
    if wanted:
        missing = sorted(set(wanted) - {p.name for p in eps})
        if missing:
            raise SystemExit(f"replay_eval: no episode folder {', '.join(missing)} in {root}")
        eps = [p for p in eps if p.name in wanted]
    if not eps:
        raise SystemExit(f"replay_eval: no e<NNN>/ folder in {root}")
    return eps


def house_for(ep: Episode, house: Path | None, layout: dict[str, Any],
              warnings: list[str]) -> FloorPlan | None:
    if house is None:
        warnings.append("no --house: floor-plan metrics are not measurable")
        return None
    if any(ep.scene.startswith(px) for px in layout["no_floorplan_scene_prefixes"]):
        return None
    path = house
    if house.is_dir():
        names = [ep.scene, ep.scene.replace(":", "_"), re.split(r"[:/]", ep.scene)[-1]]
        found = [house / f"{n}.json" for n in names if n and (house / f"{n}.json").is_file()]
        if not found:
            warnings.append(f"no house JSON for scene '{ep.scene}' in {house}")
            return None
        path = found[0]
    return load_house(json.loads(path.read_text()), layout["container_types"], ep.floor_z,
                      str(layout["house_object_origin"]))


def modes_for(ep: Episode, priors: str, layout: dict[str, Any]) -> list[str]:
    if priors != "both":
        return [priors]
    return [m for m in layout["default_modes"].get(ep.kind, list(MODES)) if m in MODES]


def call_stats(log: Log) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for route in sorted({c.route for c in log.calls}):
        cs = [c for c in log.calls if c.route == route]
        s = np.array([c.seconds for c in cs])
        out[route] = {"n": len(cs), "median_s": float(np.median(s)),
                      "p90_s": float(np.percentile(s, 90)),
                      "errors": sum(1 for c in cs if c.status >= 400)}
    return out


def job_brief(rec: Record) -> list[dict[str, Any]]:
    out = []
    for j in rec.jobs:
        inputs = j.job.get("inputs") or []
        out.append({
            "set": j.set_key, "frames": len(j.names), "job_id": j.job_id,
            "state": j.job.get("state"), "provisional_version": j.job.get("provisional_version"),
            "result_version": j.job.get("result_version"),
            "readable_s": None if j.t_readable is None else j.t_readable - j.t_start,
            "total_s": None if j.t_end is None else j.t_end - j.t_start,
            "rejected": {i["name"]: i["reason"] for i in inputs if i["status"] == "rejected"},
            "quality": j.job.get("quality"), "error": j.error,
            "recorded_job_id": j.job_id_recorded, "recorded_s": j.recorded_s})
    return out


def fmt(v: Any) -> str:
    if v is None:
        return "-"
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def _bound_text(t: dict[str, Any]) -> str:
    v = t["value"]
    if t["op"] == "between":
        return f"in [{fmt(float(v[0]))}, {fmt(float(v[1]))}]"
    return f"{t['op']} {fmt(float(v))}"


def target_text(t: dict[str, Any] | None) -> str:
    if not t or t.get("info"):
        return "-"
    if "parts" in t:
        return " & ".join(f"{k} {_bound_text(p)}" for k, p in t["parts"].items())
    return _bound_text(t)


def value_text(v: Any) -> str:
    if isinstance(v, dict):
        return " ".join(f"{k}={fmt(x)}" for k, x in v.items())
    return fmt(v)


def summary_table(result: dict[str, Any]) -> str:
    lines = [f"{'metric':32} {'value':>10} {'target':>12} {'n':>7}  status"]
    for mid, m in result["overall"].items():
        t = m["target"]
        target = target_text(t)
        status = m["status"] + (f" ({m['missing']})" if m.get("missing") else "")
        lines.append(f"{mid:32} {value_text(m['value']):>10} {target:>12} {m['n']:>7}  {status}")
    lines.append("")
    for eid, e in result["episodes"].items():
        if "error" in e:
            lines.append(f"{eid}: error: {e['error']}")
            continue
        for mode, r in e["runs"].items():
            if "error" in r:
                lines.append(f"{eid} {mode}: error: {r['error']}")
                continue
            counts: dict[str, int] = {}
            for m in r["metrics"].values():
                counts[m["status"]] = counts.get(m["status"], 0) + 1
            lines.append(f"{eid} {mode}: " + ", ".join(f"{k} {v}" for k, v in
                                                        sorted(counts.items())))
    return "\n".join(lines)


def replay_run(args: argparse.Namespace, validator: Validator, ep: Episode, mode: str,
               caps: list[str], plan: FloorPlan | None, raster: Raster | None,
               layout: dict[str, Any], params: Params, options: Options,
               targets: Targets) -> tuple[dict[str, Any], Samples | None]:
    log = Log(record_dir=None if args.record_requests is None
              else args.record_requests / f"{ep.id}-{mode}")
    api = Api(args.server, validator, log, timeout_s=args.timeout_s, retries=args.retries,
              retry_wait_s=args.retry_wait_s)
    name = f"{args.map_prefix}{ep.id}-{mode}"
    try:
        rec = Replayer(api, ep, mode, caps, name, options).run()
    except ApiError as e:
        return {"map": name, "error": str(e), "calls": call_stats(log),
                "trace": traceback.format_exc(limit=5)}, None
    truth_objects = plan.objects if plan is not None and plan.objects \
        else ep.truth_objects(layout)
    samples = Measure(rec, ep, plan, raster, truth_objects, log, params).run()
    metrics = run_metrics(samples, set(caps), plan, mode, ep.id, params, targets)
    return {"map": name, "metrics": metrics, "info": samples.info, "timings_s": rec.timings_s,
            "calls": call_stats(log), "jobs": job_brief(rec),
            "locates": len(rec.locates), "warnings": rec.warnings[:100]}, samples


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    layout = load_layout(args.layout)
    if args.camera:
        layout["camera_override"] = json.loads(args.camera)
    paths = select_episodes(args.episodes, [e for group in args.episode for e in group])
    if args.dump_layout:
        print(json.dumps([dump_layout(p, layout) for p in paths], indent=1, default=str))
        return 0
    if not args.server or not args.out:
        raise SystemExit("replay_eval: --server and --out are required (except --dump-layout)")
    t0 = time.monotonic()
    started = dt.datetime.now(dt.UTC).isoformat()
    validator = Validator(args.schema, use_jsonschema=not args.no_jsonschema)
    health_api = Api(args.server, validator, timeout_s=args.timeout_s)
    health, _ = health_api.call("GET", "/api/health", "GET health", "Health")
    caps = list(health["capabilities"])
    targets = Targets.load(args.targets)
    params = Params(align_min_spread_m=args.align_min_spread_m,
                    align_full_rotation=args.align_full_rotation,
                    worktop_placements=tuple(layout["worktop_placements"]),
                    labels=LabelRule.from_layout(layout["labels"]),
                    sweep_prefixes=tuple(layout["sweep"]["prefixes"]),
                    sweep_stand_m=float(layout["sweep"]["stand_m"]),
                    probe_interval_s=args.probe_interval_s,
                    door_centre_tol_m=targets.param("passages.geometry", "centre_tolerance_m",
                                                    0.10),
                    door_width_tol_m=targets.param("passages.geometry", "width_tolerance_m", 0.10),
                    locate_abs_m=targets.param("locate.median_pos_error", "abs_m", 0.02),
                    locate_depth_share=targets.param("locate.median_pos_error",
                                                     "scene_depth_share", 0.01),
                    eta_tolerance=targets.param("jobs.eta_accuracy", "time_tolerance_rel", 0.3),
                    heights=not args.no_height,
                    passing=None if args.passing is None
                    else tuple(e for e in args.passing.split(",") if e))
    options = Options(wait_s=args.wait_s, probe_interval_s=args.probe_interval_s,
                      probes=not args.no_probes, heights=not args.no_height, keep=args.keep,
                      accept_if=json.loads(args.accept_if), range_scans=args.range_scans,
                      job_timeout_s=args.job_timeout_s, depth=not args.no_depth,
                      sigma_pos_m=layout["prior_sigma_pos_m"],
                      sigma_rot_deg=layout["prior_sigma_rot_deg"],
                      height_sigma_m=layout["height_sigma_m"],
                      prior_trust={**layout["prior_trust"], "base": args.base_priors},
                      travel_share=dict(layout["prior_sigma_travel_share"]))
    episodes: dict[str, Any] = {}
    pooled: list[tuple[Samples, dict[str, dict[str, Any]]]] = []
    for path in paths:
        try:
            ep = load_episode(path, layout)
            warnings: list[str] = []
            plan = house_for(ep, args.house, layout, warnings)
        except (LayoutError, ValueError, KeyError, OSError) as e:
            episodes[path.name] = {"error": f"{type(e).__name__}: {e}"}
            continue
        raster = rasterise(plan) if plan is not None and plan.has_rooms else None
        entry: dict[str, Any] = {
            "robot": ep.robot, "kind": ep.kind, "scene": ep.scene, "passed": ep.passed,
            "floor_plan": plan is not None,
            "frames": len(ep.all_frames()), "events": len(ep.events),
            "warnings": (ep.warnings + warnings)[:100], "runs": {}}
        for mode in modes_for(ep, args.priors, layout):
            print(f"replay_eval: {ep.id} {mode} ...", file=sys.stderr, flush=True)
            run, samples = replay_run(args, validator, ep, mode, caps, plan, raster, layout,
                                      params, options, targets)
            entry["runs"][mode] = run
            if samples is not None:
                pooled.append((samples, run["metrics"]))
        episodes[ep.id] = entry
    health_end, _ = health_api.call("GET", "/api/health", "GET health", "Health")
    result = {
        "tool": "replay_eval", "format": 2, "targets": str(args.targets),
        "gate_parameters": json.loads(args.accept_if), "started_at": started,
        "finished_at": dt.datetime.now(dt.UTC).isoformat(), "server": args.server,
        "argv": sys.argv[1:] if argv is None else argv,
        "health": {"start": health, "end": health_end},
        "client": {"host": platform.node(), "python": platform.python_version(),
                   "jsonschema": validator.full,
                   "timings_s": {"total_s": time.monotonic() - t0}},
        "label_rule": params.labels.describe(),
        "episodes": episodes, "overall": overall(pooled, targets),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=str))
    print(summary_table(result))
    print(f"replay_eval: wrote {args.out}", file=sys.stderr)
    return 0 if pooled else 1


if __name__ == "__main__":
    sys.exit(main())
