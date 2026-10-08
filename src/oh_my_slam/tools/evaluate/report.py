"""``result.json`` (machine-readable) and ``summary.md`` (human-readable, rendered from the result
alone)."""

from __future__ import annotations

import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any, TypeGuard

import psutil

from oh_my_slam.core.atomic import atomic_write_text
from oh_my_slam.tools.evaluate.metrics import Metrics
from oh_my_slam.tools.evaluate.runner import RunRecord

SCHEMA = "oh-my-slam-evaluation/1"
# (metric id prefix, section title, note under the title); a metric goes to its longest prefix
SECTIONS: tuple[tuple[str, str, str], ...] = (
    ("perf", "Performance", ""),
    ("perf.stage", "Performance per stage",
     "`perf.<group>.stage.<stage>.*`: seconds (the median over the frames of a per-frame group, "
     "else the slowest run: per image and per mapping update) and the peak client / server memory "
     "while the stage ran."),
    ("pose", "Pose accuracy",
     "Reference: the commanded headings in the capture names. The actual headings deviate from "
     "them by several degrees, and these errors include that deviation. `examples/camera/` "
     "(`pose.camera_*`) records the pan order but not the step angle: its yaw must turn left from "
     "one pan position to the next and agree across the tilts of a pan position."),
    ("map", "Map quality", ""),
    ("map_update", "Map update",
     "`examples/office_sequence/` is mapped in one update, and split across updates as its "
     "annotation says (`split_<sizes>`). `absent_fraction` is the share of the annotated objects "
     "that changed (the cup) that the map no longer has; `hole_fraction` the share of the cells of "
     "their place where the map shows no surface; `before_present_fraction` the control (the split "
     "map after an update of the images that show the cup has it); `<split>.stability.*` compares "
     "the labels, ids and OBBs of the unchanged objects of the first update with the last "
     "(aligned by their common captures); `<split>.ids_persistent_fraction` "
     "checks every published id against every later update; `<split>.vs_one_update.*` compares "
     "the split map with the one-update map (an id may differ where an earlier update published "
     "it for that object, and the split map may keep only objects an earlier update published)."),
    ("seg", "Segmentation", ""),
    ("contract", "Contracts", ""),
    ("server_sh", "Web service (server.sh)",
     "Performance, parity with the commands (each operation of `/api/openapi.json` on every "
     "reference input, run within its request and from the shell with the same recorded "
     "inference; the response body against the command's stdout) and the web application's UI."),
    ("gt", "Ground truth: accuracy", ""),
)
NO_GROUND_TRUTH = ("No ground-truth annotations were found in `examples/ground_truth/`. Segmentation "
                   "accuracy is not measured, and pose accuracy is measured only against the "
                   "commanded headings.")
STAGE_NOTE = ("Stage times are exclusive: they add up to the run's time. Peaks include nested "
              "stages. The client figure is the command's process tree: the command's own "
              "sampled peak plus the evaluator's 0.2 s samples, which also cover child processes "
              "such as COLMAP. The server figure is the inference server's physical footprint "
              "from the same samples. A stage shorter than 0.2 s gets the samples just before "
              "and just after it.")
CAMERA_MAPS = ("camera_single", "camera_split")  # suite.CAMERA's maps
CAMERA_POSES_NOTE = ("Yaw relative to the first registered capture. The names give the pan order "
                     "but not the step angle: the yaw must grow (turn left) from one pan position "
                     "to the next and agree across the tilts of a pan position.")
TAIL_IN_SUMMARY = 12
NOT_COMPARED = ("The baseline run has no value for these metrics (they are newer than it, or it "
                "did not compute them), so they could not regress against it.")


def environment(repo: Path) -> dict[str, Any]:
    """The code and machine a run measured."""
    def git(*args: str) -> str | None:
        res = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        return res.stdout.strip() if res.returncode == 0 else None

    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                         text=True).stdout.strip()
    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status),
            "cpu": cpu or platform.processor(),
            "memory_gb": round(psutil.virtual_memory().total / 2**30, 1),
            "os": f"macOS {platform.mac_ver()[0]}" if platform.mac_ver()[0] else platform.platform(),
            "python": platform.python_version()}


def compared(baseline: dict[str, Any]) -> bool:
    """Whether the metrics were compared with a stored baseline run."""
    return baseline.get("status") == "compared"


def not_compared(metrics: Metrics) -> list[str]:
    """The metrics the baseline run has no value for: they were not compared with it."""
    return [m.id for m in metrics.items.values() if m.baseline is None]


def summary_counts(metrics: Metrics, baseline: dict[str, Any]) -> dict[str, Any]:
    """Pass/fail counts; ``regressions`` and ``without_baseline`` (the metrics the baseline has
    no value for) are None when no baseline was compared."""
    items = list(metrics.items.values())
    done = compared(baseline)
    return {"metrics": len(items), "passed": sum(m.passed is True for m in items),
            "failed": sum(m.passed is False for m in items),
            "untargeted": sum(m.passed is None for m in items),
            "baseline": str(baseline.get("status", "missing")).split(":", 1)[0],
            "regressions": sum(m.regression for m in items) if done else None,
            "without_baseline": len(not_compared(metrics)) if done else None}


def build_result(metrics: Metrics, records: list[RunRecord], details: dict[str, Any], *,
                 started: str, finished: str, duration_s: float, env: dict[str, Any],
                 targets: Path, baseline: dict[str, Any]) -> dict[str, Any]:
    about = {**baseline, "not_compared": not_compared(metrics)} if compared(baseline) \
        else baseline
    return {
        "schema": SCHEMA, "started": started, "finished": finished,
        "duration_s": round(duration_s, 1), "environment": env, "targets": str(targets),
        "baseline": about, "summary": summary_counts(metrics, baseline),
        "metrics": [m.to_dict() for m in metrics.items.values()],
        "runs": [r.to_dict() for r in records],
        "details": details,
    }


def _jsonable(o: Any) -> Any:
    if hasattr(o, "item"):  # NumPy scalars
        return o.item()
    if isinstance(o, set | frozenset | tuple):
        return list(o)
    return str(o)


def write_report(out: Path, result: dict[str, Any]) -> tuple[Path, Path]:
    res, md = Path(out) / "result.json", Path(out) / "summary.md"
    atomic_write_text(res, json.dumps(result, indent=1, default=_jsonable) + "\n")
    atomic_write_text(md, summary_md(result))
    return res, md


# ------------------------------------------------------------------------------------------------


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int | float):
        return str(int(v)) if float(v).is_integer() and abs(v) < 1e6 else f"{v:.4g}"
    return str(v)


def _cell(v: Any, width: int = 160) -> str:
    return _fmt(v).replace("|", "/").replace("\n", " ")[:width]


def _unit(t: dict[str, Any]) -> str:
    u = t.get("unit") or ""
    return f" {u}" if u else ""


def _target(m: dict[str, Any]) -> str:
    t = m.get("target")
    return "—" if t is None else f"{t['op']} {_fmt(t['value'])}{_unit(t)}"


def _verdict(m: dict[str, Any]) -> str:
    base = {True: "pass", False: "**FAIL**", None: "no target"}[m["passed"]]
    return base + (" ⚠ regression" if m.get("regression") else "")


def _finite(v: Any) -> TypeGuard[float]:
    return isinstance(v, int | float) and math.isfinite(v)


def _regression_why(m: dict[str, Any]) -> str:
    """How much worse than the baseline, and the tolerance it exceeded."""
    t, v, b = m.get("target") or {}, m.get("value"), m.get("baseline")
    if not (m.get("regression") and _finite(v) and _finite(b)):
        return ""
    worse = v - b if t.get("op") == "<=" else b - v
    text = f"worse than the baseline {_fmt(b)}{_unit(t)} by {_fmt(round(worse, 4))}"
    if "tolerance_abs" in t:
        tol = max(float(t["tolerance_abs"]), float(t.get("tolerance_rel", 0.0)) * abs(b))
        text += f" (tolerance {_fmt(round(tol, 4))})"
    return text


def _why(m: dict[str, Any]) -> str:
    """Why a metric failed or regressed: its error, value against the target, baseline delta."""
    t, v = m.get("target"), m.get("value")
    if not _finite(v):
        return m.get("error") or "no value"
    parts = []
    if m.get("passed") is False and t is not None:
        limit = f"{_fmt(t['value'])}{_unit(t)}"
        parts.append(f"{_fmt(v)}{_unit(t)} exceeds the limit {limit}" if t["op"] == "<="
                     else f"{_fmt(v)}{_unit(t)} is below the minimum {limit}")
    if m.get("regression"):
        parts.append(_regression_why(m))
    return "; ".join(parts)


def _section(title: str, head: list[str], rows: list[list[Any]], note: str = "") -> list[str]:
    """A ``## title``, an optional note and a Markdown table."""
    return [f"## {title}", "", *([note, ""] if note else []),
            "| " + " | ".join(head) + " |", "|" + "---|" * len(head),
            *("| " + " | ".join(_cell(c) for c in r) + " |" for r in rows), ""]


def _section_of(mid: str) -> str:
    """The longest ``SECTIONS`` prefix of a metric id (per-stage metrics: ``perf.stage``)."""
    if mid.startswith("perf.") and ".stage." in mid:
        return "perf.stage"
    fits = [p for p, _, _ in SECTIONS if mid == p or mid.startswith(p + ".")]
    return max(fits, key=len) if fits else mid.split(".", 1)[0]


def _baseline_text(result: dict[str, Any]) -> tuple[str, str]:
    """(header text, result-line text) about the baseline comparison."""
    base, s = result.get("baseline") or {}, result["summary"]
    status = str(base.get("status", "missing"))
    head = f"baseline `{base.get('path')}`: {status}"
    if compared(base):
        head += f" (run of {base['started']})" if base.get("started") else ""
        missing = s.get("without_baseline") or 0
        return head, (f"{s.get('regressions') or 0} regressions against the baseline"
                      + (f"; {missing} metrics not compared: the baseline has no value for them"
                         if missing else ""))
    word = status.split(":", 1)[0]
    return (head + " — not compared",
            f"baseline {word} — not compared (store one with `--set-baseline`)")


def summary_md(result: dict[str, Any]) -> str:
    s, env = result["summary"], result["environment"]
    ok = s["failed"] == 0
    base_head, base_result = _baseline_text(result)
    lines = [
        f"# oh-my-slam evaluation — {result['started']}", "",
        f"commit `{(env.get('commit') or '?')[:10]}`{' (uncommitted changes)' if env.get('dirty') else ''}"
        f" · {env.get('cpu')} · {env.get('memory_gb')} GB · {env.get('os')} · Python "
        f"{env.get('python')} · {result['duration_s'] / 60:.1f} min", "",
        f"targets `{result['targets']}` · {base_head}", "",
        f"**Result: {'PASS' if ok else 'FAIL'}** — {s['metrics']} metrics: {s['passed']} passed, "
        f"{s['failed']} failed, {s['untargeted']} without a target; {base_result}.", ""]
    metrics = result["metrics"]
    failed = [m for m in metrics if m["passed"] is False]
    if failed:
        lines += _section("Failed metrics", ["metric", "value", "target", "why"],
                          [[m["id"], m["value"], _target(m), _why(m)] for m in failed])
    regress = [m for m in metrics if m.get("regression")]
    if regress:
        lines += _section(
            "Regressions against the baseline", ["metric", "value", "baseline", "target", "why"],
            [[m["id"], m["value"], m["baseline"], _target(m), _regression_why(m)]
             for m in regress])
    if compared(result.get("baseline") or {}):
        missing = [m for m in metrics if m.get("baseline") is None]
        if missing:
            lines += _section("Not compared with the baseline", ["metric", "value"],
                              [[m["id"], m["value"]] for m in missing], NOT_COMPARED)
    for prefix, title, note in SECTIONS:
        rows = [m for m in metrics if _section_of(m["id"]) == prefix]
        if prefix == "gt" and not rows:
            found = (result["details"].get("ground_truth") or {}).get("files")
            lines += [f"## {title}", "", NO_GROUND_TRUTH if not found else
                      f"{len(found)} ground-truth files were found, but none applied (see "
                      "`details.ground_truth.skipped` in result.json).", ""]
        elif rows:
            lines += _section(
                title, ["metric", "value", "target", "result", "baseline"],
                [[m["id"], m["value"], _target(m), _verdict(m), m["baseline"]] for m in rows],
                note)
    lines += [*_stages(metrics), *_runs(result["runs"]), *_details(result["details"]),
              *_tails(result["runs"])]
    return "\n".join(lines)


def _stages(metrics: list[dict[str, Any]]) -> list[str]:
    """Per-stage time and peak memory of every performance group (per run for a group of a few
    runs, e.g. the split map's updates; median time and peak memory over a larger group)."""
    rows: list[list[Any]] = []
    for m in metrics:
        parts = m["id"].split(".")
        if len(parts) != 3 or parts[0] != "perf" or parts[2] not in ("wall_s", "render_s"):
            continue
        d = m.get("detail") or {}
        per_run = {k: v.get("stages") or {s: {"s": x} for s, x in (v.get("stages_s") or {}).items()}
                   for k, v in (d.get("per_run") or {}).items() if isinstance(v, dict)}
        if len(per_run) > 1 and all(per_run.values()):
            for tag, stages in per_run.items():
                rows += [[tag, k, x.get("s"), x.get("client_peak_mb"), x.get("server_peak_gb")]
                         for k, x in stages.items()]
            continue
        stages = d.get("stages") or {k: {"s": v} for k, v in (d.get("stages_s") or {}).items()}
        runs = int(d.get("runs") or 1)
        name = parts[1] + (f" ({runs} runs: median s, max memory)" if runs > 1 else "")
        rows += [[name, k, x.get("s"), x.get("client_peak_mb"), x.get("server_peak_gb")]
                 for k, x in stages.items()]
    if not rows:
        return []
    return _section("Per-stage time and peak memory",
                    ["run", "stage", "s", "client peak MB", "server peak GB"], rows, STAGE_NOTE)


def _runs(runs: list[dict[str, Any]]) -> list[str]:
    rows = []
    for r in runs:
        stages = ((r.get("timings") or {}).get("stages_s") or {})
        top = ", ".join(f"{k} {v:.2f}" for k, v in sorted(stages.items(), key=lambda kv: -kv[1])[:3])
        extra = f"render {r['render_s']:.2f} s" if r.get("render_s") is not None else top
        rows.append([r["tag"], r["exit_code"], r["wall_s"], r["client_peak_mb"],
                     r["server_peak_gb"], extra])
    return _section(
        "Runs (in order)", ["run", "exit", "wall s", "client peak MB", "server peak GB", "slowest stages s"], rows)


def _details(details: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for name in ("single", "split"):
        rows = details.get(f"poses.{name}")
        if rows:
            out += _section(
                f"Poses — {name} map", ["capture", "commanded yaw °", "estimated yaw °", "error °", "pitch Δ °",
                 "direction ok"],
                [[r["capture"], r["commanded_yaw_deg"], r.get("yaw_deg"), r.get("yaw_err_deg"),
                  r.get("pitch_delta_deg"), r.get("pitch_ok")] for r in rows])
    for name in CAMERA_MAPS:
        rows = details.get(f"poses.{name}")
        if rows:
            out += _section(
                f"Poses — {name} map", ["capture", "pan position", "tilt", "yaw °", "pitch Δ °",
                                        "direction ok"],
                [[r["capture"], r["pan"], r["tilt"], r.get("yaw_deg"), r.get("pitch_delta_deg"),
                  r.get("pitch_ok")] for r in rows], CAMERA_POSES_NOTE)
    seg = [*([details["segmentation.restaurant"]] if "segmentation.restaurant" in details else []),
           *details.get("segmentation.frames", []), *details.get("segmentation.camera_frames", [])]
    if seg:
        out += _section(
            "Segmentation per image", ["image", "objects", "labels", "min score", "median score"],
            [[r["image"], r.get("objects"),
              ", ".join(f"{k} {v}" for k, v in (r.get("labels") or {}).items()) or r.get("error"),
              r.get("min_score"), r.get("median_score")] for r in seg])
    for key, title in (("map.stability", "Objects: one update vs split"),
                       ("map.camera_stability", "Objects: one update vs split — camera")):
        if details.get(key):
            out += _section(
                title, ["single id", "split id", "labels", "IoU", "centre Δ m", "extent Δ"],
                [[r["single_id"], r["split_id"], f"{r['single_label']} / {r['split_label']}",
                  r["iou"], r["centre_delta_m"], r["extent_delta_rel"]] for r in details[key]])
    for name in ("single", "split", *CAMERA_MAPS):
        rows = details.get(f"map.{name}.pairs")
        if rows:
            out += _section(
                f"Least consistent overlapping keyframe pairs — {name} map",
                ["captures", "keyframes apart", "angle °", "median disagreement %", "p90 %"],
                [[r["pair"], r.get("gap"), r.get("angle_deg"), r["median_pct"], r["p90_pct"]]
                 for r in rows])
    for located, what in (("locate", ""), ("camera_locate", " — camera")):
        rows = details.get(f"poses.{located}")
        if rows:  # camera: no commanded yaw, the error is against the split map's yaw
            out += _section(
                f"Poses — held-out captures located by mapper.sh locate{what}",
                ["capture", "commanded yaw °", "located yaw °", "error °", "vs mapped °",
                 "vs mapped m"],
                [[r["capture"], r.get("commanded_yaw_deg"), r.get("yaw_deg"),
                  r.get("yaw_err_deg"), r.get("vs_mapped_rot_deg"), r.get("vs_mapped_m")]
                 for r in rows])
    out += _map_update_details(details.get("map_update") or {})
    out += _parity_details((details.get("server_sh") or {}).get("parity") or {})
    if details.get("errors"):
        out += ["## Evaluator errors", "", *(f"* {e}" for e in details["errors"]), ""]
    return out


def _absent_rows(d: dict[str, Any], key: str = "final") -> list[list[Any]]:
    return [[a["label"], o["id"], o["label"], ", ".join(o.get("detected_as") or []),
             o.get("coverage") if o.get("localised") else "label only",
             o.get("image"), ", ".join(o.get("frames") or [])]
            for a in d.get("absent", []) for o in a.get(key, [])]


def _map_update_details(d: dict[str, Any]) -> list[str]:
    out: list[str] = []
    head = ["absent", "object id", "label", "detected as", "covers region", "in image",
            "observed in"]
    none: list[list[Any]] = [["none", None, None, None, None, None, None]]
    maps = [("map of the whole sequence (one update)", d)]
    maps += [(f"{name} map after its last update", row)
             for name, row in (d.get("splits") or {}).items()]
    for title, row in maps:
        if row.get("absent"):
            out += _section(f"Map update: objects of the absent labels in the {title}", head,
                            _absent_rows(row) or none)
    if d.get("before"):
        out += _section("Map update: the control, after an update of the images that show it",
                        head, _absent_rows({"absent": d["before"]}, "before") or none)
    for name, row in (d.get("splits") or {}).items():
        if row.get("stability"):
            out += _section(
                f"Map update: {name}, objects that never changed, first update vs last",
                ["last update id", "first update id", "labels", "IoU", "centre Δ m", "extent Δ"],
                [[r["single_id"], r["split_id"], f"{r['single_label']} / {r['split_label']}",
                  r["iou"], r["centre_delta_m"], r["extent_delta_rel"]] for r in row["stability"]])
        if row.get("ids_broken"):
            out += _section(f"Map update: {name}, published ids that did not persist",
                            ["id", "label", "published by update", "missing in update", "now"],
                            [[r["id"], r["label"], r["published_by_update"], r["update"],
                              r["now"]] for r in row["ids_broken"]])
    return out


def _parity_details(d: dict[str, Any]) -> list[str]:
    rows = [[r["case"], r["status"], r.get("why")] for r in d.get("cases", [])]
    rows += [[op, "not covered", why] for op, why in (d.get("uncovered") or {}).items()]
    return _section("server.sh parity: every operation, reference input and variant",
                    ["case", "result", "why"], rows) if rows else []


def _tails(runs: list[dict[str, Any]]) -> list[str]:
    out = []
    for r in runs:
        if not r["ok"]:
            tail = "\n".join((r.get("stderr_tail") or "").splitlines()[-TAIL_IN_SUMMARY:])
            out += [f"### {r['tag']} (exit {r['exit_code']}{', ' + r['error'] if r.get('error') else ''})",
                    "", "```", tail, "```", ""]
    return ["## Failed runs — stderr tails", "", *out] if out else []
