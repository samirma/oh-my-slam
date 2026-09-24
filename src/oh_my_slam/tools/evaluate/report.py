"""``result.json`` (machine-readable) and ``summary.md`` (human-readable, rendered from the result
alone, so it can be regenerated from any stored result)."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import psutil

from oh_my_slam.core.atomic import atomic_write_text
from oh_my_slam.tools.evaluate.metrics import Metrics
from oh_my_slam.tools.evaluate.runner import RunRecord

SCHEMA = "oh-my-slam-evaluation/1"
CATEGORIES = {"perf": "Performance", "pose": "Pose accuracy", "map": "Map quality",
              "seg": "Segmentation", "contract": "Contracts", "gt": "Ground truth"}
TAIL_IN_SUMMARY = 12


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


def build_result(metrics: Metrics, records: list[RunRecord], details: dict[str, Any], *,
                 started: str, finished: str, duration_s: float, env: dict[str, Any],
                 targets: Path, baseline: dict[str, Any]) -> dict[str, Any]:
    items = list(metrics.items.values())
    return {
        "schema": SCHEMA, "started": started, "finished": finished,
        "duration_s": round(duration_s, 1), "environment": env, "targets": str(targets),
        "baseline": baseline,
        "summary": {"metrics": len(items), "passed": sum(m.passed is True for m in items),
                    "failed": sum(m.passed is False for m in items),
                    "untargeted": sum(m.passed is None for m in items),
                    "regressions": sum(m.regression for m in items)},
        "metrics": [m.to_dict() for m in items],
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


def _cell(v: Any, width: int = 120) -> str:
    return _fmt(v).replace("|", "/").replace("\n", " ")[:width]


def _target(m: dict[str, Any]) -> str:
    t = m.get("target")
    return "—" if t is None else f"{t['op']} {_fmt(t['value'])} {t.get('unit') or ''}".strip()


def _verdict(m: dict[str, Any]) -> str:
    base = {True: "pass", False: "**FAIL**", None: "no target"}[m["passed"]]
    return base + (" ⚠ regression" if m.get("regression") else "")


def _section(title: str, head: list[str], rows: list[list[Any]]) -> list[str]:
    """A ``## title`` and a Markdown table."""
    return [f"## {title}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head),
            *("| " + " | ".join(_cell(c) for c in r) + " |" for r in rows), ""]


def summary_md(result: dict[str, Any]) -> str:
    s, env, base = result["summary"], result["environment"], result["baseline"]
    ok = s["failed"] == 0
    lines = [
        f"# oh-my-slam evaluation — {result['started']}", "",
        f"commit `{(env.get('commit') or '?')[:10]}`{' (uncommitted changes)' if env.get('dirty') else ''}"
        f" · {env.get('cpu')} · {env.get('memory_gb')} GB · {env.get('os')} · Python "
        f"{env.get('python')} · {result['duration_s'] / 60:.1f} min", "",
        f"targets `{result['targets']}` · baseline `{base.get('path')}`: {base.get('status')}"
        + (f" (run of {base['started']})" if base.get("started") else ""), "",
        f"**Result: {'PASS' if ok else 'FAIL'}** — {s['metrics']} metrics: {s['passed']} passed, "
        f"{s['failed']} failed, {s['untargeted']} without a target; {s['regressions']} "
        "regressions against the baseline.", ""]
    metrics = result["metrics"]
    failed = [m for m in metrics if m["passed"] is False]
    if failed:
        lines += _section(
            "Failed metrics", ["metric", "value", "target", "why"],
            [[m["id"], m["value"], _target(m), m.get("error") or ""] for m in failed])
    regress = [m for m in metrics if m.get("regression")]
    if regress:
        lines += _section(
            "Regressions against the baseline", ["metric", "value", "baseline", "target"],
            [[m["id"], m["value"], m["baseline"], _target(m)] for m in regress])
    for prefix, title in CATEGORIES.items():
        rows = [m for m in metrics if m["id"].split(".", 1)[0] == prefix]
        if rows:
            lines += _section(
                title, ["metric", "value", "target", "result", "baseline"],
                [[m["id"], m["value"], _target(m), _verdict(m), m["baseline"]] for m in rows])
    lines += [*_runs(result["runs"]), *_details(result["details"]), *_tails(result["runs"])]
    return "\n".join(lines)


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
    seg = [*([details["segmentation.restaurant"]] if "segmentation.restaurant" in details else []),
           *details.get("segmentation.frames", [])]
    if seg:
        out += _section(
            "Segmentation per image", ["image", "objects", "labels", "min score", "median score"],
            [[r["image"], r.get("objects"),
              ", ".join(f"{k} {v}" for k, v in (r.get("labels") or {}).items()) or r.get("error"),
              r.get("min_score"), r.get("median_score")] for r in seg])
    if details.get("map.stability"):
        out += _section(
            "Objects: one update vs split", ["single id", "split id", "labels", "IoU", "centre Δ m", "extent Δ"],
            [[r["single_id"], r["split_id"], f"{r['single_label']} / {r['split_label']}",
              r["iou"], r["centre_delta_m"], r["extent_delta_rel"]]
             for r in details["map.stability"]])
    if details.get("errors"):
        out += ["## Evaluator errors", "", *(f"* {e}" for e in details["errors"]), ""]
    return out


def _tails(runs: list[dict[str, Any]]) -> list[str]:
    out = []
    for r in runs:
        if not r["ok"]:
            tail = "\n".join((r.get("stderr_tail") or "").splitlines()[-TAIL_IN_SUMMARY:])
            out += [f"### {r['tag']} (exit {r['exit_code']}{', ' + r['error'] if r.get('error') else ''})",
                    "", "```", tail, "```", ""]
    return ["## Failed runs — stderr tails", "", *out] if out else []
