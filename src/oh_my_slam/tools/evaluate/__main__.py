"""``python -m oh_my_slam.tools.evaluate [--out DIR] [--set-baseline] [--baseline PATH]
[--targets PATH] [--splits N] [--resummarise DIR|latest]`` — see the package docstring."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from oh_my_slam.core.atomic import atomic_write_bytes
from oh_my_slam.tools.evaluate.metrics import (
    Metrics,
    Target,
    TargetsError,
    baseline_values,
    load_targets,
)
from oh_my_slam.tools.evaluate.report import (
    build_result,
    environment,
    rejudged,
    write_report,
)
from oh_my_slam.tools.evaluate.runner import REPO, Runner
from oh_my_slam.tools.evaluate.suite import EXAMPLES, MIN_SPLITS, Evaluation
from oh_my_slam.tools.evaluate.viewer import BrowserProbe

DATA = Path.home() / "oh-my-slam-data" / "evaluations"
LATEST = "latest"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python -m oh_my_slam.tools.evaluate",
        description="Benchmark every entry point on examples/ (high_level_spec.md §5). Needs the "
                    "model weights; stops and restarts the inference server to time its cold "
                    "start. Writes result.json and summary.md; exit 1 if any metric fails.",
        epilog="To keep the final run as the baseline later runs are compared with: "
               "python -m oh_my_slam.tools.evaluate --resummarise latest --set-baseline")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"result folder, outside the repository (default {DATA}/<UTC time>/)")
    ap.add_argument("--targets", type=Path, default=EXAMPLES / "targets.json",
                    help="metric targets (default: %(default)s)")
    ap.add_argument("--baseline", type=Path, default=DATA / "baseline.json",
                    help="stored baseline run to compare with (default: %(default)s)")
    ap.add_argument("--set-baseline", action="store_true",
                    help="store the result (this run's, or with --resummarise that stored run's) "
                         "as the baseline, after comparing it with the previous one")
    ap.add_argument("--splits", type=int, default=MIN_SPLITS,
                    help="updates the split map is built from (default: %(default)s, minimum 3)")
    ap.add_argument("--resummarise", metavar="DIR|latest", default=None,
                    help="run nothing: judge the metric values of DIR/result.json (latest: the "
                         f"newest run under {DATA}) again against --targets and --baseline and "
                         "rewrite its result.json and summary.md")
    return ap.parse_args(argv)


def load_baseline(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """(baseline result, what the report says about it); a missing baseline is not an error."""
    if not path.exists():
        return None, {"path": str(path), "status": "missing"}
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return None, {"path": str(path), "status": f"unreadable: {exc}"}
    return doc, {"path": str(path), "status": "compared", "started": doc.get("started"),
                 "commit": (doc.get("environment") or {}).get("commit")}


def latest_run(root: Path | None = None) -> Path | None:
    """The newest run folder under ``root`` (default ``DATA``; by its ``<UTC>`` name) holding a
    result.json."""
    runs = sorted(p.parent for p in (root or DATA).glob("*/result.json"))
    return runs[-1] if runs else None


def store_baseline(result_path: Path, baseline: Path) -> None:
    atomic_write_bytes(baseline, result_path.read_bytes())
    print(f"evaluate: stored {result_path} as the baseline {baseline}", file=sys.stderr)


def report_line(s: dict[str, Any]) -> str:
    regressions = ("baseline not compared" if s.get("regressions") is None
                   else f"{s['regressions']} regressions")
    return (f"{s['passed']} passed, {s['failed']} failed, {s['untargeted']} without a target, "
            f"{regressions}")


def resummarise(args: argparse.Namespace, targets: dict[str, Target]) -> int:
    folder = latest_run() if args.resummarise == LATEST else Path(args.resummarise).expanduser()
    if folder is None or not (folder / "result.json").is_file():
        print(f"evaluate: no result.json in {folder or DATA}", file=sys.stderr)
        return 2
    result = json.loads((folder / "result.json").read_text())
    baseline_path = args.baseline.expanduser()
    baseline, about = load_baseline(baseline_path)
    metrics = Metrics.from_result(result)
    metrics.judge(targets, None if baseline is None else baseline_values(baseline))
    at = datetime.now(UTC).isoformat(timespec="seconds")
    result = rejudged(result, metrics, targets=args.targets, baseline=about, at=at)
    res, md = write_report(folder, result)
    if args.set_baseline:
        store_baseline(res, baseline_path)
    s = result["summary"]
    print(f"evaluate: {folder.name} judged again: {report_line(s)} — {md}", file=sys.stderr)
    print(md)
    return 1 if s["failed"] else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.splits < MIN_SPLITS:
        print(f"evaluate: --splits must be at least {MIN_SPLITS}", file=sys.stderr)
        return 2
    try:
        targets = load_targets(args.targets)
    except TargetsError as exc:
        print(f"evaluate: {exc}", file=sys.stderr)
        return 2
    if args.resummarise is not None:
        return resummarise(args, targets)
    start = datetime.now(UTC)
    out = (args.out or DATA / start.strftime("%Y%m%dT%H%M%SZ")).expanduser().resolve()
    if out == REPO or REPO in out.parents:
        print(f"evaluate: results must be stored outside the repository ({REPO})",
              file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    print(f"evaluate: results in {out}", file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    ev = Evaluation(out, Runner(out), BrowserProbe(), splits=args.splits)
    ev.run_all()
    baseline, about = load_baseline(args.baseline.expanduser())
    ev.metrics.judge(targets, None if baseline is None else baseline_values(baseline))
    result = build_result(
        ev.metrics, ev.runner.records, ev.details, started=start.isoformat(timespec="seconds"),
        finished=datetime.now(UTC).isoformat(timespec="seconds"),
        duration_s=time.perf_counter() - t0, env=environment(REPO), targets=args.targets,
        baseline=about)
    res, md = write_report(out, result)
    if args.set_baseline:
        store_baseline(res, args.baseline.expanduser())
    s = result["summary"]
    print(f"evaluate: {report_line(s)} — {md}", file=sys.stderr)
    print(md)
    return 1 if s["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
