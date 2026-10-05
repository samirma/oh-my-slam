"""``python -m oh_my_slam.tools.evaluate [--out DIR] [--set-baseline] [--baseline PATH]
[--targets PATH] [--street2 PATH]`` — see the package docstring."""

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
    TargetsError,
    baseline_values,
    load_targets,
)
from oh_my_slam.tools.evaluate.report import (
    build_result,
    environment,
    summary_counts,
    write_report,
)
from oh_my_slam.tools.evaluate.runner import REPO, Runner
from oh_my_slam.tools.evaluate.suite import EXAMPLES, STREET2, Evaluation
from oh_my_slam.tools.evaluate.viewer import BrowserProbe

DATA = Path.home() / "oh-my-slam-data" / "evaluations"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python -m oh_my_slam.tools.evaluate",
        description="Benchmark every entry point on examples/ (specs/high_level_spec.md §5). "
                    "Needs the model weights; stops and restarts the inference server to time "
                    "its cold start. Writes result.json and summary.md; exit 1 if any metric "
                    "fails.")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"result folder, outside the repository (default {DATA}/<UTC time>/)")
    ap.add_argument("--targets", type=Path, default=EXAMPLES / "targets.json",
                    help="metric targets (default: %(default)s)")
    ap.add_argument("--baseline", type=Path, default=DATA / "baseline.json",
                    help="stored baseline run to compare with (default: %(default)s)")
    ap.add_argument("--street2", type=Path, default=STREET2,
                    help="the street2 video every benchmark maps (default: %(default)s)")
    ap.add_argument("--rejudge", type=Path, default=None, metavar="RESULT_DIR",
                    help="run nothing: judge a stored run's result.json again with --targets and "
                         "--baseline (targets are data) and rewrite its result.json and "
                         "summary.md")
    ap.add_argument("--set-baseline", action="store_true",
                    help="store this run's result as the baseline, after comparing it with the "
                         "previous one")
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


def store_baseline(result_path: Path, baseline: Path) -> None:
    atomic_write_bytes(baseline, result_path.read_bytes())
    print(f"evaluate: stored {result_path} as the baseline {baseline}", file=sys.stderr)


def report_line(s: dict[str, Any]) -> str:
    regressions = ("baseline not compared" if s.get("regressions") is None
                   else f"{s['regressions']} regressions")
    return (f"{s['passed']} passed, {s['failed']} failed, {s['untargeted']} without a target, "
            f"{regressions}")


def rejudge(folder: Path, targets: dict[str, Any], baseline_path: Path, targets_path: Path
            ) -> dict[str, Any]:
    """A stored run judged again: its metric values with the current targets and baseline."""
    result = json.loads((folder / "result.json").read_text())
    metrics = Metrics()
    for row in result.get("metrics", []):
        metrics.add(row["id"], row.get("value"), row.get("detail"), row.get("error"))
    baseline, about = load_baseline(baseline_path)
    metrics.judge(targets, None if baseline is None else baseline_values(baseline))
    result["metrics"] = [m.to_dict() for m in metrics.items.values()]
    result["baseline"] = about
    result["targets"] = str(targets_path)
    result["summary"] = summary_counts(metrics, about)
    result["judged"] = datetime.now(UTC).isoformat(timespec="seconds")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        targets = load_targets(args.targets)
    except TargetsError as exc:
        print(f"evaluate: {exc}", file=sys.stderr)
        return 2
    if args.rejudge is not None:
        folder = args.rejudge.expanduser().resolve()
        try:
            result = rejudge(folder, targets, args.baseline.expanduser(), args.targets)
        except (OSError, ValueError, KeyError) as exc:
            print(f"evaluate: cannot judge {folder} again: {exc}", file=sys.stderr)
            return 2
        res, md = write_report(folder, result)
        if args.set_baseline:
            store_baseline(res, args.baseline.expanduser())
        print(f"evaluate: {report_line(result['summary'])} — {md}", file=sys.stderr)
        print(md)
        return 1 if result["summary"]["failed"] else 0
    start = datetime.now(UTC)
    out = (args.out or DATA / start.strftime("%Y%m%dT%H%M%SZ")).expanduser().resolve()
    if out == REPO or REPO in out.parents:
        print(f"evaluate: results must be stored outside the repository ({REPO})",
              file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    print(f"evaluate: results in {out}", file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    ev = Evaluation(out, Runner(out), BrowserProbe(), street2=args.street2.expanduser())
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
