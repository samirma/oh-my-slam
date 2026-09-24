"""Performance metrics per group of runs: end-to-end wall time (or, for ``view.sh``, the time
until the page has rendered), the command's peak resident set and the server's peak footprint;
per-stage times (``OH_MY_SLAM_TIMINGS``) are kept in the metric detail."""

from __future__ import annotations

from typing import Any

import numpy as np

from oh_my_slam.tools.evaluate.metrics import Metrics
from oh_my_slam.tools.evaluate.runner import RunRecord

# group → (time metric, aggregation over the group's runs)
PERF_GROUPS: dict[str, tuple[str, str]] = {
    "reconstruct_json": ("wall_s", "single"),
    "reconstruct_ply": ("wall_s", "single"),
    "segment_image": ("wall_s", "single"),
    "segment_frames": ("wall_s", "median"),  # per frame
    "view_image": ("render_s", "single"),
    "mapper_single": ("wall_s", "single"),
    "mapper_split": ("wall_s", "sum"),  # the whole sequence over all updates
    "segment_map": ("wall_s", "single"),
    "view_map": ("render_s", "single"),
}
PER_RUN_DETAIL_MAX = 10


def perf_ids() -> list[str]:
    return [f"perf.{g}.{k}" for g, (t, _) in PERF_GROUPS.items()
            for k in (t, "client_peak_mb", "server_peak_gb")]


def _seconds(rec: RunRecord, what: str) -> float | None:
    if not rec.ok:
        return None
    return rec.wall_s if what == "wall_s" else rec.notes.get("render_s")


def _stages(recs: list[RunRecord]) -> dict[str, float]:
    """Median over the runs of each stage's seconds."""
    per: dict[str, list[float]] = {}
    for r in recs:
        for k, v in ((r.timings or {}).get("stages_s") or {}).items():
            per.setdefault(k, []).append(float(v))
    return {k: round(float(np.median(v)), 3) for k, v in per.items()}


def perf_metrics(m: Metrics, records: list[RunRecord]) -> None:
    for group, (what, agg) in PERF_GROUPS.items():
        recs = [r for r in records if r.spec.group == group]
        ids = [f"perf.{group}.{what}", f"perf.{group}.client_peak_mb",
               f"perf.{group}.server_peak_gb"]
        ok = [r for r in recs if r.ok]
        if not ok:
            m.fail(ids, recs[0].failure() if recs else "not run")
            continue
        secs = [_seconds(r, what) for r in recs]
        timed = [s for s in secs if s is not None]
        detail: dict[str, Any] = {"runs": len(recs), "failed": [r.tag for r in recs if not r.ok],
                                  "stages_s": _stages(ok)}
        if len(recs) <= PER_RUN_DETAIL_MAX:
            detail["per_run"] = {r.tag: {what: None if s is None else round(s, 3),
                                         "stages_s": (r.timings or {}).get("stages_s")}
                                 for r, s in zip(recs, secs, strict=True)}
        else:
            detail[f"max_{what}"] = round(max(timed), 3) if timed else None
        if not timed or (agg != "median" and len(timed) < len(recs)):
            bad = next(r for r, s in zip(recs, secs, strict=True) if s is None)
            m.add(ids[0], None, detail,
                  error=bad.notes.get("render_error") or bad.failure())
        else:
            value = {"single": max(timed), "median": float(np.median(timed)),
                     "sum": sum(timed)}[agg]
            m.add(ids[0], value, detail)
        m.add(ids[1], max(r.client_peak_mb for r in ok))
        server = [r.server_peak_gb for r in ok if r.server_peak_gb is not None]
        m.add(ids[2], max(server) if server else None,
              error=None if server else "server footprint not sampled (server not running?)")
