"""Performance metrics per group of runs (spec §5 "Performance": per-stage and end-to-end wall time
and peak memory, per image and per mapping update).

End to end: the wall time (or, for ``view.sh``, the time until the page has rendered), the
command's peak resident set and the server's peak footprint. A group of several updates of one
map (``sum``) also reports its slowest update (``per_update_wall_s``).

Per stage (``OH_MY_SLAM_TIMINGS`` + ``memory.stage_peaks``): ``perf.<group>.stage.<stage>.s``,
``.client_peak_mb`` and ``.server_peak_gb``, for every stage the group's runs recorded (a stage is
data, so its metrics are too: they are targeted by pattern in ``examples/targets.json``). The
seconds are the median over the runs of a per-frame group (``median``) and the slowest run
otherwise, so a stage target holds per image and per mapping update; the memory is the peak over
the runs."""

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
    "locate": ("wall_s", "single"),  # mapper.sh locate on the reference map
    "mapper_office": ("wall_s", "single"),
    "mapper_office_split": ("wall_s", "max"),  # the slowest update of the split office maps
    "mapper_street2": ("wall_s", "single"),
}
PER_UPDATE = "per_update_wall_s"
STAGE_KEYS = ("s", "client_peak_mb", "server_peak_gb")
PER_RUN_DETAIL_MAX = 10


def perf_ids() -> list[str]:
    """The end-to-end metrics (the per-stage ones depend on the stages the runs record)."""
    ids = []
    for g, (t, agg) in PERF_GROUPS.items():
        ids += [f"perf.{g}.{k}" for k in (t, "client_peak_mb", "server_peak_gb")]
        if agg == "sum":
            ids.append(f"perf.{g}.{PER_UPDATE}")
    return ids


def stage_id(group: str, stage: str, key: str) -> str:
    return f"perf.{group}.stage.{stage}.{key}"


def _seconds(rec: RunRecord, what: str) -> float | None:
    if not rec.ok:
        return None
    return rec.wall_s if what == "wall_s" else rec.notes.get("render_s")


def _peak(values: list[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return max(known) if known else None


def _stages(recs: list[RunRecord], agg: str) -> dict[str, dict[str, float | None]]:
    """Per stage over the runs: seconds (median for a per-frame group, else the slowest run),
    peak client MB and peak server GB."""
    per: dict[str, list[dict[str, float | None]]] = {}
    for r in recs:
        for k, v in (r.stages or {}).items():
            per.setdefault(k, []).append(v)
    def pick(xs: list[float]) -> float:
        return float(np.median(xs)) if agg == "median" else max(xs)

    return {k: {"s": round(pick([float(x["s"] or 0.0) for x in v]), 3),
                "client_peak_mb": _peak([x.get("client_peak_mb") for x in v]),
                "server_peak_gb": _peak([x.get("server_peak_gb") for x in v])}
            for k, v in per.items()}


def perf_metrics(m: Metrics, records: list[RunRecord]) -> None:
    for group, (what, agg) in PERF_GROUPS.items():
        recs = [r for r in records if r.spec.group == group]
        ids = [f"perf.{group}.{what}", f"perf.{group}.client_peak_mb",
               f"perf.{group}.server_peak_gb"] + ([f"perf.{group}.{PER_UPDATE}"]
                                                  if agg == "sum" else [])
        ok = [r for r in recs if r.ok]
        if not ok:
            m.fail(ids, recs[0].failure() if recs else "not run")
            continue
        secs = [_seconds(r, what) for r in recs]
        timed = [s for s in secs if s is not None]
        stages = _stages(ok, agg)
        detail: dict[str, Any] = {"runs": len(recs), "failed": [r.tag for r in recs if not r.ok],
                                  "stages": stages}
        if len(recs) <= PER_RUN_DETAIL_MAX:
            detail["per_run"] = {r.tag: {what: None if s is None else round(s, 3),
                                         "stages": r.stages}
                                 for r, s in zip(recs, secs, strict=True)}
        else:
            detail[f"max_{what}"] = round(max(timed), 3) if timed else None
        if not timed or (agg != "median" and len(timed) < len(recs)):
            bad = next(r for r, s in zip(recs, secs, strict=True) if s is None)
            m.fail(ids[:1] + ids[3:], bad.notes.get("render_error") or bad.failure())
            m.items[ids[0]].detail = detail
        else:
            value = {"single": max(timed), "median": float(np.median(timed)),
                     "sum": sum(timed), "max": max(timed)}[agg]
            m.add(ids[0], value, detail)
            if agg == "sum":
                m.add(ids[3], max(timed), {"updates": len(timed)})
        m.add(ids[1], max(r.client_peak_mb for r in ok))
        server = [r.server_peak_gb for r in ok if r.server_peak_gb is not None]
        m.add(ids[2], max(server) if server else None,
              error=None if server else "server footprint not sampled (server not running?)")
        for stage, row in stages.items():
            for key in STAGE_KEYS:
                m.add(stage_id(group, stage, key), row[key],
                      error=None if row[key] is not None else
                      f"{key} not sampled for stage {stage} (server not running?)")
