"""The thresholds of examples/targets.json (specs/evaluation.md "Data, not code"), shared by the
replay evaluator and the reference evaluator (tools/evaluate/).

An entry is ``{op, value, ...}`` (one number), ``{parts: {name: {op, value}}}`` (a metric with
several values that passes when every part does) or ``{info: true}``. Its other keys are threshold
parameters (``abs_m``, ``time_tolerance_rel``, ...) and regression tolerances (``tolerance_abs``,
``tolerance_rel``; the file's ``defaults`` otherwise).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_TARGETS = Path(__file__).resolve().parents[2] / "examples" / "targets.json"
OPS = ("<=", ">=", "<", ">")
REDUCERS = ("share", "median", "p90", "max", "sum")


class Targets:
    def __init__(self, metrics: dict[str, dict[str, Any]],
                 defaults: dict[str, float] | None = None) -> None:
        for mid, t in metrics.items():
            check_entry(mid, t)
        self.metrics = metrics
        self.defaults = {"tolerance_rel": 0.1, "tolerance_abs": 0.0, **(defaults or {})}

    @classmethod
    def load(cls, path: Path | None = None) -> Targets:
        data = json.loads((path or DEFAULT_TARGETS).read_text())
        return cls(dict(data.get("metrics", {})), data.get("defaults"))

    def get(self, mid: str) -> dict[str, Any] | None:
        return self.metrics.get(mid)

    def param(self, mid: str, key: str, default: float) -> float:
        """A threshold parameter of ``mid`` (e.g. ``locate.median_pos_error`` ``abs_m``)."""
        t = self.metrics.get(mid) or {}
        return float(t.get(key, default))

    def tolerance(self, mid: str, baseline: float) -> float:
        t = self.metrics.get(mid) or {}
        return max(float(t.get("tolerance_abs", self.defaults["tolerance_abs"])),
                   float(t.get("tolerance_rel", self.defaults["tolerance_rel"])) * abs(baseline))


def check_entry(mid: str, t: Any) -> None:
    if not isinstance(t, dict):
        raise ValueError(f"targets: {mid}: expected an object")
    if t.get("info"):
        return
    if "parts" in t:
        if not isinstance(t["parts"], dict) or not t["parts"]:
            raise ValueError(f"targets: {mid}: parts must be a non-empty object")
        for name, p in t["parts"].items():
            check_entry(f"{mid}#{name}", p)
            if "parts" in p:
                raise ValueError(f"targets: {mid}#{name}: parts do not nest")
        return
    if t.get("op") not in OPS or not isinstance(t.get("value"), int | float):
        raise ValueError(f"targets: {mid}: expected op in {OPS} and a numeric value")


def reduce(values: list[float], how: str) -> float | None:
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    if how == "share":
        return float(a.mean())
    if how == "median":
        return float(np.median(a))
    if how == "p90":
        return float(np.percentile(a, 90))
    if how == "sum":
        return float(a.sum())
    return float(a.max())


def meets(value: float, op: str, target: float) -> bool:
    if op == "<=":
        return value <= target
    if op == ">=":
        return value >= target
    if op == "<":
        return value < target
    return value > target


def judge(value: float | dict[str, float | None] | None,
          target: dict[str, Any] | None) -> str:
    """``pass``, ``fail``, ``info``, ``no_samples`` or ``no_target``."""
    if target is None:
        return "no_target" if value is not None else "no_samples"
    if target.get("info"):
        return "info" if value is not None else "no_samples"
    if "parts" in target:
        if not isinstance(value, dict) or all(v is None for v in value.values()):
            return "no_samples"
        results = [None if value.get(k) is None else meets(float(value[k]),  # type: ignore[arg-type]
                                                          p["op"], float(p["value"]))
                   for k, p in target["parts"].items()]
        if any(r is False for r in results):
            return "fail"
        return "pass" if all(r is True for r in results) else "no_samples"
    if value is None or isinstance(value, dict):
        return "no_samples"
    return "pass" if meets(float(value), target["op"], float(target["value"])) else "fail"


def worse(mid: str, value: float, baseline: float, target: dict[str, Any] | None,
          part: str | None = None) -> bool:
    """Whether ``value`` is worse than ``baseline`` in the direction of the target's op."""
    t = target or {}
    if part is not None:
        t = (t.get("parts") or {}).get(part) or {}
    op = t.get("op")
    if op is None or not math.isfinite(value) or not math.isfinite(baseline):
        return False
    return value > baseline if op in ("<=", "<") else value < baseline


def compare(result: dict[str, dict[str, Any]], baseline: dict[str, dict[str, Any]],
            targets: Targets) -> dict[str, dict[str, Any]]:
    """Per metric present in both: the baseline value and whether the run regressed (worse than
    the baseline by more than the metric's tolerance)."""
    out: dict[str, dict[str, Any]] = {}
    for mid, m in result.items():
        b = baseline.get(mid)
        if b is None or m.get("value") is None or b.get("value") is None:
            continue
        v, bv, t = m["value"], b["value"], targets.get(mid)
        regressed = False
        if isinstance(v, dict) and isinstance(bv, dict):
            for k, x in v.items():
                y = bv.get(k)
                if x is not None and y is not None and worse(mid, x, y, t, k) \
                        and abs(x - y) > targets.tolerance(mid, y):
                    regressed = True
        elif not isinstance(v, dict) and not isinstance(bv, dict):
            regressed = worse(mid, v, bv, t) and abs(v - bv) > targets.tolerance(mid, bv)
        out[mid] = {"baseline": bv, "regressed": regressed}
    return out
