"""Metrics, their targets (data: ``examples/targets.json``) and the comparison with a baseline run.

A target is ``{"op": "<=" | ">=", "value": x}`` plus an optional regression tolerance
(``tolerance_abs``, ``tolerance_rel``; a metric without either uses those under ``"defaults"``).
A metric passes when its value
meets the target; it regresses when it is worse than the baseline's value by more than
``max(tolerance_abs, tolerance_rel * |baseline|)`` ("worse" follows the target's direction). A metric
without a value (its command failed) fails; one without a target is reported as untargeted.
"""

from __future__ import annotations

import json
import math
import sys
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OPS = ("<=", ">=")
# Metric ids renamed since earlier runs: stored results and baselines are read under the new id.
RENAMED = {"seg.map.recall": "seg.map_consistency.map_objects_detected",
           "seg.map.precision": "seg.map_consistency.detections_in_map"}


@dataclass(frozen=True)
class Target:
    op: str
    value: float
    tolerance_abs: float = 0.0
    tolerance_rel: float = 0.0
    unit: str = ""
    description: str = ""

    def met(self, value: float) -> bool:
        return value <= self.value if self.op == "<=" else value >= self.value

    def regressed(self, value: float, baseline: float) -> bool:
        worse = value - baseline if self.op == "<=" else baseline - value
        return worse > max(self.tolerance_abs, self.tolerance_rel * abs(baseline)) + 1e-12

    def text(self) -> str:
        return f"{self.op} {self.value:g}" + (f" {self.unit}" if self.unit else "")


class TargetsError(ValueError):
    """The targets file is malformed."""


def load_targets(path: Path) -> dict[str, Target]:
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise TargetsError(f"cannot read targets {path}: {exc}") from exc
    defaults = doc.get("defaults", {})
    out = {}
    for mid, t in (doc.get("metrics") or {}).items():
        if t.get("op") not in OPS or not isinstance(t.get("value"), int | float):
            raise TargetsError(f"{path}: {mid}: needs op in {OPS} and a numeric value")
        # a metric's own tolerances replace the defaults (an omitted one is then 0)
        tol = t if {"tolerance_abs", "tolerance_rel"} & t.keys() else defaults
        out[mid] = Target(t["op"], float(t["value"]), float(tol.get("tolerance_abs", 0.0)),
                          float(tol.get("tolerance_rel", 0.0)), str(t.get("unit", "")),
                          str(t.get("description", "")))
    return out


@dataclass
class Metric:
    id: str
    value: float | None
    detail: Any = None
    error: str | None = None
    target: Target | None = None
    passed: bool | None = None  # None: no target
    baseline: float | None = None
    regression: bool = False

    def judge(self, target: Target | None) -> None:
        self.target = target
        if target is None:
            self.passed = None
        elif self.value is None or not math.isfinite(self.value):
            self.passed = False
            self.error = self.error or "no value"
        else:
            self.passed = target.met(self.value)

    def compare(self, baseline: float | None) -> None:
        self.baseline = baseline
        self.regression = bool(
            self.target is not None and baseline is not None and self.value is not None
            and math.isfinite(self.value) and math.isfinite(baseline)
            and self.target.regressed(self.value, baseline))

    def to_dict(self) -> dict[str, Any]:
        t = self.target
        return {
            "id": self.id, "value": self.value, "passed": self.passed, "error": self.error,
            "target": None if t is None else {
                "op": t.op, "value": t.value, "unit": t.unit, "tolerance_abs": t.tolerance_abs,
                "tolerance_rel": t.tolerance_rel},
            "baseline": self.baseline, "regression": self.regression, "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Metric:
        """A metric of a stored ``result.json`` with its value, detail and error (to judge it
        again against other targets or another baseline)."""
        v = d.get("value")
        return cls(RENAMED.get(str(d["id"]), str(d["id"])),
                   float(v) if isinstance(v, int | float) else None, d.get("detail"),
                   d.get("error"))


class Metrics:
    """The metrics of one run, by id (a metric is recorded once)."""

    def __init__(self) -> None:
        self.items: dict[str, Metric] = {}

    @classmethod
    def from_result(cls, result: dict[str, Any]) -> Metrics:
        """The metrics of a stored ``result.json``, not yet judged."""
        out = cls()
        for d in result.get("metrics", []):
            m = Metric.from_dict(d)
            out.items[m.id] = m
        return out

    def add(self, mid: str, value: float | None, detail: Any = None,
            error: str | None = None) -> None:
        if mid in self.items:
            raise ValueError(f"metric {mid} recorded twice")
        v = None if value is None else float(value)
        self.items[mid] = Metric(mid, v, detail, error if v is None else None)

    def fail(self, mids: list[str] | tuple[str, ...], error: str) -> None:
        for mid in mids:
            if mid not in self.items:
                self.add(mid, None, error=error)

    @contextmanager
    def expect(self, *mids: str) -> Iterator[None]:
        """Every id in ``mids`` is recorded after the block: ids the block did not record fail
        with the block's exception (or as not computed)."""
        error = "not computed"
        try:
            yield
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"  error while computing {', '.join(mids)}: {error}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        self.fail(list(mids), error)

    def judge(self, targets: dict[str, Target], baseline: dict[str, float] | None) -> None:
        for m in self.items.values():
            m.judge(targets.get(m.id))
            m.compare(None if baseline is None else baseline.get(m.id))


def baseline_values(result: dict[str, Any]) -> dict[str, float]:
    """Metric values of a stored ``result.json`` (the baseline run), by current metric id."""
    return {RENAMED.get(m["id"], m["id"]): float(m["value"]) for m in result.get("metrics", [])
            if isinstance(m.get("value"), int | float)}
