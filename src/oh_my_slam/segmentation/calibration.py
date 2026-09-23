"""Score calibration: monotone piecewise-linear maps from raw model scores to calibrated
precision-like scores, one per segmentation path (``yoloe``, ``sam3``).

Maps are fitted by ``tools/calibrate_scores.py`` (isotonic regression on an LVIS subset) and
committed under ``segmentation/data``. A missing or identity map returns the raw score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from importlib import resources

import numpy as np


@dataclass(frozen=True)
class CalibrationMap:
    source: str
    knots_raw: tuple[float, ...]
    knots_calibrated: tuple[float, ...]
    status: str = "fitted"

    def __post_init__(self) -> None:
        if len(self.knots_raw) != len(self.knots_calibrated) or len(self.knots_raw) < 2:
            raise ValueError("calibration map needs >= 2 matching knots")
        if any(b < a for a, b in zip(self.knots_raw, self.knots_raw[1:], strict=False)):
            raise ValueError("raw knots must be non-decreasing")
        if any(b < a for a, b in zip(self.knots_calibrated, self.knots_calibrated[1:],
                                     strict=False)):
            raise ValueError("calibrated knots must be non-decreasing (monotone map)")

    def __call__(self, raw: float) -> float:
        return float(np.clip(np.interp(raw, self.knots_raw, self.knots_calibrated), 0.0, 1.0))

    @property
    def is_identity(self) -> bool:
        return self.status == "identity"

    @staticmethod
    def identity(source: str) -> CalibrationMap:
        return CalibrationMap(source, (0.0, 1.0), (0.0, 1.0), "identity")

    def to_dict(self) -> dict[str, object]:
        return {"source": self.source, "status": self.status,
                "knots_raw": list(self.knots_raw),
                "knots_calibrated": list(self.knots_calibrated)}


@cache
def load_map(source: str) -> CalibrationMap:
    try:
        text = (resources.files("oh_my_slam.segmentation") / "data" /
                f"calibration_{source}.json").read_text()
    except (FileNotFoundError, ModuleNotFoundError):
        return CalibrationMap.identity(source)
    d = json.loads(text)
    return CalibrationMap(
        source=d.get("source", source),
        knots_raw=tuple(float(v) for v in d["knots_raw"]),
        knots_calibrated=tuple(float(v) for v in d["knots_calibrated"]),
        status=d.get("status", "fitted"),
    )


def calibrate(raw: float, source: str) -> float:
    return load_map(source)(raw)
