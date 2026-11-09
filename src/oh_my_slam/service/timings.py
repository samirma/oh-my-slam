"""Expected durations (api.md §8): the median and p90 of the last ``WINDOW`` measurements of each
kind on this machine, as run, kept in ``<data>/oms/timings.json``. Before any measurement a kind
reports its default (README "Expected durations"), marked ``source: default``."""

from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from oh_my_slam.core.atomic import atomic_write_json

WINDOW = 50
# seconds; measured on the serving machine (M3 Pro, 18 GB) with examples/office_sequence
DEFAULTS: dict[str, float] = {
    "registration_per_frame": 6.0,
    "fusion_per_frame": 2.0,
    "locate_per_frame": 2.0,
    "analyse_per_frame": 1.5,
    "grid_read": 0.5,
    "objects_read": 0.1,
}
# the mapper's stages (``core.timing.Stage``) that fuse what registration placed
FUSION_STAGES = frozenset({"persist_frames", "validity", "objects", "cloud", "export", "commit"})


class Timings:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._data: dict[str, deque[float]] = {k: deque(maxlen=WINDOW) for k in DEFAULTS}
        if path is not None and path.exists():
            try:
                saved = json.loads(path.read_text())
                for k, v in saved.items():
                    if k in self._data:
                        self._data[k].extend(float(x) for x in v)
            except (OSError, ValueError, TypeError):
                pass

    def add(self, kind: str, seconds: float, count: int = 1) -> None:
        """Record ``count`` measurements of ``seconds`` each (a job's per-frame mean)."""
        with self._lock:
            self._data[kind].extend([float(seconds)] * max(count, 1))
            if self.path is not None:
                atomic_write_json(self.path, {k: list(v) for k, v in self._data.items()})

    def median(self, kind: str) -> float:
        with self._lock:
            v = list(self._data[kind])
        return float(np.median(v)) if v else DEFAULTS[kind]

    def body(self) -> dict[str, Any]:
        """The ``Health.timings`` object."""
        out = {}
        with self._lock:
            for k, d in self._data.items():
                v = np.array(d)
                if v.size:
                    out[k] = {"median_s": round(float(np.median(v)), 3),
                              "p90_s": round(float(np.percentile(v, 90)), 3),
                              "count": int(v.size), "source": "measured"}
                else:
                    out[k] = {"median_s": DEFAULTS[k], "p90_s": DEFAULTS[k], "count": 0,
                              "source": "default"}
        return out

    def predict_update(self, frames: int) -> float:
        """Seconds an update of ``frames`` frames is expected to take."""
        return frames * (self.median("registration_per_frame") + self.median("fusion_per_frame"))
