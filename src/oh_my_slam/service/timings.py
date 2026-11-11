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
# seconds (README "Expected durations"), measured on the serving machine (M3 Pro, 18 GB) with
# the prior-first path at 640x480 (task_map frames with priors and monocular depth, objects
# fused; ainex camera-only frames by MapAnything); the reads are placeholders until their
# groups measure them
DEFAULTS: dict[str, float] = {
    "registration_per_frame": 0.6,
    "fusion_per_frame": 1.3,
    # a job's phases: the checks of every frame sent, MapAnything's registration of a
    # camera-only frame, and what a job costs beyond its frames (clone, save, commit)
    "checks_per_frame": 0.02,
    "registration_mapanything_per_frame": 1.9,
    "job_overhead": 0.4,
    # fusion per kept frame by class (``fusion_class``: the prior's trust class and the depth
    # kind), its objects included: camera-only frames (MapAnything) fuse 5.8-11.2 s per frame,
    # frames with priors 0.7-2.0 s (the client's measurements on 04c1815)
    "fusion_per_frame_prior_depth": 1.0,
    "fusion_per_frame_prior_mono": 1.6,
    "fusion_per_frame_none_depth": 6.0,
    "fusion_per_frame_none_mono": 8.5,
    # the commit and the version's derivation before it is published, per keyframe of the
    # committed version (they grow with the map)
    "commit_per_keyframe": 0.01,
    "locate_per_frame": 2.0,
    "analyse_per_frame": 1.5,
    "grid_read": 0.5,
    "objects_read": 0.1,
}
# the mapper's stages (``core.timing.Stage``) that fuse what registration placed
FUSION_STAGES = frozenset({"persist_frames", "validity", "objects", "cloud", "export", "commit"})


FUSION_CLASSES = ("prior_depth", "prior_mono", "none_depth", "none_mono")


def fusion_class(trusted: bool, depth: bool) -> str:
    """A frame's fusion class: ``prior`` (metric or drifting) or ``none``, and ``depth`` (a
    depth image) or ``mono``."""
    return f"{'prior' if trusted else 'none'}_{'depth' if depth else 'mono'}"


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

    def predict_update(self, frames: int, camera_only: int = 0,
                       classes: dict[str, int] | None = None, keyframes: int = 0) -> float:
        """Seconds an update of ``frames`` frames is expected to take, ``camera_only`` of them
        registered by MapAnything, all of them kept, ``classes`` their fusion classes, on a map
        of ``keyframes`` keyframes (``predict_phases``)."""
        reg, fus = self.predict_phases(frames, frames - camera_only, camera_only, frames,
                                       classes, keyframes)
        return reg + fus

    def fusion_median(self, cls: str) -> float:
        """The fusion seconds per frame of a class: its own measurements, else its default."""
        return self.median(f"fusion_per_frame_{cls}")

    def predict_phases(self, sent: int, geometric: int, mapanything: int, kept: int,
                       classes: dict[str, int] | None = None, keyframes: int = 0
                       ) -> tuple[float, float]:
        """(registration, fusion) seconds of an update: its overhead, the checks of the
        ``sent`` frames, the registration of the frames each engine gets; the fusion of the
        ``kept`` ones by their fusion ``classes`` (counts; without them, the pooled median),
        their objects included, and the commit with the version's derivation, per keyframe of
        the version it commits (``keyframes`` stored plus the kept)."""
        if not sent:
            return 0.0, 0.0
        reg = (self.median("job_overhead") + sent * self.median("checks_per_frame")
               + geometric * self.median("registration_per_frame")
               + mapanything * self.median("registration_mapanything_per_frame"))
        if classes:
            share = kept / max(sum(classes.values()), 1)  # the classes of the sent frames
            fus = sum(n * share * self.fusion_median(c) for c, n in classes.items())
        else:
            fus = kept * self.median("fusion_per_frame")
        return reg, fus + (keyframes + kept) * self.median("commit_per_keyframe")
