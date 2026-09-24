"""Segmentation: detections, labels and scores per image, and their agreement with the map.

Method: the map exports, for each object, the keyframes that observed it (the object's
``frame_intervals``, keyframe indices). For every registered capture, the labels of the map objects
observed in its keyframe are compared with the labels ``segment.sh -i`` detects in the same capture:
labels are paired one-to-one, identical labels first, then compatible ones (e.g. sofa / couch).
Recall = paired map objects / map objects observed; precision = paired detections / detections,
both summed over the frames."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from oh_my_slam.segmentation.detect import compatible, normalize_label
from oh_my_slam.tools.evaluate.metrics import Metrics
from oh_my_slam.tools.evaluate.scene import DocObject

MAP_AGREEMENT_METRICS = ("recall", "precision")


def detection_row(name: str, objs: list[DocObject]) -> dict[str, Any]:
    """One report row: detections, labels and scores of one image."""
    scores = [o.score for o in objs if o.score is not None]
    return {"image": name, "objects": len(objs),
            "labels": dict(Counter(o.label for o in objs).most_common()),
            "min_score": round(min(scores), 3) if scores else None,
            "median_score": round(float(np.median(scores)), 3) if scores else None,
            "scores": [round(s, 3) for s in scores]}


def paired_labels(a: list[str], b: list[str]) -> int:
    """Size of a one-to-one pairing of labels ``a`` with ``b``: exact matches first, then
    compatible labels."""
    left = [normalize_label(x) for x in a]
    right = [normalize_label(x) for x in b]
    n = 0
    for exact in (True, False):
        for x in list(left):
            j = next((k for k, y in enumerate(right) if (x == y if exact else compatible(x, y))),
                     None)
            if j is not None:
                left.remove(x)
                right.pop(j)
                n += 1
    return n


def map_agreement(m: Metrics, prefix: str, frames: dict[str, list[DocObject]],
                  map_objs: list[DocObject], sources: dict[int, str]) -> list[dict[str, Any]]:
    """``<prefix>.recall`` / ``.precision`` of per-frame detections against the map objects that
    observed each frame; ``frames``: capture name → its ``segment.sh -i`` objects."""
    rows = []
    seen = pairs = dets = 0
    for key, capture in sorted(sources.items()):
        if capture not in frames:
            continue
        observed = [o.label for o in map_objs if key in o.frames]
        detected = [o.label for o in frames[capture]]
        k = paired_labels(observed, detected)
        rows.append({"image": capture, "keyframe": key, "map_objects": len(observed),
                     "detections": len(detected), "paired": k})
        seen, pairs, dets = seen + len(observed), pairs + k, dets + len(detected)
    detail = {"frames": len(rows), "map_observations": seen, "detections": dets, "paired": pairs}
    m.add(f"{prefix}.recall", pairs / seen if seen else None, detail,
          error=None if seen else "no map object is linked to an evaluated frame")
    m.add(f"{prefix}.precision", pairs / dets if dets else None, detail,
          error=None if dets else "no detections in the registered frames")
    return rows
