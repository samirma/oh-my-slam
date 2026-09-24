"""Instance detection: the segment endpoint → background / min-score filters → cross-label
de-duplication → stable ordering. Also owns the vocabulary and label rules."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.client import protocol as p
from oh_my_slam.client.client import InferenceClient, connect
from oh_my_slam.core import rle, timing

DEFAULT_MIN_SCORE = 0.5
DEDUPE_IOU = 0.7
MIN_AREA_PX = 64

# Prompted so that YOLOE does not assign wall/floor pixels to real objects, never reported.
BACKGROUND_LABELS = frozenset(
    {"wall", "floor", "ceiling", "road", "sidewalk", "grass", "facade", "roof", "crosswalk"}
)

# Labels that may name the same physical object (used for identity across frames and merging).
COMPATIBLE_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"sofa", "couch", "loveseat", "armchair"}),
    frozenset({"chair", "armchair", "stool", "bar stool", "high chair"}),
    frozenset({"dining table", "coffee table", "side table", "desk", "table", "kitchen island",
               "counter", "bar counter", "nightstand"}),
    frozenset({"television", "tv", "computer monitor", "monitor", "screen"}),
    frozenset({"potted plant", "plant", "flowerpot", "planter", "vase", "flower", "bush"}),
    frozenset({"lamp", "floor lamp", "table lamp", "pendant lamp", "chandelier", "ceiling light",
               "wall lamp", "spotlight"}),
    frozenset({"street light", "lamppost", "pole", "traffic light"}),
    frozenset({"cabinet", "cupboard", "wardrobe", "chest of drawers", "bookcase", "shelf",
               "tv stand", "display case", "file cabinet", "drawer"}),
    frozenset({"cup", "mug", "glass", "wine glass"}),
    frozenset({"bottle", "wine bottle"}),
    frozenset({"painting", "picture frame", "poster", "mirror"}),
    frozenset({"rug", "carpet"}),
    frozenset({"cushion", "pillow"}),
    frozenset({"window", "shop window", "window shutter", "shutter"}),
    frozenset({"door", "doorway", "gate"}),
    frozenset({"car", "taxi", "van"}),
    frozenset({"cell phone", "remote control", "tablet computer"}),
)


# Floor-standing classes: when a floor plane is known, a box whose visible bottom floats at most
# this many metres above the floor is extended down to it (lower part occluded, e.g. a chair
# behind a table). Everything else keeps its visible-surface box.
_FLOOR_GAP = 0.8
FLOOR_STANDING: dict[str, float] = {
    **{k: _FLOOR_GAP for k in (
        "chair", "armchair", "stool", "bar stool", "high chair", "bench", "sofa", "ottoman", "bed",
        "crib", "dining table", "coffee table", "side table", "desk", "table", "cabinet",
        "cupboard", "wardrobe", "chest of drawers", "bookcase", "shelf", "tv stand", "nightstand",
        "floor lamp", "potted plant", "planter", "trash can", "recycling bin", "refrigerator",
        "freezer", "dishwasher", "washing machine", "dryer", "oven", "stove", "piano",
        "kitchen island", "counter", "bar counter", "display case", "vending machine",
        "water dispenser", "coat rack", "fire hydrant", "bollard", "mailbox", "bicycle",
        "motorcycle", "scooter", "car", "taxi", "bus", "truck", "van", "street light",
        "lamppost", "traffic light", "stroller", "wheelchair", "file cabinet", "toilet",
        "bathtub", "fireplace", "radiator", "suitcase", "fan", "column", "pillar", "tree",
        "palm tree", "bush", "hedge", "fence",
    )},
    "person": 1.2,
    "dog": 0.5,
    "cat": 0.5,
    "horse": 1.0,
}


def floor_gap(label: str) -> float:
    return FLOOR_STANDING.get(normalize_label(label), 0.0)


def normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", label.replace("_", " ").strip().lower())


@cache
def default_vocabulary() -> tuple[str, ...]:
    text = (resources.files("oh_my_slam.segmentation") / "data" / "default_labels.txt").read_text()
    seen: dict[str, None] = {}
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            seen.setdefault(normalize_label(line), None)
    return tuple(seen)


def compatible(a: str, b: str) -> bool:
    a, b = normalize_label(a), normalize_label(b)
    return a == b or any(a in g and b in g for g in COMPATIBLE_GROUPS)


@dataclass
class Detection:
    label: str
    score: float
    source: str
    mask: NDArray[np.bool_]
    box: tuple[float, float, float, float]

    @property
    def area(self) -> int:
        return int(self.mask.sum())


def mask_iou(a: NDArray[Any], b: NDArray[Any]) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    return float(inter / np.logical_or(a, b).sum())


def dedupe(dets: list[Detection], iou: float = DEDUPE_IOU) -> list[Detection]:
    """Greedy cross-label suppression by mask IoU (highest score wins)."""
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda x: (-x.score, -x.area)):
        x0, y0, x1, y1 = d.box
        dup = False
        for k in kept:
            kx0, ky0, kx1, ky1 = k.box
            if x1 < kx0 or kx1 < x0 or y1 < ky0 or ky1 < y0:
                continue
            if mask_iou(d.mask, k.mask) > iou:
                dup = True
                break
        if not dup:
            kept.append(d)
    return kept


def order(dets: list[Detection]) -> list[Detection]:
    """Stable output order: score desc, then mask area desc, then label."""
    return sorted(dets, key=lambda d: (-round(d.score, 6), -d.area, d.label))


def detect(
    image_path: Path,
    *,
    min_score: float = DEFAULT_MIN_SCORE,
    max_side: int = 1024,
    client: InferenceClient | None = None,
    min_area_px: int = MIN_AREA_PX,
) -> list[Detection]:
    """Detections for one image on the ``max_side`` grid (same grid as reconstruction)."""
    with timing.part("segmentation"):
        return _detect(image_path, min_score, max_side, client or connect(), min_area_px)


def _detect(image_path: Path, min_score: float, max_side: int, client: InferenceClient,
            min_area_px: int) -> list[Detection]:
    res = client.segment_image(
        p.SegmentRequest(
            image_path=str(Path(image_path).resolve()),
            labels=list(default_vocabulary()),
            max_side=max_side,
            conf=max(0.01, min_score - 1e-6),
        )
    )
    dets: list[Detection] = []
    for inst in res.instances:
        label = normalize_label(inst.label)
        if label in BACKGROUND_LABELS:
            continue
        score = float(np.clip(inst.score, 0.0, 1.0))
        if score < min_score:
            continue
        mask = rle.decode(inst.mask)
        if mask.sum() < min_area_px:
            continue
        b = inst.box_xyxy
        dets.append(Detection(label, score, inst.source, mask,
                              (float(b[0]), float(b[1]), float(b[2]), float(b[3]))))
    return order(dedupe(dets))
