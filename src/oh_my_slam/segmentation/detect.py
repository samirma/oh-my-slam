"""Instance detection: the segment endpoint (a fine and a coarse pass) → background / min-score /
size filters on the caller's grid → fusion of the passes → cross-label de-duplication → stable
ordering. Also owns the vocabulary and label rules.

Grid-independent: what the detector is shown never depends on the caller's grid (1024 px for a
single image, 768 px for map keyframes); only the masks are resampled onto it. The same image
therefore gives the same detections to ``segment.sh -i`` / ``view.sh -i`` / ``reconstruct.sh`` and
to the mapper (``DETECT_SIDE``, ``COARSE_SIDE``).

Stable across thresholds: the server is asked for every detection above ``request_floor``
(``TRUSTED_SCORE`` for the default and higher thresholds, ``DETECTION_FLOOR`` below it) and
``--min-score`` is applied only after the objects have been resolved
(``segmentation.api.segment_frame``). Fusion, de-duplication and id order follow ``priority``
(score first), so a detection is only ever suppressed by a higher-scoring one; overlapping pixels
are resolved by nesting (``claim_order``), trusted detections first, independently of
``--min-score``. Raising the threshold therefore removes objects from the end and never changes
the ones that remain, and the detections below ``TRUSTED_SCORE`` that only a lower floor returns
never change an object at or above it."""

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
from oh_my_slam.core.images import size_at_max_side, upright_size

DEFAULT_MIN_SCORE = 0.5
# Score floor requested from the server, and so the lowest accepted --min-score (the spec sets no
# bound). Lowering it never changes an object at or above TRUSTED_SCORE: detections are ordered
# and de-duplicated by score first, those below TRUSTED_SCORE claim only pixels no trusted one
# covers, and the detector suppresses a candidate only by a higher-scoring one.
DETECTION_FLOOR = 0.05
# Detections scoring at least this claim overlapping pixels before lower-scoring ones
# (``claim_order``): the default threshold, so the default output never depends on detections
# that are only reported with a lower --min-score.
TRUSTED_SCORE = DEFAULT_MIN_SCORE
DEDUPE_IOU = 0.7
MIN_AREA_PX = 64

# Detection passes. YOLOE always runs on a DETECT_SIDE input; a pass sets the image it is given,
# downscaled to at most that many pixels (and upsampled by the detector when smaller). The fine
# pass shows the full detail at the detector's resolution: small objects (restaurant.jpg: 115
# detections at 0.5; 42 on the coarse pass alone). The coarse pass shows the image at
# COARSE_SIDE, whose lost detail misleads the detector less on large plain surfaces: a dark
# monitor showing a menu bar is a 'television' at 0.13 on the fine pass and a 'computer monitor'
# at 0.56 on the coarse one (office_sequence 20260929_122224.jpg); windows, a bag, a tree. The
# coarse pass contributes only objects covering at least COARSE_MIN_SHARE of the image (its small
# detections are the fine pass's, blurred into other classes: a wire, a bottle opener), and it is
# skipped for images no larger than COARSE_SIDE (the two inputs would be the same).
DETECT_SIDE = 1024
COARSE_SIDE = 768
COARSE_MIN_SHARE = 0.05
# The passes' detections of one object overlap by less than DEDUPE_IOU when their outlines
# differ (a window with or without its frame: IoU 0.6-0.68): a detection overlapping a
# higher-scoring detection of the other pass by PASS_IOU is that detection.
PASS_IOU = 0.5

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
    frozenset({"book", "magazine", "newspaper"}),
    frozenset({"painting", "picture frame", "poster", "mirror"}),
    frozenset({"rug", "carpet"}),
    frozenset({"cushion", "pillow"}),
    frozenset({"window", "shop window", "window shutter", "shutter"}),
    frozenset({"door", "doorway", "gate"}),
    frozenset({"car", "taxi", "van"}),
    frozenset({"cell phone", "remote control", "tablet computer"}),
)


# Floor grounding (``segmentation.obb.ground_upright``). When a floor plane is known, the box of a
# floor-standing class whose visible bottom floats at most the class's gap (metres) above the floor
# is extended down to it: its lower part is occluded (a chair behind a table, a door behind a
# kitchen island) or too thin to be seen (a table's legs). Everything else keeps its
# visible-surface box.
_FLOOR_GAP = 0.8
# Classes that also stand on furniture or hang on walls (a plant on a counter, a wall shelf): only
# a bottom trimmed off at the floor contact (depth edges, occlusion by the floor's clutter) is
# closed.
_CONTACT_GAP = 0.15
FLOOR_STANDING: dict[str, float] = {
    **{k: _FLOOR_GAP for k in (
        "chair", "armchair", "stool", "bar stool", "high chair", "bench", "sofa", "ottoman", "bed",
        "crib", "dining table", "coffee table", "side table", "desk", "table", "cabinet",
        "cupboard", "wardrobe", "chest of drawers", "bookcase", "tv stand", "nightstand",
        "floor lamp", "trash can", "recycling bin", "refrigerator",
        "freezer", "dishwasher", "washing machine", "dryer", "oven", "stove", "piano",
        "kitchen island", "counter", "bar counter", "display case", "vending machine",
        "water dispenser", "coat rack", "fire hydrant", "bollard", "mailbox", "bicycle",
        "motorcycle", "scooter", "car", "taxi", "bus", "truck", "van", "street light",
        "lamppost", "traffic light", "stroller", "wheelchair", "file cabinet", "toilet",
        "bathtub", "fireplace", "column", "pillar", "door", "doorway", "gate", "tree",
        "palm tree", "bush", "hedge", "fence",
    )},
    **{k: _CONTACT_GAP for k in (
        "shelf", "potted plant", "planter", "suitcase", "fan", "radiator",
    )},
    "person": 1.2,
    "dog": 0.5,
    "cat": 0.5,
    "horse": 1.0,
}
# Classes whose visible part is often just their top surface (a table seen from above): grounded
# whatever the visible height. For the other classes the visible part must span at least
# GROUND_MIN_VISIBLE of the grounded height (the gap at most 4x the evidence), so a box is never
# extrapolated to the floor from a sliver (a picture on a book cover detected as a person); a
# chair whose backrest shows above a table (~0.25 of its height) is still grounded.
TOP_SURFACE = frozenset({
    "dining table", "coffee table", "side table", "desk", "table", "kitchen island", "counter",
    "bar counter", "nightstand", "tv stand", "bed", "crib", "bench", "ottoman",
})
GROUND_MIN_VISIBLE = 0.2
# Horizontal surfaces that run out of the view and under the items resting on them (a counter top
# around a book lying on it, a rug under a chair): the detector splits one such surface into
# several instances of the same kind around those items. Instances of these classes with
# compatible labels that touch in one image with continuous depth are pieces of one surface
# (``split_surface``; joined by the mapper). Other classes that touch side by side (two cabinets,
# books on a shelf, chairs in a row) are usually separate objects.
SURFACES = TOP_SURFACE | frozenset({"rug", "carpet"})


def surface_label(label: str) -> bool:
    """Whether a label names a horizontal surface (``SURFACES``)."""
    return normalize_label(label) in SURFACES


def split_surface(a: str, b: str) -> bool:
    """Whether instances labelled ``a`` and ``b`` that touch with continuous depth may be pieces
    of one horizontal surface (``SURFACES``, compatible labels)."""
    return normalize_label(a) in SURFACES and normalize_label(b) in SURFACES and compatible(a, b)


def grounding(label: str) -> tuple[float, float]:
    """(largest gap to the floor that is closed, smallest visible share of the grounded height)
    for a class; (0, 0) for classes that are never grounded."""
    lab = normalize_label(label)
    gap = FLOOR_STANDING.get(lab, 0.0)
    return gap, (0.0 if lab in TOP_SURFACE or gap == 0.0 else GROUND_MIN_VISIBLE)


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
        return int(np.count_nonzero(self.mask))


def mask_iou(a: NDArray[Any], b: NDArray[Any]) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    return float(inter / np.logical_or(a, b).sum())


def priority(d: Detection) -> tuple[float, int, str, tuple[float, float, float, float]]:
    """Total, deterministic order of detections: score desc, then mask area desc, label, box."""
    return (-d.score, -d.area, d.label, d.box)


def claim_order(d: Detection) -> tuple[bool, int, float, str, tuple[float, float, float, float]]:
    """Order in which detections claim overlapping pixels (first wins): trusted detections
    (``TRUSTED_SCORE`` and above) before the others, and within each tier the smallest mask first,
    so a nested object (a plate on a table, a person on a sofa) keeps its pixels whatever the
    scores. A lower-scoring detection only gets pixels no trusted one covers: those are mostly
    fragments of the trusted object around them (a chair's back detected as another chair)."""
    return (d.score < TRUSTED_SCORE, d.area, -d.score, d.label, d.box)


def _overlapping(a: Detection, b: Detection, iou: float) -> bool:
    """Whether two masks overlap by more than ``iou`` (disjoint boxes: never)."""
    x0, y0, x1, y1 = a.box
    kx0, ky0, kx1, ky1 = b.box
    if x1 < kx0 or kx1 < x0 or y1 < ky0 or ky1 < y0:
        return False
    return mask_iou(a.mask, b.mask) > iou


def dedupe(dets: list[Detection], iou: float = DEDUPE_IOU) -> list[Detection]:
    """Greedy cross-label suppression by mask IoU (the higher-priority detection wins)."""
    kept: list[Detection] = []
    for d in sorted(dets, key=priority):
        if not any(_overlapping(d, k, iou) for k in kept):
            kept.append(d)
    return kept


def fuse_passes(passes: list[list[Detection]], iou: float = PASS_IOU) -> list[Detection]:
    """One list from the detections of several passes over one image (same grid): a detection
    overlapping a higher-priority detection of another pass by more than ``iou`` is that object
    seen again, and is dropped (label, score and mask are the higher-scoring pass's)."""
    tagged = sorted(((d, k) for k, ds in enumerate(passes) for d in ds),
                    key=lambda t: priority(t[0]))
    kept: list[tuple[Detection, int]] = []
    for d, k in tagged:
        if not any(j != k and _overlapping(d, e, iou) for e, j in kept):
            kept.append((d, k))
    return [d for d, _ in kept]


def order(dets: list[Detection]) -> list[Detection]:
    """Stable output order (``priority``)."""
    return sorted(dets, key=priority)


def request_floor(min_score: float) -> float:
    """Score floor to request for objects at ``min_score``: ``TRUSTED_SCORE`` from the default
    threshold up, ``DETECTION_FLOOR`` below it. The detections between the two only claim pixels
    no trusted detection covers and suppress only lower-scoring ones, so they never change an
    object at or above ``TRUSTED_SCORE``; the default output does not pay for them."""
    return TRUSTED_SCORE if min_score >= TRUSTED_SCORE else DETECTION_FLOOR


def detect(
    image_path: Path,
    *,
    min_score: float = DEFAULT_MIN_SCORE,
    floor: float = DETECTION_FLOOR,
    max_side: int = DETECT_SIDE,
    client: InferenceClient | None = None,
    min_area_px: int = MIN_AREA_PX,
) -> list[Detection]:
    """Detections scoring at least ``min_score``, with masks and boxes on the ``max_side`` grid
    (the reconstruction's grid of the image; ``min_area_px`` counts its pixels). The detector's
    input does not depend on ``max_side`` (``DETECT_SIDE``, ``COARSE_SIDE``), so every caller
    gets the same detections of an image. The server is asked for everything above ``floor``
    (<= ``min_score``), so the request — and the detections it returns — do not depend on
    ``min_score``."""
    if not 0.0 < floor <= min_score:
        raise ValueError(f"min_score {min_score} is below the detection floor {floor}")
    with timing.part("segmentation"):
        return _detect(image_path, min_score, floor, max_side, client or connect(), min_area_px)


def _detect(image_path: Path, min_score: float, floor: float, max_side: int,
            client: InferenceClient, min_area_px: int) -> list[Detection]:
    w, h = upright_size(Path(image_path))
    grid = size_at_max_side(w, h, max_side)
    fine = _detect_pass(image_path, DETECT_SIDE, min_score, floor, grid, client, min_area_px)
    if max(w, h) <= COARSE_SIDE:
        return order(dedupe(fine))
    large = max(min_area_px, COARSE_MIN_SHARE * grid[0] * grid[1])
    coarse = _detect_pass(image_path, COARSE_SIDE, min_score, floor, grid, client, large)
    return order(dedupe(fuse_passes([fine, coarse])))


def _detect_pass(image_path: Path, side: int, min_score: float, floor: float,
                 grid: tuple[int, int], client: InferenceClient, min_area_px: float
                 ) -> list[Detection]:
    """One pass: the image at ``side`` px shown to the detector, masks resampled onto ``grid``
    (width, height)."""
    res = client.segment_image(
        p.SegmentRequest(
            image_path=str(Path(image_path).resolve()),
            labels=list(default_vocabulary()),
            max_side=side,
            imgsz=DETECT_SIDE,
            conf=floor - 1e-6,  # the model keeps scores strictly above its threshold
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
        raw = rle.decode(inst.mask)
        mask = resample_mask(raw, grid)
        if mask.sum() < min_area_px:
            continue
        b = inst.box_xyxy
        sx, sy = grid[0] / raw.shape[1], grid[1] / raw.shape[0]
        dets.append(Detection(label, score, inst.source, mask,
                              (float(b[0]) * sx, float(b[1]) * sy, float(b[2]) * sx,
                               float(b[3]) * sy)))
    return dets


def resample_mask(mask: NDArray[np.bool_], grid: tuple[int, int]) -> NDArray[np.bool_]:
    """``mask`` on a ``grid`` (width, height) of the same image: nearest pixel centre."""
    h, w = mask.shape
    if (w, h) == grid:
        return mask
    rows = np.minimum(((np.arange(grid[1]) + 0.5) * h / grid[1]).astype(np.intp), h - 1)
    cols = np.minimum(((np.arange(grid[0]) + 0.5) * w / grid[0]).astype(np.intp), w - 1)
    return mask[np.ix_(rows, cols)]
