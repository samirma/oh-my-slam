"""Optional ground truth under ``examples/ground_truth/`` (format: that folder's README.md).

Every ``*.json`` file below the folder is discovered; its ``kind`` selects the metrics:

* ``objects`` — the objects of one example image (labels, optional camera-frame cuboids), compared
  with that image's ``segment.sh -i`` output (any example image: the evaluator segments the
  annotated images it does not segment anyway, ``annotated_images``): ``gt.objects.recall`` /
  ``.precision`` (and ``.obb_iou_median`` when cuboids are annotated).
* ``poses`` — per-image yaw / pitch of the example sequences (``ainex-captures``, ``camera``,
  ``office_sequence``), compared with the map of the image's sequence built in one update:
  ``gt.poses.yaw_err_median_deg`` / ``.yaw_err_max_deg`` (after the best common yaw offset of each
  map) and ``gt.poses.pitch_err_median_deg``.
* ``map_update`` — what changed during ``examples/office_sequence/`` (objects that must be absent
  from the final map, where they were seen) and, optionally, which objects never changed; judged by
  ``mapupdate`` into the ``map_update.*`` metrics (``MapUpdatePlan`` below).

Malformed files and files about images that were not evaluated are skipped with a reason.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from oh_my_slam.core.types import Pose
from oh_my_slam.segmentation.detect import compatible
from oh_my_slam.tools.evaluate.mapquality import match_objects
from oh_my_slam.tools.evaluate.metrics import Metrics
from oh_my_slam.tools.evaluate.names import wrap_deg
from oh_my_slam.tools.evaluate.poses import circular_mean_deg
from oh_my_slam.tools.evaluate.scene import DocObject, pitch_deg, yaw_deg
from oh_my_slam.tools.evaluate.segmentation import paired_labels

KINDS = ("objects", "poses", "map_update")
MAP_UPDATE_SEQUENCE = "office_sequence"  # the example folder (under examples/) the kind is about


@dataclass(frozen=True)
class GroundTruth:
    path: Path
    kind: str
    data: dict[str, Any]


def discover(folder: Path) -> tuple[list[GroundTruth], list[dict[str, str]]]:
    """(usable files, skipped files with the reason)."""
    found, skipped = [], []
    if not Path(folder).is_dir():
        return [], []
    for path in sorted(Path(folder).rglob("*.json")):
        try:
            data = json.loads(path.read_text())
            kind = data.get("kind") if isinstance(data, dict) else None
            if kind not in KINDS:
                raise ValueError(f"'kind' must be one of {KINDS}")
            _validate(kind, data)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            skipped.append({"file": str(path), "reason": str(exc)})
            continue
        found.append(GroundTruth(path, kind, data))
    return found, skipped


def _validate(kind: str, data: dict[str, Any]) -> None:
    if kind == "objects":
        if not isinstance(data.get("image"), str) or not isinstance(data.get("objects"), list):
            raise ValueError("an 'objects' file needs 'image' and an 'objects' list")
        for o in data["objects"]:
            cub = o.get("cuboid")
            if not isinstance(o.get("label"), str) or (cub is not None and len(cub) != 10):
                raise ValueError("each object needs a 'label' and an optional 10-value 'cuboid'")
    elif kind == "map_update":
        _map_update_parts(data)
    elif not isinstance(data.get("frames"), dict) or not all(
            isinstance(v, dict) for v in data["frames"].values()):
        raise ValueError("a 'poses' file needs a 'frames' object of capture name → values")


@dataclass(frozen=True)
class Absent:
    """An object that was in the scene early in the sequence and is gone from it later: its label
    and, per image that shows it, its region (x0, y0, x1, y1 as shares of the image size)."""

    label: str
    seen_in: dict[str, tuple[float, float, float, float]]


@dataclass(frozen=True)
class MapUpdatePlan:
    """The ``map_update`` files of one run, merged."""

    sequence: str
    absent: list[Absent]
    stable: list[str]  # labels the stability comparison is restricted to (empty: every object)
    files: list[Path]
    splits: tuple[tuple[int, ...], ...] = ()  # sizes of the consecutive updates of a split map

    def split_sizes(self, images: list[str]) -> list[tuple[int, ...]]:
        """How the sequence is split across updates: the annotated ``splits``, else one split,
        the early part (``before_images``) then the rest."""
        if self.splits:
            return list(self.splits)
        early = len(self.before_images(images))
        return [(early, len(images) - early)] if 0 < early < len(images) else []

    def before_images(self, images: list[str]) -> list[str]:
        """The images up to the last one that shows an absent object (``images``: in capture
        order): the sequence's early part, where the scene still had it."""
        seen = [images.index(n) for a in self.absent for n in a.seen_in if n in images]
        return images[:max(seen) + 1] if seen else []


def _map_update_parts(data: dict[str, Any]
                      ) -> tuple[str, list[Absent], list[str], list[tuple[int, ...]]]:
    seq, absent = data.get("sequence"), data.get("absent")
    if not isinstance(seq, str) or not isinstance(absent, list) or not absent:
        raise ValueError("a 'map_update' file needs 'sequence' and a non-empty 'absent' list")
    items = []
    for a in absent:
        seen = a.get("seen_in") if isinstance(a, dict) else None
        if not isinstance(seen, dict) or not seen or not isinstance(a.get("label"), str):
            raise ValueError("each absent object needs a 'label' and 'seen_in' (image name → "
                             "region)")
        regions = {}
        for name, box in seen.items():
            ok = isinstance(box, list) and len(box) == 4 and all(
                isinstance(v, int | float) and 0.0 <= v <= 1.0 for v in box)
            if not ok or not (box[0] < box[2] and box[1] < box[3]):
                raise ValueError(f"'seen_in' {name}: a region is [x0, y0, x1, y1] within 0..1 "
                                 "with x0 < x1 and y0 < y1")
            regions[str(name)] = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        items.append(Absent(a["label"], regions))
    stable = data.get("stable", [])
    if not isinstance(stable, list) or not all(isinstance(v, str) for v in stable):
        raise ValueError("'stable' is a list of labels")
    splits = data.get("splits", [])
    if not isinstance(splits, list) or not all(
            isinstance(sp, list) and len(sp) >= 2 and all(isinstance(n, int) and n > 0 for n in sp)
            for sp in splits):
        raise ValueError("'splits' is a list of splits, each the sizes (at least two positive "
                         "integers) of consecutive updates")
    return seq, items, list(stable), [tuple(sp) for sp in splits]


def map_update_plan(files: list[GroundTruth], skipped: list[dict[str, str]]
                    ) -> MapUpdatePlan | None:
    """The plan of the ``map_update`` files about ``MAP_UPDATE_SEQUENCE`` (their absent objects and
    stable labels are merged); files about another sequence are skipped with a reason."""
    absent: list[Absent] = []
    stable: list[str] = []
    splits: list[tuple[int, ...]] = []
    used: list[Path] = []
    for f in files:
        if f.kind != "map_update":
            continue
        seq, items, labels, sizes = _map_update_parts(f.data)
        if seq != MAP_UPDATE_SEQUENCE:
            skipped.append({"file": str(f.path), "reason": f"{seq} is not evaluated "
                                                           f"(only {MAP_UPDATE_SEQUENCE})"})
            continue
        absent += items
        stable += [v for v in labels if v not in stable]
        splits += [sp for sp in sizes if sp not in splits]
        used.append(f.path)
    return MapUpdatePlan(MAP_UPDATE_SEQUENCE, absent, stable, used, tuple(splits)) \
        if used else None


def annotated_images(files: list[GroundTruth]) -> list[str]:
    """The example images (paths relative to ``examples/``) the ``objects`` files describe."""
    return list(dict.fromkeys(f.data["image"] for f in files if f.kind == "objects"))


def _gt_objects(data: dict[str, Any]) -> list[DocObject]:
    return [DocObject(i + 1, o["label"], None, None, None,
                      None if o.get("cuboid") is None else tuple(float(v) for v in o["cuboid"]))
            for i, o in enumerate(data["objects"])]


def object_metrics(m: Metrics, files: list[GroundTruth], evaluated: dict[str, list[DocObject]],
                   skipped: list[dict[str, str]]) -> None:
    """``evaluated``: example-relative image path → its ``segment.sh -i`` objects."""
    truth = gt_n = det_n = 0
    ious: list[float] = []
    per_image = {}
    for f in files:
        image = f.data["image"]
        if image not in evaluated:
            skipped.append({"file": str(f.path), "reason": f"{image} was not evaluated (not an "
                                                           "example image, or segment.sh -i "
                                                           "failed on it)"})
            continue
        gt, det = _gt_objects(f.data), evaluated[image]
        boxes_gt = [(o, b) for o in gt if (b := o.obb()) is not None]
        boxes_det = [(o, b) for o in det if (b := o.obb()) is not None]
        if gt and len(boxes_gt) == len(gt):
            pairs = [p for p in match_objects(boxes_gt, boxes_det)
                     if compatible(boxes_gt[p[0]][0].label, boxes_det[p[1]][0].label)]
            ious += [p[2] for p in pairs]
            k = len(pairs)
        else:
            k = paired_labels([o.label for o in gt], [o.label for o in det])
        per_image[image] = {"truth": len(gt), "detections": len(det), "paired": k}
        truth, gt_n, det_n = truth + k, gt_n + len(gt), det_n + len(det)
    if not per_image:
        return
    m.add("gt.objects.recall", truth / gt_n if gt_n else None, per_image,
          error=None if gt_n else "no annotated objects")
    m.add("gt.objects.precision", truth / det_n if det_n else None, per_image,
          error=None if det_n else "no detections")
    if ious:
        m.add("gt.objects.obb_iou_median", float(np.median(ious)))


def pose_metrics(m: Metrics, files: list[GroundTruth],
                 maps: dict[str, dict[str, Pose] | None]) -> None:
    """Per-image annotated yaw (any zero per sequence, left positive) and pitch (up positive,
    from the horizon) against the poses of the one-update map that registered the image.
    ``maps``: each example sequence's one-update map, its poses by image name (None: not built);
    the yaw offset is fitted per map, since each has a frame of its own."""
    frames: dict[str, dict[str, Any]] = {}
    for f in files:
        frames.update(f.data["frames"])
    if not frames:
        return
    ids = ("gt.poses.yaw_err_median_deg", "gt.poses.yaw_err_max_deg",
           "gt.poses.pitch_err_median_deg")
    built = {k: p for k, p in maps.items() if p is not None}
    if not built:
        m.fail(ids, "no one-update map was built")
        return
    unbuilt = sorted(maps.keys() - built.keys())
    unregistered = "no annotated capture is registered in a one-update map" + (
        f" (not built: {', '.join(unbuilt)})" if unbuilt else "")
    yaw_truth = {n: float(v["yaw_deg"]) for n, v in frames.items() if v.get("yaw_deg") is not None}
    errs: list[float] = []
    offsets: dict[str, float] = {}
    for seq, poses in built.items():
        yaw = [(yaw_deg(poses[n].R), g) for n, g in yaw_truth.items() if n in poses]
        if yaw:
            offset = circular_mean_deg([e - g for e, g in yaw])
            errs += [abs(wrap_deg(e - g - offset)) for e, g in yaw]
            offsets[seq] = round(offset, 2)
    if errs:
        m.add(ids[0], float(np.median(errs)), {"frames": len(errs), "offset_deg": offsets})
        m.add(ids[1], float(np.max(errs)))
    elif yaw_truth:
        m.fail(ids[:2], unregistered)
    pitch_truth = {n: float(v["pitch_deg"]) for n, v in frames.items()
                   if v.get("pitch_deg") is not None}
    pitch = [abs(pitch_deg(poses[n].R) - g) for n, g in pitch_truth.items()
             for poses in built.values() if n in poses]
    if pitch:
        m.add(ids[2], float(np.median(pitch)), {"frames": len(pitch)})
    elif pitch_truth:
        m.fail(ids[2:], unregistered)
