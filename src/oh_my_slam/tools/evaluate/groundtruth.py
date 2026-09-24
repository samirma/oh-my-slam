"""Optional ground truth under ``examples/ground_truth/`` (format: that folder's README.md).

Every ``*.json`` file below the folder is discovered; its ``kind`` selects the metrics:

* ``objects`` — the objects of one example image (labels, optional camera-frame cuboids), compared
  with that image's ``segment.sh -i`` output: ``gt.objects.recall`` / ``.precision`` (and
  ``.obb_iou_median`` when cuboids are annotated).
* ``poses`` — per-capture yaw / pitch of ``ainex-captures``, compared with the one-update map:
  ``gt.poses.yaw_err_median_deg`` / ``.yaw_err_max_deg`` (after the best common yaw offset) and
  ``gt.poses.pitch_err_median_deg``.

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
from oh_my_slam.tools.evaluate.scene import DocObject, pitch_deg, yaw_deg
from oh_my_slam.tools.evaluate.segmentation import paired_labels

KINDS = ("objects", "poses")


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
    elif not isinstance(data.get("frames"), dict) or not all(
            isinstance(v, dict) for v in data["frames"].values()):
        raise ValueError("a 'poses' file needs a 'frames' object of capture name → values")


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
            skipped.append({"file": str(f.path), "reason": f"{image} was not evaluated"})
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


def pose_metrics(m: Metrics, files: list[GroundTruth], poses: dict[str, Pose] | None) -> None:
    """Per-capture annotated yaw (any zero, left positive) and pitch (up positive, from the
    horizon) against the one-update map's poses."""
    frames: dict[str, dict[str, Any]] = {}
    for f in files:
        frames.update(f.data["frames"])
    if not frames:
        return
    ids = ("gt.poses.yaw_err_median_deg", "gt.poses.yaw_err_max_deg",
           "gt.poses.pitch_err_median_deg")
    if poses is None:
        m.fail(ids, "the one-update map was not built")
        return
    unregistered = "no annotated capture is registered in the map"
    yaw_truth = {n: float(v["yaw_deg"]) for n, v in frames.items() if v.get("yaw_deg") is not None}
    yaw = [(yaw_deg(poses[n].R), g) for n, g in yaw_truth.items() if n in poses]
    if yaw:
        d = np.radians([e - g for e, g in yaw])
        offset = float(np.degrees(np.angle(np.mean(np.exp(1j * d)))))
        errs = [abs(wrap_deg(e - g - offset)) for e, g in yaw]
        m.add(ids[0], float(np.median(errs)), {"frames": len(errs), "offset_deg": round(offset, 2)})
        m.add(ids[1], float(np.max(errs)))
    elif yaw_truth:
        m.fail(ids[:2], unregistered)
    pitch_truth = {n: float(v["pitch_deg"]) for n, v in frames.items()
                   if v.get("pitch_deg") is not None}
    pitch = [abs(pitch_deg(poses[n].R) - g) for n, g in pitch_truth.items() if n in poses]
    if pitch:
        m.add(ids[2], float(np.median(pitch)), {"frames": len(pitch)})
    elif pitch_truth:
        m.fail(ids[2:], unregistered)
