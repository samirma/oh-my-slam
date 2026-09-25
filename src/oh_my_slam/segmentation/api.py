"""Segmentation API: detections → exclusive masks → 3D points → upright OBBs → ids and colours.

``segment_frame`` serves ``segment.sh -i``, ``reconstruct.sh`` (JSON) and ``view.sh -i``;
``detect_alongside`` and ``lift_detections`` serve the mapper (detections of a keyframe while the
mapper's own reconstruction call runs; instances in map coordinates), ``fit_object_obb`` fits the
boxes of both; ``export_map`` draws a map's persistent objects on its keyframes. Emitted clouds are
derived in ``segmentation.cloud``. Other packages use segmentation through this module (and
``cloud``, ``scene``, ``artifacts``), never its internals (import-linter contract).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.client.client import InferenceClient
from oh_my_slam.core import timing
from oh_my_slam.core.geometry import depth_edge_mask
from oh_my_slam.core.types import Pose
from oh_my_slam.reconstruction.api import (
    SINGLE_IMAGE_TOKENS,
    FrameReconstruction,
    reconstruct_image,
)
from oh_my_slam.reconstruction.gravity import DEFAULT_UP_CAM
from oh_my_slam.reconstruction.pointcloud import MAX_GRID_SIDE
from oh_my_slam.segmentation.colors import UNSEGMENTED as UNSEGMENTED
from oh_my_slam.segmentation.colors import color_for_id, color_hex_for_id
from oh_my_slam.segmentation.detect import (
    DEFAULT_MIN_SCORE,
    DETECTION_FLOOR,
    claim_order,
    detect,
    grounding,
    priority,
)
from oh_my_slam.segmentation.detect import Detection as Detection
from oh_my_slam.segmentation.detect import compatible as compatible
from oh_my_slam.segmentation.detect import split_surface as split_surface
from oh_my_slam.segmentation.detect import surface_label as surface_label
from oh_my_slam.segmentation.lift import MIN_POINTS, Lifted, lift_mask
from oh_my_slam.segmentation.obb import OBB as OBB
from oh_my_slam.segmentation.obb import fit_obb
from oh_my_slam.segmentation.obb import obb_iou_upright as obb_iou_upright


@dataclass
class SceneObject:
    id: int
    label: str
    score: float
    obb: OBB
    pixel_count: int
    point_count: int
    observations: int = 1
    confirmed: bool = True
    frames: list[int] = field(default_factory=list)
    # every label the object was detected as, most evidence first (a map object whose label
    # flickered between keyframes); empty or one label: just ``label``
    labels: tuple[str, ...] = ()

    @property
    def color(self) -> tuple[int, int, int]:
        return color_for_id(self.id)

    @property
    def color_hex(self) -> str:
        return color_hex_for_id(self.id)


@dataclass
class LiftedInstance:
    detection: Detection
    mask: NDArray[np.bool_]  # exclusive mask on the grid
    lifted: Lifted


def join_instances(pieces: list[LiftedInstance]) -> LiftedInstance:
    """One instance from pieces of one object in the same image (e.g. a surface the detector split
    around the items on it, ``split_surface``): the union of their exclusive masks and lifted
    points, under the label, score and source of the piece that comes first in ``priority``."""
    if len(pieces) == 1:
        return pieces[0]
    best = min(pieces, key=lambda p: priority(p.detection))
    mask = np.logical_or.reduce([p.mask for p in pieces])
    boxes = np.array([p.detection.box for p in pieces])
    box = (float(boxes[:, 0].min()), float(boxes[:, 1].min()), float(boxes[:, 2].max()),
           float(boxes[:, 3].max()))
    det = Detection(best.detection.label, best.detection.score, best.detection.source, mask, box)
    lifted = Lifted(np.concatenate([p.lifted.points for p in pieces]),
                    np.concatenate([p.lifted.pixels for p in pieces]),
                    sum(p.lifted.mask_pixels for p in pieces))
    return LiftedInstance(det, mask, lifted)


@dataclass
class FrameSegmentation:
    frame: FrameReconstruction
    objects: list[SceneObject]
    label_map: NDArray[np.int32]  # object id per grid pixel (0 = none), exclusive masks
    point_pixels: dict[int, NDArray[np.int64]]  # lifted pixel indices per object id
    points: dict[int, NDArray[np.float64]]  # lifted points per object id (camera frame)

    def point_labels(self) -> NDArray[np.int32]:
        """Object id per grid pixel for the pixels lifted into an object's points, 0 elsewhere
        (mask pixels dropped by lifting stay unsegmented)."""
        out = np.zeros(self.label_map.shape, np.int32)
        flat = out.reshape(-1)
        for oid, pix in self.point_pixels.items():
            flat[pix] = oid
        return out


def pixel_owners(dets: list[Detection], shape: tuple[int, int]) -> NDArray[np.int32]:
    """Index of the detection owning each pixel (-1: none), claimed in ``claim_order``: among the
    trusted detections (the default threshold and above) the smallest mask covering a pixel wins,
    so nested objects keep their pixels; lower-scoring detections only get the pixels left over.
    The result depends on the detections alone, never on the ``--min-score`` applied later."""
    owner = np.full(shape, -1, np.int32)
    for i in sorted(range(len(dets)), key=lambda k: claim_order(dets[k])):
        mask = dets[i].mask
        rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
        if len(rows) == 0:
            continue
        box = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))  # a view of owner
        sub = owner[box]
        sub[mask[box] & (sub < 0)] = i
    return owner


def fit_object_obb(points: NDArray[Any], label: str, up: NDArray[Any],
                   floor_level: float | None = None) -> OBB:
    """Upright OBB of an object's points; floor-standing classes are grounded on the floor at
    ``floor_level`` (the floor's coordinate along ``up``) when their visible bottom floats just
    above it and the visible part is not a sliver of the grounded box (``detect.grounding``)."""
    return fit_obb(points, up, floor_level, *grounding(label))


def lift_detections(
    frame: FrameReconstruction,
    dets: list[Detection],
    T_parent_cam: Pose | None = None,
    depth: NDArray[Any] | None = None,
    valid: NDArray[Any] | None = None,
    *,
    min_score: float = 0.0,
) -> list[LiftedInstance]:
    """Exclusive masks lifted to 3D (``depth`` overrides the frame's, e.g. scale-aligned).

    Overlaps are resolved among all of ``dets``; only the detections scoring at least
    ``min_score`` are lifted, so the result for one of them does not depend on ``min_score``."""
    with timing.part("lift"):
        return _lift_detections(frame, dets, T_parent_cam, depth, valid, min_score)


def _lift_detections(frame: FrameReconstruction, dets: list[Detection], T_parent_cam: Pose | None,
                     depth: NDArray[Any] | None, valid: NDArray[Any] | None, min_score: float
                     ) -> list[LiftedInstance]:
    d = frame.depth if depth is None else depth
    v = frame.valid if valid is None else valid
    edges = depth_edge_mask(np.where(v & (d > 0), d, 0.0))
    owner = pixel_owners(dets, d.shape)
    out = []
    for i, det in enumerate(dets):
        if det.score < min_score:
            continue
        m = owner == i
        if not m.any():
            continue
        lifted = lift_mask(m, d, frame.K_grid, v, T_parent_cam, edges=edges)
        if len(lifted.points) >= MIN_POINTS:
            out.append(LiftedInstance(det, m, lifted))
    return out


def segment_frame(
    frame: FrameReconstruction,
    *,
    min_score: float = DEFAULT_MIN_SCORE,
    client: InferenceClient | None = None,
    detections: list[Detection] | None = None,
) -> FrameSegmentation:
    """Objects of one image in its camera frame.

    ``detections`` must be every detection down to ``DETECTION_FLOOR`` (as ``detect`` and
    ``reconstruct_and_detect`` return them), whatever ``min_score``: overlapping pixels are
    resolved once among all of them (``pixel_owners``), and only then are the objects below
    ``min_score`` dropped. An object's mask, points and OBB therefore never depend on
    ``min_score``, nor does whether it survives lifting.

    Ids are 1..N over the objects kept, in detection ``priority`` order (score desc, then area,
    label, box), so the same image and options give the same ids, and a higher ``min_score`` only
    drops objects from the end: the objects kept at both thresholds have the same id, colour,
    mask, points and OBB. The pixels of a detection below ``min_score`` that a larger object
    would otherwise cover stay unsegmented (they are not that object's surface)."""
    if not DETECTION_FLOOR <= min_score <= 1.0:
        raise ValueError(f"min_score {min_score} is outside [{DETECTION_FLOOR}, 1]")
    if detections is None:
        detections = detect(frame.image_path, min_score=DETECTION_FLOOR,
                            max_side=max(frame.grid_size), client=client)
    instances = sorted(lift_detections(frame, detections, min_score=min_score),
                       key=lambda i: priority(i.detection))
    up = frame.gravity.up_cam if frame.gravity is not None else DEFAULT_UP_CAM
    floor = None
    if frame.gravity is not None and frame.gravity.floor_height is not None:
        floor = -float(frame.gravity.floor_height)  # floor points satisfy up·x = -height
    label_map = np.zeros(frame.depth.shape, np.int32)
    objects: list[SceneObject] = []
    point_pixels: dict[int, NDArray[np.int64]] = {}
    points: dict[int, NDArray[np.float64]] = {}
    for oid, inst in enumerate(instances, start=1):
        box = fit_object_obb(inst.lifted.points, inst.detection.label, up, floor)
        label_map[inst.mask] = oid
        objects.append(SceneObject(
            id=oid, label=inst.detection.label, score=inst.detection.score, obb=box,
            pixel_count=int(inst.mask.sum()), point_count=len(inst.lifted.points), frames=[0],
        ))
        point_pixels[oid] = inst.lifted.pixels
        points[oid] = inst.lifted.points
    return FrameSegmentation(frame, objects, label_map, point_pixels, points)


def detect_alongside(
    image_path: Path,
    client: InferenceClient,
    reconstruct: Callable[[InferenceClient], FrameReconstruction],
    *,
    max_side: int,
    min_score: float = DEFAULT_MIN_SCORE,
    floor: float | None = None,
) -> tuple[FrameReconstruction, list[Detection]]:
    """Detections of ``image_path`` requested on a second connection while the caller's
    ``reconstruct(client)`` runs, so the server's queue stays busy while this process decodes and
    post-processes. ``max_side`` must be the reconstruction's grid; the server is asked for
    everything above ``floor`` (default: ``min_score``)."""
    from concurrent.futures import ThreadPoolExecutor

    det_client = client.clone()
    try:
        with ThreadPoolExecutor(1) as pool:
            fut = pool.submit(detect, image_path, min_score=min_score,
                              floor=min_score if floor is None else floor, max_side=max_side,
                              client=det_client)
            frame = reconstruct(client)
            dets = fut.result()
    finally:
        if det_client is not client:
            det_client.close()
    return frame, dets


def reconstruct_and_detect(
    image_path: Path,
    client: InferenceClient,
) -> tuple[FrameReconstruction, list[Detection]]:
    """Single-image reconstruction (delegated to ``reconstruction``) and detection, issued
    concurrently. Returns every detection down to ``DETECTION_FLOOR``: ``segment_frame`` applies
    ``--min-score`` after resolving overlaps, so the objects kept do not depend on it."""

    def reconstruct(c: InferenceClient) -> FrameReconstruction:
        return reconstruct_image(image_path, want_gravity=True, client=c, max_side=MAX_GRID_SIDE,
                                 num_tokens=SINGLE_IMAGE_TOKENS)

    return detect_alongside(image_path, client, reconstruct, max_side=MAX_GRID_SIDE,
                            min_score=DETECTION_FLOOR, floor=DETECTION_FLOOR)


@dataclass
class KeyframeLabels:
    """A keyframe image (grid) with its stored per-pixel persistent object ids."""

    name: str
    rgb: NDArray[np.uint8]
    label_map: NDArray[np.int32]


@dataclass
class MapSegmentation:
    segmented: NDArray[np.uint8]  # contact sheet of <= 6 keyframes
    tiles: list[str]  # keyframe names on the sheet


def export_map(objects: list[SceneObject], keyframes: list[KeyframeLabels]) -> MapSegmentation:
    """``segmented.png`` of a persisted map (objects keep their ids and colours); the segments
    cloud is derived by ``segmentation.cloud``."""
    from oh_my_slam.segmentation.render import choose_contact_frames, contact_sheet

    ids = {o.id for o in objects}
    id_list = sorted(ids)
    sets = [set(np.unique(k.label_map).tolist()) & ids for k in keyframes]
    chosen = choose_contact_frames(sets)
    if not chosen and keyframes:
        chosen = [0]
    tiles = [
        (keyframes[i].name, keyframes[i].rgb,
         np.where(np.isin(keyframes[i].label_map, id_list), keyframes[i].label_map, 0))
        for i in chosen
    ]
    return MapSegmentation(contact_sheet(tiles), [keyframes[i].name for i in chosen])
