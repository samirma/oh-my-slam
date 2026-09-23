"""Segmentation API: detections → exclusive masks → 3D points → upright OBBs → ids and colours.

``segment_frame`` serves ``segment.sh -i``, ``reconstruct.sh`` (JSON) and ``view.sh -i``;
``lift_detections`` serves the mapper (per keyframe, in map coordinates); ``export_map`` turns a
map's persistent objects into the segment artefacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.client.client import InferenceClient
from oh_my_slam.core import timing
from oh_my_slam.core.geometry import depth_edge_mask
from oh_my_slam.core.ply import PointCloud
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.reconstruction.api import (
    KEYFRAME_TOKENS,
    SINGLE_IMAGE_TOKENS,
    FrameReconstruction,
    reconstruct_image,
)
from oh_my_slam.reconstruction.gravity import DEFAULT_UP_CAM
from oh_my_slam.reconstruction.pointcloud import MAX_GRID_SIDE
from oh_my_slam.segmentation.colors import UNSEGMENTED, color_for_id, color_hex_for_id
from oh_my_slam.segmentation.detect import DEFAULT_MIN_SCORE, Detection, detect, floor_gap
from oh_my_slam.segmentation.lift import MIN_POINTS, Lifted, lift_mask
from oh_my_slam.segmentation.obb import OBB, fit_obb

KEYFRAME_GRID_SIDE = 768  # depth/segmentation grid of map keyframes

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


@dataclass
class FrameSegmentation:
    frame: FrameReconstruction
    objects: list[SceneObject]
    label_map: NDArray[np.int32]  # object id per grid pixel (0 = none), exclusive masks
    point_pixels: dict[int, NDArray[np.int64]]  # lifted pixel indices per object id
    points: dict[int, NDArray[np.float64]]  # lifted points per object id (camera frame)

    def segments_cloud(self) -> PointCloud:
        """Every frame point, grey unless it is one of an object's lifted points."""
        cloud, idx = self.frame.camera_cloud()
        owner = np.zeros(self.label_map.size, np.int32)
        for oid, pix in self.point_pixels.items():
            owner[pix] = oid
        labels = owner[idx]
        rgb = np.tile(np.array(UNSEGMENTED, np.uint8), (len(idx), 1))
        for obj in self.objects:
            rgb[labels == obj.id] = obj.color
        return PointCloud(cloud.xyz, rgb, labels)


def exclusive_masks(dets: list[Detection], shape: tuple[int, int]) -> list[NDArray[np.bool_]]:
    """Resolve overlaps: larger masks are painted first, smaller ones on top."""
    owner = np.full(shape, -1, np.int32)
    for i in sorted(range(len(dets)), key=lambda k: -dets[k].area):
        owner[dets[i].mask] = i
    return [owner == i for i in range(len(dets))]


def lift_detections(
    frame: FrameReconstruction,
    dets: list[Detection],
    T_parent_cam: Pose | None = None,
    depth: NDArray[Any] | None = None,
    valid: NDArray[Any] | None = None,
) -> list[LiftedInstance]:
    """Exclusive masks lifted to 3D (``depth`` overrides the frame's, e.g. scale-aligned)."""
    with timing.part("lift"):
        return _lift_detections(frame, dets, T_parent_cam, depth, valid)


def _lift_detections(frame: FrameReconstruction, dets: list[Detection], T_parent_cam: Pose | None,
                     depth: NDArray[Any] | None, valid: NDArray[Any] | None
                     ) -> list[LiftedInstance]:
    d = frame.depth if depth is None else depth
    v = frame.valid if valid is None else valid
    edges = depth_edge_mask(np.where(v & (d > 0), d, 0.0))
    masks = exclusive_masks(dets, d.shape)
    out = []
    for det, m in zip(dets, masks, strict=True):
        if m.sum() == 0:
            continue
        lifted = lift_mask(m, d, frame.K_grid, v, T_parent_cam, edges=edges)
        if len(lifted.points) >= MIN_POINTS:
            out.append(LiftedInstance(det, m, lifted))
    return out


def segment_frame(
    frame: FrameReconstruction,
    *,
    labels: list[str] | None = None,
    min_score: float = DEFAULT_MIN_SCORE,
    client: InferenceClient | None = None,
    detections: list[Detection] | None = None,
) -> FrameSegmentation:
    """Objects of one image in its camera frame; ids 1..N by score, then area."""
    if detections is None:
        detections = detect(frame.image_path, labels=labels, min_score=min_score,
                            max_side=max(frame.grid_size), client=client)
    instances = lift_detections(frame, detections)
    up = frame.gravity.up_cam if frame.gravity is not None else DEFAULT_UP_CAM
    floor = None
    if frame.gravity is not None and frame.gravity.floor_height is not None:
        floor = -float(frame.gravity.floor_height)  # floor points satisfy up·x = -height
    label_map = np.zeros(frame.depth.shape, np.int32)
    objects: list[SceneObject] = []
    point_pixels: dict[int, NDArray[np.int64]] = {}
    points: dict[int, NDArray[np.float64]] = {}
    for oid, inst in enumerate(instances, start=1):
        box = fit_obb(inst.lifted.points, up, floor, floor_gap(inst.detection.label))
        label_map[inst.mask] = oid
        objects.append(SceneObject(
            id=oid, label=inst.detection.label, score=inst.detection.score, obb=box,
            pixel_count=int(inst.mask.sum()), point_count=len(inst.lifted.points), frames=[0],
        ))
        point_pixels[oid] = inst.lifted.pixels
        points[oid] = inst.lifted.points
    return FrameSegmentation(frame, objects, label_map, point_pixels, points)


def reconstruct_and_detect(
    image_path: Path,
    client: InferenceClient,
    *,
    labels: list[str] | None = None,
    min_score: float = DEFAULT_MIN_SCORE,
    keyframe: bool = False,
    intrinsics: Intrinsics | None = None,
    work_dir: Path | None = None,
) -> tuple[FrameReconstruction, list[Detection]]:
    """Reconstruction and detection with the detection request issued concurrently, so the
    server's queue stays busy while this process decodes and post-processes.

    ``keyframe=True`` uses the mapper's settings (smaller grid, fewer tokens, descriptor)."""
    from concurrent.futures import ThreadPoolExecutor

    side = KEYFRAME_GRID_SIDE if keyframe else MAX_GRID_SIDE
    det_client = client.clone()
    try:
        with ThreadPoolExecutor(1) as pool:
            fut = pool.submit(detect, image_path, labels=labels, min_score=min_score,
                              max_side=side, client=det_client)
            frame = reconstruct_image(
                image_path, want_gravity=True, client=client, intrinsics=intrinsics,
                max_side=side, num_tokens=KEYFRAME_TOKENS if keyframe else SINGLE_IMAGE_TOKENS,
                want_descriptor=keyframe, work_dir=work_dir)
            dets = fut.result()
    finally:
        if det_client is not client:
            det_client.close()
    return frame, dets


@dataclass
class KeyframeLabels:
    """A keyframe image (grid) with its stored per-pixel persistent object ids."""

    name: str
    rgb: NDArray[np.uint8]
    label_map: NDArray[np.int32]


@dataclass
class MapSegmentation:
    segmented: NDArray[np.uint8]  # contact sheet of <= 6 keyframes
    segments: PointCloud  # map cloud coloured by object (grey elsewhere), label = object id
    tiles: list[str]  # keyframe names on the sheet


def export_map(objects: list[SceneObject], keyframes: list[KeyframeLabels],
               cloud: PointCloud) -> MapSegmentation:
    """Segment artefacts for a persisted map (objects keep their ids and colours)."""
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
    return MapSegmentation(contact_sheet(tiles), colorize_cloud(cloud, ids),
                           [keyframes[i].name for i in chosen])


def colorize_cloud(cloud: PointCloud, valid_ids: set[int]) -> PointCloud:
    """Segment colours for a labelled cloud (label = object id, 0 = none)."""
    labels = cloud.label if cloud.label is not None else np.zeros(len(cloud), np.int32)
    labels = np.where(np.isin(labels, list(valid_ids)), labels, 0).astype(np.int32)
    rgb = np.tile(np.array(UNSEGMENTED, np.uint8), (len(cloud), 1))
    for oid in np.unique(labels):
        if oid > 0:
            rgb[labels == oid] = color_for_id(int(oid))
    return PointCloud(cloud.xyz, rgb, labels)
