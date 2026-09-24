"""The emitted point cloud (spec §2.2): one derivation for every PLY writer and the viewer.

A source holds data that is already computed — one image's depth grid (reconstruction) with the
per-pixel object ids of its segmentation, or a map's points with their object ids — and
``derive_cloud`` applies the attributes to it without any inference: pixel selection (``stride``,
depth range, ``edge``; images only), unprojection, ``voxel`` thinning (one representative point per
voxel, the first in pixel/storage order, colours never averaged), then colour, normals and label.
Object colours are the colour contract of ``segmentation.colors``; unsegmented points are grey.
The same source and attributes always give the same cloud, and attributes never change objects,
ids or colours.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.cloud_attrs import CloudAttrs, CloudScope
from oh_my_slam.core.geometry import voxel_downsample_indices
from oh_my_slam.core.ply import PointCloud, ply_bytes
from oh_my_slam.core.types import Intrinsics
from oh_my_slam.reconstruction.pointcloud import (
    depth_normals,
    pixel_mask,
    pixel_points,
    point_normals,
)
from oh_my_slam.segmentation.colors import height_colors, segment_colors

if TYPE_CHECKING:
    from oh_my_slam.reconstruction.api import FrameReconstruction
    from oh_my_slam.segmentation.api import FrameSegmentation

IMAGE_FRAME = "oh-my-slam camera frame (OpenCV axes: x right, y down, z forward), metres"
MAP_FRAME = "oh-my-slam map frame (z up), metres"


@dataclass(frozen=True, eq=False)
class ImageCloudSource:
    """One image on its depth grid; points are in the camera frame (OpenCV axes, metres)."""

    depth: NDArray[np.float32]  # (H, W) metres, 0 = invalid
    valid: NDArray[np.bool_]  # (H, W) model validity
    rgb: NDArray[np.uint8]  # (H, W, 3) the image resized to the grid
    K: Intrinsics  # intrinsics of the grid
    labels: NDArray[np.int32] | None = None  # (H, W) object id per pixel (0 = none), if segmented
    up: NDArray[np.float64] | None = None  # unit up direction (camera frame); None = not estimated

    @cached_property
    def normals(self) -> NDArray[np.float32]:
        """(H, W, 3) normals from the full-resolution depth grid (computed once)."""
        return depth_normals(self.depth, self.K, self.valid)


@dataclass(frozen=True, eq=False)
class MapCloudSource:
    """A map's points in map coordinates (metres, z up)."""

    xyz: NDArray[Any]  # (N, 3)
    rgb: NDArray[np.uint8]  # (N, 3)
    labels: NDArray[np.int32] | None  # (N,) object id per point, 0 = none
    viewpoints: NDArray[np.float64]  # (M, 3) camera centres, to orient normals

    @cached_property
    def normals(self) -> NDArray[np.float32]:
        """(N, 3) normals of the whole cloud (computed once)."""
        return point_normals(self.xyz, self.viewpoints)


CloudSource = ImageCloudSource | MapCloudSource


def scope_of(source: CloudSource) -> CloudScope:
    return CloudScope.IMAGE if isinstance(source, ImageCloudSource) else CloudScope.MAP


def image_cloud_source(frame: FrameReconstruction, seg: FrameSegmentation | None = None
                       ) -> ImageCloudSource:
    """Source of one reconstructed image; with ``seg`` its pixels carry the id of the object whose
    points they became (``FrameSegmentation.point_labels``)."""
    return ImageCloudSource(
        depth=frame.depth, valid=frame.valid, rgb=frame.rgb, K=frame.K_grid,
        labels=None if seg is None else seg.point_labels(),
        up=None if frame.gravity is None else np.asarray(frame.gravity.up_cam, np.float64),
    )


def map_cloud_source(xyz: NDArray[Any], rgb: NDArray[np.uint8], labels: NDArray[Any] | None,
                     object_ids: set[int], viewpoints: NDArray[Any]) -> MapCloudSource:
    """Source of a map cloud; point labels of objects not in ``object_ids`` become 0."""
    lab = None
    if labels is not None:
        lab = np.asarray(labels, np.int32).reshape(-1)
        lab = np.where(np.isin(lab, sorted(object_ids)), lab, 0).astype(np.int32)
    return MapCloudSource(np.asarray(xyz), np.asarray(rgb, np.uint8), lab,
                          np.asarray(viewpoints, np.float64).reshape(-1, 3))


def derive_cloud(source: CloudSource, attrs: CloudAttrs) -> PointCloud:
    """The cloud ``attrs`` describe, from data already computed (no inference)."""
    if (attrs.color == "segment" or attrs.label) and source.labels is None:
        raise ValueError("color=segment and label=on need a segmented source")
    # xyz[i] is the point of source row rows[i] (a flat pixel index, or a map point index)
    if isinstance(source, ImageCloudSource):
        if attrs.color == "height" and source.up is None:
            raise ValueError("color=height needs the image's estimated gravity")
        mask = pixel_mask(source.depth, source.valid, attrs)
        xyz, rows = pixel_points(source.depth, source.K, mask)
        rgb, labels = source.rgb.reshape(-1, 3), source.labels
        normals = source.normals.reshape(-1, 3) if attrs.normals else None
    else:
        xyz = np.asarray(source.xyz, np.float64).reshape(-1, 3)
        rows = np.arange(len(xyz))
        rgb, labels = source.rgb, source.labels
        normals = source.normals if attrs.normals else None
    if attrs.voxel > 0:
        keep = voxel_downsample_indices(xyz, attrs.voxel, keep="first")
        xyz, rows = xyz[keep], rows[keep]
    lab = None if labels is None else labels.reshape(-1)[rows]
    colour: NDArray[np.uint8] | None
    if attrs.color == "rgb":
        colour = rgb[rows]
    elif attrs.color == "segment":
        assert lab is not None
        colour = segment_colors(lab)
    elif attrs.color == "height":
        up = source.up if isinstance(source, ImageCloudSource) else np.array([0.0, 0.0, 1.0])
        colour = height_colors(xyz @ np.asarray(up, np.float64))
    else:
        colour = None
    return PointCloud(xyz.astype(np.float32), colour, lab if attrs.label else None,
                      None if normals is None else normals[rows])


def cloud_ply(source: CloudSource, attrs: CloudAttrs) -> bytes:
    """``derive_cloud`` as a PLY whose header names the frame and records the effective
    attributes (defaults included; for a map only those that apply)."""
    scope = scope_of(source)
    frame = IMAGE_FRAME if scope == CloudScope.IMAGE else MAP_FRAME
    return ply_bytes(derive_cloud(source, attrs), encoding=attrs.encoding,
                     comments=[frame, f"attributes {attrs.describe(scope)}"])
