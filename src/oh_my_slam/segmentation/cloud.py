"""The emitted point cloud (spec §2.2): one derivation for every PLY writer and the viewer.

A source holds data that is already computed — one image's depth grid (reconstruction) with the
per-pixel object ids of its segmentation, or a map's points with their object ids — and
``derive_cloud`` applies the attributes to it without any inference: pixel selection (``stride``,
depth range, ``edge``; images only), unprojection, ``voxel`` thinning (one representative point per
voxel, the first in pixel/storage order, colours never averaged), then colour, label and normals.
Object colours are the colour contract of ``segmentation.colors``; unsegmented points are grey.
The same source and attributes always give the same cloud, and attributes never change objects,
ids or colours.

Normals are computed last, for the emitted points only: an image's come from its depth grid, a
map's from the k nearest neighbours in the whole map (``PointNormals``, kept per source), so a
point's normal is the same whatever ``voxel`` says, and their cost follows the emitted points.
``derive_thinned`` also selects, for a cloud larger than a display budget (the viewer, spec §2.5),
one original point per occupied voxel of the finest grid that fits the budget
(``core.geometry.budget_voxel_indices``) before the normals, with every value of a selected point
exactly that of ``derive_cloud``; nothing is averaged. PLY writers never thin.

Memory: a map cloud can hold ten million points, so a derivation copies nothing it does not have
to. Positions stay in the source's dtype (float64 only where arithmetic needs it: ``voxel``,
``color=height``), and when no point is dropped the cloud shares the source's position, colour
and label arrays through read-only views instead of copying them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.cloud_attrs import CloudAttrs, CloudScope
from oh_my_slam.core.geometry import budget_voxel_indices, voxel_downsample_indices
from oh_my_slam.core.ply import PointCloud, ply_bytes
from oh_my_slam.core.types import Intrinsics
from oh_my_slam.reconstruction.pointcloud import (
    PointNormals,
    depth_normals,
    pixel_mask,
    pixel_points,
)
from oh_my_slam.segmentation.colors import height_colors, segment_colors

if TYPE_CHECKING:
    from oh_my_slam.reconstruction.api import FrameReconstruction
    from oh_my_slam.segmentation.api import FrameSegmentation

# element-wise passes over a whole map cloud work on this many points at a time (small temporaries)
CHUNK_POINTS = 1 << 18
IMAGE_FRAME = "oh-my-slam camera frame (OpenCV axes: x right, y down, z forward), metres"
MAP_FRAME = "oh-my-slam map frame (z up), metres"


@dataclass(eq=False)
class DisplaySelections:
    """A source's display selections (``derive_thinned``), by what decides the positions and the
    budget: the voxel edge of each one found (a few bytes; it saves the search when a control comes
    back), and the indices of the latest only (they can be 100 MB)."""

    edges: dict[Any, float] = field(default_factory=dict)
    latest: tuple[Any, NDArray[np.int64]] | None = None

    def indices(self, key: Any, xyz: NDArray[Any], max_points: int) -> tuple[NDArray[np.int64], float]:
        if self.latest is not None and self.latest[0] == key:
            return self.latest[1], self.edges[key]
        self.latest = None  # free the old indices first
        keep, edge = budget_voxel_indices(xyz, max_points, self.edges.get(key))
        self.edges[key], self.latest = edge, (key, keep)
        return keep, edge


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

    @cached_property
    def selections(self) -> DisplaySelections:
        """Display selections found so far (``derive_thinned``)."""
        return DisplaySelections()


@dataclass(frozen=True, eq=False)
class MapCloudSource:
    """A map's points in map coordinates (metres, z up)."""

    xyz: NDArray[Any]  # (N, 3)
    rgb: NDArray[np.uint8]  # (N, 3)
    labels: NDArray[np.int32] | None  # (N,) object id per point, 0 = none
    viewpoints: NDArray[np.float64]  # (M, 3) camera centres, to orient normals

    @cached_property
    def normals(self) -> PointNormals:
        """Normals of the map's points, computed on demand for the points emitted and kept."""
        return PointNormals(self.xyz, self.viewpoints)

    @cached_property
    def selections(self) -> DisplaySelections:
        """Display selections found so far (``derive_thinned``)."""
        return DisplaySelections()


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
        given, ids = np.asarray(labels, np.int32).reshape(-1), sorted(object_ids)
        lab = np.empty(len(given), np.int32)
        for s in range(0, len(given), CHUNK_POINTS):  # temporaries of one chunk only
            part = given[s:s + CHUNK_POINTS]
            lab[s:s + CHUNK_POINTS] = np.where(np.isin(part, ids), part, 0)
    return MapCloudSource(np.asarray(xyz), np.asarray(rgb, np.uint8), lab,
                          np.asarray(viewpoints, np.float64).reshape(-1, 3))


@dataclass(frozen=True)
class ThinnedCloud:
    """A derived cloud, possibly thinned: ``cloud`` holds, of the ``total`` points
    ``derive_cloud`` gives for the same attributes, one per occupied voxel of edge ``voxel``
    metres (0: every point), with exactly their values."""

    cloud: PointCloud
    total: int
    voxel: float


def derive_cloud(source: CloudSource, attrs: CloudAttrs) -> PointCloud:
    """The cloud ``attrs`` describe, from data already computed (no inference)."""
    return derive_thinned(source, attrs, None).cloud


def derive_thinned(source: CloudSource, attrs: CloudAttrs, max_points: int | None
                   ) -> ThinnedCloud:
    """``derive_cloud``; with more than ``max_points`` points, only the first point of each
    occupied voxel of the smallest edge whose grid has at most ``max_points`` of them
    (``budget_voxel_indices``; deterministic, kept in derivation order). Colours are those of the
    whole cloud (the height ramp's range included); normals are computed for the kept points
    only. The selection depends on the positions only: its edge is kept per source and attributes,
    and its indices are reused while only colour or normals change."""
    if (attrs.color == "segment" or attrs.label) and source.labels is None:
        raise ValueError("color=segment and label=on need a segmented source")
    # xyz[i] is the point of source row rows[i] (a flat pixel index, or a map point index);
    # rows None: every map point in storage order, whose arrays the cloud then shares (read-only)
    # with the source instead of copying them
    rows: NDArray[np.int64] | None
    if isinstance(source, ImageCloudSource):
        if attrs.color == "height" and source.up is None:
            raise ValueError("color=height needs the image's estimated gravity")
        mask = pixel_mask(source.depth, source.valid, attrs)
        xyz, rows = pixel_points(source.depth, source.K, mask)
        rgb, labels = source.rgb.reshape(-1, 3), source.labels
    else:
        xyz, rows = _readonly(np.asarray(source.xyz).reshape(-1, 3)), None
        rgb, labels = source.rgb, source.labels
    if attrs.voxel > 0:
        keep = voxel_downsample_indices(xyz, attrs.voxel, keep="first")
        xyz, rows = xyz[keep], keep if rows is None else rows[keep]

    def pick(a: NDArray[Any]) -> NDArray[Any]:
        return _readonly(a) if rows is None else a[rows]

    lab = None if labels is None else pick(labels.reshape(-1))
    colour: NDArray[np.uint8] | None
    if attrs.color == "rgb":
        colour = pick(rgb)
    elif attrs.color == "segment":
        assert lab is not None
        colour = segment_colors(lab)
    elif attrs.color == "height":
        up = source.up if isinstance(source, ImageCloudSource) else np.array([0.0, 0.0, 1.0])
        colour = height_colors(np.asarray(xyz, np.float64) @ np.asarray(up, np.float64))
    else:
        colour = None
    total, edge = len(xyz), 0.0
    if max_points is not None and total > max_points:
        key = (_position_key(attrs), max_points)
        keep, edge = source.selections.indices(key, xyz, max_points)
        if len(keep) < total:
            xyz = xyz[keep]
            rows = keep if rows is None else rows[keep]
            lab = None if lab is None else lab[keep]
            colour = None if colour is None else colour[keep]
    normals: NDArray[np.float32] | None = None
    if attrs.normals:
        if isinstance(source, ImageCloudSource):
            assert rows is not None
            normals = source.normals.reshape(-1, 3)[rows]
        else:
            normals = source.normals.at(np.arange(total) if rows is None else rows)
    cloud = PointCloud(xyz, colour, lab if attrs.label else None, normals)
    return ThinnedCloud(cloud, total, edge)


def display_selection(xyz: NDArray[Any], max_points: int
                      ) -> tuple[NDArray[np.int64] | None, float]:
    """(indices, edge) of the points of ``xyz`` drawn within the display budget: ``None`` (every
    point) when there are at most ``max_points``, else ``budget_voxel_indices``'s selection — the
    one ``derive_thinned`` makes. Only positions are needed, so a caller can pick the other
    attributes of the kept points alone (e.g. from a memory-mapped file)."""
    if len(xyz) <= max_points:
        return None, 0.0
    return budget_voxel_indices(xyz, max_points)


def thin_cloud(cloud: PointCloud, max_points: int) -> ThinnedCloud:
    """A cloud already derived (e.g. read from a PLY file the commands wrote) within the display
    budget, by the selection ``derive_thinned`` makes: with more than ``max_points`` points, the
    first point of each occupied voxel of the smallest edge whose grid has at most ``max_points``
    of them (``budget_voxel_indices``). Each kept point keeps exactly its values (position,
    colour, label, normal)."""
    total = len(cloud)
    keep, edge = display_selection(cloud.xyz, max_points)
    if keep is None:
        return ThinnedCloud(cloud, total, 0.0)

    def pick(a: NDArray[Any] | None) -> NDArray[Any] | None:
        return None if a is None else a[keep]

    thinned = PointCloud(cloud.xyz[keep], pick(cloud.rgb), pick(cloud.label), pick(cloud.normals))
    return ThinnedCloud(thinned, total, edge)


def _position_key(attrs: CloudAttrs) -> CloudAttrs:
    """``attrs`` with what does not move or drop points (colour, normals, PLY-only keys) reset."""
    return replace(attrs, color="rgb", normals=False, label=False, encoding="binary")


def _readonly(a: NDArray[Any]) -> NDArray[Any]:
    """A view of ``a`` that cannot write into it (a derived cloud sharing its source's array)."""
    v = a.view()
    v.flags.writeable = False
    return v


def cloud_ply(source: CloudSource, attrs: CloudAttrs, extra_comments: Sequence[str] = ()
              ) -> bytes:
    """``derive_cloud`` as a PLY whose header names the frame and records the effective
    attributes (defaults included; for a map only those that apply), then ``extra_comments``
    (one header line each: the located cameras of ``mapper.sh locate``)."""
    scope = scope_of(source)
    frame = IMAGE_FRAME if scope == CloudScope.IMAGE else MAP_FRAME
    return ply_bytes(derive_cloud(source, attrs), encoding=attrs.encoding,
                     comments=[frame, f"attributes {attrs.describe(scope)}", *extra_comments])
