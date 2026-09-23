"""Coloured point clouds from a depth grid: validity (invalid and depth-edge pixels dropped),
per-pixel colours from the resized image, camera- or map-frame coordinates."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.geometry import depth_edge_mask, unproject_pixels
from oh_my_slam.core.ply import PointCloud
from oh_my_slam.core.types import Intrinsics, Pose

EDGE_REL_THRESHOLD = 0.04
MAX_GRID_SIDE = 1024


def cloud_mask(depth: NDArray[Any], valid: NDArray[Any] | None = None,
               edge_rel: float = EDGE_REL_THRESHOLD, max_depth: float | None = None) -> NDArray[Any]:
    """Pixels that become points: finite positive depth, valid, not on a depth edge."""
    d = np.asarray(depth)
    m = np.isfinite(d) & (d > 0)
    if valid is not None:
        m &= np.asarray(valid, dtype=bool)
    if max_depth is not None:
        m &= d < max_depth
    m &= ~depth_edge_mask(np.where(m, d, 0.0), rel_threshold=edge_rel)
    return m


def pixel_points(depth: NDArray[Any], K: Intrinsics, mask: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
    """Camera-frame points for ``mask`` pixels and their flat pixel indices (row-major)."""
    v, u = np.nonzero(mask)
    z = np.asarray(depth)[v, u].astype(np.float64)
    pts = unproject_pixels(u, v, z, K.K())
    return pts, v * mask.shape[1] + u


def frame_cloud(
    depth: NDArray[Any],
    rgb: NDArray[np.uint8],
    K: Intrinsics,
    mask: NDArray[Any],
    T_parent_cam: Pose | None = None,
) -> tuple[PointCloud, NDArray[Any]]:
    """Coloured cloud for ``mask`` pixels, optionally transformed; also returns pixel indices."""
    if rgb.shape[:2] != depth.shape:
        raise ValueError(f"image {rgb.shape[:2]} and depth {depth.shape} grids differ")
    pts, idx = pixel_points(depth, K, mask)
    if T_parent_cam is not None:
        pts = T_parent_cam.apply(pts)
    colors = rgb.reshape(-1, 3)[idx]
    return PointCloud(pts.astype(np.float32), colors), idx
