"""Geometry-first object candidates: what stands on the support plane a view looks down at.

A view from above onto a light worktop shows flat, low-contrast items (a white plate, a grey
bowl, a mug's rim) that the detector scores low or not at all on the whole frame, although its
depth shows each of them plainly as a raised disc or rim on the plane. ``raised_regions`` finds
those: in a view whose optical axis looks down (``LOOK_DOWN_COS``), the support is the height
most of its depth samples share (``SUPPORT_SHARE`` within ``SUPPORT_BAND_M``); a candidate is a
connected area raised above it by more than the depth's noise (``RAISE_MIN_M``, or
``RAISE_NOISE`` times the depth's relative noise at its distance) and at most ``RAISE_MAX_M``,
its holes filled (the inside of a plate's rim or of a mug), not on the robot (nearer the camera
than ``ON_ROBOT_M``), of an object's size (``REGION_MIN_PX``, ``REGION_MAX_M``). The detector
then looks at each candidate closer (``detect.detect_regions``): a crop around it, upsampled to
the detector's input, and a detection there is kept only when it lies on the raised area
(``segmentation.api.label_regions``). The geometry finds the place; the label is always the
detector's."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from oh_my_slam.core.geometry import unproject_pixels
from oh_my_slam.core.types import Intrinsics

LOOK_DOWN_COS = 0.7  # the optical axis within 45 degrees of straight down
SUPPORT_BIN_M = 0.005
SUPPORT_BAND_M = 0.01
SUPPORT_SHARE = 0.3  # of the view's depth samples within SUPPORT_BAND_M of the support's level
RAISE_MIN_M = 0.008
RAISE_NOISE = 2.0
RAISE_MAX_M = 0.4
ON_ROBOT_M = 0.15  # depth nearer than this is the robot (a gripper under the camera)
ROBOT_RING_PX = 3  # a raised area this share of whose outline touches the robot is part of it
ROBOT_CONTACT = 0.2
OPEN_PX = 2  # opening of the raised pixels: depth speckle is no object
REGION_MIN_PX = 300
REGION_MAX_M = 0.6  # a candidate's largest footprint side


@dataclass
class Region:
    """A raised area of a view: its mask (holes filled) and box (x0, y0, x1, y1) on the depth
    grid, its largest height above the support (metres)."""

    mask: NDArray[np.bool_]
    box: tuple[int, int, int, int]
    height_m: float


def support_level(heights: NDArray[Any]) -> float | None:
    """The height most samples share (``SUPPORT_BIN_M`` bins), when ``SUPPORT_SHARE`` of them lie
    within ``SUPPORT_BAND_M`` of it; None otherwise (no plane dominates the view)."""
    h = np.asarray(heights, np.float64)
    if len(h) == 0:
        return None
    lo = float(h.min())
    bins = np.floor((h - lo) / SUPPORT_BIN_M).astype(np.int64)
    mode = lo + (float(np.argmax(np.bincount(bins))) + 0.5) * SUPPORT_BIN_M
    near = np.abs(h - mode) <= SUPPORT_BAND_M
    return float(np.median(h[near])) if float(np.mean(near)) >= SUPPORT_SHARE else None


def raised_regions(depth: NDArray[Any], K: Intrinsics, T_map_cam: NDArray[Any],
                   depth_rel: float, valid: NDArray[Any] | None = None) -> list[Region]:
    """The candidates of a view (see the module docstring): ``depth`` in metres on the grid of
    ``K`` (0: none), ``T_map_cam`` the view's pose in a frame whose z is up, ``depth_rel`` the
    depth's relative noise; none unless the view looks down at a dominant support."""
    T = np.asarray(T_map_cam, np.float64)
    if -float(T[2, 2]) < LOOK_DOWN_COS:
        return []
    d = np.asarray(depth, np.float64)
    ok = np.isfinite(d) & (d > 0)
    if valid is not None:
        ok &= np.asarray(valid, bool)
    v, u = np.nonzero(ok)
    if not len(v):
        return []
    z = unproject_pixels(u, v, d[v, u], K.K()) @ T[2, :3] + T[2, 3]
    level = support_level(z)
    if level is None:
        return []
    rise = np.full(d.shape, -np.inf)
    rise[v, u] = z - level
    floor_m = np.maximum(RAISE_MIN_M, RAISE_NOISE * depth_rel * np.where(ok, d, 0.0))
    raised = (rise > floor_m) & (rise <= RAISE_MAX_M) & ok & (d >= ON_ROBOT_M)
    raised = ndimage.binary_opening(raised, iterations=OPEN_PX)
    robot = ok & (d < ON_ROBOT_M)
    labels, _ = ndimage.label(raised)
    out = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        m = np.zeros((d.shape[0], d.shape[1]), bool)
        m[sl] = ndimage.binary_fill_holes(labels[sl] == i)
        if m.sum() < REGION_MIN_PX:
            continue
        ring = ndimage.binary_dilation(m, iterations=ROBOT_RING_PX) & ~m
        if robot[ring].mean() > ROBOT_CONTACT:
            continue  # a part of the robot beyond ON_ROBOT_M, joined to its nearer part
        rows, cols = np.nonzero(m)
        pts = unproject_pixels(cols, rows, np.where(ok[rows, cols], d[rows, cols], np.nan),
                               K.K()) @ T[:3, :3].T + T[:3, 3]
        span = np.nanmax(pts[:, :2], axis=0) - np.nanmin(pts[:, :2], axis=0)
        if float(np.max(span)) > REGION_MAX_M:
            continue
        out.append(Region(m, (int(cols.min()), int(rows.min()), int(cols.max()),
                              int(rows.max())), round(float(np.max(rise[m & ok])), 4)))
    return out
