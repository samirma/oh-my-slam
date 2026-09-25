"""Upright oriented bounding boxes (gravity-aligned z axis, yaw only).

Fit: rotate the points so "up" is +z, try the directions of the 2D convex-hull edges in the
horizontal plane, keep the yaw whose robust (2nd-98th percentile) rectangle has the least area,
then take robust extents along the chosen axes. Canonical yaw: x = the longer horizontal side
(width), y = depth, z = height; yaw in (-90, 90] degrees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, QhullError

from oh_my_slam.core.geometry import rot_z, rotation_between
from oh_my_slam.core.types import Pose

LOW_PCT = 2.0
HIGH_PCT = 98.0
MAX_FIT_SAMPLES = 5000
MIN_SIZE = 0.01  # metres, avoids zero-thickness boxes
TINY = 0.1  # metres: a box under this in every dimension is never grounded


@dataclass
class OBB:
    center: NDArray[np.float64]  # (3,)
    R: NDArray[np.float64]  # columns: box x (width), y (depth), z (height) in the parent frame
    size: NDArray[np.float64]  # (width, depth, height) metres

    @property
    def volume(self) -> float:
        return float(np.prod(self.size))

    def corners(self) -> NDArray[np.float64]:
        h = self.size / 2
        c = np.array([[x, y, z] for x in (-h[0], h[0]) for y in (-h[1], h[1])
                      for z in (-h[2], h[2])])
        return c @ self.R.T + self.center

    def transformed(self, T: Pose) -> OBB:
        return OBB(T.R @ self.center + T.t, T.R @ self.R, self.size.copy())

    def contains(self, points: NDArray[Any], margin: float = 0.0) -> NDArray[np.bool_]:
        local = (np.asarray(points) - self.center) @ self.R
        return np.all(np.abs(local) <= self.size / 2 + margin, axis=1)

    def to_dict(self) -> dict[str, Any]:
        return {"center": self.center.tolist(), "R": self.R.tolist(), "size": self.size.tolist()}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> OBB:
        return OBB(np.asarray(d["center"], float), np.asarray(d["R"], float),
                   np.asarray(d["size"], float))


def _robust_extent(x: NDArray[Any]) -> tuple[float, float]:
    lo, hi = np.percentile(x, [LOW_PCT, HIGH_PCT])
    return float(lo), float(hi)


def _candidate_yaws(xy: NDArray[Any]) -> NDArray[np.float64]:
    if len(xy) < 4:
        return np.array([0.0])
    try:
        hull = ConvexHull(xy)
    except QhullError:
        return np.array([0.0])
    pts = xy[hull.vertices]
    edges = np.roll(pts, -1, axis=0) - pts
    yaws = np.mod(np.arctan2(edges[:, 1], edges[:, 0]), np.pi / 2)
    return np.unique(np.round(yaws, 4))


def fit_upright_obb(points: NDArray[Any], seed: int = 0) -> OBB:
    """OBB for points already in a z-up frame."""
    p = np.asarray(points, dtype=np.float64)
    if len(p) == 0:
        raise ValueError("cannot fit a box to zero points")
    sample = p
    if len(p) > MAX_FIT_SAMPLES:
        rng = np.random.default_rng(seed)
        sample = p[rng.choice(len(p), MAX_FIT_SAMPLES, replace=False)]
    xy = sample[:, :2]
    best_yaw, best_area = 0.0, np.inf
    for yaw in _candidate_yaws(xy):
        c, s = np.cos(yaw), np.sin(yaw)
        a = xy @ np.array([c, s])
        b = xy @ np.array([-s, c])
        lo_a, hi_a = _robust_extent(a)
        lo_b, hi_b = _robust_extent(b)
        area = (hi_a - lo_a) * (hi_b - lo_b)
        if area < best_area - 1e-12:
            best_area, best_yaw = area, float(yaw)
    c, s = np.cos(best_yaw), np.sin(best_yaw)
    a = p[:, :2] @ np.array([c, s])
    b = p[:, :2] @ np.array([-s, c])
    lo_a, hi_a = _robust_extent(a)
    lo_b, hi_b = _robust_extent(b)
    lo_z, hi_z = _robust_extent(p[:, 2])
    w, d, h = hi_a - lo_a, hi_b - lo_b, hi_z - lo_z
    ca, cb, cz = (lo_a + hi_a) / 2, (lo_b + hi_b) / 2, (lo_z + hi_z) / 2
    center = np.array([ca * c - cb * s, ca * s + cb * c, cz])
    yaw = best_yaw
    if d > w:
        w, d = d, w
        yaw += np.pi / 2
    yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2  # (-90, 90]
    if yaw <= -np.pi / 2 + 1e-12:
        yaw += np.pi
    size = np.maximum(np.array([w, d, h]), MIN_SIZE)
    return OBB(center, rot_z(yaw), size)


def ground_upright(box: OBB, floor_z: float, max_gap: float, min_visible: float = 0.0) -> OBB:
    """Extend a z-up box down to the floor when its visible bottom floats at most ``max_gap``
    above it (floor-standing objects whose lower part is occluded), its visible height is at least
    ``min_visible`` of the grounded height (no extrapolation from a sliver) and it is not tiny
    (``TINY``: under that in every dimension, it cannot stand for a floor-standing object)."""
    bottom = box.center[2] - box.size[2] / 2
    top = box.center[2] + box.size[2] / 2
    gap = bottom - floor_z
    if not (0.0 < gap <= max_gap) or top <= floor_z:
        return box
    if box.size[2] < min_visible * (top - floor_z) or float(np.max(box.size)) < TINY:
        return box
    center = box.center.copy()
    center[2] = (top + floor_z) / 2
    size = box.size.copy()
    size[2] = top - floor_z
    return OBB(center, box.R, size)


def fit_obb(points: NDArray[Any], up: NDArray[Any], floor_level: float | None = None,
            max_gap: float = 0.0, min_visible: float = 0.0) -> OBB:
    """Upright OBB for points in any frame, given that frame's unit "up" direction.

    ``floor_level`` is the floor's coordinate along ``up`` (``up · x`` for floor points); with
    ``max_gap > 0`` the box is grounded (see :func:`ground_upright`).
    """
    up = np.asarray(up, dtype=np.float64)
    up /= np.linalg.norm(up)
    R_g = rotation_between(up, np.array([0.0, 0.0, 1.0]))  # parent -> gravity-aligned
    box = fit_upright_obb(np.asarray(points, dtype=np.float64) @ R_g.T)
    if floor_level is not None and max_gap > 0:
        box = ground_upright(box, floor_level, max_gap, min_visible)
    return OBB(R_g.T @ box.center, R_g.T @ box.R, box.size)


def obb_iou_upright(a: OBB, b: OBB, samples: int = 20000, seed: int = 0) -> float:
    """Monte-Carlo 3D IoU of two boxes (used by tests and identity checks)."""
    rng = np.random.default_rng(seed)
    lo = np.minimum(a.corners().min(0), b.corners().min(0))
    hi = np.maximum(a.corners().max(0), b.corners().max(0))
    pts = rng.uniform(lo, hi, size=(samples, 3))
    ia, ib = a.contains(pts), b.contains(pts)
    union = np.logical_or(ia, ib).sum()
    return float(np.logical_and(ia, ib).sum() / union) if union else 0.0
