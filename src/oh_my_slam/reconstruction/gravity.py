"""Gravity: GeoCalib's up direction, refined by a RANSAC floor plane within 5 degrees."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.geometry import angle_between_deg, ransac_plane

FLOOR_MAX_ANGLE_DEG = 5.0
DEFAULT_UP_CAM = np.array([0.0, -1.0, 0.0])  # level camera, OpenCV axes


@dataclass
class GravityEstimate:
    """Unit "up" direction in camera coordinates (OpenCV axes)."""

    up_cam: NDArray[np.float64] = field(default_factory=lambda: DEFAULT_UP_CAM.copy())
    source: str = "default"  # default | geocalib | geocalib+floor
    roll_unc_deg: float = 10.0
    pitch_unc_deg: float = 10.0
    floor_height: float | None = None  # camera height above the floor plane, metres
    floor_inliers: int = 0
    prior_up_cam: list[float] | None = None  # GeoCalib's estimate before floor refinement

    @property
    def confidence(self) -> float:
        """Weight for averaging (inverse variance of the angular uncertainty)."""
        unc = max(0.1, float(np.hypot(self.roll_unc_deg, self.pitch_unc_deg)))
        return 1.0 / unc**2

    def to_dict(self) -> dict[str, Any]:
        return {
            "up_cam": [float(v) for v in self.up_cam],
            "source": self.source,
            "roll_unc_deg": self.roll_unc_deg,
            "pitch_unc_deg": self.pitch_unc_deg,
            "floor_height": self.floor_height,
            "floor_inliers": self.floor_inliers,
            "prior_up_cam": self.prior_up_cam,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> GravityEstimate:
        return GravityEstimate(
            up_cam=np.asarray(d["up_cam"], dtype=np.float64),
            source=d.get("source", "default"),
            roll_unc_deg=float(d.get("roll_unc_deg", 10.0)),
            pitch_unc_deg=float(d.get("pitch_unc_deg", 10.0)),
            floor_height=d.get("floor_height"),
            floor_inliers=int(d.get("floor_inliers", 0)),
            prior_up_cam=d.get("prior_up_cam"),
        )


def floor_candidate_height(h: NDArray[Any], bin_m: float = 0.03, min_support: float = 0.02,
                           max_below: float = 0.03, margin: float = 0.12) -> float | None:
    """Height (along up) of the lowest horizontal-surface peak with almost nothing beneath it.

    Horizontal surfaces (floor, table tops, seats) are peaks of the height histogram. The floor is
    the lowest peak with enough support such that at most ``max_below`` of all points lie more
    than ``margin`` below it — table tops seen from above carry more points than an occluded
    floor, so "most inliers" would pick them.
    """
    n = len(h)
    lo, hi = np.percentile(h, [0.5, 99.5])
    if hi - lo < 3 * bin_m:
        return None
    edges = np.arange(lo, hi + bin_m, bin_m)
    counts, edges = np.histogram(h, bins=edges)
    smooth = np.convolve(counts, np.ones(3), mode="same")
    centers = (edges[:-1] + edges[1:]) / 2
    need = max(300, min_support * n)
    h_sorted = np.sort(h)
    for i in range(len(smooth)):
        left = smooth[i - 1] if i > 0 else -1
        right = smooth[i + 1] if i + 1 < len(smooth) else -1
        if smooth[i] < need or smooth[i] < left or smooth[i] < right:
            continue
        below = np.searchsorted(h_sorted, centers[i] - margin) / n
        if below <= max_below:
            return float(centers[i])
    return None


def refine_with_floor(
    points_cam: NDArray[Any],
    prior: GravityEstimate,
    max_angle_deg: float = FLOOR_MAX_ANGLE_DEG,
    min_inliers: int = 300,
    seed: int = 0,
) -> GravityEstimate:
    """Refine ``prior.up_cam`` with the floor plane (lowest well-supported horizontal surface).

    The plane normal must be within ``max_angle_deg`` of the prior; otherwise the prior is kept.
    """
    pts = np.asarray(points_cam, dtype=np.float64)
    up = prior.up_cam / np.linalg.norm(prior.up_cam)
    if len(pts) < 200:
        return prior
    if len(pts) > 80000:
        rng = np.random.default_rng(seed)
        pts = pts[rng.choice(len(pts), 80000, replace=False)]
    h = pts @ up
    floor_h = floor_candidate_height(h)
    if floor_h is None:
        return prior
    band = pts[np.abs(h - floor_h) < 0.15]
    if len(band) < min_inliers:
        return prior
    z_med = float(np.median(np.linalg.norm(band, axis=1)))
    thr = max(0.02, 0.006 * z_med)
    res = ransac_plane(band, thr, iterations=300, seed=seed, normal_prior=up,
                       max_angle_deg=max_angle_deg)
    if res is None:
        return prior
    n, d, inliers = res
    count = int(inliers.sum())
    if count < min_inliers or angle_between_deg(n, up) > max_angle_deg:
        return prior
    # the floor must be below the camera: n·x + d = 0 with n pointing up => camera height = d
    height = float(d)
    if height <= 0:
        return prior
    return GravityEstimate(
        up_cam=n,
        source=prior.source + "+floor",
        roll_unc_deg=min(prior.roll_unc_deg, 1.0),
        pitch_unc_deg=min(prior.pitch_unc_deg, 1.0),
        floor_height=height,
        floor_inliers=count,
        prior_up_cam=[float(v) for v in up],
    )


def mean_up(estimates: list[NDArray[Any]], weights: list[float]) -> NDArray[np.float64]:
    """Confidence-weighted mean direction (already expressed in a common frame)."""
    v = np.zeros(3)
    for e, w in zip(estimates, weights, strict=True):
        v += w * np.asarray(e, dtype=np.float64) / np.linalg.norm(e)
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.array([0.0, 0.0, 1.0])
