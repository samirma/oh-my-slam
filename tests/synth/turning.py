"""A head turning in place inside a box room, analytically (no rendering, no COLMAP): true poses,
ray-cast z-depth grids and the verified matches a feature matcher would return, for testing the
multi-view pose refinement (``mapping.panorama``)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.geometry import rot_z
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.mapping.panorama import PairMatches, View

K = Intrinsics(435.0, 435.0, 320.0, 240.0, 640, 480)
ROOM = (3.0, 2.5, 0.0, 2.6)  # half-width x, half-depth y, floor z, ceiling z
PIVOT = np.array([0.3, -0.2, 1.3])
NECK = 0.05  # camera centre in front of the pivot (m)


def head_pose(yaw_deg: float, pitch_deg: float = -12.0, step: NDArray[np.float64] | None = None
              ) -> Pose:
    """Camera-to-map pose (OpenCV axes) of a head at ``PIVOT`` (+ ``step``) turned by yaw/pitch."""
    base = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # looks along +x
    p = np.radians(pitch_deg)
    tilt = np.array([[np.cos(p), 0.0, -np.sin(p)], [0.0, 1.0, 0.0], [np.sin(p), 0.0, np.cos(p)]])
    R = rot_z(np.radians(yaw_deg)) @ tilt @ base
    pivot = PIVOT + (np.zeros(3) if step is None else step)
    return Pose(R, pivot + R @ np.array([0.0, 0.0, NECK]))


def _cast(C: NDArray[np.float64], d: NDArray[np.float64]) -> NDArray[np.float64]:
    """Distance along unit rays ``d`` from ``C`` to the room's walls, floor or ceiling."""
    hx, hy, z0, z1 = ROOM
    t = np.full(len(d), np.inf)
    for axis, lo, hi in ((0, -hx, hx), (1, -hy, hy), (2, z0, z1)):
        with np.errstate(divide="ignore", invalid="ignore"):
            for plane in (lo, hi):
                tt = (plane - C[axis]) / d[:, axis]
                t = np.where((tt > 1e-6) & (tt < t), tt, t)
    return t


def depth_grid(pose: Pose, K: Intrinsics = K) -> NDArray[np.float32]:
    """True z-depth at the grid's pixel centres."""
    v, u = np.mgrid[0:K.height, 0:K.width]
    rays = np.stack([(u + 0.5 - K.cx) / K.fx, (v + 0.5 - K.cy) / K.fy, np.ones_like(u, float)],
                    -1).reshape(-1, 3)
    n = np.linalg.norm(rays, axis=1)
    t = _cast(pose.t, (rays / n[:, None]) @ pose.R.T)
    return (t / n).reshape(K.height, K.width).astype(np.float32)


@dataclass
class Rig:
    poses: dict[str, Pose]
    depth: dict[str, NDArray[np.float32]]
    pairs: list[PairMatches]

    def views(self, poses: dict[str, Pose] | None = None, depth_scale: dict[str, float] | None
              = None, K_used: Intrinsics = K) -> dict[str, View]:
        poses = poses or self.poses
        return {n: View(poses[n], K_used, self.depth[n] * (depth_scale or {}).get(n, 1.0), K,
                        (K.width, K.height)) for n in self.poses}


def turning_rig(poses: dict[str, Pose], points: int = 6000, noise_px: float = 0.4,
                seed: int = 0, max_per_pair: int = 200) -> Rig:
    """Matches between every two views that share >= 20 of ``points`` random wall/floor points
    (pixel noise ``noise_px``, COLMAP pixel convention)."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(points, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    X = PIVOT + _cast(PIVOT, dirs)[:, None] * dirs
    obs: dict[str, tuple[NDArray[np.float64], NDArray[np.bool_]]] = {}
    for n, T in poses.items():
        pc = T.inverse().apply(X)
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = np.stack([K.fx * pc[:, 0] / z + K.cx, K.fy * pc[:, 1] / z + K.cy], 1)
        vis = (z > 0.1) & (uv[:, 0] > 2) & (uv[:, 0] < K.width - 2) & (uv[:, 1] > 2) \
            & (uv[:, 1] < K.height - 2)
        # occlusion is impossible in an empty convex room: every point in view is seen
        obs[n] = (uv + rng.normal(scale=noise_px, size=uv.shape), vis)
    names = sorted(poses)
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = obs[a][1] & obs[b][1]
            idx = np.flatnonzero(both)
            if len(idx) < 20:
                continue
            idx = idx[:max_per_pair]
            pairs.append(PairMatches(a, b, obs[a][0][idx], obs[b][0][idx]))
    return Rig(dict(poses), {n: depth_grid(T) for n, T in poses.items()}, pairs)


def perturb(pose: Pose, rot_deg: float, trans_m: float, rng: np.random.Generator) -> Pose:
    from oh_my_slam.mapping.panorama import _exp

    w = rng.normal(size=3)
    w *= np.radians(rot_deg) / np.linalg.norm(w)
    t = rng.normal(size=3)
    t *= trans_m / np.linalg.norm(t)
    return Pose(_exp(w) @ pose.R, pose.t + t)


def rot_err_deg(A: Pose, B: Pose) -> float:
    c = (np.trace(A.R.T @ B.R) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
