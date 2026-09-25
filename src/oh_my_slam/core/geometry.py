"""Rigid/similarity transforms, quaternions (scalar last), camera projection and point utilities.

Quaternions are ``(qx, qy, qz, qw)`` as in the OpenLABEL cuboid, normalised with ``qw >= 0``.
Cameras use OpenCV axes. All functions are pure NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

F64 = NDArray[np.float64]


# --- rotations -----------------------------------------------------------------------------------


def quat_to_rot(q: NDArray[Any]) -> F64:
    """Rotation matrix from a scalar-last quaternion (normalised internally)."""
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot_to_quat(R: NDArray[Any]) -> F64:
    """Scalar-last unit quaternion with ``qw >= 0`` from a rotation matrix."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return q


def rot_z(angle_rad: float) -> F64:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_between(a: NDArray[Any], b: NDArray[Any]) -> F64:
    """Smallest rotation taking direction ``a`` onto direction ``b``."""
    a = np.asarray(a, dtype=np.float64) / np.linalg.norm(a)
    b = np.asarray(b, dtype=np.float64) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -1 + 1e-9:
        # 180 degrees: rotate about any axis orthogonal to a
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def angle_between_deg(a: NDArray[Any], b: NDArray[Any]) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


# --- SE(3) / Sim(3) ------------------------------------------------------------------------------


def se3(R: NDArray[Any], t: NDArray[Any]) -> F64:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


@dataclass(frozen=True)
class Sim3:
    """Similarity ``x' = s * R @ x + t``."""

    s: float
    R: F64
    t: F64

    @staticmethod
    def identity() -> Sim3:
        return Sim3(1.0, np.eye(3), np.zeros(3))

    def apply(self, points: NDArray[Any]) -> F64:
        return self.s * (np.asarray(points, dtype=np.float64) @ self.R.T) + self.t

    def compose(self, other: Sim3) -> Sim3:
        """``self ∘ other`` (apply ``other`` first)."""
        return Sim3(self.s * other.s, self.R @ other.R, self.s * (self.R @ other.t) + self.t)

    def inverse(self) -> Sim3:
        Ri = self.R.T
        return Sim3(1.0 / self.s, Ri, -(Ri @ self.t) / self.s)

    def transform_pose(self, T_world_cam: NDArray[Any]) -> F64:
        """Map a camera-to-world pose into the target frame (camera stays metric-rigid)."""
        T = np.asarray(T_world_cam, dtype=np.float64)
        return se3(self.R @ T[:3, :3], self.apply(T[:3, 3][None])[0])

    def matrix(self) -> F64:
        M = np.eye(4)
        M[:3, :3] = self.s * self.R
        M[:3, 3] = self.t
        return M


def umeyama(src: NDArray[Any], dst: NDArray[Any], with_scale: bool = True) -> Sim3:
    """Least-squares similarity (or rigid if ``with_scale=False``) mapping ``src`` onto ``dst``."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or len(src) < 3:
        raise ValueError("umeyama needs two (N>=3, 3) arrays of equal shape")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    if with_scale:
        var_s = (xs**2).sum() / len(src)
        s = float(np.trace(np.diag(S) @ D) / var_s) if var_s > 0 else 1.0
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return Sim3(s, R, t)


# --- camera projection ---------------------------------------------------------------------------


def unproject_pixels(u: NDArray[Any], v: NDArray[Any], z: NDArray[Any], K: NDArray[Any]) -> F64:
    x = (np.asarray(u, dtype=np.float64) - K[0, 2]) / K[0, 0] * z
    y = (np.asarray(v, dtype=np.float64) - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, np.asarray(z, dtype=np.float64)], axis=-1)


def project(points_cam: NDArray[Any], K: NDArray[Any]) -> tuple[F64, F64]:
    """Project camera-frame points; returns (uv (N, 2), z (N,)). Points behind get NaN uv."""
    p = np.asarray(points_cam, dtype=np.float64)
    z = p[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * p[:, 0] / z + K[0, 2]
        v = K[1, 1] * p[:, 1] / z + K[1, 2]
    uv = np.stack([u, v], axis=1)
    uv[z <= 0] = np.nan
    return uv, z


def depth_edge_mask(depth: NDArray[Any], rel_threshold: float = 0.04, size: int = 3) -> NDArray[Any]:
    """True where depth jumps by more than ``rel_threshold`` (relative) within a window, and on
    valid pixels touching invalid ones. ``rel_threshold <= 0`` disables the test (no edges)."""
    d = np.asarray(depth, dtype=np.float64)
    if rel_threshold <= 0:
        return np.zeros(d.shape, bool)
    valid = np.isfinite(d) & (d > 0)
    big = np.where(valid, d, np.nan)
    fill_hi = np.where(valid, d, -np.inf)
    fill_lo = np.where(valid, d, np.inf)
    dmax = ndimage.maximum_filter(fill_hi, size=size, mode="nearest")
    dmin = ndimage.minimum_filter(fill_lo, size=size, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = (dmax - dmin) / big
    edges = np.asarray(rel > rel_threshold)
    # a valid pixel touching an invalid one is also an edge
    touching_invalid = ndimage.binary_dilation(~valid, structure=np.ones((size, size), dtype=bool))
    return np.asarray((edges | touching_invalid) & valid)


# --- point utilities -----------------------------------------------------------------------------


def voxel_keys(points: NDArray[Any], voxel: float) -> NDArray[np.int64]:
    return np.floor(np.asarray(points, dtype=np.float64) / voxel).astype(np.int64)


def voxel_downsample_indices(points: NDArray[Any], voxel: float, keep: str = "last") -> NDArray[Any]:
    """Indices of one representative point per voxel.

    ``keep="last"`` keeps the latest point in each voxel (points ordered oldest → newest), which is
    how "latest colour wins" is implemented; ``"first"`` keeps the earliest.
    """
    if len(points) == 0:
        return np.zeros(0, dtype=np.int64)
    keys = _packed(voxel_keys(points, voxel))
    if keep == "last":
        rev = keys[::-1]
        _, idx = np.unique(rev, axis=0, return_index=True)
        out = len(points) - 1 - idx
    else:
        _, out = np.unique(keys, axis=0, return_index=True)
    return np.sort(out)


def _packed(keys: NDArray[np.int64]) -> NDArray[np.int64]:
    """(N, 3) voxel keys as one int64 per voxel (a 1-D unique sorts far faster than rows), or the
    rows themselves when the key range does not fit. Equal rows <=> equal packed values."""
    lo = keys.min(axis=0)
    span = (keys.max(axis=0) - lo + 1).astype(object)  # Python ints: no overflow in the check
    if span[0] * span[1] * span[2] >= 2**62:
        return keys
    k = keys - lo
    return (k[:, 0] * int(span[1] * span[2]) + k[:, 1] * int(span[2]) + k[:, 2]).astype(np.int64)


def fit_plane(points: NDArray[Any]) -> tuple[F64, float]:
    """Least-squares plane: unit normal n and offset d with n·x + d = 0."""
    p = np.asarray(points, dtype=np.float64)
    c = p.mean(0)
    _, _, Vt = np.linalg.svd(p - c, full_matrices=False)
    n = Vt[2]
    return n, float(-n @ c)


def ransac_plane(
    points: NDArray[Any],
    threshold: float,
    iterations: int = 300,
    seed: int = 0,
    normal_prior: NDArray[Any] | None = None,
    max_angle_deg: float = 180.0,
) -> tuple[F64, float, NDArray[np.bool_]] | None:
    """RANSAC plane with an optional constraint on the normal's angle to ``normal_prior``.

    Returns (normal, d, inlier mask) with the normal oriented along the prior when given.
    """
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 3:
        return None
    rng = np.random.default_rng(seed)
    best: tuple[int, F64, float] | None = None
    cos_max = np.cos(np.radians(max_angle_deg))
    prior = None if normal_prior is None else np.asarray(normal_prior) / np.linalg.norm(normal_prior)
    for _ in range(iterations):
        i = rng.choice(len(p), 3, replace=False)
        a, b, c = p[i]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        if prior is not None:
            if n @ prior < 0:
                n = -n
            if n @ prior < cos_max:
                continue
        d = -n @ a
        count = int((np.abs(p @ n + d) < threshold).sum())
        if best is None or count > best[0]:
            best = (count, n, float(d))
    if best is None:
        return None
    _, n, d = best
    inliers = np.abs(p @ n + d) < threshold
    n2, d2 = fit_plane(p[inliers])
    if (prior is not None and n2 @ prior < 0) or (prior is None and n2 @ n < 0):
        n2, d2 = -n2, -d2
    inliers = np.abs(p @ n2 + d2) < threshold
    return n2, d2, inliers
