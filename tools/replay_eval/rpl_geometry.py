"""Poses, rotations and the Sim(3) alignment from the map frame to the simulator world."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

Arr = npt.NDArray[np.float64]


def mat4(value: Any) -> Arr:
    """A 4x4 pose from nested rows (or 16 numbers, row-major)."""
    m = np.asarray(value, dtype=np.float64)
    if m.shape == (16,):
        m = m.reshape(4, 4)
    if m.shape != (4, 4):
        raise ValueError(f"expected a 4x4 pose, got shape {m.shape}")
    return m


def make_T(R: Arr, t: Arr) -> Arr:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv_T(T: Arr) -> Arr:
    R = T[:3, :3]
    return make_T(R.T, -R.T @ T[:3, 3])


def rot_z(yaw_rad: float) -> Arr:
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_to_R(q: Any, order: str = "wxyz") -> Arr:
    """Rotation matrix of a quaternion given in ``order`` (``wxyz`` or ``xyzw``)."""
    v = [float(x) for x in q]
    if len(v) != 4:
        raise ValueError(f"expected a quaternion of 4 numbers, got {len(v)}")
    w, x, y, z = v if order == "wxyz" else (v[3], v[0], v[1], v[2])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def yaw_of_R(R: Arr) -> float:
    """Yaw (rad, CCW from +x) of a body frame whose forward axis is x."""
    return math.atan2(float(R[1, 0]), float(R[0, 0]))


def optical_yaw_deg(R: Arr) -> float:
    """Heading (deg, CCW from +x) of an optical frame: its z axis projected on the plane."""
    return math.degrees(math.atan2(float(R[1, 2]), float(R[0, 2])))


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def angle_deg(Ra: Arr, Rb: Arr) -> float:
    """Angle (deg) of the rotation between two rotation matrices."""
    c = (float(np.trace(Ra.T @ Rb)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


@dataclass(frozen=True)
class Sim3:
    """world = s R map + t."""

    s: float
    R: Arr
    t: Arr

    def point(self, p: Any) -> Arr:
        return self.s * (self.R @ np.asarray(p, dtype=np.float64)) + self.t

    def points(self, P: Arr) -> Arr:
        return self.s * (P @ self.R.T) + self.t

    def inverse_point(self, p: Any) -> Arr:
        return (self.R.T @ (np.asarray(p, dtype=np.float64) - self.t)) / self.s

    def pose(self, T: Arr) -> Arr:
        """A map-frame pose in the world (rotation rotated, position transformed)."""
        return make_T(self.R @ T[:3, :3], self.point(T[:3, 3]))

    @property
    def yaw_deg(self) -> float:
        return math.degrees(yaw_of_R(self.R))


def umeyama(src: Arr, dst: Arr, min_ratio: float = 1e-3) -> tuple[Sim3, float]:
    """Least-squares similarity dst ~ s R src + t and its RMS residual.

    Both frames are gravity-aligned (z up), so when the points are nearly collinear the full 3-D
    rotation is ill-posed and a yaw-only fit is used instead.
    """
    if len(src) != len(dst) or len(src) < 2:
        raise ValueError("need at least two point pairs")
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - mu_s, dst - mu_d
    var_s = float((xs ** 2).sum() / len(src))
    if var_s <= 1e-12:
        raise ValueError("the source points coincide")
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    sv = np.linalg.svd(xs, compute_uv=False)
    if len(src) >= 3 and sv[1] > min_ratio * sv[0]:
        S = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[2, 2] = -1.0
        R = U @ S @ Vt
        s = float(np.trace(np.diag(D) @ S) / var_s)
    else:
        num = float((xs[:, 0] * xd[:, 1] - xs[:, 1] * xd[:, 0]).sum())
        den = float((xs[:, 0] * xd[:, 0] + xs[:, 1] * xd[:, 1]).sum())
        R = rot_z(math.atan2(num, den))
        s = float((xd * (xs @ R.T)).sum() / (xs ** 2).sum())
    t = mu_d - s * R @ mu_s
    sim = Sim3(s, R, t)
    rms = float(np.sqrt(((sim.points(src) - dst) ** 2).sum(axis=1).mean()))
    return sim, rms


# optical axes in a link frame (x forward, y left, z up): optical x = -y, y = -z, z = x
R_LINK_OPT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def rigid_fit(src: Arr, dst: Arr, R0: Arr | None = None, min_ratio: float = 1e-3) -> Arr:
    """The rigid 4x4 T with dst ~ T src (Kabsch). When the points are nearly collinear or fewer
    than three, only the translation is fitted, keeping rotation ``R0`` (default identity)."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - mu_s, dst - mu_d
    sv = np.linalg.svd(xs, compute_uv=False)
    if len(src) < 3 or sv[1] <= min_ratio * max(float(sv[0]), 1e-12):
        R0 = np.eye(3) if R0 is None else R0
        return make_T(R0, mu_d - R0 @ mu_s)
    U, _, Vt = np.linalg.svd(xd.T @ xs)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    return make_T(R, mu_d - R @ mu_s)


def project_rotation(M: Arr) -> Arr:
    """The rotation nearest a 3x3 matrix (Frobenius)."""
    U, _, Vt = np.linalg.svd(M)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1.0
    return np.asarray(U @ S @ Vt, dtype=np.float64)


def mean_rotation(Rs: list[Arr]) -> tuple[Arr, float]:
    """A robust mean of rotations (chordal mean, refitted without those farther than
    max(3 x the median angle, 1 deg)) and the median angle (deg) of all of them to it."""
    M = project_rotation(np.sum(Rs, axis=0))
    ang = [angle_deg(M, R) for R in Rs]
    keep = [R for R, a in zip(Rs, ang, strict=True) if a <= max(3 * float(np.median(ang)), 1.0)]
    M = project_rotation(np.sum(keep, axis=0))
    return M, float(np.median([angle_deg(M, R) for R in Rs]))
