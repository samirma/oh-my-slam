"""Rigid/similarity transforms, quaternions (scalar last), camera projection and point utilities.

Quaternions are ``(qx, qy, qz, qw)`` as in the OpenLABEL cuboid, normalised with ``qw >= 0``.
Cameras use OpenCV axes. All functions are pure NumPy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

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
    if size % 2 == 0:  # an even window is off-centre: ndimage's convention
        dmax = ndimage.maximum_filter(fill_hi, size=size, mode="nearest")
        dmin = ndimage.minimum_filter(fill_lo, size=size, mode="nearest")
        touching_invalid = ndimage.binary_dilation(~valid, structure=np.ones((size, size), bool))
    else:  # the same windows, as shifted views (several times faster)
        dmax = _window(fill_hi, size, np.maximum, "edge")
        dmin = _window(fill_lo, size, np.minimum, "edge")
        touching_invalid = _window(~valid, size, np.logical_or, "constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = (dmax - dmin) / big
    edges = np.asarray(rel > rel_threshold)
    # a valid pixel touching an invalid one is also an edge
    return np.asarray((edges | touching_invalid) & valid)


def _window(a: NDArray[Any], size: int, op: Any, mode: Literal["edge", "constant"]
            ) -> NDArray[Any]:
    """``op`` (an associative ufunc: maximum, minimum, logical_or) over the ``size`` x ``size``
    window centred on each element (``size`` odd), beyond the border as ``np.pad``'s ``mode``
    extends the array ("edge": ndimage's "nearest"; "constant": zeros / False, a binary
    dilation's border): ndimage's maximum/minimum filter and binary dilation, separably."""
    r = size // 2
    p = np.pad(a, r, mode=mode)
    h, w = a.shape
    rows = p[0:h]
    for k in range(1, size):
        rows = op(rows, p[k:k + h])
    out = rows[:, 0:w]
    for k in range(1, size):
        out = op(out, rows[:, k:k + w])
    return np.asarray(out)


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


def unique_rows(keys: NDArray[np.int64]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """``np.unique(keys, axis=0, return_inverse=True)`` of (N, 3) integer keys (the rows in
    lexicographic order, and the row of each key), through one int64 per key (``_packed``)."""
    keys = np.asarray(keys, np.int64)
    packed = _packed(keys) if len(keys) else keys
    if packed.ndim == 2:
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        return uniq, inv.reshape(-1)
    _, first, inv = np.unique(packed, return_index=True, return_inverse=True)
    return keys[first], inv.reshape(-1)


def _packed(keys: NDArray[np.int64]) -> NDArray[np.int64]:
    """(N, 3) voxel keys as one int64 per voxel (a 1-D unique sorts far faster than rows), or the
    rows themselves when the key range does not fit. Equal rows <=> equal packed values."""
    lo = keys.min(axis=0)
    span = (keys.max(axis=0) - lo + 1).astype(object)  # Python ints: no overflow in the check
    if span[0] * span[1] * span[2] >= 2**62:
        return keys
    k = keys - lo
    return (k[:, 0] * int(span[1] * span[2]) + k[:, 1] * int(span[2]) + k[:, 2]).astype(np.int64)


# --- voxel-grid selection within a point budget --------------------------------------------------

# The edge found is within a factor 1 + this of the smallest one that fits. Finer means nothing: at
# that scale the grid's alignment alone moves the count by about 0.01 % between nearby edges.
BUDGET_EDGE_REL_TOL = 1e-2
_BUDGET_CHUNK = 1 << 20  # points whose voxel codes are computed at a time (small temporaries)


def _voxel_codes(points: NDArray[Any], edge: float) -> NDArray[np.int64]:
    """The voxel of every point (``voxel_keys``: the grid of edge ``edge`` anchored at the origin)
    as one int64 each (equal codes <=> same voxel), computed chunk by chunk; ``(N, 3)`` keys when
    the grid's extent does not fit one int64."""
    lo = np.floor(points.min(axis=0).astype(np.float64) / edge).astype(np.int64)
    hi = np.floor(points.max(axis=0).astype(np.float64) / edge).astype(np.int64)
    span = [int(s) for s in hi - lo + 1]
    if span[0] * span[1] * span[2] >= 2**62:
        return voxel_keys(points, edge)
    codes = np.empty(len(points), np.int64)
    for s in range(0, len(points), _BUDGET_CHUNK):
        k = voxel_keys(points[s:s + _BUDGET_CHUNK], edge) - lo
        codes[s:s + _BUDGET_CHUNK] = (k[:, 0] * span[1] + k[:, 1]) * span[2] + k[:, 2]
    return codes


def _occupied(points: NDArray[Any], edge: float) -> int:
    codes = _voxel_codes(points, edge)
    if codes.ndim == 2:
        return len(np.unique(codes, axis=0))
    codes.sort()
    return 1 + int(np.count_nonzero(codes[1:] != codes[:-1])) if len(codes) else 0


def budget_voxel_edge(points: NDArray[Any], max_points: int,
                      rel_tol: float = BUDGET_EDGE_REL_TOL) -> float:
    """The smallest voxel edge (metres, to within a factor ``1 + rel_tol``) whose grid
    (``voxel_keys``) has at most ``max_points`` occupied voxels; 0 when the points need no
    thinning (at most ``max_points`` of them, or no more distinct places than that).

    Each step counts the occupied voxels of one edge (a sort of the points' voxel codes). The
    search brackets the edge by factors of 4, closes the bracket by Brent's method on the count
    against log(edge) — a handful of counts where bisection would take a dozen — and returns the
    smallest edge counted that fits, with a count above ``max_points`` less than ``rel_tol`` below
    it."""
    from scipy.optimize import brentq

    pts = np.asarray(points).reshape(-1, 3)
    if len(pts) <= max_points:
        return 0.0
    if max_points < 1:
        raise ValueError("max_points must be at least 1")
    extent = float(np.max(pts.max(axis=0).astype(np.float64) - pts.min(axis=0)))
    if extent == 0.0:
        return 0.0  # a single place: one voxel at any edge
    counted: dict[float, int] = {}  # log(edge) -> occupied voxels

    def f(x: float) -> float:  # > 0: too many voxels
        if x not in counted:
            counted[x] = _occupied(pts, math.exp(x))
        return counted[x] - (max_points + 0.5)

    # bracket by factors of 4 from the edge at which max_points voxels would tile the extent's
    # square (a scene's points lie on surfaces)
    x, step = math.log(extent / math.sqrt(max_points)), math.log(4.0)
    if f(x) > 0:
        while f(x) > 0:
            x += step
    else:
        floor = math.log(extent * 1e-9)  # below this the points are at no more distinct places
        while f(x) <= 0:
            if x < floor:
                return 0.0
            x -= step
    width = math.log1p(rel_tol)

    def bracket() -> tuple[float, float]:
        """The tightest counted bracket: the smallest fitting edge, the largest one below it
        that does not fit."""
        hi = min(k for k, n in counted.items() if n <= max_points)
        return max(k for k, n in counted.items() if n > max_points and k < hi), hi

    lo, hi = bracket()
    if hi - lo > width:
        brentq(f, lo, hi, xtol=width / 4, rtol=4 * np.finfo(float).eps)
    lo, hi = bracket()
    while hi - lo > width:  # Brent's last estimate may sit on one side only
        f((lo + hi) / 2)
        lo, hi = bracket()
    return math.exp(hi)


def budget_voxel_indices(points: NDArray[Any], max_points: int,
                         rel_tol: float = BUDGET_EDGE_REL_TOL) -> tuple[NDArray[np.int64], float]:
    """(indices, edge): one original point per occupied voxel of the grid of edge
    ``budget_voxel_edge`` — the first in the points' order — in ascending order, so at most
    ``max_points`` of them; every index and an edge of 0 when no thinning is needed. Points are
    selected, never merged: each keeps its own attributes."""
    pts = np.asarray(points).reshape(-1, 3)
    edge = budget_voxel_edge(pts, max_points, rel_tol)
    if edge == 0.0:
        if len(pts) <= max_points:
            return np.arange(len(pts), dtype=np.int64), 0.0
        _, first = np.unique(pts, axis=0, return_index=True)  # duplicates only: one per place
        return np.sort(first).astype(np.int64), 0.0
    codes = _voxel_codes(pts, edge)
    if codes.ndim == 2:
        _, first = np.unique(codes, axis=0, return_index=True)
        return np.sort(first).astype(np.int64), edge
    # the first point of each voxel: the smallest index of each run of equal codes (an unstable
    # argsort and a reduction are several times faster than a stable sort)
    order = np.argsort(codes)
    sorted_codes = codes[order]
    starts = np.flatnonzero(np.r_[True, sorted_codes[1:] != sorted_codes[:-1]])
    return np.sort(np.minimum.reduceat(order, starts)).astype(np.int64), edge


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
