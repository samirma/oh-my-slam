"""Rigid/similarity transforms, quaternions (scalar last), camera projection and point utilities.

Quaternions are ``(qx, qy, qz, qw)`` as in the OpenLABEL cuboid, normalised with ``qw >= 0``.
Cameras use OpenCV axes. All functions are pure NumPy.
"""

from __future__ import annotations

import math
import os
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

# The edge returned is the smallest that fits to within this factor: the search ends with a counted
# edge that does not fit less than 2 % below it (finer is meaningless: at that scale the grid's
# alignment alone moves the count non-monotonically).
BUDGET_EDGE_TOL = 0.02
# While unbracketed, the secant aims this fraction past the budget, so that it lands on the far side.
_BUDGET_AIM = 0.0025
_BUDGET_CHUNK = 1 << 20  # points whose voxel codes are computed at a time (small temporaries)
_BUDGET_MAX_STEPS = 64
_MIN_EDGE_BITS = 60  # no grid finer than extent / 2**60: at that scale only distinct places count


@dataclass(frozen=True)
class BudgetGrid:
    """The grid ``budget_voxel_grid`` chose: voxel ``edge`` (metres; 0 = no grid, every finite
    point or every distinct place fits), the ``count`` of points it keeps, the tightest counted
    edge below it that did not fit (``finer``, with ``finer_count`` voxels; None when none was
    counted), and how many occupancy ``counts`` the search took."""

    edge: float
    count: int
    finer: float | None
    finer_count: int | None
    counts: int


def _voxel_codes(points: NDArray[Any], edge: float,
                 bounds: tuple[NDArray[Any], NDArray[Any]] | None = None) -> NDArray[np.int64]:
    """The voxel of every point (``voxel_keys``: the grid of edge ``edge`` anchored at the origin)
    as one int64 each (equal codes <=> same voxel), computed chunk by chunk on a few threads
    (NumPy releases the GIL); ``(N, 3)`` keys when the grid's extent does not fit one int64.
    ``bounds``: the points' (min, max), when already known."""
    mn, mx = bounds if bounds is not None else (points.min(axis=0), points.max(axis=0))
    lo = np.floor(np.asarray(mn, np.float64) / edge).astype(np.int64)
    hi = np.floor(np.asarray(mx, np.float64) / edge).astype(np.int64)
    span = [int(s) for s in hi - lo + 1]
    if span[0] * span[1] * span[2] >= 2**62:
        return voxel_keys(points, edge)
    codes = np.empty(len(points), np.int64)

    def chunk(s: int) -> None:
        k = voxel_keys(points[s:s + _BUDGET_CHUNK], edge) - lo
        codes[s:s + _BUDGET_CHUNK] = (k[:, 0] * span[1] + k[:, 1]) * span[2] + k[:, 2]

    starts = range(0, len(points), _BUDGET_CHUNK)
    if len(starts) > 2:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
            list(pool.map(chunk, starts))
    else:
        for s in starts:
            chunk(s)
    return codes


def _runs(codes: NDArray[np.int64]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """(order, starts): an unstable argsort of 1-D codes (faster than a sort on a map's codes, and
    reused for the selection) and where each run of equal codes begins in it."""
    order = np.argsort(codes)
    sorted_codes = codes[order]
    return order, np.flatnonzero(np.r_[True, sorted_codes[1:] != sorted_codes[:-1]])


def _occupied(points: NDArray[Any], edge: float) -> int:
    """Occupied voxels of the grid of edge ``edge``."""
    codes = _voxel_codes(np.asarray(points).reshape(-1, 3), edge)
    if codes.ndim == 2:
        return len(np.unique(codes, axis=0))
    return len(_runs(codes)[1]) if len(codes) else 0


def _first_per_voxel(codes: NDArray[np.int64],
                     runs: tuple[NDArray[np.int64], NDArray[np.int64]] | None = None
                     ) -> NDArray[np.int64]:
    """Index of the first point of each voxel, ascending: the smallest index of each run of equal
    codes (``runs``: ``_runs(codes)`` when already known)."""
    if codes.ndim == 2:
        _, first = np.unique(codes, axis=0, return_index=True)
        return np.sort(first).astype(np.int64)
    order, starts = runs if runs is not None else _runs(codes)
    keep = np.zeros(len(codes), bool)
    keep[np.minimum.reduceat(order, starts)] = True
    return np.flatnonzero(keep).astype(np.int64)


def _distinct_places(points: NDArray[Any]) -> NDArray[np.int64]:
    """Index of the first point at each distinct position, ascending."""
    _, first = np.unique(points, axis=0, return_index=True)
    return np.sort(first).astype(np.int64)


def _finite(points: NDArray[Any]) -> tuple[NDArray[Any], NDArray[np.int64] | None]:
    """The points with finite coordinates, and their rows (None: all of them)."""
    ok = np.isfinite(points).all(axis=1)
    if ok.all():
        return points, None
    rows = np.flatnonzero(ok)
    return points[rows], rows


def budget_voxel_grid(points: NDArray[Any], max_points: int,
                      edge_tol: float = BUDGET_EDGE_TOL) -> tuple[BudgetGrid, NDArray[np.int64]]:
    """(grid, indices): the selection that draws at most ``max_points`` of ``points`` — one
    original point per occupied voxel (the first in the points' order), ascending — for the
    smallest voxel edge whose grid has at most ``max_points`` occupied voxels. Points with a
    non-finite coordinate are never selected (they cannot be drawn).

    * Every finite point when there are at most ``max_points`` of them (edge 0).
    * Otherwise the grid is searched: each step counts the occupied voxels of one edge (an
      argsort of the points' voxel codes, reused for the selection itself). Steps are secants of
      logit(count / points) against log(edge) — nearly straight both far from and close to every
      point having its own voxel — aimed just past the budget until both sides are counted, then
      regula falsi (Illinois) inside the counted bracket.
    * The search ends when the bracket is tight: the returned edge was counted with at most
      ``max_points`` voxels, and ``grid.finer``, less than ``edge_tol`` (2 %) below it, was counted
      with more (the count is not strictly monotonic at that scale, so an edge in between may
      still fit). The result never exceeds ``max_points``.
    * When the points are at no more than ``max_points`` distinct places (duplicates), no grid
      thins them further: one point per place, edge 0 (``finer`` None).
    * A ``ValueError`` when no grid holds them in ``max_points`` voxels: points around the origin
      occupy up to 8 voxels of every grid (the grids are anchored there)."""
    if max_points < 1:
        raise ValueError("max_points must be at least 1")
    pts, rows = _finite(np.asarray(points).reshape(-1, 3))

    def out(idx: NDArray[np.int64]) -> NDArray[np.int64]:
        return idx if rows is None else rows[idx]

    n = len(pts)
    if n <= max_points:
        return BudgetGrid(0.0, n, None, None, 0), out(np.arange(n, dtype=np.int64))
    extent = float(np.max(pts.max(axis=0).astype(np.float64) - pts.min(axis=0)))
    floor = math.log(extent) - _MIN_EDGE_BITS * math.log(2.0) if extent > 0 else math.inf

    counted: dict[float, int] = {}  # log(edge) -> occupied voxels
    last: list[Any] = [None, None, None]  # log(edge), codes, runs of the latest count
    bounds = (pts.min(axis=0), pts.max(axis=0))

    def count(x: float) -> int:  # each edge is counted once: every step is a new one
        last[:] = [None, None, None]  # free the previous count's arrays first
        codes = _voxel_codes(pts, math.exp(x), bounds)
        runs = _runs(codes) if codes.ndim == 1 else None
        counted[x] = len(runs[1]) if runs is not None else len(np.unique(codes, axis=0))
        last[:] = [x, codes, runs]
        return counted[x]

    tight = math.log1p(edge_tol)

    def y(c: float) -> float:  # logit of the kept fraction: straight-ish in log(edge)
        return math.log(c) - math.log(n + 0.5 - c)

    def places() -> tuple[BudgetGrid, NDArray[np.int64]]:
        """One point per distinct place: all points at one place, or no grid fit at all."""
        idx = _distinct_places(pts)
        if len(idx) > max_points:  # points around the origin occupy up to 8 voxels of any grid
            raise ValueError(f"no voxel grid holds these points in {max_points} voxels")
        return BudgetGrid(0.0, len(idx), None, None, len(counted)), out(idx)

    def fitting() -> tuple[float, int]:
        x = min(k for k, c in counted.items() if c <= max_points)
        return x, counted[x]

    def result(x: float, c: int) -> tuple[BudgetGrid, NDArray[np.int64]]:
        fail = [k for k, v in counted.items() if v > max_points and k < x]
        finer = max(fail) if fail else None
        if last[0] == x:
            idx = _first_per_voxel(last[1], last[2])
        else:
            idx = _first_per_voxel(_voxel_codes(pts, math.exp(x), bounds))
        last[:] = [None, None, None]
        return (BudgetGrid(math.exp(x), c, None if finer is None else math.exp(finer),
                           None if finer is None else counted[finer], len(counted)), out(idx))

    if extent == 0.0:
        return places()
    x = math.log(extent / math.sqrt(max_points))  # max_points voxels tiling the extent's square
    side = 0  # Illinois: which end the last step moved
    tried_places = False
    for _ in range(_BUDGET_MAX_STEPS):
        c = count(x)
        fits = [k for k, v in counted.items() if v <= max_points]
        fails = [k for k, v in counted.items() if v > max_points]
        hi = min(fits) if fits else None  # finest edge that fits
        lo = max((k for k in fails if hi is None or k < hi), default=None)  # coarsest that fails
        if hi is not None and lo is None:
            finest = sorted(fits)[:2]
            if x <= floor or (not tried_places and len(finest) == 2
                              and counted[finest[0]] == counted[finest[1]]):
                # finer grids keep no more points: perhaps duplicates only
                tried_places = True
                idx = _distinct_places(pts)
                if len(idx) <= max_points:
                    return BudgetGrid(0.0, len(idx), None, None, len(counted)), out(idx)
                if x <= floor:
                    return result(*fitting())
        if hi is not None and lo is not None:
            width = hi - lo
            if width <= tight:
                return result(*fitting())
            f_hi, f_lo = y(counted[hi]) - y(max_points), y(counted[lo]) - y(max_points)
            moved = 1 if c <= max_points else -1
            if moved == side:  # the same end moved twice: halve the other end's weight
                if moved == 1:
                    f_lo /= 2
                else:
                    f_hi /= 2
            side = moved
            nx = lo + width * f_lo / (f_lo - f_hi) if f_lo != f_hi else (lo + hi) / 2
            nx = min(max(nx, lo + 0.01 * width), hi - 0.01 * width)
            # close to an end: step just inside the tolerance from it, so that one count closes it
            if hi - nx < 0.9 * tight:
                nx = hi - 0.9 * tight
            elif nx - lo < 0.9 * tight:
                nx = lo + 0.9 * tight
        else:
            # one side only: a secant through the last two counts (slope -2 to begin with)
            ks = sorted(counted)
            slope = -2.0
            if len(ks) >= 2:
                a, b = sorted(ks, key=lambda k: abs(k - x))[:2]  # the two counts nearest
                if counted[a] != counted[b] and a != b:
                    slope = min((y(counted[a]) - y(counted[b])) / (a - b), -0.25)
            aim = (min(max_points * (1 + _BUDGET_AIM), (max_points + n) / 2) if c <= max_points
                   else max_points * (1 - _BUDGET_AIM))
            nx = x + (y(aim) - y(c)) / slope
            if c <= max_points:  # at least a tolerance finer, to find a finer edge that fails
                nx = min(nx, x - 0.9 * tight)
            step = math.log(64.0)
            nx = min(max(nx, x - step), x + step)  # at most 64 x per step while unbracketed
            nx = max(nx, floor)
        x = nx
    return result(*fitting()) if any(v <= max_points for v in counted.values()) else places()


def budget_voxel_indices(points: NDArray[Any], max_points: int, edge: float | None = None
                         ) -> tuple[NDArray[np.int64], float]:
    """(indices, edge) of ``budget_voxel_grid``; with ``edge`` (one it found before for the same
    points and budget) the selection at that edge, without a search."""
    if edge is None:
        grid, idx = budget_voxel_grid(points, max_points)
        return idx, grid.edge
    pts, rows = _finite(np.asarray(points).reshape(-1, 3))
    if edge == 0.0:
        idx = np.arange(len(pts), dtype=np.int64) if len(pts) <= max_points else _distinct_places(pts)
    else:
        idx = _first_per_voxel(_voxel_codes(pts, edge))
    return (idx if rows is None else rows[idx]), edge


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
