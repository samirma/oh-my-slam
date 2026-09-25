"""Depth alignment: robust per-frame scale against sparse SfM depths and a global metric scale."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

MIN_POINTS = 50
MAD_K = 3.0


@dataclass
class ScaleFit:
    scale: float  # multiply the first argument by this to match the second
    inliers: int
    spread: float  # IQR / median of the per-point ratios (inliers)

    @property
    def ok(self) -> bool:
        return self.inliers >= MIN_POINTS and np.isfinite(self.scale) and self.scale > 0


def robust_ratio(num: NDArray[Any], den: NDArray[Any], min_points: int = MIN_POINTS) -> ScaleFit:
    """Median of ``num/den`` after 3-MAD outlier rejection (in log space)."""
    a = np.asarray(num, dtype=np.float64)
    b = np.asarray(den, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if ok.sum() < max(3, min_points):
        return ScaleFit(float("nan"), int(ok.sum()), float("inf"))
    r = np.log(a[ok] / b[ok])
    med = np.median(r)
    mad = np.median(np.abs(r - med)) * 1.4826 + 1e-9
    keep = np.abs(r - med) <= MAD_K * mad
    rk = r[keep]
    ratios = np.exp(rk)
    med_ratio = float(np.exp(np.median(rk)))
    q75, q25 = np.percentile(ratios, [75, 25])
    return ScaleFit(med_ratio, int(keep.sum()), float((q75 - q25) / med_ratio))


def fit_frame_scale(pred_depth: NDArray[Any], ref_depth: NDArray[Any],
                    min_points: int = MIN_POINTS) -> ScaleFit:
    """Scale s so that ``s * pred ≈ ref`` for sparse samples at the same pixels."""
    return robust_ratio(ref_depth, pred_depth, min_points)


def global_scale(per_frame: list[float]) -> tuple[float, float]:
    """Median of per-frame ratios and IQR/median (spread)."""
    v = np.asarray([s for s in per_frame if np.isfinite(s) and s > 0], dtype=np.float64)
    if len(v) == 0:
        return float("nan"), float("inf")
    med = float(np.median(v))
    q75, q25 = np.percentile(v, [75, 25])
    return med, float((q75 - q25) / med)


def dense_scale(
    depth: NDArray[Any],
    K: NDArray[Any],
    T_world_cam: NDArray[Any],
    refs: list[tuple[NDArray[Any], NDArray[Any], NDArray[Any]]],
    step: int = 6,
    iterations: int = 2,
    min_points: int = 500,
) -> ScaleFit:
    """Scale s so that ``s * depth`` agrees with already-aligned reference depth maps.

    Pixels of ``depth`` (every ``step``) are lifted with the current scale, projected into each
    reference (depth, K, T_world_cam) and compared with the depth that reference observed along
    the same ray; the robust median ratio updates s (two iterations absorb small baselines).
    Used for keyframes with too few triangulated points for a sparse fit.
    """
    d = np.asarray(depth, np.float64)
    h, w = d.shape
    vv, uu = np.mgrid[step // 2:h:step, step // 2:w:step]
    z0 = d[vv, uu]
    ok = z0 > 0
    u, v, z0 = uu[ok].astype(np.float64), vv[ok].astype(np.float64), z0[ok]
    Kinv_rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], 1)
    Rw, tw = np.asarray(T_world_cam)[:3, :3], np.asarray(T_world_cam)[:3, 3]
    s = 1.0
    fit = ScaleFit(float("nan"), 0, float("inf"))
    for _ in range(iterations):
        pts = (Kinv_rays * (s * z0)[:, None]) @ Rw.T + tw
        num, den = [], []
        for rd, rK, rT in refs:
            Rc = np.asarray(rT)[:3, :3].T
            tc = -Rc @ np.asarray(rT)[:3, 3]
            pc = pts @ Rc.T + tc
            zc = pc[:, 2]
            front = zc > 0.05
            with np.errstate(divide="ignore", invalid="ignore"):
                pu = np.rint(rK[0, 0] * pc[:, 0] / zc + rK[0, 2])
                pv = np.rint(rK[1, 1] * pc[:, 1] / zc + rK[1, 2])
            rh, rw = rd.shape
            inside = front & (pu >= 0) & (pu < rw) & (pv >= 0) & (pv < rh)
            if inside.sum() < 50:
                continue
            obs = rd[pv[inside].astype(int), pu[inside].astype(int)]
            good = obs > 0
            num.append(obs[good])
            den.append(zc[inside][good])
        if not num:
            return fit
        fit = robust_ratio(np.concatenate(num), np.concatenate(den), min_points)
        if not fit.ok:
            return fit
        s *= fit.scale
    return ScaleFit(s, fit.inliers, fit.spread)


# ------------------------------------------------------------------------------------------------
# global adjustment of per-keyframe depth scales
#
# ``dense_scale`` aligns a keyframe to a few already-aligned neighbours, one keyframe after another:
# the scale error of each step accumulates along a long sequence, and where a loop closes the last
# keyframes disagree with the first ones by 10-20 %. ``adjust_log_scales`` instead solves for one
# scale correction per keyframe from the depth ratios of *all* overlapping keyframe pairs at once
# (sequence neighbours and loop closures alike), so the loop's disagreement is spread thinly over
# the whole loop instead of piling up where it closes.

PAIR_STEP = 8  # every 8th pixel of the source grid (both axes) is transferred
PAIR_GATE = 0.4  # |log ratio| beyond this (a factor 1.5) is another surface (occlusion)
PAIR_MIN_POINTS = 300  # fewer transferred points in both directions: the pair is not measured
PAIR_FULL_POINTS = 2000  # a pair with at least this many points gets its full weight
# A pair's weight is its overlap over its spread², the spread floored at the depth noise: a pair is
# never more reliable than monocular depth, and pairs tighter than that (narrow overlaps at wide
# angles) must not outvote the others; a wider spread (a disagreement no single scale reconciles)
# lowers the weight.
SPREAD_FLOOR = 0.05
# graduated Cauchy scale (log units) of the pair residuals: plain least squares first, so the loop
# closure's large initial residuals are distributed rather than rejected as outliers, then
# shrinking to the depth noise so pairs that no single scale reconciles lose their pull
ROBUST_SCALES = (float("inf"), 0.1, 0.05)
RIDGE = 1e-6  # relative pull of every correction towards 0 (keyframes without pairs keep theirs)


@dataclass
class DepthView:
    """A keyframe's metric z-depth grid (0 where invalid or unreliable), grid intrinsics ``K``
    and camera-to-world pose ``T_world_cam`` (4x4)."""

    depth: NDArray[Any]
    K: NDArray[Any]
    T_world_cam: NDArray[Any]
    step: int = PAIR_STEP
    _samples: NDArray[np.float64] | None = None

    def samples(self) -> NDArray[np.float64]:
        """Camera-frame points of every ``step``-th valid pixel (cached)."""
        if self._samples is None:
            d = np.asarray(self.depth, np.float64)
            h, w = d.shape
            vv, uu = np.mgrid[self.step // 2:h:self.step, self.step // 2:w:self.step]
            z = d[vv, uu]
            ok = np.isfinite(z) & (z > 0)
            u, v, z = uu[ok].astype(np.float64), vv[ok].astype(np.float64), z[ok]
            K = self.K
            self._samples = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z,
                                      z], 1)
        return self._samples


@dataclass(frozen=True)
class PairRatio:
    """How much deeper keyframe ``a`` places the surfaces both keyframes see than ``b`` does:
    ``log_ratio`` ≈ log(s_a / s_b) for per-keyframe scale errors s (symmetric: half the
    difference of the a→b and b→a transfers), the robust spread of the per-pixel log ratios, and
    the number of transferred points."""

    log_ratio: float
    spread: float
    points: int


def _transfer_log_ratios(src: DepthView, dst: DepthView, scale_src: float, scale_dst: float
                         ) -> NDArray[np.float64]:
    """log(z / d) of ``src``'s sampled points (depth times ``scale_src``) in ``dst``'s camera:
    z their depth there, d ``dst``'s depth (times ``scale_dst``) at the pixel they land on; same
    surface only (|log| <= ``PAIR_GATE``)."""
    pts = src.samples()
    if not len(pts):
        return np.zeros(0)
    Ts, Td = np.asarray(src.T_world_cam), np.asarray(dst.T_world_cam)
    R = Td[:3, :3].T @ Ts[:3, :3]
    t = Td[:3, :3].T @ (Ts[:3, 3] - Td[:3, 3])
    pc = (scale_src * pts) @ R.T + t
    z = pc[:, 2]
    K = dst.K
    front = z > 0.05
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.rint(K[0, 0] * pc[:, 0] / z + K[0, 2])
        v = np.rint(K[1, 1] * pc[:, 1] / z + K[1, 2])
    h, w = np.shape(dst.depth)
    inside = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    d = np.asarray(dst.depth)[v[inside].astype(np.int64), u[inside].astype(np.int64)]
    good = np.isfinite(d) & (d > 0)
    r = np.log(z[inside][good] / (scale_dst * d[good].astype(np.float64)))
    return r[np.abs(r) <= PAIR_GATE]


def _robust_centre(r: NDArray[np.float64]) -> tuple[float, float]:
    """Median and 1.4826 MAD after 3-MAD rejection."""
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med))) * 1.4826 + 1e-9
    keep = r[np.abs(r - med) <= MAD_K * mad]
    med = float(np.median(keep))
    return med, float(np.median(np.abs(keep - med))) * 1.4826


def pair_log_ratio(a: DepthView, b: DepthView, scale_a: float = 1.0, scale_b: float = 1.0,
                   min_points: int = PAIR_MIN_POINTS) -> PairRatio | None:
    """``PairRatio`` of two keyframes with their depths multiplied by ``scale_a`` / ``scale_b``;
    None when neither direction transfers ``min_points`` points onto the same surface."""
    ab = _transfer_log_ratios(a, b, scale_a, scale_b)
    ba = _transfer_log_ratios(b, a, scale_b, scale_a)
    parts = [(r, sign) for r, sign in ((ab, 1.0), (ba, -1.0)) if len(r) >= min_points]
    if not parts:
        return None
    centres = [(sign * m, s) for (r, sign) in parts for m, s in [_robust_centre(r)]]
    return PairRatio(float(np.mean([m for m, _ in centres])),
                     float(np.mean([s for _, s in centres])),
                     int(sum(len(r) for r, _ in parts)))


def pair_weight(p: PairRatio) -> float:
    """Weight of a pair's log ratio: its overlap (up to ``PAIR_FULL_POINTS``) over its spread²."""
    return min(1.0, p.points / PAIR_FULL_POINTS) / (p.spread ** 2 + SPREAD_FLOOR ** 2)


@dataclass
class ScaleAdjustment:
    log_scale: NDArray[np.float64]  # correction per keyframe: multiply its depth by exp(x)
    residuals_before: NDArray[np.float64]  # |log ratio| per pair before / after
    residuals_after: NDArray[np.float64]


def adjust_log_scales(n: int, pairs: list[tuple[int, int, float, float]], fixed: set[int],
                      scales: tuple[float, ...] = ROBUST_SCALES, iterations: int = 10
                      ) -> ScaleAdjustment:
    """Per-keyframe log scale corrections ``x`` (``x[k] = 0`` for ``k`` in ``fixed``) minimising
    Σ w ρ(m + x_i − x_j) over the ``pairs`` (i, j, m = measured log ratio of i over j, w), with
    the Cauchy loss ρ at each robust scale of ``scales`` in turn (iteratively reweighted least
    squares; ``inf``: plain least squares) and a tiny ridge towards 0."""
    x = np.zeros(n)
    if not pairs:
        return ScaleAdjustment(x, np.zeros(0), np.zeros(0))
    ia = np.array([p[0] for p in pairs], np.int64)
    ja = np.array([p[1] for p in pairs], np.int64)
    meas = np.array([p[2] for p in pairs], np.float64)
    wts = np.array([p[3] for p in pairs], np.float64)
    free = [k for k in range(n) if k not in fixed]
    if not free:
        r = np.abs(meas)
        return ScaleAdjustment(x, r, r)
    col = np.full(n, -1, np.int64)
    col[free] = np.arange(len(free))
    ci, cj = col[ia], col[ja]
    ridge = RIDGE * float(wts.mean())

    def solve(w: NDArray[np.float64]) -> NDArray[np.float64]:
        H = np.zeros((len(free), len(free)))
        g = np.zeros(len(free))
        for c, sign in ((ci, 1.0), (cj, -1.0)):
            ok = c >= 0
            np.add.at(g, c[ok], -sign * w[ok] * meas[ok])
        for ca, cb, s in ((ci, ci, 1.0), (cj, cj, 1.0), (ci, cj, -1.0), (cj, ci, -1.0)):
            ok = (ca >= 0) & (cb >= 0)
            np.add.at(H, (ca[ok], cb[ok]), s * w[ok])
        H[np.diag_indices_from(H)] += ridge
        out = np.zeros(n)
        out[free] = np.linalg.solve(H, g)
        return out

    for c in scales:
        for _ in range(iterations):
            r = meas + x[ia] - x[ja]
            w = wts if not np.isfinite(c) else wts / (1.0 + (r / c) ** 2)
            x_new = solve(w)
            done = float(np.abs(x_new - x).max()) < 1e-6
            x = x_new
            if done or not np.isfinite(c):
                break
    return ScaleAdjustment(x, np.abs(meas), np.abs(meas + x[ia] - x[ja]))


def sample_depth(depth: NDArray[Any], uv: NDArray[Any]) -> NDArray[np.float64]:
    """Nearest-pixel depth at (u, v) positions; NaN outside the grid or where invalid."""
    d = np.asarray(depth)
    h, w = d.shape
    u = np.rint(uv[:, 0]).astype(np.int64)
    v = np.rint(uv[:, 1]).astype(np.int64)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    out = np.full(len(uv), np.nan)
    vals = d[v[inside], u[inside]].astype(np.float64)
    vals[vals <= 0] = np.nan
    out[inside] = vals
    return out
