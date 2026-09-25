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
# global adjustment of the keyframes' depth
#
# ``dense_scale`` aligns a keyframe to a few already-aligned neighbours, one keyframe after another:
# the scale error of each step accumulates along a long sequence, and where a loop closes the last
# keyframes disagree with the first ones by 10-20 %. Nor is a keyframe's error a single scale:
# monocular depth of the near field (a counter top 0.3-0.7 m away, seen at a grazing angle) differs
# between neighbouring keyframes by 10-30 % where their far surfaces agree within 1-2 %, in either
# direction from one keyframe to the next, so one scale per keyframe that reconciles the near field
# of one pair breaks the far field of another (on the example sequence, a scale-only adjustment left
# keyframes two apart disagreeing by 21 %). ``adjust_depth_corrections`` therefore solves for a
# log-affine correction per keyframe — a scale and a near/far tilt about its median depth
# (``DepthCorrection``) — from the depth ratios of *all* overlapping keyframe pairs at once
# (sequence neighbours and loop closures alike), each measured separately in bins of depth
# (``pair_bins``): the loop's disagreement is spread thinly over the whole loop instead of piling up
# where it closes, and a keyframe's near field is brought to its neighbours' without moving its far
# field.

PAIR_STEP = 8  # every 8th pixel of the source grid (both axes) is transferred
PAIR_GATE = 0.4  # |log ratio| beyond this (a factor 1.5) is another surface (occlusion)
PAIR_MIN_POINTS = 300  # fewer transferred points in a direction: that direction is not measured
PAIR_FULL_POINTS = 2000  # a direction with at least this many points gets its full weight
DEPTH_BINS = 6  # a direction's points are split into this many bins of equal count by depth
BIN_MIN_POINTS = 30
# graduated Cauchy scale (log units) of the bin residuals: plain least squares first, so the loop
# closure's large initial residuals are distributed rather than rejected as outliers, then 20 %,
# so bins that no correction reconciles (another surface behind a thin one, a misplaced keyframe)
# lose their pull
ROBUST_SCALES = (float("inf"), 0.2)
RIDGE = 1e-6  # relative pull of every log scale towards 0 (keyframes without pairs keep theirs)
# The prior "no tilt" (slope 0) weighs as much as SLOPE_PRIOR bins of average weight: a keyframe
# tilts only as far as many bins of its pairs agree (a keyframe of one depth range — a wall seen
# face on — cannot tilt at all), and at most MAX_SLOPE (a correction of ±35 % between 0.4 and 3 m).
SLOPE_PRIOR = 10.0
MAX_SLOPE = 0.3
MAX_FACTOR = 1.5  # a correction never scales a pixel's depth by more than this (or less than 1/it)


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

    def log_median(self) -> float:
        """Log of the median valid depth (0 without any)."""
        d = np.asarray(self.depth)
        d = d[np.isfinite(d) & (d > 0)]
        return float(np.log(np.median(d))) if len(d) else 0.0

    def corrected(self, c: DepthCorrection) -> DepthView:
        """The view with ``c`` applied to its depth (itself when ``c`` is the identity)."""
        if c.identity:
            return self
        return DepthView(c.apply(self.depth).astype(np.float32), self.K, self.T_world_cam,
                         self.step)


@dataclass(frozen=True)
class DepthCorrection:
    """``log d' = log d + log_scale + slope · (log d − pivot)`` (``pivot``: log metres, the
    keyframe's median depth when solved): a scale and a near/far tilt; the factor ``d'/d`` is
    clamped to [1/``MAX_FACTOR``, ``MAX_FACTOR``]."""

    log_scale: float = 0.0
    slope: float = 0.0
    pivot: float = 0.0

    @property
    def identity(self) -> bool:
        return self.log_scale == 0.0 and self.slope == 0.0

    @property
    def scale(self) -> float:
        """The factor at the pivot."""
        return float(np.exp(self.log_scale))

    @property
    def exponent(self) -> float:
        """``d' ∝ d ** exponent``."""
        return 1.0 + self.slope

    def factor(self, depth: Any) -> NDArray[np.float64]:
        """``d'/d`` at each depth (metres; 1 where the depth is not positive)."""
        d = np.asarray(depth, np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.exp(self.log_scale + self.slope * (np.log(np.where(d > 0, d, 1.0)) - self.pivot))
        f = np.clip(f, 1.0 / MAX_FACTOR, MAX_FACTOR)
        return np.asarray(np.where(d > 0, f, 1.0), np.float64)

    def apply(self, depth: Any) -> NDArray[np.float64]:
        """The corrected depth (non-positive depths unchanged)."""
        d = np.asarray(depth, np.float64)
        return np.asarray(d * self.factor(d), np.float64)

    def then(self, other: DepthCorrection) -> DepthCorrection:
        """This correction followed by ``other`` (defined on the corrected depth), about this
        one's pivot (clamps aside)."""
        b1, b2 = self.slope, other.slope
        a = (1 + b2) * (self.pivot + self.log_scale) + other.log_scale - b2 * other.pivot \
            - self.pivot
        return DepthCorrection(float(a), float((1 + b1) * (1 + b2) - 1), self.pivot)


def _transfer(src: DepthView, dst: DepthView, scale_src: float = 1.0, scale_dst: float = 1.0
              ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """``src``'s sampled points (depth times ``scale_src``) in ``dst``'s camera, on the same
    surface as ``dst``'s depth (times ``scale_dst``) at the pixel they land on (|log ratio| <=
    ``PAIR_GATE``): (log(z / d), log of their depth in ``src``, log d) — z their depth in ``dst``'s
    camera, d ``dst``'s depth there."""
    pts = src.samples()
    empty = np.zeros(0)
    if not len(pts):
        return empty, empty, empty
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
    log_d = np.log(scale_dst * d[good].astype(np.float64))
    r = np.log(z[inside][good]) - log_d
    log_src = np.log(scale_src * pts[:, 2][inside][good])
    same = np.abs(r) <= PAIR_GATE
    return r[same], log_src[same], log_d[same]


def transfer_log_ratios(src: DepthView, dst: DepthView, scale_src: float = 1.0,
                        scale_dst: float = 1.0) -> NDArray[np.float64]:
    """log(z / d) of ``src``'s sampled points on the same surface in ``dst`` (``_transfer``)."""
    return _transfer(src, dst, scale_src, scale_dst)[0]


def _robust_centre(r: NDArray[np.float64]) -> tuple[float, float]:
    """Median and 1.4826 MAD after 3-MAD rejection."""
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med))) * 1.4826 + 1e-9
    keep = r[np.abs(r - med) <= MAD_K * mad]
    med = float(np.median(keep))
    return med, float(np.median(np.abs(keep - med))) * 1.4826


@dataclass(frozen=True)
class PairRatio:
    """How much deeper keyframe ``a`` places the surfaces both keyframes see than ``b`` does:
    ``log_ratio`` ≈ log(s_a / s_b) for per-keyframe scale errors s (symmetric: half the
    difference of the a→b and b→a transfers), the robust spread of the per-pixel log ratios, and
    the number of transferred points."""

    log_ratio: float
    spread: float
    points: int


def pair_log_ratio(a: DepthView, b: DepthView, scale_a: float = 1.0, scale_b: float = 1.0,
                   min_points: int = PAIR_MIN_POINTS) -> PairRatio | None:
    """``PairRatio`` of two keyframes with their depths multiplied by ``scale_a`` / ``scale_b``;
    None when neither direction transfers ``min_points`` points onto the same surface."""
    ab = _transfer(a, b, scale_a, scale_b)[0]
    ba = _transfer(b, a, scale_b, scale_a)[0]
    parts = [(r, sign) for r, sign in ((ab, 1.0), (ba, -1.0)) if len(r) >= min_points]
    if not parts:
        return None
    centres = [(sign * m, s) for (r, sign) in parts for m, s in [_robust_centre(r)]]
    return PairRatio(float(np.mean([m for m, _ in centres])),
                     float(np.mean([s for _, s in centres])),
                     int(sum(len(r) for r, _ in parts)))


@dataclass(frozen=True)
class BinRatio:
    """Keyframe ``src``'s sampled points of one depth bin, transferred into keyframe ``dst``:
    ``log_ratio`` the median log(z / d) (how much deeper ``src`` places the surfaces both see),
    ``log_src`` / ``log_dst`` the median log depths of those points in either keyframe, and the
    bin's weight (its direction's overlap, shared by its bins)."""

    src: int
    dst: int
    log_ratio: float
    log_src: float
    log_dst: float
    weight: float


def pair_bins(i: int, j: int, a: DepthView, b: DepthView, bins: int = DEPTH_BINS,
              min_points: int = PAIR_MIN_POINTS) -> list[BinRatio]:
    """``BinRatio`` of keyframes ``i`` (view ``a``) and ``j`` (view ``b``) in both directions:
    a direction that transfers at least ``min_points`` points onto the same surface is split into
    ``bins`` bins of equal count by the points' depth in the source keyframe."""
    out: list[BinRatio] = []
    for s, t, src, dst in ((i, j, a, b), (j, i, b, a)):
        r, ls, ld = _transfer(src, dst)
        if len(r) < min_points:
            continue
        w = min(1.0, len(r) / PAIR_FULL_POINTS) / bins
        order = np.argsort(ls, kind="stable")
        for part in np.array_split(order, bins):
            if len(part) < BIN_MIN_POINTS:
                continue
            out.append(BinRatio(s, t, float(np.median(r[part])), float(np.median(ls[part])),
                                float(np.median(ld[part])), w))
    return out


@dataclass
class DepthAdjustment:
    corrections: list[DepthCorrection]  # per keyframe (identity for the fixed ones)
    residuals_before: NDArray[np.float64]  # |log ratio| per bin before / after
    residuals_after: NDArray[np.float64]


def adjust_depth_corrections(n: int, bins: list[BinRatio], pivots: NDArray[Any], fixed: set[int],
                             scales: tuple[float, ...] = ROBUST_SCALES, iterations: int = 10
                             ) -> DepthAdjustment:
    """Per-keyframe corrections (log scale a, slope b about ``pivots[k]``; identity for ``k`` in
    ``fixed``) minimising Σ w ρ(r + a_s + b_s (ℓ_s − p_s) − a_t − b_t (ℓ_t − p_t)) over the
    ``bins`` (r their log ratio, ℓ their log depths in the source and target keyframes), with the
    Cauchy loss ρ at each robust scale of ``scales`` in turn (iteratively reweighted least
    squares; ``inf``: plain least squares), a tiny ridge on a and the prior b = 0 weighing
    ``SLOPE_PRIOR`` average bins; slopes are clamped to ±``MAX_SLOPE``."""
    from scipy.sparse import csr_matrix

    ident = [DepthCorrection(0.0, 0.0, float(pivots[k])) for k in range(n)]
    if not bins:
        return DepthAdjustment(ident, np.zeros(0), np.zeros(0))
    src = np.array([b.src for b in bins], np.int64)
    dst = np.array([b.dst for b in bins], np.int64)
    r = np.array([b.log_ratio for b in bins], np.float64)
    wts = np.array([b.weight for b in bins], np.float64)
    piv = np.asarray(pivots, np.float64)
    ls = np.array([b.log_src for b in bins], np.float64) - piv[src]
    lt = np.array([b.log_dst for b in bins], np.float64) - piv[dst]
    free = [k for k in range(n) if k not in fixed]
    if not free:
        return DepthAdjustment(ident, np.abs(r), np.abs(r))
    col = np.full(n, -1, np.int64)
    col[free] = np.arange(len(free))
    nf = len(free)
    rows, cols, vals = [], [], []
    for k, (cc, coef) in enumerate(((col[src], 1.0), (col[dst], -1.0))):
        lever = ls if k == 0 else lt
        ok = cc >= 0
        idx = np.flatnonzero(ok)
        rows += [idx, idx]
        cols += [cc[ok], nf + cc[ok]]
        vals += [np.full(len(idx), coef), coef * lever[ok]]
    A = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(len(bins), 2 * nf))
    reg = np.r_[np.full(nf, RIDGE), np.full(nf, SLOPE_PRIOR)] * float(wts.mean())
    x = np.zeros(2 * nf)
    for c in scales:
        for _ in range(iterations):
            res = r + A @ x
            w = wts if not np.isfinite(c) else wts / (1.0 + (res / c) ** 2)
            H = (A.T @ A.multiply(w[:, None])).toarray()
            H[np.diag_indices_from(H)] += reg
            x_new = np.linalg.solve(H, -(A.T @ (w * r)))
            done = float(np.abs(x_new - x).max()) < 1e-6
            x = x_new
            if done or not np.isfinite(c):
                break
    x[nf:] = np.clip(x[nf:], -MAX_SLOPE, MAX_SLOPE)
    out = list(ident)
    for k in free:
        out[k] = DepthCorrection(float(x[col[k]]), float(x[nf + col[k]]), float(piv[k]))
    return DepthAdjustment(out, np.abs(r), np.abs(r + A @ x))


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
