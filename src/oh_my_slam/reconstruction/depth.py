"""Depth alignment: robust per-frame scale against sparse SfM depths and a global metric scale."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.geometry import unproject_pixels

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


# Dense scale fits compare a keyframe's depth with what its neighbours see along the same rays.
# A surface much nearer than the rest of a view (a pole 0.2-0.5 m from a camera turning about a
# point a few cm behind its lens, where the walls are 2-3 m away) is seen by each neighbour beside
# where the keyframe sees it, against the wall behind: on ``examples/camera`` such pixels made
# 25-45 % of a keyframe's comparisons, at ratios of 7-9 (or 0.1 the other way), and fits gave
# scales up to 9.3 where the walls agree at 1.0. Pixels nearer than
# ``NEAR_SHARE`` of the view's median depth therefore take no part, on either side, and the ratio
# is taken around the dominant cluster (``_dominant``): the samples within ``CLUSTER_GATE`` (log,
# a factor 1.5: another surface) of the densest ``CLUSTER_WIDTH`` window.
NEAR_SHARE = 0.5
CLUSTER_WIDTH = 0.1
CLUSTER_GATE = 0.4


def _dominant(r: NDArray[np.float64]) -> NDArray[np.bool_]:
    """The log ratios ``r`` within ``CLUSTER_GATE`` of the centre of their densest window
    (``CLUSTER_WIDTH`` wide)."""
    s = np.sort(r)
    ends = np.searchsorted(s, s + CLUSTER_WIDTH, side="right")
    i = int(np.argmax(ends - np.arange(len(s))))
    centre = float(np.median(s[i:ends[i]]))
    return np.asarray(np.abs(r - centre) <= CLUSTER_GATE)


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
    the same ray; the robust median ratio of the dominant cluster updates s (two iterations
    absorb small baselines). The near field (``NEAR_SHARE``) of either view takes no part. Used
    for keyframes with too few triangulated points for a sparse fit.
    """
    d = np.asarray(depth, np.float64)
    h, w = d.shape
    vv, uu = np.mgrid[step // 2:h:step, step // 2:w:step]
    z0 = d[vv, uu]
    ok = z0 > 0
    if ok.any():
        ok &= z0 >= NEAR_SHARE * np.median(z0[ok])
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
            seen = rd[rd > 0]
            good = (obs > 0) & (obs >= NEAR_SHARE * (np.median(seen) if len(seen) else 0.0))
            num.append(obs[good])
            den.append(zc[inside][good])
        if not num:
            return fit
        a, b = np.concatenate(num), np.concatenate(den)
        if len(a):
            keep = _dominant(np.log(a / b))
            a, b = a[keep], b[keep]
        fit = robust_ratio(a, b, min_points)
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
# field. Nor is it the same over the image: monocular depth bends surfaces (a wall seen at the
# image's side placed 10-15 % nearer than where its neighbour sees it at the centre, with opposite
# signs for the same keyframe against neighbours on either side), so each keyframe also takes a
# smooth field over its image (``FIELD_NODES``, bilinear between them, of mean 0 so that it moves no
# scale), measured on the pairs in bins of image position as well as depth. On ``examples/camera``
# it brought the pairs' p90 disagreement from 11.7 % to 4.7 %, the share of pairs above 10 % from
# 14 % to 0 and the worst pair from 22.5 % to 8.1 %.

PAIR_STEP = 8  # every 8th pixel of the source grid (both axes) is transferred
PAIR_GATE = 0.4  # |log ratio| beyond this (a factor 1.5) is another surface (occlusion)
PAIR_MIN_POINTS = 300  # fewer transferred points in a direction: that direction is not measured
PAIR_FULL_POINTS = 2000  # a direction with at least this many points gets its full weight
DEPTH_BINS = 6  # a direction's points are split into at most this many bins of equal count by depth
BIN_MIN_POINTS = 30
# a keyframe's correction field: node columns x rows over its image; a direction's points are split
# by their place in the source image into the cells between the nodes, each cell's by depth into
# bins of about BIN_POINTS (at most DEPTH_BINS)
FIELD_NODES = (4, 3)
BIN_POINTS = 60
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
# Each node of a keyframe's field is pulled to 0 as much as FIELD_PRIOR bins of average weight and
# to its neighbours (each difference between adjacent nodes to 0) as much as FIELD_SMOOTH (a field
# bends only where the pairs say so, and smoothly), and the field's mean over the image to 0 as
# much as FIELD_MEAN_PRIOR bins (the scale is the log scale's alone, which a held keyframe holds).
# On examples/camera, 0.1 / 1 left a pan's walls leaning 3° and a window frame 2°, 0.3 / 3 the
# pairs' p90 disagreement at 5.2 %; 0.2 / 2: 4.7 %, walls and frames within 2° of vertical.
# The pairs of a camera turning in place hardly see the field every keyframe shares: where two
# keyframes overlap, the surfaces lie at the edge of both (to the left in one, to the right in the
# other; at the top of one, at the bottom of the other), so all of them bending their edges alike
# agrees as well as none. The fields summed over the keyframes are therefore held at 0, node by
# node, as much as FIELD_MEAN_PRIOR bins: monocular depth bends either way from one keyframe to the
# next, and the network's own average shape stands.
FIELD_PRIOR = 0.2
FIELD_SMOOTH = 2.0
FIELD_MEAN_PRIOR = 100.0
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
    _places: NDArray[np.float64] | None = None

    def samples(self) -> NDArray[np.float64]:
        """Camera-frame points of every ``step``-th valid pixel (cached)."""
        if self._samples is None:
            d = np.asarray(self.depth, np.float64)
            h, w = d.shape
            vv, uu = np.mgrid[self.step // 2:h:self.step, self.step // 2:w:self.step]
            z = d[vv, uu]
            ok = np.isfinite(z) & (z > 0)
            u, v, z = uu[ok].astype(np.float64), vv[ok].astype(np.float64), z[ok]
            self._samples = unproject_pixels(u, v, z, self.K)
            self._places = np.stack([(u + 0.5) / w, (v + 0.5) / h], 1)
        return self._samples

    def places(self) -> NDArray[np.float64]:
        """The samples' places in the image (``field_weights``)."""
        self.samples()
        assert self._places is not None
        return self._places

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


def field_weights(places: Any, nodes: tuple[int, int] = FIELD_NODES) -> NDArray[np.float64]:
    """Bilinear weights (N, columns x rows, row-major) of a field's nodes at ``places`` (N, 2): x
    and y over the image's width and height, 0 to 1 (nodes at both edges; one node: constant
    along that axis)."""
    at = np.asarray(places, np.float64).reshape(-1, 2)
    wx, wy = _along(at[:, 0], nodes[0]), _along(at[:, 1], nodes[1])
    return np.asarray((wy[:, :, None] * wx[:, None, :]).reshape(len(at), -1))


def _along(t: NDArray[Any], n: int) -> NDArray[np.float64]:
    """Linear weights (N, n) of ``n`` nodes spread over 0..1 at the positions ``t`` (N,)."""
    if n == 1:
        return np.ones((len(t), 1))
    u = np.clip(t, 0.0, 1.0) * (n - 1)
    i = _cell(t, n)
    out = np.zeros((len(t), n))
    out[np.arange(len(t)), i] = 1 - (u - i)
    out[np.arange(len(t)), i + 1] = u - i
    return out


def _cell(t: NDArray[Any], n: int) -> NDArray[np.int64]:
    """The cell between ``n`` nodes over 0..1 that each position ``t`` lies in (0 for one node)."""
    return np.clip(np.floor(np.clip(t, 0.0, 1.0) * (n - 1)).astype(np.int64), 0, max(n - 2, 0))


def _axis_weights(n: int, size: int) -> NDArray[np.float64]:
    """Linear weights (size, n) of ``n`` nodes at the pixel centres of an image axis."""
    return _along((np.arange(size) + 0.5) / size, n)


FIELD_MEAN = np.outer(_axis_weights(FIELD_NODES[1], 1200).mean(axis=0),
                      _axis_weights(FIELD_NODES[0], 1200).mean(axis=0)).ravel()  # image mean


def _roughness(nodes: tuple[int, int]) -> NDArray[np.float64]:
    """``LᵀL`` of the differences ``L`` between adjacent nodes of a field (row-major)."""
    gx, gy = nodes
    index = np.arange(gx * gy).reshape(gy, gx)
    pairs = [(a, b) for a, b in zip(index[:, :-1].ravel(), index[:, 1:].ravel(), strict=True)]
    pairs += [(a, b) for a, b in zip(index[:-1].ravel(), index[1:].ravel(), strict=True)]
    L = np.zeros((len(pairs), gx * gy))
    for row, (a, b) in enumerate(pairs):
        L[row, a], L[row, b] = 1.0, -1.0
    return L.T @ L


FIELD_ROUGHNESS = _roughness(FIELD_NODES)




@dataclass(frozen=True)
class DepthCorrection:
    """``log d' = log d + log_scale + slope · (log d − pivot) + field(x, y)`` (``pivot``: log
    metres, the keyframe's median depth when solved; ``field``: the values of a smooth field over
    the image at its ``FIELD_NODES``, row-major, bilinear between them; empty: 0): a scale, a
    near/far tilt and a bend over the image; the factor ``d'/d`` is clamped to
    [1/``MAX_FACTOR``, ``MAX_FACTOR``]."""

    log_scale: float = 0.0
    slope: float = 0.0
    pivot: float = 0.0
    field: tuple[float, ...] = ()

    @property
    def identity(self) -> bool:
        return self.log_scale == 0.0 and self.slope == 0.0 and not any(self.field)

    @property
    def scale(self) -> float:
        """The factor at the pivot (over the image, on average)."""
        return float(np.exp(self.log_scale))

    @property
    def exponent(self) -> float:
        """``d' ∝ d ** exponent``."""
        return 1.0 + self.slope

    @property
    def bend(self) -> float:
        """The field's largest factor away from 1 (0: none)."""
        return float(np.expm1(np.abs(self.field)).max()) if self.field else 0.0

    def factor(self, depth: Any, places: Any = None) -> NDArray[np.float64]:
        """``d'/d`` at each depth (metres; 1 where the depth is not positive), at its ``places``
        in the image (``field_weights``; needed with a field)."""
        d = np.asarray(depth, np.float64)
        log_f = self.log_scale + self.slope * (np.log(np.where(d > 0, d, 1.0)) - self.pivot)
        if self.field:
            if places is None:
                raise ValueError("a depth correction with a field needs the depths' places")
            log_f = log_f + (field_weights(places) @ np.asarray(self.field)).reshape(d.shape)
        f = np.clip(np.exp(log_f), 1.0 / MAX_FACTOR, MAX_FACTOR)
        return np.asarray(np.where(d > 0, f, 1.0), np.float64)

    def apply(self, depth: Any) -> NDArray[np.float64]:
        """The corrected depth grid (non-positive depths unchanged)."""
        d = np.asarray(depth, np.float64)
        log_f = self.log_scale + self.slope * (np.log(np.where(d > 0, d, 1.0)) - self.pivot)
        if self.field:
            h, w = d.shape
            F = np.asarray(self.field).reshape(FIELD_NODES[1], FIELD_NODES[0])
            log_f = log_f + _axis_weights(FIELD_NODES[1], h) @ F @ _axis_weights(
                FIELD_NODES[0], w).T
        f = np.clip(np.exp(log_f), 1.0 / MAX_FACTOR, MAX_FACTOR)
        return np.asarray(d * np.where(d > 0, f, 1.0), np.float64)

    def then(self, other: DepthCorrection) -> DepthCorrection:
        """This correction followed by ``other`` (defined on the corrected depth), about this
        one's pivot (clamps aside)."""
        b1, b2 = self.slope, other.slope
        a = (1 + b2) * (self.pivot + self.log_scale) + other.log_scale - b2 * other.pivot \
            - self.pivot
        n = FIELD_NODES[0] * FIELD_NODES[1]
        f1 = np.asarray(self.field or (0.0,) * n)
        f2 = np.asarray(other.field or (0.0,) * n)
        field = (1 + b2) * f1 + f2
        return DepthCorrection(float(a), float((1 + b1) * (1 + b2) - 1), self.pivot,
                               tuple(float(v) for v in field) if field.any() else ())


def _transfer(src: DepthView, dst: DepthView, scale_src: float = 1.0, scale_dst: float = 1.0
              ) -> tuple[NDArray[np.float64], ...]:
    """``src``'s sampled points (depth times ``scale_src``) in ``dst``'s camera, on the same
    surface as ``dst``'s depth (times ``scale_dst``) at the pixel they land on (|log ratio| <=
    ``PAIR_GATE``): (log(z / d), log of their depth in ``src``, log d, their places in ``src`` and
    in ``dst`` (``field_weights``)) — z their depth in ``dst``'s camera, d ``dst``'s depth
    there."""
    pts = src.samples()
    empty = np.zeros(0)
    if not len(pts):
        return empty, empty, empty, np.zeros((0, 2)), np.zeros((0, 2))
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
    at_src = src.places()[inside][good]
    at_dst = np.stack([(u[inside][good] + 0.5) / w, (v[inside][good] + 0.5) / h], 1)
    same = np.abs(r) <= PAIR_GATE
    return r[same], log_src[same], log_d[same], at_src[same], at_dst[same]


@dataclass(frozen=True)
class BinRatio:
    """Keyframe ``src``'s sampled points of one bin (of image place and depth), transferred into
    keyframe ``dst``: ``log_ratio`` the median log(z / d) (how much deeper ``src`` places the
    surfaces both see), ``log_src`` / ``log_dst`` the median log depths of those points in either
    keyframe, the bin's weight (its direction's overlap, shared by its bins by their points) and
    the mean weights of either keyframe's field nodes at the points (``field_weights``; empty:
    no field)."""

    src: int
    dst: int
    log_ratio: float
    log_src: float
    log_dst: float
    weight: float
    field_src: tuple[float, ...] = ()
    field_dst: tuple[float, ...] = ()


def pair_bins(i: int, j: int, a: DepthView, b: DepthView, bins: int = DEPTH_BINS,
              min_points: int = PAIR_MIN_POINTS) -> list[BinRatio]:
    """``BinRatio`` of keyframes ``i`` (view ``a``) and ``j`` (view ``b``) in both directions:
    a direction that transfers at least ``min_points`` points onto the same surface is split by
    the points' place in the source keyframe into the cells between its field's nodes, each cell
    into bins of equal count (about ``BIN_POINTS``, at most ``bins``) by the points' depth in the
    source keyframe."""
    out: list[BinRatio] = []
    for s, t, src, dst in ((i, j, a, b), (j, i, b, a)):
        r, ls, ld, at_s, at_d = _transfer(src, dst)
        if len(r) < min_points:
            continue
        w = min(1.0, len(r) / PAIR_FULL_POINTS) / len(r)
        ws, wd = field_weights(at_s), field_weights(at_d)
        cell = (_cell(at_s[:, 1], FIELD_NODES[1]) * FIELD_NODES[0]
                + _cell(at_s[:, 0], FIELD_NODES[0]))
        for c in np.unique(cell):
            idx = np.flatnonzero(cell == c)
            order = idx[np.argsort(ls[idx], kind="stable")]
            for part in np.array_split(order, int(np.clip(len(idx) // BIN_POINTS, 1, bins))):
                if len(part) < BIN_MIN_POINTS:
                    continue
                out.append(BinRatio(s, t, float(np.median(r[part])), float(np.median(ls[part])),
                                    float(np.median(ld[part])), w * len(part),
                                    tuple(ws[part].mean(axis=0).tolist()),
                                    tuple(wd[part].mean(axis=0).tolist())))
    return out


@dataclass
class DepthAdjustment:
    corrections: list[DepthCorrection]  # per keyframe (identity for the fixed ones)
    residuals_before: NDArray[np.float64]  # |log ratio| per bin before / after
    residuals_after: NDArray[np.float64]


def adjust_depth_corrections(n: int, bins: list[BinRatio], pivots: NDArray[Any], fixed: set[int],
                             scales: tuple[float, ...] = ROBUST_SCALES, iterations: int = 10,
                             scale_fixed: set[int] | None = None,
                             flat: set[int] | None = None) -> DepthAdjustment:
    """Per-keyframe corrections (log scale a, slope b about ``pivots[k]`` and, when the ``bins``
    carry their places, a field F over the image; identity for ``k`` in ``fixed``, a = 0 for ``k``
    in ``scale_fixed``: its scale holds, its tilt and field are solved) minimising
    Σ w ρ(r + a_s + b_s (ℓ_s − p_s) + φ_s·F_s − a_t − b_t (ℓ_t − p_t) − φ_t·F_t) over the bins (r
    their log ratio, ℓ their log depths and φ their field weights in the source and target
    keyframes), with the Cauchy loss ρ at each robust scale of ``scales`` in turn (iteratively
    reweighted least squares; ``inf``: plain least squares), a tiny ridge on a, the prior b = 0
    weighing ``SLOPE_PRIOR`` average bins (which also holds the one tilt the pairs cannot see: the
    same tilt of every keyframe), each field node's prior 0 weighing ``FIELD_PRIOR``, adjacent
    nodes' differences 0 ``FIELD_SMOOTH``, and the field's mean over the image and the fields
    summed over the measured keyframes (node by node) held at 0 by ``FIELD_MEAN_PRIOR``; slopes
    are clamped to ±``MAX_SLOPE``. ``flat``: keyframes that take no field (their scale and tilt
    alone are solved)."""
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
    tilted = [k for k in range(n) if k not in fixed]
    scaled = [k for k in tilted if k not in (scale_fixed or set())]
    if not tilted:
        return DepthAdjustment(ident, np.abs(r), np.abs(r))
    nodes = len(bins[0].field_src) if all(b.field_src for b in bins) else 0
    col_a = np.full(n, -1, np.int64)
    col_a[scaled] = np.arange(len(scaled))
    col_b = np.full(n, -1, np.int64)
    col_b[tilted] = len(scaled) + np.arange(len(tilted))
    bent = [k for k in tilted if k not in (flat or set())]
    col_f = np.full(n, -1, np.int64)
    col_f[bent] = len(scaled) + len(tilted) + nodes * np.arange(len(bent))
    nx = len(scaled) + len(tilted) + nodes * len(bent)
    rows, cols, vals = [], [], []
    for ends, coef, lever in ((src, 1.0, ls), (dst, -1.0, lt)):
        for col, v in ((col_a[ends], np.full(len(ends), coef)), (col_b[ends], coef * lever)):
            ok = col >= 0
            rows.append(np.flatnonzero(ok))
            cols.append(col[ok])
            vals.append(v[ok])
    if nodes:
        for ends, coef, attr in ((src, 1.0, "field_src"), (dst, -1.0, "field_dst")):
            phi = np.array([getattr(b, attr) for b in bins], np.float64)
            ok = col_f[ends] >= 0
            rows.append(np.repeat(np.flatnonzero(ok), nodes))
            cols.append((col_f[ends][ok][:, None] + np.arange(nodes)).ravel())
            vals.append((coef * phi[ok]).ravel())
    A = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(len(bins), nx))
    avg = float(wts.mean())
    reg = np.r_[np.full(len(scaled), RIDGE), np.full(len(tilted), SLOPE_PRIOR),
                np.full(len(bent) * nodes, FIELD_PRIOR)] * avg
    prior = np.diag(reg)
    if nodes:
        block = avg * (FIELD_MEAN_PRIOR * np.outer(FIELD_MEAN, FIELD_MEAN)
                       + FIELD_SMOOTH * FIELD_ROUGHNESS)
        shared = np.zeros((nodes, nx))
        measured = set(src.tolist()) | set(dst.tolist())
        for k in bent:
            sl = slice(int(col_f[k]), int(col_f[k]) + nodes)
            prior[sl, sl] += block
            shared[:, sl] = np.eye(nodes) * (k in measured)  # a keyframe without pairs: no field
        prior += FIELD_MEAN_PRIOR * avg * (shared.T @ shared)
    x = np.zeros(nx)
    for c in scales:
        for _ in range(iterations):
            res = r + A @ x
            w = wts if not np.isfinite(c) else wts / (1.0 + (res / c) ** 2)
            H = (A.T @ A.multiply(w[:, None])).toarray() + prior
            x_new = np.linalg.solve(H, -(A.T @ (w * r)))
            done = float(np.abs(x_new - x).max()) < 1e-6
            x = x_new
            if done or not np.isfinite(c):
                break
    b0 = len(scaled)
    x[b0:b0 + len(tilted)] = np.clip(x[b0:b0 + len(tilted)], -MAX_SLOPE, MAX_SLOPE)
    out = list(ident)
    for k in tilted:
        a = float(x[col_a[k]]) if col_a[k] >= 0 else 0.0
        field = tuple(float(v) for v in x[col_f[k]:col_f[k] + nodes]) if col_f[k] >= 0 else ()
        out[k] = DepthCorrection(a, float(x[col_b[k]]), float(piv[k]), field)
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
