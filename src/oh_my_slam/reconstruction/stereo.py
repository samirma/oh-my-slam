"""Motion-stereo depth (api.md §7 "Motion stereo", oms/2.11): a reference frame's monocular depth
brought onto the points triangulated from other frames of the same camera at known relative
poses (``reports/motion_stereo_eval.md`` §3 method ``b2``, §9). Everything here is depth and
point generation on given rays and poses; the matches and the poses' refinement are mapping's
(``mapping.motion``).

* **Triangulation** (``triangulate``): each match's depth in the reference, the midpoint of its
  two rays, with σ(ln z) = hypot(``MATCH_SIGMA_PX``, σ_rot · f) over the pixels of parallax per
  unit ln z (the match's pixel noise and the relative rotation's posterior sigma, against how
  far the point's image in the other frame moves when its depth changes).
* **Fit** (``fit_scale``): ln z_tri − ln z_mono = a + b (ln z_mono − c) over the points with
  σ(ln z) below ``BASELINE_POINT_SIGMA_MAX``: a scale and a slope in ln z, iteratively
  re-weighted (inverse variance σ² + spread², Huber), the slope bounded by ``FIT_SLOPE_MAX``.
  ``sigma_a`` is the scale's standard error, ``spread`` the robust (1.4826 MAD) spread of the
  residuals: the monocular shape against the triangulation.
* **Sparse-to-dense correction** (``sparse_dense``): per pixel, the inverse-variance weighted
  residual ln(z_tri / z_fit) of the points within 3 × ``S2D_RADIUS_PX`` (its ``S2D_NEIGHBOURS``
  nearest), each weighted by a Gaussian of ``S2D_RADIUS_PX`` in the image and ``S2D_LNZ`` in
  ln z, shrunk to 0 by a prior of weight 1 / ``S2D_PRIOR``²; the local σ(ln z) left is
  sqrt(1 / (Σ weights + 1 / spread²)): the spread far from every point, less near them."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.geometry import depth_edge_mask

Array = NDArray[np.float64]

MATCH_SIGMA_PX = 0.7  # a SIFT keypoint's localisation, pixels (1 sigma)
BASELINE_POINT_SIGMA_MAX = 0.25  # sigma(ln z) of a triangulated point that takes part in a fit
MIN_TRIANGULATED_Z_M = 0.05  # a point this near either camera is no point
PARALLAX_STEP = 0.01  # the relative depth step the parallax per unit ln z is measured over
FIT_MIN_POINTS = 6  # a fit needs this many points (a pair qualifies from many more)
FIT_ITERATIONS = 8
FIT_HUBER = 1.5  # in sigmas
FIT_SLOPE_MAX = 0.6  # |b|: the monocular shape is stretched at most this much in ln z
SPREAD_MIN = 0.02  # the residuals' spread is at least this (ln z)
S2D_RADIUS_PX = 25.0  # the correction's Gaussian in the image (pixels of the depth grid)
S2D_LNZ = 0.15  # ... and in ln z
S2D_PRIOR = 0.10  # ... a residual is believed beyond this (the prior's sigma, ln z)
S2D_NEIGHBOURS = 16
S2D_POINT_FLOOR = 0.03  # a point's sigma(ln z) in the correction's weights is at least this


@dataclass
class Triangulated:
    """Triangulated matches: the depth ``z`` in the reference (metres of the translation),
    σ(ln z) ``sigma`` and the parallax angle (degrees), with ``ok`` over the matches given (the
    ones these values are for)."""

    z: Array
    sigma: Array
    parallax_deg: Array
    ok: NDArray[np.bool_]


def triangulate(ra: Array, rb: Array, R: Array, t: Array, f: float, sigma_rot_rad: float
                ) -> Triangulated:
    """The depth in the reference of matches whose rays (unit z) are ``ra`` in the reference and
    ``rb`` in the other frame, that frame at ``X_ref = R X_other + t``: the midpoint of the two
    rays, with σ(ln z) (module docstring; ``f`` the focal length in pixels, ``sigma_rot_rad``
    the relative rotation's sigma). A point behind or within ``MIN_TRIANGULATED_Z_M`` of either
    camera, on parallel rays, or without a finite sigma, is not ``ok``."""
    a = np.asarray(ra, np.float64).reshape(-1, 3)
    b = np.asarray(rb, np.float64).reshape(-1, 3) @ np.asarray(R, np.float64).T
    t = np.asarray(t, np.float64)
    aa, bb, ab = np.sum(a * a, 1), np.sum(b * b, 1), np.sum(a * b, 1)
    at, bt = a @ t, b @ t
    det = aa * bb - ab * ab
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (at * bb - ab * bt) / det
        z2 = (ab * at - aa * bt) / det
        X = a * z[:, None]
        Xo = (X - t) @ R
        Xs = (X * (1.0 + PARALLAX_STEP) - t) @ R
        moved = np.hypot(Xs[:, 0] / Xs[:, 2] - Xo[:, 0] / Xo[:, 2],
                         Xs[:, 1] / Xs[:, 2] - Xo[:, 1] / Xo[:, 2]) * f / PARALLAX_STEP
        sigma = math.hypot(MATCH_SIGMA_PX, sigma_rot_rad * f) / moved
        cosang = ab / np.sqrt(aa * bb)
    ang = np.degrees(np.arccos(np.clip(np.nan_to_num(cosang, nan=1.0), -1.0, 1.0)))
    ok = (np.isfinite(z) & (z > MIN_TRIANGULATED_Z_M) & np.isfinite(z2) & (z2 > 0)
          & (Xo[:, 2] > MIN_TRIANGULATED_Z_M) & np.isfinite(sigma))
    return Triangulated(np.where(ok, z, np.nan), np.where(ok, sigma, np.inf), ang, ok)


def sample(depth: NDArray[Any], uv: Array) -> Array:
    """A depth grid bilinearly at pixels ``uv`` (N x 2, integer centres); NaN off the grid, at
    a pixel without depth or on a depth edge (``core.geometry.depth_edge_mask``) among the four
    it interpolates."""
    d = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0).astype(np.float64)
    h, w = d.shape
    bad = (d <= 0) | depth_edge_mask(d)
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    out = np.full(len(uv), np.nan)
    with np.errstate(invalid="ignore"):
        inside = (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
    x, y = uv[inside, 0], uv[inside, 1]
    x0 = np.minimum(np.floor(x).astype(np.intp), max(w - 2, 0))
    y0 = np.minimum(np.floor(y).astype(np.intp), max(h - 2, 0))
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    v = (d[y0, x0] * (1 - fx) * (1 - fy) + d[y0, x1] * fx * (1 - fy)
         + d[y1, x0] * (1 - fx) * fy + d[y1, x1] * fx * fy)
    poor = bad[y0, x0] | bad[y0, x1] | bad[y1, x0] | bad[y1, x1]
    out[inside] = np.where(poor, np.nan, v)
    return out


@dataclass
class ScaleFit:
    """ln z_tri − ln z_mono = a + b (ln z_mono − c) (module docstring) over ``n`` points."""

    a: float
    b: float
    c: float
    sigma_a: float
    spread: float
    n: int

    def apply(self, depth: NDArray[Any]) -> NDArray[np.float32]:
        """The depth grid brought onto the fit (0 stays 0)."""
        d = np.asarray(depth, np.float64)
        v = np.isfinite(d) & (d > 0)
        out = np.zeros(d.shape, np.float64)
        lm = np.log(d[v])
        out[v] = np.exp(lm + self.a + self.b * (lm - self.c))
        return out.astype(np.float32)


def fit_scale(zm: Array, zt: Array, sig: Array) -> ScaleFit | None:
    """The fit of the monocular depths ``zm`` to the triangulated ``zt`` (σ(ln z) ``sig``) at
    the same points (module docstring); None with fewer than ``FIT_MIN_POINTS`` usable points
    (both depths positive and finite, σ below ``BASELINE_POINT_SIGMA_MAX``)."""
    zm, zt, sig = (np.asarray(v, np.float64) for v in (zm, zt, sig))
    with np.errstate(invalid="ignore"):
        ok = (np.isfinite(zm) & (zm > 0) & np.isfinite(zt) & (zt > 0)
              & (sig < BASELINE_POINT_SIGMA_MAX))
    if int(ok.sum()) < FIT_MIN_POINTS:
        return None
    lm, s = np.log(zm[ok]), sig[ok]
    y = np.log(zt[ok]) - lm
    c = float(np.median(lm))
    X = np.c_[np.ones_like(lm), lm - c]
    beta = np.array([float(np.median(y)), 0.0])
    spread = SPREAD_MIN
    for _ in range(FIT_ITERATIONS):
        r = y - X @ beta
        spread = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), SPREAD_MIN)
        w0 = 1.0 / (s ** 2 + spread ** 2)
        u = np.abs(r) * np.sqrt(w0)
        w = w0 * np.where(u < FIT_HUBER, 1.0, FIT_HUBER / np.maximum(u, 1e-9))
        A = X * w[:, None]
        beta = np.linalg.solve(X.T @ A + 1e-9 * np.eye(2), A.T @ y)
        beta[1] = float(np.clip(beta[1], -FIT_SLOPE_MAX, FIT_SLOPE_MAX))
    w0 = 1.0 / (s ** 2 + spread ** 2)
    cov = np.linalg.inv(X.T @ (X * w0[:, None]) + 1e-9 * np.eye(2))
    return ScaleFit(float(beta[0]), float(beta[1]), c, float(math.sqrt(cov[0, 0])),
                    float(spread), int(ok.sum()))


def sparse_dense(depth: NDArray[np.float32], uv: Array, zt: Array, sig: Array, spread: float
                 ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """The fitted ``depth`` corrected towards the triangulated points (``zt`` at grid pixels
    ``uv``, σ(ln z) ``sig``) and the local σ(ln z) left per pixel (module docstring; 0 where
    the depth is 0). The points below ``BASELINE_POINT_SIGMA_MAX`` with a depth at their pixel
    take part; without any, the depth as it is and ``spread`` everywhere."""
    from scipy.spatial import cKDTree

    d = np.asarray(depth, np.float32)
    valid = np.isfinite(d) & (d > 0)
    local = np.where(valid, np.float32(spread), np.float32(0.0)).astype(np.float32)
    zi = sample(d, uv)
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(zi) & np.isfinite(zt) & (zt > 0) & (sig < BASELINE_POINT_SIGMA_MAX)
    if not ok.any():
        return d.copy(), local
    xy, r = np.asarray(uv, np.float64)[ok], np.log(zt[ok]) - np.log(zi[ok])
    lzi = np.log(zi[ok])
    wi = 1.0 / (np.maximum(sig[ok], S2D_POINT_FLOOR) ** 2)
    vv, uu = np.nonzero(valid)
    k = min(S2D_NEIGHBOURS, len(xy))
    dist, j = cKDTree(xy).query(np.c_[uu, vv].astype(np.float64), k=k,
                                distance_upper_bound=3.0 * S2D_RADIUS_PX)
    dist, j = dist.reshape(len(vv), k), j.reshape(len(vv), k)
    near = np.isfinite(dist)
    jj = np.where(near, j, 0)
    lzq = np.log(d[vv, uu].astype(np.float64))[:, None]
    g = (np.exp(-0.5 * (np.where(near, dist, 0.0) / S2D_RADIUS_PX) ** 2)
         * np.exp(-0.5 * ((lzq - lzi[jj]) / S2D_LNZ) ** 2))
    g = np.where(near, g, 0.0) * wi[jj]
    total = g.sum(1)
    corr = (g * r[jj]).sum(1) / (total + 1.0 / S2D_PRIOR ** 2)
    out = d.copy()
    out[vv, uu] = (d[vv, uu] * np.exp(corr)).astype(np.float32)
    local[vv, uu] = np.sqrt(1.0 / (total + 1.0 / spread ** 2)).astype(np.float32)
    return out, local


@dataclass
class StereoDepth:
    """A reference depth brought onto triangulated points: the corrected grid, its local
    σ(ln z) per pixel (``sparse_dense``) and the fit."""

    depth: NDArray[np.float32]
    local_sigma: NDArray[np.float32]
    fit: ScaleFit


def stereo_depth(depth: NDArray[np.float32], uv: Array, zt: Array, sig: Array
                 ) -> StereoDepth | None:
    """The reference's monocular ``depth`` grid fitted to triangulated points (``zt`` at grid
    pixels ``uv``, σ(ln z) ``sig``: ``fit_scale`` on the depth sampled there) and corrected
    towards them (``sparse_dense``); None without a fit."""
    fit = fit_scale(sample(depth, uv), zt, sig)
    if fit is None:
        return None
    out, local = sparse_dense(fit.apply(depth), uv, zt, sig, fit.spread)
    return StereoDepth(out, local, fit)
