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
