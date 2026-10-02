"""Multi-view depth consensus before fusion, as MVS depth fusion does it: a pixel's depth becomes
the median of its own and of the depths its neighbouring views give along its ray where they see
the same surface (MVSNet §4.2 averages the consistent reprojected depths, COLMAP's fusion takes
their median), and a pixel that several views see through, which few support, is left out
(the free-space violations of Merrell et al., ICCV 2007).

These are the array steps over one view's sampled pixels; which views are neighbours and which
pixels are sampled is the caller's (``mapping.geometry.consensus_depths``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.types import Intrinsics, Pose

MIN_DEPTH = 0.05  # m: a point nearer to a neighbour's camera than this is not judged by it


@dataclass
class Samples:
    """A view's sampled pixels: which samples of the ``step`` grid they are (``sel``), their
    depth, and their points in the view's camera frame (3 x N)."""

    sel: NDArray[np.bool_]
    z: NDArray[np.float64]
    cam: NDArray[np.float32]


def sample_pixels(depth: NDArray[Any], ok: NDArray[np.bool_], K: Intrinsics, step: int
                  ) -> Samples:
    """The ``ok`` pixels among every ``step``-th pixel of a depth grid (the centre of its
    ``step`` x ``step`` cell). Pixel indices are read as the TSDF reads them: u = fx x / z + cx."""
    h, w = depth.shape
    vv, uu = np.mgrid[step // 2:h:step, step // 2:w:step]
    z = np.asarray(depth, np.float64)[vv, uu]
    sel = ok[vv, uu] & np.isfinite(z) & (z > 0)
    v, u, zs = vv[sel], uu[sel], z[sel]
    cam = np.stack([(u - K.cx) / K.fx * zs, (v - K.cy) / K.fy * zs, zs]).astype(np.float32)
    return Samples(sel, zs, cam)


@dataclass
class Views:
    """Depth maps of several views in one flat array, each as the depth of the pixels it
    vouches for (0 elsewhere), so that the pixels of many views are read in one gather."""

    flat: NDArray[np.float32]
    offset: NDArray[np.int64]  # per view, where its pixels start in ``flat`` (row-major)
    shape: NDArray[np.int64]  # per view, (rows, cols)

    @staticmethod
    def of(depths: list[NDArray[Any]], oks: list[NDArray[np.bool_]]) -> Views:
        sizes = [int(d.size) for d in depths]
        offset = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
        flat = np.zeros(int(sum(sizes)), np.float32)
        for d, ok, o in zip(depths, oks, offset.tolist(), strict=True):
            flat[o:o + d.size] = np.where(ok, d, 0.0).reshape(-1)
        return Views(flat, offset, np.array([d.shape for d in depths], np.int64).reshape(-1, 2))


def neighbour_views(s: Samples, T_map_cam: Pose, views: Views, nbs: list[int],
                    Ks: list[Intrinsics], Ts: list[Pose], carves: NDArray[np.bool_],
                    same: float, free: float
                    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64],
                               NDArray[np.int64]]:
    """What neighbouring views (``nbs``: their indices in ``views``, intrinsics ``Ks`` and poses
    ``Ts``) say about a view's samples ``s`` (the view's pose ``T_map_cam``): (for each sample
    whose surface a neighbour sees — its depth at the sample's projection within ``same`` in log
    ratio —: the neighbour's position in ``nbs``, the sample, the neighbour's depth along the
    sample's ray; the samples that a neighbour whose free space counts (``carves``) sees through
    — its depth there lies beyond the sample by more than ``free`` in log ratio —, once per such
    neighbour).

    The neighbour's surface point lies on its own ray through the sample's point X; moved there
    from X by the ratio rho of the two depths along that ray, its depth in the view is
    rho z + (1 - rho) b, with b the depth of the neighbour's centre in the view."""
    B, n = len(nbs), len(s.z)
    M = np.empty((B, 3, 3))
    c = np.empty((B, 3))
    b = np.empty(B)
    for k, (K, T) in enumerate(zip(Ks, Ts, strict=True)):
        # view camera -> neighbour pixels (+0.5: truncation then rounds to the nearest pixel)
        Kp = np.array([[K.fx, 0.0, K.cx + 0.5], [0.0, K.fy, K.cy + 0.5], [0.0, 0.0, 1.0]])
        M[k] = Kp @ T.R.T @ T_map_cam.R
        c[k] = Kp @ T.R.T @ (T_map_cam.t - T.t)
        b[k] = (T.t - T_map_cam.t) @ T_map_cam.R[:, 2]
    Y = (M.reshape(3 * B, 3).astype(np.float32) @ s.cam).reshape(B, 3, n)
    Y += c.astype(np.float32)[:, :, None]
    zz = Y[:, 2]
    rows, cols = views.shape[nbs, 0][:, None], views.shape[nbs, 1][:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        pu = Y[:, 0] / zz
        pv = Y[:, 1] / zz
        kk, idx = np.nonzero((zz > MIN_DEPTH) & (pu >= 0) & (pu < cols) & (pv >= 0)
                             & (pv < rows))
    lin = (views.offset[nbs][kk] + pv[kk, idx].astype(np.int64) * cols[kk, 0]
           + pu[kk, idx].astype(np.int64))
    d = views.flat[lin]
    good = d > 0
    kk, idx = kk[good], idx[good]
    rho = d[good].astype(np.float64) / zz[kk, idx]
    hit = (rho >= np.exp(-same)) & (rho <= np.exp(same))
    on, rh = idx[hit], rho[hit]
    thru = (rho > np.exp(free)) & np.asarray(carves, bool)[kk]
    return kk[hit], on, rh * s.z[on] + (1.0 - rh) * b[kk[hit]], idx[thru]


def median_ratio(z: NDArray[np.float64], along: NDArray[np.float64]) -> NDArray[np.float64]:
    """Each sample's median of its own depth ``z`` and its neighbours' depths ``along`` its ray
    (N x k, NaN: none), over its own depth."""
    vals = np.sort(np.column_stack([z, along]), axis=1)  # NaN sorts last
    n = np.sum(np.isfinite(vals), axis=1)
    rows = np.arange(len(z))
    return 0.5 * (vals[rows, (n - 1) // 2] + vals[rows, n // 2]) / z


def free_space(through: NDArray[Any], support: NDArray[Any], free_min: int,
               free_share: float) -> NDArray[np.bool_]:
    """Whether samples are left out: at least ``free_min`` neighbours see through them
    (``through``), more than ``free_share`` of those that see through them or ``support``
    them."""
    through = np.asarray(through)
    return np.asarray((through >= free_min) & (through > free_share * (through + support)))


def spread_cells(depth: NDArray[Any], ok: NDArray[np.bool_], s: Samples,
                 ratio: NDArray[np.float64], drop: NDArray[np.bool_], step: int, same: float
                 ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """A view's depth with each sample's ``ratio`` applied to the pixels of its ``step`` x
    ``step`` cell on its surface (``ok``, within ``same`` in log ratio of its depth), and the
    pixels left out (``drop``, on the same surfaces)."""
    h, w = depth.shape
    d = np.asarray(depth, np.float32)
    # the cell of sample (r, c) holds pixels [r*step, (r+1)*step) x [c*step, (c+1)*step); the
    # last row or column of cells may have no sample (a side of 4k + 1 pixels, step 4)
    cells = (-(-h // step), -(-w // step))
    sr, sc = s.sel.shape
    gr = np.ones(cells)
    gz = np.zeros(cells)
    gd = np.zeros(cells, bool)
    gr[:sr, :sc][s.sel], gz[:sr, :sc][s.sel], gd[:sr, :sc][s.sel] = ratio, s.z, drop
    full = np.repeat(np.repeat(gr, step, axis=0), step, axis=1)[:h, :w]
    zcell = np.repeat(np.repeat(gz, step, axis=0), step, axis=1)[:h, :w]
    dcell = np.repeat(np.repeat(gd, step, axis=0), step, axis=1)[:h, :w]
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(ok & (zcell > 0), d / np.where(zcell > 0, zcell, 1.0), 0.0)
    on = (rel >= np.exp(-same)) & (rel <= np.exp(same))
    out = np.where(on & (full != 1.0), d * full, d).astype(np.float32)
    return out, on & dcell
