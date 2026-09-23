"""Lift an instance mask to 3D: shrink the mask, drop invalid and depth-edge pixels, keep the
largest spatial cluster and remove statistical outliers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from oh_my_slam.core.geometry import depth_edge_mask, unproject_pixels
from oh_my_slam.core.types import Intrinsics, Pose

SHRINK_PX = 3
MIN_EPS = 0.03
EPS_FRACTION = 0.01
SOR_K = 8
SOR_STD = 2.0
MIN_POINTS = 20

_OFFSETS = np.array(
    [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
     if (dx, dy, dz) > (0, 0, 0)],
    dtype=np.int64,
)


@dataclass
class Lifted:
    points: NDArray[np.float64]  # (N, 3) in the parent frame (camera frame if no pose)
    pixels: NDArray[np.int64]  # flat pixel indices on the depth grid
    mask_pixels: int  # pixels of the (unshrunk) mask


def shrink_mask(mask: NDArray[Any], px: int = SHRINK_PX) -> NDArray[np.bool_]:
    if px <= 0:
        return np.asarray(mask, bool)
    return ndimage.binary_erosion(np.asarray(mask, bool), structure=np.ones((3, 3), bool),
                                  iterations=px)


def largest_cluster(points: NDArray[Any], eps: float) -> NDArray[np.bool_]:
    """Largest set of points connected through occupied ``eps`` voxels (26-neighbourhood).

    A single-linkage, voxelised DBSCAN (min_points = 1) that runs in O(N log N) with NumPy.
    """
    n = len(points)
    if n == 0:
        return np.zeros(0, bool)
    keys = np.floor(np.asarray(points) / eps).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    m = len(uniq)
    if m == 1:
        return np.ones(n, bool)
    u = uniq - uniq.min(0)
    dims = u.max(0) + 3
    code = (u[:, 0] * dims[1] + u[:, 1]) * dims[2] + u[:, 2]
    order = np.argsort(code)
    sorted_code = code[order]
    rows, cols = [], []
    for off in _OFFSETS:
        nb = u + off
        ncode = (nb[:, 0] * dims[1] + nb[:, 1]) * dims[2] + nb[:, 2]
        pos = np.searchsorted(sorted_code, ncode)
        pos = np.clip(pos, 0, m - 1)
        hit = sorted_code[pos] == ncode
        rows.append(np.nonzero(hit)[0])
        cols.append(order[pos[hit]])
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    graph = coo_matrix((np.ones(len(r), np.int8), (r, c)), shape=(m, m))
    _, labels = connected_components(graph, directed=False)
    counts = np.bincount(labels[inv])
    return labels[inv] == int(np.argmax(counts))


def statistical_outliers(points: NDArray[Any], k: int = SOR_K,
                         std_ratio: float = SOR_STD) -> NDArray[np.bool_]:
    """True for inliers: mean k-NN distance within mean + std_ratio * std."""
    n = len(points)
    if n <= k + 1:
        return np.ones(n, bool)
    tree = cKDTree(points)
    d, _ = tree.query(points, k=k + 1)
    md = d[:, 1:].mean(1)
    return md <= md.mean() + std_ratio * md.std()


def lift_mask(
    mask: NDArray[Any],
    depth: NDArray[Any],
    K: Intrinsics,
    valid: NDArray[Any] | None = None,
    T_parent_cam: Pose | None = None,
    edges: NDArray[Any] | None = None,
    shrink_px: int = SHRINK_PX,
) -> Lifted:
    mask = np.asarray(mask, bool)
    d = np.asarray(depth)
    if edges is None:
        ok_full = np.isfinite(d) & (d > 0)
        if valid is not None:
            ok_full &= np.asarray(valid, bool)
        edges = depth_edge_mask(np.where(ok_full, d, 0.0))
    rows = np.flatnonzero(mask.any(1))
    cols = np.flatnonzero(mask.any(0))
    if len(rows) == 0:
        return Lifted(np.zeros((0, 3)), np.zeros(0, np.int64), 0)
    # work on the mask's bounding box (plus a margin for the erosion)
    r0, r1 = max(0, rows[0] - shrink_px - 1), min(mask.shape[0], rows[-1] + shrink_px + 2)
    c0, c1 = max(0, cols[0] - shrink_px - 1), min(mask.shape[1], cols[-1] + shrink_px + 2)
    mc = mask[r0:r1, c0:c1]
    dc = d[r0:r1, c0:c1]
    ok = np.isfinite(dc) & (dc > 0) & ~np.asarray(edges)[r0:r1, c0:c1]
    if valid is not None:
        ok &= np.asarray(valid, bool)[r0:r1, c0:c1]
    m = shrink_mask(mc, shrink_px) & ok
    if m.sum() < MIN_POINTS:
        m = mc & ok
    v, u = np.nonzero(m)
    v = v + r0
    u = u + c0
    pix = (v * mask.shape[1] + u).astype(np.int64)
    if len(pix) == 0:
        return Lifted(np.zeros((0, 3)), pix, int(mask.sum()))
    z = d[v, u].astype(np.float64)
    pts = unproject_pixels(u, v, z, K.K())
    eps = max(MIN_EPS, EPS_FRACTION * float(np.median(z)))
    keep = largest_cluster(pts, eps)
    pts, pix = pts[keep], pix[keep]
    keep = statistical_outliers(pts)
    pts, pix = pts[keep], pix[keep]
    if T_parent_cam is not None:
        pts = T_parent_cam.apply(pts)
    return Lifted(pts, pix, int(mask.sum()))
