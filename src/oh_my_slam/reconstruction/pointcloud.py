"""Points from a depth grid: the pixel selection of the §2.2 point-cloud attributes (validity,
depth range, flying pixels, stride), unprojection with per-pixel colours from the resized image,
normals from the depth map, and normals of an unordered (map) cloud."""

from __future__ import annotations

import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.cloud_attrs import CloudAttrs
from oh_my_slam.core.geometry import depth_edge_mask, unproject_pixels
from oh_my_slam.core.ply import PointCloud
from oh_my_slam.core.types import Intrinsics, Pose

MAX_GRID_SIDE = 1024
NORMAL_NEIGHBOURS = 16  # k of the k-NN plane fit for map normals
_NORMAL_CHUNK = 65_536  # points per worker task

_DEFAULT = CloudAttrs()


def pixel_mask(depth: NDArray[Any], valid: NDArray[Any] | None = None,
               attrs: CloudAttrs = _DEFAULT) -> NDArray[np.bool_]:
    """Pixels that become points: finite positive depth, valid, not a flying pixel (``edge``;
    edges are found on the full valid grid), within ``min-depth``/``max-depth`` and on the
    ``stride`` lattice. The default attributes give the cloud every command emits by default."""
    d = np.asarray(depth)
    ok = np.isfinite(d) & (d > 0)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    m = ok.copy()
    if attrs.edge > 0:
        m &= ~depth_edge_mask(np.where(ok, d, 0.0), rel_threshold=attrs.edge)
    if attrs.min_depth > 0:
        m &= d >= attrs.min_depth
    if math.isfinite(attrs.max_depth):
        m &= d <= attrs.max_depth
    if attrs.stride > 1:
        lattice = np.zeros(d.shape, bool)
        lattice[::attrs.stride, ::attrs.stride] = True
        m &= lattice
    return m


def pixel_points(depth: NDArray[Any], K: Intrinsics, mask: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
    """Camera-frame points for ``mask`` pixels and their flat pixel indices (row-major)."""
    v, u = np.nonzero(mask)
    z = np.asarray(depth)[v, u].astype(np.float64)
    pts = unproject_pixels(u, v, z, K.K())
    return pts, v * mask.shape[1] + u


def frame_cloud(
    depth: NDArray[Any],
    rgb: NDArray[np.uint8],
    K: Intrinsics,
    mask: NDArray[Any],
    T_parent_cam: Pose | None = None,
) -> tuple[PointCloud, NDArray[Any]]:
    """Coloured cloud for ``mask`` pixels, optionally transformed; also returns pixel indices."""
    if rgb.shape[:2] != depth.shape:
        raise ValueError(f"image {rgb.shape[:2]} and depth {depth.shape} grids differ")
    pts, idx = pixel_points(depth, K, mask)
    if T_parent_cam is not None:
        pts = T_parent_cam.apply(pts)
    colors = rgb.reshape(-1, 3)[idx]
    return PointCloud(pts.astype(np.float32), colors), idx


# --- normals --------------------------------------------------------------------------------------


def _tangent(P: NDArray[np.float64], z: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    """Per-pixel tangent along ``axis``: the one-sided difference towards the neighbour with the
    smaller depth jump, so normals next to a depth discontinuity follow their own surface."""
    D = np.diff(P, axis=axis)
    J = np.abs(np.diff(z, axis=axis))
    J = np.where(np.isfinite(J), J, np.inf)
    shape = list(z.shape)
    fwd = np.full((*shape, 3), np.nan)
    bwd = np.full((*shape, 3), np.nan)
    jf = np.full(shape, np.inf)
    jb = np.full(shape, np.inf)
    if axis == 1:
        fwd[:, :-1], jf[:, :-1] = D, J
        bwd[:, 1:], jb[:, 1:] = D, J
    else:
        fwd[:-1], jf[:-1] = D, J
        bwd[1:], jb[1:] = D, J
    return np.where((jf <= jb)[..., None], fwd, bwd)


def depth_normals(depth: NDArray[Any], K: Intrinsics, valid: NDArray[Any] | None = None
                  ) -> NDArray[np.float32]:
    """Unit normal per pixel from the unprojected depth grid (cross product of the image-axis
    tangents), oriented towards the camera. Pixels without a usable neighbour get the viewing
    direction; invalid pixels get zeros."""
    d = np.asarray(depth, dtype=np.float64)
    ok = np.isfinite(d) & (d > 0)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    z = np.where(ok, d, np.nan)
    P = unproject_pixels(u, v, z, K.K())
    with np.errstate(invalid="ignore", divide="ignore"):
        n = np.cross(_tangent(P, z, 1), _tangent(P, z, 0))
        length = np.linalg.norm(n, axis=-1, keepdims=True)
        view = -P / np.linalg.norm(P, axis=-1, keepdims=True)
        n = np.where(np.isfinite(length) & (length > 0), n / length, view)
        n = np.where(((n * P).sum(-1) > 0)[..., None], -n, n)
    return np.where(ok[..., None], n, 0.0).astype(np.float32)


class PointNormals:
    """Normals of an unordered cloud, computed only for the points asked for, and kept.

    The normal of point ``i`` is the smallest principal axis of its ``k`` nearest neighbours in
    the *whole* cloud, oriented towards the nearest of ``viewpoints`` (camera centres; the origin
    when there are none). It depends only on the cloud, never on which other points are asked
    for or in which order: the neighbour index is built once, and every normal is computed with
    element-wise arithmetic in a fixed order (no reductions whose summation order could vary),
    so asking for a voxel-thinned subset gives exactly the rows of the full result. The cost is
    therefore proportional to the points asked for (plus one index build per cloud)."""

    def __init__(self, xyz: NDArray[Any], viewpoints: NDArray[Any], k: int = NORMAL_NEIGHBOURS
                 ) -> None:
        self.pts = np.ascontiguousarray(np.asarray(xyz, dtype=np.float64).reshape(-1, 3))
        vp = np.asarray(viewpoints, dtype=np.float64).reshape(-1, 3)
        self.viewpoints = vp if len(vp) else np.zeros((1, 3))
        self.k = max(1, min(k, len(self.pts)))
        self._normals = np.zeros((len(self.pts), 3), np.float32)
        self._done = np.zeros(len(self.pts), bool)
        self._lock = threading.Lock()
        self._trees: tuple[Any, Any] | None = None

    def _index(self) -> tuple[Any, Any]:
        if self._trees is None:
            from scipy.spatial import cKDTree

            self._trees = (cKDTree(self.pts), cKDTree(self.viewpoints))
        return self._trees

    def at(self, rows: NDArray[Any]) -> NDArray[np.float32]:
        """``(len(rows), 3)`` unit normals of the points ``rows`` (indices into the cloud)."""
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        with self._lock:
            todo = np.unique(rows[~self._done[rows]])
            if len(todo):
                self._compute(todo)
                self._done[todo] = True
            return self._normals[rows]

    def _compute(self, rows: NDArray[np.int64]) -> None:
        tree, vp_tree = self._index()
        chunks = [rows[s:s + _NORMAL_CHUNK] for s in range(0, len(rows), _NORMAL_CHUNK)]

        def work(chunk: NDArray[np.int64]) -> None:
            p = self.pts[chunk]
            _, nb = tree.query(p, k=self.k)
            nb_t = np.asarray(nb).reshape(len(chunk), self.k).T  # (k, m): neighbour j of each
            dev = []  # per axis, (k, m) offsets of the neighbours from their mean
            for axis in range(3):
                x = self.pts[:, axis][nb_t]
                total = x[0]
                for j in range(1, self.k):
                    total = total + x[j]
                dev.append(x - total / self.k)
            cov = np.empty((len(chunk), 3, 3))
            for a in range(3):
                for b in range(a, 3):
                    acc = dev[a][0] * dev[b][0]
                    for j in range(1, self.k):
                        acc = acc + dev[a][j] * dev[b][j]
                    cov[:, a, b] = cov[:, b, a] = acc
            n = np.linalg.eigh(cov)[1][:, :, 0]  # one LAPACK call per 3 x 3 matrix
            _, near = vp_tree.query(p, k=1)
            to_view = self.viewpoints[np.asarray(near)] - p
            flip = (n[:, 0] * to_view[:, 0] + n[:, 1] * to_view[:, 1]
                    + n[:, 2] * to_view[:, 2]) < 0
            n[flip] *= -1
            self._normals[chunk] = n.astype(np.float32)

        if len(chunks) == 1:
            work(chunks[0])
            return
        with ThreadPoolExecutor(max_workers=min(len(chunks), os.cpu_count() or 1)) as pool:
            list(pool.map(work, chunks))  # each chunk writes its own rows


def point_normals(xyz: NDArray[Any], viewpoints: NDArray[Any], k: int = NORMAL_NEIGHBOURS
                  ) -> NDArray[np.float32]:
    """Unit normals of every point of an unordered cloud (see :class:`PointNormals`)."""
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    return PointNormals(pts, viewpoints, k).at(np.arange(len(pts)))
