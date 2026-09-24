"""Points from a depth grid: the pixel selection of the §2.2 point-cloud attributes (validity,
depth range, flying pixels, stride), unprojection with per-pixel colours from the resized image,
normals from the depth map, and normals of an unordered (map) cloud."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.cloud_attrs import CloudAttrs
from oh_my_slam.core.geometry import depth_edge_mask, unproject_pixels
from oh_my_slam.core.ply import PointCloud
from oh_my_slam.core.types import Intrinsics, Pose

MAX_GRID_SIDE = 1024
NORMAL_NEIGHBOURS = 16  # k of the k-NN plane fit for map normals
_NORMAL_CHUNK = 200_000

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


# --- normals ---------------------------------------------------------------------------------------


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


def point_normals(xyz: NDArray[Any], viewpoints: NDArray[Any], k: int = NORMAL_NEIGHBOURS
                  ) -> NDArray[np.float32]:
    """Unit normals of an unordered cloud: the smallest principal axis of each point's ``k``
    nearest neighbours, oriented towards the nearest of ``viewpoints`` (camera centres; the
    origin when there are none). Deterministic."""
    from scipy.spatial import cKDTree

    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    n = len(pts)
    out = np.zeros((n, 3))
    if n == 0:
        return out.astype(np.float32)
    kk = min(k, n)
    _, nb = cKDTree(pts).query(pts, k=kk, workers=-1)
    nb = np.asarray(nb).reshape(n, kk)
    for s in range(0, n, _NORMAL_CHUNK):
        q = pts[nb[s:s + _NORMAL_CHUNK]]
        q = q - q.mean(axis=1, keepdims=True)
        _, vecs = np.linalg.eigh(np.einsum("nki,nkj->nij", q, q))
        out[s:s + _NORMAL_CHUNK] = vecs[:, :, 0]
    vp = np.asarray(viewpoints, dtype=np.float64).reshape(-1, 3)
    if len(vp) == 0:
        vp = np.zeros((1, 3))
    _, j = cKDTree(vp).query(pts, k=1)
    flip = (out * (vp[np.asarray(j)] - pts)).sum(axis=1) < 0
    out[flip] *= -1
    return out.astype(np.float32)
