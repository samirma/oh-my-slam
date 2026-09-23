"""Latest wins, per pixel: parts of older keyframes that newer, well-registered keyframes
contradict are invalidated (their ``valid.png`` loses those pixels), so fusion, the cloud and the
objects stop using them.

Two tests on a 4-px grid, both directions (design step 10):
* free space — an old point projects into a new keyframe that sees clearly *behind* it;
* occlusion of old free space — a new surface lies in front of what an old keyframe observed along
  the same ray (the old ray would carve the new surface).
Margin τ(z) = max(0.15 m, 0.10 z). A cell is invalidated with 2 votes, or 1 vote beyond 1.5 τ.
Depth-edge pixels never vote.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from oh_my_slam.core.geometry import depth_edge_mask, project, unproject_pixels
from oh_my_slam.core.images import load_png, png_bytes
from oh_my_slam.core.types import Intrinsics, Pose

GRID = 4
TAU_MIN = 0.15
TAU_REL = 0.10
STRONG = 1.5
MIN_OBSERVATIONS = 100
MAX_REPROJ = 1.5
BORDER = 0.08


def tau(z: NDArray[Any]) -> NDArray[Any]:
    return np.maximum(TAU_MIN, TAU_REL * z)


@dataclass
class View:
    """Depth grid of one keyframe in the map (aligned metric depth, validity, pose)."""

    depth: NDArray[np.float32]
    valid: NDArray[np.bool_]
    K: Intrinsics  # grid intrinsics
    T_map_cam: Pose

    def usable(self) -> NDArray[np.bool_]:
        """Valid pixels away from depth edges and the image border (monocular depth of objects
        cut by the border is unreliable)."""
        ok = self.valid & (self.depth > 0)
        ok &= ~depth_edge_mask(np.where(ok, self.depth, 0.0))
        h, w = ok.shape
        mh, mw = max(1, int(BORDER * h)), max(1, int(BORDER * w))
        ok[:mh] = ok[-mh:] = False
        ok[:, :mw] = ok[:, -mw:] = False
        return ok

    def grid_points(self) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Map-frame points at the centres of GRID cells: (points, rows, cols) of cells."""
        use = self.usable()
        h, w = self.depth.shape
        vv, uu = np.mgrid[GRID // 2:h:GRID, GRID // 2:w:GRID]
        m = use[vv, uu]
        v, u = vv[m], uu[m]
        z = self.depth[v, u].astype(np.float64)
        pts = self.T_map_cam.apply(unproject_pixels(u, v, z, self.K.K()))
        return pts, v // GRID, u // GRID

    def lookup(self, pts_map: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Project map points: (inside mask, point z in this camera, observed depth there)."""
        pc = self.T_map_cam.inverse().apply(pts_map)
        uv, z = project(pc, self.K.K())
        h, w = self.depth.shape
        with np.errstate(invalid="ignore"):
            u = np.rint(uv[:, 0])
            v = np.rint(uv[:, 1])
            inside = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        d = np.zeros(len(pts_map))
        ui, vi = u[inside].astype(int), v[inside].astype(int)
        use = self.usable()
        ok = use[vi, ui]
        d_in = np.where(ok, self.depth[vi, ui], 0.0)
        d[inside] = d_in
        inside[np.flatnonzero(inside)[~ok]] = False
        return inside, z, d


MIN_OVERLAP = 200
MAX_GLOBAL_BIAS = 0.25


def _normalised(d: NDArray[Any], z: NDArray[Any]) -> NDArray[Any] | None:
    """Observed depths divided by the median observed/predicted ratio of the overlap.

    Monocular depth maps of two keyframes disagree by a smooth, largely global factor even for
    a static scene; real changes are local. Removing the overlap's median ratio keeps only
    local contradictions. ``None`` when the overlap is too small or the bias implausible."""
    if len(d) < MIN_OVERLAP:
        return None
    ratio = np.median(d / np.maximum(z, 1e-6))
    if not (1 - MAX_GLOBAL_BIAS <= ratio <= 1 + MAX_GLOBAL_BIAS):
        return None
    return d / ratio


def contradicted_cells(old: View, new_views: list[View]) -> NDArray[np.bool_]:
    """Boolean GRID-cell mask of ``old`` contradicted by the new views (a cell needs votes from
    two different new keyframes, or one vote beyond 1.5 τ)."""
    h, w = old.depth.shape
    gh, gw = (h + GRID - 1) // GRID, (w + GRID - 1) // GRID
    votes = np.zeros((gh, gw), np.int32)
    strong = np.zeros((gh, gw), bool)
    pts, rows, cols = old.grid_points()
    for nv in new_views:
        voted = np.zeros((gh, gw), bool)
        # (1) old point now in free space in front of a new surface
        if len(pts):
            inside, z, d = nv.lookup(pts)
            dn = _normalised(d[inside], z[inside])
            if dn is not None:
                zi = z[inside]
                diff = dn - zi
                t = tau(zi)
                ri, ci = rows[inside], cols[inside]
                hit = diff > t
                voted[ri[hit], ci[hit]] = True
                s = diff > STRONG * t
                strong[ri[s], ci[s]] = True
        # (2) new surface in front of what the old keyframe observed
        q, _, _ = nv.grid_points()
        if len(q):
            inside, z, d = old.lookup(q)
            dn = _normalised(d[inside], z[inside])
            if dn is not None:
                zi = z[inside]
                diff = dn - zi
                t = tau(zi)
                pc = old.T_map_cam.inverse().apply(q[inside])
                uv, _ = project(pc, old.K.K())
                r = np.clip(np.rint(uv[:, 1]).astype(int) // GRID, 0, gh - 1)
                c = np.clip(np.rint(uv[:, 0]).astype(int) // GRID, 0, gw - 1)
                hit = diff > t
                voted[r[hit], c[hit]] = True
                s = diff > STRONG * t
                strong[r[s], c[s]] = True
        votes += voted
    return (votes >= 2) | strong


def cells_to_pixels(cells: NDArray[Any], shape: tuple[int, int]) -> NDArray[np.bool_]:
    full = np.kron(cells, np.ones((GRID, GRID), bool))[: shape[0], : shape[1]]
    return ndimage.binary_dilation(full, iterations=1)


def well_registered(stats: dict[str, Any], pose_source: str) -> bool:
    if pose_source in ("identity", "multiview"):
        return True
    return (stats.get("observations", 0) >= MIN_OBSERVATIONS
            and stats.get("reproj_error", 0.0) <= MAX_REPROJ)


def apply_latest_wins(ctx: Any, records: list[Any], progress: Any) -> None:
    """Update ``valid.png`` of old keyframes contradicted by this update's keyframes."""
    new = [nf for nf in ctx.new if nf.record is not None and nf.depth is not None
           and well_registered(nf.record.stats, nf.record.pose_source)]
    if not new or not ctx.old_frames:
        return
    tx = ctx.tx
    new_views = [View(nf.depth, nf.frame.valid & (nf.depth > 0), nf.record.K_grid,
                      nf.record.T_map_cam) for nf in new]
    changed, pixels = 0, 0
    for rec in ctx.old_frames:
        d = f"per_frame/{rec.name}"
        dp = tx.current(f"{d}/depth.npy")
        if not dp.exists():
            continue
        depth = np.load(dp).astype(np.float32)
        vp = tx.current(f"{d}/valid.png")
        valid = (load_png(vp) > 0) if vp.exists() else depth > 0
        old = View(depth, valid, rec.K_grid, rec.T_map_cam)
        cells = contradicted_cells(old, new_views)
        if not cells.any():
            continue
        kill = cells_to_pixels(cells, depth.shape) & valid
        if not kill.any():
            continue
        tx.write_bytes(f"{d}/valid.png", png_bytes((valid & ~kill).astype(np.uint8) * 255))
        changed += 1
        pixels += int(kill.sum())
    ctx.notes["latest_wins"] = {"frames_changed": changed, "pixels_invalidated": pixels}
    if changed:
        progress(f"latest wins: {pixels} pixels invalidated in {changed} older keyframes")
