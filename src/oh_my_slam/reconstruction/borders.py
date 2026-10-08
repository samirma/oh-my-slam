"""Depth at the image border, corrected by the views that see the same surface centrally.

Monocular depth is least reliable near the image border: on the turning head of
``ainex-captures`` the keyframes that saw a wall in their outer 15 % placed it up to 6 % nearer or
farther than those that saw it near their centre (one scale and near/far tilt per keyframe cannot
take out an error that varies across the image), and the TSDF, whose band is 4 cm, kept each
placement as a layer of its own: a wall 12-15 cm thick, doubled below the light switch. Where
another view sees the surface of a border pixel near its own centre, the border pixel takes that
view's depth ratio along its ray (``correct_borders``).

These are the depth steps over a set of posed depth views; which views are fused, and when, is
the caller's (``mapping.geometry._setup``)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.geometry import project, unproject_pixels
from oh_my_slam.core.types import Intrinsics, Pose

BORDER_BAND = 0.15  # the outer share of the image, on each side, whose depth defers to others
BORDER_STEP = 4  # border pixels are measured every 4th pixel; each corrects its 4x4 cell
BORDER_NEIGHBOURS = 8  # the views, nearest by viewpoint, that may see a border centrally
BORDER_SAME = 0.1  # |log ratio| within which the two depths are one surface (else: occlusion)
BORDER_MAX_ANGLE_DEG = 60.0
BORDER_MIN_DEPTH = 0.05  # m: a point nearer to a view's camera than this is not judged by it


@dataclass(frozen=True)
class DepthView:
    """A posed depth grid: its depth, validity, intrinsics (of the grid), camera-to-map pose and
    the median of its valid depth (None: none)."""

    depth: NDArray[np.float32]
    valid: NDArray[np.bool_]
    K: Intrinsics
    T_map_cam: Pose
    median_depth: float | None


def central(u: NDArray[Any], v: NDArray[Any], w: int, h: int, band: float) -> NDArray[np.bool_]:
    """Whether the image positions (u, v) lie in the central part of a ``w`` x ``h`` image, inside
    its outer ``band`` on each side."""
    return np.asarray((u >= band * w) & (u < (1 - band) * w) & (v >= band * h)
                      & (v < (1 - band) * h))


def viewpoint_order(views: Sequence[DepthView], max_angle_deg: float = BORDER_MAX_ANGLE_DEG
                    ) -> list[list[int]]:
    """Per view, the other views whose optical axis lies within ``max_angle_deg`` of its own,
    nearest by viewpoint first (centre distance over the scene's median depth, plus 1 - the
    cosine of the axes' angle)."""
    C = np.array([v.T_map_cam.t for v in views])
    F = np.array([v.T_map_cam.R[:, 2] for v in views])
    cos = np.clip(F @ F.T, -1.0, 1.0)
    meds = [m for m in (v.median_depth for v in views) if m is not None]
    scene = max(0.5, float(np.median(meds))) if meds else 1.0
    dist = np.linalg.norm(C[:, None] - C[None], axis=2) / scene + (1.0 - cos)
    near = cos > np.cos(np.radians(max_angle_deg))
    return [[j for j in np.argsort(dist[i], kind="stable").tolist() if j != i and near[i, j]]
            for i in range(len(views))]


def correct_borders(views: Sequence[DepthView], band: float = BORDER_BAND,
                    step: int = BORDER_STEP) -> tuple[list[NDArray[np.float32]], int]:
    """(the depth of each view with its border pixels — the outer ``band`` of the image — given
    the depth of the views that see the same surface near their centre, the number of corrected
    pixels): each sampled border pixel is lifted, projected into its ``BORDER_NEIGHBOURS``
    nearest views by viewpoint and, where it lands in one's central part on the same surface
    (within ``BORDER_SAME``), its depth is scaled by their mean depth ratio along its ray; the
    correction covers the pixel's ``step`` x ``step`` cell. Border pixels no other view sees
    centrally keep their depth; every view is judged against the others' depth as it was
    before."""
    if len(views) < 2:
        return [np.asarray(v.depth, np.float32) for v in views], 0
    order = viewpoint_order(views)
    corrected = 0
    new_depths = []
    for i, view in enumerate(views):
        d = np.asarray(view.depth, np.float32)
        h, w = d.shape
        vv, uu = np.mgrid[step // 2:h:step, step // 2:w:step]
        z = d[vv, uu].astype(np.float64)
        ok = view.valid[vv, uu] & (z > 0) & ~central(uu + 0.5, vv + 0.5, w, h, band)
        # pixel indices as the TSDF reads them: u = fx x / z + cx
        cand = order[i]
        if not ok.any() or not cand:
            new_depths.append(d)
            continue
        zs = z[ok]
        X = view.T_map_cam.apply(unproject_pixels(uu[ok], vv[ok], zs, view.K.K()))
        total = np.zeros(len(zs))
        count = np.zeros(len(zs))
        for j in cand[:BORDER_NEIGHBOURS]:
            g = views[j]
            Tg = g.T_map_cam
            puv, zz = project((X - Tg.t) @ Tg.R, g.K.K())
            gh, gw = g.depth.shape
            with np.errstate(invalid="ignore"):
                inside = (zz > BORDER_MIN_DEPTH) & central(puv[:, 0] + 0.5, puv[:, 1] + 0.5, gw,
                                                           gh, band)
            idx = np.flatnonzero(inside)
            iu = np.clip(np.floor(puv[idx, 0] + 0.5).astype(np.int64), 0, gw - 1)
            iv = np.clip(np.floor(puv[idx, 1] + 0.5).astype(np.int64), 0, gh - 1)
            dg = g.depth[iv, iu].astype(np.float64)
            good = g.valid[iv, iu] & (dg > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.log(np.where(good, dg, 1.0) / zz[idx])
            same = good & (np.abs(r) <= BORDER_SAME)
            total[idx[same]] += r[same]
            count[idx[same]] += 1
        field = np.ones(z.shape)
        seen = count > 0
        vals = np.ones(len(zs))
        vals[seen] = np.exp(total[seen] / count[seen])
        field[ok] = vals
        full = np.repeat(np.repeat(field, step, axis=0), step, axis=1)[:h, :w]
        zcell = np.repeat(np.repeat(z, step, axis=0), step, axis=1)[:h, :w]
        border = ~central(np.arange(w)[None, :] + 0.5, np.arange(h)[:, None] + 0.5, w, h, band)
        with np.errstate(divide="ignore", invalid="ignore"):
            # a cell's pixels on the sampled pixel's surface (not across a depth edge)
            same = np.abs(np.log(np.where(d > 0, d, 1.0) / np.where(zcell > 0, zcell, 1.0))) \
                <= BORDER_SAME
        fix = border & (full != 1.0) & view.valid & (d > 0) & same
        corrected += int(fix.sum())
        new_depths.append(np.where(fix, d * full, d).astype(np.float32))
    return new_depths, corrected
