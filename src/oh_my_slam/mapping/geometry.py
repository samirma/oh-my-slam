"""Map geometry for an update: the coloured cloud (surface of a TSDF fusion of the valid, aligned
depth maps; colour and object id per point from the latest update that sees it), built with the
reconstruction package's fusion code. A surface is kept where ``CLOUD_MIN_VIEWS`` keyframes
updated it, or where every keyframe that has it in view sees it (``_few_views``); the places of
removed objects are drawn from the keyframes that saw through them (``_vacated``).

Object ids come from the keyframes' instance masks, which a detector draws generously: a "carpet"
mask that covers a counter top and the floor beyond it, of which only the counter was lifted into
the object (lifting keeps a mask's largest spatial cluster). A keyframe's vote for an object
therefore counts only for points inside the object's box grown by the depth noise at its viewing
distance (``attribution_margin``), so an object's points in the cloud (``segments.ply``,
``color=segment``, its ``point_count``) coincide with its box. The vote needs a third of the
keyframes that see a point, and an object detected in fewer of them wins only part of its
surface — a refrigerator detected in 7 of the ~15 keyframes that see its front, half of it — or
nothing — a dishwasher detected in 2 of ~13, a light switch on a wall. Each confirmed object
therefore also takes the unlabelled cloud points of its gate nearest to its own lifted points
(``support_labels``) — if a keyframe that detected it fused it (``fused_objects``: not a car
detected only 40-100 m away, beyond the fused depth) — and it is exported only when that makes it
visibly drawn (``objects.min_cloud_points``, at the map's sampling at its nearest detection)."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core import rle, timing
from oh_my_slam.core.geometry import project
from oh_my_slam.core.images import load_rgb
from oh_my_slam.core.ply import PointCloud, ply_bytes
from oh_my_slam.mapping import store
from oh_my_slam.mapping.objects import (
    ObjectState,
    Vacated,
    fit_points,
    label_map_for,
    load_vacated,
)
from oh_my_slam.reconstruction.consensus import (
    Views,
    free_space,
    median_ratio,
    neighbour_views,
    sample_pixels,
    spread_cells,
)
from oh_my_slam.reconstruction.depth import MAX_FACTOR
from oh_my_slam.reconstruction.fusion import (
    TsdfFusion,
    choose_voxel_size,
    fusion_step,
    member_points,
)
from oh_my_slam.reconstruction.pointcloud import pixel_mask
from oh_my_slam.segmentation.api import OBB, UNSEGMENTED

# Map cloud = surface of a fine TSDF (voxel/2, wide band so frames that disagree by a few
# centimetres still average into one surface), attributed from the latest update that sees it.
CLOUD_TRUNC_VOXELS = 8.0
# A surface voxel must be seen by this many frames (fewer in tiny maps), or by every frame that has
# it in view when fewer do (``_few_views``: a laptop at the corner of two photos).
CLOUD_MIN_VIEWS = 3
# A surface that fewer than CLOUD_MIN_VIEWS keyframes see is drawn only from keyframes whose depth
# fits the map: a scale measured with a spread (IQR / median of the per-point depth ratios) of at
# most FEW_VIEWS_MAX_SPREAD. A keyframe placed or scaled worse has nothing to check it there.
FEW_VIEWS_MAX_SPREAD = 0.1
# m: points are bucketed in cubes this size so that a keyframe projects only the points of the
# cubes its image can hold (``_Cells``): a keyframe of a street walk sees a small part of the map
CULL_CELL = 0.25
# Threads for the per-keyframe projections of the cloud's points (numpy releases the GIL in them):
# the points are split into parts of the map, judged independently, so the result is the same
# whatever the number of threads.
WORKERS = max(1, min(8, (os.cpu_count() or 2) - 2))
PROJECT_BATCH = 1_000_000  # points a thread projects at a time (bounds its temporaries)
# A map of more voxel blocks than this is fused in slabs of about as many blocks, one at a time
# (``_tiles``): the result is the same (every step is local), the memory a slab's (a block of
# the fine TSDF holds 512 voxels; a street walk fuses millions of blocks).
TILE_BLOCKS = 300_000
VIS_TOL_MIN = 0.02
VIS_TOL_REL = 0.03
LABEL_SHARE_DIVISOR = 3  # an object id needs the votes of >= 1/3 of the views that see a point
ATTRIBUTE_CHUNK = 1_000_000  # points attributed at a time (bounds the vote's memory)
# A vote for an object counts only within its box grown by max(ATTRIBUTE_MARGIN_M,
# ATTRIBUTE_MARGIN_REL · its viewing distance): the fused surface lies within a few centimetres of
# the points the box was fitted to (TSDF band 4 cm, box at the 2-98 % extent), plus the depth
# disagreement of the keyframes that saw it (p90 ~3 % of the distance after the global depth
# adjustment), which matters most across a thin box (a dishwasher front 6 cm deep at 3 m).
ATTRIBUTE_MARGIN_M = 0.05
ATTRIBUTE_MARGIN_REL = 0.03
# The surface of a confirmed object beyond its votes: the SUPPORT_NEIGHBOURS unlabelled cloud points
# nearest each of its own lifted points (at least SUPPORT_MIN_POINTS of them), within
# max(SUPPORT_RADIUS_MIN, SUPPORT_RADIUS_REL · its viewing distance) — the depth disagreement of
# the keyframes that saw it and of the fused surface — and inside its attribution gate; a point
# several objects pick goes to the one whose own points are nearest.
SUPPORT_MIN_POINTS = 30
SUPPORT_NEIGHBOURS = 4
SUPPORT_RADIUS_MIN = 0.03
SUPPORT_RADIUS_REL = 0.02
# The place of a removed object in a keyframe that detected it: its retired pixels, at or behind its
# surface there up to the depth noise max(VACATED_MARGIN_M, VACATED_MARGIN_REL · depth) in front
# (``objects.absence_tau``'s floor).
VACATED_MARGIN_M = 0.05
VACATED_MARGIN_REL = 0.08


@dataclass
class FrameData:
    rec: store.FrameRecord
    depth: NDArray[np.float32]
    valid: NDArray[np.bool_]
    rgb: NDArray[np.uint8]
    labels: NDArray[np.int32]
    is_new: bool
    # pixels left out of the fusion although valid: what several keyframes see through
    # (``consensus_depths``); None: none
    drop: NDArray[np.bool_] | None = None

    @cached_property
    def median_depth(self) -> float | None:
        """Median of its valid depth (None: none), once: ``depth`` and ``valid`` are not
        modified after the fusion's setup (``correct_borders`` and ``consensus_depths``, which
        drop this value), and ``fusion_depth_max`` asks for it per part of the map."""
        d = self.depth[self.valid & (self.depth > 0)]
        return float(np.median(d)) if len(d) else None


def _parallel[T, R](fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
    """``[fn(x) for x in items]`` on ``WORKERS`` threads (in order)."""
    items = list(items)
    if WORKERS <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(min(WORKERS, len(items))) as pool:
        return list(pool.map(fn, items))


class _Cells:
    """Points bucketed in cubes of ``CULL_CELL`` metres, to cull the points a keyframe cannot
    project into its image (``select``) before projecting them: a keyframe of a large map (a
    street walk) sees a small part of it, and projecting every point for every keyframe costs
    points x keyframes. The culling is conservative: every point that projects into the image
    (or the window) nearer than the depth limit is selected."""

    def __init__(self, pts: NDArray[Any], cell: float = CULL_CELL,
                 ids: NDArray[np.int64] | None = None) -> None:
        """``ids``: the indices ``select`` returns for ``pts`` (default: their positions)."""
        self.cell = cell
        self.rad = cell * float(np.sqrt(3.0)) / 2
        self.order = np.zeros(0, np.int64)
        self.first = np.zeros(0, np.int64)
        self.count = np.zeros(0, np.int64)
        self.centres: NDArray[Any] = np.zeros((0, 3))
        self._bound: tuple[NDArray[Any], float] | None = None  # a ball holding every cell
        self.size = len(pts)
        if not len(pts):
            return
        step = 4_000_000  # points at a time: bounded temporaries

        def keys(s: int) -> NDArray[np.int64]:
            return np.floor(np.asarray(pts[s:s + step], np.float64) / cell).astype(np.int64)

        # floor is monotonic: the lowest and highest cells are those of the extreme coordinates
        base = np.floor(np.asarray(pts.min(axis=0), np.float64) / cell).astype(np.int64)
        span = np.floor(np.asarray(pts.max(axis=0), np.float64) / cell).astype(np.int64) - base + 1
        ck = np.empty(len(pts), np.int64)
        for s in range(0, len(pts), step):
            k = keys(s) - base
            ck[s:s + step] = (k[:, 0] * span[1] + k[:, 1]) * span[2] + k[:, 2]
        order = np.argsort(ck)  # the points cell by cell
        uk, first, count = np.unique(ck[order], return_index=True, return_counts=True)
        self.order = order if ids is None else np.asarray(ids, np.int64)[order]
        self.first, self.count = first, count
        uc = np.stack([uk // (span[1] * span[2]), (uk // span[2]) % span[1], uk % span[2]], axis=1)
        self.centres = (uc + base + 0.5) * cell

    @staticmethod
    def ball(centre: NDArray[Any], radius: float) -> _Cells:
        """One cell: the ball of ``radius`` around ``centre`` (``select`` gives [0] for a
        keyframe that may see into it)."""
        out = _Cells(np.zeros((0, 3)))
        out.rad = float(radius)
        out.centres = np.asarray(centre, np.float64).reshape(1, 3)
        out.first, out.count = np.zeros(1, np.int64), np.ones(1, np.int64)
        out.order = np.zeros(1, np.int64)
        out.size = 1
        return out

    def split(self, parts: int) -> list[tuple[NDArray[np.int64], _Cells]]:
        """The points in up to ``parts`` groups of whole cells, about equal in size, each as
        (its points' indices, its cells over them: ``select`` gives positions in the group)."""
        out: list[tuple[NDArray[np.int64], _Cells]] = []
        if not len(self.centres):
            return out
        ends = np.searchsorted(np.cumsum(self.count), np.linspace(0, self.size, parts + 1)[1:-1])
        bounds = [0, *sorted(set((ends + 1).tolist()) - {0, len(self.centres)}),
                  len(self.centres)]
        for c0, c1 in pairwise(bounds):
            sub = _Cells(np.zeros((0, 3)), self.cell)
            f0 = int(self.first[c0])
            n = int(self.first[c1 - 1] + self.count[c1 - 1]) - f0
            sub.order = np.arange(n, dtype=np.int64)
            sub.first = self.first[c0:c1] - f0
            sub.count = self.count[c0:c1]
            sub.centres = self.centres[c0:c1]
            sub.size = n
            out.append((self.order[f0:f0 + n], sub))
        return out

    def points_of(self, c: NDArray[np.int64]) -> NDArray[np.int64]:
        """Indices of the points of cells ``c``."""
        n = self.count[c]
        return self.order[np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n - self.first[c], n)]

    def near(self, centre: NDArray[Any], radius: float) -> NDArray[np.int64]:
        """Indices of the points of the cells that reach within ``radius`` of ``centre``."""
        d = np.linalg.norm(self.centres - np.asarray(centre, np.float64), axis=1)
        return self.points_of(np.flatnonzero(d <= radius + self.rad))

    def select(self, fd: FrameData, zmax: float = float("inf"),
               window: tuple[float, float, float, float] | None = None) -> NDArray[np.int64]:
        """Indices of the points of the cells that may project into keyframe ``fd``'s image
        (``window``: u0, v0, u1, v1 within it) at a depth below ``zmax``."""
        if not len(self.centres):
            return np.zeros(0, np.int64)
        if len(self.centres) > 1:  # first all of them, as one ball
            if self._bound is None:
                lo, hi = self.centres.min(axis=0), self.centres.max(axis=0)
                self._bound = ((lo + hi)[None] / 2,
                               float(np.linalg.norm(hi - lo)) / 2 + self.rad)
            if not _in_frustum(self._bound[0], self._bound[1], fd, zmax, window)[0]:
                return np.zeros(0, np.int64)
        return self.points_of(np.flatnonzero(_in_frustum(self.centres, self.rad, fd, zmax,
                                                         window)))


def _in_frustum(centres: NDArray[Any], rad: float, fd: FrameData, zmax: float,
                window: tuple[float, float, float, float] | None) -> NDArray[np.bool_]:
    """Whether balls of radius ``rad`` around ``centres`` may reach into keyframe ``fd``'s image
    (``window``: u0, v0, u1, v1 within it) nearer than ``zmax``."""
    h, w = fd.depth.shape
    u0, v0, u1, v1 = window if window is not None else (0.0, 0.0, float(w), float(h))
    K = fd.rec.K_grid.K()
    cam = fd.rec.T_map_cam.inverse()
    uvc, zc = project(centres @ cam.R.T + cam.t, K)
    # a point within rad of the centre (x, y, z) projects within
    # f · rad · (1 + |x / z|) / (z - rad) of the centre's pixel along u (v: y)
    gap = 1.0 / np.maximum(zc - rad, 1e-6)
    with np.errstate(invalid="ignore"):
        mu = rad * float(K[0, 0]) * gap + rad * np.abs(uvc[:, 0] - K[0, 2]) * gap
        mv = rad * float(K[1, 1]) * gap + rad * np.abs(uvc[:, 1] - K[1, 2]) * gap
        near = (zc > -rad) & (zc - rad < zmax)
        framed = ((uvc[:, 0] > u0 - mu) & (uvc[:, 0] < u1 + mu)
                  & (uvc[:, 1] > v0 - mv) & (uvc[:, 1] < v1 + mv))
    return np.asarray(near & (framed | (zc < 2 * rad)))


def _depth_limit(fd: FrameData) -> float:
    """A depth beyond which keyframe ``fd`` sees no point (its deepest pixel, plus the visibility
    tolerance): the ``_Cells.select`` limit of ``_visible`` and ``_seen_through``."""
    d = float(fd.depth.max()) if fd.depth.size else 0.0
    return d * (1.0 + VIS_TOL_REL) + VIS_TOL_MIN if np.isfinite(d) else float("inf")


@dataclass
class MapGeometry:
    cloud: PointCloud  # map cloud, label = object id
    new_cloud: PointCloud  # points of this update's keyframes
    stats: dict[str, Any]
    nearest: dict[int, float] = field(default_factory=dict)  # nearest_detections


@dataclass
class FusedCloud:
    """The fused surface of the map's confident keyframes (``fuse_map``), before object ids are
    attributed to it: the objects of the update use it (``objects.update_objects``)."""

    frames: list[FrameData]  # the confident keyframes (their labels are set by build_geometry)
    xyz: NDArray[np.float64]
    voxel: float
    seconds: float
    focal: float = 0.0  # focal length (px) of the fused depth grids (median over the keyframes)
    depth_max: float = float("inf")  # the fusion's depth cut (``fusion_depth_max`` per keyframe)
    vacated: list[Vacated] = field(default_factory=list)  # places of removed objects (``_vacated``)
    retired: list[FrameData] = field(default_factory=list)  # every keyframe, fused or not
    consensus: dict[str, Any] = field(default_factory=dict)  # ``_Setup.consensus``


def _frame_data(ctx: Any, rec: store.FrameRecord, new_by_name: dict[str, Any]) -> FrameData:
    """A keyframe's aligned depth, validity and colour (object ids not yet set)."""
    tx = ctx.tx
    nf = new_by_name.get(rec.name)
    if nf is not None and nf.depth is not None:
        depth = nf.depth
        rgb = nf.frame.rgb
    else:
        depth = store.load_depth(tx.current, rec.name)
        rgb = load_rgb(tx.current(rec.image), max_side=max(rec.grid_width, rec.grid_height))
    valid = store.load_valid(tx.current, rec.name, depth)
    return FrameData(rec, depth, valid, rgb, np.zeros(depth.shape, np.int32), nf is not None)


def _frame_labels(ctx: Any, rec: store.FrameRecord, shape: tuple[int, ...], objs: ObjectState
                  ) -> NDArray[np.int32]:
    """Per-pixel persistent object ids of a keyframe (its instances as stored or staged)."""
    inst_p = ctx.tx.current(store.frame_file(rec.name, "instances.json"))
    insts = json.loads(inst_p.read_text()).get("instances", []) if inst_p.exists() else []
    return label_map_for(insts, (int(shape[0]), int(shape[1])), objs)


def fusion_depth_max(fd: FrameData, depth_max: float) -> float:
    """How deep keyframe ``fd`` is fused: ``depth_max`` where the keyframe placed it before its
    near/far correction (``mapping.api._adjust_depth_scales``: about its median depth m,
    d' = m · (d / m) ** exponent, the factor within [1/MAX_FACTOR, MAX_FACTOR]).

    The correction moves a keyframe's surfaces; it does not change which of them the keyframe
    contributes. Outdoors the keyframes tilt by exponents up to ~1.2 (far field 15-35 % deeper):
    cut at a fixed ``depth_max``, the keyframes that see a facade from 25-30 m no longer counted
    for it, and the facades 10-20 m from a street walk fell below ``CLOUD_MIN_VIEWS`` (street.mp4:
    8.2 M cloud points untilted, 6.9 M tilted)."""
    return _depth_cut(fd.median_depth, fd.rec, depth_max)


def _depth_cut(median: float | None, rec: store.FrameRecord, depth_max: float) -> float:
    """``fusion_depth_max`` of a keyframe whose valid depth has this median (None: no depth)."""
    e = float(rec.stats.get("depth_exponent", 1.0))
    if e == 1.0 or median is None:
        return depth_max
    return float(np.clip(median * (depth_max / median) ** e, depth_max / MAX_FACTOR,
                         depth_max * MAX_FACTOR))


# Monocular depth is least reliable near the image border: on the turning head of
# ``ainex-captures`` the keyframes that saw a wall in their outer 15 % placed it up to 6 % nearer or
# farther than those that saw it near their centre (one scale and near/far tilt per keyframe cannot
# take out an error that varies across the image), and the TSDF, whose band is 4 cm, kept each
# placement as a layer of its own: a wall 12-15 cm thick, doubled below the light switch. Where
# another keyframe sees the surface of a border pixel near its own centre, the border pixel takes
# that keyframe's depth ratio (``correct_borders``).
BORDER_BAND = 0.15  # the outer share of the image, on each side, whose depth defers to others
BORDER_STEP = 4  # border pixels are measured every 4th pixel; each corrects its 4x4 cell
BORDER_NEIGHBOURS = 8  # the keyframes, nearest by viewpoint, that may see a border centrally
BORDER_SAME = 0.1  # |log ratio| within which the two depths are one surface (else: occlusion)
BORDER_MAX_ANGLE_DEG = 60.0


def _central(u: NDArray[Any], v: NDArray[Any], w: int, h: int, band: float) -> NDArray[np.bool_]:
    return np.asarray((u >= band * w) & (u < (1 - band) * w) & (v >= band * h)
                      & (v < (1 - band) * h))


def _viewpoint_order(frames: list[FrameData], max_angle_deg: float = BORDER_MAX_ANGLE_DEG
                     ) -> list[list[int]]:
    """Per keyframe, the other keyframes whose optical axis lies within ``max_angle_deg`` of its
    own, nearest by viewpoint first (centre distance over the scene's median depth, plus
    1 - the cosine of the axes' angle)."""
    C = np.array([fd.rec.T_map_cam.t for fd in frames])
    F = np.array([fd.rec.T_map_cam.R[:, 2] for fd in frames])
    cos = np.clip(F @ F.T, -1.0, 1.0)
    meds = [m for m in (fd.median_depth for fd in frames) if m is not None]
    scene = max(0.5, float(np.median(meds))) if meds else 1.0
    dist = np.linalg.norm(C[:, None] - C[None], axis=2) / scene + (1.0 - cos)
    near = cos > np.cos(np.radians(max_angle_deg))
    return [[j for j in np.argsort(dist[i], kind="stable").tolist() if j != i and near[i, j]]
            for i in range(len(frames))]


def correct_borders(frames: list[FrameData], band: float = BORDER_BAND,
                    step: int = BORDER_STEP) -> int:
    """Give each keyframe's border pixels (outer ``band`` of the image) the depth of the
    keyframes that see the same surface near their centre (in place; see ``BORDER_BAND``): each
    sampled border pixel is lifted, projected into its ``BORDER_NEIGHBOURS`` nearest keyframes by
    viewpoint and, where it lands in one's central part on the same surface (within
    ``BORDER_SAME``), its depth is scaled by their mean depth ratio along its ray; the correction
    covers the pixel's ``step`` x ``step`` cell. Border pixels no other keyframe sees centrally
    keep their depth. Returns the number of corrected pixels."""
    if len(frames) < 2:
        return 0
    order = _viewpoint_order(frames)
    corrected = 0
    new_depths = []
    for i, fd in enumerate(frames):
        d = np.asarray(fd.depth, np.float32)
        h, w = d.shape
        vv, uu = np.mgrid[step // 2:h:step, step // 2:w:step]
        z = d[vv, uu].astype(np.float64)
        ok = fd.valid[vv, uu] & (z > 0) & ~_central(uu + 0.5, vv + 0.5, w, h, band)
        # pixel indices as the TSDF reads them: u = fx x / z + cx
        cand = order[i]
        if not ok.any() or not cand:
            new_depths.append(d)
            continue
        K = fd.rec.K_grid
        T = fd.rec.T_map_cam
        zs = z[ok]
        X = np.column_stack([(uu[ok] - K.cx) / K.fx * zs, (vv[ok] - K.cy) / K.fy * zs,
                             zs]) @ T.R.T + T.t
        total = np.zeros(len(zs))
        count = np.zeros(len(zs))
        for j in cand[:BORDER_NEIGHBOURS]:
            g = frames[j]
            Kg, Tg = g.rec.K_grid, g.rec.T_map_cam
            Y = (X - Tg.t) @ Tg.R
            zz = Y[:, 2]
            front = zz > 0.05
            with np.errstate(divide="ignore", invalid="ignore"):
                pu = Kg.fx * Y[:, 0] / zz + Kg.cx
                pv = Kg.fy * Y[:, 1] / zz + Kg.cy
            gh, gw = g.depth.shape
            inside = front & _central(pu + 0.5, pv + 0.5, gw, gh, band)
            idx = np.flatnonzero(inside)
            iu = np.clip(np.floor(pu[idx] + 0.5).astype(np.int64), 0, gw - 1)
            iv = np.clip(np.floor(pv[idx] + 0.5).astype(np.int64), 0, gh - 1)
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
        border = ~_central(np.arange(w)[None, :] + 0.5, np.arange(h)[:, None] + 0.5, w, h, band)
        with np.errstate(divide="ignore", invalid="ignore"):
            # a cell's pixels on the sampled pixel's surface (not across a depth edge)
            same = np.abs(np.log(np.where(d > 0, d, 1.0) / np.where(zcell > 0, zcell, 1.0))) \
                <= BORDER_SAME
        fix = border & (full != 1.0) & fd.valid & (d > 0) & same
        corrected += int(fix.sum())
        new_depths.append(np.where(fix, d * full, d).astype(np.float32))
    for fd, nd in zip(frames, new_depths, strict=True):
        fd.depth = nd
        fd.__dict__.pop("median_depth", None)  # of the corrected depth from now on
    return corrected


# Neighbouring keyframes still disagree by ~3 % in depth after the depth adjustment (p10-p90
# 0.96-1.03 on the office's desk legs and monitors, outliers 7-10 %), and the TSDF (a 4 cm band)
# keeps each placement beyond the band as a layer of its own: legs come out fat, doubled or
# tripled, monitors as slabs with offset copies. Before fusion the keyframes are therefore made to
# agree, as MVS depth fusion does (``reconstruction.consensus``): every CONSENSUS_STEP-th pixel
# (its CONSENSUS_STEP x CONSENSUS_STEP cell) is projected into the keyframes whose optical axis
# lies within CONSENSUS_MAX_ANGLE_DEG, up to CONSENSUS_NEIGHBOURS of them nearest by viewpoint.
# Where one sees the same surface there (within CONSENSUS_SAME in log ratio), its depth along the
# pixel's ray is collected, and the cell takes the median of these and its own. Where at least
# CONSENSUS_FREE_MIN keyframes see beyond the pixel's point (by more than CONSENSUS_FREE: it lies
# in their free space), more than CONSENSUS_FREE_SHARE of those that see through it or see it, the
# cell is left out of the fusion (``FrameData.drop``). The nearest keyframes by viewpoint alone
# (the sequence neighbours) agree with each other: the offset copies come from keyframes of other
# passes 40-55° away (the office's monitors 9-15 % nearer from the start of the walk than from its
# middle), so every keyframe in view counts.
CONSENSUS_STEP = 2
CONSENSUS_NEIGHBOURS = 96
CONSENSUS_MAX_ANGLE_DEG = 60.0
CONSENSUS_SAME = 0.03
CONSENSUS_FREE = 0.03
CONSENSUS_FREE_MIN = 2
CONSENSUS_FREE_SHARE = 0.5
CONSENSUS_BATCH = 16  # neighbours projected at a time (bounds the temporaries: 16 x samples)


def _vouched(fd: FrameData) -> NDArray[np.bool_]:
    """The pixels whose depth keyframe ``fd`` vouches for to the others: valid, edge-free
    (``pixel_mask``), not left out (``drop``)."""
    m = pixel_mask(fd.depth, fd.valid)
    return m if fd.drop is None else m & ~fd.drop


def consensus_depths(frames: list[FrameData], step: int = CONSENSUS_STEP,
                     neighbours: int = CONSENSUS_NEIGHBOURS) -> tuple[int, int]:
    """Give each keyframe's pixels the consensus depth of the keyframes that see them, and leave
    out those that several keyframes see through (in place, in memory: the stored depth is
    unchanged; see ``CONSENSUS_STEP``). A keyframe vouches for its valid, edge-free pixels that
    are not left out (``_vouched``); every keyframe is judged against the others' depth as it was
    before. A keyframe of an older update is not one that sees through a pixel: what it saw
    through may have been placed there since (latest wins).
    Returns (pixels whose depth changed, pixels left out)."""
    if len(frames) < 2:
        return 0, 0
    order = _viewpoint_order(frames, CONSENSUS_MAX_ANGLE_DEG)
    views = Views.of([fd.depth for fd in frames], _parallel(_vouched, frames))

    def judge(i: int) -> tuple[NDArray[np.float32], NDArray[np.bool_]] | None:
        fd = frames[i]
        own = fd.valid & (fd.depth > 0)
        nbs = order[i][:neighbours]
        if not nbs:
            return None
        s = sample_pixels(fd.depth, own, fd.rec.K_grid, step)
        n = len(s.z)
        if not n:
            return None
        along = np.full((n, len(nbs)), np.nan)
        support = np.zeros(n, np.int64)
        through = np.zeros(n, np.int64)
        for b0 in range(0, len(nbs), CONSENSUS_BATCH):
            part = nbs[b0:b0 + CONSENSUS_BATCH]
            # an older update's keyframe does not carve: what it saw through may stand there now
            carves = np.array([frames[j].rec.update_id >= fd.rec.update_id for j in part])
            kk, on, depth, thru = neighbour_views(
                s, fd.rec.T_map_cam, views, part, [frames[j].rec.K_grid for j in part],
                [frames[j].rec.T_map_cam for j in part], carves, CONSENSUS_SAME, CONSENSUS_FREE)
            along[on, b0 + kk] = depth
            support += np.bincount(on, minlength=n)
            through += np.bincount(thru, minlength=n)
        ratio = median_ratio(s.z, along)
        drop = free_space(through, support, CONSENSUS_FREE_MIN, CONSENSUS_FREE_SHARE)
        return spread_cells(fd.depth, own, s, ratio, drop, step, CONSENSUS_SAME)

    results = _parallel(judge, range(len(frames)))
    moved = dropped = 0
    for fd, res in zip(frames, results, strict=True):
        if res is None:
            continue
        depth, drop = res
        moved += int(np.count_nonzero(depth != fd.depth))
        dropped += int(np.count_nonzero(drop))
        fd.depth = depth
        fd.drop = drop if fd.drop is None else fd.drop | drop
        fd.__dict__.pop("median_depth", None)  # of the consensus depth from now on
    return moved, dropped


def _map_depth_max(medians: list[Any]) -> tuple[float, float]:
    """(voxel, depth cut) of the map's fusion from its keyframes' median depths."""
    med = float(np.median(medians)) if medians else 2.0
    return choose_voxel_size(med), float(np.clip(2.5 * med, 3.0, 30.0))


def keyframe_depth_cuts(ctx: Any, records: list[store.FrameRecord]) -> dict[int, float]:
    """Per keyframe index, how deep the map's fusion would fuse it now (``fusion_depth_max``,
    from the keyframes' depth and validity as stored or staged): the depth within which the
    map holds surfaces drawn from it."""
    new_by_name = {nf.kf.name: nf for nf in ctx.new if nf.record is not None}
    medians: dict[int, float | None] = {}
    for rec in records:
        nf = new_by_name.get(rec.name)
        depth = nf.depth if nf is not None and nf.depth is not None else store.load_depth(
            ctx.tx.current, rec.name)
        valid = store.load_valid(ctx.tx.current, rec.name, depth)
        d = depth[valid & (depth > 0)]
        medians[rec.index] = float(np.median(d)) if len(d) else None
    _, depth_max = _map_depth_max([m for m in medians.values() if m is not None])
    return {rec.index: _depth_cut(medians[rec.index], rec, depth_max) for rec in records}


@dataclass
class _Blocks:
    """What a fusion of keyframes integrates (``_frame_blocks``)."""

    fused: list[NDArray[np.bool_]]  # per keyframe, the pixels it fuses
    order: list[int]  # the keyframes in fusion order
    coords: dict[int, NDArray[np.int32]]  # per keyframe, the voxel blocks it updates
    n_fused: int  # keyframes with a pixel to fuse
    block: float  # block edge, metres


def _fused_pixels(fd: FrameData, depth_max: float) -> NDArray[np.bool_]:
    """The pixels keyframe ``fd`` fuses: valid and edge-free (``pixel_mask``), within its fused
    depth (``fusion_depth_max``), not left out (``drop``)."""
    m = pixel_mask(fd.depth, fd.valid) & (fd.depth < fusion_depth_max(fd, depth_max))
    return m if fd.drop is None else m & ~fd.drop


def _frame_blocks(frames: list[FrameData], voxel: float, depth_max: float,
                  region: tuple[NDArray[Any], NDArray[Any]] | None = None) -> _Blocks:
    """The pixels and voxel blocks each keyframe fuses; with ``region``, blocks only of the
    keyframes whose frustum may reach the box (the others count among the fused keyframes)."""
    fused = [_fused_pixels(fd, depth_max) for fd in frames]
    order = sorted(range(len(frames)), key=lambda i: frames[i].rec.order_key)
    probe = TsdfFusion(voxel, depth_max, block_count=1, trunc_voxels=CLOUD_TRUNC_VOXELS)
    bs = probe.block_size
    reach: _Cells | None = None
    if region is not None:
        lo, hi = np.asarray(region[0], np.float64), np.asarray(region[1], np.float64)
        outer = (np.floor(lo / bs).astype(np.int64) - 1, np.floor(hi / bs).astype(np.int64) + 1)
        reach = _Cells.ball((outer[0] + outer[1] + 1) * bs / 2,
                            float(np.linalg.norm((outer[1] - outer[0] + 1) * bs)) / 2)
    coords: dict[int, NDArray[np.int32]] = {}
    n_fused = 0
    for i in order:
        fd = frames[i]
        cut = fusion_depth_max(fd, depth_max)
        if reach is not None and not len(reach.select(fd, cut + probe.trunc)):
            n_fused += _fuses(fused[i])
            continue
        c = probe.block_coords(np.where(fused[i], fd.depth, 0.0), fd.rec.K_grid.K(),
                               fd.rec.T_map_cam, depth_max=cut)
        if c is not None:
            coords[i] = c
            n_fused += 1
    return _Blocks(fused, order, coords, n_fused, bs)


def fused_cloud_points(frames: list[FrameData], voxel: float, depth_max: float,
                       vacated: list[Vacated] | None = None,
                       retired: list[FrameData] | None = None,
                       region: tuple[NDArray[Any], NDArray[Any]] | None = None,
                       blocks: _Blocks | None = None) -> NDArray[np.float64]:
    """Surface points of a fine TSDF of all frames' valid, edge-free depth, each frame up to
    ``depth_max`` as it placed it before its near/far correction (``fusion_depth_max``).

    Each keyframe's monocular depth disagrees with its neighbours by a few percent even after
    alignment, so back-projecting every frame leaves one offset copy of each surface per view;
    the TSDF averages them into a single surface. A surface is kept when ``CLOUD_MIN_VIEWS``
    frames fused it, or, where fewer frames have it in view, when all of them did
    (``_few_views``): speckle that other frames look at and do not see is dropped, a laptop at
    the corner of two photos is not. Frames are fused in ``FrameRecord.order_key`` order and the
    points are returned sorted, so the cloud does not depend on the keyframes' order within an
    update.

    ``vacated``: the places of removed objects, drawn from the keyframes that saw through them
    (``_vacated``; ``retired``: the keyframes whose pixels were retired, fused or not).

    ``region`` (lowest, highest corner): only the points inside this box, the same as those of
    the whole fusion there — every step is local (a voxel's values depend only on the frames, a
    surface point on its two voxels, the tests of ``_few_views`` and ``_vacated`` on the point),
    so only the voxel blocks within a block of the box are fused. ``blocks``: the keyframes'
    blocks (``_frame_blocks`` of the same keyframes, computed once for several regions).
    """
    lo = hi = None
    if region is not None:
        lo, hi = np.asarray(region[0], np.float64), np.asarray(region[1], np.float64)
    if blocks is None:
        blocks = _frame_blocks(frames, voxel, depth_max, region)
    fused, order, coords, bs = blocks.fused, blocks.order, blocks.coords, blocks.block
    views = max(1, min(CLOUD_MIN_VIEWS, blocks.n_fused))
    if lo is None or hi is None:
        tiles: list[_Tile] = _tiles(coords, bs)
    else:
        outer = (np.floor(lo / bs).astype(np.int64) - 1, np.floor(hi / bs).astype(np.int64) + 1)
        tiles = [(outer[0], outer[1], lambda p: np.asarray(np.all((p >= lo) & (p <= hi), axis=1)))]
    parts = []
    for t_lo, t_hi, inside in tiles:
        mine: dict[int, NDArray[np.int32]] = {}
        for i, c in coords.items():
            sel = c if t_lo is None or t_hi is None else c[np.all((c >= t_lo) & (c <= t_hi),
                                                                  axis=1)]
            if len(sel):
                mine[i] = sel
        if not mine:
            continue
        n_blocks = len(_unique_blocks(list(mine.values())))
        fusion = TsdfFusion(voxel, depth_max, block_count=n_blocks + 1,
                            trunc_voxels=CLOUD_TRUNC_VOXELS)
        for i in order:
            if i in mine:
                fd = frames[i]
                fusion.integrate(np.where(fused[i], fd.depth, 0.0), fd.rec.K_grid.K(),
                                 fd.rec.T_map_cam, depth_max=fusion_depth_max(fd, depth_max),
                                 blocks=mine[i])
        del mine
        parts.append(_surface(fusion, views, inside, frames, fused, depth_max, vacated,
                              retired))
    pts = np.concatenate(parts) if parts else np.zeros((0, 3), np.float32)
    # Open3D's parallel hash map returns the points in no fixed order
    pts = pts[np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))]
    return pts.astype(np.float64)


def _fuses(mask: NDArray[np.bool_]) -> int:
    """1 when a frame whose fused pixels are ``mask`` counts among the fused frames (it has a
    pixel on the fusion's grid: ``fusion_step``), else 0."""
    step = fusion_step(mask.shape)
    return int(bool(mask[::step, ::step].any()))


def _unique_blocks(coords: list[NDArray[np.int32]]) -> NDArray[np.int64]:
    """The distinct block coordinates (N x 3) of several lists of them."""
    c = np.concatenate(coords).astype(np.int64) + (1 << 20)  # block coordinates within ±2^20
    key = np.unique((c[:, 0] << 42) | (c[:, 1] << 21) | c[:, 2])
    m = (1 << 21) - 1
    return np.stack([key >> 42, (key >> 21) & m, key & m], axis=1) - (1 << 20)


_Inside = Callable[[NDArray[Any]], NDArray[np.bool_]]
_Tile = tuple[NDArray[Any] | None, NDArray[Any] | None, _Inside | None]


def _slab(axis: int, s0: int, s1: int, block: float) -> _Inside:
    """Whether points lie in the blocks ``s0`` to ``s1`` (excluded) along ``axis``."""
    def inside(p: NDArray[Any]) -> NDArray[np.bool_]:
        b = np.floor(np.asarray(p[:, axis], np.float64) / block)
        return np.asarray((b >= s0) & (b < s1))
    return inside


def _tiles(coords: dict[int, NDArray[np.int32]], block: float) -> list[_Tile]:
    """The map's voxel blocks in slabs of at most about ``TILE_BLOCKS`` blocks along the axis
    they spread most, fused one at a time (``fused_cloud_points``): (lowest, highest block of the
    slab grown by a block — the neighbours its surface points need —, which points are the
    slab's). One slab covering everything (no bounds) when the map is small."""
    if not coords:
        return []
    allb = _unique_blocks(list(coords.values()))
    if len(allb) <= TILE_BLOCKS:
        return [(None, None, None)]
    axis = int(np.argmax(np.ptp(allb, axis=0)))
    k = int(np.ceil(len(allb) / TILE_BLOCKS))
    at = np.sort(allb[:, axis])
    cuts = sorted(set(at[(np.arange(1, k) * len(at)) // k].tolist()))  # slab starts (blocks)
    starts = [int(at[0]), *[c for c in cuts if c > at[0]]]
    ends = [*starts[1:], int(at[-1]) + 1]
    big = np.iinfo(np.int32).max // 2
    out: list[_Tile] = []
    for s0, s1 in zip(starts, ends, strict=True):
        lo = np.full(3, -big, np.int64)
        hi = np.full(3, big, np.int64)
        lo[axis], hi[axis] = s0 - 1, s1
        out.append((lo, hi, _slab(axis, s0, s1, block)))
    return out


def _surface(fusion: TsdfFusion, views: int,
             inside: Callable[[NDArray[Any]], NDArray[np.bool_]] | None,
             frames: list[FrameData], fused: list[NDArray[np.bool_]], depth_max: float,
             vacated: list[Vacated] | None, retired: list[FrameData] | None
             ) -> NDArray[np.float32]:
    """The kept surface points of a fusion (``fused_cloud_points``; ``views``: the frames a
    surface needs), of those ``inside`` says are its own (default: all). The fusion is freed."""
    def own(p: NDArray[np.float32]) -> NDArray[np.float32]:
        return p if inside is None else p[inside(p)]

    pts = own(_extract(fusion, views))
    if views <= 1:
        return pts
    two = own(_extract(fusion, 2)) if views > 2 else None
    low = own(_extract(fusion, 1))
    fusion.release()  # the TSDF is no longer needed: free it before the points are judged
    few = low[~_member(low, pts)]  # the surfaces that fewer than ``views`` frames updated
    del low
    weight = (np.where(_member(few, two), 2, 1).astype(np.int8) if two is not None
              else np.ones(len(few), np.int8))
    del two
    few_cells = _Cells(few)
    add = _few_views(few, weight, frames, fused, depth_max, views, few_cells)
    keep = np.ones(len(pts), bool)
    if vacated:
        keep, witnessed = _vacated(pts, few, frames, vacated, retired or frames, few_cells)
        add |= witnessed
    return np.concatenate([pts[keep], few[add]])


def _extract(fusion: TsdfFusion, views: int) -> NDArray[np.float32]:
    """The fused surface where at least ``views`` frames updated the voxels (Open3D's weight
    counts the frames that updated a voxel: the ones that saw its surface or saw through it),
    as Open3D gives it (float32)."""
    pts = fusion.extract_points(weight_threshold=views - 0.5)  # Open3D keeps weight > threshold
    return np.ascontiguousarray(pts, dtype=np.float32).reshape(-1, 3)


def _member(a: NDArray[Any], b: NDArray[Any]) -> NDArray[np.bool_]:
    """Whether each point of ``a`` is one of ``b`` (the same float32 coordinates: surface points
    that TSDF extractions at two weight thresholds share are bit-identical)."""
    return member_points(a, b)


def _trusted(fd: FrameData) -> bool:
    """Whether a keyframe may draw alone what fewer than ``CLOUD_MIN_VIEWS`` keyframes see: its
    depth scale was measured with a spread of at most ``FEW_VIEWS_MAX_SPREAD``."""
    spread = fd.rec.stats.get("depth_scale_spread")
    return spread is None or float(spread) <= FEW_VIEWS_MAX_SPREAD


def _few_views(pts: NDArray[Any], weight: NDArray[Any], frames: list[FrameData],
               fused: list[NDArray[np.bool_]], depth_max: float, views: int,
               cells: _Cells | None = None) -> NDArray[np.bool_]:
    """Which surface points that fewer than ``views`` frames updated (``weight`` of them) stay:
    as many frames updated them as have them in view, and a frame that sees them (``_visible``'s
    depth agreement) is ``_trusted``. A frame has a point in view when the point lies within its
    fused depth (``fusion_depth_max``) and projects onto a pixel it fuses (``fused``: valid,
    edge-free, within that depth), whatever the pixel shows — a nearer surface, the point, or a
    farther one.

    The weight, not a count of the frames whose depth lies near the point, decides: a surface
    that 5 frames see has, a centimetre or two beside it, a second zero crossing that 1 or 2 of
    them updated (the rim of the TSDF band), which only the weight tells from the surface.

    A surface seen by one or two keyframes because no other keyframe looks there (a laptop at the
    corner of two photos, the half of a monitor that one photo shows, the sill that the latest
    photos show where a removed cup stood) is kept; one that other keyframes look at and do not
    see (they see a nearer surface, or through it) needs ``views`` of them, as before.

    A point that more frames have in view than updated it is dropped whatever the other frames
    show, so it is not projected again (most of these points are the offset copies of far
    surfaces that one keyframe's depth placed apart from the others'). ``cells``: ``pts``
    bucketed (``_Cells``)."""
    if not len(pts):
        return np.zeros(0, bool)
    cells = cells if cells is not None else _Cells(pts)
    weight = np.asarray(weight)
    keep = np.zeros(len(pts), bool)
    groups = cells.split(4 * WORKERS)  # parts of the map, judged independently

    def judge(group: tuple[NDArray[np.int64], _Cells]) -> NDArray[np.bool_]:
        ids, sub = group
        return _few_views_part(pts[ids], weight[ids], frames, fused, depth_max, views, sub)

    for (ids, _), kept in zip(groups, _parallel(judge, groups), strict=True):
        keep[ids] = kept
    return keep


def _few_views_part(pts: NDArray[Any], weight: NDArray[Any], frames: list[FrameData],
                    fused: list[NDArray[np.bool_]], depth_max: float, views: int,
                    cells: _Cells) -> NDArray[np.bool_]:
    """``_few_views`` of the points ``pts`` (``cells``: over them)."""
    limit = np.minimum(weight, views)  # the most frames that may have it in view
    in_view = np.zeros(len(pts), np.int32)
    trusted = np.zeros(len(pts), bool)
    alive = np.ones(len(pts), bool)
    n_alive = len(pts)
    for fd, m in zip(frames, fused, strict=True):
        if n_alive == 0:
            break
        if n_alive < cells.size // 2:  # re-bucket the points still undecided
            ids = np.flatnonzero(alive)
            cells = _Cells(pts[ids], ids=ids)
        cut = fusion_depth_max(fd, depth_max)
        K = fd.rec.K_grid.K()
        cam = fd.rec.T_map_cam.inverse()
        h, w = fd.depth.shape
        cand = cells.select(fd, cut)  # cull the cells outside the frustum
        cand = cand[alive[cand]]
        trust = _trusted(fd)
        for b in range(0, len(cand), PROJECT_BATCH):  # bounded temporaries per thread
            sel = cand[b:b + PROJECT_BATCH]
            uv, z = project(pts[sel] @ cam.R.T + cam.t, K)
            with np.errstate(invalid="ignore"):
                u = np.floor(uv[:, 0] + 0.5)
                v = np.floor(uv[:, 1] + 0.5)
                ok = (z > 0) & (z < cut) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            sel, z = sel[ok], z[ok]
            uu, vv = u[ok].astype(np.int64), v[ok].astype(np.int64)
            on = m[vv, uu]
            sel, z, uu, vv = sel[on], z[on], uu[on], vv[on]
            in_view[sel] += 1
            if trust:
                sees = np.abs(fd.depth[vv, uu] - z) < np.maximum(VIS_TOL_MIN, VIS_TOL_REL * z)
                trusted[sel[sees]] = True
            dead = sel[in_view[sel] > limit[sel]]
            alive[dead] = False
            n_alive -= len(dead)
    return (weight >= np.minimum(views, in_view)) & trusted


def _vacated_region(pts: NDArray[np.float64], place: Vacated, by_name: dict[str, FrameData],
                    cells: _Cells | None = None) -> NDArray[np.bool_]:
    """Whether each point lies in a removed object's place: it projects onto a pixel retired from
    a keyframe that detected the object, at or behind the object's surface there (up to the depth
    noise, max(VACATED_MARGIN_M, VACATED_MARGIN_REL · depth), in front of it): the object itself
    and what it hid from that keyframe; or it lies in the object's box grown by the depth noise
    (``attribution_margin``): what stood by it, such as its shadow on the surface it stood on."""
    cells = cells if cells is not None else _Cells(pts)
    out = np.zeros(len(pts), bool)
    if place.box is not None and len(pts):
        margin = attribution_margin(place.obs_depth)
        c = cells.near(place.box.center, float(np.linalg.norm(place.box.size)) / 2 + margin)
        out[c[place.box.contains(pts[c], margin)]] = True
    for name, enc in place.masks.items():
        fd = by_name.get(name)
        if fd is None:
            continue
        mask = rle.decode(enc)
        if mask.shape != fd.depth.shape or not mask.any():
            continue
        rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
        c = cells.select(fd, window=(cols[0] - 0.5, rows[0] - 0.5, cols[-1] + 0.5, rows[-1] + 0.5))
        cam = fd.rec.T_map_cam.inverse()
        uv, z = project(pts[c] @ cam.R.T + cam.t, fd.rec.K_grid.K())
        h, w = mask.shape
        with np.errstate(invalid="ignore"):
            u = np.floor(uv[:, 0] + 0.5)
            v = np.floor(uv[:, 1] + 0.5)
            idx = np.flatnonzero((z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h))
        uu, vv = u[idx].astype(np.int64), v[idx].astype(np.int64)
        z = z[idx]
        idx = c[idx]
        d = fd.depth[vv, uu].astype(np.float64)
        hit = mask[vv, uu] & (d > 0)
        hit &= z >= d - np.maximum(VACATED_MARGIN_M, VACATED_MARGIN_REL * d)
        out[idx[hit]] = True
    return out


def _candidates(fd: FrameData, pts: NDArray[np.float64], cells: _Cells | None
                ) -> NDArray[np.int64] | None:
    """The points keyframe ``fd`` may see (``_Cells.select`` up to ``_depth_limit``), or None
    (all) without ``cells``."""
    return None if cells is None else cells.select(fd, _depth_limit(fd))


def _residuals(fd: FrameData, pts: NDArray[np.float64], cells: _Cells | None = None
               ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(how far beyond each point keyframe ``fd`` sees: the depth of the valid pixel it projects
    onto minus its own depth, NaN where it projects onto none; its depth)."""
    cand = _candidates(fd, pts, cells)
    q = pts if cand is None else pts[cand]
    cam = fd.rec.T_map_cam.inverse()
    uv, z = project(q @ cam.R.T + cam.t, fd.rec.K_grid.K())
    h, w = fd.depth.shape
    with np.errstate(invalid="ignore"):
        u = np.floor(uv[:, 0] + 0.5)
        v = np.floor(uv[:, 1] + 0.5)
        idx = np.flatnonzero((z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h))
    uu, vv = u[idx].astype(np.int64), v[idx].astype(np.int64)
    ok = fd.valid[vv, uu]
    idx, uu, vv = idx[ok], uu[ok], vv[ok]
    res = np.full(len(pts), np.nan)
    depth = np.full(len(pts), np.nan)
    at = idx if cand is None else cand[idx]
    res[at] = fd.depth[vv, uu] - z[idx]
    depth[at] = z[idx]
    return res, depth


def _seen_through(fd: FrameData, pts: NDArray[np.float64], cells: _Cells | None = None
                  ) -> NDArray[np.bool_]:
    """Whether keyframe ``fd`` sees clearly behind each point: it projects onto a valid pixel whose
    depth lies beyond it by more than the visibility tolerance (``_visible``)."""
    res, z = _residuals(fd, pts, cells)
    with np.errstate(invalid="ignore"):
        return np.asarray(res >= np.maximum(VIS_TOL_MIN, VIS_TOL_REL * z))


def _witnessed(witnesses: list[FrameData], q: NDArray[np.float64], cells: _Cells | None = None
               ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """(which points the witnesses of a removed object see, which they see through), from the
    witnesses together: per point, the median over the witnesses that project it onto a valid
    pixel of how far beyond it they see, within the visibility tolerance (``_visible``) or beyond
    it. The witnesses are the place's latest observation as one: their monocular depth of the
    empty place disagrees by a few percent (``office_sequence``: the two photos that see the cup
    gone place the sill 3.5 cm in front of and 3.2 cm behind the surface the other keyframes
    agree on), so one witness that sees a little farther than the others does not carve the
    surface they show (a hole where the cup stood), and a surface only one of them places apart
    is not drawn as a second copy."""
    res = np.full((len(witnesses), len(q)), np.nan)
    z = np.full(len(q), np.nan)
    for k, fd in enumerate(witnesses):
        res[k], zk = _residuals(fd, q, cells)
        z = np.where(np.isnan(z), zk, z)
    viewed = ~np.isnan(res).all(axis=0)
    med = np.full(len(q), np.nan)
    if viewed.any():
        med[viewed] = np.nanmedian(res[:, viewed], axis=0)
    with np.errstate(invalid="ignore"):
        tol = np.maximum(VIS_TOL_MIN, VIS_TOL_REL * z)
        return np.asarray(np.abs(med) < tol), np.asarray(med >= tol)


def _vacated(pts: NDArray[Any], few: NDArray[Any], frames: list[FrameData],
             vacated: list[Vacated], retired: list[FrameData], few_cells: _Cells | None = None
             ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Latest wins in the places of removed objects (``Vacated``): what the keyframes that saw
    through an object (its witnesses) see there is the map's surface. Returns (which of the
    well-seen surface points ``pts`` stay, which of the surface points fewer frames fused
    ``few`` are added).

    The pixels of the keyframes that detected the object are retired, so the surface it stood on
    or hid is seen only by its witnesses, where a surface usually needs ``CLOUD_MIN_VIEWS``
    views. In the place, a point that a witness sees is kept however few frames fused it, and a
    point that a witness sees through, which neither a witness nor a keyframe of a later update
    sees, is dropped: what is left of the object, drawn by keyframes that did not detect it or by
    pixels beside its masks. A later update that sees the place again is fused like any other."""
    by_name = {fd.rec.name: fd for fd in retired}
    fused = {fd.rec.name: fd for fd in frames}
    keep = np.ones(len(pts), bool)
    add = np.zeros(len(few), bool)
    cells = {False: _Cells(pts), True: few_cells if few_cells is not None else _Cells(few)}

    def judge(place: Vacated) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """(indices of ``pts`` dropped, of ``few`` added) in one place."""
        drop, added = np.zeros(0, np.int64), np.zeros(0, np.int64)
        witnesses = [fused[n] for n in place.witnesses if n in fused]
        if not witnesses:
            return drop, added
        later = [fd for fd in frames if fd.rec.update_id > place.update]
        for cloud, is_few in ((pts, False), (few, True)):
            region = np.flatnonzero(_vacated_region(cloud, place, by_name, cells[is_few]))
            if not len(region):
                continue
            q = cloud[region]
            qcells = _Cells(q) if len(q) > 50_000 else None
            seen, through = _witnessed(witnesses, q, qcells)
            if is_few:
                added = region[seen]
                continue
            again = np.zeros(len(q), bool)
            for fd in later:
                again[_visible(fd, q, qcells)[0]] = True
            drop = region[through & ~seen & ~again]
        return drop, added

    for drop, added in _parallel(judge, vacated):  # the places are judged independently
        keep[drop] = False
        add[added] = True
    return keep, add


def _visible(fd: FrameData, pts: NDArray[np.float64], cells: _Cells | None = None
             ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64],
                        NDArray[np.float64]]:
    """(point indices, pixel rows, pixel cols, footprint in m/px) of the points ``fd`` sees: they
    project onto a valid pixel whose depth agrees within max(VIS_TOL_MIN, VIS_TOL_REL·z).
    ``cells``: ``pts`` bucketed (``_Cells``), to project only the points the keyframe may see."""
    cand = _candidates(fd, pts, cells)
    if cand is not None:
        idx, vv, uu, fp = _visible(fd, pts[cand])
        return cand[idx], vv, uu, fp
    cam = fd.rec.T_map_cam.inverse()
    pc = pts @ cam.R.T + cam.t
    K = fd.rec.K_grid
    uv, z = project(pc, K.K())
    h, w = fd.depth.shape
    with np.errstate(invalid="ignore"):
        u = np.floor(uv[:, 0] + 0.5)
        v = np.floor(uv[:, 1] + 0.5)
        idx = np.nonzero((z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h))[0]
    uu = u[idx].astype(np.int64)
    vv = v[idx].astype(np.int64)
    dz = fd.depth[vv, uu]
    vis = fd.valid[vv, uu] & (np.abs(z[idx] - dz) < np.maximum(VIS_TOL_MIN, VIS_TOL_REL * z[idx]))
    idx, uu, vv = idx[vis], uu[vis], vv[vis]
    return idx, vv, uu, z[idx] / K.fx


def attribution_margin(obs_depth: float) -> float:
    """How far beyond its box an object's cloud points may lie (its attribution gate), for an
    object seen from ``obs_depth`` metres."""
    return max(ATTRIBUTE_MARGIN_M, ATTRIBUTE_MARGIN_REL * float(obs_depth))


Gates = dict[int, tuple[OBB, float]]  # object id -> (box, margin of its attribution gate)


def attribution_gates(objs: ObjectState) -> Gates:
    """Each object's box and attribution margin (``attribution_margin``)."""
    return {o.id: (o.obb, attribution_margin(o.obs_depth)) for o in objs.objects
            if o.obb is not None}


def _in_boxes(pts: NDArray[np.float64], lab: NDArray[Any], gates: Gates) -> NDArray[np.bool_]:
    """Whether each point lies in the attribution gate of the object it is labelled with;
    objects without a box have none."""
    out = np.zeros(len(pts), bool)
    for oid in np.unique(lab).tolist():
        gate = gates.get(int(oid))
        if gate is not None:
            sel = lab == oid
            out[sel] = gate[0].contains(pts[sel], gate[1])
    return out


def _attribute_update(pts: NDArray[np.float64], frames: list[FrameData],
                      boxes: Gates | None = None, cells: _Cells | None = None
                      ) -> tuple[NDArray[np.bool_], NDArray[np.uint8], NDArray[np.int32]]:
    """(seen, colour, object id) of each point from the keyframes of one update, whatever their
    order: the colour of the finest view (smallest footprint; ties: larger colour, then larger id)
    and the object id with most votes among the views that see the point (ties: finest view),
    kept only if at least a third of those views give it. With ``boxes`` (object id → box and
    margin), a vote counts only for a point inside the voted object's gate (``_in_boxes``)."""
    n = len(pts)
    views = np.zeros(n, np.int32)
    best = np.full(n, np.inf)
    best_key = np.full(n, -1, np.int64)
    rgb = np.zeros((n, 3), np.uint8)
    p_idx, p_lab, p_fp = [], [], []
    for fd in frames:
        idx, vv, uu, fp = _visible(fd, pts, cells)
        views[idx] += 1
        col = fd.rgb[vv, uu]
        lab = fd.labels[vv, uu]
        key = ((col[:, 0].astype(np.int64) << 16) | (col[:, 1].astype(np.int64) << 8)
               | col[:, 2].astype(np.int64)) * (1 << 31) + lab
        better = (fp < best[idx]) | ((fp == best[idx]) & (key > best_key[idx]))
        sel = idx[better]
        best[sel], best_key[sel], rgb[sel] = fp[better], key[better], col[better]
        on = lab > 0
        if boxes is not None and on.any():
            on[on] = _in_boxes(pts[idx[on]], lab[on], boxes)
        p_idx.append(idx[on])
        p_lab.append(lab[on])
        p_fp.append(fp[on])
    label = np.zeros(n, np.int32)
    P = np.concatenate(p_idx) if p_idx else np.zeros(0, np.int64)
    if len(P):
        L, F = np.concatenate(p_lab), np.concatenate(p_fp)
        o = np.lexsort((F, L, P))
        P, L, F = P[o], L[o], F[o]
        start = np.flatnonzero(np.r_[True, (P[1:] != P[:-1]) | (L[1:] != L[:-1])])
        votes = np.diff(np.r_[start, len(P)])
        gp, gl, gf = P[start], L[start], F[start]  # gf: finest view voting for (point, id)
        o = np.lexsort((gl, gf, -votes, gp))
        gp, gl, votes = gp[o], gl[o], votes[o]
        win = np.r_[True, gp[1:] != gp[:-1]]
        wp, wl, wv = gp[win], gl[win], votes[win]
        ok = wv * LABEL_SHARE_DIVISOR >= views[wp]
        label[wp[ok]] = wl[ok]
    return views > 0, rgb, label


def attribute_points(xyz: NDArray[Any], frames: list[FrameData],
                     boxes: Gates | None = None, vacated: list[Vacated] | None = None,
                     retired: list[FrameData] | None = None
                     ) -> tuple[NDArray[np.uint8], NDArray[np.int32], NDArray[np.bool_]]:
    """Colour and object id of each point from the latest update whose keyframes see it.

    Updates are applied oldest → newest, so a later update wins wherever it sees a point. The
    keyframes of one update are one observation: within it, their order never matters
    (``_attribute_update``; ``boxes`` gate the votes). In the place of an object an update
    removed (``vacated``; ``retired``: the keyframes whose pixels were retired), the keyframes
    that saw through it are the latest observation: they colour what they see there, unless a
    later update sees it too (what its detecting keyframes still see beside their retired masks,
    such as its shadow, is not its latest state). Returns (rgb, label, seen by a new frame);
    unseen points keep mid-grey and label 0.
    """
    n = len(xyz)
    rgb = np.full((n, 3), UNSEGMENTED, np.uint8)
    label = np.zeros(n, np.int32)
    seen_new = np.zeros(n, bool)
    pts_all = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    by_update: dict[int, list[FrameData]] = {}
    for fd in frames:
        by_update.setdefault(fd.rec.update_id, []).append(fd)
    def chunk(start: int) -> None:  # the chunks write disjoint slices
        sl = slice(start, min(n, start + ATTRIBUTE_CHUNK))
        pts = pts_all[sl]
        cells = _Cells(pts)  # the chunk (the points are usually sorted: a slab of the map)
        for uid in sorted(by_update):
            seen, col, lab = _attribute_update(pts, by_update[uid], boxes, cells)
            where = np.flatnonzero(seen) + start
            rgb[where] = col[seen]
            label[where] = lab[seen]
            if any(fd.is_new for fd in by_update[uid]):
                seen_new[where] = True
        _attribute_vacated(pts, start, frames, vacated or [], retired or frames, boxes, rgb, label,
                           cells)

    _parallel(chunk, range(0, n, ATTRIBUTE_CHUNK))
    return rgb, label, seen_new


def _attribute_vacated(pts: NDArray[np.float64], start: int, frames: list[FrameData],
                       vacated: list[Vacated], retired: list[FrameData], boxes: Gates | None,
                       rgb: NDArray[np.uint8], label: NDArray[np.int32],
                       cells: _Cells | None = None) -> None:
    """Colour and object id, in the places of removed objects, from the keyframes that saw
    through them (``attribute_points``); ``pts`` are the points from index ``start``."""
    by_name = {fd.rec.name: fd for fd in retired}
    fused = {fd.rec.name: fd for fd in frames}
    for place in sorted(vacated, key=lambda v: (v.update, v.object)):
        witnesses = [fused[n] for n in place.witnesses if n in fused]
        if not witnesses:
            continue
        region = np.flatnonzero(_vacated_region(pts, place, by_name, cells))
        if not len(region):
            continue
        q = pts[region]
        seen, col, lab = _attribute_update(q, witnesses, boxes)
        for fd in frames:
            if fd.rec.update_id > place.update:
                seen[_visible(fd, q)[0]] = False
        rgb[region[seen] + start] = col[seen]
        label[region[seen] + start] = lab[seen]


def _sighting_depths(objs: ObjectState, frames: list[FrameData], depth_max: float
                     ) -> dict[int, list[tuple[float, float]]]:
    """Per object: (depth in the keyframe's camera, that keyframe's fused depth) of each of its
    sightings in the fused keyframes ``frames``."""
    cams = {fd.rec.index: (fd.rec.T_map_cam, fusion_depth_max(fd, depth_max)
                           if np.isfinite(depth_max) else depth_max) for fd in frames}
    out: dict[int, list[tuple[float, float]]] = {}
    for o in objs.objects:
        for sg in o.sightings:
            cam = cams.get(sg.frame)
            if cam is not None:
                z = float((np.asarray(sg.centroid, np.float64) - cam[0].t) @ cam[0].R[:, 2])
                out.setdefault(o.id, []).append((z, cam[1]))
    return out


def nearest_detections(objs: ObjectState, frames: list[FrameData]) -> dict[int, float]:
    """Per object, the depth (in that keyframe's camera) of its nearest sighting among the fused
    keyframes ``frames``: its finest view, which sampled it most densely
    (``objects.min_cloud_points``). Objects without such a sighting are left out."""
    ahead = {oid: [z for z, _ in zs if z > 0]
             for oid, zs in _sighting_depths(objs, frames, float("inf")).items()}
    return {oid: min(zs) for oid, zs in ahead.items() if zs}


def fused_objects(objs: ObjectState, frames: list[FrameData], depth_max: float) -> set[int]:
    """Ids of the objects that a keyframe which detected them fused: one of their sightings (its
    centroid, in that keyframe's camera) lies within the keyframe's fused depth
    (``fusion_depth_max``) — ``frames`` are the fused keyframes.

    Only these take support (``support_labels``). An object detected only beyond that depth (a
    car 40-100 m down a street) has no surface of its own in the cloud: its lifted points are
    placed by far monocular depth, its gate and support radius grow to 1-3 m, and the cloud
    points near them are other surfaces (the road, a facade). Its votes still count: a point
    that its keyframes see at the depth where they detected it."""
    return {oid for oid, zs in _sighting_depths(objs, frames, depth_max).items()
            if any(0.0 < z < cut for z, cut in zs)}


def support_labels(xyz: NDArray[Any], label: NDArray[np.int32], objs: ObjectState,
                   fused: set[int] | None = None) -> int:
    """Give the confirmed objects the unlabelled cloud points of their own surface: each of an
    object's own lifted points (``objects.fit_points``) picks the ``SUPPORT_NEIGHBOURS`` nearest
    unlabelled cloud points inside the object's attribution gate, within ``support_radius``; a
    point that several objects pick takes the id of the object whose own point is nearest (ties:
    the lower id), so the order of the objects does not matter. With ``fused``, only the objects
    it holds take points (``fused_objects``). ``label`` is modified in place. Returns how many
    objects took points."""
    from scipy.spatial import cKDTree

    pts_all = np.asarray(xyz, np.float64).reshape(-1, 3)
    free = label == 0
    best_d = np.full(len(pts_all), np.inf)
    best_id = np.zeros(len(pts_all), np.int32)
    x = pts_all[:, 0]
    by_x = bool(len(x)) and bool(np.all(x[1:] >= x[:-1]))  # the map cloud is sorted by x
    for o in sorted(objs.objects, key=lambda o: o.id):
        if not o.confirmed or o.obb is None or (fused is not None and o.id not in fused):
            continue
        own = np.asarray(fit_points(o), np.float64)
        if len(own) < SUPPORT_MIN_POINTS:
            continue
        margin = attribution_margin(o.obs_depth)
        lo, hi = 0, len(pts_all)
        if by_x:  # the slab of the cloud the grown box spans along x
            half = float(np.abs(o.obb.R[0]) @ (o.obb.size / 2)) + margin
            lo = int(np.searchsorted(x, o.obb.center[0] - half, side="left"))
            hi = int(np.searchsorted(x, o.obb.center[0] + half, side="right"))
        cand = lo + np.flatnonzero(free[lo:hi] & o.obb.contains(pts_all[lo:hi], margin))
        if not len(cand):
            continue
        k = min(SUPPORT_NEIGHBOURS, len(cand))
        d, j = cKDTree(pts_all[cand]).query(own, k=k,
                                            distance_upper_bound=support_radius(o.obs_depth))
        d, j = np.reshape(d, -1), np.reshape(j, -1)
        ok = np.isfinite(d)
        picked, dist = cand[j[ok]], d[ok]
        if not len(picked):
            continue
        order = np.lexsort((dist, picked))  # per picked point, its nearest own point first
        picked, dist = picked[order], dist[order]
        first = np.r_[True, picked[1:] != picked[:-1]]
        picked, dist = picked[first], dist[first]
        better = dist < best_d[picked]  # strict: a tie keeps the lower id
        best_d[picked[better]] = dist[better]
        best_id[picked[better]] = o.id
    take = best_id > 0
    label[take] = best_id[take]
    return len(np.unique(best_id[take]))


def support_radius(obs_depth: float) -> float:
    """How far a cloud point may lie from an object's own lifted points to be its surface, for
    an object seen from ``obs_depth`` metres."""
    return max(SUPPORT_RADIUS_MIN, SUPPORT_RADIUS_REL * float(obs_depth))


@dataclass
class _Setup:
    """What a fusion of the map uses (``_setup``)."""

    frames: list[FrameData]  # every keyframe
    confident: list[FrameData]  # the fused ones
    voxel: float
    depth_max: float
    vacated: list[Vacated]
    # ``consensus_depths``: share of the fused keyframes' valid pixels left out, seconds
    consensus: dict[str, Any] = field(default_factory=dict)


def _setup(ctx: Any, records: list[store.FrameRecord]) -> _Setup:
    """The keyframes (as stored or staged), the fusion's voxel and depth cut from their median
    depth, and the places of removed objects. The fused (confident) keyframes' image borders
    take the depth of the keyframes that see the surface centrally (``correct_borders``), then
    every pixel the consensus of the keyframes that see it (``consensus_depths``)."""
    new_by_name = {nf.kf.name: nf for nf in ctx.new if nf.record is not None}
    frames = [_frame_data(ctx, r, new_by_name) for r in sorted(records, key=lambda r: r.order_key)]
    voxel, depth_max = _map_depth_max([fd.median_depth for fd in frames
                                       if fd.median_depth is not None])
    confident = [fd for fd in frames if not fd.rec.low_confidence]
    correct_borders(confident)
    t0 = time.perf_counter()
    _, dropped = consensus_depths(confident)
    valid = sum(int(np.count_nonzero(fd.valid)) for fd in confident)
    consensus = {"dropped_share": round(dropped / max(valid, 1), 4),
                 "seconds": round(time.perf_counter() - t0, 2)}
    return _Setup(frames, confident, max(0.005, voxel / 2), depth_max,
                  load_vacated(ctx.tx.current), consensus)


class SurfaceQuery:
    """The map's fused surface (``fuse_map``) within a box, fused when asked (only the voxel
    blocks there: ``fused_cloud_points``' ``region``), from the keyframes as they are when it is
    created: the objects of an update ask for it around a few pairs of pieces
    (``objects._Surfaces``), before the update's latest wins retire pixels, so the map is fused
    as a whole only once, afterwards."""

    def __init__(self, ctx: Any, records: list[store.FrameRecord]) -> None:
        self.ctx = ctx
        self.records = list(records)
        self._setup: _Setup | None = None
        self._blocks: _Blocks | None = None
        self.calls = 0
        self.seconds = 0.0

    def __call__(self, lo: NDArray[Any], hi: NDArray[Any]) -> NDArray[np.float64]:
        t0 = time.perf_counter()
        if self._setup is None:
            self._setup = _setup(self.ctx, self.records)
        st = self._setup
        if self._blocks is None:  # every keyframe's blocks, once for all the boxes
            self._blocks = _frame_blocks(st.confident, st.voxel, st.depth_max)
        out = fused_cloud_points(st.confident, st.voxel, st.depth_max, st.vacated, st.frames,
                                 region=(np.asarray(lo, np.float64), np.asarray(hi, np.float64)),
                                 blocks=self._blocks)
        self.calls += 1
        self.seconds += time.perf_counter() - t0
        return out

    def release(self) -> None:
        """Drop the keyframes it loaded."""
        self._setup = None
        self._blocks = None


def fuse_map(ctx: Any, records: list[store.FrameRecord]) -> FusedCloud:
    """The fused surface of the map's confident keyframes (timed as the stage ``cloud``)."""
    t0 = time.perf_counter()
    with timing.stage("cloud"):
        st = _setup(ctx, records)
        confident = st.confident
        xyz = fused_cloud_points(confident, st.voxel, st.depth_max, st.vacated, st.frames)
        focal = float(np.median([fd.rec.K_grid.fx / fusion_step(fd.depth.shape)
                                 for fd in confident])) if confident else 0.0
    return FusedCloud(confident, xyz, st.voxel, time.perf_counter() - t0, focal, st.depth_max,
                      st.vacated, st.frames, st.consensus)


def build_geometry(ctx: Any, records: list[store.FrameRecord], objs: ObjectState,
                   progress: Any, fused: FusedCloud | None = None) -> MapGeometry:
    """The map cloud with its object ids (timed as the stage ``cloud``): the fused surface
    (``fused``, else fused here) attributed from the keyframes' instances."""
    tx = ctx.tx
    fused = fused if fused is not None else fuse_map(ctx, records)
    t0 = time.perf_counter()
    with timing.stage("cloud"):
        for fd in fused.frames:
            fd.labels = _frame_labels(ctx, fd.rec, fd.depth.shape, objs)
        xyz = fused.xyz
        rgb, label, seen_new = attribute_points(xyz, fused.frames, attribution_gates(objs),
                                                fused.vacated, fused.retired)
        supported = support_labels(xyz, label, objs,
                                   fused_objects(objs, fused.frames, fused.depth_max))
        cloud = PointCloud(xyz, rgb, label)
        new_cloud = cloud.subset(np.nonzero(seen_new)[0])
        assert cloud.label is not None
        tx.write_bytes(store.CLOUD_PLY, ply_bytes(PointCloud(cloud.xyz, cloud.rgb),
                                                  comments=["oh-my-slam map cloud, metres, z up"]))
        tx.save_npy(store.CLOUD_OBJECTS, cloud.label.astype(np.int32))
    progress(f"cloud: {len(cloud)} points (voxel {fused.voxel * 100:.1f} cm) in "
             f"{fused.seconds + time.perf_counter() - t0:.0f} s")
    stats = {"cloud_points": len(cloud), "voxel": fused.voxel, "focal_px": round(fused.focal, 2),
             "support_objects": supported, "consensus": fused.consensus}
    ctx.notes["geometry"] = stats
    return MapGeometry(cloud, new_cloud, stats, nearest_detections(objs, fused.frames))
