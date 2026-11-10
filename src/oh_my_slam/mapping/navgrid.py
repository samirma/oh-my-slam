"""The navigation grid (map.md §6): free space from camera rays, a 2-D grid per height band, and
its regions, passages and frontiers.

**Free space.** The TSDF of ``reconstruction.fusion`` keeps voxels near surfaces only, so it
cannot say where a camera saw through. The grid keeps its own occupancy volume instead
(``Volume``): voxels of ``VOXEL_M`` (horizontal) by ``VOXEL_Z_M``, and per voxel the number of
frames whose rays passed through it (``free``), ended in it (``hit``), and ended in it on a
horizontal surface (``flat``, normal within ``FLAT_DEG`` of vertical), plus per column the number of
frames that observed it (``obs``). A frame adds its depth's rays — every ``ray_stride`` pixel, off
depth edges, up to ``MAX_RANGE_M`` — once per voxel, so the counts are frame counts and a version's
volume is the previous one plus its new frames (``Volume.integrate``).

**A band** (``layers``). Relative to the floor ``f`` and the band ``[z_min, z_max]``:

* a voxel is a surface when ``hit >= OCC_RATIO * (hit + free)``, except a floor-like voxel
  (mostly ``flat`` hits within ``FLOOR_TOL_M`` of the floor);
* a cell is ``occupied`` when a surface voxel's centre lies in the band, ``free`` when it is not
  occupied and a ray passed through it in the band (or its floor was seen), else ``unknown``;
* ``height_m`` is the highest surface below ``max(z_max, HEIGHT_CAP_M)``; ``floor_observed`` is a
  floor-like voxel there.

**Camera stands** (``mark_stands``). The cell under each keyframe camera, and the robot's footprint
there, are free whatever was seen; so are the unknown cells of the disc the camera could not see:
radius ``h / tan(depression)`` of its lower field-of-view edge at height ``h`` above the floor
(``blind_radius``, at most ``BLIND_RING_MAX_M``).

**Features** (``features``). Passages are chords across narrow free gaps that lie in a wall:
through every free cell on the medial axis of the free space (its clearance a local maximum across
some direction) with a clearance below ``PASSAGE_MAX_W / 2``, the chord toward its nearest
obstacle and on to the opposite one, refined to the shortest within ``CHORD_REFINE_DEG``. A chord
is kept when the obstacle goes on beyond both of its ends along its line (``_walled``: a gap
between furniture, or between furniture and a wall, is no passage), the shortest one of each gap.
Its ends are the jambs, its length the clear width (the free width at the narrowest point: an
open leaf is an obstacle), its midpoint the centre, its normal the crossing direction. Every chord
is cut out of the free space, and the connected parts are the regions (at least
``MIN_REGION_M2``); a chord whose two sides stay connected is no passage. Frontiers are the free
cells next to unknown ones, in connected runs of at
least ``MIN_FRONTIER_M``."""

from __future__ import annotations

import io
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

VOXEL_M = 0.05
VOXEL_Z_M = 0.05
MAX_RANGE_M = 6.0  # rays end here: monocular depth beyond it is not trusted for free space
STRIDE_RANGE_M = 4.0  # ray spacing: one voxel at this range
MAX_RAYS = 40000  # free-space rays per frame
MAX_HITS = 400000  # surface points per frame
FREE_MARGIN_M = VOXEL_M  # a ray is free up to this (plus FREE_MARGIN_REL of its length) short of its hit
FREE_MARGIN_REL = 0.02
FLAT_DEG = 25.0
HIT_INTO_M = 0.01  # a hit counts this far behind the surface: a face on a voxel boundary is solid
EDGE_REL = 0.04  # depth edges (relative jump) give no rays: they are neither surface
PAD_M = 1.0  # the volume grows by this beyond what a frame reaches
BELOW_CAMERA_M = 3.0  # vertical extent kept around the cameras
ABOVE_CAMERA_M = 2.0
MAX_CELLS_XY = 2400 * 2400  # a volume never grows beyond this many columns
SAME_POSE_M = 0.005  # a frame is re-integrated only when its pose moved more than this
SAME_POSE_DEG = 0.1

OCC_RATIO = 0.3
FLOOR_TOL_M = 0.10
HEIGHT_CAP_M = 2.0
BLIND_RING_MAX_M = 2.0
MIN_DEPRESSION_DEG = 2.0

MIN_REGION_M2 = 0.25
MIN_FRONTIER_M = 0.3
CHORD_REFINE_DEG = 10.0
PASSAGE_MAX_W = 2.0  # wider gaps are not narrowings
MIN_PASSAGE_W = 0.2
NMS_M = 0.3
NARROW_SHIFT_M = 0.5
NARROW_TURN_DEG = 20.0
NARROW_EXACT = 8  # screened chords measured exactly
CHORD_MAX_M = 4.0  # a chord is measured this far each way
JAMB_RUN_M = 0.4  # a passage's jambs: wall that goes on this far along the gap's line
JAMB_OCCUPIED = 0.5
DOOR_MIN_W = 0.55
DOOR_MAX_W = 1.3
DOOR_OBJECT_M = 0.6  # a door-labelled object this close to a passage's centre makes it a door
LEAF_MIN_M = 0.3
LEAF_WALL_M = 0.15  # cells farther than this from the jamb line, near a jamb, are leaf cells

UNKNOWN, FREE, OCCUPIED = 0, 1, 2
FLOOR_BY_RULE = 2  # floor_observed: free by rule (camera stand, blind disc), floor not seen

Array = NDArray[np.float64]


# ------------------------------------------------------------------------------------------------
# the occupancy volume


def ray_stride(fx: float, width: int, height: int, voxel: float = VOXEL_M) -> int:
    """Pixel spacing of a frame's rays: one voxel apart at ``STRIDE_RANGE_M``, at most
    ``MAX_RAYS`` rays."""
    s = max(1, int(fx * voxel / STRIDE_RANGE_M))
    return max(s, math.ceil(math.sqrt(width * height / MAX_RAYS)))


def pose_moved(a: Sequence[float], b: Sequence[float]) -> bool:
    """Whether two 4x4 poses (flattened) differ by more than ``SAME_POSE_M`` / ``SAME_POSE_DEG``."""
    A, B = np.asarray(a, np.float64).reshape(4, 4), np.asarray(b, np.float64).reshape(4, 4)
    c = (float(np.trace(A[:3, :3].T @ B[:3, :3])) - 1.0) / 2.0
    deg = math.degrees(math.acos(max(-1.0, min(1.0, c))))
    return bool(np.linalg.norm(A[:3, 3] - B[:3, 3]) > SAME_POSE_M or deg > SAME_POSE_DEG)


@dataclass
class Volume:
    origin: Array = field(default_factory=lambda: np.zeros(3))  # corner of voxel [0, 0, 0]
    voxel: float = VOXEL_M
    vz: float = VOXEL_Z_M
    free: NDArray[np.uint16] = field(default_factory=lambda: np.zeros((0, 0, 0), np.uint16))
    hit: NDArray[np.uint16] = field(default_factory=lambda: np.zeros((0, 0, 0), np.uint16))
    flat: NDArray[np.uint16] = field(default_factory=lambda: np.zeros((0, 0, 0), np.uint16))
    obs: NDArray[np.uint16] = field(default_factory=lambda: np.zeros((0, 0), np.uint16))
    frames: dict[str, list[float]] = field(default_factory=dict)  # key -> flattened T_map_cam

    @property
    def shape(self) -> tuple[int, int, int]:
        z, y, x = self.free.shape
        return int(z), int(y), int(x)

    @property
    def empty(self) -> bool:
        return self.free.size == 0

    # -- extent ---------------------------------------------------------------------------------

    def ensure(self, lo: Array, hi: Array) -> bool:
        """Grow the arrays to hold the box ``[lo, hi]`` (plus ``PAD_M``); False when that would
        exceed ``MAX_CELLS_XY`` columns (the volume is unchanged)."""
        step = np.array([self.voxel, self.voxel, self.vz])
        lo = np.floor((np.asarray(lo) - PAD_M * np.array([1, 1, 0])) / step) * step
        hi = np.ceil((np.asarray(hi) + PAD_M * np.array([1, 1, 0])) / step) * step
        if self.empty:
            n = np.maximum(np.round((hi - lo) / step).astype(int), 1)
            if n[0] * n[1] > MAX_CELLS_XY:
                return False
            self.origin = lo.astype(np.float64)
            self.free = np.zeros((n[2], n[1], n[0]), np.uint16)
            self.hit = np.zeros_like(self.free)
            self.flat = np.zeros_like(self.free)
            self.obs = np.zeros((n[1], n[0]), np.uint16)
            return True
        Z, Y, X = self.shape
        cur_hi = self.origin + step * np.array([X, Y, Z])
        new_lo = np.minimum(self.origin, lo)
        new_hi = np.maximum(cur_hi, hi)
        before = np.round((self.origin - new_lo) / step).astype(int)
        after = np.round((new_hi - cur_hi) / step).astype(int)
        if not (before.any() or after.any()):
            return True
        nx, ny = X + before[0] + after[0], Y + before[1] + after[1]
        if nx * ny > MAX_CELLS_XY:
            return False
        pad3 = ((before[2], after[2]), (before[1], after[1]), (before[0], after[0]))
        self.free = np.pad(self.free, pad3)
        self.hit = np.pad(self.hit, pad3)
        self.flat = np.pad(self.flat, pad3)
        self.obs = np.pad(self.obs, pad3[1:])
        self.origin = self.origin - before * step
        return True

    def _index(self, P: NDArray[Any]) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
        """(flat voxel index, inside) of points ``P`` (N, 3)."""
        Z, Y, X = self.shape
        step = np.array([self.voxel, self.voxel, self.vz])
        ijk = np.floor((np.asarray(P, np.float64) - self.origin) / step).astype(np.int64)
        inside = ((ijk >= 0) & (ijk < np.array([X, Y, Z]))).all(axis=1)
        flat = (ijk[:, 2] * Y + ijk[:, 1]) * X + ijk[:, 0]
        return flat, inside

    # -- integration ----------------------------------------------------------------------------

    def integrate(self, key: str, depth: NDArray[Any], K: NDArray[Any], T_map_cam: NDArray[Any],
                  valid: NDArray[Any] | None = None) -> bool:
        """Add a frame's rays: ``depth`` (metres, z along the optical axis, 0 = none) on the grid
        of intrinsics ``K`` (3x3), camera pose ``T_map_cam`` (4x4). False when it added
        nothing (no valid ray, or beyond the volume's limits)."""
        from oh_my_slam.core.geometry import depth_edge_mask, unproject_pixels

        d_full = np.asarray(depth, np.float64)
        H, W = d_full.shape
        ok_full = np.isfinite(d_full) & (d_full > 0)
        if valid is not None:
            ok_full &= np.asarray(valid, bool)
        ok_full &= ~depth_edge_mask(np.where(ok_full, d_full, 0.0), EDGE_REL)
        hs = max(1, math.ceil(math.sqrt(H * W / MAX_HITS)))  # surface points: nearly every pixel
        rows = np.arange(hs // 2, H, hs)
        cols = np.arange(hs // 2, W, hs)
        d = d_full[np.ix_(rows, cols)]
        ok = ok_full[np.ix_(rows, cols)]
        if not ok.any():
            return False
        T = np.asarray(T_map_cam, np.float64)
        R, C = T[:3, :3], T[:3, 3]
        Kf = np.asarray(K, np.float64)
        u, v = np.meshgrid(cols + 0.5, rows + 0.5)
        rays = unproject_pixels(u, v, np.ones_like(u), Kf)
        P = (rays * np.where(ok, d, 0.0)[..., None]) @ R.T + C  # (h, w, 3)
        # normals of the point grid (map frame), for the flat surfaces
        dx = np.zeros_like(P)
        dy = np.zeros_like(P)
        dx[:, :-1] = P[:, 1:] - P[:, :-1]
        dy[:-1] = P[1:] - P[:-1]
        n = np.cross(dx, dy)
        nn = np.linalg.norm(n, axis=-1)
        has_n = ok.copy()
        has_n[:, :-1] &= ok[:, 1:]
        has_n[:-1] &= ok[1:]
        has_n[-1, :] = False
        has_n[:, -1] = False
        with np.errstate(invalid="ignore", divide="ignore"):
            flat_px = has_n & (np.abs(n[..., 2]) / np.where(nn > 0, nn, 1.0)
                               > math.cos(math.radians(FLAT_DEG)))
        dirs_all = rays @ R.T
        rng_all = d * np.linalg.norm(rays, axis=-1)
        dirs_all /= np.linalg.norm(dirs_all, axis=-1, keepdims=True)
        surf = ok & (rng_all <= MAX_RANGE_M)
        ends = P[surf] + dirs_all[surf] * HIT_INTO_M
        flats = P[surf & flat_px] + dirs_all[surf & flat_px] * HIT_INTO_M
        # the free-space rays: every ``ray_stride`` pixel
        k = max(1, ray_stride(float(Kf[0, 0]), W, H, self.voxel) // hs)
        sub = np.zeros_like(ok)
        sub[k // 2::k, k // 2::k] = True
        sub &= ok
        dirs = dirs_all[sub]
        rng = rng_all[sub]
        hit_ok = rng <= MAX_RANGE_M
        reach = np.minimum(rng, MAX_RANGE_M)
        far = C + dirs * reach[:, None]
        pts = np.vstack([ends, far, C[None]])
        lo, hi = pts.min(0), pts.max(0)
        lo[2] = max(lo[2], C[2] - BELOW_CAMERA_M)
        hi[2] = min(hi[2], C[2] + ABOVE_CAMERA_M)
        if not self.ensure(lo, hi):
            return False
        # free samples along each ray, short of its hit
        step = 0.7 * self.voxel
        free_len = np.where(hit_ok, rng - FREE_MARGIN_M - FREE_MARGIN_REL * rng, reach)
        counts = np.maximum(np.floor(free_len / step).astype(np.int64), 0)
        total = int(counts.sum())
        idx_ray = np.repeat(np.arange(len(dirs)), counts)
        start = np.repeat(np.cumsum(counts) - counts, counts)
        t = (np.arange(total) - start + 0.5) * step
        samples = C + dirs[idx_ray] * t[:, None]
        _Z, Y, X = self.shape
        cols_seen = np.zeros(Y * X, bool)
        for arr, at in ((self.free, samples), (self.hit, ends), (self.flat, flats)):
            fi, inside = self._index(at)
            fi = np.unique(fi[inside])
            arr.reshape(-1)[fi] = np.minimum(arr.reshape(-1)[fi].astype(np.int64) + 1,
                                             np.iinfo(np.uint16).max).astype(np.uint16)
            cols_seen[fi % (Y * X)] = True
        seen = np.flatnonzero(cols_seen)
        self.obs.reshape(-1)[seen] = np.minimum(self.obs.reshape(-1)[seen].astype(np.int64) + 1,
                                                np.iinfo(np.uint16).max).astype(np.uint16)
        self.frames[key] = T.reshape(-1).tolist()
        return True

    # -- storage --------------------------------------------------------------------------------

    def to_bytes(self) -> bytes:
        import json

        buf = io.BytesIO()
        np.savez_compressed(buf, origin=self.origin, voxel=np.array([self.voxel, self.vz]),
                            free=self.free, hit=self.hit, flat=self.flat, obs=self.obs,
                            frames=np.frombuffer(json.dumps(self.frames).encode(), np.uint8))
        return buf.getvalue()

    @staticmethod
    def from_bytes(data: bytes) -> Volume:
        import json

        z = np.load(io.BytesIO(data))
        vox = z["voxel"]
        return Volume(np.asarray(z["origin"], np.float64), float(vox[0]), float(vox[1]),
                      z["free"], z["hit"], z["flat"], z["obs"],
                      json.loads(bytes(z["frames"]).decode()))

    def estimate_floor(self) -> float | None:
        """The floor's height when the map has none: the lowest layer holding at least a quarter
        of the most flat hits of any layer (its centre), None without flat hits."""
        if self.empty:
            return None
        per = self.flat.reshape(self.shape[0], -1).sum(axis=1).astype(np.float64)
        if per.max() <= 0:
            return None
        k = int(np.argmax(per >= 0.25 * per.max()))
        return float(self.origin[2] + (k + 0.5) * self.vz)


# ------------------------------------------------------------------------------------------------
# a band


@dataclass
class Layers:
    origin: Array  # x, y of cell [0, 0]'s corner
    cell: float
    state: NDArray[np.uint8]  # (rows y, cols x)
    observations: NDArray[np.uint16]
    height: NDArray[np.float32]
    floor_observed: NDArray[np.uint8]

    @property
    def shape(self) -> tuple[int, int]:
        r, c = self.state.shape
        return int(r), int(c)

    def cell_of(self, xy: Sequence[float]) -> tuple[int, int]:
        """(row, col) of a map point (may lie outside the grid)."""
        return (math.floor((xy[1] - self.origin[1]) / self.cell),
                math.floor((xy[0] - self.origin[0]) / self.cell))

    def centre(self, row: float, col: float) -> Array:
        return np.array([self.origin[0] + (col + 0.5) * self.cell,
                         self.origin[1] + (row + 0.5) * self.cell])


def layers(vol: Volume, floor_z: float, z_min: float, z_max: float) -> Layers:
    """The band's layers at the volume's resolution (see the module docstring)."""
    Z, Y, X = vol.shape
    zc = vol.origin[2] + (np.arange(Z) + 0.5) * vol.vz - floor_z  # relative to the floor
    hit = vol.hit.astype(np.float32)
    free = vol.free.astype(np.float32)
    surface = (vol.hit > 0) & (hit >= OCC_RATIO * (hit + free))
    floorlike = (vol.flat.astype(np.float32) * 2 >= hit) & (vol.hit > 0) & \
        (np.abs(zc) <= FLOOR_TOL_M)[:, None, None]
    surface &= ~floorlike
    band = ((zc >= z_min) & (zc <= z_max))[:, None, None]
    occupied = (surface & band).any(axis=0)
    floor_obs = floorlike.any(axis=0)
    free2 = (((vol.free > 0) & band).any(axis=0) | floor_obs) & ~occupied
    state = np.where(occupied, OCCUPIED, np.where(free2, FREE, UNKNOWN)).astype(np.uint8)
    cap = max(z_max, HEIGHT_CAP_M)
    top = (surface | floorlike) & (zc <= cap)[:, None, None]
    has = top.any(axis=0)
    k = Z - 1 - np.argmax(top[::-1], axis=0)
    height = np.where(has, zc[np.clip(k, 0, max(Z - 1, 0))], np.nan).astype(np.float32) \
        if Z else np.full((Y, X), np.nan, np.float32)
    return Layers(vol.origin[:2].copy(), vol.voxel, state, vol.obs.copy(), height,
                  floor_obs.astype(np.uint8))


def resample(lay: Layers, cell: float) -> Layers:
    """The layers on cells of ``cell`` metres over the same extent: nearest for finer cells; for
    coarser ones, the worst state (occupied, then free), the most observations, the highest
    surface and any observed floor of the cells each covers."""
    if abs(cell - lay.cell) < 1e-9:
        return lay
    R, C = lay.shape
    rows = max(1, math.ceil(R * lay.cell / cell - 1e-9))
    cols = max(1, math.ceil(C * lay.cell / cell - 1e-9))
    if cell < lay.cell:
        r = np.minimum(((np.arange(rows) + 0.5) * cell / lay.cell).astype(int), R - 1)
        c = np.minimum(((np.arange(cols) + 0.5) * cell / lay.cell).astype(int), C - 1)
        g = np.ix_(r, c)
        return Layers(lay.origin, cell, lay.state[g], lay.observations[g], lay.height[g],
                      lay.floor_observed[g])
    r = np.minimum(((np.arange(R) + 0.5) * lay.cell / cell).astype(int), rows - 1)
    c = np.minimum(((np.arange(C) + 0.5) * lay.cell / cell).astype(int), cols - 1)
    rr, cc = np.meshgrid(r, c, indexing="ij")
    target = (rr * cols + cc).ravel()
    prio = np.array([0, 1, 2], np.uint8)[lay.state.ravel()]  # unknown < free < occupied
    state = np.zeros(rows * cols, np.uint8)
    np.maximum.at(state, target, prio)
    obs = np.zeros(rows * cols, np.uint16)
    np.maximum.at(obs, target, lay.observations.ravel())
    fl = np.zeros(rows * cols, np.uint8)  # seen (1) over free by rule (2) over not seen (0)
    np.maximum.at(fl, target, np.array([0, 2, 1], np.uint8)[lay.floor_observed.ravel()])
    fl = np.array([0, 2, 1], np.uint8)[fl]
    h = np.full(rows * cols, -np.inf, np.float32)
    np.maximum.at(h, target, np.where(np.isnan(lay.height), -np.inf, lay.height).ravel())
    h[np.isinf(h)] = np.nan
    return Layers(lay.origin, cell, state.reshape(rows, cols), obs.reshape(rows, cols),
                  h.reshape(rows, cols), fl.reshape(rows, cols))


# ------------------------------------------------------------------------------------------------
# camera stands


@dataclass(frozen=True)
class Stand:
    """A keyframe camera: its pose and the intrinsics of its image."""

    T_map_cam: Array
    K: Array  # 3x3
    width: int
    height: int


def blind_radius(stand: Stand, floor_z: float) -> float:
    """Horizontal radius of the floor the camera could not see in front of it: where its lower
    field-of-view edge (the bottom row's centre) meets the floor; 0 when that edge looks
    less than ``MIN_DEPRESSION_DEG`` below the horizon; at most ``BLIND_RING_MAX_M``."""
    from oh_my_slam.core.geometry import unproject_pixels

    T = np.asarray(stand.T_map_cam, np.float64)
    h = float(T[2, 3]) - floor_z
    if h <= 0:
        return 0.0
    K = np.asarray(stand.K, np.float64)
    ray = unproject_pixels(np.array([K[0, 2]]), np.array([float(stand.height)]), np.ones(1), K)[0]
    d = T[:3, :3] @ ray
    d /= np.linalg.norm(d)
    depression = math.degrees(math.asin(max(-1.0, min(1.0, -float(d[2])))))
    if depression < MIN_DEPRESSION_DEG:
        return 0.0
    return min(h / math.tan(math.radians(depression)), BLIND_RING_MAX_M)


def _disc(lay: Layers, xy: Array, radius: float) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """(rows, cols) of the cells whose centres lie within ``radius`` of ``xy``, plus its own."""
    R, C = lay.shape
    r0, c0 = lay.cell_of(xy)
    k = math.ceil(radius / lay.cell) + 1
    rr, cc = np.mgrid[r0 - k:r0 + k + 1, c0 - k:c0 + k + 1]
    cen_x = lay.origin[0] + (cc + 0.5) * lay.cell
    cen_y = lay.origin[1] + (rr + 0.5) * lay.cell
    inside = (np.hypot(cen_x - xy[0], cen_y - xy[1]) <= radius) | ((rr == r0) & (cc == c0))
    inside &= (rr >= 0) & (rr < R) & (cc >= 0) & (cc < C)
    return rr[inside], cc[inside]


def mark_stands(lay: Layers, stands: Sequence[Stand], floor_z: float,
                footprint_radius: float | None) -> None:
    """Free the camera cells (and the footprint) whatever was seen there, and the unknown cells
    of each camera's blind disc that the camera cell reaches without crossing an occupied cell
    (in place). Those cells whose floor nobody saw get ``floor_observed`` 2 (free by rule)."""
    for st in stands:
        xy = np.asarray(st.T_map_cam, np.float64)[:2, 3]
        r, c = _disc(lay, xy, footprint_radius or 0.0)
        lay.state[r, c] = FREE
        _by_rule(lay, r, c)
        ring = blind_radius(st, floor_z)
        if ring <= lay.cell:
            continue
        r, c = _disc(lay, xy, ring)
        if not r.size:  # the camera lies outside the grid
            continue
        r0, c0 = r.min(), c.min()
        box = np.zeros((r.max() - r0 + 1, c.max() - c0 + 1), bool)
        box[r - r0, c - c0] = lay.state[r, c] != OCCUPIED
        lab, _ = ndimage.label(box)
        me = lay.cell_of(xy)
        if not (0 <= me[0] - r0 < box.shape[0] and 0 <= me[1] - c0 < box.shape[1]):
            continue
        reach = lab[r - r0, c - c0] == lab[me[0] - r0, me[1] - c0]
        unknown = reach & (lay.state[r, c] == UNKNOWN)
        lay.state[r[unknown], c[unknown]] = FREE
        _by_rule(lay, r[reach], c[reach])


def _by_rule(lay: Layers, r: NDArray[np.int64], c: NDArray[np.int64]) -> None:
    """``floor_observed`` 2 on free cells whose floor was not seen."""
    m = (lay.floor_observed[r, c] == 0) & (lay.state[r, c] == FREE)
    lay.floor_observed[r[m], c[m]] = FLOOR_BY_RULE


# ------------------------------------------------------------------------------------------------
# features


@dataclass
class Region:
    cells: NDArray[np.bool_]
    area_m2: float
    centroid: Array
    polygon: list[list[float]]
    coverage_share: float
    id: int = 0


@dataclass
class Passage:
    centre: Array
    width: float
    normal_deg: float
    jambs: tuple[Array, Array]
    regions: tuple[int, int]  # indices into the regions list until ids are assigned
    kind: str
    leaf: dict[str, Any] | None
    id: int = 0


@dataclass
class Frontier:
    centre: Array
    length: float
    region: int | None  # index into the regions list until ids are assigned
    id: int = 0


@dataclass
class Features:
    regions: list[Region]
    passages: list[Passage]
    frontiers: list[Frontier]
    labels: NDArray[np.int32]  # region index + 1 per cell (0: none)


def _geodesic_labels(free: NDArray[np.bool_], seeds: NDArray[np.int32]) -> NDArray[np.int32]:
    """Each free cell's nearest seed label through free cells (8-connected breadth-first, by
    steps), 0 where none reaches."""
    R, C = free.shape
    lab = np.where(free, seeds, 0).astype(np.int32)
    pad_free = np.pad(free, 1)
    lab = np.pad(lab, 1)
    W = C + 2
    front = np.flatnonzero(lab.ravel())
    offs = np.array([-W - 1, -W, -W + 1, -1, 1, W - 1, W, W + 1])
    flat_lab, flat_free = lab.ravel(), pad_free.ravel()
    while front.size:
        nb = (front[:, None] + offs[None, :]).ravel()
        src = np.repeat(front, len(offs))
        ok = flat_free[nb] & (flat_lab[nb] == 0)
        nb, src = nb[ok], src[ok]
        if not nb.size:
            break
        nb, first = np.unique(nb, return_index=True)
        flat_lab[nb] = flat_lab[src[first]]
        front = nb
    return flat_lab.reshape(R + 2, C + 2)[1:-1, 1:-1]


def _march(free: NDArray[np.bool_], p: Array, u: Array, max_t: float) -> tuple[float, tuple[int, int] | None]:
    """Distance (in cells) from ``p`` (cell coordinates: col + 0.5, row + 0.5 is a centre) along
    the unit direction ``u`` (col, row) to the boundary of the first cell that is not free (at
    most ``max_t``), and that cell (None beyond the grid: the grid's edge is the boundary)."""
    R, C = free.shape
    step = 0.25
    t = np.arange(1, int(max_t / step) + 1) * step
    x = np.floor(p[0] + u[0] * t).astype(np.int64)
    y = np.floor(p[1] + u[1] * t).astype(np.int64)
    inside = (x >= 0) & (x < C) & (y >= 0) & (y < R)
    blocked = ~inside
    blocked[inside] = ~free[y[inside], x[inside]]
    if not blocked.any():
        return max_t, None
    k = int(np.argmax(blocked))
    col, row, tk = int(x[k]), int(y[k]), float(t[k])
    # entry into that cell (slab method), within the last step
    ts = [min((lo - p[a]) / u[a], (lo + 1 - p[a]) / u[a])
          for a, lo in ((0, col), (1, row)) if abs(u[a]) > 1e-12]
    entry = min(max(max(ts), tk - step), tk)
    return entry, (row, col) if bool(inside[k]) else None


def _march_many(free: NDArray[np.bool_], P: Array, U: Array, max_t: float) -> Array:
    """``_march`` of many rays at once, to the first step inside a cell not free (a screening:
    up to half a cell past its boundary)."""
    R, C = free.shape
    step = 0.5
    t = np.arange(1, int(max_t / step) + 1) * step
    x = np.floor(P[:, 0:1] + U[:, 0:1] * t[None]).astype(np.int64)
    y = np.floor(P[:, 1:2] + U[:, 1:2] * t[None]).astype(np.int64)
    inside = (x >= 0) & (x < C) & (y >= 0) & (y < R)
    blocked = ~inside
    blocked[inside] = ~free[y[inside], x[inside]]
    first = np.argmax(blocked, axis=1)
    return np.where(blocked.any(axis=1), t[first], max_t)


def _walled_many(state: NDArray[np.uint8], E1: Array, E2: Array, cell: float
                 ) -> NDArray[np.bool_]:
    """``_walled`` of many chords at once (ends (N, 2) in cell coordinates)."""
    R, C = state.shape
    L = np.linalg.norm(E2 - E1, axis=1)
    U = (E2 - E1) / np.maximum(L, 1e-9)[:, None]
    nrm = np.column_stack([-U[:, 1], U[:, 0]])
    n = max(2, round(JAMB_RUN_M / cell * 4))
    t = (np.arange(n) + 0.5) * (JAMB_RUN_M / cell) / n
    ok = L > 1e-9
    for end, d in ((E2, U), (E1, -U)):
        pts = end[:, None, :] + d[:, None, :] * t[None, :, None]
        occ = np.zeros((len(E1), n), bool)
        for off in (0.0, -1.0, 1.0):
            q = pts + off * nrm[:, None, :]
            x, y = np.floor(q[..., 0]).astype(np.int64), np.floor(q[..., 1]).astype(np.int64)
            inside = (x >= 0) & (x < C) & (y >= 0) & (y < R)
            ok &= inside.all(axis=1)
            v = np.where(inside, state[np.clip(y, 0, R - 1), np.clip(x, 0, C - 1)], UNKNOWN)
            occ |= v == OCCUPIED
            if off == 0.0:
                ok &= ~(v == FREE).any(axis=1)
        ok &= occ.mean(axis=1) >= JAMB_OCCUPIED
    return ok


def _chord_at(free: NDArray[np.bool_], p: Array, theta: float, max_t: float
              ) -> tuple[float, Array, Array, tuple[int, int] | None, tuple[int, int] | None]:
    """The free chord through ``p`` (cell coordinates) at angle ``theta``: (length in cells, end
    a, end b, the non-free cells it ends on)."""
    u = np.array([math.cos(theta), math.sin(theta)])
    t1, c1 = _march(free, p, u, max_t)
    t2, c2 = _march(free, p, -u, max_t)
    return t1 + t2, p + u * t1, p - u * t2, c1, c2


def _passage_chords(state: NDArray[np.uint8], cell: float
                    ) -> list[tuple[float, Array, Array, tuple[int, int] | None, tuple[int, int] | None]]:
    """The passages' chords: through every free cell on the ridge of the clearance (its distance
    to anything not free a local maximum) with a clearance below ``PASSAGE_MAX_W / 2``, the chord
    toward its nearest obstacle, refined to the shortest within ``CHORD_REFINE_DEG``; kept when it
    lies in a wall (``_walled``), the shortest of those whose centres lie within half its length
    or ``NMS_M`` of each other, then narrowed beside it (``_narrowest``)."""
    free = state == FREE
    edt, idx = ndimage.distance_transform_edt(np.pad(free, 1), return_indices=True)
    edt = edt[1:-1, 1:-1]
    iy, ix = idx[0][1:-1, 1:-1] - 1, idx[1][1:-1, 1:-1] - 1
    pad = np.pad(edt, 1)
    R, C = edt.shape
    axis = np.zeros(edt.shape, bool)  # a local maximum across some direction: the medial axis
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = pad[1 + dr:R + 1 + dr, 1 + dc:C + 1 + dc]
        b = pad[1 - dr:R + 1 - dr, 1 - dc:C + 1 - dc]
        axis |= (edt >= a) & (edt >= b)
    ridge = free & axis & (edt * cell <= PASSAGE_MAX_W / 2 + cell) & \
        (edt * cell >= MIN_PASSAGE_W / 2)
    max_t = CHORD_MAX_M / cell
    rc = np.argwhere(ridge)
    P = rc[:, ::-1] + 0.5
    D = np.column_stack([ix[rc[:, 0], rc[:, 1]], iy[rc[:, 0], rc[:, 1]]]) + 0.5 - P
    lim = PASSAGE_MAX_W / cell + 1
    found = []
    for theta in (np.arctan2(D[:, 1], D[:, 0]),):  # toward the nearest obstacle
        U = np.column_stack([np.cos(theta), np.sin(theta)])
        t1 = _march_many(free, P, U, lim)
        t2 = _march_many(free, P, -U, lim)
        keep = (t1 + t2) * cell <= PASSAGE_MAX_W + cell
        keep[keep] = _walled_many(state, P[keep] - U[keep] * t2[keep, None],
                                  P[keep] + U[keep] * t1[keep, None], cell)
        for k in np.flatnonzero(keep):
            L, e1, e2, _, _ = _chord_at(free, P[k], float(theta[k]), max_t)
            if L * cell <= PASSAGE_MAX_W and _walled(state, e1, e2, cell):
                found.append((L, P[k], float(theta[k])))
    out: list[tuple[float, Array, Array, tuple[int, int] | None, tuple[int, int] | None]] = []
    centres: list[tuple[Array, float]] = []
    for _, p, th in sorted(found, key=lambda f: (f[0], f[1][1], f[1][0])):
        if any(float(np.linalg.norm(p - q)) * cell < max(Lq * cell / 2, NMS_M)
               for q, Lq in centres):
            continue
        walled = [ch for ch in (_chord_at(free, p, th + math.radians(a), max_t)
                                for a in np.arange(-CHORD_REFINE_DEG, CHORD_REFINE_DEG + 0.5, 1.0))
                  if _walled(state, ch[1], ch[2], cell)]
        best = min(walled, key=lambda ch: round(ch[0], 6)) if walled else \
            _chord_at(free, p, th, max_t)
        centres.append((p, best[0]))
        narrow = _narrowest(free, best, cell, max_t)
        mid = (narrow[1] + narrow[2]) / 2
        if all(float(np.linalg.norm(mid - (o[1] + o[2]) / 2)) * cell >= NMS_M for o in out):
            out.append(narrow)
    return out


def _narrowest(free: NDArray[np.bool_], ch: tuple[float, Array, Array, Any, Any], cell: float,
               max_t: float) -> tuple[float, Array, Array, tuple[int, int] | None,
                                      tuple[int, int] | None]:
    """The narrowest free chord of a passage: shifted along its normal by up to
    ``NARROW_SHIFT_M``, turned by up to ``NARROW_TURN_DEG``, ending on obstacles at both ends,
    its midpoint between the walled chord ``ch``'s jambs (along their line). An open leaf, or a
    door frame deeper than the wall, narrows the gap beside the wall line."""
    L0, a, b = ch[0], ch[1], ch[2]
    u = (b - a) / max(L0, 1e-9)
    n = np.array([-u[1], u[0]])
    mid = (a + b) / 2
    th0 = math.atan2(u[1], u[0])
    shift = NARROW_SHIFT_M / cell
    S, A = np.meshgrid(np.arange(-shift, shift + 1e-9, 1.0),
                       np.radians(np.arange(-NARROW_TURN_DEG, NARROW_TURN_DEG + 1e-9, 2.5)))
    P = mid + np.outer(S.ravel(), n)
    th = th0 + A.ravel()
    U = np.column_stack([np.cos(th), np.sin(th)])
    lim = L0 + 2.0  # a screening of every shift and turn, then the best ones exactly
    t1 = _march_many(free, P, U, lim)
    t2 = _march_many(free, P, -U, lim)
    along = ((P + U * ((t1 - t2) / 2)[:, None]) - a) @ u
    ok = (t1 < lim) & (t2 < lim) & (along >= 0) & (along <= L0)
    best = ch
    for k in np.flatnonzero(ok)[np.argsort((t1 + t2)[ok], kind="stable")][:NARROW_EXACT]:
        c = _chord_at(free, P[k], float(th[k]), max_t)
        mid_along = float(((c[1] + c[2]) / 2 - a) @ u)
        if c[0] < best[0] - 1e-9 and c[3] is not None and c[4] is not None \
                and 0.0 <= mid_along <= L0:
            best = (c[0], c[2], c[1], c[4], c[3])  # from a's side to b's, as ``ch``
    return best


def _walled(state: NDArray[np.uint8], e1: Array, e2: Array, cell: float) -> bool:
    """Whether a chord (ends in cell coordinates) lies in a wall: beyond each end, along the chord,
    the obstacle goes on for ``JAMB_RUN_M`` with no free cell and at least ``JAMB_OCCUPIED`` of
    its cells occupied. A gap between pieces of furniture, or between furniture and a wall, has
    an end whose obstacle stops (a box's unseen inside, the room behind a wall): it is no
    passage, and the free space on both sides stays one region."""
    R, C = state.shape
    L = float(np.linalg.norm(e2 - e1))
    if L < 1e-9:
        return False
    u = (e2 - e1) / L
    n = max(2, round(JAMB_RUN_M / cell * 4))
    nrm = np.array([-u[1], u[0]])
    for end, d in ((e2, u), (e1, -u)):
        t = (np.arange(n) + 0.5) * (JAMB_RUN_M / cell) / n
        pts = end[None] + d[None] * t[:, None]
        occ = np.zeros(n, bool)
        on = None
        for off in (0.0, -1.0, 1.0):  # a wall seen from one side: its occupied face beside
            q = pts + off * nrm
            x, y = np.floor(q[:, 0]).astype(int), np.floor(q[:, 1]).astype(int)
            if not ((x >= 0) & (x < C) & (y >= 0) & (y < R)).all():
                return False
            v = state[y, x]
            occ |= v == OCCUPIED
            on = v if on is None else on
        assert on is not None
        if (on == FREE).any() or occ.mean() < JAMB_OCCUPIED:
            return False
    return True


def _polygon(mask: NDArray[np.bool_], lay: Layers) -> list[list[float]]:
    """The outer boundary of a region's cells (map x, y), traced at half-cell resolution."""
    import cv2

    up = np.kron(mask.astype(np.uint8), np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(np.pad(up, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(cnt, 1.0, True).reshape(-1, 2).astype(np.float64)
    if len(approx) < 3:  # a one-cell region: its square
        approx = cnt.reshape(-1, 2).astype(np.float64)
    pts = []
    for x, y in approx - 1:  # undo the pad
        pts.append([round(float(lay.origin[0] + (x + 0.5) * lay.cell / 2), 4),
                    round(float(lay.origin[1] + (y + 0.5) * lay.cell / 2), 4)])
    return pts


def _leaf(state: NDArray[np.uint8], lay: Layers, j1: Array, j2: Array) -> dict[str, Any] | None:
    """An open door leaf at the gap ``j1``-``j2`` (map x, y): the occupied cells off the jamb line by
    more than ``LEAF_WALL_M`` (and less than the gap's width), along it within ``LEAF_WALL_M`` of the
    gap, spanning at least ``LEAF_MIN_M``; its footprint, and whether it reaches into the gap
    between the jambs."""
    gap = float(np.linalg.norm(j2 - j1))
    u = (j2 - j1) / max(gap, 1e-9)
    n = np.array([-u[1], u[0]])
    occ = np.argwhere(state == OCCUPIED)
    if not len(occ):
        return None
    cen = np.column_stack([lay.origin[0] + (occ[:, 1] + 0.5) * lay.cell,
                           lay.origin[1] + (occ[:, 0] + 0.5) * lay.cell])
    along = (cen - j1) @ u
    off = np.abs((cen - j1) @ n)
    pick = (off > LEAF_WALL_M) & (off <= gap + lay.cell) & (along >= -LEAF_WALL_M) & \
        (along <= gap + LEAF_WALL_M)
    pts = cen[pick]
    if len(pts) < 2:
        return None
    span = float(np.max(np.linalg.norm(pts[:, None] - pts[None], axis=-1)))
    if span + lay.cell < LEAF_MIN_M:
        return None
    a_ = (pts - j1) @ u  # the leaf's rectangle, along the jamb line and across it
    b_ = (pts - j1) @ n
    h_ = lay.cell / 2
    poly = [[round(float(x), 4), round(float(y), 4)] for x, y in
            (j1 + u * (a_.min() - h_) + n * (b_.min() - h_), j1 + u * (a_.max() + h_) + n * (b_.min() - h_),
             j1 + u * (a_.max() + h_) + n * (b_.max() + h_), j1 + u * (a_.min() - h_) + n * (b_.max() + h_))]
    inside = along[pick]
    intrudes = bool(np.any((inside > lay.cell) & (inside < gap - lay.cell)))
    return {"polygon_m": poly, "intrudes": intrudes}


def features(lay: Layers, footprint_radius: float | None = None,
             doors: Sequence[Sequence[float]] = ()) -> Features:
    """Regions, passages and frontiers of a grid (see the module docstring); ``doors``: the map x,
    y of door-labelled objects."""
    state = lay.state
    free = state == FREE
    R, C = state.shape
    cell = lay.cell
    chords = _passage_chords(state, cell)
    # cut the chords out of the free space; the parts are the regions
    import cv2

    barrier = np.zeros((R, C), np.uint8)
    for _, e1, e2, *_ in chords:
        for pt_a, pt_b in ((e1, e2),):
            cv2.line(barrier, (math.floor(pt_a[0]), math.floor(pt_a[1])),
                     (math.floor(pt_b[0]), math.floor(pt_b[1])), 1, 1)
    cut = free & (barrier == 0)
    parts, n_parts = ndimage.label(cut)
    areas = ndimage.sum(cut, parts, np.arange(1, n_parts + 1)) * cell * cell
    index = np.zeros(n_parts + 1, np.int32)
    regions: list[Region] = []
    for i, area in enumerate(areas, start=1):
        if area < MIN_REGION_M2:
            continue
        cells = parts == i
        filled = ndimage.binary_fill_holes(cells)
        enclosed_unknown = int((filled & (state == UNKNOWN)).sum())
        n_cells = int(cells.sum())
        seen = int((cells & (lay.floor_observed == 1)).sum())
        rows, cols = np.nonzero(cells)
        regions.append(Region(cells, round(float(area), 4),
                              lay.centre(float(rows.mean()), float(cols.mean())),
                              _polygon(cells, lay),
                              round(seen / (n_cells + enclosed_unknown), 4)))
        index[i] = len(regions)
    labels = index[parts].astype(np.int32)
    near_lab = _geodesic_labels(free, labels)  # barrier cells and small parts: nearest region
    passages: list[Passage] = []
    for _, e1, e2, c1, c2 in chords:
        u = (e2 - e1) / max(float(np.linalg.norm(e2 - e1)), 1e-9)
        n = np.array([-u[1], u[0]])
        mid = (e1 + e2) / 2
        sides = []
        for sgn in (-1, 1):
            votes: dict[int, int] = {}
            for t in np.arange(1.0, 3.5, 0.5):
                x, y = mid + sgn * n * t
                r, c = math.floor(y), math.floor(x)
                if 0 <= r < R and 0 <= c < C and labels[r, c] > 0:
                    votes[int(labels[r, c])] = votes.get(int(labels[r, c]), 0) + 1
            sides.append(max(votes, key=lambda v: votes[v]) if votes else 0)
        if sides[0] == 0 or sides[1] == 0 or sides[0] == sides[1]:
            continue
        j1 = np.array([lay.origin[0] + e1[0] * cell, lay.origin[1] + e1[1] * cell])
        j2 = np.array([lay.origin[0] + e2[0] * cell, lay.origin[1] + e2[1] * cell])
        width = float(np.linalg.norm(j2 - j1))
        centre = (j1 + j2) / 2
        normal = math.degrees(math.atan2(n[1], n[0]))  # from sides[0] (behind) to sides[1]
        walls = c1 is not None and c2 is not None and state[c1] == OCCUPIED and \
            state[c2] == OCCUPIED
        door_obj = any(math.hypot(d[0] - centre[0], d[1] - centre[1]) <= DOOR_OBJECT_M
                       for d in doors)
        if footprint_radius and width < 2 * footprint_radius:
            kind = "narrow"
        elif door_obj or (walls and DOOR_MIN_W <= width <= DOOR_MAX_W):
            kind = "door"
        else:
            kind = "opening"
        leaf = _leaf(state, lay, j1, j2) if kind == "door" else None
        passages.append(Passage(centre, width, normal, (j1, j2), (sides[0] - 1, sides[1] - 1),
                                kind, leaf))
    # frontiers: free cells next to unknown ones (or the grid's edge)
    unknown = np.pad(state == UNKNOWN, 1, constant_values=True)
    next_unknown = unknown[:-2, 1:-1] | unknown[2:, 1:-1] | unknown[1:-1, :-2] | unknown[1:-1, 2:]
    front = free & next_unknown
    flab, nf = ndimage.label(front, structure=np.ones((3, 3)))
    frontiers: list[Frontier] = []
    for i in range(1, nf + 1):
        cells = np.argwhere(flab == i)
        length = len(cells) * cell
        if length < MIN_FRONTIER_M:
            continue
        mean = cells.mean(axis=0)
        r, c = cells[int(np.argmin(np.hypot(cells[:, 0] - mean[0], cells[:, 1] - mean[1])))]
        labs = near_lab[cells[:, 0], cells[:, 1]]
        labs = labs[labs > 0]
        region = int(np.bincount(labs).argmax()) - 1 if labs.size else None
        frontiers.append(Frontier(lay.centre(float(r), float(c)), round(length, 4), region))
    return Features(regions, passages, frontiers, near_lab)


# ------------------------------------------------------------------------------------------------
# objects


def object_region(feat: Features, lay: Layers, centre: Sequence[float], size: Sequence[float],
                  yaw_deg: float, seen_from: Sequence[Sequence[float]] = ()) -> int | None:
    """The region index of an object (map.md §5): the region its footprint (or centre) overlaps
    most, its boundary included (one cell around); else, for an object on a wall, the region of
    the cameras that saw it (``seen_from``: their map x, y); None when it touches no region."""
    R, C = lay.shape
    a = math.radians(yaw_deg)
    u = np.array([math.cos(a), math.sin(a)])
    v = np.array([-u[1], u[0]])
    half = np.array([max(size[0], lay.cell) / 2 + lay.cell, max(size[1], lay.cell) / 2 + lay.cell])
    c0 = np.asarray(centre[:2], np.float64)
    reach = float(np.linalg.norm(half)) + lay.cell
    r0, cc0 = lay.cell_of(c0 - reach)
    r1, cc1 = lay.cell_of(c0 + reach)
    rr, cc = np.meshgrid(np.arange(max(r0, 0), min(r1, R - 1) + 1),
                         np.arange(max(cc0, 0), min(cc1, C - 1) + 1), indexing="ij")
    cx = lay.origin[0] + (cc + 0.5) * lay.cell - c0[0]
    cy = lay.origin[1] + (rr + 0.5) * lay.cell - c0[1]
    inside = (np.abs(cx * u[0] + cy * u[1]) <= half[0]) & (np.abs(cx * v[0] + cy * v[1]) <= half[1])
    labs = feat.labels[rr[inside], cc[inside]]
    labs = labs[labs > 0]
    if labs.size:
        return int(np.bincount(labs).argmax()) - 1
    votes = []
    for xy in seen_from:
        r, c = lay.cell_of(xy)
        if 0 <= r < R and 0 <= c < C and feat.labels[r, c] > 0:
            votes.append(int(feat.labels[r, c]))
    return int(np.bincount(votes).argmax()) - 1 if votes else None


def thin_points(xyz: NDArray[Any], lo: Array, hi: Array, voxel: float, max_points: int
                ) -> tuple[NDArray[np.int64], bool]:
    """(indices, truncated) of the points inside the box ``[lo, hi]``: one per ``voxel``
    (``core.geometry.voxel_downsample_indices``), then, beyond ``max_points``, the budget grid
    (``core.geometry.budget_voxel_indices``)."""
    from oh_my_slam.core.geometry import budget_voxel_indices, voxel_downsample_indices

    idx = np.flatnonzero(np.all((xyz >= lo) & (xyz <= hi), axis=1))
    sel = idx[np.sort(voxel_downsample_indices(xyz[idx], voxel))] if idx.size else idx
    if sel.size <= max_points:
        return sel, False
    keep, _ = budget_voxel_indices(xyz[sel], max_points)
    return sel[np.sort(keep)], True
