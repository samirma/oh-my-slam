"""Latest wins, per pixel: parts of older keyframes that newer, well-registered keyframes
contradict are invalidated (their ``valid.png`` loses those pixels), so fusion, the cloud and the
objects stop using them.

Two tests on a 4-px grid, both directions (design step 10):
* free space — an old point projects into a new keyframe that sees clearly *behind* it;
* occlusion of old free space — a new surface lies in front of what an old keyframe observed along
  the same ray (the old ray would carve the new surface).
Margin τ(z) = max(0.15 m, 0.10 z); on the old keyframe's far field, at least the disagreement that
the two keyframes show there (``far_tolerance``). The newer keyframes judge as one observation of
the scene, so a cell is invalidated only when they contradict it together: votes from 2 of them, or
1 vote beyond 1.5 times the margin, and more of them contradicting the cell than re-observing it
within the margin. Depth-edge pixels never vote.

* Across updates (``apply_latest_wins``, before the objects): the keyframes of an update judge the
  map's older keyframes.
* Within an update (``apply_input_order``, after the objects and before the cloud is fused): a
  later keyframe wins over an earlier one, in input order (spec §2.3), so each keyframe is judged
  by the update's keyframes after it, from the latest back. Only keyframes the cloud is drawn
  from vote (well registered and not low confidence), so what they contradict is drawn from
  them: no hole. The objects judge their own changes within the update
  (``objects.update_objects``: an object that an update removes loses the pixels of its masks,
  ``objects.retire_pixels``); this test covers every other surface, and frees the later
  keyframes' view of a change from being carved by the earlier keyframes that saw through it
  (``mapping.geometry.consensus_depths``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from oh_my_slam.core.geometry import depth_edge_mask, project, unproject_pixels
from oh_my_slam.core.images import png_bytes
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.mapping import store

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
    _usable: NDArray[np.bool_] | None = field(default=None, init=False, repr=False,
                                              compare=False)
    _grid: tuple[NDArray[Any], NDArray[Any], NDArray[Any]] | None = field(
        default=None, init=False, repr=False, compare=False)

    def usable(self) -> NDArray[np.bool_]:
        """Valid pixels away from depth edges and the image border (monocular depth of objects
        cut by the border is unreliable). Computed once per view (depth and validity are not
        modified after construction)."""
        if self._usable is None:
            ok = self.valid & (self.depth > 0)
            ok &= ~depth_edge_mask(np.where(ok, self.depth, 0.0))
            h, w = ok.shape
            mh, mw = max(1, int(BORDER * h)), max(1, int(BORDER * w))
            ok[:mh] = ok[-mh:] = False
            ok[:, :mw] = ok[:, -mw:] = False
            self._usable = ok
        return self._usable

    def grid_points(self) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Map-frame points at the centres of GRID cells: (points, rows, cols) of cells. Computed
        once per view, as ``usable``."""
        if self._grid is None:
            use = self.usable()
            h, w = self.depth.shape
            vv, uu = np.mgrid[GRID // 2:h:GRID, GRID // 2:w:GRID]
            m = use[vv, uu]
            v, u = vv[m], uu[m]
            z = self.depth[v, u].astype(np.float64)
            pts = self.T_map_cam.apply(unproject_pixels(u, v, z, self.K.K()))
            self._grid = (pts, v // GRID, u // GRID)
        return self._grid

    def lookup(self, pts_map: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Project map points: (inside mask, point z in this camera, observed depth there)."""
        inside, z, d, _, _ = self.lookup_pixels(pts_map)
        return inside, z, d

    def lookup_pixels(self, pts_map: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any],
                                                             NDArray[Any], NDArray[np.int64],
                                                             NDArray[np.int64]]:
        """``lookup`` plus the pixel (u, v) each point projects to (0 where not inside)."""
        pc = self.T_map_cam.inverse().apply(pts_map)
        uv, z = project(pc, self.K.K())
        h, w = self.depth.shape
        with np.errstate(invalid="ignore"):
            u = np.rint(uv[:, 0])
            v = np.rint(uv[:, 1])
            inside = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        d = np.zeros(len(pts_map))
        ui = np.zeros(len(pts_map), np.int64)
        vi = np.zeros(len(pts_map), np.int64)
        ui[inside], vi[inside] = u[inside], v[inside]
        use = self.usable()
        ok = use[vi[inside], ui[inside]]
        d[inside] = np.where(ok, self.depth[vi[inside], ui[inside]], 0.0)
        inside[np.flatnonzero(inside)[~ok]] = False
        return inside, z, d, ui, vi


def keyframe_view(nf: Any) -> View:
    """View of a placed keyframe of this update (aligned depth, model validity)."""
    return View(nf.depth, nf.frame.valid & (nf.depth > 0), nf.record.K_grid, nf.record.T_map_cam)


def stored_view(path_of: Any, rec: store.FrameRecord) -> View | None:
    """View of a stored keyframe with its current validity; None when it has no depth."""
    if not path_of(store.frame_file(rec.name, "depth.npy")).exists():
        return None
    depth = store.load_depth(path_of, rec.name)
    valid = store.load_valid(path_of, rec.name, depth)
    return View(depth, valid & (depth > 0), rec.K_grid, rec.T_map_cam)


MIN_OVERLAP = 200
MAX_GLOBAL_BIAS = 0.25


def _normalised(d: NDArray[Any], z: NDArray[Any]) -> NDArray[Any] | None:
    """Observed depths divided by the median observed/predicted ratio of the overlap's plausible
    samples (ratio within ``MAX_GLOBAL_BIAS`` of 1).

    Monocular depth maps of two keyframes disagree by a smooth, largely global factor even for
    a static scene; real changes are local. Removing that factor keeps only local
    contradictions. It is measured where the two keyframes may see one surface: the samples one
    of them sees far nearer (occluded) or far beyond (changed) would bias it: with the median of
    every sample, the views of a static rendered room with exact depth, from around it, lost
    thousands of pixels each to votes of the others. ``None`` when the overlap is too small or
    most of it is implausible."""
    if len(d) < MIN_OVERLAP:
        return None
    r = d / np.maximum(z, 1e-6)
    plausible = r[(r >= 1 - MAX_GLOBAL_BIAS) & (r <= 1 + MAX_GLOBAL_BIAS)]
    if 2 * len(plausible) <= len(r):
        return None
    return d / np.median(plausible)


# The far field of a keyframe (beyond FAR_REL times its median depth) is where monocular depth
# disagrees most between keyframes of a static scene: on examples/camera the corridor seen
# through the door (4 times the median depth) lay at a third of that depth in the keyframes panned
# beside, and their votes invalidated it in the keyframes that saw it well (the corridor floor:
# 1 300 points of the 35 000 without the test; 25 000 with this margin). On the old keyframe's
# far cells, a later keyframe's margin is at least the FAR_QUANTILE of the log depth ratios of the
# pair's far samples (from FAR_MIN_SAMPLES of them): there it contradicts only what disagrees more
# than the pair does on most of its far field. A change filling that far field is the pair's
# disagreement itself and is not seen; nearer cells (the office cup) keep τ.
FAR_REL = 1.6
FAR_QUANTILE = 0.9
FAR_MIN_SAMPLES = 50


def far_tolerance(log_ratios: NDArray[Any]) -> float:
    """The log depth ratio a pair of keyframes' far samples stay within (``FAR_QUANTILE`` of
    their magnitudes; 0 below ``FAR_MIN_SAMPLES``)."""
    if len(log_ratios) < FAR_MIN_SAMPLES:
        return 0.0
    return float(np.quantile(np.abs(log_ratios), FAR_QUANTILE))


def contradicted_cells(old: View, new_views: list[View], mapper: Any = map
                       ) -> NDArray[np.bool_]:
    """Boolean GRID-cell mask of ``old`` contradicted by the keyframes ``new_views`` as one
    observation (their order does not matter): votes from two different keyframes, or one vote
    beyond 1.5 times the margin, and more keyframes contradicting the cell than re-observing it
    within the margin (τ; on ``old``'s far field at least ``far_tolerance`` of the pair).
    ``mapper``: how the keyframes' verdicts are computed (``map``, or a thread pool's)."""
    h, w = old.depth.shape
    gh, gw = (h + GRID - 1) // GRID, (w + GRID - 1) // GRID
    pts, rows, cols = old.grid_points()
    usable = old.usable()
    far_at = FAR_REL * float(np.median(old.depth[usable])) if usable.any() else np.inf
    far_cell = old.depth[rows * GRID + GRID // 2, cols * GRID + GRID // 2] > far_at

    def verdicts(nv: View) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
        """(voted, agreed, strong) cells of one keyframe."""
        voted = np.zeros((gh, gw), bool)
        agreed = np.zeros((gh, gw), bool)
        strong = np.zeros((gh, gw), bool)
        # (diff: how far the old keyframe's depth lies before the new one's, the predicted depth,
        # whether the old cell is far, its row, its column), each normalised by the overlap
        samples: list[tuple[NDArray[Any], ...]] = []
        # (1) old point now in free space in front of a new surface
        if len(pts):
            inside, z, d = nv.lookup(pts)
            dn = _normalised(d[inside], z[inside])
            if dn is not None:
                samples.append((dn - z[inside], z[inside], far_cell[inside], rows[inside],
                                cols[inside]))
        # (2) new surface in front of what the old keyframe observed
        q, _, _ = nv.grid_points()
        if len(q):
            inside, z, d = old.lookup(q)
            dn = _normalised(d[inside], z[inside])
            if dn is not None:
                pc = old.T_map_cam.inverse().apply(q[inside])
                uv, _ = project(pc, old.K.K())
                r = np.clip(np.rint(uv[:, 1]).astype(int) // GRID, 0, gh - 1)
                c = np.clip(np.rint(uv[:, 0]).astype(int) // GRID, 0, gw - 1)
                samples.append((dn - z[inside], z[inside], d[inside] > far_at, r, c))
        if not samples:
            return voted, agreed, strong
        diff, z, far, r, c = (np.concatenate(x) for x in zip(*samples, strict=True))
        t = tau(z)
        with np.errstate(divide="ignore", invalid="ignore"):
            spread = far_tolerance(np.log1p(diff[far] / z[far]))
        t[far] = np.maximum(t[far], z[far] * np.expm1(spread))
        hit = diff > t
        voted[r[hit], c[hit]] = True
        s = diff > STRONG * t
        strong[r[s], c[s]] = True
        ok = np.abs(diff) <= t
        agreed[r[ok], c[ok]] = True
        return voted, agreed, strong

    votes = np.zeros((gh, gw), np.int32)
    support = np.zeros((gh, gw), np.int32)
    strong = np.zeros((gh, gw), bool)
    for voted, agreed, s in mapper(verdicts, new_views):
        votes += voted
        support += agreed & ~voted
        strong |= s
    return ((votes >= 2) | strong) & (votes > support)


def cells_to_pixels(cells: NDArray[Any], shape: tuple[int, int]) -> NDArray[np.bool_]:
    full = np.kron(cells, np.ones((GRID, GRID), bool))[: shape[0], : shape[1]]
    return ndimage.binary_dilation(full, iterations=1)


POSE_MAX_RESIDUAL_DEG = 1.0
POSE_MIN_MATCHES = 20


def pose_supported(stats: dict[str, Any], max_residual_deg: float = POSE_MAX_RESIDUAL_DEG
                   ) -> bool:
    """A multi-view pose refined with feature matches (``stats`` has ``pose_matches``) is
    supported when its keyframe has matches that agree with it (median residual <= 1°, or
    ``max_residual_deg``)."""
    if "pose_matches" not in stats:
        return True
    res = stats.get("pose_residual_deg")
    return (stats["pose_matches"] >= POSE_MIN_MATCHES and res is not None
            and res <= max_residual_deg)


# A re-placed pose that its own matches contradict by this much is wrong, not merely uncertain:
# over 181 stored maps, 99 % of 5963 refined keyframes are within 0.43° and all but two within
# 7.6°; those two (a hallway photo at 23°, an office one at 30°) had been placed 3.8 km and 3.9 m
# from the rest.
POSE_REJECT_RESIDUAL_DEG = 10.0


def pose_contradicted(stats: dict[str, Any], max_residual_deg: float = POSE_REJECT_RESIDUAL_DEG
                      ) -> bool:
    """A multi-view pose refined with feature matches (``stats`` has ``pose_matches``) that
    enough of them (``POSE_MIN_MATCHES``) contradict: median residual above
    ``max_residual_deg``. Too few matches judge nothing."""
    res = stats.get("pose_residual_deg")
    return (stats.get("pose_matches", 0) >= POSE_MIN_MATCHES and res is not None
            and res > max_residual_deg)


def well_registered(stats: dict[str, Any], pose_source: str) -> bool:
    if pose_source in ("identity", "multiview"):
        return pose_supported(stats)
    return (stats.get("observations", 0) >= MIN_OBSERVATIONS
            and stats.get("reproj_error", 0.0) <= MAX_REPROJ)


def apply_latest_wins(ctx: Any, records: list[Any], progress: Any) -> None:
    """Update ``valid.png`` of old keyframes contradicted by this update's keyframes."""
    new = [nf for nf in ctx.new if nf.record is not None and nf.depth is not None
           and well_registered(nf.record.stats, nf.record.pose_source)]
    if not new or not ctx.old_frames:
        return
    tx = ctx.tx
    new_views = [keyframe_view(nf) for nf in new]
    changed, pixels = 0, 0
    for rec in ctx.old_frames:
        old = stored_view(tx.current, rec)
        if old is None:
            continue
        cells = contradicted_cells(old, new_views)
        if not cells.any():
            continue
        # every contradicted cell holds the usable (so valid) pixel that voted for it
        kill = cells_to_pixels(cells, old.depth.shape) & old.valid
        tx.write_bytes(store.frame_file(rec.name, "valid.png"),
                       png_bytes((old.valid & ~kill).astype(np.uint8) * 255))
        changed += 1
        pixels += int(kill.sum())
    ctx.notes["latest_wins"] = {"frames_changed": changed, "pixels_invalidated": pixels}
    if changed:
        progress(f"latest wins: {pixels} pixels invalidated in {changed} older keyframes")


# Which keyframes may judge one another within an update: those whose images overlap, which a
# coarse sample of their grid points tells (every OVERLAP_SAMPLE-th GRID cell on each axis; a
# keyframe whose points never land in another's image, nor the other's in its, has fewer than
# MIN_OVERLAP points to compare with it, and ``_normalised`` would not let it vote).
OVERLAP_SAMPLE = 4
INPUT_ORDER_WORKERS = 8


def _voter(rec: Any) -> bool:
    """Whether a keyframe of the update judges the ones before it: well registered, and drawn
    into the cloud (not low confidence)."""
    return well_registered(rec.stats, rec.pose_source) and not rec.low_confidence


def _sees_any(view: View, pts: NDArray[Any]) -> bool:
    """Whether any of the map points ``pts`` lands in front of ``view`` inside its image."""
    if not len(pts):
        return False
    uv, z = project(view.T_map_cam.inverse().apply(pts), view.K.K())
    h, w = view.depth.shape
    with np.errstate(invalid="ignore"):
        return bool(np.any((z > 0) & (uv[:, 0] >= -0.5) & (uv[:, 0] < w - 0.5)
                           & (uv[:, 1] >= -0.5) & (uv[:, 1] < h - 0.5)))


def apply_input_order(ctx: Any, progress: Any) -> None:
    """Latest wins within the update, in input order: the update's placed keyframes, ordered by
    update then input order (a rebuild maps the stored keyframes again with the update's), each
    have the pixels that the voters after them contradict (``contradicted_cells``: the later
    keyframes that ``_voter`` admits, as one observation) invalidated in their staged
    ``valid.png``. The keyframes are judged from the latest back, each by what the later ones
    still hold: a keyframe that a later one contradicted no longer vouches for what it saw there
    (a cup that arrives in the last keyframe is not kept out by the keyframe before it, which
    saw the place empty too)."""
    from concurrent.futures import ThreadPoolExecutor

    placed = sorted((nf for nf in ctx.new if nf.record is not None and nf.depth is not None),
                    key=lambda nf: (nf.record.update_id, nf.record.index))
    voters = [i for i, nf in enumerate(placed) if _voter(nf.record)]
    if not voters or voters[-1] == 0:
        return
    tx = ctx.tx
    views = [View(nf.depth, store.load_valid(tx.current, nf.record.name, nf.depth)
                  & (nf.depth > 0), nf.record.K_grid, nf.record.T_map_cam) for nf in placed]

    def sample(v: View) -> NDArray[Any]:
        pts, r, c = v.grid_points()
        return pts[(r % OVERLAP_SAMPLE == 0) & (c % OVERLAP_SAMPLE == 0)]

    samples = [sample(v) for v in views]
    changed, pixels, pairs = 0, 0, 0
    with ThreadPoolExecutor(max_workers=INPUT_ORDER_WORKERS) as pool:
        for i in reversed(range(voters[-1])):  # the last voter: nothing later judges it
            cand = [j for j in voters if j > i]
            overlap = pool.map(lambda j, i=i: _sees_any(views[j], samples[i])
                               or _sees_any(views[i], samples[j]), cand)
            later = [j for j, ok in zip(cand, overlap, strict=True) if ok]
            pairs += len(later)
            if not later:
                continue
            cells = contradicted_cells(views[i], [views[j] for j in later], pool.map)
            kill = cells_to_pixels(cells, views[i].depth.shape) & views[i].valid
            if not kill.any():
                continue
            v = views[i]
            views[i] = View(v.depth, v.valid & ~kill, v.K, v.T_map_cam)
            views[i].grid_points()  # cached before the threads read it
            tx.write_bytes(store.frame_file(placed[i].record.name, "valid.png"),
                           png_bytes(views[i].valid.astype(np.uint8) * 255))
            changed += 1
            pixels += int(kill.sum())
    ctx.notes["latest_wins_within"] = {"frames_changed": changed, "pixels_invalidated": pixels,
                                       "pairs": pairs}
    if changed:
        progress(f"latest wins within the update: {pixels} pixels invalidated in {changed} "
                 "earlier keyframes")
