"""Pose refinement of multi-view keyframes from feature matches and monocular depth.

Rotation-dominant input (a camera turning in place) is posed by multi-view inference, whose
rotations are degrees and whose camera centres decimetres off. Bundle adjustment cannot repair
that: with almost no parallax, few matches become triangulated points and those few are
ill-conditioned, so the rotations stay poorly constrained, weakly observed keyframes drift, and a
map extended by a later update disagrees with its earlier part. Every verified feature match still
constrains its two keyframes, though: the keypoint of keyframe ``a``, lifted with ``a``'s monocular
depth, must reproject onto the matched keypoint of ``b`` (and vice versa). ``refine_poses``
minimises the robust (Cauchy) reprojection error of all those lifted keypoints over the rotations
and camera centres of the free keyframes — optionally also one focal length shared by all
keyframes — with the other keyframes (the map's, when extending it) fixed. Keypoints without depth
are points at infinity (they constrain the rotations only). The depth's own per-keyframe scale
error only scales the recovered baselines (centimetres for a turning head); it does not bias the
rotations. ``refine_turning`` stages it for rotation-dominant input: rotations first, then the
centres restart where the overlapping fixed keyframes are.

Solver: Levenberg-Marquardt on left rotation increments and centres, iteratively reweighted, with
the robust scale shrinking from about a hundred pixels to a few (the initial poses can be degrees
off); weak priors keep each centre near its initial value where the matches say nothing about it
(pure rotation, points at infinity) and the focal length near its initial value.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.types import Intrinsics, Pose

USED_CONFIGS = (2, 3, 4, 5, 6)  # COLMAP TwoViewGeometry: calibrated … planar-or-panoramic
MIN_INLIERS = 15
MAX_PER_PAIR = 250
ROBUST_SCALES_PX = (100.0, 30.0, 8.0, 2.5)  # graduated Cauchy scale of the reprojection error
MAX_ITERATIONS = 12  # per robust scale
CENTRE_PRIOR_M = 0.25  # weak prior (standard deviation) of each centre around its initial value
FOCAL_PRIOR_LOG = 0.1  # weak prior (standard deviation) of the log focal-length change
NOISE_PX = 1.5  # keypoint noise the priors are weighed against


@dataclass(frozen=True)
class PairMatches:
    """Inlier matches of one verified image pair (full-resolution pixel coordinates)."""

    a: str
    b: str
    uv_a: NDArray[np.float64]
    uv_b: NDArray[np.float64]


@dataclass
class View:
    """A keyframe taking part in the refinement."""

    pose: Pose  # camera-to-map (initial value for a free keyframe)
    K: Intrinsics  # full resolution
    depth: NDArray[Any] | None = None  # z-depth on the grid (metric); None: rays only
    K_grid: Intrinsics | None = None
    full_size: tuple[int, int] | None = None


@dataclass
class PoseFit:
    poses: dict[str, Pose]  # refined camera-to-map poses of the free keyframes
    focal_scale: float = 1.0  # multiply the shared focal length by this
    pairs: int = 0
    matches: int = 0
    median_before_deg: float = float("nan")
    median_after_deg: float = float("nan")
    per_frame_deg: dict[str, float] = field(default_factory=dict)  # median residual per keyframe
    per_frame_matches: dict[str, int] = field(default_factory=dict)  # matches per keyframe

    def summary(self) -> dict[str, Any]:
        return {"pairs": self.pairs, "matches": self.matches,
                "median_before_deg": round(self.median_before_deg, 4),
                "median_after_deg": round(self.median_after_deg, 4),
                "focal_scale": round(self.focal_scale, 6)}


def verified_matches(db_path: Path, names: Collection[str], min_inliers: int = MIN_INLIERS,
                     max_per_pair: int = MAX_PER_PAIR) -> list[PairMatches]:
    """Inlier matches of every verified pair among ``names`` (at most ``max_per_pair`` per pair,
    evenly subsampled), sorted by image names."""
    import pycolmap

    wanted = set(names)
    db = pycolmap.Database.open(str(db_path))
    try:
        id_name = {im.image_id: im.name for im in db.read_all_images() if im.name in wanted}
        pair_ids, geoms = db.read_two_view_geometries()
        keypoints: dict[int, NDArray[np.float64]] = {}
        out = []
        for pid, g in zip(pair_ids, geoms, strict=True):
            if int(g.config) not in USED_CONFIGS or len(g.inlier_matches) < min_inliers:
                continue
            a, b = pycolmap.pair_id_to_image_pair(int(pid))
            if a not in id_name or b not in id_name:
                continue
            for i in (a, b):
                if i not in keypoints:
                    keypoints[i] = np.asarray(db.read_keypoints(i), np.float64)[:, :2]
            m = np.asarray(g.inlier_matches)
            if len(m) > max_per_pair:
                m = m[np.linspace(0, len(m) - 1, max_per_pair).astype(int)]
            na, nb = id_name[a], id_name[b]
            ua, ub = keypoints[a][m[:, 0]], keypoints[b][m[:, 1]]
            out.append(PairMatches(na, nb, ua, ub) if na < nb else PairMatches(nb, na, ub, ua))
    finally:
        db.close()
    return sorted(out, key=lambda p: (p.a, p.b))


# ------------------------------------------------------------------------------------------------
# solver


def _exp(w: NDArray[Any]) -> NDArray[np.float64]:
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def _skew(v: NDArray[Any]) -> NDArray[np.float64]:
    """Per-row cross-product matrices, (N, 3) → (N, 3, 3)."""
    z = np.zeros(len(v))
    return np.stack([np.stack([z, -v[:, 2], v[:, 1]], 1),
                     np.stack([v[:, 2], z, -v[:, 0]], 1),
                     np.stack([-v[:, 1], v[:, 0], z], 1)], 1)


def _usable_depth(view: View) -> NDArray[np.float64] | None:
    """The view's depth with non-positive values and depth-edge pixels set to inf (no depth)."""
    if view.depth is None or view.K_grid is None or view.full_size is None:
        return None
    from oh_my_slam.core.geometry import depth_edge_mask

    d = np.asarray(view.depth, np.float64)
    ok = (d > 0) & np.isfinite(d)
    ok &= ~depth_edge_mask(np.where(ok, d, 0.0))
    return np.where(ok, d, np.inf)


def _depth_at(view: View, usable: NDArray[np.float64] | None, uv: NDArray[Any]
              ) -> NDArray[np.float64]:
    """Monocular z-depth at full-resolution keypoints; inf where there is none (a point at
    infinity)."""
    if usable is None or view.K_grid is None or view.full_size is None:
        return np.full(len(uv), np.inf)
    from oh_my_slam.mapping.frame import grid_uv

    g = grid_uv(uv, view.full_size, view.K_grid)
    u, v = np.rint(g[:, 0]).astype(int), np.rint(g[:, 1]).astype(int)
    h, w = usable.shape
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    out = np.full(len(uv), np.inf)
    out[inside] = usable[v[inside], u[inside]]
    return out


@dataclass
class _Problem:
    """Every match in both directions: the keypoint of ``src`` lifted with its depth and
    reprojected into ``dst``, compared with the matched keypoint there. Directions are stored in
    contiguous segments (one per pair and direction)."""

    src: NDArray[np.int64]  # (N,) view index
    dst: NDArray[np.int64]
    q: NDArray[np.float64]  # (N, 2) src keypoint, normalised at the initial focal length
    m: NDArray[np.float64]  # (N, 2) dst keypoint relative to dst's principal point (px)
    f: NDArray[np.float64]  # (N, 2) dst's initial (fx, fy)
    depth: NDArray[np.float64]  # (N,) z-depth in src (inf: point at infinity)
    starts: NDArray[np.int64]  # (S,) first match of each segment
    seg_views: list[tuple[int, int]]  # (src, dst) view index of each segment


BEHIND_PX = 1000.0  # residual of a point that projects behind the destination camera
CHUNK = 20000  # matches per block when accumulating the normal equations


def _evaluate(pb: _Problem, R: NDArray[Any], C: NDArray[Any], phi: float, jac: bool
              ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Reprojection residuals (N, 2) in pixels and, with ``jac``, their Jacobian (N, 2, 13) with
    respect to (δR_src, C_src, δR_dst, C_dst, φ); ``R`` (V, 3, 3) and ``C`` (V, 3) are the views'
    camera-to-map rotations and centres, the focal lengths the initial ones times exp(φ)."""
    Ra, Rb = R[pb.src], R[pb.dst]
    s = np.exp(phi)
    pa = np.column_stack([pb.q / s, np.ones(len(pb.q))])  # z = 1 ray in src
    fin = np.isfinite(pb.depth)
    z = np.where(fin, pb.depth, 1.0)
    va = np.einsum("nij,nj->ni", Ra, pa)
    P = np.where(fin[:, None], (C[pb.src] - C[pb.dst]) + z[:, None] * va, va)
    X = np.einsum("nji,nj->ni", Rb, P)  # Rb^T P, in dst's camera frame
    Z = X[:, 2]
    front = 1e-3 * np.linalg.norm(X, axis=1) < Z
    Zs = np.where(front, Z, 1.0)
    fx, fy = pb.f[:, 0] * s, pb.f[:, 1] * s
    pred = np.column_stack([fx * X[:, 0] / Zs, fy * X[:, 1] / Zs])
    r = np.where(front[:, None], pred - pb.m, BEHIND_PX)
    if not jac:
        return r, None
    A = np.zeros((len(r), 2, 3))
    A[:, 0, 0] = fx / Zs
    A[:, 0, 2] = -fx * X[:, 0] / Zs**2
    A[:, 1, 1] = fy / Zs
    A[:, 1, 2] = -fy * X[:, 1] / Zs**2
    A[~front] = 0.0
    ARt = np.einsum("nij,nkj->nik", A, Rb)  # de/dP
    J = np.zeros((len(r), 2, 13))
    J[:, :, 0:3] = -np.einsum("nij,njk->nik", ARt, z[:, None, None] * _skew(va))
    J[:, :, 3:6] = np.where(fin[:, None, None], ARt, 0.0)
    J[:, :, 6:9] = np.einsum("nij,njk->nik", ARt, _skew(P))
    J[:, :, 9:12] = -J[:, :, 3:6]
    dpa = np.column_stack([-pa[:, 0], -pa[:, 1], np.zeros(len(pa))])
    J[:, :, 12] = np.einsum("nij,nj->ni", ARt, z[:, None] * np.einsum("nij,nj->ni", Ra, dpa)) \
        + np.where(front[:, None], pred, 0.0)
    return r, J


def _cost(r: NDArray[Any], c: float) -> float:
    return float(0.5 * c * c * np.log1p(np.sum(r * r, axis=1) / (c * c)).sum())


def _normal_equations(pb: _Problem, r: NDArray[Any], J: NDArray[Any], c: float,
                      free: dict[int, int], npar: int) -> tuple[NDArray[np.float64],
                                                                NDArray[np.float64]]:
    """Σ w JᵀJ and Σ w Jᵀr (Cauchy weights at scale ``c``) over the free views' parameters."""
    H = np.zeros((npar, npar))
    g = np.zeros(npar)
    w = 1.0 / (1.0 + np.sum(r * r, axis=1) / (c * c))
    ends = np.r_[pb.starts[1:], len(r)]
    k = 0
    while k < len(pb.starts):
        k1 = k + 1
        while k1 < len(pb.starts) and ends[k1] - pb.starts[k] <= CHUNK:
            k1 += 1
        lo, hi = int(pb.starts[k]), int(ends[k1 - 1])
        Jc, wc = J[lo:hi], w[lo:hi]
        rel = pb.starts[k:k1] - lo
        HB = np.add.reduceat(np.einsum("n,nij,nik->njk", wc, Jc, Jc), rel, axis=0)
        GB = np.add.reduceat(np.einsum("n,nij,ni->nj", wc, Jc, r[lo:hi]), rel, axis=0)
        for j in range(k, k1):
            gi: list[int] = []
            li: list[int] = []
            for view, off in zip(pb.seg_views[j], (0, 6), strict=True):
                if view in free:
                    gi.extend(range(6 * free[view], 6 * free[view] + 6))
                    li.extend(range(off, off + 6))
            gi.append(npar - 1)
            li.append(12)
            H[np.ix_(gi, gi)] += HB[j - k][np.ix_(li, li)]
            g[gi] += GB[j - k][li]
        k = k1
    return H, g


def refine_poses(pairs: list[PairMatches], views: dict[str, View], free: Collection[str],
                 refine_focal: bool = False, use_depth: bool = True,
                 scales: tuple[float, ...] = ROBUST_SCALES_PX) -> PoseFit:
    """Refine the camera-to-map poses of the ``free`` keyframes (see the module docstring).

    ``views``: every keyframe that may take part (free and fixed); pairs with a keyframe outside
    it, or with no free keyframe, are ignored. ``refine_focal`` scales every keyframe's focal
    length by one common factor (one shared camera); ``use_depth=False`` is the pure-rotation
    model (centres stay put)."""
    names = sorted(n for n in free if n in views)
    used_pairs = [p for p in pairs if p.a in views and p.b in views
                  and (p.a in free or p.b in free)]
    fit = PoseFit({n: views[n].pose for n in names}, 1.0, len(used_pairs),
                  sum(len(p.uv_a) for p in used_pairs))
    if not used_pairs or not names:
        return fit
    order = sorted({n for p in used_pairs for n in (p.a, p.b)} | set(names))
    vid = {n: k for k, n in enumerate(order)}
    free_idx = {vid[n]: k for k, n in enumerate(names)}
    usable = {n: _usable_depth(views[n]) if use_depth else None for n in order}
    cols: dict[str, list[Any]] = {k: [] for k in ("src", "dst", "q", "m", "f", "depth")}
    starts, seg_views = [], []
    count = 0
    for p in used_pairs:
        for a, b, ua, ub in ((p.a, p.b, p.uv_a, p.uv_b), (p.b, p.a, p.uv_b, p.uv_a)):
            Ka, Kb = views[a].K, views[b].K
            size = len(ua)
            cols["src"].append(np.full(size, vid[a]))
            cols["dst"].append(np.full(size, vid[b]))
            cols["q"].append((ua - [Ka.cx, Ka.cy]) / [Ka.fx, Ka.fy])
            cols["m"].append(ub - [Kb.cx, Kb.cy])
            cols["f"].append(np.tile([Kb.fx, Kb.fy], (size, 1)))
            cols["depth"].append(_depth_at(views[a], usable[a], ua))
            starts.append(count)
            seg_views.append((vid[a], vid[b]))
            count += size
    pb = _Problem(*(np.concatenate(cols[k]) for k in ("src", "dst", "q", "m", "f", "depth")),
                  starts=np.asarray(starts, np.int64), seg_views=seg_views)
    R = np.stack([views[n].pose.R for n in order]).astype(np.float64)
    C = np.stack([views[n].pose.t for n in order]).astype(np.float64)
    fi = np.array([vid[n] for n in names])
    C0 = C[fi].copy()
    npar = 6 * len(names) + 1
    phi = 0.0
    prior_c = (NOISE_PX / CENTRE_PRIOR_M) ** 2
    prior_f = (NOISE_PX / FOCAL_PRIOR_LOG) ** 2
    f_px = float(np.median(pb.f))

    def residuals(Rs: NDArray[Any], Cs: NDArray[Any], ph: float) -> NDArray[np.float64]:
        return _evaluate(pb, Rs, Cs, ph, jac=False)[0]

    def total(r: NDArray[Any], Cs: NDArray[Any], ph: float, c: float) -> float:
        prior = prior_c * float(np.sum((Cs[fi] - C0) ** 2))
        return _cost(r, c) + 0.5 * (prior + prior_f * ph * ph)

    def degrees(r: NDArray[Any]) -> NDArray[np.float64]:
        return np.degrees(np.linalg.norm(r, axis=1) / f_px)

    fit.median_before_deg = float(np.median(degrees(residuals(R, C, phi))))
    fixed = [] if use_depth else [6 * k + j for k in range(len(names)) for j in (3, 4, 5)]
    if not refine_focal:
        fixed.append(npar - 1)
    for c in scales:
        lam = 1e-3
        r = residuals(R, C, phi)
        current = total(r, C, phi, c)
        for _ in range(MAX_ITERATIONS):
            r, J = _evaluate(pb, R, C, phi, jac=True)
            assert J is not None
            H, g = _normal_equations(pb, r, J, c, free_idx, npar)
            for k in range(len(names)):  # weak centre prior
                sl = slice(6 * k + 3, 6 * k + 6)
                H[sl, sl] += prior_c * np.eye(3)
                g[sl] += prior_c * (C[fi[k]] - C0[k])
            H[-1, -1] += prior_f
            g[-1] += prior_f * phi
            for i in fixed:
                H[i, :] = 0.0
                H[:, i] = 0.0
                H[i, i] = 1.0
                g[i] = 0.0
            improved = False
            delta = np.zeros(npar)
            for _ in range(8):
                A = H + lam * np.diag(np.maximum(np.diag(H), 1e-9))
                try:
                    delta = -np.linalg.solve(A, g)
                except np.linalg.LinAlgError:
                    lam *= 10
                    continue
                Rn, Cn = R.copy(), C.copy()
                for k, v in enumerate(fi):
                    Rn[v] = _exp(delta[6 * k:6 * k + 3]) @ R[v]
                    Cn[v] = C[v] + delta[6 * k + 3:6 * k + 6]
                phn = phi + float(delta[-1])
                cn = total(residuals(Rn, Cn, phn), Cn, phn, c)
                if cn < current:
                    R, C, phi, current = Rn, Cn, phn, cn
                    lam = max(lam / 3, 1e-7)
                    improved = True
                    break
                lam *= 10
            if not improved or float(np.abs(delta).max()) < 1e-7:
                break
    final = degrees(residuals(R, C, phi))
    fit.poses = {n: Pose(R[vid[n]], C[vid[n]]) for n in names}
    fit.focal_scale = float(np.exp(phi))
    fit.median_after_deg = float(np.median(final))
    for n in names:
        mine = (pb.src == vid[n]) | (pb.dst == vid[n])
        if mine.any():
            fit.per_frame_deg[n] = float(np.median(final[mine]))
            fit.per_frame_matches[n] = int(mine.sum() // 2)
    return fit


def centre_hints(pairs: list[PairMatches], views: dict[str, View], free: Collection[str]
                 ) -> dict[str, NDArray[np.float64]]:
    """For each free keyframe, the median centre of the fixed keyframes it shares matches with
    (of all fixed keyframes when it shares none)."""
    fixed = [n for n in views if n not in free]
    if not fixed:
        return {n: views[n].pose.t.copy() for n in free if n in views}
    everywhere = np.median([views[n].pose.t for n in fixed], axis=0)
    near: dict[str, list[str]] = {}
    for p in pairs:
        for a, b in ((p.a, p.b), (p.b, p.a)):
            if a in free and b in views and b not in free:
                near.setdefault(a, []).append(b)
    return {n: (np.median([views[m].pose.t for m in near[n]], axis=0) if n in near
                else everywhere.copy()) for n in free if n in views}


def refine_turning(pairs: list[PairMatches], views: dict[str, View], free: Collection[str],
                   refine_focal: bool = False) -> PoseFit:
    """``refine_poses`` for rotation-dominant input, whose multi-view camera centres are noise
    (decimetres, where the head moves centimetres) that can trap the joint refinement: first the
    rotations alone (pure-rotation model, centres irrelevant), then every free centre restarts at
    the centre of the fixed keyframes it overlaps (``centre_hints``), then rotations and centres
    are refined together with the depth."""
    rot = refine_poses(pairs, views, free, use_depth=False)
    hints = centre_hints(pairs, views, free)
    staged = {n: replace(v, pose=Pose(rot.poses[n].R, hints[n])) if n in rot.poses else v
              for n, v in views.items()}
    fit = refine_poses(pairs, staged, free, refine_focal=refine_focal)
    fit.median_before_deg = rot.median_before_deg
    return fit
