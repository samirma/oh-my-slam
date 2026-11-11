"""One standpoint and rotation consistency without priors (map.md §1, §4; README "Frames without
priors").

**Two-view fit** (``two_view``). Two frames' epipolar inliers (``anchored.epipolar``: SIFT
matches that agree with some relative pose, essential matrix and RANSAC), as unit rays; a
rotation-only fit (RANSAC on the rays) gives the relative rotation of a turn in place. The
pair's *two-view parallax* (map.md §1, the one parallax measure) is the median, over the
epipolar inliers, of the rotation-only residual (pixels of the images as sent: its angle times
the pair's focal length) divided by the sine of the ray's angle to the epipolar fit's baseline (at least
``SIN_MIN``). About 0 when the camera only turned. Below ``PARALLAX_MAX_PX`` the pair is
translation-unconstrained: one standpoint. The pair's *overlap* is the smaller of the two images'
8 x 8 inlier-cell coverages (``anchored.cell_share``, map.md §1 "Overlap"). The image-measured
relative rotation is the epipolar fit's (unbiased whether the camera moved or not), the
rotation-only fit's only for a translation-unconstrained pair whose epipolar rotation leaves it by
more than ``EPIPOLAR_AGREE_DEG`` (a degenerate essential matrix), independent of any joint solve.

**Standpoint groups** (``groups``). Frames without priors whose pairwise parallax is below
``PARALLAX_MAX_PX`` (pairs overlapping by at least ``VIEW_SHARE``) form one standpoint:
connected through such pairs. A group is registered as a pure rotation about a common centre:
the position of the overlapped stored image-confirmed keyframe with the most pair inliers with
the group (ties: the lowest keyframe id; stored keyframes never move), else the mean of its
frames' image-only centres; its position sigma is their spread (RMS distance to that centre). Each frame keeps the
engine's image-only rotation: a two-view rotation of a small baseline absorbs the translation (a
robot turning with its camera ahead of its axis: 1.5-2 deg per pair on the test room, adding up
along a chain), the engine's multi-view or lifted-point rotation does not. Its rotation sigma is
the noise of its path along the strongest pairs (a maximum spanning tree by inliers) from that
stored keyframe, else the group's first frame in request order.

**Rotation consistency** (``consistency``). Over the pairs of overlapping neighbours (view share
at least ``VIEW_SHARE``) among the update's registered frames and the confirmed keyframes they
are measured against, each with at least ``CHECK_MIN_INLIERS`` epipolar inliers: the angle
between their committed relative rotation and the pair's image-measured one; the hard limit ``rotation_inconsistent`` fails when the median exceeds
``MEDIAN_MAX_DEG`` or any pair ``PAIR_MAX_DEG``. No measurable pair passes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.types import Intrinsics
from oh_my_slam.mapping import sigmas
from oh_my_slam.mapping.anchored import (
    EPIPOLAR_MIN_MATCHES,
    AnchoredMap,
    Features,
    FrameIn,
    Keyframe,
    Placement,
    cell_share,
    epipolar,
    kabsch,
    pose,
    rot_deg,
    rotation_ransac,
    unit,
)

Array = NDArray[np.float64]

PARALLAX_MAX_PX = 1.0  # map.md §1: a pair below this two-view parallax (px) is one standpoint
VIEW_SHARE = 0.10  # map.md §4: overlapping neighbours overlap by at least this
MEDIAN_MAX_DEG = 3.0  # rotation_inconsistent: the pairs' median disagreement above this...
PAIR_MAX_DEG = 10.0  # ... or one pair's above this
MIN_MATCHES = EPIPOLAR_MIN_MATCHES  # a pair with fewer epipolar inliers is not measured
# the rotation check measures a pair only on this many epipolar inliers: a weakly overlapping
# pair's two-view rotation is off by 4-8 deg on 50 inliers (test room), which no committed map
# could meet within the 10 deg pair limit
CHECK_MIN_INLIERS = 100
EPIPOLAR_AGREE_DEG = 2.0  # a pair's epipolar rotation this near its rotation fit's is taken
SIN_MIN = 0.3  # a ray this near the baseline direction has its parallax scaled by at most 1/this
NEIGHBOURS = 6  # each frame is measured against this many nearest frames (by committed pose)


@dataclass
class View:
    """A frame as the two-view fits see it: its name, intrinsics, features, committed rotation
    and centre (image-only for a new frame), whether it is a stored confirmed keyframe, and its
    request order (new frames)."""

    name: str
    K: Intrinsics
    feats: Features
    R: Array
    c: Array
    stored: bool = False
    order: int = 0
    kid: str = ""  # a stored keyframe's id (the standpoint centre's tie-break)


@dataclass
class TwoView:
    R_ab: Array  # the image-measured rotation of b's camera in a's: R_a^T R_b
    parallax: float  # the two-view parallax (px, map.md §1)
    inliers: int
    share: float  # the overlap: the smaller of the two coverages
    noise_deg: float  # the rotation-only fit's residual angle over sqrt(inliers)

    @property
    def unconstrained(self) -> bool:
        """Translation-unconstrained (map.md §1): below ``PARALLAX_MAX_PX``."""
        return self.parallax < PARALLAX_MAX_PX

    @property
    def standpoint(self) -> bool:
        return self.unconstrained and self.share >= VIEW_SHARE


def two_view(a: View, b: View) -> TwoView | None:
    """The two-view fit of a pair (module docstring), None below ``MIN_MATCHES``."""
    e = epipolar(a.feats, a.K, b.feats, b.K)
    if e is None:
        return None
    thr = e.thr
    da, db = unit(e.ra), unit(e.rb)
    rot = rotation_ransac(da, db, 2.0 * thr)
    if rot is None:
        return None
    R_rot = kabsch(db[rot[1]], da[rot[1]]) if int(rot[1].sum()) >= 3 else rot[0]
    ang = np.arccos(np.clip(np.sum((db @ R_rot.T) * da, axis=1), -1.0, 1.0))
    # recoverPose: x_b = R x_a + t, so R is a's camera in b's and b's centre in a's is -R^T t
    _, R_e, t_e, _ = cv2.recoverPose(e.E, e.ra[:, :2], e.rb[:, :2], np.eye(3))
    base = unit(-(np.asarray(R_e, np.float64).T @ np.asarray(t_e, np.float64).ravel()))
    # a ray's residual grows with (baseline / depth) x sin(its angle to the baseline): near the
    # epipole (a camera moving forward) it vanishes, so each residual is divided by that sine
    sin = np.sqrt(np.clip(1.0 - (da @ base) ** 2, 0.0, 1.0))
    focal = math.sqrt(a.K.fx * b.K.fx)  # the residual angle in pixels
    parallax = float(np.median(ang * focal / np.maximum(sin, SIN_MIN)))
    R_epi = np.asarray(R_e, np.float64).T
    # the epipolar rotation is unbiased whether or not the camera moved; the rotation-only fit
    # absorbs a small translation as rotation (a robot turning with its camera ahead of its
    # axis). It stands in only where the epipolar fit is degenerate: a pair without parallax
    # whose epipolar rotation leaves the rotation fit's by more than ``EPIPOLAR_AGREE_DEG``
    R_ab = R_rot if parallax < PARALLAX_MAX_PX and rot_deg(R_epi.T @ R_rot) > EPIPOLAR_AGREE_DEG \
        else R_epi
    n = len(e.ua)
    noise = math.degrees(float(np.sqrt(np.mean(np.minimum(ang, 2.0 * thr) ** 2))))
    share = min(cell_share(e.ua, a.K), cell_share(e.ub, b.K))
    return TwoView(np.asarray(R_ab, np.float64), parallax, n, share,
                   noise / math.sqrt(max(n, 1)))


class Pairs:
    """The two-view fits of an update, each pair fitted once."""

    def __init__(self) -> None:
        self.fits: dict[tuple[str, str], TwoView | None] = {}

    def get(self, a: View, b: View) -> TwoView | None:
        key = (a.name, b.name)
        if key not in self.fits:
            rev = self.fits.get((b.name, a.name), False)
            if rev is not False:
                self.fits[key] = None if rev is None else TwoView(
                    rev.R_ab.T, rev.parallax, rev.inliers, rev.share, rev.noise_deg)
            else:
                self.fits[key] = two_view(a, b)
        return self.fits[key]


def neighbours(views: Sequence[View], i: int, k: int = NEIGHBOURS) -> list[int]:
    """The ``k`` views nearest view ``i`` by centre distance (0.5 m) plus optical-axis angle
    (30°), and its neighbours in request order."""
    v = views[i]
    score = []
    for j, w in enumerate(views):
        if j == i:
            continue
        d = float(np.linalg.norm(w.c - v.c))
        a = rot_deg(v.R.T @ w.R)
        score.append((d / 0.5 + a / 30.0, j))
    near = [j for _, j in sorted(score)[:k]]
    for j, w in enumerate(views):
        if not w.stored and not v.stored and abs(w.order - v.order) == 1 and j not in near:
            near.append(j)
    return near


@dataclass
class Placed:
    R: Array
    c: Array
    sigma_pos_m: float
    sigma_rot_deg: float  # the path's pair noise and the reference's (uncalibrated)
    group: int


def groups(views: Sequence[View], pairs: Pairs, ref_sigma_rot_deg: float = 0.0
           ) -> dict[str, Placed]:
    """The new frames of ``views`` that share a standpoint (module docstring), each placed as a
    pure rotation about its group's centre; frames of no group are left out."""
    n = len(views)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges: list[tuple[int, int, int, TwoView]] = []
    for i, v in enumerate(views):
        if v.stored:
            continue
        for j in neighbours(views, i):
            tv = pairs.get(v, views[j])  # R_ab: j's camera in i's
            if tv is not None and tv.standpoint:
                edges.append((i, j, tv.inliers, tv))
                parent[find(i)] = find(j)
    out: dict[str, Placed] = {}
    members: dict[int, list[int]] = {}
    for i in range(n):
        members.setdefault(find(i), []).append(i)
    for gid, (_, mem) in enumerate(sorted(members.items())):
        new = [i for i in mem if not views[i].stored]
        if len(mem) < 2 or not new:
            continue
        stored = [i for i in mem if views[i].stored]
        # the overlapped stored keyframe with the most pair inliers with the group, ties to the
        # lowest keyframe id (map.md §1): its centre
        strength = {i: sum(e[2] for e in edges if i in (e[0], e[1])) for i in stored}
        root = min(stored, key=lambda i: (-strength[i], views[i].kid, views[i].name)) \
            if stored else min(new, key=lambda i: views[i].order)
        centre = views[root].c if stored else np.mean([views[i].c for i in new], axis=0)
        spread = math.sqrt(float(np.mean([np.sum((views[i].c - centre) ** 2) for i in new])))
        # rotations along a maximum spanning tree of the group's standpoint pairs
        R = {root: views[root].R}
        var = {root: (0.0 if views[root].stored else ref_sigma_rot_deg) ** 2}
        mine = sorted((e for e in edges if e[0] in mem and e[1] in mem), key=lambda e: -e[2])
        grown = True
        while grown:
            grown = False
            for i, j, _, tv in mine:
                if j in R and i not in R:
                    R[i], var[i] = R[j] @ tv.R_ab.T, var[j] + tv.noise_deg ** 2
                    grown = True
                elif i in R and j not in R and not views[j].stored:
                    R[j], var[j] = R[i] @ tv.R_ab, var[i] + tv.noise_deg ** 2
                    grown = True
        for i in new:
            if i not in R:  # reached only through a stored keyframe other than the root
                continue
            # its rotation stays the engine's image-only one (``join_standpoint``: a two-view
            # rotation of a small baseline is biased, and the tree adds the biases up); the tree
            # gives its rotation sigma
            out[views[i].name] = Placed(views[i].R.copy(), centre.copy(), spread,
                                        math.sqrt(var[i]), gid)
    return out


def consistency(views: Sequence[View], pairs: Pairs) -> tuple[float | None, float | None]:
    """The median and largest disagreement (deg) between committed and image-measured relative
    rotations over the overlapping pairs of the new frames of ``views`` (with each other and
    with the stored ones); (None, None) without a measurable pair."""
    dis: list[float] = []
    seen: set[frozenset[str]] = set()
    for i, v in enumerate(views):
        if v.stored:
            continue
        for j in neighbours(views, i):
            w = views[j]
            key = frozenset((v.name, w.name))
            if key in seen:
                continue
            seen.add(key)
            tv = pairs.get(v, w)
            if tv is None or tv.share < VIEW_SHARE or tv.inliers < CHECK_MIN_INLIERS:
                continue
            committed = v.R.T @ w.R
            dis.append(rot_deg(committed.T @ tv.R_ab))
    if not dis:
        return None, None
    return float(np.median(dis)), float(max(dis))


def inconsistent(median_deg: float | None, max_deg: float | None) -> bool:
    """The hard limit ``rotation_inconsistent`` (map.md §4)."""
    return median_deg is not None and max_deg is not None \
        and (median_deg > MEDIAN_MAX_DEG or max_deg > PAIR_MAX_DEG)


# -- the engines' orchestration (map.md §1, §4): the stored views, the standpoint registration and
# the rotation hard limit of an update ---------------------------------------------------------


def stored_views(am: AnchoredMap, near: list[tuple[Any, Any]], exclude: list[Keyframe] | None = None
                 ) -> list[View]:
    """The map's image-confirmed keyframes (but ``exclude``) as views: those among the
    ``NEIGHBOURS`` nearest (by centre and optical axis) of each pose of ``near`` ((R, centre)
    pairs), with their features."""
    conf = [k for k in am.confirmed() if all(k is not n for n in exclude or [])]
    chosen: dict[str, Keyframe] = {}
    for R, c in near:
        score = sorted((float(np.linalg.norm(k.centre - c)) / 0.5
                        + rot_deg(R.T @ k.T[:3, :3]) / 30.0, i)
                       for i, k in enumerate(conf))
        for _, i in score[:NEIGHBOURS]:
            chosen[conf[i].id] = conf[i]
    out = []
    for k in chosen.values():
        am._load_kf(k)
        if k.feats is not None and len(k.feats.kps):
            out.append(View(k.name, k.K, k.feats, k.T[:3, :3].copy(), k.centre.copy(), True,
                            kid=k.id))
    return out


@dataclass
class NewFrame:
    """A frame without a prior registered by an engine, for ``standpoints``: its name,
    intrinsics, request index, whether it has a depth image, its image-only pose and features."""

    name: str
    K: Intrinsics
    index: int
    depth: bool
    T: Array
    feats: Features


def standpoints(am: AnchoredMap, new: list[NewFrame], pairs: Pairs
                ) -> dict[int, tuple[Array, float, float]]:
    """The frames of ``new`` that share a standpoint with each other or with a stored confirmed
    keyframe (``groups``, map.md §1): by request index, their pure-rotation pose about the
    group's centre and its calibrated sigmas (the position's: the group's spread)."""
    views = stored_views(am, [(n.T[:3, :3], n.T[:3, 3]) for n in new])
    views += [View(n.name, n.K, n.feats, n.T[:3, :3].copy(), n.T[:3, 3].copy(), False, n.index)
              for n in new]
    got = groups(views, pairs)
    out = {}
    for n in new:
        g = got.get(n.name)
        if g is not None:
            sp, sr = sigmas.calibrated(g.sigma_pos_m, g.sigma_rot_deg, n.depth, None)
            out[n.index] = (pose(g.R, g.c), round(sp, 4), round(sr, 3))
    return out


def join_standpoint(am: AnchoredMap, pairs: Pairs, f: FrameIn, feats: Features, pl: Placement
                    ) -> Placement:
    """A frame without a prior that the prior-first engine registered, placed as one
    standpoint with the confirmed keyframe it shares one with (the most inliers, ties to the
    lowest keyframe id; map.md §1): a pure rotation about that keyframe's centre, its position
    sigma the distance of its image-only centre to it. Its rotation stays its image-only one
    (PnP on the map's lifted points): a two-view rotation of a small baseline is biased by the
    translation it absorbs (1.5-2 deg on a robot turning with its camera 10 cm ahead of its
    axis, test room), the lifted points' is not. Its displacement is the rotation between its
    committed and its image-only pose (0 here). As it was when it shares none."""
    assert pl.T is not None
    new = View(f.name, f.K, feats, pl.T[:3, :3].copy(), pl.T[:3, 3].copy())
    best = None
    for v in stored_views(am, [(pl.T[:3, :3], pl.T[:3, 3])]):
        tv = pairs.get(new, v)
        if tv is not None and tv.standpoint and (
                best is None or (-tv.inliers, v.kid) < (-best[1].inliers, best[0].kid)):
            best = (v, tv)
    if best is None:
        return pl
    v, tv = best
    spread = float(np.linalg.norm(pl.T[:3, 3] - v.c))
    sp = sigmas.calibrated(spread, 0.0, f.depth is not None, None)[0]
    image_T = pl.T if pl.align_T is None else pl.align_T
    T = pose(image_T[:3, :3], v.c)
    return Placement(pl.name, "registered", None, T, round(sp, 4), pl.sigma_rot_deg, pl.inliers,
                     "rotation", lifted=pl.lifted, align_T=image_T,
                     map_residual=(0.0, rot_deg(T[:3, :3].T @ image_T[:3, :3])))


def rotation_check(am: AnchoredMap, placed: list[Placement], pairs: Pairs) -> bool:
    """The hard limit ``rotation_inconsistent`` (map.md §4) over the update's registered
    frames ``placed`` (their keyframes) and the stored confirmed keyframes they neighbour."""
    kfs = [k for pl in placed if pl.status == "registered" and pl.keyframe is not None
           for k in [am.keyframe(pl.keyframe)] if k is not None and k.feats is not None]
    if not kfs:
        return False
    views = stored_views(am, [(k.T[:3, :3], k.centre) for k in kfs], kfs)
    views += [View(k.name, k.K, k.feats, k.T[:3, :3].copy(), k.centre.copy(), False, i)
              for i, k in enumerate(kfs) if k.feats is not None]
    return inconsistent(*consistency(views, pairs))


def views_inconsistent(new: list[View], stored: list[View]) -> bool:
    """The hard limit ``rotation_inconsistent`` (map.md §4) for an engine that hands its
    committed poses as views (the older mapper): the new views against each other and the
    stored ones among the ``NEIGHBOURS`` nearest of each."""
    near: dict[str, View] = {}
    for v in new:
        score = sorted((float(np.linalg.norm(w.c - v.c)) / 0.5 + rot_deg(v.R.T @ w.R) / 30.0, i)
                       for i, w in enumerate(stored))
        for _, i in score[:NEIGHBOURS]:
            near[stored[i].name] = stored[i]
    return inconsistent(*consistency([*near.values(), *new], Pairs()))
