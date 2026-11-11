"""One standpoint and rotation consistency without priors (map.md §1, §4; README "Frames without
priors").

**Two-view fit** (``two_view``). Two frames' SIFT matches (``anchored.match``), as unit rays.
The epipolar fit (essential matrix, RANSAC) keeps the matches that agree with some relative
pose; a rotation-only fit (RANSAC on the rays) gives the relative rotation of a turn in place.
The pair's *parallax* — baseline over median scene depth — is the median, among the epipolar
inliers, of the angle (radians) the rotation-only fit leaves, each over the sine of its ray's
angle to the epipolar fit's baseline direction (at least ``SIN_MIN``): a ray's parallax is the
baseline over its depth times that sine. About 0 when the camera only turned, about the baseline
over the depth when it moved (forward too). The pair's *view share* is the smaller of the two
images' shares of ``GRID`` x ``GRID`` cells holding an epipolar inlier. The image-measured
relative rotation is the rotation-only fit's when the parallax is below ``PARALLAX_MAX``, else the
epipolar fit's, independent of any joint solve.

**Standpoint groups** (``groups``). Frames without priors whose pairwise parallax is below
``PARALLAX_MAX`` (pairs sharing at least ``VIEW_SHARE`` of their view) form one standpoint:
connected through such pairs. A group is registered as a pure rotation about a common centre:
the position of the stored image-confirmed keyframe it overlaps most (the most inliers over its
pairs with the group; stored keyframes never move), else the mean of its frames' image-only
centres; its position sigma is their spread (RMS distance to that centre). Rotations follow the
strongest pairs (a maximum spanning tree by inliers) from a reference: that stored keyframe,
else the group's first frame in request order (the engine's rotation); each frame's rotation
sigma adds, in quadrature, its path's pair noise.

**Rotation consistency** (``consistency``). Over the pairs of overlapping neighbours (view share
at least ``VIEW_SHARE``) among the update's registered frames and the confirmed keyframes they
are measured against: the angle between their committed relative rotation and the pair's
image-measured one; the hard limit ``rotation_inconsistent`` fails when the median exceeds
``MEDIAN_MAX_DEG`` or any pair ``PAIR_MAX_DEG``. No measurable pair passes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.types import Intrinsics
from oh_my_slam.mapping.anchored import (
    REF_WIDTH,
    REPROJ_PX,
    Features,
    kabsch,
    match,
    rays,
    rot_deg,
    rotation_ransac,
    unit,
)

Array = NDArray[np.float64]

PARALLAX_MAX = 0.02  # map.md §1: one standpoint below this baseline over median scene depth
VIEW_SHARE = 0.10  # map.md §4: overlapping neighbours share at least this of their view
MEDIAN_MAX_DEG = 3.0  # rotation_inconsistent: the pairs' median disagreement above this...
PAIR_MAX_DEG = 10.0  # ... or one pair's above this
MIN_MATCHES = 30  # a pair with fewer epipolar inliers is not measured
GRID = 8  # view share: cells per image side
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


@dataclass
class TwoView:
    R_ab: Array  # the image-measured rotation of b's camera in a's: R_a^T R_b
    parallax: float  # baseline over median scene depth (rad of residual angle)
    inliers: int
    share: float
    noise_deg: float  # the rotation-only fit's residual angle over sqrt(inliers)

    @property
    def standpoint(self) -> bool:
        return self.parallax < PARALLAX_MAX and self.share >= VIEW_SHARE


def _share(uv: Array, K: Intrinsics) -> float:
    cx = np.clip((uv[:, 0] / K.width * GRID).astype(int), 0, GRID - 1)
    cy = np.clip((uv[:, 1] / K.height * GRID).astype(int), 0, GRID - 1)
    return len(set(zip(cx.tolist(), cy.tolist(), strict=True))) / float(GRID * GRID)


def two_view(a: View, b: View) -> TwoView | None:
    """The two-view fit of a pair (module docstring), None below ``MIN_MATCHES``."""
    ib, ia = match(b.feats.desc, a.feats.desc)  # one per keypoint of a
    if len(ia) < MIN_MATCHES:
        return None
    ua, ub = a.feats.kps[ia].astype(np.float64), b.feats.kps[ib].astype(np.float64)
    ra, rb = rays(a.K, ua), rays(b.K, ub)
    thr = REPROJ_PX * a.K.width / REF_WIDTH / a.K.fx
    E, mask = cv2.findEssentialMat(ra[:, :2], rb[:, :2], np.eye(3), cv2.RANSAC, 0.999, thr)
    if E is None or mask is None or int(mask.sum()) < MIN_MATCHES:
        return None
    inl = mask.ravel().astype(bool)
    da, db = unit(ra[inl]), unit(rb[inl])
    rot = rotation_ransac(da, db, 2.0 * thr)
    if rot is None:
        return None
    R_rot = kabsch(db[rot[1]], da[rot[1]]) if int(rot[1].sum()) >= 3 else rot[0]
    ang = np.arccos(np.clip(np.sum((db @ R_rot.T) * da, axis=1), -1.0, 1.0))
    # recoverPose: x_b = R x_a + t, so R is a's camera in b's and b's centre in a's is -R^T t
    _, R_e, t_e, _ = cv2.recoverPose(E[:3], ra[inl, :2], rb[inl, :2], np.eye(3))
    base = unit(-(np.asarray(R_e, np.float64).T @ np.asarray(t_e, np.float64).ravel()))
    # a ray's parallax is (baseline / depth) x sin(its angle to the baseline): near the epipole
    # (a camera moving forward) it vanishes, so each residual is scaled by that sine
    sin = np.sqrt(np.clip(1.0 - (da @ base) ** 2, 0.0, 1.0))
    parallax = float(np.median(ang / np.maximum(sin, SIN_MIN)))
    R_ab = R_rot if parallax < PARALLAX_MAX else np.asarray(R_e, np.float64).T
    n = int(inl.sum())
    noise = math.degrees(float(np.sqrt(np.mean(np.minimum(ang, 2.0 * thr) ** 2))))
    share = min(_share(ua[inl], a.K), _share(ub[inl], b.K))
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
        # the stored keyframe the group overlaps most (by its pairs' inliers): its centre
        strength = {i: sum(e[2] for e in edges if i in (e[0], e[1])) for i in stored}
        root = max(stored, key=lambda i: (strength[i], -i)) if stored \
            else min(new, key=lambda i: views[i].order)
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
            out[views[i].name] = Placed(R[i], centre.copy(), spread, math.sqrt(var[i]), gid)
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
            if tv is None or tv.share < VIEW_SHARE:
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
