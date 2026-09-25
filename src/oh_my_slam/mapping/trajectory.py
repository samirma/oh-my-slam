"""Keyframe trajectories: which SfM poses to trust, joining reconstructions, capture-order anchors.

COLMAP's global mapper (GLOMAP) averages rotations over the verified pairs, then solves the camera
centres from point bearings. A part of the view graph that hangs on the rest by one weak link — a
walk into another room joined by a few dozen matches (a bridge), or a stretch attached through a
single keyframe (an articulation) — has no scale of its own there: every scale about the link fits
the constraints, and which one the solver returns depends on its random start (on the user's
apartment walk the kitchen came out shrunk onto one point, at 0.44x, 0.68x or 23x across seeds);
its orientation rests on that one link too, and can be tens of degrees off. Bundle adjustment
keeps both. The checks:

* support — an SfM pose is accepted only with ``SUPPORT_MIN_POINTS`` triangulated observations
  (``sfm.vet``): a part shrunk onto a point triangulates (almost) nothing; the keyframe counts as
  unplaced and goes to the fallbacks.
* blocks (``fix_blocks``) — the monocular metric depth measures each keyframe's SfM scale (median
  depth ratio at its triangulated points, a few per cent apart within a rigid reconstruction), and
  its gravity estimate the tilt of its orientation (within 1-3° on correctly posed keyframes).
  Co-visible keyframes whose ratios agree form blocks; a block whose ratio differs from the largest
  block's by more than ``SCALE_BLOCK_TOL``, or whose gravity is more than ``TILT_BLOCK_TOL_DEG``
  off, is scaled and levelled about the keyframe it hangs on (``pivot_keyframe``) — the free mode of
  a bridge or an articulation — and the mapper then refines its poses with the matches and depth.
* the collapse guard (``collapsed_keyframes``) — a keyframe whose centre coincides with another
  keyframe's while the two show different content (optical axes apart) is not accepted as posed
  unless its own evidence supports it or the pair's matches say the camera turned in place; and a
  re-placed keyframe whose gravity still disagrees by ``TILT_REJECT_DEG`` is not accepted either.

Keyframes left out of the main reconstruction join it through a secondary reconstruction that
shares keyframes with it (``merge_by_shared``: similarity from the shared poses, verified on each
of them), else through the anchored multi-view path of the mapper; for video its anchors are the
posed keyframes next to them in capture order (``capture_runs``, ``temporal_anchors``).
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.types import Pose

SUPPORT_MIN_POINTS = 20  # triangulated observations an SfM pose needs to be trusted
COLLAPSE_STEP_FRACTION = 0.1  # centres closer than this fraction of the median step coincide
COLLAPSE_MIN_ANGLE_DEG = 3.0  # optical axes further apart: the two keyframes show different content
MERGE_MIN_SHARED = 3
MERGE_ROT_TOL_DEG = 2.0
MERGE_POS_TOL = 0.05  # of the shared keyframes' spread
COVIS_MIN_POINTS = 15  # triangulated points two keyframes share to be compared
SCALE_PAIR_TOL = 1.15  # co-visible keyframes whose depth ratios differ more are cut apart
SCALE_BLOCK_TOL = 1.12  # a block whose ratio differs more from the reference block's is mis-scaled
SCALE_BLOCK_MIN = 3  # keyframes with a ratio a block needs to be judged
ANCHOR_MIN_POINTS = 50  # shared points that tie a block to a keyframe outside it
ANCHOR_APART_STEPS = 0.5  # two such keyframes this many median steps apart pin the block
TILT_PAIR_TOL_DEG = 10.0  # co-visible keyframes whose gravity estimates differ more are cut apart
TILT_BLOCK_TOL_DEG = 5.0  # a block whose gravity is further off the reference block's is levelled
TILT_REJECT_DEG = 25.0  # a re-placed keyframe whose gravity is further off is not accepted
FLOAT_JUMP_STEPS = 10.0  # a joined video run this many median steps from its neighbours floats


def rotation_deg(Ra: NDArray[Any], Rb: NDArray[Any]) -> float:
    """Angle of the rotation between two orientations, degrees."""
    c = (np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def steps(poses: dict[str, Pose], order: list[str]) -> NDArray[np.float64]:
    """Distances between the camera centres of capture-order neighbours among ``poses``."""
    names = [n for n in order if n in poses]
    if len(names) < 2:
        return np.zeros(0)
    C = np.array([poses[n].t for n in names])
    return np.asarray(np.linalg.norm(np.diff(C, axis=0), axis=1), np.float64)


def collapse_radius(poses: dict[str, Pose], order: list[str], supported: Collection[str]) -> float:
    """``COLLAPSE_STEP_FRACTION`` of the median distance between capture-order neighbours among
    the ``supported`` keyframes (0 when they do not move)."""
    ok = set(supported)
    st = steps({n: T for n, T in poses.items() if n in ok}, order)
    st = st[st > 0]
    return float(COLLAPSE_STEP_FRACTION * np.median(st)) if len(st) else 0.0


def collapsed_keyframes(poses: dict[str, Pose], supported: Collection[str], radius: float,
                        rotation_pairs: Collection[frozenset[str]] = (),
                        min_angle_deg: float = COLLAPSE_MIN_ANGLE_DEG) -> set[str]:
    """Keyframes not to accept as posed: placed within ``radius`` of another keyframe's centre
    while their optical axes differ by more than ``min_angle_deg`` (different content), unless
    the keyframe is ``supported`` by its own evidence (triangulated points, matches that agree
    with a refined pose) or the pair is in ``rotation_pairs`` (its matches say the camera turned in
    place). Near-duplicate keyframes (same centre, same view) are fine."""
    names = sorted(poses)
    if len(names) < 2 or radius <= 0:
        return set()
    C = np.array([poses[n].t for n in names])
    F = np.array([poses[n].R[:, 2] for n in names])
    dist = np.linalg.norm(C[:, None] - C[None], axis=2)
    cos = np.clip(F @ F.T, -1.0, 1.0)
    clash = (dist <= radius) & (cos < np.cos(np.radians(min_angle_deg)))
    np.fill_diagonal(clash, False)
    ok = set(supported)
    turns = {frozenset(p) for p in rotation_pairs}
    out: set[str] = set()
    for i, j in zip(*np.nonzero(np.triu(clash)), strict=True):
        a, b = names[int(i)], names[int(j)]
        if frozenset((a, b)) in turns:
            continue
        out.update(n for n in (a, b) if n not in ok)
    return out


def merge_by_shared(main: dict[str, Pose], other: dict[str, Pose],
                    min_shared: int = MERGE_MIN_SHARED) -> dict[str, Pose] | None:
    """Poses (in ``main``'s frame) of the keyframes of a secondary reconstruction ``other`` that
    ``main`` lacks, through the similarity that maps the keyframes both hold onto ``main``'s poses.
    None when they share fewer than ``min_shared`` keyframes, their centres coincide (no scale), or
    a shared keyframe does not land on its pose in ``main`` (the two disagree)."""
    from oh_my_slam.mapping.frame import similarity_by_poses, transform_pose

    shared = sorted(set(main) & set(other))
    extra = sorted(set(other) - set(main))
    if len(shared) < min_shared or not extra:
        return None
    ref = [main[n] for n in shared]
    centres = np.array([r.t for r in ref])
    spread = float(np.linalg.norm(centres - centres.mean(0), axis=1).max())
    if spread <= 1e-9:
        return None
    sim = similarity_by_poses([other[n] for n in shared], ref)
    for n, r in zip(shared, ref, strict=True):
        T = transform_pose(sim, other[n])
        if (rotation_deg(T.R, r.R) > MERGE_ROT_TOL_DEG
                or np.linalg.norm(T.t - r.t) > MERGE_POS_TOL * spread):
            return None
    return {n: transform_pose(sim, other[n]) for n in extra}


def capture_runs(order: list[str], todo: Collection[str]) -> list[list[str]]:
    """Maximal runs of ``todo`` keyframes that are consecutive in the capture ``order``."""
    want = set(todo)
    runs: list[list[str]] = []
    current: list[str] = []
    for n in order:
        if n in want:
            current.append(n)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def temporal_anchors(first: str, last: str, posed: list[str], k: int) -> list[str]:
    """Up to ``k`` of the ``posed`` keyframes (in capture order) nearest in capture order to a run
    ``first`` … ``last``: alternately the nearest before it and the nearest after it."""
    before = [n for n in posed if n < first][::-1]
    after = [n for n in posed if n > last]
    out: list[str] = []
    while len(out) < k and (before or after):
        for side in (before, after):
            if side and len(out) < k:
                out.append(side.pop(0))
    return out


def summary(poses: dict[str, Pose], order: list[str]) -> dict[str, Any]:
    """Capture-order step statistics of a trajectory: median and largest step, and where."""
    names = [n for n in order if n in poses]
    st = steps(poses, order)
    if not len(st):
        return {"keyframes": len(names)}
    k = int(np.argmax(st))
    return {"keyframes": len(names), "median_step": round(float(np.median(st)), 4),
            "max_step": round(float(st[k]), 4), "max_step_between": [names[k], names[k + 1]]}


def _median_ratio(ratios: dict[str, float], block: Collection[str], min_ratios: int = 1
                  ) -> float:
    vals = [ratios[n] for n in block if n in ratios and ratios[n] > 0]
    return float(np.median(vals)) if len(vals) >= min_ratios else float("nan")


def scale_blocks(ratios: dict[str, float], covis: dict[frozenset[str], int],
                 names: Collection[str], ups: dict[str, NDArray[Any]] | None = None,
                 min_points: int = COVIS_MIN_POINTS, pair_tol: float = SCALE_PAIR_TOL,
                 block_tol: float = SCALE_BLOCK_TOL, tilt_tol_deg: float = TILT_PAIR_TOL_DEG
                 ) -> list[set[str]]:
    """Blocks of keyframes with one SfM scale and tilt: components of the keyframes with a depth
    ratio (``ratios``: monocular / SfM depth) linked by sharing ``min_points`` triangulated points
    (``covis``), ratios within ``pair_tol`` and, where both have one, gravity estimates (``ups``,
    in the reconstruction) within ``tilt_tol_deg``. Keyframes without a ratio then join the block
    they share the most points with (none: a block of their own), and blocks that share points and
    agree in their median ratio (within ``block_tol``) and mean gravity are one (a keyframe without a ratio may be all
    that links them; it never links blocks that disagree). Largest block first."""
    up = ups or {}
    ratioed = {n for n in names if n in ratios and ratios[n] > 0}
    parent = {n: n for n in names}

    def level(a: str | set[str], b: str | set[str]) -> bool:
        ua = mean_direction([up[n] for n in sorted({a} if isinstance(a, str) else a) if n in up])
        ub = mean_direction([up[n] for n in sorted({b} if isinstance(b, str) else b) if n in up])
        return ua is None or ub is None or angle_deg(ua, ub) <= tilt_tol_deg

    def find(n: str) -> str:
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    strong = {pair: k for pair, k in covis.items()
              if k >= min_points and all(n in parent for n in pair)}
    for pair in strong:
        a, b = sorted(pair)
        if a in ratioed and b in ratioed and abs(np.log(ratios[a] / ratios[b])) <= np.log(
                pair_tol) and level(a, b):
            parent[find(a)] = find(b)
    assigned = set(ratioed)
    for _ in range(3):  # chains of keyframes without a ratio
        for n in names:
            if n in assigned:
                continue
            votes: dict[str, int] = {}
            for pair, k in strong.items():
                if n in pair:
                    (m,) = pair - {n}
                    if m in assigned:
                        votes[find(m)] = votes.get(find(m), 0) + k
            if votes:
                parent[find(n)] = max(sorted(votes), key=lambda b: votes[b])
                assigned.add(n)
    merged = True
    while merged:
        merged = False
        groups: dict[str, set[str]] = {}
        for n in names:
            groups.setdefault(find(n), set()).add(n)
        med = {g: _median_ratio(ratios, members) for g, members in groups.items()}
        for pair in strong:
            a, b = (find(n) for n in pair)
            if a != b and np.isfinite(med[a]) and np.isfinite(med[b]) and abs(
                    np.log(med[a] / med[b])) <= np.log(block_tol) and level(groups[a], groups[b]):
                parent[a] = b
                merged = True
                break
    groups = {}
    for n in names:
        groups.setdefault(find(n), set()).add(n)
    return sorted(groups.values(), key=lambda g: (-len(g), min(g)))


def block_factors(ratios: dict[str, float], blocks: list[set[str]],
                  tol: float = SCALE_BLOCK_TOL, min_ratios: int = SCALE_BLOCK_MIN) -> list[float]:
    """Per block (the first is the reference), the factor its SfM geometry must be multiplied by
    to agree in scale with the reference: its median depth ratio over the reference's when that
    differs by more than ``tol`` and both have ``min_ratios`` ratios, else 1."""
    r_ref = _median_ratio(ratios, blocks[0], min_ratios) if blocks else float("nan")
    out = []
    for b in blocks:
        q = _median_ratio(ratios, b, min_ratios) / r_ref
        out.append(float(q) if np.isfinite(q) and abs(np.log(q)) > np.log(tol) else 1.0)
    return out


def mean_direction(vectors: list[NDArray[Any]]) -> NDArray[np.float64] | None:
    """Robust mean of directions: the normalised component-wise median of the unit vectors."""
    if not vectors:
        return None
    u = np.array([np.asarray(v, np.float64) / np.linalg.norm(v) for v in vectors])
    m = np.median(u, axis=0)
    n = float(np.linalg.norm(m))
    return m / n if n > 1e-9 else None


def angle_deg(a: NDArray[Any], b: NDArray[Any]) -> float:
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def block_tilts(ups: dict[str, NDArray[Any]], blocks: list[set[str]],
                tol_deg: float = TILT_BLOCK_TOL_DEG, min_frames: int = SCALE_BLOCK_MIN
                ) -> list[NDArray[np.float64]]:
    """Per block (the first is the reference), the rotation that levels it with the reference:
    the smallest rotation taking the block's mean gravity direction (``ups``: each keyframe's up
    estimate rotated into the reconstruction) onto the reference's, when they are more than
    ``tol_deg`` apart and both have ``min_frames`` estimates, else the identity. (Gravity says
    nothing about the heading.)"""
    from oh_my_slam.core.geometry import rotation_between

    def mean(block: set[str]) -> NDArray[np.float64] | None:
        vals = [ups[n] for n in sorted(block) if n in ups]
        return mean_direction(vals) if len(vals) >= min_frames else None

    ref = mean(blocks[0]) if blocks else None
    out = []
    for b in blocks:
        u = mean(b)
        if ref is None or u is None or angle_deg(u, ref) <= tol_deg:
            out.append(np.eye(3))
        else:
            out.append(rotation_between(u, ref))
    return out


def pivot_keyframe(block: Collection[str], anchored: Collection[str],
                   links: dict[frozenset[str], int]) -> str | None:
    """The ``anchored`` keyframe (outside ``block``) with the most verified matches (``links``:
    inliers per pair) to the block — where the block hangs on the rest; None without any."""
    inside, fixed = set(block), set(anchored) - set(block)
    score: dict[str, int] = {}
    for pair, k in links.items():
        a, b = sorted(pair)
        for x, y in ((a, b), (b, a)):
            if x in fixed and y in inside:
                score[x] = score.get(x, 0) + k
    return max(sorted(score), key=lambda n: score[n]) if score else None


def anchoring_keyframes(block: Collection[str], covis: dict[frozenset[str], int],
                        min_points: int = ANCHOR_MIN_POINTS) -> set[str]:
    """Keyframes outside ``block`` that share ``min_points`` triangulated points with it."""
    inside = set(block)
    shared: dict[str, int] = {}
    for pair, k in covis.items():
        a, b = sorted(pair)
        if (a in inside) != (b in inside):
            out = b if a in inside else a
            shared[out] = shared.get(out, 0) + k
    return {n for n, k in shared.items() if k >= min_points}


def pinned(anchors: Collection[str], poses: dict[str, Pose], apart: float) -> bool:
    """Whether ``anchors`` hold two keyframes at least ``apart`` from each other: a block tied to
    two viewpoints by shared points has its scale and orientation fixed; tied to one (an
    articulation, or neighbouring keyframes at one spot) it has not."""
    C = np.array([poses[n].t for n in sorted(anchors) if n in poses]).reshape(-1, 3)
    if len(C) < 2 or apart <= 0:
        return False
    return bool((np.linalg.norm(C[:, None] - C[None], axis=2) >= apart).any())


@dataclass
class BlockFix:
    poses: dict[str, Pose]  # new poses of the keyframes that move
    moves: list[dict[str, Any]]  # per moved block: keyframes, factor, tilt, pivot
    unanchored: set[str]  # keyframes of mis-scaled or tilted blocks that hang on nothing placed


def fix_blocks(poses: dict[str, Pose], ratios: dict[str, float],
               covis: dict[frozenset[str], int], links: dict[frozenset[str], int],
               ups: dict[str, NDArray[Any]] | None = None) -> BlockFix:
    """Make an SfM reconstruction agree in scale with its keyframes' metric depth and in tilt with
    their gravity, block by block (``scale_blocks``, ``block_factors``, ``block_tilts``; ``ups``:
    each keyframe's up estimate rotated into the reconstruction). The blocks are placed along the
    maximum spanning tree of their verified matches (``links``: inliers per keyframe pair), from
    the largest block: each is scaled by its factor and levelled about the keyframe it hangs on in
    its parent (``pivot_keyframe``) — an articulation, whose points the block shares — or, where
    they share next to no points (a bridge of matches), about the block's own keyframe at the other
    end of the link; and it follows the pivot where the pivot moved — a block right in scale and
    tilt that hangs on a moved one moves with it. A block that shares many points with placed
    keyframes some way apart (``anchoring_keyframes``, ``pinned``) is not judged: only a bridge or
    an articulation (one viewpoint) leaves scale and orientation free. Blocks that need nothing
    and hang on no moved block keep their poses."""
    blocks = scale_blocks(ratios, covis, list(poses), ups)
    if len(blocks) < 2:
        return BlockFix({}, [], set())
    factors = block_factors(ratios, blocks)
    tilts = block_tilts(ups or {}, blocks)
    st = steps(poses, sorted(poses))
    apart = ANCHOR_APART_STEPS * float(np.median(st[st > 0])) if (st > 0).any() else 0.0
    block_of = {n: i for i, b in enumerate(blocks) for n in b}
    between: dict[tuple[int, int], int] = {}  # inliers between two blocks
    for pair, k in links.items():
        a, b = sorted(pair)
        if a in block_of and b in block_of and block_of[a] != block_of[b]:
            key = (min(block_of[a], block_of[b]), max(block_of[a], block_of[b]))
            between[key] = between.get(key, 0) + k
    placed = {0}
    new: dict[str, Pose] = {}
    moves: list[dict[str, Any]] = []
    while True:
        edges = [(k, i if j in placed else j, j if j in placed else i)
                 for (i, j), k in between.items() if (i in placed) != (j in placed)]
        if not edges:
            break
        _, child, parent = max(edges)
        q, Rt, block = factors[child], tilts[child], blocks[child]
        pivot = pivot_keyframe(block, blocks[parent], links)
        assert pivot is not None
        held = {n for i in placed for n in blocks[i]}
        if pinned(anchoring_keyframes(block, covis) & held, poses, apart):
            # points shared with placed keyframes far enough apart pin its scale and orientation:
            # a disagreement with the depth or gravity is theirs, not the reconstruction's
            q, Rt = 1.0, np.eye(3)
        tilted = not np.allclose(Rt, np.eye(3))
        if q != 1.0 or tilted or pivot in new:
            shift = (new[pivot].t - poses[pivot].t) if pivot in new else np.zeros(3)
            about = pivot
            if sum(covis.get(frozenset((pivot, n)), 0) for n in block) < COVIS_MIN_POINTS:
                # a bridge (matches, next to no shared points): the length of the link is not
                # the block's to scale — scale about the block's end of it
                end = pivot_keyframe([pivot], block, links)
                assert end is not None
                about = end
            centre = poses[about].t
            new.update({n: Pose(Rt @ poses[n].R, centre + shift + q * Rt @ (poses[n].t - centre))
                        for n in block})
            moves.append({"keyframes": sorted(block), "factor": round(q, 4),
                          "tilt_deg": round(rotation_deg(Rt, np.eye(3)), 2), "pivot": pivot,
                          "about": about})
        placed.add(child)
    unanchored = {n for i, b in enumerate(blocks) if i not in placed
                  and (factors[i] != 1.0 or not np.allclose(tilts[i], np.eye(3))) for n in b}
    return BlockFix(new, moves, unanchored)


def tilted_keyframes(poses: dict[str, Pose], ups_cam: dict[str, NDArray[Any]],
                     reference: Collection[str], candidates: Collection[str],
                     tol_deg: float = TILT_REJECT_DEG) -> set[str]:
    """``candidates`` whose gravity estimate (``ups_cam``: up in the camera), rotated by their
    pose, is more than ``tol_deg`` off the ``reference`` keyframes' mean up."""
    ref = mean_direction([poses[n].R @ ups_cam[n] for n in sorted(reference)
                          if n in poses and n in ups_cam])
    if ref is None:
        return set()
    return {n for n in candidates if n in poses and n in ups_cam
            and angle_deg(poses[n].R @ ups_cam[n], ref) > tol_deg}


def level_block(poses: dict[str, Pose], block: Collection[str], about: str,
                ups_cam: dict[str, NDArray[Any]], up: NDArray[Any],
                tol_deg: float = TILT_BLOCK_TOL_DEG) -> dict[str, Pose]:
    """The block's poses rotated rigidly about keyframe ``about``'s centre so that its mean
    gravity (``ups_cam``: up in the camera, rotated by the poses) is ``up``; nothing when it is
    within ``tol_deg`` already (or unknown). A rigid rotation keeps the block's own geometry."""
    from oh_my_slam.core.geometry import rotation_between

    u = mean_direction([poses[n].R @ ups_cam[n] for n in sorted(block)
                        if n in poses and n in ups_cam])
    if u is None or about not in poses or angle_deg(u, up) <= tol_deg:
        return {}
    Rt = rotation_between(u, up)
    c = poses[about].t
    return {n: Pose(Rt @ poses[n].R, c + Rt @ (poses[n].t - c)) for n in block if n in poses}


def floating_runs(poses: dict[str, Pose], order: list[str], joined: Collection[str],
                  jump_steps: float = FLOAT_JUMP_STEPS
                  ) -> list[tuple[list[str], str | None, str | None]]:
    """Runs of ``joined`` keyframes (consecutive in the capture ``order``) placed implausibly far
    from the keyframes around them — a camera does not jump ``jump_steps`` median steps between
    two keyframes of a video — with their neighbours before and after (None at an end)."""
    fixed = [n for n in order if n in poses and n not in set(joined)]
    st = steps({n: poses[n] for n in fixed}, fixed)
    st = st[st > 0]
    if not len(st):
        return []
    limit = jump_steps * float(np.median(st))
    out = []
    for run in capture_runs([n for n in order if n in poses], [n for n in joined if n in poses]):
        before = next((n for n in reversed(fixed) if n < run[0]), None)
        after = next((n for n in fixed if n > run[-1]), None)
        jumps = [float(np.linalg.norm(poses[a].t - poses[b].t))
                 for a, b in ((before, run[0]), (run[-1], after)) if a is not None and b is not None]
        if jumps and max(jumps) > limit:
            out.append((run, before, after))
    return out


def attach_run(poses: dict[str, Pose], run: list[str], before: str | None, after: str | None,
               up: NDArray[Any]) -> dict[str, Pose]:
    """The run moved rigidly next to its capture-order neighbours: with both, turned about
    ``up`` so that its first-to-last direction is the before-to-after one (in the horizontal) and
    centred between them; with one, shifted so that its end sits at that neighbour's centre. Its
    own shape, scale and tilt are kept."""
    first, last = poses[run[0]].t, poses[run[-1]].t
    u = np.asarray(up, np.float64) / np.linalg.norm(up)
    R = np.eye(3)
    if before is not None and after is not None:
        a, b = poses[before].t, poses[after].t
        d_run = (last - first) - ((last - first) @ u) * u
        d_nb = (b - a) - ((b - a) @ u) * u
        if np.linalg.norm(d_run) > 1e-6 and np.linalg.norm(d_nb) > 1e-6:
            yaw = np.arctan2(u @ np.cross(d_run, d_nb), d_run @ d_nb)
            K = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])
            R = np.eye(3) + np.sin(yaw) * K + (1.0 - np.cos(yaw)) * K @ K
        src, dst = (first + last) / 2, (a + b) / 2
    elif before is not None:
        src, dst = first, poses[before].t
    elif after is not None:
        src, dst = last, poses[after].t
    else:
        return {}
    return {n: Pose(R @ poses[n].R, dst + R @ (poses[n].t - src)) for n in run}
