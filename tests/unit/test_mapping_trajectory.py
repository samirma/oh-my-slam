"""Keyframe trajectories (``mapping.trajectory``, ``sfm.vet``): joining a secondary reconstruction
through shared keyframes, the collapse guard, capture-order runs and anchors."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from oh_my_slam.core.geometry import Sim3, rot_z
from oh_my_slam.core.types import Pose
from oh_my_slam.mapping import trajectory as traj
from oh_my_slam.mapping.frame import transform_pose
from tests.synth.scene import look_at


def walk(n: int, step: float = 0.4, seed: int = 0) -> dict[str, Pose]:
    """A handheld walk: camera centres along a gently curving path, the view turning with it."""
    rng = np.random.default_rng(seed)
    out = {}
    heading = 0.0
    c = np.array([0.0, 0.0, 1.5])
    for k in range(n):
        heading += rng.uniform(-0.25, 0.25)
        d = np.array([np.cos(heading), np.sin(heading), 0.0])
        c = c + step * d
        out[f"f{k:06d}.jpg"] = look_at(c, c + d + np.array([0.0, 0.0, -0.3]))
    return out


def close(a: Pose, b: Pose, tol: float = 1e-6) -> bool:
    return bool(np.allclose(a.R, b.R, atol=tol) and np.allclose(a.t, b.t, atol=tol))


def test_merge_by_shared_places_a_secondary_reconstruction() -> None:
    truth = walk(14)
    names = sorted(truth)
    main = {n: truth[n] for n in names[:9]}  # main reconstruction: the map frame
    # a secondary reconstruction in its own (similarity-related) frame, sharing four keyframes
    sim = Sim3(0.37, rot_z(1.1) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float),
               np.array([4.0, -2.0, 0.5]))
    other = {n: transform_pose(sim, truth[n]) for n in names[5:]}
    extra = traj.merge_by_shared(main, other)
    assert extra is not None
    assert sorted(extra) == names[9:]
    for n, T in extra.items():
        assert close(T, truth[n]), n


def test_merge_by_shared_refuses_what_it_cannot_verify() -> None:
    truth = walk(10)
    names = sorted(truth)
    main = {n: truth[n] for n in names[:6]}
    # two shared keyframes: too few to check the similarity
    assert traj.merge_by_shared(main, {n: truth[n] for n in names[4:]}) is None
    # nothing new to add
    assert traj.merge_by_shared(main, {n: truth[n] for n in names[:5]}) is None
    # a shared keyframe the two reconstructions disagree on (a misregistration)
    other = {n: truth[n] for n in names[3:]}
    other[names[4]] = Pose(other[names[4]].R, other[names[4]].t + np.array([0.0, 0.6, 0.0]))
    assert traj.merge_by_shared(main, other) is None
    # shared keyframes at one centre: no scale
    still = {n: Pose(truth[n].R, np.zeros(3)) for n in names[:6]}
    assert traj.merge_by_shared(still, {n: Pose(truth[n].R, np.zeros(3)) for n in names[3:]}
                                ) is None


def collapsed_walk() -> tuple[dict[str, Pose], list[str], set[str]]:
    """A walk whose keyframes 5-11 a global mapper shrank onto one centre (orientations kept,
    no triangulated support); returns (poses, capture order, supported keyframes)."""
    poses = walk(20)
    order = sorted(poses)
    centre = poses[order[5]].t.copy()
    bad = order[5:12]
    rng = np.random.default_rng(1)
    for n in bad:
        poses[n] = Pose(poses[n].R, centre + 1e-3 * rng.normal(size=3))
    return poses, order, set(order) - set(bad)


def test_collapse_guard_flags_a_segment_shrunk_onto_one_centre() -> None:
    poses, order, supported = collapsed_walk()
    radius = traj.collapse_radius(poses, order, supported)
    assert 0.02 < radius < 0.06  # a tenth of the 0.4 m step
    assert traj.collapsed_keyframes(poses, supported, radius) == set(order[5:12])
    # the same shapes with support (e.g. triangulated points of a camera turning in place) pass
    assert traj.collapsed_keyframes(poses, set(order), radius) == set()
    # and so do pairs whose matches are a pure rotation (panoramic two-view geometry)
    turns = {frozenset((a, b)) for a in order[5:12] for b in order[5:12] if a != b}
    assert traj.collapsed_keyframes(poses, supported, radius, turns) == set()


def test_collapse_guard_keeps_near_duplicates_and_a_moving_camera() -> None:
    poses = walk(12)
    order = sorted(poses)
    # a near-duplicate keyframe (same place, same view: two video frames 1/15 s apart)
    dup = Pose(poses[order[3]].R, poses[order[3]].t + 1e-4)
    poses["f000003b.jpg"] = dup
    order = sorted(poses)
    radius = traj.collapse_radius(poses, order, set())  # nothing supported: no scale at all
    assert radius == 0.0
    assert traj.collapsed_keyframes(poses, set(), radius) == set()
    radius = traj.collapse_radius(poses, order, set(order))
    assert traj.collapsed_keyframes(poses, set(), radius) == set()


def test_collapse_guard_on_a_single_other_keyframe() -> None:
    """Two keyframes suffice: an unsupported pose on another keyframe's centre looking elsewhere
    is rejected, the supported one stays."""
    poses = walk(8)
    order = sorted(poses)
    a, b = order[2], order[6]
    poses[b] = Pose(rot_z(0.5) @ poses[a].R, poses[a].t.copy())  # 29° to the left
    supported = set(order) - {b}
    radius = traj.collapse_radius(poses, order, supported)
    assert traj.collapsed_keyframes(poses, supported, radius) == {b}


def test_capture_runs_and_temporal_anchors() -> None:
    order = [f"f{k:06d}.jpg" for k in range(12)]
    todo = {order[k] for k in (2, 3, 4, 8, 11)}
    runs = traj.capture_runs(order, todo)
    assert runs == [[order[2], order[3], order[4]], [order[8]], [order[11]]]
    posed = [n for n in order if n not in todo]
    assert traj.temporal_anchors(order[2], order[4], posed, 4) == [
        order[1], order[5], order[0], order[6]]
    # only one side has posed keyframes: all anchors come from it
    assert traj.temporal_anchors(order[11], order[11], posed, 3) == [
        order[10], order[9], order[7]]
    assert traj.temporal_anchors(order[2], order[4], [], 4) == []


def test_summary_reports_the_largest_jump() -> None:
    poses = walk(6)
    order = sorted(poses)
    for n in order[4:]:  # a 5 m jump between the 4th and 5th keyframe
        poses[n] = Pose(poses[n].R, poses[n].t + np.array([0.0, 5.0, 0.0]))
    s = traj.summary(poses, order)
    assert s["keyframes"] == 6
    assert s["max_step_between"] == [order[3], order[4]]
    assert s["max_step"] > 4.5
    assert s["median_step"] == pytest.approx(0.4, abs=0.05)


def chain(names: list[str], k: int) -> dict[frozenset[str], int]:
    """Consecutive keyframes linked with weight ``k`` (shared points or inlier matches)."""
    return {frozenset(p): k for p in pairwise(names)}


def articulated_walk() -> tuple[dict[str, Pose], dict[str, Pose], dict[str, float], list[str]]:
    """(truth, SfM solution, depth ratios, names) of a walk whose stretch 10-15 hangs on keyframe 9
    through an articulation: the solver returned it at 1/2.5 of its size about 9, and 16-19,
    which hang on 15, came along. Monocular/SfM depth ratios: 0.7, and 2.5x that for 10-15."""
    truth = walk(20, seed=5)
    names = sorted(truth)
    p9 = truth[names[9]].t
    solved = dict(truth)
    for n in names[10:16]:
        solved[n] = Pose(truth[n].R, p9 + (truth[n].t - p9) / 2.5)
    shift = solved[names[15]].t - truth[names[15]].t
    for n in names[16:]:
        solved[n] = Pose(truth[n].R, truth[n].t + shift)
    ratios = {n: 0.7 * (2.5 if n in names[10:16] else 1.0) for n in names}
    return truth, solved, ratios, names


def test_scale_blocks_are_cut_where_the_depth_ratios_disagree() -> None:
    _, _, ratios, names = articulated_walk()
    rng = np.random.default_rng(3)
    noisy = {n: r * (1 + rng.uniform(-0.06, 0.06)) for n, r in ratios.items()}
    del noisy[names[13]]  # too few points for a ratio: joins the block it shares points with
    blocks = traj.scale_blocks(noisy, chain(names, 40), names)
    assert blocks == [set(names[:10]), set(names[10:16]), set(names[16:])]
    factors = traj.block_factors(noisy, blocks)
    assert factors[0] == 1.0 and factors[2] == 1.0
    assert factors[1] == pytest.approx(2.5, rel=0.08)


def test_rescale_blocks_repairs_a_misscaled_stretch_and_what_hangs_on_it() -> None:
    truth, solved, ratios, names = articulated_walk()
    links = chain(names, 120) | {frozenset((names[3], names[17])): 16}  # a weak loop closure
    fix = traj.fix_blocks(solved, ratios, chain(names, 40), links)
    assert sorted(fix.poses) == names[10:]
    for n, T in fix.poses.items():
        assert close(T, truth[n]), n
    assert [(len(m["keyframes"]), m["factor"], m["pivot"]) for m in fix.moves] == [
        (6, 2.5, names[9]), (4, 1.0, names[15])]
    assert fix.unanchored == set()


def test_rescale_blocks_leaves_a_consistent_reconstruction_alone() -> None:
    names = [f"f{k:06d}.jpg" for k in range(30)]
    poses = walk(30)
    rng = np.random.default_rng(4)
    ratios = {n: 1.3 * (1 + rng.uniform(-0.08, 0.08)) for n in names}
    ratios[names[7]] *= 1.3  # one keyframe with a biased depth: a block too small to judge
    fix = traj.fix_blocks(poses, ratios, chain(names, 40), chain(names, 100))
    assert fix.poses == {} and fix.unanchored == set()
    # two parts that share no points but agree in scale
    covis = chain(names[:15], 40) | chain(names[15:], 40)
    fix = traj.fix_blocks(poses, {n: 1.3 for n in names}, covis, chain(names, 100))
    assert fix.poses == {} and fix.unanchored == set()


def test_misscaled_block_without_a_link_is_not_guessed() -> None:
    _, solved, ratios, names = articulated_walk()
    links = chain(names[:10], 120) | chain(names[10:], 120)  # nothing links 9 and 10
    fix = traj.fix_blocks(solved, ratios, chain(names, 40), links)
    assert fix.poses == {}
    assert fix.unanchored == set(names[10:16])
    assert traj.pivot_keyframe(names[10:16], names[:10], links) is None


def test_rescale_blocks_across_a_bridge_keeps_the_link() -> None:
    """A block hanging on the rest by matches only (a bridge 9-10: no shared points): its scale
    is free about its own end of the link, so that is where it is scaled."""
    truth = walk(20, seed=6)
    names = sorted(truth)
    p10 = truth[names[10]].t
    solved = dict(truth)
    for n in names[10:]:
        solved[n] = Pose(truth[n].R, p10 + (truth[n].t - p10) * 0.3)
    ratios = {n: 0.7 / (0.3 if n in names[10:] else 1.0) for n in names}
    covis = chain(names[:10], 40) | chain(names[10:], 40)
    fix = traj.fix_blocks(solved, ratios, covis, chain(names, 21))
    assert sorted(fix.poses) == names[10:]
    for n, T in fix.poses.items():
        assert close(T, truth[n]), n


def test_fix_blocks_levels_a_tilted_block_with_its_gravity() -> None:
    """A stretch whose orientation rests on one weak link (a bridge of matches 9-10, next to no
    shared points) came out tilted by 30° about its end of the link: the gravity estimates of its
    keyframes show it, and it is levelled about that keyframe. Scale agrees; what hangs on it
    follows."""
    from oh_my_slam.core.geometry import rotation_between

    truth = walk(20, seed=7)
    names = sorted(truth)
    a = np.radians(30.0)
    tilt = rotation_between(np.array([0.0, 0.0, 1.0]), np.array([np.sin(a), 0.0, np.cos(a)]))
    p10 = truth[names[10]].t
    solved = dict(truth)
    for n in names[10:16]:
        solved[n] = Pose(tilt @ truth[n].R, p10 + tilt @ (truth[n].t - p10))
    shift = solved[names[15]].t - truth[names[15]].t
    for n in names[16:]:
        solved[n] = Pose(truth[n].R, truth[n].t + shift)
    up_cam = {n: truth[n].R.T @ np.array([0.0, 0.0, 1.0]) for n in names}  # exact gravity
    ups = {n: solved[n].R @ up_cam[n] for n in names}
    ratios = {n: 0.7 for n in names}
    covis = chain(names, 40)
    covis[frozenset((names[9], names[10]))] = 3  # the bridge
    covis[frozenset((names[15], names[16]))] = 3  # and the next weak link
    fix = traj.fix_blocks(solved, ratios, covis, chain(names, 120), ups)
    assert sorted(fix.poses) == names[10:]
    for n, T in fix.poses.items():
        assert close(T, truth[n]), n
    assert fix.moves[0]["tilt_deg"] == pytest.approx(30.0, abs=0.01)
    assert fix.moves[0]["factor"] == 1.0
    # after the fix nothing disagrees with gravity; before, the block did
    assert traj.tilted_keyframes({**solved, **fix.poses}, up_cam, names[:10], names) == set()
    assert traj.tilted_keyframes(solved, up_cam, names[:10], names) == set(names[10:16])


def test_block_tilts_ignore_small_and_unknown_disagreement() -> None:
    blocks = [{"a", "b", "c"}, {"d", "e", "f"}, {"g"}]
    ups = {"a": np.array([0.0, 0.0, 1.0]), "b": np.array([0.02, 0.0, 1.0]),
           "c": np.array([0.0, 0.03, 1.0]), "d": np.array([0.05, 0.0, 1.0]),
           "e": np.array([0.06, 0.0, 1.0]), "f": np.array([0.04, 0.02, 1.0]),
           "g": np.array([1.0, 0.0, 0.0])}
    tilts = traj.block_tilts(ups, blocks)
    for R in tilts:  # 3° apart: within tolerance; one estimate: too few to judge
        np.testing.assert_allclose(R, np.eye(3))


def test_level_block_rotates_a_block_rigidly_onto_gravity() -> None:
    from oh_my_slam.core.geometry import rotation_between

    truth = walk(12, seed=8)
    names = sorted(truth)
    block = names[6:]
    up_cam = {n: truth[n].R.T @ np.array([0.0, 0.0, 1.0]) for n in names}
    a = np.radians(18.0)
    tilt = rotation_between(np.array([0.0, 0.0, 1.0]), np.array([0.0, np.sin(a), np.cos(a)]))
    c = truth[names[6]].t
    tilted = {**truth, **{n: Pose(tilt @ truth[n].R, c + tilt @ (truth[n].t - c)) for n in block}}
    new = traj.level_block(tilted, block, names[6], up_cam, np.array([0.0, 0.0, 1.0]))
    assert sorted(new) == block
    for n in block:
        assert close(new[n], truth[n]), n
    # within tolerance: left alone
    assert traj.level_block(truth, block, names[6], up_cam, np.array([0.0, 0.0, 1.0])) == {}


def test_a_block_pinned_by_two_keyframes_is_not_moved() -> None:
    """A stretch whose depth ratios are 30 % off (monocular depth biased by what it looks at) but
    that shares many points with two keyframes of the rest, some way apart: the reconstruction's
    scale holds there."""
    poses = walk(20)
    names = sorted(poses)
    ratios = {n: 0.7 * (1.3 if n in names[10:16] else 1.0) for n in names}
    covis = chain(names, 80) | {frozenset((names[7], names[10])): 60}
    assert traj.anchoring_keyframes(names[10:16], covis) == {names[7], names[9], names[16]}
    fix = traj.fix_blocks(poses, ratios, covis, chain(names, 120))
    assert fix.poses == {} and fix.moves == [] and fix.unanchored == set()
    # tied to one viewpoint only (keyframes 9 and 9b at one spot): judged, and scaled
    poses["f000009b.jpg"] = Pose(poses[names[9]].R, poses[names[9]].t + 0.01)
    ratios["f000009b.jpg"] = 0.7
    covis = chain(names, 80) | {frozenset(("f000009b.jpg", names[10])): 60,
                                frozenset(("f000009b.jpg", names[9])): 500}
    fix = traj.fix_blocks(poses, ratios, covis, chain(names, 120))
    assert fix.moves[0]['keyframes'] == names[10:16]  # and 16-19, hanging on it, follow
    assert sorted(fix.poses) == names[10:]


def test_a_floating_run_is_moved_between_its_capture_order_neighbours() -> None:
    """Keyframes 8-13 were placed 11 m away and turned 70° (multi-view poses anchored on
    keyframes they do not overlap): they keep their shape and go back between 7 and 14."""
    truth = walk(20, seed=9)
    names = sorted(truth)
    run = names[8:14]
    turn = rot_z(np.radians(70.0))
    far = {n: Pose(turn @ truth[n].R, turn @ truth[n].t + np.array([9.0, -6.0, 0.0])) for n in run}
    placed = {**truth, **far}
    found = traj.floating_runs(placed, names, run)
    assert found == [(run, names[7], names[14])]
    new = traj.attach_run(placed, run, names[7], names[14], np.array([0.0, 0.0, 1.0]))
    # the run's own shape is kept; its ends land about one step from the neighbours
    for a, b in pairwise(run):
        assert np.linalg.norm(new[a].t - new[b].t) == pytest.approx(
            np.linalg.norm(truth[a].t - truth[b].t), abs=1e-9)
    for n in run:
        assert traj.rotation_deg(new[n].R, truth[n].R) < 5.0
        assert np.linalg.norm(new[n].t - truth[n].t) < 0.5
    # a run in place does not float; a run at the end of the video moves next to its one neighbour
    assert traj.floating_runs(truth, names, run) == []
    tail = {n: Pose(truth[n].R, truth[n].t + 12.0) for n in names[16:]}
    found = traj.floating_runs({**truth, **tail}, names, names[16:])
    assert found == [(names[16:], names[15], None)]
    new = traj.attach_run({**truth, **tail}, names[16:], names[15], None,
                          np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(new[names[16]].t, truth[names[15]].t)
