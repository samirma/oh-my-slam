"""Multi-view pose refinement from feature matches and monocular depth (``mapping.panorama``) on
an analytic head turning in place: rotations to a small fraction of a degree, camera centres to
about a centimetre, parallax of side steps modelled, focal length recovered, fixed keyframes held."""

from __future__ import annotations

import numpy as np

from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.mapping.panorama import refine_poses
from tests.synth.turning import K, head_pose, perturb, rot_err_deg, turning_rig


def _sweep(n: int = 18, step_deg: float = 20.0) -> dict[str, Pose]:
    """A full turn, alternating level / up / down views."""
    return {f"f{k:03d}": head_pose(k * step_deg, -12.0 + (0, 14, -14)[k % 3]) for k in range(n)}


def _noisy(poses: dict[str, Pose], fixed: set[str], rot: float, trans: float, seed: int
           ) -> dict[str, Pose]:
    rng = np.random.default_rng(seed)
    return {n: T if n in fixed else perturb(T, rot, trans, rng) for n, T in poses.items()}


def test_rotations_and_centres_are_recovered_from_matches_and_depth() -> None:
    truth = _sweep()
    rig = turning_rig(truth)
    init = _noisy(truth, {"f000"}, rot=4.0, trans=0.12, seed=1)
    rng = np.random.default_rng(2)
    scale = {n: float(rng.uniform(0.85, 1.15)) for n in truth}  # monocular depth scale errors
    fit = refine_poses(rig.pairs, rig.views(init, scale), set(truth) - {"f000"})
    assert set(fit.poses) == set(truth) - {"f000"}
    rot = [rot_err_deg(fit.poses[n], truth[n]) for n in fit.poses]
    ctr = [float(np.linalg.norm(fit.poses[n].t - truth[n].t)) for n in fit.poses]
    assert max(rot) < 0.25 and np.median(rot) < 0.1, rot  # ±15 % depth scale errors
    assert max(ctr) < 0.03, ctr
    assert fit.median_after_deg < 0.1 < fit.median_before_deg
    assert fit.focal_scale == 1.0


def test_side_steps_are_parallax_not_rotation() -> None:
    """Views taken after a sideways step see near surfaces shifted: the depth-aware model puts
    that into the centre; a pure-rotation model would turn the view instead."""
    truth = _sweep(12, 30.0)
    for k, n in enumerate(("s0", "s1", "s2")):
        truth[n] = head_pose(30.0 * k, -12.0, step=np.array([0.0, 0.12, 0.0]))
    rig = turning_rig(truth, seed=3)
    init = _noisy(truth, {"f000"}, rot=2.0, trans=0.08, seed=4)
    free = set(truth) - {"f000"}
    fit = refine_poses(rig.pairs, rig.views(init), free)
    for n in ("s0", "s1", "s2"):
        assert rot_err_deg(fit.poses[n], truth[n]) < 0.1
        assert np.linalg.norm(fit.poses[n].t - truth[n].t) < 0.02
    rot_only = refine_poses(rig.pairs, rig.views(init), free, use_depth=False)
    assert max(rot_err_deg(rot_only.poses[n], truth[n]) for n in ("s0", "s1", "s2")) > 0.5


def test_shared_focal_length_is_recovered_without_degenerating() -> None:
    truth = _sweep()
    rig = turning_rig(truth, seed=5)
    wrong = Intrinsics(K.fx * 0.95, K.fy * 0.95, K.cx, K.cy, K.width, K.height)
    init = _noisy(truth, {"f000"}, rot=2.0, trans=0.05, seed=6)
    fit = refine_poses(rig.pairs, rig.views(init, K_used=wrong), set(truth) - {"f000"},
                       refine_focal=True)
    assert abs(fit.focal_scale * 0.95 - 1.0) < 0.005, fit.focal_scale
    assert max(rot_err_deg(fit.poses[n], truth[n]) for n in fit.poses) < 0.15
    # without the focal length the same data leaves the rotations visibly off
    stuck = refine_poses(rig.pairs, rig.views(init, K_used=wrong), set(truth) - {"f000"})
    assert max(rot_err_deg(stuck.poses[n], truth[n]) for n in stuck.poses) > 0.5


def test_fixed_keyframes_anchor_an_extension() -> None:
    """The map's keyframes hold still and the new ones are placed in their frame, including the
    loop back to where the map started; a keyframe without matches keeps its pose."""
    truth = _sweep()
    rig = turning_rig(truth, seed=7)
    old = {n for n in truth if int(n[1:]) < 9}
    init = _noisy(truth, old, rot=5.0, trans=0.2, seed=8)
    lonely = Pose(truth["f000"].R, np.array([9.0, 9.0, 9.0]))
    views = rig.views(init)
    views["lonely"] = views["f000"].__class__(lonely, K)
    fit = refine_poses(rig.pairs, views, (set(truth) - old) | {"lonely"})
    assert set(fit.poses) == (set(truth) - old) | {"lonely"}
    assert max(rot_err_deg(fit.poses[n], truth[n]) for n in set(truth) - old) < 0.1
    assert max(float(np.linalg.norm(fit.poses[n].t - truth[n].t)) for n in set(truth) - old) \
        < 0.02
    np.testing.assert_allclose(fit.poses["lonely"].matrix(), lonely.matrix())
