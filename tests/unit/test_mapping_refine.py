"""Multi-view pose refinement from feature matches and monocular depth (``mapping.panorama``) on
an analytic head turning in place: rotations to a small fraction of a degree, camera centres to
about a centimetre, parallax of side steps modelled, focal length recovered, fixed keyframes held."""

from __future__ import annotations

import numpy as np
import pytest

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
    assert fit.distortion == 0.0  # a pinhole's: what little the fit finds moves no corner
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


def test_normal_equations_match_a_per_match_sum() -> None:
    """Σ w JᵀJ and Σ w Jᵀr over the free views' parameters equal the plain per-match sum, with
    fixed views left out and segments of any length (one match, many, views repeated)."""
    from oh_my_slam.mapping.panorama import _normal_equations, _Problem

    rng = np.random.default_rng(3)
    seg_views = [(0, 1), (1, 0), (1, 2), (2, 3), (3, 1), (0, 3)]
    lengths = [1, 40, 7, 300, 2, 55]
    starts = np.cumsum([0, *lengths[:-1]])
    n = sum(lengths)
    src = np.concatenate([[a] * k for (a, _), k in zip(seg_views, lengths, strict=True)])
    dst = np.concatenate([[b] * k for (_, b), k in zip(seg_views, lengths, strict=True)])
    pb = _Problem(src, dst, np.zeros((n, 2)), np.zeros((n, 2)), np.ones((n, 2)), np.ones(n),
                  starts.astype(np.int64), seg_views)
    J, r, c = rng.normal(size=(n, 2, 14)), rng.normal(size=(n, 2)), 1.7
    free = {1: 0, 3: 1}  # views 0 and 2 are held fixed
    npar = 6 * len(free) + 2
    H, g = _normal_equations(pb, r, J, c, free, npar)
    Hr, gr = np.zeros((npar, npar)), np.zeros(npar)
    w = 1.0 / (1.0 + np.sum(r * r, axis=1) / (c * c))
    for i in range(n):
        gi, li = [], []
        for view, off in ((src[i], 0), (dst[i], 6)):
            if view in free:
                gi += range(6 * free[view], 6 * free[view] + 6)
                li += range(off, off + 6)
        gi += [npar - 2, npar - 1]  # the shared camera: focal length and distortion
        li += [12, 13]
        Hr[np.ix_(gi, gi)] += w[i] * J[i][:, li].T @ J[i][:, li]
        gr[gi] += w[i] * J[i][:, li].T @ r[i]
    np.testing.assert_allclose(H, Hr, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(g, gr, rtol=1e-10, atol=1e-10)


def _lens(uv: np.ndarray, k: float) -> np.ndarray:
    """Where a division-model lens (coefficient ``k`` at ``K``'s focal length) shows the keypoints
    its pinhole sees at ``uv``."""
    u = (uv - [K.cx, K.cy]) / [K.fx, K.fy]
    d = 2 * u / (1 + np.sqrt(1 - 4 * k * np.sum(u * u, axis=1, keepdims=True)))
    return np.asarray(d * [K.fx, K.fy] + [K.cx, K.cy])


def _lens_depth(depth: np.ndarray, k: float) -> np.ndarray:
    """The depth grid (``K``'s size) as the lens shows it: each pixel takes the depth its
    pinhole sees where the lens bends its ray."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    d = (np.column_stack([u.ravel() + 0.5, v.ravel() + 0.5]) - [K.cx, K.cy]) / [K.fx, K.fy]
    p = d / (1 + k * np.sum(d * d, axis=1, keepdims=True)) * [K.fx, K.fy] + [K.cx, K.cy]
    pu, pv = np.floor(p[:, 0]).astype(int), np.floor(p[:, 1]).astype(int)
    inside = (pu >= 0) & (pu < w) & (pv >= 0) & (pv < h)
    out = np.zeros(h * w, np.float32)
    out[inside] = depth[pv[inside], pu[inside]]
    return out.reshape(h, w)


def test_a_wide_angle_lens_is_refined_with_the_focal_length() -> None:
    """Keypoints of a barrel-distorted lens (a corner 60 px in): refining the shared camera
    recovers its distortion and focal length, and the rotations (the keyframes' depth as the
    image shows it, before the map knows the lens)."""
    from oh_my_slam.mapping.panorama import PairMatches, View

    k = -0.15
    truth = _sweep()
    rig = turning_rig(truth, seed=9)
    pairs = [PairMatches(p.a, p.b, _lens(p.uv_a, k), _lens(p.uv_b, k)) for p in rig.pairs]
    wrong = Intrinsics(K.fx * 0.95, K.fy * 0.95, K.cx, K.cy, K.width, K.height)
    init = _noisy(truth, {"f000"}, rot=2.0, trans=0.05, seed=10)
    views = {n: View(init[n], wrong, _lens_depth(rig.depth[n], k), K, (K.width, K.height))
             for n in truth}
    fit = refine_poses(pairs, views, set(truth) - {"f000"}, refine_focal=True)
    assert abs(fit.focal_scale * 0.95 - 1.0) < 0.01, fit.focal_scale
    assert abs(fit.distortion - k) < 0.01, fit.distortion
    assert max(rot_err_deg(fit.poses[n], truth[n]) for n in fit.poses) < 0.15
    assert fit.summary()["distortion"] == round(fit.distortion, 6)
    # the same keypoints where the pinhole sees them, the depth on the pinhole's grid (a map whose
    # lens is known): nothing left to refine but the poses
    flat = {n: View(init[n], K, rig.depth[n], K, (K.width, K.height)) for n in truth}
    held = refine_poses(rig.pairs, flat, set(truth) - {"f000"})
    assert held.distortion == 0.0 and held.focal_scale == 1.0
    assert max(rot_err_deg(held.poses[n], truth[n]) for n in held.poses) < 0.15
    # a known lens is held: the focal length alone is refined
    from oh_my_slam.mapping.panorama import refine_turning

    known = refine_turning(rig.pairs, flat, set(truth) - {"f000"}, refine_focal=True,
                           hold_distortion=True)
    assert known.distortion == 0.0 and abs(known.focal_scale - 1.0) < 0.01


def test_the_jacobian_is_the_residuals_derivative() -> None:
    """Finite differences of the residuals in every parameter, with a lens (``k`` < 0), for a
    pinhole whose coefficient is refined (``k`` = 0), and with the coefficient held (its column
    0); points with and without depth."""
    from oh_my_slam.mapping.panorama import _evaluate, _exp, _Problem

    rng = np.random.default_rng(11)
    n = 60
    pb = _Problem(np.zeros(n, np.int64), np.ones(n, np.int64), rng.uniform(-0.6, 0.6, (n, 2)),
                  rng.uniform(-300, 300, (n, 2)), np.tile([900.0, 900.0], (n, 1)),
                  np.where(rng.random(n) < 0.3, np.inf, rng.uniform(1, 4, n)),
                  np.array([0]), [(0, 1)])
    R = np.stack([_exp(np.array([0.05, -0.1, 0.02])), _exp(np.array([-0.02, 0.15, 0.01]))])
    C = np.array([[0.0, 0.0, 0.0], [0.03, -0.01, 0.02]])
    for kappa, lens in ((-0.25, True), (0.0, True), (0.0, False)):
        r, J = _evaluate(pb, R, C, 0.07, kappa, jac=True, lens=lens)
        assert J is not None
        eps = 1e-6
        for k in range(14 if lens else 13):
            Rp, Cp, ph, ka = R.copy(), C.copy(), 0.07, kappa
            view, j = divmod(k, 6) if k < 12 else (0, 0)
            if k < 12 and j < 3:
                Rp[view] = _exp(np.eye(3)[j] * eps) @ R[view]
            elif k < 12:
                Cp[view, j - 3] += eps
            elif k == 12:
                ph += eps
            else:
                ka += eps
            rp, _ = _evaluate(pb, Rp, Cp, ph, ka, jac=False)
            np.testing.assert_allclose(J[:, :, k], (rp - r) / eps, rtol=1e-4, atol=1e-3)
        if not lens:
            assert not J[:, :, 13].any()


def test_a_focal_length_that_runs_off_its_prior_is_held(monkeypatch: pytest.MonkeyPatch
                                                         ) -> None:
    """A shared focal length that leaves its prior by more than ``FOCAL_MAX_FACTOR`` (here a
    5 % change against a 1 % limit) is a wrong camera: the refinement starts again with it held,
    the distortion alone refined."""
    from oh_my_slam.mapping import panorama

    monkeypatch.setattr(panorama, "FOCAL_MAX_FACTOR", 1.01)
    truth = _sweep()
    rig = turning_rig(truth, seed=5)
    wrong = Intrinsics(K.fx * 0.95, K.fy * 0.95, K.cx, K.cy, K.width, K.height)
    init = _noisy(truth, {"f000"}, rot=2.0, trans=0.05, seed=6)
    fit = refine_poses(rig.pairs, rig.views(init, K_used=wrong), set(truth) - {"f000"},
                       refine_focal=True)
    assert fit.focal_scale == 1.0


def test_depth_on_an_undistorted_grid_is_read_where_that_image_shows_the_keypoints() -> None:
    """A keyframe of a known lens has its depth on its undistorted image's grid
    (``Intrinsics.pinhole``: a shorter focal length than the lens's), its matches where the
    lens's own pinhole sees them: the depth is read where the undistorted image shows them."""
    from oh_my_slam.mapping.panorama import View, _depth_at, _usable_depth

    lens = Intrinsics(K.fx, K.fy, K.cx, K.cy, K.width, K.height, "colmap", -0.15)
    grid = lens.pinhole().resized(K.width // 4, K.height // 4)
    rows, cols = np.mgrid[0:grid.height, 0:grid.width]
    depth = (2.0 + 0.001 * rows + 0.0001 * cols).astype(np.float32)
    uv = np.array([[K.cx + 200.0, K.cy - 100.0], [K.cx - 250.0, K.cy + 60.0]])
    at = (uv - [K.cx, K.cy]) * lens.undistorted_scale + [K.cx, K.cy]
    g = np.rint(at / 4 - 0.5).astype(int)
    view = View(Pose.identity(), lens, depth, grid, (K.width, K.height), lens)
    np.testing.assert_array_equal(_depth_at(view, _usable_depth(view), uv), depth[g[:, 1], g[:, 0]])
    # the camera fitting its lens again (its keypoints the image's own): read where the
    # undistorted image shows them
    raw = lens.image_pixels(at)
    refit = View(Pose.identity(), K, depth, grid, (K.width, K.height), lens)
    g2 = np.rint(at / 4 - 0.5).astype(int)
    np.testing.assert_array_equal(_depth_at(refit, _usable_depth(refit), raw),
                                  depth[g2[:, 1], g2[:, 0]])
    plain = View(Pose.identity(), lens, depth, grid, (K.width, K.height))  # the image's own grid
    g = np.rint(uv / 4 - 0.5).astype(int)
    np.testing.assert_array_equal(_depth_at(plain, _usable_depth(plain), uv), depth[g[:, 1], g[:, 0]])


def _up_tilt_deg(T: Pose, up_cam: np.ndarray) -> float:
    up = T.R @ up_cam
    return float(np.degrees(np.arccos(np.clip(up[2] / np.linalg.norm(up), -1.0, 1.0))))


def test_gravity_levels_what_the_matches_leave_free() -> None:
    """A keyframe the matches do not hold (as a block hanging on the map by a few weak matches
    turns about them) is levelled with its gravity estimate, its heading kept; keyframes the
    matches hold are placed as without it."""
    from oh_my_slam.mapping.panorama import _exp

    truth = _sweep()
    rig = turning_rig(truth, seed=9)
    old = {n for n in truth if int(n[1:]) < 9}
    init = _noisy(truth, old, rot=3.0, trans=0.1, seed=10)
    true_lonely = truth["f010"]
    tilted = Pose(true_lonely.R @ _exp(np.radians([8.0, 0.0, 0.0])), true_lonely.t)
    views = rig.views(init)
    views["lonely"] = views["f000"].__class__(tilted, K)
    free = (set(truth) - old) | {"lonely"}
    gravity = {n: (truth[n].R.T @ np.array([0.0, 0.0, 1.0]), 2.0) for n in set(truth) - old}
    gravity["lonely"] = (true_lonely.R.T @ np.array([0.0, 0.0, 1.0]), 2.0)
    fit = refine_poses(rig.pairs, views, free, gravity=gravity)
    assert _up_tilt_deg(tilted, gravity["lonely"][0]) > 7.9
    assert _up_tilt_deg(fit.poses["lonely"], gravity["lonely"][0]) < 0.05
    heading = [np.arctan2(*T.R[:2, 2][::-1]) for T in (true_lonely, fit.poses["lonely"])]
    assert abs(np.degrees(heading[0] - heading[1])) < 0.5
    assert max(rot_err_deg(fit.poses[n], truth[n]) for n in set(truth) - old) < 0.1
    # without the prior it keeps its tilt
    plain = refine_poses(rig.pairs, views, free)
    assert _up_tilt_deg(plain.poses["lonely"], gravity["lonely"][0]) > 7.9


def test_the_tilt_jacobian_is_its_derivative() -> None:
    from oh_my_slam.mapping.panorama import _exp, _tilt

    rng = np.random.default_rng(11)
    up = rng.normal(size=3)
    up /= np.linalg.norm(up)
    e, J = _tilt(up)
    d = 1e-6 * rng.normal(size=3)
    np.testing.assert_allclose(_tilt(_exp(d) @ up)[0] - e, J @ d, atol=1e-11)
