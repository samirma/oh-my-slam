"""Pair screening of the rotation-dominant refinement (``mapping.panorama``): repetitive texture
gives pairs of keyframes coherent matches that a wrong turn explains; screened against the initial
poses they cannot pull the rotations away, and the refinement's notes count the pairs it could not
explain."""

from __future__ import annotations

import numpy as np
import pytest

from oh_my_slam.core.geometry import rot_z
from oh_my_slam.core.types import Pose
from oh_my_slam.mapping import panorama
from oh_my_slam.mapping.panorama import (
    PairMatches,
    pair_rotation,
    refine_poses,
    refine_turning,
    screen_pairs,
)
from tests.synth.turning import K, head_pose, perturb, rot_err_deg, turning_rig


def _sweep(n: int = 18, step_deg: float = 20.0) -> dict[str, Pose]:
    return {f"f{k:03d}": head_pose(k * step_deg, -12.0 + (0, 14, -14)[k % 3]) for k in range(n)}


def _bearings(uv: np.ndarray) -> np.ndarray:
    x = np.column_stack([(uv[:, 0] - K.cx) / K.fx, (uv[:, 1] - K.cy) / K.fy, np.ones(len(uv))])
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _turned(Ra: np.ndarray, Rb: np.ndarray, rng: np.random.Generator, n: int,
            noise_px: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    """Matches of two views turned by the camera-to-map rotations ``Ra``, ``Rb`` about one centre
    (at most ``n``, keypoints in both images)."""
    uv = np.column_stack([rng.uniform(5, K.width - 5, 4 * n), rng.uniform(5, K.height - 5, 4 * n)])
    x = (_bearings(uv) @ Ra.T) @ Rb
    ok = x[:, 2] > 0.2
    with np.errstate(divide="ignore", invalid="ignore"):
        ub = np.column_stack([K.fx * x[:, 0] / x[:, 2] + K.cx, K.fy * x[:, 1] / x[:, 2] + K.cy])
    ok &= (ub[:, 0] > 2) & (ub[:, 0] < K.width - 2) & (ub[:, 1] > 2) & (ub[:, 1] < K.height - 2)
    idx = np.flatnonzero(ok)[:n]
    return (uv[idx] + rng.normal(scale=noise_px, size=(len(idx), 2)),
            ub[idx] + rng.normal(scale=noise_px, size=(len(idx), 2)))


def _repetitive(truth: dict[str, Pose], rng: np.random.Generator, share: float = 0.5,
                twist_deg: float = 70.0) -> list[PairMatches]:
    """Pairs between the two halves of a sweep whose matches one wrong turn explains coherently:
    the second half seen turned by ``twist_deg`` about the vertical, as a texture repeating across
    a counter top matches places of it that are not the same."""
    names = sorted(truth)
    half = len(names) // 2
    twist = rot_z(np.radians(twist_deg))
    out = []
    for a in names[:half]:
        for b in names[half:]:
            if rng.random() > share:
                continue
            ua, ub = _turned(truth[a].R, twist @ truth[b].R, rng, 200)
            if len(ua) >= 30:
                out.append(PairMatches(a, b, ua, ub))
    return out


@pytest.mark.parametrize("seed", range(4))
def test_a_pairs_rotation_is_its_matches_turn_among_outliers(seed: int) -> None:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=3)
    w *= np.radians(rng.uniform(5.0, 60.0)) / np.linalg.norm(w)
    Ra, Rb = np.eye(3), panorama._exp(w)
    ua, ub = _turned(Ra, Rb, rng, 300)
    bad = rng.random(len(ua)) < 0.3  # a third of the matches anywhere
    ub[bad] = np.column_stack([rng.uniform(0, K.width, bad.sum()),
                               rng.uniform(0, K.height, bad.sum())])
    M = pair_rotation(K, K, ua, ub, np.random.default_rng(seed))
    assert panorama._angle_deg(M @ (Ra.T @ Rb)) < 0.1


@pytest.mark.parametrize("seed", range(3))
def test_repetitive_texture_pairs_are_screened_and_cannot_turn_the_sweep(seed: int) -> None:
    """Coherent wrong pairs between the two halves of a sweep (a fifth of all matches): refined
    with them, the rotations end degrees off; screened against the initial poses, every wrong pair
    and no right one is left out, and the rotations come out as without them."""
    rng = np.random.default_rng(seed)
    truth = _sweep()
    rig = turning_rig(truth, seed=seed)
    wrong = _repetitive(truth, rng)
    pairs = sorted(rig.pairs + wrong, key=lambda p: (p.a, p.b))
    init = {n: T if n == "f000" else perturb(T, 4.0, 0.15, rng) for n, T in truth.items()}
    views = rig.views(init)
    free = set(truth) - {"f000"}
    kept, gate = screen_pairs(pairs, views, free)
    kept_ids = {id(p) for p in kept}
    assert not any(id(p) in kept_ids for p in wrong)
    assert all(id(p) in kept_ids for p in rig.pairs)
    assert gate.dropped == len(wrong) and gate.gate_deg >= panorama.SCREEN_MIN_DEG
    unscreened = refine_poses(pairs, views, free, use_depth=False)
    assert max(rot_err_deg(unscreened.poses[n], truth[n]) for n in free) > 2.0
    fit = refine_turning(pairs, views, free)
    assert max(rot_err_deg(fit.poses[n], truth[n]) for n in free) < 0.25
    assert fit.screened == len(wrong)
    assert fit.summary()["screened_pairs"] == len(wrong)
    assert "screened_pairs" not in unscreened.summary()
    # the notes tell a refinement that gave pairs up from one that explains them all
    assert unscreened.summary()["unexplained_pairs"] > 0 == fit.summary()["unexplained_pairs"]


def test_no_pair_to_screen_keeps_the_smallest_gate() -> None:
    truth = _sweep(4)
    rig = turning_rig(truth)
    kept, gate = screen_pairs(rig.pairs, rig.views(), set())
    assert kept == [] and gate == panorama.ScreenGate(0, panorama.SCREEN_MIN_DEG)
    # an earlier screening's gate is the smallest: refined poses keep the pairs it kept
    kept, gate = screen_pairs(rig.pairs, rig.views(), set(truth) - {"f000"}, min_gate_deg=30.0)
    assert len(kept) == len(rig.pairs) and gate.gate_deg == 30.0


def test_a_gate_follows_the_initial_poses_error() -> None:
    """The gate is a multiple of the median disagreement: initial poses ten degrees off keep
    every right pair."""
    truth = _sweep()
    rig = turning_rig(truth, seed=4)
    rng = np.random.default_rng(5)
    init = {n: T if n == "f000" else perturb(T, 10.0, 0.0, rng) for n, T in truth.items()}
    kept, gate = screen_pairs(rig.pairs, rig.views(init), set(truth) - {"f000"})
    assert len(kept) == len(rig.pairs) and gate.dropped == 0
    assert gate.gate_deg > 3 * panorama.SCREEN_MIN_DEG


def test_per_pair_residuals_are_each_pairs_median() -> None:
    truth = _sweep(6)
    rig = turning_rig(truth, seed=3)
    fit = refine_poses(rig.pairs, rig.views(), set(truth) - {"f000"}, scales=())
    assert set(fit.per_pair_px) == {(p.a, p.b) for p in rig.pairs}
    assert max(fit.per_pair_px.values()) < 1.5  # true poses: the keypoint noise


def test_a_pair_whose_matches_no_turn_explains_keeps_its_best_hypothesis() -> None:
    """Two matches 10° apart in one view and 40° apart in the other: no turn explains both, the
    refit has no consensus of two, and the best hypothesis (one that explains one match) stays."""
    ua = np.array([[K.cx, K.cy], [K.cx + K.fx * np.tan(np.radians(10.0)), K.cy]])
    ub = np.array([[K.cx, K.cy], [K.cx + K.fx * np.tan(np.radians(40.0)), K.cy]])
    M = pair_rotation(K, K, ua, ub, np.random.default_rng(0))
    np.testing.assert_allclose(M @ M.T, np.eye(3), atol=1e-12)
    a, b = _bearings(ua), _bearings(ub)
    agree = np.degrees(np.arccos(np.clip(np.sum((a @ M.T) * b, axis=1), -1.0, 1.0)))
    assert np.sort(agree)[0] < 1e-3 and np.sort(agree)[1] > panorama.FIT_TOL_DEG
