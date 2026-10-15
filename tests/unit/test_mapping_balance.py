"""Depth-balanced pose refinement (``mapping.panorama.depth_balance``): every depth stratum of a
pair direction carries the same weight, so the few far matches that carry a moving head's
parallax are not outvoted by the many on one near surface; on consistent matches it finds the
same poses as the plain refinement."""

from __future__ import annotations

import numpy as np
import pytest

from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.mapping import panorama
from oh_my_slam.mapping.panorama import PairMatches, View, depth_balance, refine_poses

K = Intrinsics(400.0, 400.0, 320.0, 130.0, 640, 260)


def _look(yaw_deg: float, centre: np.ndarray, pitch_deg: float = 50.0) -> Pose:
    """Camera-to-map pose looking ``pitch_deg`` down towards ``yaw_deg`` (z up, OpenCV axes)."""
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    fwd = np.array([np.cos(y) * np.cos(p), np.sin(y) * np.cos(p), -np.sin(p)])
    right = np.array([np.sin(y), -np.cos(y), 0.0])
    return Pose(np.stack([right, np.cross(fwd, right), fwd], 1), centre)


def _scene_depth(T: Pose) -> np.ndarray:
    """z-depth on K's grid: a counter top 0.8 m below the camera out to 1.2 m from it, beyond it a
    floor 1.9 m below."""
    v, u = np.mgrid[0:K.height, 0:K.width]
    rays = np.stack([(u + 0.5 - K.cx) / K.fx, (v + 0.5 - K.cy) / K.fy, np.ones(u.shape)], -1)
    world = rays @ T.R.T
    z_counter = -0.8 / np.minimum(world[..., 2], -1e-6)
    hit = T.t + z_counter[..., None] * world
    on_counter = np.linalg.norm(hit[..., :2] - T.t[:2], axis=-1) < 1.2
    z_floor = -1.9 / np.minimum(world[..., 2], -1e-6)
    return np.where(on_counter, z_counter, z_floor).astype(np.float32)


def _pair(rng: np.random.Generator, Ta: Pose, Tb: Pose, depth: np.ndarray, n: int = 1500
          ) -> PairMatches:
    uv = np.column_stack([rng.uniform(1, K.width - 1, n), rng.uniform(1, K.height - 1, n)])
    z = depth[uv[:, 1].astype(int), uv[:, 0].astype(int)]
    X = np.column_stack([(uv[:, 0] - K.cx) / K.fx * z, (uv[:, 1] - K.cy) / K.fy * z, z])
    Y = (X @ Ta.R.T + Ta.t - Tb.t) @ Tb.R
    ub = Y[:, :2] / Y[:, 2:3] * [K.fx, K.fy] + [K.cx, K.cy]
    ok = (Y[:, 2] > 0.1) & (ub[:, 0] > 1) & (ub[:, 0] < K.width - 1) & (ub[:, 1] > 1) \
        & (ub[:, 1] < K.height - 1)
    return PairMatches("a", "b", uv[ok] + rng.normal(scale=0.3, size=(ok.sum(), 2)),
                       ub[ok] + rng.normal(scale=0.3, size=(ok.sum(), 2)))


@pytest.mark.parametrize("seed", range(3))
def test_a_balanced_refinement_of_consistent_matches_finds_the_same_poses(seed: int) -> None:
    """A head swinging 4.5 cm sideways over a counter top, the matches crowding on the counter and
    the depth right: balanced or not, the refinement finds the translation and the turn."""
    rng = np.random.default_rng(seed)
    Ta = _look(0.0, np.array([0.13, 0.0, 0.0]))
    yaw = 20.0
    Tb = _look(yaw, 0.13 * np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw)), 0.0]))
    pair = _pair(rng, Ta, Tb, _scene_depth(Ta))
    start = Pose(Tb.R, Ta.t.copy())  # the turn known, the centre not
    views = {"a": View(Ta, K, _scene_depth(Ta), K, (K.width, K.height)),
             "b": View(start, K, _scene_depth(Tb), K, (K.width, K.height))}
    truth = float(np.linalg.norm(Tb.t - Ta.t))
    for balance in (False, True):
        fit = refine_poses([pair], views, {"b"}, balance_depth=balance)
        step = float(np.linalg.norm(fit.poses["b"].t - Ta.t))
        assert abs(step - truth) < 0.03 * truth
        assert float(np.linalg.norm(fit.poses["b"].t - Tb.t)) < 0.003


def test_depth_balance_gives_every_stratum_of_a_direction_the_same_weight() -> None:
    """90 matches at 1 m and 10 at 3 m share a direction's weight equally (its total kept);
    matches without depth keep 1; a direction of one stratum is unchanged; the boost is capped."""
    depth = np.r_[np.full(90, 1.0), np.full(10, 3.0), np.full(5, np.inf), np.full(40, 2.0),
                  np.full(399, 1.0), np.full(1, 4.0)]
    starts = np.array([0, 105, 145])
    n = len(depth)
    pb = panorama._Problem(np.zeros(n, np.int64), np.ones(n, np.int64), np.zeros((n, 2)),
                           np.zeros((n, 2)), np.ones((n, 2)), depth, starts,
                           [(0, 1), (1, 0), (0, 1)])
    w = depth_balance(pb)
    assert w[:90].sum() == pytest.approx(50.0) and w[90:100].sum() == pytest.approx(50.0)
    np.testing.assert_array_equal(w[100:105], 1.0)
    np.testing.assert_allclose(w[105:145], 1.0)
    assert w[145:544] == pytest.approx(400 / (2 * 399))
    assert w[-1] == panorama.BALANCE_MAX  # 400 / 2 = 200, capped
    empty = panorama._Problem(np.zeros(2, np.int64), np.ones(2, np.int64), np.zeros((2, 2)),
                              np.zeros((2, 2)), np.ones((2, 2)), np.full(2, np.inf),
                              np.array([0]), [(0, 1)])
    np.testing.assert_array_equal(depth_balance(empty), 1.0)
