"""Dense depth-scale fits (``reconstruction.depth.dense_scale``) of a camera turning about a point a
few centimetres behind its lens, with a pole 0.3 m in front of a wall 3 m away that a neighbour does
not see where the keyframe does (parallax, or monocular depth that misses it): the fit must find
the wall's scale, and find it consistent."""

from __future__ import annotations

import numpy as np

from oh_my_slam.reconstruction.depth import _dominant, dense_scale

W, H, F = 160, 120, 100.0
K = np.array([[F, 0.0, W / 2], [0.0, F, H / 2], [0.0, 0.0, 1.0]])


def _pose(yaw_deg: float, centre: tuple[float, float, float]) -> np.ndarray:
    """Camera-to-world: turned ``yaw_deg`` about the vertical (y) axis, at ``centre``."""
    a = np.radians(yaw_deg)
    T = np.eye(4)
    T[:3, :3] = [[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]]
    T[:3, 3] = centre
    return T


def _render(T: np.ndarray, pole: bool = True) -> np.ndarray:
    """z-depth of a wall at world z = 3 and a pole (|x| < 0.08, z = 0.3) in front of it."""
    v, u = np.mgrid[0:H, 0:W] + 0.5
    rays = np.stack([(u - K[0, 2]) / F, (v - K[1, 2]) / F, np.ones_like(u)], -1) @ T[:3, :3].T
    c = T[:3, 3]
    t_wall = (3.0 - c[2]) / rays[..., 2]
    t = t_wall
    if pole:
        t_pole = (0.3 - c[2]) / rays[..., 2]
        x = c[0] + t_pole * rays[..., 0]
        t = np.where((np.abs(x) < 0.08) & (t_pole > 0), t_pole, t_wall)
    cam = (c + t[..., None] * rays - c) @ T[:3, :3]  # in the camera frame
    return cam[..., 2]


def test_the_wall_scale_is_found_past_a_near_pole() -> None:
    A, B = _pose(0.0, (0.0, 0.0, 0.0)), _pose(4.0, (0.03, 0.0, -0.02))
    da, db = _render(A), _render(B, pole=False)
    assert (da < 0.5).mean() > 0.3  # the pole takes a third of the view: ratios of 10 there
    fit = dense_scale(da / 1.3, K, A, [(db, K, B)], step=2)
    assert fit.ok and abs(fit.scale - 1.3) < 0.02 and fit.spread < 0.05


def test_a_dominant_cluster_wins_over_a_smaller_one() -> None:
    r = np.concatenate([np.full(70, 0.0), np.full(30, np.log(8.0))])
    keep = _dominant(r + np.random.default_rng(0).normal(0, 0.01, 100))
    assert keep[:70].all() and not keep[70:].any()


def test_no_depth_or_nothing_seen_gives_no_fit() -> None:
    A = _pose(0.0, (0.0, 0.0, 0.0))
    empty = np.zeros((H, W))
    assert not dense_scale(empty, K, A, [(_render(A), K, A)]).ok
    assert not dense_scale(_render(A), K, A, [(empty, K, A)]).ok
