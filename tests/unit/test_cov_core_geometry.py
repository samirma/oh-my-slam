"""``core.geometry`` at its edges: half-turn rotations, mirrored point sets, disabled edge tests,
planes without a prior, and the display budget's voxel-grid search (spec §2.5) on layouts that
reach its limits — grids too fine for one int64 per voxel, points below the finest grid it may
use, points no grid can hold, and the parallel voxel coding of large clouds."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from oh_my_slam.core import geometry as g


def test_a_half_turn_rotation_between_opposite_directions() -> None:
    for a in ([1.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.3, -0.4, 0.5]):
        R = g.rotation_between(a, -np.asarray(a))
        np.testing.assert_allclose(R @ np.asarray(a), -np.asarray(a), atol=1e-12)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)


def test_umeyama_returns_a_rotation_even_for_mirrored_points(rng: np.random.Generator) -> None:
    src = rng.normal(size=(30, 3))
    mirrored = src * [-1.0, 1.0, 1.0]  # a reflection fits best, but it is not a rotation
    sim = g.umeyama(src, mirrored)
    assert np.linalg.det(sim.R) == pytest.approx(1.0)
    np.testing.assert_allclose(sim.R @ sim.R.T, np.eye(3), atol=1e-12)
    rigid = g.umeyama(src, src @ g.rot_z(0.5).T + 1.0, with_scale=False)
    assert rigid.s == 1.0
    np.testing.assert_allclose(rigid.R, g.rot_z(0.5), atol=1e-12)


def test_a_non_positive_edge_threshold_disables_the_edge_test() -> None:
    depth = np.array([[1.0, 1.0, 9.0], [1.0, 0.0, 9.0], [1.0, 1.0, 9.0]])
    assert g.depth_edge_mask(depth, rel_threshold=0.04).any()
    for off in (0.0, -1.0):
        mask = g.depth_edge_mask(depth, rel_threshold=off)
        assert mask.shape == depth.shape and not mask.any()


def test_ransac_plane_without_a_prior(rng: np.random.Generator) -> None:
    plane = np.c_[rng.uniform(-1, 1, (300, 2)), 0.5 + rng.normal(0, 0.002, 300)]
    outliers = rng.uniform(-1, 1, (60, 3))
    res = g.ransac_plane(np.r_[plane, outliers], 0.01, iterations=200, seed=3)
    assert res is not None
    n, d, inliers = res
    assert abs(n[2]) == pytest.approx(1.0, abs=1e-3) and d == pytest.approx(-0.5 * n[2], abs=2e-3)
    assert inliers[:300].mean() > 0.98 and inliers[300:].mean() < 0.2
    assert g.ransac_plane(np.zeros((2, 3)), 0.01) is None
    assert g.ransac_plane(np.zeros((10, 3)), 0.01) is None  # degenerate: no plane


# --- the display budget's voxel-grid search ------------------------------------------------------


def test_grids_too_fine_for_one_code_per_voxel_use_the_keys() -> None:
    """Corners a few kilometres apart around a nanometre cluster: the grid that fits the budget
    has more cells than one int64 can number, so voxels are compared by their (x, y, z) keys —
    with the same selection rules."""
    rng = np.random.default_rng(0)
    corners = np.array(list(itertools.product([-1e6, 1e6], repeat=3)))
    pts = np.r_[corners, rng.uniform(0, 1e-9, (1000, 3))]
    grid, idx = g.budget_voxel_grid(pts, 500)
    assert g._voxel_codes(pts, grid.edge).ndim == 2  # (N, 3) keys
    assert len(idx) == grid.count == g._occupied(pts, grid.edge) <= 500
    assert grid.finer is not None and grid.edge / grid.finer <= 1 + g.BUDGET_EDGE_TOL
    assert set(range(8)) <= set(idx.tolist())  # every corner keeps its own voxel
    keys = g.voxel_keys(pts, grid.edge)
    _, first = np.unique(keys, axis=0, return_index=True)
    np.testing.assert_array_equal(idx, np.sort(first))  # the first point of each voxel
    again, edge = g.budget_voxel_indices(pts, 500, grid.edge)
    assert edge == grid.edge and np.array_equal(again, idx)


def test_points_below_the_finest_grid_share_its_voxel() -> None:
    """No grid is finer than the extent / 2**60: a cluster smaller than that next to a far point
    is one voxel of it, and the selection still fits the budget."""
    rng = np.random.default_rng(1)
    pts = np.r_[rng.uniform(0, 1e-8, (200, 3)), [[1e12, 0.0, 0.0]]]
    grid, idx = g.budget_voxel_grid(pts, 50)
    assert grid.edge == pytest.approx(1e12 / 2**60) and grid.finer is None
    assert grid.count == len(idx) == 2 and idx.tolist() == [0, 200]


def test_points_no_grid_can_hold_in_the_budget_are_an_error() -> None:
    """Grids are anchored at the origin: points in all eight octants occupy at least 8 voxels of
    any of them, so a budget below that cannot be met (and is not silently exceeded)."""
    rng = np.random.default_rng(2)
    octants = np.repeat(np.array(list(itertools.product([-1.0, 1.0], repeat=3))), 3, axis=0)
    pts = octants + rng.normal(0, 0.01, octants.shape)
    with pytest.raises(ValueError, match="no voxel grid holds these points in 4 voxels"):
        g.budget_voxel_grid(pts, 4)
    grid, idx = g.budget_voxel_grid(pts, 8)
    assert grid.count == len(idx) == 8


def test_duplicates_at_a_known_zero_edge_keep_one_point_per_place() -> None:
    pts = np.repeat(np.arange(24, dtype=np.float32).reshape(8, 3), 5, axis=0)
    idx, edge = g.budget_voxel_indices(pts, 10, edge=0.0)
    assert edge == 0.0 and idx.tolist() == [5 * k for k in range(8)]
    idx, _ = g.budget_voxel_indices(pts, 40, edge=0.0)
    assert idx.tolist() == list(range(40))  # within the budget: every point


def test_large_clouds_are_coded_in_parallel_chunks_with_the_same_result(
        monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(4)
    pts = rng.uniform(0, 5, (5_000, 3)).astype(np.float32)
    expected = g._voxel_codes(pts, 0.1)
    grid, idx = g.budget_voxel_grid(pts, 900)
    monkeypatch.setattr(g, "_BUDGET_CHUNK", 700)  # 8 chunks: coded on a thread pool
    np.testing.assert_array_equal(g._voxel_codes(pts, 0.1), expected)
    grid2, idx2 = g.budget_voxel_grid(pts, 900)
    assert grid2 == grid and np.array_equal(idx2, idx)
