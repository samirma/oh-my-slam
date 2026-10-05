"""Spec §2.5 display budget: the voxel-grid selection of a cloud above the budget
(``core.geometry.budget_voxel_indices``) — at most N points, the smallest edge that fits, one
original point per occupied voxel (the first), deterministic, every occupied cell covered."""

from __future__ import annotations

import numpy as np
import pytest

from oh_my_slam.core import geometry as g


def surface_cloud(n: int, seed: int = 3) -> np.ndarray:
    """A floor and two walls (a room corner, 4 x 3 x 2.5 m), unevenly dense, float32 like a map."""
    rng = np.random.default_rng(seed)
    k = n // 3
    floor = np.c_[rng.uniform(0, 4, k) ** 1.3, rng.uniform(0, 3, k), rng.normal(0, 0.002, k)]
    wall = np.c_[rng.uniform(0, 4, k), rng.normal(0, 0.002, k), rng.uniform(0, 2.5, k)]
    side = np.c_[rng.normal(0, 0.002, n - 2 * k), rng.uniform(0, 3, n - 2 * k),
                 rng.uniform(0, 2.5, n - 2 * k) ** 0.8]
    return np.vstack([floor, wall, side]).astype(np.float32)


def keys(points: np.ndarray, edge: float) -> np.ndarray:
    return g.voxel_keys(points, edge)


@pytest.mark.parametrize("budget", [1_000, 25_000, 59_000, 59_990])
def test_at_most_the_budget_with_the_smallest_edge(budget: int) -> None:
    pts = surface_cloud(60_000)
    grid, idx = g.budget_voxel_grid(pts, budget)
    assert grid.edge > 0 and len(idx) == grid.count == g._occupied(pts, grid.edge)
    # never more than the budget, and close to it (a coarse grid's count moves in larger steps)
    assert (0.97 if budget >= 10_000 else 0.9) * budget <= len(idx) <= budget
    # minimal: a finer edge less than 2 % below was counted and does not fit
    assert grid.finer is not None and grid.edge / grid.finer <= 1 + g.BUDGET_EDGE_TOL
    assert grid.finer_count == g._occupied(pts, grid.finer) > budget
    # the edge found once gives the same selection without a search (the per-source cache)
    again, edge = g.budget_voxel_indices(pts, budget, grid.edge)
    assert edge == grid.edge and np.array_equal(again, idx)


def test_the_budget_holds_for_any_layout() -> None:
    """Points packed far below a far outlier's scale: never more than the budget."""
    rng = np.random.default_rng(0)
    tiny = np.r_[rng.uniform(0, 1e-10, (200, 3)), [[100.0, 0.0, 0.0]]]
    grid, idx = g.budget_voxel_grid(tiny, 50)
    assert 0 < len(idx) <= 50 and grid.edge > 0
    assert grid.finer is not None and grid.edge / grid.finer <= 1 + g.BUDGET_EDGE_TOL
    far = np.r_[rng.uniform(0, 1, (1000, 3)), [[1e12, 0.0, 0.0]]]
    grid, idx = g.budget_voxel_grid(far, 100)
    assert 0 < len(idx) <= 100 and 1000 in idx  # the outlier keeps its own voxel
    assert grid.finer is not None and grid.edge / grid.finer <= 1 + g.BUDGET_EDGE_TOL


def test_non_finite_points_are_never_selected() -> None:
    pts = surface_cloud(5_000).astype(np.float64)
    pts[[3, 70]] = np.nan
    pts[99, 1] = np.inf
    grid, idx = g.budget_voxel_grid(pts, 1_000)  # terminates (a NaN extent once looped forever)
    assert 0 < len(idx) <= 1_000 and not {3, 70, 99} & set(idx.tolist())
    assert np.isfinite(pts[idx]).all()
    idx, edge = g.budget_voxel_indices(pts[:1_001], 1_000)  # 998 finite points: all of them
    assert edge == 0.0 and len(idx) == 998 and not {3, 70, 99} & set(idx.tolist())


def test_one_original_point_per_occupied_voxel_the_first() -> None:
    pts = surface_cloud(30_000)
    idx, edge = g.budget_voxel_indices(pts, 8_000)
    assert np.all(np.diff(idx) > 0)  # original rows, ascending, each once
    k_all, k_sel = keys(pts, edge), keys(pts[idx], edge)
    # exactly one selected point in every occupied voxel …
    assert len(np.unique(k_sel, axis=0)) == len(idx) == len(np.unique(k_all, axis=0))
    # … the first one of it in the points' order
    _, first = np.unique(k_all, axis=0, return_index=True)
    np.testing.assert_array_equal(np.sort(first), idx)


def test_every_occupied_coarse_cell_is_covered() -> None:
    pts = surface_cloud(30_000)
    idx, edge = g.budget_voxel_indices(pts, 5_000)
    for factor in (1, 2, 5, 40):
        coarse = edge * factor
        assert {tuple(k) for k in keys(pts[idx], coarse)} == {tuple(k) for k in keys(pts, coarse)}


def test_points_keep_their_own_values() -> None:
    """Selected, never merged: a selected point is bit for bit an input point."""
    pts = surface_cloud(20_000)
    idx, _ = g.budget_voxel_indices(pts, 3_000)
    sel = pts[idx]
    assert sel.dtype == np.float32
    np.testing.assert_array_equal(sel, pts[idx])
    assert {r.tobytes() for r in sel} <= {r.tobytes() for r in pts}


def test_deterministic() -> None:
    pts = surface_cloud(20_000)
    a = g.budget_voxel_indices(pts, 4_000)
    b = g.budget_voxel_indices(pts.copy(), 4_000)
    assert a[1] == b[1]
    np.testing.assert_array_equal(a[0], b[0])


def test_no_thinning_needed() -> None:
    pts = surface_cloud(1_000)
    idx, edge = g.budget_voxel_indices(pts, 1_000)
    assert edge == 0.0 and np.array_equal(idx, np.arange(1_000))
    same = np.repeat([[1.0, 2.0, 3.0]], 50, axis=0)
    idx, edge = g.budget_voxel_indices(same, 10)  # one place: its first point
    assert edge == 0.0 and idx.tolist() == [0]
    dup = np.repeat(pts[:8], 5, axis=0)  # 40 points at 8 places, budget 10: one per place
    idx, edge = g.budget_voxel_indices(dup, 10)
    assert edge == 0.0 and idx.tolist() == [5 * k for k in range(8)]
    idx, edge = g.budget_voxel_indices(np.repeat(pts[:200], 2, axis=0), 50)  # 200 places > 50
    assert edge > 0 and 0 < len(idx) <= 50
    with pytest.raises(ValueError, match="at least 1"):
        g.budget_voxel_indices(pts, 0)
