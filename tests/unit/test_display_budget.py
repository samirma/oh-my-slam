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


@pytest.mark.parametrize("budget", [1_000, 25_000, 59_000])
def test_at_most_the_budget_with_the_smallest_edge(budget: int) -> None:
    pts = surface_cloud(60_000)
    idx, edge = g.budget_voxel_indices(pts, budget)
    assert edge > 0 and 0 < len(idx) <= budget
    assert g._occupied(pts, edge) == len(idx)
    # minimal to within the tolerance: some edge less than 1 % smaller does not fit (the count is
    # not strictly monotonic at that scale: the grid's alignment moves it slightly), and a clearly
    # finer grid never does
    tol = g.BUDGET_EDGE_REL_TOL
    assert any(g._occupied(pts, edge / (1 + tol) ** (k / 10)) > budget for k in range(1, 11))
    assert g._occupied(pts, edge / 1.1) > budget
    assert g.budget_voxel_edge(pts, budget) == edge


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
    with pytest.raises(ValueError, match="at least 1"):
        g.budget_voxel_indices(pts, 0)
