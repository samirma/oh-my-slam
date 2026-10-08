"""Each keyframe's depth correction also bends smoothly over its image
(``reconstruction.depth.DepthCorrection.field``): monocular depth that places one side of an image
deeper than the other, differently in every keyframe, is brought back to its neighbours', which
one scale and tilt per keyframe cannot do."""

from __future__ import annotations

import numpy as np
import pytest

from oh_my_slam.reconstruction.depth import (
    FIELD_MEAN,
    FIELD_NODES,
    BinRatio,
    DepthCorrection,
    DepthView,
    adjust_depth_corrections,
    field_weights,
    pair_bins,
)
from tests.unit.test_depth_scales import KG, N, loop_poses, overlapping, solve, worst_pair

NODES = FIELD_NODES[0] * FIELD_NODES[1]


def _bent(depth: np.ndarray, side: float) -> np.ndarray:
    """``depth`` placed ``side`` (log) deeper at the image's right edge than at its left."""
    x = (np.arange(depth.shape[1]) + 0.5) / depth.shape[1]
    return depth * np.exp(side * (x - 0.5))[None, :]


def test_a_bend_over_each_image_is_removed() -> None:
    from tests.synth.turning import depth_grid

    poses = loop_poses()
    true = [depth_grid(T, KG).astype(np.float64) for T in poses]
    side = np.random.default_rng(5).uniform(-0.15, 0.15, N)
    side[0] = 0.0
    views = [DepthView(_bent(d, s), KG.K(), T.matrix())
             for d, s, T in zip(true, side, poses, strict=True)]
    pairs = overlapping(poses)
    corr = solve(views, pairs, {0})
    flat = [DepthCorrection(c.log_scale, c.slope, c.pivot) for c in corr]
    before = worst_pair(views, [DepthCorrection() for _ in views], pairs)
    assert before > 0.05
    assert worst_pair(views, corr, pairs) < 0.6 * min(before, worst_pair(views, flat, pairs))
    # and the keyframes stay as near their true depth as they were (within 0.2 %): the pairs see
    # only the differences between keyframes, so part of each bend stays (the prior holds it
    # back) and the rest of the loop takes up the difference as scale
    errs = []
    for d, v, c in zip(true, views, corr, strict=True):
        ok = d > 0
        errs.append([float(np.median(np.abs(np.log(x[ok] / d[ok])))) for x in (v.depth,
                                                                           c.apply(v.depth))])
        assert c.bend < 0.2
    e = np.array(errs)
    assert e[:, 1].mean() <= e[:, 0].mean() + 0.002 and e[:, 1].max() < 0.06


def test_field_weights_are_bilinear_between_the_nodes() -> None:
    w = field_weights(np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [2.0, -1.0]]))
    assert w.shape == (4, NODES)
    np.testing.assert_allclose(w.sum(axis=1), 1.0)
    assert w[0, 0] == 1.0 and w[1, NODES - 1] == 1.0
    assert w[3, FIELD_NODES[0] - 1] == 1.0  # beyond the image: its nearest edge
    np.testing.assert_allclose(FIELD_MEAN.sum(), 1.0)


def test_a_field_of_one_node_along_an_axis_is_constant_along_it() -> None:
    w = field_weights(np.array([[0.25, 0.0], [0.25, 0.9]]), (3, 1))
    np.testing.assert_allclose(w, [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])


def test_a_field_applies_alike_on_a_grid_and_at_places() -> None:
    field = tuple(np.linspace(-0.1, 0.1, NODES))
    c = DepthCorrection(0.05, 0.1, float(np.log(2.0)), field)
    depth = np.full((24, 32), 2.0)
    depth[0, 0] = 0.0
    grid = c.apply(depth)
    v, u = np.mgrid[0:24, 0:32]
    at = np.column_stack([(u.ravel() + 0.5) / 32, (v.ravel() + 0.5) / 24])
    expected = np.where(depth > 0, grid / np.where(depth > 0, depth, 1.0), 1.0)
    np.testing.assert_allclose(c.factor(depth.ravel(), at), expected.ravel(), rtol=1e-9)
    assert grid[0, 0] == 0.0 and grid[0, 31] > grid[0, 1]
    assert c.bend == pytest.approx(np.expm1(0.1)) and not c.identity
    assert DepthCorrection(field=(0.0,) * NODES).identity and DepthCorrection().bend == 0.0
    with pytest.raises(ValueError, match="needs the depths' places"):
        c.factor(np.array([2.0]))


def test_fields_compose() -> None:
    f1 = tuple(np.linspace(-0.05, 0.05, NODES))
    f2 = tuple(np.linspace(0.04, -0.02, NODES))
    a = DepthCorrection(0.02, -0.1, float(np.log(1.5)), f1)
    b = DepthCorrection(-0.01, 0.05, float(np.log(1.4)), f2)
    depth = np.random.default_rng(0).uniform(0.5, 4.0, (12, 16))
    np.testing.assert_allclose(a.then(b).apply(depth), b.apply(a.apply(depth)), rtol=1e-9)
    plain = DepthCorrection(0.02, 0.0, 0.0)
    assert plain.then(DepthCorrection()).field == ()  # no field either side: none
    np.testing.assert_allclose(plain.then(b).field, f2)


def test_bins_carry_their_places_and_the_solver_bends_held_keyframes_too() -> None:
    """Pairs of a held keyframe (its scale holds) with a free one: both bend; the field's mean
    over the image stays 0 (the scale is the log scale's alone)."""
    from tests.synth.turning import depth_grid

    poses = loop_poses()[:3]
    true = [depth_grid(T, KG).astype(np.float64) for T in poses]
    views = [DepthView(_bent(d, s), KG.K(), T.matrix())
             for d, s, T in zip(true, (0.1, -0.1, 0.05), poses, strict=True)]
    bins = [b for i, j in ((0, 1), (1, 2)) for b in pair_bins(i, j, views[i], views[j])]
    assert bins and all(len(b.field_src) == len(b.field_dst) == NODES for b in bins)
    assert len({b.field_src for b in bins}) > len({(b.src, b.dst) for b in bins})  # places vary
    piv = np.array([v.log_median() for v in views])
    adj = adjust_depth_corrections(3, bins, piv, set(), scale_fixed={0})
    held, free = adj.corrections[0], adj.corrections[1]
    assert held.log_scale == 0.0 and held.bend > 0.01 and free.bend > 0.01
    for c in adj.corrections:
        assert abs(float(FIELD_MEAN @ np.asarray(c.field))) < 0.01
    # the keyframes do not bend alike: their fields sum to 0, node by node
    total = np.sum([np.asarray(c.field) for c in adj.corrections], axis=0)
    assert np.abs(total).max() < 0.01 < max(c.bend for c in adj.corrections)
    # bins without places (another source of ratios): scale and tilt only
    plain = [BinRatio(b.src, b.dst, b.log_ratio, b.log_src, b.log_dst, b.weight) for b in bins]
    assert all(c.field == () for c in adjust_depth_corrections(3, plain, piv, {0}).corrections)


def test_a_keyframe_without_pairs_keeps_its_depth() -> None:
    """Its field is no part of the fields held to sum to 0, so it takes none; and without any
    bins every correction is the identity."""
    from tests.synth.turning import depth_grid

    poses = loop_poses()[:2]
    views = [DepthView(_bent(depth_grid(T, KG).astype(np.float64), s), KG.K(), T.matrix())
             for T, s in zip(poses, (0.1, -0.1), strict=True)]
    bins = pair_bins(0, 1, views[0], views[1])
    piv = np.array([v.log_median() for v in views] + [0.7])
    adj = adjust_depth_corrections(3, bins, piv, {0})
    assert adj.corrections[2].identity and not adj.corrections[1].identity
    assert all(c.identity for c in adjust_depth_corrections(3, [], piv, set()).corrections)
