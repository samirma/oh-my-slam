"""Faster implementations that must give exactly what the ones they replace gave."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from oh_my_slam.core.geometry import depth_edge_mask, unique_rows


def _reference_edges(depth: np.ndarray, rel_threshold: float = 0.04, size: int = 3
                     ) -> np.ndarray:
    """``depth_edge_mask`` as written with ndimage's filters."""
    d = np.asarray(depth, dtype=np.float64)
    valid = np.isfinite(d) & (d > 0)
    big = np.where(valid, d, np.nan)
    dmax = ndimage.maximum_filter(np.where(valid, d, -np.inf), size=size, mode="nearest")
    dmin = ndimage.minimum_filter(np.where(valid, d, np.inf), size=size, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = (dmax - dmin) / big
    touching = ndimage.binary_dilation(~valid, structure=np.ones((size, size), bool))
    return np.asarray(((rel > rel_threshold) | touching) & valid)


@pytest.mark.parametrize("size", [3, 4, 5])
@pytest.mark.parametrize("shape", [(48, 64), (7, 3), (1, 9)])
def test_depth_edge_mask_is_the_ndimage_one(rng: np.random.Generator, size: int,
                                            shape: tuple[int, int]) -> None:
    d = rng.uniform(0.5, 4.0, shape) * (1 + 0.1 * rng.normal(size=shape))
    d[rng.random(shape) < 0.1] = 0.0
    d[rng.random(shape) < 0.03] = np.nan
    d[rng.random(shape) < 0.02] = np.inf
    d[:, : shape[1] // 3] *= 1.2  # a depth step
    for thr in (0.04, 0.1):
        np.testing.assert_array_equal(depth_edge_mask(d, thr, size), _reference_edges(d, thr, size))
    np.testing.assert_array_equal(depth_edge_mask(d.astype(np.float32)),
                                  _reference_edges(d.astype(np.float32)))


def test_unique_rows_is_numpys(rng: np.random.Generator) -> None:
    for keys in (rng.integers(-50, 50, (5000, 3)), rng.integers(-3, 3, (200, 3)),
                 np.array([[2**40, -(2**40), 0], [0, 0, 0], [2**40, -(2**40), 0]]),  # wide span
                 np.zeros((1, 3), np.int64)):
        uniq, inv = unique_rows(keys)
        ref_u, ref_i = np.unique(keys, axis=0, return_inverse=True)
        np.testing.assert_array_equal(uniq, ref_u)
        np.testing.assert_array_equal(inv, ref_i.reshape(-1))


def test_overlap_fraction_looks_up_only_what_can_hit(rng: np.random.Generator) -> None:
    from scipy.spatial import cKDTree

    from oh_my_slam.mapping.objects import overlap_fraction

    def reference(a: np.ndarray, b: np.ndarray, radius: float) -> float:
        sa = a if len(a) <= 4000 else a[np.linspace(0, len(a) - 1, 4000).astype(int)]
        d, _ = cKDTree(b).query(sa, k=1, distance_upper_bound=radius)
        return float(np.isfinite(d).mean())

    hits = 0
    for _ in range(100):
        a, b = ((rng.normal(size=(int(rng.integers(1, 9000)), 3)) * rng.uniform(0.05, 1)
                 + rng.normal(size=3)).astype(np.float32) for _ in range(2))
        r = float(rng.uniform(0.01, 0.3))
        got = overlap_fraction(a, b, r, cKDTree(b))
        assert got == reference(a, b, r) == overlap_fraction(a, b, r)
        hits += got > 0
    assert 20 < hits < 100  # overlapping and apart pairs both tested


def test_fusion_depth_cut_uses_the_median_of_the_valid_depth(rng: np.random.Generator) -> None:
    from oh_my_slam.mapping.geometry import FrameData, fusion_depth_max

    class Rec:
        def __init__(self) -> None:
            self.stats = {"depth_exponent": 1.1}

    depth = rng.uniform(0.5, 6.0, (40, 50)).astype(np.float32)
    depth[rng.random(depth.shape) < 0.2] = 0.0
    valid = rng.random(depth.shape) < 0.8
    fd = FrameData(Rec(), depth, valid, np.zeros((40, 50, 3), np.uint8),  # type: ignore[arg-type]
                   np.zeros((40, 50), np.int32), True)
    m = float(np.median(depth[valid & (depth > 0)]))
    assert fd.median_depth == m
    assert fusion_depth_max(fd, 7.5) == float(np.clip(m * (7.5 / m) ** 1.1, 7.5 / 1.5, 7.5 * 1.5))
