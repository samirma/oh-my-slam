"""Lifting a detection of a see-through object (``segmentation.lift``): a chair back's mask covers
the wall seen between its spindles, and the wall is its largest cluster. The object is the nearer
layer that encloses what lies behind it (``object_layer``); a mask that holds only what is seen
through its object is marked (``Lifted.through``). Randomised over image size, focal length, the
object's distance and size and the gap to the wall behind it."""

from __future__ import annotations

import numpy as np
import pytest

from oh_my_slam.core.types import Intrinsics
from oh_my_slam.segmentation.lift import (
    MIN_POINTS,
    _inside_hull,
    lift_mask,
    object_layer,
    seen_through,
)


def camera(rng: np.random.Generator) -> Intrinsics:
    """A random pinhole: 320-1280 px wide, 4:3, 45-100° across."""
    w = int(rng.choice([320, 480, 640, 768, 1024, 1280]))
    h = w * 3 // 4
    f = (w / 2) / np.tan(np.radians(rng.uniform(45.0, 100.0)) / 2)
    return Intrinsics(f, f, w / 2, h / 2, w, h)


def lattice(rng: np.random.Generator, K: Intrinsics, spindle_px: float, frame_px: float
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """A wall with a see-through object in front of it, at the image centre: a rectangular frame
    ``frame_px`` wide with vertical spindles ``spindle_px`` wide. Returns (depth, silhouette
    mask, interior mask, the object's distance, the wall's)."""
    z0 = rng.uniform(1.0, 5.0)
    # Keep the lattice resolved and fully inside the image at every random focal length.
    size = rng.uniform(0.4, 0.45) * K.width * z0 / K.fx
    wall = z0 + rng.uniform(1.5 * size, 1.5 * size + 3.0)  # behind it by more than its size
    depth = np.full((K.height, K.width), wall, np.float32)
    wpx, hpx = size * K.fx / z0, 1.5 * size * K.fy / z0
    c0, r0 = int(K.cx - wpx / 2), int(K.cy - hpx / 2)
    c1, r1 = int(K.cx + wpx / 2), int(K.cy + hpx / 2)
    max_bar = max(1, int(wpx / 24))
    bar = min(max_bar, max(1, round(spindle_px)))
    rim = max(1, round(frame_px)) if spindle_px <= 2 else min(max_bar, max(1, round(frame_px)))
    sil = np.zeros(depth.shape, bool)
    sil[r0:r1, c0:c1] = True
    inner = np.zeros_like(sil)
    inner[r0 + rim:r1 - rim, c0 + rim:c1 - rim] = True
    obj = sil & ~inner  # the frame
    gap = max(3 * bar, (c1 - c0) // 7)
    for c in range(c0 + rim + gap, c1 - rim - bar, gap):  # the spindles
        obj[r0:r1, c:c + bar] = True
    depth[obj] = z0
    return depth, sil, inner, z0, wall


@pytest.mark.parametrize("seed", range(12))
def test_a_see_through_object_is_lifted_at_its_own_depth_not_the_wall_behind(seed: int) -> None:
    rng = np.random.default_rng(seed)
    K = camera(rng)
    # spindles wide enough that some of each survives the depth edges (3 x 3 window)
    width = rng.uniform(5.0, 9.0)
    depth, sil, _, z0, wall = lattice(rng, K, width, width)
    lifted = lift_mask(sil, depth, K)
    z = lifted.points[:, 2]
    assert len(z) >= MIN_POINTS
    # the largest cluster is the wall (most of the silhouette), the object its nearer layer
    assert (depth[sil] == wall).mean() > 0.5
    assert np.median(z) == pytest.approx(z0, rel=0.02)
    assert not lifted.through


@pytest.mark.parametrize("seed", range(12))
def test_a_mask_that_holds_only_what_is_seen_through_its_object_is_marked(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    K = camera(rng)
    # spindles a pixel or two wide: depth edges, nothing of them is left to lift; the mask holds
    # the inside of the frame
    depth, _, inner, _, wall = lattice(rng, K, rng.uniform(1.0, 2.0), rng.uniform(10.0, 20.0))
    lifted = lift_mask(inner, depth, K)
    assert lifted.through and np.median(lifted.points[:, 2]) == pytest.approx(wall, rel=0.01)


@pytest.mark.parametrize("seed", range(6))
def test_what_is_not_seen_through_keeps_the_largest_cluster(seed: int) -> None:
    rng = np.random.default_rng(200 + seed)
    K = camera(rng)
    z0 = rng.uniform(1.0, 5.0)
    w = rng.uniform(0.4, 1.0)
    depth = np.full((K.height, K.width), z0 + 3.0, np.float32)
    px = int(w * K.fx / z0)
    r, c = int(K.cy), int(K.cx)
    # an object's own far side, within its size behind its front: the larger far slab stays
    near = np.zeros(depth.shape, bool)
    near[r - px // 2:r + px // 2, c - px // 2:c - px // 4] = True
    far = np.zeros(depth.shape, bool)
    far[r - px // 2:r + px // 2, c - px // 4:c + px // 2] = True
    depth[near], depth[far] = z0, z0 + 0.5 * w
    lifted = lift_mask(near | far, depth, K)
    assert np.median(lifted.points[:, 2]) == pytest.approx(z0 + 0.5 * w, rel=0.02)
    assert not lifted.through
    # a surface beside a nearer one, not enclosed by it (a sill below a window, the far part of
    # a rug behind a chair leg): the larger, farther one stays
    depth = np.full((K.height, K.width), z0 + 3.0, np.float32)
    top = np.zeros(depth.shape, bool)
    top[r - px:r, c - px // 2:c + px // 2] = True
    sill = np.zeros(depth.shape, bool)
    sill[r + 2:r + 2 + px // 4, c - px // 2:c + px // 2] = True
    depth[top], depth[sill] = z0 + 2.0 * w, z0
    lifted = lift_mask(top | sill, depth, K)
    assert np.median(lifted.points[:, 2]) == pytest.approx(z0 + 2.0 * w, rel=0.02)


def test_an_object_behind_a_nearer_one_below_it_is_not_seen_through() -> None:
    """A chair behind a table: the table is nearer below the chair's back only, not all round."""
    K = Intrinsics(500.0, 500.0, 320.0, 240.0, 640, 480)
    depth = np.full((480, 640), 5.0, np.float32)
    back = np.zeros((480, 640), bool)
    back[150:260, 280:360] = True
    depth[back] = 3.4
    depth[260:330, 200:440] = 2.8  # the table top in front of the chair's lower half
    lifted = lift_mask(back, depth, K)
    assert not lifted.through and np.median(lifted.points[:, 2]) == pytest.approx(3.4)


def test_the_edge_cases_of_the_hull_and_the_ring() -> None:
    line = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    assert not _inside_hull(line, np.array([[1.0, 0.0]])).any()  # spans no area
    assert not _inside_hull(line[:2], np.array([[0.5, 0.0]])).any()  # too few points
    square = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [1.0, 1.0]])
    assert _inside_hull(square, np.array([[1.0, 1.5], [3.0, 1.0]])).tolist() == [True, False]
    # nearer clusters that span no area enclose nothing: the largest cluster stays
    pts = np.array([[x, 0.0, 2.0] for x in range(30)] + [[x, y, 5.0] for x in range(10)
                                                         for y in range(10)], np.float64)
    uv = np.array([[x, 50.0] for x in range(30)] + [[x + 5, y + 40] for x in range(10)
                                                   for y in range(10)], np.float64)
    labels = np.r_[np.zeros(30, np.int64), np.ones(100, np.int64)]
    assert object_layer(pts, uv, labels, 0.1)[30:].all()
    # no lifted pixels, or no ring around the mask (it fills the image): nothing seen through
    mask = np.ones((20, 20), bool)
    d = np.full((20, 20), 2.0)
    assert not seen_through(mask, d, mask, np.zeros(0, np.int64), 0.1)
    assert not seen_through(mask, d, mask, np.arange(10), 0.1)
    # a ring nearer on most of it but on too few pixels to enclose anything
    mask = np.zeros((40, 40), bool)
    mask[10:30, 10:30] = True
    d = np.full((40, 40), 5.0)
    ok = np.zeros((40, 40), bool)
    ok[5, 5:30] = True  # the only comparable ring pixels: one nearer row
    d[5, 5:30] = 1.0
    pixels = np.flatnonzero(mask.reshape(-1))
    assert not seen_through(mask, d, ok | mask, pixels, 0.1)
