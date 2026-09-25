from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from oh_my_slam.segmentation import colors as c
from oh_my_slam.segmentation.render import DIM

# One just-noticeable difference in OKLab (ΔE_OK), as CSS Color 4's gamut mapping takes it.
JND = 0.02
VIEWER_CANVAS = (0x15, 0x17, 0x1C)


def colours(n: int) -> np.ndarray:
    return np.array([c.color_for_id(i) for i in range(1, n + 1)])


def pairwise(rgb: np.ndarray) -> np.ndarray:
    lab = c.oklab(rgb)
    d = np.linalg.norm(lab[:, None] - lab[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    return d


def test_first_cycle_is_the_fixed_palette() -> None:
    assert c.color_hex_for_id(1) == "#e6194b"
    assert [c.color_hex_for_id(i) for i in range(1, 20)] == list(c.PALETTE_HEX)
    assert len(c.PALETTE_HEX) == 19
    assert pairwise(colours(19)).min() >= 0.09  # ~5 JND between any two palette colours


def test_ids_1_to_1000_unique_admissible_and_never_grey() -> None:
    cols = colours(1000)
    assert len({tuple(x) for x in cols}) == 1000
    assert not (cols == c.UNSEGMENTED).all(axis=1).any()
    assert c.admissible(cols).all()
    assert c.delta_e(cols, c.UNSEGMENTED).min() >= c.MIN_GREY_DISTANCE


def test_visible_on_the_viewer_and_on_the_dimmed_image() -> None:
    """WCAG 1.4.11 contrast (3:1) on the viewer's canvas and panel; brighter than every pixel of
    the dimmed photo of segmented.png (whose brightest pixel is 35 % of white)."""
    cols = colours(300)
    assert c.contrast_ratio(cols, VIEWER_CANVAS).min() >= 3.0
    assert c.contrast_ratio(cols, c.DARK_SURFACE).min() >= 3.0
    dimmed_white = np.full(3, int(255 * DIM))
    lum = c._linear(cols) @ c._LUMA
    assert lum.min() > float(c._linear(dimmed_white) @ c._LUMA)
    assert cols.max(axis=1).min() > int(255 * DIM)  # the evaluator's mask test holds
    for navy in ((0, 0, 117), (128, 0, 0)):  # the old palette's near-invisible entries
        assert not c.admissible(navy)


def test_min_pairwise_delta_e_for_ids_1_to_120() -> None:
    """Any two of the first 120 objects differ by >= 0.045 ΔE_OK (> 2 JND; the old hue-rotated
    palette had pairs at 0.009, below one JND). 0.045 is about 80 % of what farthest-point
    sampling of the same admissible gamut reaches for 120 colours (0.058), the price of keeping
    the hue cycle; neighbouring ids, which are often the same kind of object, differ far more."""
    d = pairwise(colours(120))
    assert d.min() >= 0.045
    idx = np.arange(120)
    gap = np.abs(idx[:, None] - idx[None])
    assert d[gap == 1].min() >= 0.15  # consecutive ids: 7 JND
    assert d[gap <= 10].min() >= 0.07  # ids at most 10 apart: 3.5 JND


def test_later_ids_cycle_by_hue_and_vary_lightness() -> None:
    cols = colours(19 + 200)[19:]
    lab = c.oklab(cols)
    hue = np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) % 360
    aim = np.round(np.arange(200) * c.GOLDEN_ANGLE_DEG) % 360
    dev = np.abs((hue - aim + 180) % 360 - 180)
    assert dev.max() <= c.HUE_WINDOW_DEG + 1.0  # + 8-bit rounding
    # lightness and chroma change along with the hue: consecutive ids do not share a tier
    light = lab[:, 0]
    assert np.mean(np.abs(np.diff(light)) > 0.03) > 0.5
    assert light.max() - light.min() > 0.25


def test_deterministic_across_processes() -> None:
    ids = (1, 20, 57, 120, 500, 999)
    code = ("from oh_my_slam.segmentation.colors import color_hex_for_id as f;"
            f"print(','.join(f(i) for i in {ids!r}))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ",".join(c.color_hex_for_id(i) for i in ids)
    # the table depends only on the ids, not on the order they are asked for
    code = ("from oh_my_slam.segmentation.colors import color_hex_for_id as f;"
            f"print(','.join(f(i) for i in {tuple(reversed(ids))!r}))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ",".join(c.color_hex_for_id(i) for i in reversed(ids))


def test_hex_helpers_and_invalid_id() -> None:
    assert c.hex_to_rgb("#0a0B0c") == (10, 11, 12)
    assert c.rgb_to_hex((10, 11, 12)) == "#0a0b0c"
    with pytest.raises(ValueError):
        c.color_for_id(0)
    assert np.array(c.UNSEGMENTED).tolist() == [128, 128, 128] and c.UNSEGMENTED_HEX == "#808080"
    assert c.oklab((255, 255, 255))[0] == pytest.approx(1.0, abs=1e-4)
    assert c.contrast_ratio((0, 0, 0), (255, 255, 255)) == pytest.approx(21.0)
