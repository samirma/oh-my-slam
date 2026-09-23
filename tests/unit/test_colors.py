from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from oh_my_slam.segmentation import colors as c


def test_first_cycle_is_the_fixed_palette() -> None:
    assert c.color_hex_for_id(1) == "#e6194b"
    assert c.color_hex_for_id(19) == "#000075"
    assert [c.color_hex_for_id(i) for i in range(1, 20)] == list(c.PALETTE_HEX)
    assert len(c.PALETTE_HEX) == 19


def test_ids_1_to_1000_unique_and_never_grey() -> None:
    cols = [c.color_for_id(i) for i in range(1, 1001)]
    assert len(set(cols)) == 1000
    assert c.UNSEGMENTED not in cols
    assert all(0 <= v <= 255 for col in cols for v in col)


def test_hue_cycling_keeps_lightness_and_saturation() -> None:
    import colorsys

    for i in (1, 7, 19):
        a = c.color_for_id(i)
        b = c.color_for_id(i + 19)
        ha, la, sa = colorsys.rgb_to_hls(*(v / 255 for v in a))
        hb, lb, sb = colorsys.rgb_to_hls(*(v / 255 for v in b))
        assert abs(la - lb) < 0.02 and abs(sa - sb) < 0.05
        dh = (hb - ha) % 1.0
        assert min(abs(dh - c.HUE_STEP), abs(dh - c.HUE_STEP - 1)) < 0.03 or i == 19


def test_deterministic_across_processes() -> None:
    code = ("from oh_my_slam.segmentation.colors import color_hex_for_id as f;"
            "print(','.join(f(i) for i in (1, 20, 57, 500, 999)))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ",".join(c.color_hex_for_id(i) for i in (1, 20, 57, 500, 999))


def test_hex_helpers_and_invalid_id() -> None:
    assert c.hex_to_rgb("#0a0B0c") == (10, 11, 12)
    assert c.rgb_to_hex((10, 11, 12)) == "#0a0b0c"
    with pytest.raises(ValueError):
        c.color_for_id(0)
    assert np.array(c.UNSEGMENTED).tolist() == [128, 128, 128] and c.UNSEGMENTED_HEX == "#808080"
