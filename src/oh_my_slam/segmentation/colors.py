"""The colour contract: one sRGB triple per object id, a pure function of the id.

19 perceptually distinct colours (Sasha Trubetskoy's list without grey, white and black) for ids
1-19; every further cycle ``k = (id - 1) // 19`` rotates the hue of the same 19 colours by
``k * 0.381966`` of a turn in HLS space (lightness and saturation unchanged, so no colour is
ever grey). Unsegmented points are ``UNSEGMENTED`` (#808080), never an object colour.
"""

from __future__ import annotations

import colorsys
from functools import cache

PALETTE_HEX = (
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6",
    "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075",
)
HUE_STEP = 0.381966  # 1 - 1/golden ratio, in turns
UNSEGMENTED = (128, 128, 128)
UNSEGMENTED_HEX = "#808080"

RGB = tuple[int, int, int]


def hex_to_rgb(h: str) -> RGB:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(rgb: RGB) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


_BASE = tuple(hex_to_rgb(h) for h in PALETTE_HEX)


def _rotated(base: RGB, turns: float) -> RGB:
    r, g, b = base
    h, lum, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    rr, gg, bb = colorsys.hls_to_rgb((h + turns) % 1.0, lum, s)
    return (round(rr * 255), round(gg * 255), round(bb * 255))


_TABLE: list[RGB] = []
_USED: set[RGB] = {UNSEGMENTED}


def _extend(n: int) -> None:
    """Grow the id -> colour table deterministically up to id ``n``.

    Hue rotation can make two palette entries coincide after 8-bit rounding (e.g. maroon and
    olive are 60 degrees apart). A colliding colour is nudged by 0.002-turn steps until unused,
    so every id has a unique colour; the table depends only on the ids, never on run order.
    """
    while len(_TABLE) < n:
        object_id = len(_TABLE) + 1
        k, i = divmod(object_id - 1, len(_BASE))
        col = _BASE[i] if k == 0 else _rotated(_BASE[i], k * HUE_STEP)
        step = 0
        while col in _USED:
            step += 1
            col = _rotated(_BASE[i], k * HUE_STEP + 0.002 * step)
        _TABLE.append(col)
        _USED.add(col)


@cache
def color_for_id(object_id: int) -> RGB:
    """sRGB triple (0-255) for a positive object id."""
    if object_id < 1:
        raise ValueError(f"object ids start at 1 (got {object_id})")
    _extend(object_id)
    return _TABLE[object_id - 1]


def color_hex_for_id(object_id: int) -> str:
    return rgb_to_hex(color_for_id(object_id))
