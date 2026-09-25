"""The colour contract: one sRGB triple per object id, a pure function of the id.

Every object colour must read on the viewer's dark surfaces and on the dimmed ``segmented.png``,
and must never pass for the unsegmented mid-grey (``UNSEGMENTED``, #808080). A colour is
*admissible* when

* its WCAG 2.x contrast against ``DARK_SURFACE`` (#1d2027, the viewer's panel; its canvas
  #15171c is darker) is at least ``MIN_CONTRAST`` = 3:1, the WCAG 1.4.11 minimum for graphical
  objects. That makes it brighter (relative luminance >= 0.143) than every pixel of the dimmed
  photo behind the masks (35 % of white, luminance <= 0.100);
* its OKLab lightness is at most ``MAX_LIGHTNESS`` (no near-white: lines stay visible over white
  walls), its OKLab chroma at least ``MIN_CHROMA`` and its OKLab distance from #808080 at least
  ``MIN_GREY_DISTANCE``.

Ids 1-19 use ``PALETTE_HEX``: Sasha Trubetskoy's distinct colours without the ones that fail
the rules above (navy, maroon, purple, beige, the palest pastels), replaced by admissible ones;
every pair differs by >= 0.095 in OKLab (about five just-noticeable differences; CSS Color 4
takes ΔE_OK 0.02 as one).

Further ids cycle by hue. Id ``19 + n + 1`` aims at the OKLCh hue ``n x GOLDEN_ANGLE_DEG`` and
chooses among admissible 8-bit colours on a fixed grid — the hues within ``HUE_WINDOW_DEG`` of
that aim, on every lightness tier of ``RING_LIGHTNESS`` and chroma level of ``RING_CHROMA`` (each
clipped to the sRGB gamut) — the one farthest from every colour issued before it, with the last
``NEAR_IDS`` ids counted ``NEAR_WEIGHT`` times as close, so that neighbouring ids differ most.
Lightness and chroma therefore change along with the hue instead of repeating the palette's.
Ties go to the smallest hue offset, then to the tier ``n mod len(tiers)``. The table depends
only on the ids (built once, in id order), so every run and every process agrees; a colour that
would repeat an earlier id after 8-bit rounding is nudged to the nearest unused triple.

Also the ``color=height`` ramp of the point-cloud attributes (viridis, low → high).
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
from numpy.typing import NDArray

PALETTE_HEX = (
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#9c4dff", "#42d4f4", "#f032e6",
    "#a8f04a", "#ff9ec7", "#1fb5a3", "#dcbeff", "#9a6324", "#8ef0c0", "#808000", "#ffc49b",
    "#5a9cff", "#b4339c", "#c9a227",
)
UNSEGMENTED = (128, 128, 128)
UNSEGMENTED_HEX = "#808080"

DARK_SURFACE = (0x1D, 0x20, 0x27)
MIN_CONTRAST = 3.0
MAX_LIGHTNESS = 0.93
MIN_CHROMA = 0.08
MIN_GREY_DISTANCE = 0.11

GOLDEN_ANGLE_DEG = 137.50776405003785  # 360 / phi^2
HUE_WINDOW_DEG = 30
RING_LIGHTNESS = (0.62, 0.68, 0.74, 0.80, 0.86, 0.91)
RING_CHROMA = (0.24, 0.15, 0.10)
NEAR_IDS = 10
NEAR_WEIGHT = 1.5

RGB = tuple[int, int, int]


def hex_to_rgb(h: str) -> RGB:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(rgb: RGB) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


# ------------------------------------------------------------------------------------------------
# colour science: sRGB (8-bit) <-> linear light <-> OKLab (Björn Ottosson, 2020)

_LMS = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                 [0.2119034982, 0.6806995451, 0.1073969566],
                 [0.0883024619, 0.2817188376, 0.6299787005]])
_LAB = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                 [1.9779984951, -2.4285922050, 0.4505937099],
                 [0.0259040371, 0.7827717662, -0.8086757660]])
_LAB_INV = np.linalg.inv(_LAB)
_LMS_INV = np.linalg.inv(_LMS)
_LUMA = np.array([0.2126, 0.7152, 0.0722])


def _linear(rgb: NDArray[Any]) -> NDArray[np.float64]:
    u = np.asarray(rgb, np.float64) / 255.0
    return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)


def _encode(lin: NDArray[Any]) -> NDArray[np.int64]:
    x = np.clip(lin, 0.0, 1.0)
    u = np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
    return np.rint(u * 255.0).astype(np.int64)


def oklab(rgb: Any) -> NDArray[np.float64]:
    """OKLab ``(L, a, b)`` of 8-bit sRGB triples, shape ``(..., 3)``."""
    return np.cbrt(_linear(rgb) @ _LMS.T) @ _LAB.T


def delta_e(rgb1: Any, rgb2: Any) -> NDArray[np.float64]:
    """Euclidean OKLab distance (ΔE_OK) between 8-bit sRGB triples."""
    return np.asarray(np.linalg.norm(oklab(rgb1) - oklab(rgb2), axis=-1))


def contrast_ratio(rgb1: Any, rgb2: Any) -> NDArray[np.float64]:
    """WCAG 2.x contrast ratio between 8-bit sRGB triples (1 to 21)."""
    y1, y2 = _linear(rgb1) @ _LUMA, _linear(rgb2) @ _LUMA
    return np.asarray((np.maximum(y1, y2) + 0.05) / (np.minimum(y1, y2) + 0.05))


def admissible(rgb: Any) -> NDArray[np.bool_]:
    """The rules every object colour satisfies (module docstring)."""
    lab = oklab(rgb)
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    return np.asarray((contrast_ratio(rgb, DARK_SURFACE) >= MIN_CONTRAST)
                      & (lab[..., 0] <= MAX_LIGHTNESS) & (chroma >= MIN_CHROMA)
                      & (np.linalg.norm(lab - oklab(UNSEGMENTED), axis=-1) >= MIN_GREY_DISTANCE))


def _oklch_linear(L: NDArray[Any], C: NDArray[Any], h: NDArray[Any]) -> NDArray[np.float64]:
    lab = np.stack([L, C * np.cos(h), C * np.sin(h)], axis=-1)
    return np.asarray(((lab @ _LAB_INV.T) ** 3) @ _LMS_INV.T)


def _gamut_chroma(L: NDArray[Any], h: NDArray[Any]) -> NDArray[np.float64]:
    """Largest OKLCh chroma inside the sRGB gamut at lightness ``L`` and hue ``h`` (radians)."""
    lo, hi = np.zeros_like(L), np.full_like(L, 0.4)
    for _ in range(32):
        mid = (lo + hi) / 2
        lin = _oklch_linear(L, mid, h)
        inside = np.all((lin >= 0.0) & (lin <= 1.0), axis=-1)
        lo, hi = np.where(inside, mid, lo), np.where(inside, hi, mid)
    return lo


# ------------------------------------------------------------------------------------------------
# the id -> colour table

_HUES = 360  # grid resolution: one degree


class _Table:
    """Colours of ids 1..n, grown in id order (so it depends only on the ids)."""

    def __init__(self) -> None:
        self.rgb: list[RGB] = [hex_to_rgb(h) for h in PALETTE_HEX]
        self.used: set[RGB] = {UNSEGMENTED, *self.rgb}
        self.lab: list[NDArray[np.float64]] = [oklab(c) for c in self.rgb]
        self._grid: tuple[NDArray[Any], NDArray[Any], NDArray[Any]] | None = None
        self._dmin: NDArray[np.float64] | None = None

    def grid(self) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Candidate colours ``(rings, 360, 3)`` as 8-bit sRGB and OKLab, and admissibility."""
        if self._grid is None:
            rings = [(L, C) for L in RING_LIGHTNESS for C in RING_CHROMA]
            h = np.radians(np.arange(_HUES, dtype=np.float64))
            rgb = []
            for L, C in rings:
                Ls = np.full(_HUES, L)
                chroma = np.minimum(C, _gamut_chroma(Ls, h) * 0.995)
                rgb.append(_encode(_oklch_linear(Ls, chroma, h)))
            g_rgb = np.stack(rgb)
            self._grid = (g_rgb, oklab(g_rgb), admissible(g_rgb))
            self._dmin = np.full(g_rgb.shape[:2], np.inf)
            for lab in self.lab:
                self._forget(lab)
        return self._grid

    def _forget(self, lab: NDArray[np.float64]) -> None:
        assert self._grid is not None and self._dmin is not None
        self._dmin = np.minimum(self._dmin, np.linalg.norm(self._grid[1] - lab, axis=-1))

    def extend(self, n: int) -> None:
        while len(self.rgb) < n:
            self.rgb.append(self._next())

    def _next(self) -> RGB:
        g_rgb, g_lab, g_ok = self.grid()
        assert self._dmin is not None
        n = len(self.rgb) - len(PALETTE_HEX)  # extension index
        rings = g_rgb.shape[0]
        aim = round(n * GOLDEN_ANGLE_DEG) % _HUES
        offsets = sorted(range(-HUE_WINDOW_DEG, HUE_WINDOW_DEG + 1), key=lambda o: (abs(o), -o))
        ring_order = [(n + k) % rings for k in range(rings)]
        r = np.repeat([ring_order], len(offsets), axis=0).ravel()
        h = np.repeat([(aim + o) % _HUES for o in offsets], rings)
        lab = g_lab[r, h]
        recent = np.array(self.lab[-NEAR_IDS:])
        near = np.linalg.norm(lab[:, None] - recent[None], axis=-1).min(axis=1)
        score = np.where(g_ok[r, h], np.minimum(self._dmin[r, h], near / NEAR_WEIGHT), -1.0)
        k = int(np.argmax(score))  # the first best: smallest hue offset, preferred tier
        red, green, blue = (int(v) for v in g_rgb[r[k], h[k]])
        col = self._unused((red, green, blue))
        self.used.add(col)
        self.lab.append(oklab(col))
        self._forget(self.lab[-1])
        return col

    def _unused(self, col: RGB) -> RGB:
        """``col``, or the nearest 8-bit triple not issued yet (after 6480 grid colours)."""
        step = 0
        cand = col
        while cand in self.used or not bool(admissible(cand)):
            step += 1
            for axis in range(6):
                d = [0, 0, 0]
                d[axis % 3] = step if axis < 3 else -step
                cand = (min(255, max(0, col[0] + d[0])), min(255, max(0, col[1] + d[1])),
                        min(255, max(0, col[2] + d[2])))
                if cand not in self.used and bool(admissible(cand)):
                    break
        return cand


_TABLE = _Table()


@cache
def color_for_id(object_id: int) -> RGB:
    """sRGB triple (0-255) for a positive object id."""
    if object_id < 1:
        raise ValueError(f"object ids start at 1 (got {object_id})")
    _TABLE.extend(object_id)
    return _TABLE.rgb[object_id - 1]


def color_hex_for_id(object_id: int) -> str:
    return rgb_to_hex(color_for_id(object_id))


def segment_colors(labels: NDArray[Any]) -> NDArray[np.uint8]:
    """Per-point object colour for object ids (``UNSEGMENTED`` grey for 0)."""
    lab = np.asarray(labels, dtype=np.int64).reshape(-1)
    ids, inverse = np.unique(lab, return_inverse=True)
    table = np.array([UNSEGMENTED if i <= 0 else color_for_id(int(i)) for i in ids], np.uint8)
    return table.reshape(-1, 3)[inverse.reshape(-1)]


# viridis sampled at 0, 0.1, ..., 1 (matplotlib), linearly interpolated
_VIRIDIS = np.array([hex_to_rgb(h) for h in (
    "#440154", "#482475", "#414487", "#355f8d", "#2a788e", "#21918c", "#22a884", "#44bf70",
    "#7ad151", "#bddf26", "#fde725")], np.float64)
HEIGHT_RANGE_PERCENTILES = (1.0, 99.0)


def height_colors(heights: NDArray[Any]) -> NDArray[np.uint8]:
    """Viridis ramp over the robust range (1st-99th percentile) of ``heights`` (along up)."""
    h = np.asarray(heights, dtype=np.float64).reshape(-1)
    if len(h) == 0:
        return np.zeros((0, 3), np.uint8)
    lo, hi = np.percentile(h, HEIGHT_RANGE_PERCENTILES)
    t = np.clip((h - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.full(len(h), 0.5)
    x = t * (len(_VIRIDIS) - 1)
    i = np.minimum(x.astype(np.int64), len(_VIRIDIS) - 2)
    f = (x - i)[:, None]
    rgb = _VIRIDIS[i] * (1 - f) + _VIRIDIS[i + 1] * f
    return np.rint(rgb).astype(np.uint8)
