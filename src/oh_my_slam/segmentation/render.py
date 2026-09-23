"""``segmented.png``: the image dimmed, each object's exclusive mask painted opaque in exactly its
colour (no blending, no anti-aliasing). For maps, a contact sheet of <= 6 keyframes chosen by set
cover so that every object appears at least once (where possible)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from oh_my_slam.segmentation.colors import color_for_id

DIM = 0.35
HEADER_PX = 28
HEADER_COLOR = (24, 24, 24)
TEXT_COLOR = (235, 235, 235)
TILE_WIDTH = 640
MAX_TILES = 6


def segmented_image(rgb: NDArray[np.uint8], label_map: NDArray[Any], dim: float = DIM
                    ) -> NDArray[np.uint8]:
    out = (rgb.astype(np.float32) * dim).astype(np.uint8)
    for oid in np.unique(label_map):
        if oid > 0:
            out[label_map == oid] = color_for_id(int(oid))
    return out


def choose_contact_frames(frame_ids: list[set[int]], max_tiles: int = MAX_TILES) -> list[int]:
    """Greedy set cover of all object ids by frames; ties → earlier frame."""
    remaining = set().union(*frame_ids) if frame_ids else set()
    chosen: list[int] = []
    while remaining and len(chosen) < max_tiles:
        best = max(range(len(frame_ids)), key=lambda i: (len(frame_ids[i] & remaining), -i))
        gain = frame_ids[best] & remaining
        if not gain:
            break
        chosen.append(best)
        remaining -= gain
    return sorted(chosen)


def _nearest_resize(a: NDArray[Any], width: int) -> NDArray[Any]:
    h, w = a.shape[:2]
    height = max(1, round(h * width / w))
    ys = (np.arange(height) * h / height).astype(int)
    xs = (np.arange(width) * w / width).astype(int)
    return a[ys][:, xs]


def contact_sheet(tiles: list[tuple[str, NDArray[np.uint8], NDArray[Any]]],
                  tile_width: int = TILE_WIDTH) -> NDArray[np.uint8]:
    """Grid (up to 3 columns) of segmented keyframes with a header strip naming each frame.

    Tiles are resized with nearest-neighbour sampling so mask colours stay exact.
    """
    if not tiles:
        return np.zeros((HEADER_PX, tile_width, 3), np.uint8)
    rendered = []
    for name, rgb, labels in tiles:
        seg = segmented_image(rgb, labels)
        seg = _nearest_resize(seg, tile_width)
        header = Image.new("RGB", (tile_width, HEADER_PX), HEADER_COLOR)
        ImageDraw.Draw(header).text((8, 6), name, fill=TEXT_COLOR, font=ImageFont.load_default())
        rendered.append(np.vstack([np.asarray(header), seg]))
    cols = min(3, len(rendered))
    rows = int(np.ceil(len(rendered) / cols))
    th = max(t.shape[0] for t in rendered)
    sheet = np.full((rows * th, cols * tile_width, 3), HEADER_COLOR, np.uint8)
    for i, t in enumerate(rendered):
        r, c = divmod(i, cols)
        sheet[r * th: r * th + t.shape[0], c * tile_width: (c + 1) * tile_width] = t
    return sheet
