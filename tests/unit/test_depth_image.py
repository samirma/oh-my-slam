"""The depth image of ``reconstruct.sh -f depth`` (spec §2.2, ``reconstruction.depthimage``): one
16-bit single-channel PNG of the input's pixel size, the metric depth along the optical axis in
1/256 m, 0 where the model gives no valid depth, resampled from the depth grid by nearest
neighbour (no depth interpolated across an edge), with no point-cloud attribute applied."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from oh_my_slam.core.constants import DEPTH_UNITS_PER_METRE, NO_DEPTH
from oh_my_slam.core.images import png_bytes
from oh_my_slam.core.types import Intrinsics
from oh_my_slam.reconstruction.api import FrameReconstruction
from oh_my_slam.reconstruction.depthimage import (
    DESCRIPTION,
    MAX_VALUE,
    depth_image,
    depth_png,
    depth_values,
    resample_nearest,
)


def test_the_scale_and_the_reserved_value() -> None:
    assert (DEPTH_UNITS_PER_METRE, NO_DEPTH, MAX_VALUE) == (256, 0, 65535)
    depth = np.array([[1.0, 2.5, 0.0, np.nan],
                      [-1.0, np.inf, 1e-4, 300.0],
                      [0.5 / 256, 1.5 / 256, 3.0, 255.996]], np.float32)
    valid = np.ones(depth.shape, bool)
    valid[2, 2] = False  # outside the model's mask: no valid depth, whatever the value
    v = depth_values(depth, valid)
    assert v.dtype == np.uint16
    assert v.tolist() == [[256, 640, 0, 0],
                          [0, 0, 1, 65535],  # a valid depth never takes the reserved value
                          [1, 2, 0, 65535]]  # rounded to the nearest unit, saturated above
    np.testing.assert_allclose(v[0, :2] / DEPTH_UNITS_PER_METRE, depth[0, :2])


def test_nearest_neighbour_keeps_every_value_exact() -> None:
    a = np.array([[1, 2, 3], [4, 5, 6]], np.uint16)
    assert resample_nearest(a, 3, 2).tolist() == a.tolist()
    up = resample_nearest(a, 7, 5)
    assert up.shape == (5, 7) and up.dtype == np.uint16
    assert up.tolist() == [[1, 1, 2, 2, 2, 3, 3],
                           [1, 1, 2, 2, 2, 3, 3],
                           [4, 4, 5, 5, 5, 6, 6],  # each pixel takes the grid pixel holding its
                           [4, 4, 5, 5, 5, 6, 6],  # centre: nothing in between is made up
                           [4, 4, 5, 5, 5, 6, 6]]
    assert set(np.unique(up)) <= set(np.unique(a))
    assert resample_nearest(np.arange(16).reshape(4, 4), 2, 2).tolist() == [[5, 7], [13, 15]]


def frame(depth: np.ndarray, valid: np.ndarray, width: int, height: int) -> FrameReconstruction:
    h, w = depth.shape
    return FrameReconstruction(
        image_path=Path("x.jpg"), rgb=np.zeros((h, w, 3), np.uint8),
        depth=depth.astype(np.float32), valid=valid, K_grid=Intrinsics(50, 50, w / 2, h / 2, w, h),
        intrinsics=Intrinsics(100, 100, width / 2, height / 2, width, height))


def test_the_image_has_the_inputs_pixel_size() -> None:
    depth = np.array([[1.0, 2.0], [0.0, 4.0]])
    f = frame(depth, depth > 0, width=5, height=3)  # the input is larger than the depth grid
    img = depth_image(f)
    assert img.shape == (3, 5) and img.dtype == np.uint16
    assert img.tolist() == [[256, 256, 512, 512, 512],
                            [0, 0, 1024, 1024, 1024],
                            [0, 0, 1024, 1024, 1024]]


def test_the_png_is_16_bit_greyscale_and_says_how_to_read_it() -> None:
    depth = np.array([[1.0, 2.0, 3.0], [0.0, 4.0, 5.0]])
    data = depth_png(frame(depth, depth > 0, width=6, height=4))
    assert data[12:16] == b"IHDR" and data[24:26] == bytes([16, 0])  # bit depth 16, greyscale
    with Image.open(io.BytesIO(data)) as im:
        assert im.size == (6, 4) and im.mode.startswith("I;16")
        assert im.info == {"Description": DESCRIPTION}
        values = np.asarray(im)
    assert values.tolist() == [[256, 256, 512, 512, 768, 768],
                               [256, 256, 512, 512, 768, 768],
                               [0, 0, 1024, 1024, 1280, 1280],
                               [0, 0, 1024, 1024, 1280, 1280]]
    assert "metres = value / 256" in DESCRIPTION and "0 = no valid depth" in DESCRIPTION
    assert depth_png(frame(depth, depth > 0, width=6, height=4)) == data  # deterministic


def test_png_bytes_text_chunks_are_optional() -> None:
    rgb = np.zeros((2, 3, 3), np.uint8)
    plain = png_bytes(rgb)
    assert b"tEXt" not in plain and png_bytes(rgb, {}) == plain
    with Image.open(io.BytesIO(png_bytes(rgb, {"Comment": "hi"}))) as im:
        assert im.info == {"Comment": "hi"} and im.mode == "RGB"
