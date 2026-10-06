"""The depth image of ``reconstruct.sh -f depth`` (spec §2.2): the model's own metric depth along
the optical axis as one 16-bit single-channel PNG with the pixel size of the input.

* A pixel holds the depth in units of 1/``DEPTH_UNITS_PER_METRE`` m (256 per metre, steps of about
  3.9 mm): depth in metres = value / 256, rounded to the nearest unit. ``NO_DEPTH`` (0) is reserved
  for the pixels where the model gives no valid depth (outside its validity mask, or a depth that
  is not finite and positive). A valid depth that would round to 0 is stored as 1, and one beyond
  65535 / 256 ≈ 256 m as 65535.
* The depth is the reconstruction's grid (long side at most ``MAX_GRID_SIDE`` px, the grid of the
  scene description and the point cloud) resampled to the input's upright pixel size (EXIF
  orientation applied, as everywhere) by nearest neighbour: each pixel takes the value of the grid
  pixel whose footprint holds its centre. No depth is interpolated across an edge or into a hole,
  and the reserved value stays exact. Depth along the optical axis does not change with the image
  scale.
* No point-cloud attribute applies: no flying-pixel filter, depth range or stride.
* A ``tEXt`` chunk (``Description``) states the scale and the reserved value, so the file says how
  to read it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.constants import DEPTH_UNITS_PER_METRE, NO_DEPTH
from oh_my_slam.core.images import png_bytes

if TYPE_CHECKING:
    from oh_my_slam.reconstruction.api import FrameReconstruction

MAX_VALUE = np.iinfo(np.uint16).max
DESCRIPTION = (f"oh-my-slam depth along the optical axis: metres = value / {DEPTH_UNITS_PER_METRE}; "
               f"{NO_DEPTH} = no valid depth")


def depth_values(depth: NDArray[Any], valid: NDArray[Any]) -> NDArray[np.uint16]:
    """The 16-bit value of each pixel of a metric depth grid (``valid``: the model's mask)."""
    d = np.asarray(depth, np.float64)
    ok = np.asarray(valid, bool) & np.isfinite(d) & (d > 0)
    units = np.rint(np.where(ok, d, 0.0) * DEPTH_UNITS_PER_METRE)
    out = np.clip(units, 1, MAX_VALUE).astype(np.uint16)
    out[~ok] = NO_DEPTH
    return out


def resample_nearest(a: NDArray[Any], width: int, height: int) -> NDArray[Any]:
    """``a`` (H, W) at ``width`` x ``height`` by nearest neighbour: each output pixel takes the
    input pixel whose footprint holds its centre."""
    h, w = a.shape[:2]
    rows = np.minimum(((np.arange(height) + 0.5) * h / height).astype(np.int64), h - 1)
    cols = np.minimum(((np.arange(width) + 0.5) * w / width).astype(np.int64), w - 1)
    return a[rows[:, None], cols[None, :]]


def depth_image(frame: FrameReconstruction) -> NDArray[np.uint16]:
    """The depth image of a reconstructed frame, (height, width) of the input (upright)."""
    return resample_nearest(depth_values(frame.depth, frame.valid), frame.intrinsics.width,
                            frame.intrinsics.height)


def depth_png(frame: FrameReconstruction) -> bytes:
    """``reconstruct.sh -f depth``'s result: the depth image as a 16-bit greyscale PNG."""
    return png_bytes(depth_image(frame), text={"Description": DESCRIPTION})
