"""The segmented image (``segment.sh -f png``, ``segmented.png``, the image of ``view.sh -i``): the
image dimmed, each object's exclusive mask painted opaque in exactly its colour (no blending, no
anti-aliasing), on the reconstruction's grid."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.images import png_bytes
from oh_my_slam.segmentation.colors import color_for_id

DIM = 0.35


def segmented_image(rgb: NDArray[np.uint8], label_map: NDArray[Any]) -> NDArray[np.uint8]:
    out = (rgb.astype(np.float32) * DIM).astype(np.uint8)
    for oid in np.unique(label_map):
        if oid > 0:
            out[label_map == oid] = color_for_id(int(oid))
    return out


def segmented_png(rgb: NDArray[np.uint8], label_map: NDArray[Any]) -> bytes:
    """The segmented image as the PNG every output that carries it shares, byte for byte."""
    return png_bytes(segmented_image(rgb, label_map))
