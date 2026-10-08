"""Out-of-focus near field: monocular depth of something so close to the lens that the camera
renders it as a blur.

A fixed-focus camera blurs what lies much nearer than the rest of its scene (``examples/camera``'s
p05-p07 frames see a dark ring 0.2-0.3 m from the lens, out of focus, over a third of the image).
The depth network has no edges to go by there: it places the blur anywhere from the lens to the
wall behind, differently in every frame, and keyframes that share it disagree by 10-20 % however
their depth is scaled. ``defocused_near_field`` finds such regions in a keyframe: connected parts
of its near field (closer than ``NEAR_SHARE`` of its median depth) whose outline is out of focus —
at least ``DEFOCUS_SHARE`` of its edges ``DEFOCUS_RATIO`` times wider than the image's typical
edge (the rest of the outline may run along sharp edges of what lies behind). Both tests are
relative to the image, so a sharp rendering (every edge as narrow as the rest) or a blurred video
frame (every edge as wide) has none, and a near counter top in focus stays. On
``examples/camera`` the ring's outlines have 41-77 % such edges, the near field in focus 0-5 %.

Edge width: at an edge pixel (local contrast at least ``EDGE_CONTRAST`` over ``EDGE_WINDOW`` px),
the intensity range across the window over the steepest gradient in it, in pixels — about 1-2 px
for an edge in focus, the width of the blur for one out of focus.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

NEAR_SHARE = 0.5  # the near field: closer than this share of the keyframe's median depth
DEFOCUS_RATIO = 2.0  # an edge out of focus: this many times wider than the image's median edge
DEFOCUS_SHARE = 0.25  # a region out of focus: at least this share of its outline's edges out of focus
EDGE_CONTRAST = 0.12  # an edge: an intensity range (0-1) of at least this over the window
EDGE_WINDOW = 9  # px of the depth grid
BAND = 6  # px: a region's outline band (where its edges are measured), and its blurred rim
MIN_REGION = 0.01  # share of the keyframe's valid pixels: smaller near regions are not judged
MIN_EDGES = 20  # edge pixels a region's outline needs to be judged


def edge_width(rgb: NDArray[np.uint8]) -> NDArray[np.float64]:
    """Width in pixels of the edge at each pixel of an image (nan where there is none)."""
    from scipy import ndimage as ndi

    g = np.asarray(rgb, np.float64).mean(axis=2) / 255.0
    grad = np.hypot(ndi.sobel(g, 1), ndi.sobel(g, 0)) / 8.0
    span = ndi.maximum_filter(g, EDGE_WINDOW) - ndi.minimum_filter(g, EDGE_WINDOW)
    steep = ndi.maximum_filter(grad, EDGE_WINDOW)
    edge = (span >= EDGE_CONTRAST) & (grad >= 0.5 * steep) & (steep > 0)
    out = np.full(g.shape, np.nan)
    out[edge] = span[edge] / steep[edge]
    return out


def defocused_near_field(rgb: NDArray[np.uint8], depth: NDArray[Any], valid: NDArray[Any]
                         ) -> NDArray[np.bool_]:
    """The pixels of a keyframe (image ``rgb`` on its depth grid) in an out-of-focus near region,
    with its blurred rim (module docstring); none when nothing is judged."""
    from scipy import ndimage as ndi

    d = np.asarray(depth, np.float64)
    ok = np.asarray(valid, bool) & np.isfinite(d) & (d > 0)
    out = np.zeros(d.shape, bool)
    if not ok.any():
        return out
    near = ok & (d < NEAR_SHARE * np.median(d[ok]))
    width = edge_width(rgb)
    rest = width[ok & ~near]
    rest = rest[np.isfinite(rest)]
    if len(rest) < MIN_EDGES:
        return out
    typical = float(np.median(rest))
    labels, n = ndi.label(ndi.binary_opening(near, iterations=2))
    for k in range(1, n + 1):
        region = labels == k
        if region.sum() < MIN_REGION * ok.sum():
            continue
        outline = (ndi.binary_dilation(region, iterations=BAND)
                   & ~ndi.binary_erosion(region, iterations=BAND))
        w = width[outline]
        w = w[np.isfinite(w)]
        if len(w) >= MIN_EDGES and float(np.mean(w > DEFOCUS_RATIO * typical)) >= DEFOCUS_SHARE:
            out |= ndi.binary_dilation(region, iterations=BAND)
    return out & ok
