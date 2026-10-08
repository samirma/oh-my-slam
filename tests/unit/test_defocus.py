"""An out-of-focus near field has no depth (``reconstruction.defocus``): a near region whose outline
is blurred, relative to the image's own edges, is found; a near region in focus, a uniformly
blurred image and an image without edges are not."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from oh_my_slam.reconstruction.defocus import defocused_near_field, edge_width

H, W = 120, 200


def _scene(blur: float, image_blur: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A textured wall 2 m away (sharp vertical bars) behind a dark band 0.25 m from the lens over
    the left third, its outline blurred by ``blur`` px (the whole image by ``image_blur``)."""
    g = np.full((H, W), 0.7)
    g[:, ::12] = 0.2
    g[:, 1::12] = 0.2
    band = np.zeros((H, W), bool)
    band[:, 10:70] = True
    g[band] = 0.05
    if blur:
        soft = ndi.gaussian_filter(band.astype(float), blur)
        wall = g.copy()
        wall[band] = 0.7
        g = np.where(band | (soft > 0.01), wall * (1 - soft) + 0.05 * soft, g)
    if image_blur:
        g = ndi.gaussian_filter(g, image_blur)
    rgb = np.repeat((g * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    depth = np.where(band, 0.25, 2.0).astype(np.float32)
    return rgb, depth, np.ones((H, W), bool)


def test_edges_in_focus_are_narrow_and_blurred_ones_wide() -> None:
    sharp, _, _ = _scene(0.0)
    soft, _, _ = _scene(4.0)
    assert np.nanmedian(edge_width(sharp)) < 3.0
    assert np.nanmedian(edge_width(soft)[:, 60:70]) > 2 * np.nanmedian(edge_width(sharp))
    assert np.isnan(edge_width(np.full((H, W, 3), 128, np.uint8))).all()  # no edge at all


def test_a_blurred_near_region_is_out_of_focus() -> None:
    rgb, depth, valid = _scene(4.0)
    out = defocused_near_field(rgb, depth, valid)
    assert out[:, 20:60].all() and not out[:, 90:].any()
    assert out[:, 70:74].any()  # its blurred rim too


def test_near_regions_in_focus_or_in_a_blurred_image_stay() -> None:
    rgb, depth, valid = _scene(0.0)
    assert not defocused_near_field(rgb, depth, valid).any()  # in focus
    rgb, depth, valid = _scene(4.0, image_blur=4.0)
    assert not defocused_near_field(rgb, depth, valid).any()  # every edge as wide
    # no depth, no edges beyond the near field, a near region too small to judge
    assert not defocused_near_field(rgb, depth, np.zeros((H, W), bool)).any()
    plain = np.full((H, W, 3), 90, np.uint8)
    assert not defocused_near_field(plain, depth, valid).any()
    rgb, depth, valid = _scene(4.0)
    tiny = np.full((H, W), 2.0, np.float32)
    tiny[50:52, 30:32] = 0.25
    assert not defocused_near_field(rgb, tiny, valid).any()
