"""Lens distortion (``Intrinsics.k``, COLMAP's division model): pixels to the undistorted image's
and back, the undistorted image (which holds the whole lens), grids over it, the OpenCV
coefficients and the OpenLABEL camera stream that records it (and gives it back)."""

from __future__ import annotations

import numpy as np
import pytest

from oh_my_slam.core.images import grid_index, undistort_rgb
from oh_my_slam.core.types import Intrinsics
from oh_my_slam.schema import openlabel as ol

LENS = Intrinsics(1392.0, 1392.0, 960.0, 444.0, 1920, 888, "colmap", -0.524)


def test_a_pinhole_has_no_distortion() -> None:
    K = LENS.pinhole()
    assert K.k == 0.0 and (K.cx, K.cy, K.width, K.height) == (LENS.cx, LENS.cy, 1920, 888)
    assert K.pinhole() == K and K.undistorted_scale == 1.0
    uv = np.array([[10.0, 20.0], [1900.0, 880.0]])
    assert np.array_equal(K.pinhole_pixels(uv), uv) and np.array_equal(K.image_pixels(uv), uv)
    np.testing.assert_allclose(K.rays(uv), (uv - [960, 444]) / K.fx)
    assert K.opencv_distortion() == [0.0] * 5
    assert "k" not in K.to_dict() and Intrinsics.from_dict(K.to_dict()) == K


def test_the_undistorted_image_holds_the_whole_lens() -> None:
    """Barrel distortion: a shorter focal length than the lens's, so that the image's corners — its
    farthest pixels — stay in the undistorted image; pincushion distortion keeps the lens's focal
    length; a lens past 90° is held at a tenth."""
    s = 1 - 0.524 * ((960 / 1392) ** 2 + (444 / 1392) ** 2)
    assert LENS.undistorted_scale == pytest.approx(s)
    assert LENS.pinhole().fx == pytest.approx(1392 * s) == LENS.pinhole().fy
    corners = np.array([[0.0, 0.0], [1920.0, 0.0], [0.0, 888.0], [1920.0, 888.0]])
    np.testing.assert_allclose(LENS.pinhole_pixels(corners), corners, atol=1e-9)
    edges = LENS.pinhole_pixels(np.array([[0.0, 444.0], [960.0, 0.0]]))
    assert 0 < edges[0, 0] < 150 and 0 < edges[1, 1] < 150  # the edges' middles lie inside
    pincushion = Intrinsics(100.0, 100.0, 50.0, 50.0, 100, 100, "colmap", 0.5)
    assert pincushion.undistorted_scale == 1.0
    assert Intrinsics(100.0, 100.0, 50.0, 50.0, 100, 100, "colmap", -4.0).undistorted_scale == 0.1


def test_the_lens_maps_pixels_to_its_pinhole_and_back() -> None:
    uv = np.array([[960.0, 444.0], [0.0, 0.0], [1919.0, 887.0], [1500.0, 300.0]])
    pin = LENS.pinhole_pixels(uv)
    np.testing.assert_allclose(pin[0], [960.0, 444.0])
    d = (uv[3] - [960, 444]) / 1392.0
    np.testing.assert_allclose(LENS.rays(uv[3:]), [d / (1 - 0.524 * d @ d)])
    np.testing.assert_allclose(pin[3], d / (1 - 0.524 * d @ d) * LENS.pinhole().fx + [960, 444])
    # barrel distortion: the lens's corners are the undistorted image's
    np.testing.assert_allclose(pin[1], [0.0, 0.0], atol=1e-9)
    assert 1916 < pin[2, 0] < 1920 and 885 < pin[2, 1] < 888
    np.testing.assert_allclose(LENS.image_pixels(pin), uv, atol=1e-9)
    # a ray beyond what a pincushion lens (k > 0) can show
    pincushion = Intrinsics(100.0, 100.0, 50.0, 50.0, 100, 100, "colmap", 0.5)
    assert np.isnan(pincushion.image_pixels(np.array([[200.0, 50.0]]))).all()


def test_the_lens_is_kept_across_sizes_sources_and_json() -> None:
    small = LENS.resized(960, 444)
    assert small.k == LENS.k and small.fx == pytest.approx(696.0)
    assert LENS.with_source("given").k == LENS.k
    d = LENS.to_dict()
    assert d["k"] == -0.524 and Intrinsics.from_dict(d) == LENS


def test_opencv_coefficients_bend_the_rays_as_the_lens_does() -> None:
    import cv2

    c = LENS.opencv_distortion()
    assert len(c) == 8 and c[2] == c[3] == 0.0
    uv = np.random.default_rng(0).uniform([0, 0], [1920, 888], (500, 2))
    rays = np.column_stack([LENS.rays(uv), np.ones(len(uv))])
    proj, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), LENS.K(), np.array(c))
    assert np.abs(proj[:, 0] - uv).max() < 0.01


def test_the_undistorted_image_is_the_pinholes() -> None:
    rgb = np.full((888, 1920, 3), 60, np.uint8)
    rgb[:, ::40] = 255  # vertical lines, bent by the lens away from the centre
    assert undistort_rgb(rgb, LENS.pinhole()) is rgb
    out = undistort_rgb(rgb, LENS)
    assert out.shape == rgb.shape and out.dtype == np.uint8
    row = out[444, :, 0].astype(int)
    assert row[959:962].max() > 200  # the line through the centre stays there
    # the next one is nearer: the undistorted image has a shorter focal length than the lens
    assert 960 + 40 * LENS.undistorted_scale - 2 < 962 + np.argmax(row[962:1010]) < 992
    # the undistorted image's corner is the lens's; the middle of its top edge lies beyond the
    # lens, where it takes the colours of the lens's edge
    src = LENS.image_pixels(np.array([[0.5, 0.5]]))[0]
    assert 0 < src[0] < 2 and 0 < src[1] < 2
    assert LENS.image_pixels(np.array([[960.5, 0.5]]))[0, 1] < 0 and out.min() == 60
    with pytest.raises(ValueError, match="intrinsics of 1920x888"):
        undistort_rgb(rgb[:100], LENS)


def test_a_grid_over_the_undistorted_image_indexes_the_image_grid() -> None:
    """``grid_index``: each pixel of a grid over the undistorted image -> the pixel of a grid over
    the image (or over another undistorted image) that shows the same point; -1 where none."""
    plain = LENS.pinhole()
    np.testing.assert_array_equal(grid_index(plain, (8, 4), (8, 4)), np.arange(32).reshape(4, 8))
    idx = grid_index(LENS, (192, 89), (96, 44))
    assert idx.shape == (89, 192) and idx.dtype == np.intp
    assert idx[44, 96] == 22 * 96 + 48  # the centre
    assert idx[0, 96] == -1 and idx[44, 0] == -1 and idx[0, 0] == 0  # beyond the lens: -1
    rows, cols = np.divmod(idx[44, 100:170], 96)
    assert (rows == 22).all() and (np.diff(cols) >= 0).all()  # monotonic along the centre row
    # from one undistorted grid to another of the same lens: the same pixels
    same = grid_index(LENS, (96, 44), (96, 44), LENS)
    shown = same >= 0
    rows, cols = np.divmod(same[shown], 96)
    v, u = np.nonzero(shown)
    np.testing.assert_array_equal(rows, v)
    np.testing.assert_array_equal(cols, u)
    assert (same[0, 48] == -1) and (same >= -1).all() and shown[22, 48]


def test_a_camera_stream_records_its_lens() -> None:
    pin = ol.camera_stream(LENS.pinhole(), uri="a.jpg")["stream_properties"]
    assert pin["intrinsics_pinhole"]["distortion_coeffs"] == [0.0] * 5
    assert "intrinsics_custom" not in pin
    lens = ol.camera_stream(LENS)["stream_properties"]
    assert lens["intrinsics_pinhole"]["distortion_coeffs"] == LENS.opencv_distortion()
    custom = lens["intrinsics_custom"]
    assert custom["model"] == "division" and custom["k"] == -0.524
    assert (custom["focal_length_px"], custom["center_x_px"]) == (1392.0, 960.0)


def test_a_camera_stream_gives_its_camera_back() -> None:
    for cam in (LENS, LENS.pinhole().with_source("exif")):
        back = ol.stream_intrinsics(ol.camera_stream(cam)["stream_properties"])
        assert (back.width, back.height, back.source, back.k) == (cam.width, cam.height,
                                                                  cam.source, cam.k)
        np.testing.assert_allclose([back.fx, back.fy, back.cx, back.cy],
                                   [cam.fx, cam.fy, cam.cx, cam.cy], rtol=1e-6)
