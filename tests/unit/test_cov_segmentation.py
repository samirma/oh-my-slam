"""Segmentation (the single owner of segmentation, OBB fitting and colour assignment, spec §4) at
its edges: a detection whose mask is empty claims no pixel, a mask lifted with only the model's
validity mask, a support strip that cannot be judged on too few points, a box fitted to points on
a line, a colour that would repeat an earlier id or break the colour rules, no heights to ramp,
an image cloud thinned for display, and a scene description without gravity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from oh_my_slam.core.cloud_attrs import CloudAttrs
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.reconstruction.api import reconstruct_image
from oh_my_slam.schema.validate import validation_errors
from oh_my_slam.segmentation import colors
from oh_my_slam.segmentation.api import Detection, pixel_owners, segment_frame
from oh_my_slam.segmentation.cloud import derive_cloud, derive_thinned, image_cloud_source
from oh_my_slam.segmentation.lift import MIN_POINTS, lift_mask, support_fringe
from oh_my_slam.segmentation.obb import fit_upright_obb
from oh_my_slam.segmentation.scene import single_image_scene
from tests.fakes.client import FakeClient, FakeFrame, FakeInstance

K = Intrinsics(100.0, 100.0, 40.0, 30.0, 80, 60)


def rect(r0: int, r1: int, c0: int, c1: int, shape: tuple[int, int] = (60, 80)) -> np.ndarray:
    m = np.zeros(shape, bool)
    m[r0:r1, c0:c1] = True
    return m


def test_a_detection_whose_mask_is_empty_claims_no_pixel() -> None:
    empty = Detection("cup", 0.9, "yoloe", np.zeros((60, 80), bool), (0.0, 0.0, 0.0, 0.0))
    table = Detection("table", 0.8, "yoloe", rect(10, 50, 10, 70), (10.0, 10.0, 70.0, 50.0))
    owner = pixel_owners([empty, table], (60, 80))
    assert np.array_equal(owner == 1, table.mask) and not (owner == 0).any()


def test_a_mask_lifted_with_the_models_validity_mask_only() -> None:
    """Without precomputed depth edges, they are found on the valid depth; invalid pixels never
    become points."""
    depth = np.full((60, 80), 2.0)
    valid = np.ones((60, 80), bool)
    valid[:, :40] = False
    mask = rect(10, 50, 20, 60)
    lifted = lift_mask(mask, depth, K, valid)
    cols = lifted.pixels % 80
    assert len(lifted.points) >= MIN_POINTS and (cols >= 40).all()
    assert lifted.mask_pixels == int(mask.sum())
    assert np.allclose(lifted.points[:, 2], 2.0)


def test_a_support_strip_is_not_judged_on_too_few_points() -> None:
    mask = rect(10, 50, 20, 60)
    few = np.flatnonzero(mask.reshape(-1))[:MIN_POINTS - 1]
    up = Pose(np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]]), np.zeros(3))  # camera → z up
    drop = support_fringe(mask, np.full((60, 80), 2.0), K, np.ones((60, 80), bool), up, few)
    assert drop.shape == mask.shape and not drop.any()


def test_a_box_fitted_to_points_on_a_line_is_axis_aligned() -> None:
    """Points on a vertical plane through one horizontal line: no hull in the horizontal plane,
    so the box keeps yaw 0 and spans the line."""
    t = np.linspace(0.0, 1.0, 50)
    pts = np.c_[t, np.zeros(50), np.linspace(0.0, 0.5, 50) ** 2]
    box = fit_upright_obb(pts)
    assert np.allclose(box.R, np.eye(3))
    assert box.size[0] == pytest.approx(0.96, abs=1e-6) and box.size[1] == pytest.approx(0.01)


def test_a_colour_that_would_repeat_an_id_or_break_the_rules_is_nudged() -> None:
    """The nearest 8-bit triple that is neither issued yet nor outside the colour rules."""
    table = colors._Table()  # a table of its own (the module's is shared)
    taken = table.rgb[0]
    nudged = table._unused(taken)
    assert nudged != taken and nudged not in table.used and bool(colors.admissible(nudged))
    assert max(abs(a - b) for a, b in zip(nudged, taken, strict=True)) <= 3
    grey = colors.UNSEGMENTED  # reserved, and never admissible
    fixed = table._unused((130, 128, 128))
    assert fixed not in table.used and bool(colors.admissible(fixed)) and fixed != grey
    fresh = (12, 200, 90)
    assert bool(colors.admissible(fresh)) and table._unused(fresh) == fresh


def test_no_heights_have_no_ramp() -> None:
    assert colors.height_colors(np.zeros(0)).shape == (0, 3)


def frame_with_objects(tmp_path: Path, gravity: bool):  # type: ignore[no-untyped-def]
    client = FakeClient()
    depth = np.full((60, 80), 2.0, np.float32)
    depth[20:40, 30:50] = 1.5  # a box in front of the wall
    rgb = np.random.default_rng(5).integers(0, 255, (60, 80, 3), dtype=np.uint8)
    img = client.add(tmp_path / "img.png", rgb, FakeFrame(
        depth, K, np.array([0.0, -1.0, 0.0]), [FakeInstance("box", 0.9, rect(20, 40, 30, 50))]))
    frame = reconstruct_image(img, client=client, want_gravity=gravity)
    return frame, segment_frame(frame, client=client)


def test_an_image_cloud_thinned_for_display_keeps_exact_points(tmp_path: Path) -> None:
    frame, seg = frame_with_objects(tmp_path, gravity=False)
    source = image_cloud_source(frame, seg)
    attrs = CloudAttrs(color="segment", label=True)
    full = derive_cloud(source, attrs)
    thin = derive_thinned(source, attrs, 500)
    assert thin.total == len(full.xyz) > 500 >= len(thin.cloud.xyz) and thin.voxel > 0
    rows = {tuple(p): i for i, p in enumerate(full.xyz.tolist())}
    picked = [rows[tuple(p)] for p in thin.cloud.xyz.tolist()]  # each one an original point
    assert np.array_equal(thin.cloud.rgb, full.rgb[picked])
    assert np.array_equal(thin.cloud.label, full.label[picked])
    again = derive_thinned(source, CloudAttrs(color="rgb"), 500)  # colour only: same selection
    assert np.array_equal(again.cloud.xyz, thin.cloud.xyz) and again.voxel == thin.voxel


def test_a_scene_without_gravity_says_none(tmp_path: Path) -> None:
    frame, seg = frame_with_objects(tmp_path, gravity=False)
    assert frame.gravity is None and [o.label for o in seg.objects] == ["box"]
    doc = single_image_scene(seg, tool="segment")
    meta = doc["openlabel"]["metadata"]
    assert "gravity" not in meta and meta["tool"] == "segment"
    assert validation_errors(doc) == []
