"""Reconstruction (the single owner of depth and point-cloud generation, spec §4) at its edges: a
server whose depth grid is not the image's, gravity without the floor refinement, a floor
refinement that keeps the prior (too little floor, two floor levels, a "floor" above the camera),
pixel selection and normals without a validity mask, depth pairs that overlap too little, an
iteration cap on the depth adjustment, the fusion's subsampled grid, a frame with no voxel block
to update and an Open3D failure that is not "no surface"."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.reconstruction import depth as rdepth
from oh_my_slam.reconstruction import fusion
from oh_my_slam.reconstruction.api import estimate_gravity, reconstruct_image
from oh_my_slam.reconstruction.gravity import GravityEstimate, refine_with_floor
from oh_my_slam.reconstruction.pointcloud import depth_normals, pixel_mask
from tests.fakes.client import FakeClient, FakeFrame

K = Intrinsics(100.0, 100.0, 80.0, 60.0, 160, 120)
UP = np.array([0.0, -1.0, 0.0])  # a level camera (OpenCV axes)


def image(tmp_path: Path, client: FakeClient, depth: np.ndarray) -> Path:
    rgb = np.random.default_rng(3).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    return client.add(tmp_path / "img.png", rgb, FakeFrame(depth.astype(np.float32), K, UP))


# -- single image --------------------------------------------------------------------------------


def test_a_depth_grid_that_is_not_the_images_is_refused(tmp_path: Path) -> None:
    client = FakeClient()
    img = image(tmp_path, client, np.full((60, 80), 2.0))  # the server answered another grid
    with pytest.raises(RuntimeError, match=r"grid mismatch: image \(120, 160\) vs depth "
                                           r"\(60, 80\)"):
        reconstruct_image(img, client=client, want_gravity=False)


def test_gravity_without_the_floor_refinement_is_geocalibs(tmp_path: Path) -> None:
    client = FakeClient()
    img = image(tmp_path, client, np.full((120, 160), 2.0))
    frame = reconstruct_image(img, client=client, want_gravity=False)
    g = estimate_gravity(frame, client, refine=False)
    assert g.source == "geocalib" and np.allclose(g.up_cam, UP) and g.floor_height is None
    assert frame.meta["geocalib_focal_px"] == K.fx


def _plane(height: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` camera-frame points of a horizontal surface ``height`` m along up (UP)."""
    return np.c_[rng.uniform(-2, 2, n), np.full(n, -height), rng.uniform(1, 5, n)]


def _above(lo: float, hi: float, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.c_[rng.uniform(-1, 1, n), -rng.uniform(lo, hi, n), rng.uniform(2, 4, n)]


@pytest.mark.parametrize(("case", "min_inliers"), [
    ("two floor levels", 1000),  # the band holds both; the plane only one of them
    ("too little floor", 5000),  # the band itself is smaller than needed
    ("floor above the camera", 300),  # the lowest surface is over the camera's head
])
def test_the_floor_refinement_keeps_the_prior(case: str, min_inliers: int) -> None:
    rng = np.random.default_rng(7)
    if case == "floor above the camera":
        pts = np.r_[_plane(0.5, 600, rng), _above(0.7, 1.5, 500, rng)]
    else:
        pts = np.r_[_plane(-1.5, 600, rng), _plane(-1.4, 600, rng), _above(-1.2, -0.5, 500, rng)]
    prior = GravityEstimate(up_cam=UP.copy(), source="geocalib", roll_unc_deg=2.0)
    assert refine_with_floor(pts, prior, min_inliers=min_inliers) is prior


def test_a_single_floor_level_refines_the_prior() -> None:
    rng = np.random.default_rng(7)
    pts = np.r_[_plane(-1.5, 1200, rng), _above(-1.2, -0.5, 500, rng)]
    prior = GravityEstimate(up_cam=UP.copy(), source="geocalib")
    g = refine_with_floor(pts, prior, min_inliers=1000)
    assert g.source == "geocalib+floor" and g.floor_height == pytest.approx(1.5, abs=1e-6)


def test_pixel_selection_and_normals_without_a_validity_mask() -> None:
    depth = np.full((40, 50), 2.0, np.float32)
    depth[:5] = 0.0  # no depth there
    everywhere = np.ones(depth.shape, bool)
    assert np.array_equal(pixel_mask(depth), pixel_mask(depth, everywhere))
    assert not pixel_mask(depth)[:5].any() and pixel_mask(depth)[10:].all()
    k =Intrinsics(40.0, 40.0, 25.0, 20.0, 50, 40)
    n = depth_normals(depth, k)
    assert np.array_equal(n, depth_normals(depth, k, everywhere))
    assert np.allclose(n[20, 25], [0.0, 0.0, -1.0], atol=1e-5)  # a wall facing the camera
    assert not n[:5].any()  # no depth: no normal


# -- depth alignment -----------------------------------------------------------------------------


def _view(depth: np.ndarray, x: float = 0.0) -> rdepth.DepthView:
    T = np.eye(4)
    T[0, 3] = x
    k = np.array([[100.0, 0, depth.shape[1] / 2], [0, 100.0, depth.shape[0] / 2], [0, 0, 1]])
    return rdepth.DepthView(depth.astype(np.float32), k, T)


def test_depth_pairs_that_overlap_too_little_give_no_bins() -> None:
    wall = np.full((192, 256), 2.0)
    assert rdepth.pair_bins(0, 1, _view(np.zeros((192, 256))), _view(wall)) == []  # no depth
    # 36 points on the same surface in each direction: enough to measure, too few to bin
    small = np.full((48, 48), 2.0)
    assert rdepth.pair_bins(0, 1, _view(small), _view(small), min_points=10) == []
    bins = rdepth.pair_bins(0, 1, _view(wall), _view(wall, 0.05))
    assert bins and {(b.src, b.dst) for b in bins} == {(0, 1), (1, 0)}


def test_the_depth_adjustment_stops_at_its_iteration_cap() -> None:
    """With a robust scale, each keyframe's correction is re-solved until it moves less than
    1e-6 or ``iterations`` is reached: a loop whose ratios disagree has not converged after one."""
    def b(s: int, t: int, r: float) -> rdepth.BinRatio:
        return rdepth.BinRatio(s, t, r, 0.7 + 0.1 * s, 0.7 + 0.1 * t, 0.2)

    bins = [b(0, 1, 0.1), b(1, 2, 0.3), b(2, 0, -0.05), b(0, 2, 0.5), b(2, 1, 0.02)]
    pivots = np.log([2.0, 2.2, 2.4])
    once = rdepth.adjust_depth_corrections(3, bins, pivots, {0}, iterations=1)
    done = rdepth.adjust_depth_corrections(3, bins, pivots, {0}, iterations=50)
    assert once.corrections[0].identity and done.corrections[0].identity
    assert once.corrections[1].log_scale != pytest.approx(done.corrections[1].log_scale,
                                                          abs=1e-6)
    assert done.residuals_after.sum() < done.residuals_before.sum()


# -- fusion --------------------------------------------------------------------------------------


def test_a_large_depth_grid_is_subsampled_for_fusion() -> None:
    depth = np.arange(64 * 48, dtype=np.float32).reshape(48, 64)
    k = np.array([[60.0, 0, 32.0], [0, 60.0, 24.0], [0, 0, 1]])
    d, k2 = fusion._downsample(depth, k, 32)
    assert fusion.fusion_step(depth.shape, 32) == 2
    assert np.array_equal(d, depth[::2, ::2])
    assert np.allclose(k2, [[30.0, 0, 16.0], [0, 30.0, 12.0], [0, 0, 1]]) and k[0, 0] == 60.0
    same, k3 = fusion._downsample(depth, k, 64)
    assert same is depth and k3 is k


def test_a_frame_with_no_block_to_update_integrates_nothing() -> None:
    f = fusion.TsdfFusion(voxel_size=0.02, depth_max=5.0)
    depth = (2.0 + 0.004 * np.mgrid[0:96, 0:128][1]).astype(np.float32)  # a slanted wall
    k = np.array([[100.0, 0, 64.0], [0, 100.0, 48.0], [0, 0, 1]])
    blocks = f.block_coords(depth, k, Pose.identity())
    assert blocks is not None and len(blocks) > 0
    f.integrate(depth, k, Pose.identity(), blocks=np.zeros((0, 3), np.int32))
    assert f.stats.frames == 1 and len(f.extract_points()) == 0
    for _ in range(2):  # (a surface needs more than one observation's weight)
        f.integrate(depth, k, Pose.identity(), blocks=blocks)
    assert len(f.extract_points()) > 0


def test_an_open3d_failure_that_is_not_no_surface_is_raised() -> None:
    f = fusion.TsdfFusion(voxel_size=0.02, depth_max=5.0)

    def fails(**_kw: Any) -> Any:
        raise RuntimeError("[Open3D Error] out of memory")

    f.vbg = SimpleNamespace(hashmap=lambda: SimpleNamespace(size=lambda: 1),
                            extract_point_cloud=fails)
    with pytest.raises(RuntimeError, match="out of memory"):
        f.extract_points()
