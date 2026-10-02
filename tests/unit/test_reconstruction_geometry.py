"""Point clouds, gravity refinement, depth alignment and fusion on synthetic scenes (no server)."""

from __future__ import annotations

import numpy as np
import pytest

from oh_my_slam.core.geometry import angle_between_deg, rotation_between
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.reconstruction import depth as dmod
from oh_my_slam.reconstruction.fusion import TsdfFusion, choose_voxel_size
from oh_my_slam.reconstruction.gravity import GravityEstimate, mean_up, refine_with_floor
from oh_my_slam.reconstruction.pointcloud import frame_cloud, pixel_mask
from tests.synth.scene import default_room, look_at, orbit_poses, render

K = Intrinsics(260.0, 260.0, 160.0, 120.0, 320, 240)


def test_frame_cloud_colours_and_edges() -> None:
    room = default_room()
    pose = look_at(np.array([2.5, 2.0, 1.5]), np.array([0.0, 0.0, 0.5]))
    r = render(room, pose, K)
    m = pixel_mask(r.depth, r.depth > 0)
    assert m.sum() > 0.8 * (r.depth > 0).sum()
    cloud, idx = frame_cloud(r.depth, r.rgb, K, m, pose)
    np.testing.assert_array_equal(cloud.rgb, r.rgb.reshape(-1, 3)[idx])
    # floor pixels unproject to z ~ 0 in the map frame
    floor = (r.ids.reshape(-1)[idx] == 0)
    assert np.abs(cloud.xyz[floor, 2]).max() < 0.02
    with pytest.raises(ValueError):
        frame_cloud(r.depth, r.rgb[:10], K, m)


def test_gravity_floor_refinement_within_5_degrees(rng: np.random.Generator) -> None:
    room = default_room()
    pose = look_at(np.array([2.5, 2.0, 1.5]), np.array([0.0, 0.0, 0.3]))
    r = render(room, pose, K)
    cloud, _ = frame_cloud(r.depth, r.rgb, K, r.depth > 0)
    true_up = pose.R.T @ np.array([0.0, 0.0, 1.0])
    # a prior 3 degrees off
    tilt = rotation_between(true_up, true_up + np.array([0.05, 0.0, 0.0]))
    prior = GravityEstimate(tilt @ true_up, "geocalib", 1.0, 1.5)
    est = refine_with_floor(cloud.xyz, prior)
    assert est.source == "geocalib+floor"
    assert angle_between_deg(est.up_cam, true_up) < 0.5
    assert est.floor_height == pytest.approx(1.5, abs=0.03)
    # a prior 20 degrees off is kept (floor normal outside the 5 degree window)
    far = rotation_between(true_up, true_up + np.array([0.4, 0.0, 0.0])) @ true_up
    kept = refine_with_floor(cloud.xyz, GravityEstimate(far, "geocalib"))
    assert kept.source == "geocalib"
    assert refine_with_floor(cloud.xyz[:10], prior) is prior
    d = est.to_dict()
    assert GravityEstimate.from_dict(d).floor_inliers == est.floor_inliers
    assert est.confidence > prior.confidence
    up = mean_up([np.array([0, 0, 1.0]), np.array([0, 0.1, 1.0])], [1.0, 1.0])
    assert angle_between_deg(up, [0, 0.05, 1]) < 0.5


def test_depth_scale_fit_with_noise(rng: np.random.Generator) -> None:
    ref = rng.uniform(1, 5, 500)
    pred = ref / 1.37 * rng.normal(1, 0.02, 500)
    pred[:40] *= 3  # outliers
    fit = dmod.fit_frame_scale(pred, ref)
    assert fit.ok and fit.scale == pytest.approx(1.37, rel=0.01)
    assert fit.spread < 0.1
    bad = dmod.fit_frame_scale(pred[:10], ref[:10])
    assert not bad.ok
    s, spread = dmod.global_scale([1.0, 1.1, 0.9, float("nan")])
    assert s == pytest.approx(1.0) and spread > 0
    assert np.isnan(dmod.global_scale([])[0])
    d = np.arange(12, dtype=np.float32).reshape(3, 4)
    got = dmod.sample_depth(d, np.array([[1.2, 0.9], [9, 9], [0, 0]]))
    assert got[0] == 5 and np.isnan(got[1]) and np.isnan(got[2])


def test_fusion_points() -> None:
    room = default_room()
    poses = orbit_poses(12)
    fusion = TsdfFusion(voxel_size=0.02, depth_max=6.0)
    for p in poses:
        fusion.integrate(render(room, p, K).depth, K.K(), p)
    pts = fusion.extract_points()
    assert len(pts) > 1000
    # geometry lies inside the room
    assert pts[:, 2].min() > -0.1 and pts[:, 2].max() < 2.7
    assert fusion.stats.frames == 12
    assert choose_voxel_size(1.0) == 0.01 and choose_voxel_size(3.0) == 0.015
    assert choose_voxel_size(50.0) == 0.04
    fusion.integrate(np.zeros((240, 320), np.float32), K.K(), Pose.identity())
    assert fusion.stats.frames == 12  # empty depth skipped
    assert len(TsdfFusion(voxel_size=0.02, depth_max=6.0).extract_points()) == 0


def test_consensus_reads_a_neighbours_depth_along_the_pixels_ray() -> None:
    """A neighbour that sees a wall 2 % deeper than the view does gives, for each of the view's
    pixels, the depth in the view of its own surface point on its ray through the pixel's point
    (rho z + (1 - rho) b): exact for the forward step it took away from the wall. A neighbour 6 %
    deeper sees through the pixels; one that never sees the wall says nothing; and one whose free
    space does not count (``carves``) supports but never sees through."""
    from oh_my_slam.reconstruction.consensus import (
        Views,
        free_space,
        median_ratio,
        neighbour_views,
        sample_pixels,
    )

    wall = 2.0
    view = Pose.identity()
    back = Pose(np.eye(3), np.array([0.05, 0.02, -0.5]))  # 0.5 m behind the view
    away = Pose(np.diag([-1.0, 1.0, -1.0]), np.zeros(3))  # looking the other way
    depth = np.full((240, 320), wall, np.float32)
    nb_depth = np.full((240, 320), (wall + 0.5) * 1.02, np.float32)  # 2 % deeper
    far_depth = np.full((240, 320), (wall + 0.5) * 1.06, np.float32)  # 6 % deeper
    s = sample_pixels(depth, depth > 0, K, 4)
    assert len(s.z) == 60 * 80 and np.allclose(s.cam[2], wall)
    views = Views.of([nb_depth, far_depth, nb_depth], [nb_depth > 0] * 3)
    kk, on, along, thru = neighbour_views(s, view, views, [0, 1, 2], [K] * 3,
                                          [back, back, away], np.array([True, True, True]),
                                          same=0.03, free=0.03)
    assert set(kk.tolist()) == {0} and len(on) > 0.9 * len(s.z)
    # the neighbour's point on its ray through (x, y, 2): its depth in the view
    X = s.cam[:, on].T.astype(np.float64)
    expected = (back.t + 1.02 * (X - back.t))[:, 2]
    np.testing.assert_allclose(along, expected, rtol=1e-5)
    assert np.all(np.abs(along / wall - 1.0) > 0.019)  # 2 % deeper along the ray too
    assert len(thru) > 0.9 * len(s.z) and len(np.unique(thru)) == len(thru)  # the 6 % one
    _, _, _, thru2 = neighbour_views(s, view, views, [1], [K], [back], np.array([False]),
                                     same=0.03, free=0.03)
    assert len(thru2) == 0
    # the median of the sample's own depth and its neighbours' (NaN: no view)
    z = np.array([2.0, 2.0, 2.0])
    nbs = np.array([[2.04, 2.02, np.nan], [np.nan, np.nan, np.nan], [1.9, 2.1, 2.2]])
    np.testing.assert_allclose(median_ratio(z, nbs), [1.01, 1.0, 1.025])
    # left out: at least 2 views see through it, more than half of those that see it or through
    np.testing.assert_array_equal(free_space(np.array([1, 2, 2, 3]), np.array([0, 1, 2, 2]),
                                             2, 0.5), [False, True, False, True])


def test_consensus_cells_cover_grids_of_any_size() -> None:
    """Each sample's ratio and drop cover its step x step cell on its own surface (not across a
    depth step), on grids whose sides are not multiples of the step (the last cells then have no
    sample and keep their depth)."""
    from oh_my_slam.reconstruction.consensus import sample_pixels, spread_cells

    for h, w in ((240, 320), (241, 321), (242, 323)):
        d = np.full((h, w), 2.0, np.float32)
        d[:, 101:] = 3.0  # a depth step inside the cell of columns 100-103 (sampled at 102)
        Kg = Intrinsics(260.0, 260.0, w / 2, h / 2, w, h)
        s = sample_pixels(d, d > 0, Kg, 4)
        drop = s.z > 2.5  # the far surface is left out
        out, left = spread_cells(d, d > 0, s, np.full(len(s.z), 1.01), drop, 4, 0.03)
        assert out.shape == (h, w) and left.shape == (h, w)
        rows = (h // 4) * 4  # pixel rows whose cell has a sample
        expected = d[:rows] * np.float32(1.01)
        expected[:, 100] = 2.0  # off its cell sample's surface: kept
        np.testing.assert_allclose(out[:rows, :320], expected[:, :320], rtol=1e-6)
        np.testing.assert_array_equal(out[rows:], d[rows:])
        far = d[:rows, :320] > 2.5
        np.testing.assert_array_equal(left[:rows, :320], far)
        assert not left[rows:].any()
