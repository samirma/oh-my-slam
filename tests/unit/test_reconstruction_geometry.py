"""Point clouds, gravity refinement, depth alignment and fusion on synthetic scenes (no server)."""

from __future__ import annotations

import numpy as np
import pytest

from oh_my_slam.core.geometry import angle_between_deg, rotation_between
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.reconstruction import depth as dmod
from oh_my_slam.reconstruction.fusion import TsdfFusion, choose_voxel_size
from oh_my_slam.reconstruction.gravity import GravityEstimate, mean_up, refine_with_floor
from oh_my_slam.reconstruction.pointcloud import cloud_mask, frame_cloud
from tests.synth.scene import default_room, look_at, orbit_poses, render

K = Intrinsics(260.0, 260.0, 160.0, 120.0, 320, 240)


def test_frame_cloud_colours_and_edges() -> None:
    room = default_room()
    pose = look_at(np.array([2.5, 2.0, 1.5]), np.array([0.0, 0.0, 0.5]))
    r = render(room, pose, K)
    m = cloud_mask(r.depth, r.depth > 0)
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
