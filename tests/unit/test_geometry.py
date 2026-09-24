from __future__ import annotations

import numpy as np
import pytest

from oh_my_slam.core import geometry as g
from oh_my_slam.core.types import Intrinsics, Pose


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    return g.quat_to_rot(q / np.linalg.norm(q))


def test_quaternion_roundtrip(rng: np.random.Generator) -> None:
    for _ in range(200):
        R = random_rotation(rng)
        q = g.rot_to_quat(R)
        assert q[3] >= 0
        assert abs(np.linalg.norm(q) - 1) < 1e-12
        np.testing.assert_allclose(g.quat_to_rot(q), R, atol=1e-10)


def test_quaternion_scalar_last_convention() -> None:
    # 90 degrees about z: (0, 0, sin45, cos45)
    q = g.rot_to_quat(g.rot_z(np.pi / 2))
    np.testing.assert_allclose(q, [0, 0, np.sqrt(0.5), np.sqrt(0.5)], atol=1e-12)
    np.testing.assert_allclose(g.quat_to_rot([0, 0, 0, 1]), np.eye(3))


def test_pose_compose_inverse(rng: np.random.Generator) -> None:
    a = Pose(random_rotation(rng), rng.normal(size=3))
    b = Pose(random_rotation(rng), rng.normal(size=3))
    p = rng.normal(size=(10, 3))
    np.testing.assert_allclose(a.compose(b).apply(p), a.apply(b.apply(p)), atol=1e-10)
    np.testing.assert_allclose(a.inverse().apply(a.apply(p)), p, atol=1e-10)
    d = a.to_dict()
    np.testing.assert_allclose(Pose.from_dict(d).matrix(), a.matrix(), atol=1e-10)


def test_sim3_and_umeyama(rng: np.random.Generator) -> None:
    true = g.Sim3(2.5, random_rotation(rng), rng.normal(size=3))
    src = rng.normal(size=(50, 3))
    dst = true.apply(src)
    est = g.umeyama(src, dst)
    assert est.s == pytest.approx(2.5, rel=1e-9)
    np.testing.assert_allclose(est.R, true.R, atol=1e-9)
    np.testing.assert_allclose(est.apply(src), dst, atol=1e-9)
    inv = true.inverse()
    np.testing.assert_allclose(inv.apply(dst), src, atol=1e-9)
    comp = true.compose(inv)
    np.testing.assert_allclose(comp.apply(src), src, atol=1e-9)
    rigid = g.umeyama(src, src @ true.R.T + 1.0, with_scale=False)
    assert rigid.s == 1.0


def test_umeyama_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        g.umeyama(np.zeros((2, 3)), np.zeros((2, 3)))


def test_rotation_between_and_angles(rng: np.random.Generator) -> None:
    for _ in range(50):
        a, b = rng.normal(size=3), rng.normal(size=3)
        R = g.rotation_between(a, b)
        np.testing.assert_allclose(R @ (a / np.linalg.norm(a)), b / np.linalg.norm(b), atol=1e-9)
        assert abs(np.linalg.det(R) - 1) < 1e-9
    R = g.rotation_between([0, 0, 1], [0, 0, -1])
    np.testing.assert_allclose(R @ [0, 0, 1], [0, 0, -1], atol=1e-9)
    assert g.angle_between_deg([1, 0, 0], [0, 1, 0]) == pytest.approx(90)


def test_project_unproject_roundtrip(rng: np.random.Generator) -> None:
    K = Intrinsics(500, 510, 320, 240, 640, 480).K()
    depth = rng.uniform(1, 5, size=(48, 64))
    v, u = np.nonzero(depth > 0)
    pts = g.unproject_pixels(u, v, depth[v, u], K)
    uv, z = g.project(pts, K)
    np.testing.assert_allclose(uv[:, 0], u, atol=1e-9)
    np.testing.assert_allclose(uv[:, 1], v, atol=1e-9)
    np.testing.assert_allclose(z, depth[v, u])
    behind, _ = g.project(np.array([[0.0, 0.0, -1.0]]), K)
    assert np.isnan(behind).all()


def test_unproject_pixels_matches() -> None:
    K = Intrinsics(100, 100, 50, 50, 100, 100).K()
    p = g.unproject_pixels(np.array([60.0]), np.array([50.0]), np.array([2.0]), K)
    np.testing.assert_allclose(p, [[0.2, 0.0, 2.0]])


def test_depth_edge_mask() -> None:
    d = np.full((20, 20), 2.0)
    d[:, 10:] = 4.0
    d[0, 0] = np.nan
    edges = g.depth_edge_mask(d, rel_threshold=0.1)
    assert edges[:, 9].all() and edges[:, 10].all()
    assert not edges[5:15, 3].any()
    assert edges[1, 1]  # touches the invalid pixel
    assert not edges[0, 0]  # invalid pixels are never edges


def test_voxel_downsample_latest_wins() -> None:
    pts = np.array([[0.01, 0.01, 0.01], [0.02, 0.02, 0.02], [1.0, 1.0, 1.0]])
    idx = g.voxel_downsample_indices(pts, 0.1, keep="last")
    assert idx.tolist() == [1, 2]
    idx = g.voxel_downsample_indices(pts, 0.1, keep="first")
    assert idx.tolist() == [0, 2]
    assert g.voxel_downsample_indices(np.zeros((0, 3)), 0.1).size == 0


def test_ransac_plane_with_prior(rng: np.random.Generator) -> None:
    floor = np.c_[rng.uniform(-2, 2, 500), rng.uniform(-2, 2, 500), rng.normal(0, 0.005, 500)]
    wall = np.c_[rng.uniform(-2, 2, 800), np.full(800, 1.0), rng.uniform(0, 2, 800)]
    pts = np.vstack([floor, wall])
    res = g.ransac_plane(pts, 0.02, normal_prior=[0, 0, 1], max_angle_deg=20)
    assert res is not None
    n, d, inl = res
    assert g.angle_between_deg(n, [0, 0, 1]) < 1.0
    assert inl[:500].mean() > 0.95 and inl[500:].mean() < 0.1
    n2, d2 = g.fit_plane(floor)
    assert abs(abs(n2[2]) - 1) < 1e-3
    assert g.ransac_plane(np.zeros((2, 3)), 0.1) is None


def test_intrinsics_resize_and_dict() -> None:
    intr = Intrinsics(2658, 2658, 2000, 1500, 4000, 3000, "exif")
    small = intr.resized(1024, 768)
    assert small.fx == pytest.approx(2658 * 1024 / 4000)
    assert small.cx == pytest.approx(512)
    assert Intrinsics.from_dict(intr.to_dict()) == intr
    assert intr.with_source("colmap").source == "colmap"
    assert 70 < intr.fov_x_deg < 75
