from __future__ import annotations

import numpy as np
import pytest

from oh_my_slam.core.geometry import rot_z, rotation_between
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.segmentation.lift import (
    largest_cluster,
    lift_mask,
    shrink_mask,
    statistical_outliers,
)
from oh_my_slam.segmentation.obb import OBB, fit_obb, fit_upright_obb, obb_iou_upright


def box_surface(rng: np.random.Generator, size: tuple[float, float, float], yaw: float,
                center: tuple[float, float, float], n: int = 4000,
                faces: tuple[int, ...] = (0, 1, 2, 3, 4, 5)) -> np.ndarray:
    w, d, h = size
    pts = []
    per = n // len(faces)
    for f in faces:
        u = rng.uniform(-0.5, 0.5, (per, 3)) * [w, d, h]
        axis, sign = divmod(f, 2)
        u[:, axis] = (0.5 if sign else -0.5) * [w, d, h][axis]
        pts.append(u)
    p = np.vstack(pts)
    return p @ rot_z(yaw).T + np.asarray(center)


@pytest.mark.parametrize("yaw", [0.0, 0.3, -1.2, np.pi / 4, 1.5])
def test_rotated_box_recovered(rng: np.random.Generator, yaw: float) -> None:
    pts = box_surface(rng, (1.2, 0.5, 0.8), yaw, (1.0, -2.0, 0.4))
    box = fit_upright_obb(pts)
    np.testing.assert_allclose(box.size, [1.2, 0.5, 0.8], atol=0.06)
    np.testing.assert_allclose(box.center, [1.0, -2.0, 0.4], atol=0.03)
    truth = OBB(np.array([1.0, -2.0, 0.4]), rot_z(yaw), np.array([1.2, 0.5, 0.8]))
    assert obb_iou_upright(box, truth) > 0.85
    # canonical: x is the longer horizontal side, R is a z rotation with yaw in (-90, 90]
    assert box.size[0] >= box.size[1]
    assert np.allclose(box.R[:, 2], [0, 0, 1])
    yaw_est = np.arctan2(box.R[1, 0], box.R[0, 0])
    assert -np.pi / 2 < yaw_est <= np.pi / 2 + 1e-9


def test_outliers_do_not_inflate_the_box(rng: np.random.Generator) -> None:
    pts = box_surface(rng, (0.6, 0.4, 0.9), 0.2, (0, 0, 0.45), n=6000)
    outliers = rng.uniform(-3, 3, (40, 3))
    box = fit_upright_obb(np.vstack([pts, outliers]))
    assert box.size.max() < 1.0


def test_partial_surface_gives_visible_extent(rng: np.random.Generator) -> None:
    # only the front face and top visible (single image): depth extent is small, never negative
    pts = box_surface(rng, (1.0, 0.6, 0.7), 0.0, (0, 2.0, 0.35), faces=(2, 5))
    box = fit_upright_obb(pts)
    assert box.size[0] == pytest.approx(1.0, abs=0.06)
    assert 0.01 <= box.size[1] <= 0.65


def test_fit_obb_in_camera_frame_uses_up(rng: np.random.Generator) -> None:
    world = box_surface(rng, (1.0, 0.5, 0.8), 0.4, (0.5, 3.0, 0.4))
    # camera looking along +y (map), pitched down 20 degrees; points in camera coordinates
    R_cam_map = rotation_between([0, 0, 1], [0, -np.sin(0.35), np.cos(0.35)])
    T_map_cam = Pose(R_cam_map.T @ np.diag([1, 1, 1]), np.array([0.0, 0.0, 1.5]))
    cam = T_map_cam.inverse().apply(world)
    up_cam = T_map_cam.inverse().R @ np.array([0, 0, 1.0])
    box_cam = fit_obb(cam, up_cam)
    box_map = box_cam.transformed(T_map_cam)
    np.testing.assert_allclose(box_map.size, [1.0, 0.5, 0.8], atol=0.06)
    np.testing.assert_allclose(box_map.R[:, 2], [0, 0, 1], atol=1e-6)
    d = OBB.from_dict(box_cam.to_dict())
    np.testing.assert_allclose(d.corners(), box_cam.corners())
    assert box_cam.volume == pytest.approx(np.prod(box_cam.size))
    with pytest.raises(ValueError):
        fit_upright_obb(np.zeros((0, 3)))
    few = fit_upright_obb(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.0]]))
    assert few.size.min() >= 0.01


def test_largest_cluster_and_sor(rng: np.random.Generator) -> None:
    a = rng.normal(0, 0.05, (500, 3))
    b = rng.normal(0, 0.05, (100, 3)) + [2, 0, 0]
    keep = largest_cluster(np.vstack([a, b]), 0.05)
    assert keep[:500].mean() > 0.95 and not keep[500:].any()
    assert largest_cluster(np.zeros((0, 3)), 0.1).size == 0
    assert largest_cluster(np.zeros((3, 3)), 0.1).all()
    pts = np.vstack([rng.normal(0, 0.01, (300, 3)), [[1, 1, 1]]])
    inl = statistical_outliers(pts)
    assert not inl[-1] and inl[:300].mean() > 0.9
    assert statistical_outliers(pts[:3]).all()


def test_lift_mask_with_bleeding() -> None:
    """A mask that bleeds 4 px onto the background wall: shrink + clustering drop the wall."""
    K = Intrinsics(200, 200, 50, 50, 100, 100)
    depth = np.full((100, 100), 5.0)
    depth[30:70, 30:70] = 2.0  # object 3 m in front of the wall
    mask = np.zeros((100, 100), bool)
    mask[26:74, 26:74] = True
    lifted = lift_mask(mask, depth, K)
    z = lifted.points[:, 2]
    assert np.all(np.abs(z - 2.0) < 1e-6)
    assert lifted.mask_pixels == mask.sum()
    moved = lift_mask(mask, depth, K, T_parent_cam=Pose(np.eye(3), np.array([0, 0, 1.0])))
    assert np.allclose(moved.points[:, 2], 3.0)
    empty = lift_mask(np.zeros((100, 100), bool), depth, K)
    assert len(empty.points) == 0
    assert shrink_mask(mask, 0).sum() == mask.sum()
    assert shrink_mask(mask, 3).sum() == (48 - 6) ** 2
