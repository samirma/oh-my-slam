"""Synthetic rooms with boxes, rendered by ray casting (Open3D) into RGB + metric depth.

Map frame: z up, metres. Cameras: OpenCV axes, ``Pose`` = camera-to-map.
Surfaces get a procedural high-frequency texture so feature matching works on the renders.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.geometry import rot_z, rotation_between
from oh_my_slam.core.types import Intrinsics, Pose


@dataclass
class Box:
    center: NDArray[np.float64]  # map frame
    size: NDArray[np.float64]  # width (x), depth (y), height (z)
    yaw: float = 0.0
    color: tuple[int, int, int] = (200, 60, 60)
    label: str = "box"

    def corners(self) -> NDArray[np.float64]:
        sx, sy, sz = np.asarray(self.size) / 2
        c = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
        return c @ rot_z(self.yaw).T + self.center

    def mesh(self):  # type: ignore[no-untyped-def]
        import open3d as o3d

        m = o3d.geometry.TriangleMesh.create_box(*self.size)
        m.translate(-np.asarray(self.size) / 2)
        m.rotate(rot_z(self.yaw), center=(0, 0, 0))
        m.translate(self.center)
        return m


@dataclass
class Room:
    size: tuple[float, float, float] = (6.0, 5.0, 2.6)
    boxes: list[Box] = field(default_factory=list)
    floor_color: tuple[int, int, int] = (150, 120, 90)
    wall_color: tuple[int, int, int] = (190, 190, 180)

    def meshes(self) -> list[tuple[object, tuple[int, int, int], int]]:
        """(mesh, base colour, id) — id 0 floor, 1 walls/ceiling, k+2 box k."""
        import open3d as o3d

        W, D, H = self.size
        out: list[tuple[object, tuple[int, int, int], int]] = []
        floor = o3d.geometry.TriangleMesh.create_box(W, D, 0.01)
        floor.translate((-W / 2, -D / 2, -0.01))
        out.append((floor, self.floor_color, 0))
        shell = o3d.geometry.TriangleMesh()
        for (sx, sy, sz), (tx, ty, tz) in [
            ((W, 0.01, H), (-W / 2, D / 2, 0)),
            ((W, 0.01, H), (-W / 2, -D / 2 - 0.01, 0)),
            ((0.01, D, H), (W / 2, -D / 2, 0)),
            ((0.01, D, H), (-W / 2 - 0.01, -D / 2, 0)),
            ((W, D, 0.01), (-W / 2, -D / 2, H)),
        ]:
            b = o3d.geometry.TriangleMesh.create_box(sx, sy, sz)
            b.translate((tx, ty, tz))
            shell += b
        out.append((shell, self.wall_color, 1))
        for k, box in enumerate(self.boxes):
            out.append((box.mesh(), box.color, k + 2))
        return out


def look_at(eye: NDArray, target: NDArray, up: NDArray = np.array([0.0, 0.0, 1.0])) -> Pose:
    """Camera-to-map pose with OpenCV axes looking from ``eye`` at ``target``."""
    eye = np.asarray(eye, float)
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return Pose(np.stack([x, y, z], axis=1), eye)


def _texture(points: NDArray, base: NDArray) -> NDArray:
    """Deterministic high-frequency pattern modulating the base colour."""
    p = points * 9.0
    n = (np.sin(p[:, 0] * 3.1 + np.cos(p[:, 1] * 2.3)) + np.sin(p[:, 1] * 4.7 + p[:, 2] * 1.9)
         + np.cos(p[:, 2] * 5.3 + p[:, 0] * 1.3))
    h = np.floor(p * 2.0).astype(np.int64)
    hashv = ((h[:, 0] * 73856093) ^ (h[:, 1] * 19349663) ^ (h[:, 2] * 83492791)) % 97 / 97.0
    f = 0.55 + 0.15 * n / 3.0 + 0.35 * hashv
    return np.clip(base[None, :] * f[:, None] * 1.1, 0, 255)


@dataclass
class Render:
    rgb: NDArray[np.uint8]
    depth: NDArray[np.float32]  # z-depth, 0 = no hit
    ids: NDArray[np.int32]  # surface id per pixel (-1 none, 0 floor, 1 walls, k+2 box k)


def render(room: Room, pose: Pose, K: Intrinsics) -> Render:
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    geo_ids: dict[int, tuple[NDArray, int]] = {}
    for mesh, color, sid in room.meshes():
        gid = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        geo_ids[int(gid)] = (np.asarray(color, float), sid)
    u, v = np.meshgrid(np.arange(K.width), np.arange(K.height))
    dirs_cam = np.stack([(u - K.cx) / K.fx, (v - K.cy) / K.fy, np.ones_like(u, float)], -1)
    dirs = dirs_cam.reshape(-1, 3) @ pose.R.T
    origins = np.broadcast_to(pose.t, dirs.shape)
    rays = np.concatenate([origins, dirs], 1).astype(np.float32)
    ans = scene.cast_rays(o3d.core.Tensor(rays))
    t = ans["t_hit"].numpy()
    gid = ans["geometry_ids"].numpy()
    hit = np.isfinite(t)
    depth = np.where(hit, t, 0).astype(np.float32)  # dirs have unit z in camera frame
    pts = origins + dirs * np.where(hit, t, 0)[:, None]
    rgb = np.zeros((len(t), 3))
    ids = np.full(len(t), -1, np.int32)
    for g, (base, sid) in geo_ids.items():
        m = hit & (gid == g)
        if m.any():
            rgb[m] = _texture(pts[m], base)
            ids[m] = sid
    return Render(
        rgb=rgb.reshape(K.height, K.width, 3).astype(np.uint8),
        depth=depth.reshape(K.height, K.width),
        ids=ids.reshape(K.height, K.width),
    )


def orbit_poses(n: int, radius: float = 1.6, height: float = 1.4, target_z: float = 0.5,
                start: float = 0.0, span: float = 2 * np.pi) -> list[Pose]:
    out = []
    for k in range(n):
        a = start + span * k / max(n, 1)
        eye = np.array([radius * np.cos(a), radius * np.sin(a), height])
        out.append(look_at(eye, np.array([0.0, 0.0, target_z])))
    return out


def default_room() -> Room:
    return Room(boxes=[
        Box(np.array([1.2, 0.8, 0.4]), np.array([0.9, 0.5, 0.8]), 0.3, (220, 40, 40), "cabinet"),
        Box(np.array([-1.3, -0.6, 0.25]), np.array([0.6, 0.6, 0.5]), -0.4, (40, 180, 60),
            "box"),
        Box(np.array([0.2, -1.5, 0.45]), np.array([1.6, 0.4, 0.9]), 0.0, (50, 70, 210), "sofa"),
    ])


def gravity_rotation(up_cam: NDArray) -> NDArray:
    """Rotation taking a camera-frame up vector onto map +z (helper for tests)."""
    return rotation_between(up_cam, np.array([0.0, 0.0, 1.0]))
