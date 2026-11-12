"""A textured box room with a few boxes in it, ray-cast analytically in numpy (no Open3D): RGB
images that SIFT matches like a real scene's, and exact z-depth, for any camera pose. For the
prior-first registration (``mapping.anchored``) and its service path.

Map frame: z up, metres, floor at z = 0. Camera poses ``T_map_cam`` (4x4, optical frame x right,
y down, z forward)."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from oh_my_slam.core.types import Intrinsics

Array = NDArray[np.float64]

K_ROBOT = Intrinsics(270.0, 270.0, 320.0, 240.0, 640, 480)


@dataclass
class Room:
    half: tuple[float, float] = (3.0, 2.5)  # half-width x, half-depth y
    height: float = 2.6
    # boxes: (min corner, max corner)
    boxes: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = field(
        default_factory=lambda: [((1.2, 0.6, 0.0), (2.0, 1.4, 0.8)),
                                 ((-2.2, -1.8, 0.0), (-1.4, -1.2, 1.2)),
                                 ((-0.5, 1.7, 0.0), (0.7, 2.5, 0.75)),
                                 ((1.0, -2.5, 0.0), (2.2, -1.9, 1.6))])
    # the floor's texture: ``texture`` as the walls', or ``tiles`` (identical, 4-fold symmetric
    # square tiles of ``tile`` m: repetitive content whose largest epipolar consensus between
    # frames turned by about 90 deg is a false geometry, map.md §1 "Measurable pair")
    floor: str = "texture"
    tile: float = 0.30

    def cast(self, C: Array, d: Array) -> Array:
        """Distance along rays ``d`` (N x 3, any length: the parameter t) from ``C``."""
        hx, hy = self.half
        t = np.full(len(d), np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            for axis, lo, hi in ((0, -hx, hx), (1, -hy, hy), (2, 0.0, self.height)):
                for plane in (lo, hi):
                    tt = (plane - C[axis]) / d[:, axis]
                    t = np.where((tt > 1e-6) & (tt < t), tt, t)
            for lo_c, hi_c in self.boxes:
                lo_a, hi_a = np.asarray(lo_c), np.asarray(hi_c)
                t1 = (lo_a - C) / d
                t2 = (hi_a - C) / d
                tn = np.nanmax(np.minimum(t1, t2), axis=1)
                tf = np.nanmin(np.maximum(t1, t2), axis=1)
                hit = (tn <= tf) & (tn > 1e-6)
                t = np.where(hit & (tn < t), tn, t)
        return t


def texture(p: Array) -> Array:
    """A deterministic pattern of the surface point ``p`` (N x 3): random blocks and waves."""
    q = p * 9.0
    n = (np.sin(q[:, 0] * 3.1 + np.cos(q[:, 1] * 2.3)) + np.sin(q[:, 1] * 4.7 + q[:, 2] * 1.9)
         + np.cos(q[:, 2] * 5.3 + q[:, 0] * 1.3))
    h = np.floor(p * 18.0).astype(np.int64)
    hv = ((h[:, 0] * 73856093) ^ (h[:, 1] * 19349663) ^ (h[:, 2] * 83492791)) % 97 / 97.0
    g = np.floor(p * 4.0).astype(np.int64)
    base = ((g[:, 0] * 31 + g[:, 1] * 17 + g[:, 2] * 7) % 5) / 5.0
    v = 0.35 + 0.12 * n / 3.0 + 0.3 * hv + 0.25 * base
    tint = np.stack([0.9 + 0.1 * np.sin(g[:, 0] + g[:, 2]), 0.85 + 0.1 * np.cos(g[:, 1]),
                     0.8 + 0.15 * np.sin(g[:, 1] * 2 + g[:, 0])], 1)
    out: Array = np.clip(v[:, None] * tint * 255.0, 0, 255)
    return out


def tiles(p: Array, tile: float) -> Array:
    """The colour of floor points ``p`` (N x 3) on a lattice of identical square tiles, each
    4-fold symmetric about its centre: rings, a diagonal star and dark grout."""
    f = p[:, :2] / tile
    f = f - np.floor(f) - 0.5
    ax, ay = np.abs(f[:, 0]), np.abs(f[:, 1])
    ring, diag = np.maximum(ax, ay), np.minimum(ax, ay)
    v = 0.55 + 0.25 * np.cos(ring * 2 * np.pi * 3) + 0.15 * np.cos(diag * 2 * np.pi * 4)
    v = np.where(ring > 0.46, 0.15, v)
    v = np.where((np.abs(ax - ay) < 0.04) & (ring < 0.35), 0.9, v)
    out: Array = np.clip(v[:, None] * np.array([0.9, 0.8, 0.7]) * 255.0, 0, 255)
    return out


def render(room: Room, T: Array, K: Intrinsics = K_ROBOT, supersample: int = 2
           ) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """RGB (antialiased by ``supersample``) and z-depth of the camera ``T_map_cam``."""
    s = supersample
    v, u = np.mgrid[0:K.height * s, 0:K.width * s]
    uu, vv = (u + 0.5) / s - 0.5, (v + 0.5) / s - 0.5
    rays = np.stack([(uu - K.cx) / K.fx, (vv - K.cy) / K.fy, np.ones_like(uu, float)],
                    -1).reshape(-1, 3)
    R, C = T[:3, :3], T[:3, 3]
    d = rays @ R.T
    t = room.cast(C, d)
    p = C + d * t[:, None]
    col = texture(p)
    if room.floor == "tiles":
        on = p[:, 2] < 1e-4
        col[on] = tiles(p[on], room.tile)
    rgb = col.reshape(K.height * s, K.width * s, 3)
    rgb = rgb.reshape(K.height, s, K.width, s, 3).mean(axis=(1, 3))
    z = t.reshape(K.height * s, K.width * s)[s // 2::s, s // 2::s]  # unit-z rays: t is z-depth
    return rgb.astype(np.uint8), z.astype(np.float32)


def look(yaw_deg: float, centre: tuple[float, float, float], pitch_deg: float = 0.0) -> Array:
    """A camera at ``centre`` looking along ``yaw`` (CCW from +x), pitched up by ``pitch``."""
    a, p = math.radians(yaw_deg), math.radians(pitch_deg)
    fwd = np.array([math.cos(a) * math.cos(p), math.sin(a) * math.cos(p), math.sin(p)])
    right = np.array([math.sin(a), -math.cos(a), 0.0])
    down = np.cross(fwd, right)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([right, down, fwd])
    T[:3, 3] = centre
    return T


def base_pose(yaw_deg: float, x: float = 0.0, y: float = 0.0, height: float = 0.5,
              offset: float = 0.1, pitch_deg: float = -10.0) -> Array:
    """A wheeled base's camera: ``offset`` ahead of the base centre (x, y), at ``height``."""
    a = math.radians(yaw_deg)
    return look(yaw_deg, (x + offset * math.cos(a), y + offset * math.sin(a), height), pitch_deg)


def jpeg(rgb: NDArray[np.uint8], quality: int = 92) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def png_depth(z: NDArray[np.float32], scale_m: float = 0.001) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.clip(np.round(z / scale_m), 0, 65535).astype(np.uint16)).save(buf, "PNG")
    return buf.getvalue()
