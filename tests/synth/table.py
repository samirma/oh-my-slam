"""A table scene for the objects of an anchored map (``mapping.anchored_objects``): the textured
room of ``tests.synth.room`` with a table top, items on it and a hollow container, ray-cast with
the index of the box each pixel sees, so detections are exact masks.

Map frame: z up, metres, floor at z = 0, the table top at ``TOP`` m."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.types import Intrinsics
from tests.synth.room import Room, look, texture

K_TABLE = Intrinsics(400.0, 400.0, 320.0, 240.0, 640, 480)
TOP = 0.75
Bx = tuple[tuple[float, float, float], tuple[float, float, float]]


@dataclass
class Item:
    label: str
    boxes: list[Bx]
    rgb: tuple[int, int, int] | None = None  # a solid colour, else the room's texture


def container(x0: float, y0: float, z0: float, side: float, height: float, wall: float = 0.01
              ) -> list[Bx]:
    """A hollow open box: a bottom and four walls."""
    x1, y1, z1 = x0 + side, y0 + side, z0 + height
    return [((x0, y0, z0), (x1, y1, z0 + wall)),
            ((x0, y0, z0), (x0 + wall, y1, z1)), ((x1 - wall, y0, z0), (x1, y1, z1)),
            ((x0, y0, z0), (x1, y0 + wall, z1)), ((x0, y1 - wall, z0), (x1, y1, z1))]


def default_items() -> list[Item]:
    return [
        Item("table", [((0.0, 0.0, TOP - 0.03), (1.2, 0.8, TOP)),
                       ((0.0, 0.0, 0.0), (0.05, 0.05, TOP - 0.03)),
                       ((1.15, 0.0, 0.0), (1.2, 0.05, TOP - 0.03)),
                       ((0.0, 0.75, 0.0), (0.05, 0.8, TOP - 0.03)),
                       ((1.15, 0.75, 0.0), (1.2, 0.8, TOP - 0.03))]),
        Item("box", [((0.15, 0.15, TOP), (0.35, 0.3, TOP + 0.12))], (200, 40, 40)),
        Item("basket", container(0.65, 0.3, TOP, 0.3, 0.12), (40, 60, 200)),
        Item("apple", [((0.76, 0.41, TOP + 0.01), (0.84, 0.49, TOP + 0.08))], (60, 190, 50)),
    ]


@dataclass
class TableScene:
    items: list[Item] = field(default_factory=default_items)
    room: Room = field(default_factory=lambda: Room(boxes=[]))

    def boxes(self) -> list[tuple[Bx, int]]:
        return [(b, i) for i, it in enumerate(self.items) for b in it.boxes]

    def cast(self, C: NDArray[np.float64], d: NDArray[np.float64]
             ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Ray parameter and item index (-1: the room) along rays ``d`` from ``C``."""
        t = self.room.cast(C, d)
        idx = np.full(len(d), -1, np.int64)
        with np.errstate(divide="ignore", invalid="ignore"):
            for (lo_c, hi_c), i in self.boxes():
                t1 = (np.asarray(lo_c) - C) / d
                t2 = (np.asarray(hi_c) - C) / d
                tn = np.nanmax(np.minimum(t1, t2), axis=1)
                tf = np.nanmin(np.maximum(t1, t2), axis=1)
                hit = (tn <= tf) & (tn > 1e-6) & (tn < t)
                t = np.where(hit, tn, t)
                idx = np.where(hit, i, idx)
        return t, idx

    def render(self, T: NDArray[np.float64], K: Intrinsics = K_TABLE
               ) -> tuple[NDArray[np.uint8], NDArray[np.float32], NDArray[np.int64]]:
        """RGB, z-depth and the item index of every pixel of the camera ``T_map_cam``."""
        v, u = np.mgrid[0:K.height, 0:K.width]
        rays = np.stack([(u - K.cx) / K.fx, (v - K.cy) / K.fy, np.ones_like(u, float)],
                        -1).reshape(-1, 3)
        d = rays @ T[:3, :3].T
        t, idx = self.cast(T[:3, 3], d)
        p = T[:3, 3] + d * t[:, None]
        rgb = texture(p)
        for i, it in enumerate(self.items):
            if it.rgb is not None:
                rgb[idx == i] = it.rgb
        return (rgb.reshape(K.height, K.width, 3).astype(np.uint8),
                t.reshape(K.height, K.width).astype(np.float32),
                idx.reshape(K.height, K.width))


def look_at(eye: tuple[float, float, float], target: tuple[float, float, float]
            ) -> NDArray[np.float64]:
    """A camera at ``eye`` looking at ``target`` (level horizon)."""
    dx, dy, dz = (target[k] - eye[k] for k in range(3))
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
    return look(yaw, eye, pitch)
