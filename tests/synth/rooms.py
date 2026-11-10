"""Scenes of axis-aligned boxes (walls, floor, furniture) rendered analytically into z-depth, for
the navigation grid: two rooms joined by a door, cameras standing in them."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

K640 = np.array([[320.0, 0.0, 160.0], [0.0, 320.0, 120.0], [0.0, 0.0, 1.0]])  # 320x240


@dataclass
class Scene:
    boxes: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = field(default_factory=list)

    def add(self, lo: tuple[float, float, float], hi: tuple[float, float, float]) -> Scene:
        self.boxes.append((np.array(lo, np.float64), np.array(hi, np.float64)))
        return self

    def depth(self, T: NDArray[np.float64], K: NDArray[np.float64] = K640, width: int = 320,
              height: int = 240) -> NDArray[np.float32]:
        """Z-depth (0: nothing hit) of a camera ``T`` (camera-to-map, optical axes)."""
        u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
        rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
        d = rays.reshape(-1, 3) @ T[:3, :3].T
        C = T[:3, 3]
        best = np.full(len(d), np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            for lo, hi in self.boxes:
                t1 = (lo - C) / d
                t2 = (hi - C) / d
                tmin = np.nanmax(np.minimum(t1, t2), axis=1)
                tmax = np.nanmin(np.maximum(t1, t2), axis=1)
                hit = (tmax >= np.maximum(tmin, 0)) & (tmin > 1e-6)
                best = np.where(hit & (tmin < best), tmin, best)
        z = np.where(np.isfinite(best), best, 0.0)  # rays have unit z: t is the z-depth
        return z.reshape(height, width).astype(np.float32)


def camera(x: float, y: float, z: float, yaw_deg: float, pitch_deg: float = -15.0
           ) -> NDArray[np.float64]:
    """A camera at (x, y, z) looking along ``yaw`` (CCW from +x), pitched down by ``-pitch``."""
    a, p = math.radians(yaw_deg), math.radians(-pitch_deg)
    fwd = np.array([math.cos(a) * math.cos(p), math.sin(a) * math.cos(p), -math.sin(p)])
    right = np.array([math.sin(a), -math.cos(a), 0.0])
    down = np.cross(fwd, right)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([right, down, fwd])
    T[:3, 3] = [x, y, z]
    return T


WALL = 0.1
H = 2.5
DOOR = (1.5, 2.4)  # the door gap's y extent in the wall x in [4.0, 4.1]


def two_rooms(door: tuple[float, float] = DOOR, leaf: bool = False, table: bool = False,
              leaf_t: float = 0.04) -> Scene:
    """Room A (x 0-4) and room B (x 4.1-8), y 0-4, joined by a door in the wall between them."""
    s = Scene()
    s.add((-1.0, -1.0, -0.05), (9.0, 5.0, 0.0))  # floor
    s.add((-WALL, -WALL, 0.0), (8.0 + WALL, 0.0, H))  # south
    s.add((-WALL, 4.0, 0.0), (8.0 + WALL, 4.0 + WALL, H))  # north
    s.add((-WALL, 0.0, 0.0), (0.0, 4.0, H))  # west
    s.add((8.0, 0.0, 0.0), (8.0 + WALL, 4.0, H))  # east
    s.add((4.0, 0.0, 0.0), (4.1, door[0], H))  # the wall between, below the door
    s.add((4.0, door[1], 0.0), (4.1, 4.0, H))  # and above it
    s.add((4.0, door[0], 2.1), (4.1, door[1], H))  # lintel
    if leaf:  # an open leaf, hinged at the door's upper jamb, swung 90 deg into room B
        s.add((4.1, door[1] - leaf_t, 0.0), (4.1 + 0.8, door[1], 2.0))
    if table:  # a table top at 0.75 m on thin legs, in room A
        s.add((1.0, 2.5, 0.72), (2.0, 3.3, 0.76))
        for x, y in ((1.0, 2.5), (1.95, 2.5), (1.0, 3.25), (1.95, 3.25)):
            s.add((x, y, 0.0), (x + 0.05, y + 0.05, 0.72))
    return s


def tour(height: float = 1.0) -> list[NDArray[np.float64]]:
    """Cameras turning in place at two spots of each room, and looking through the door."""
    out = []
    for x, y in ((1.2, 1.2), (2.8, 2.8), (5.2, 1.2), (6.8, 2.8)):
        for yaw in range(0, 360, 30):
            out.append(camera(x, y, height, yaw))
    out.append(camera(3.0, 1.95, height, 0.0))
    out.append(camera(5.2, 1.95, height, 180.0))
    return out
