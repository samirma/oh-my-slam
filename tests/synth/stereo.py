"""AiNex-like motion-stereo pairs (``reports/motion_stereo_eval.md`` §2, its ``scripts/synth.py``):
a 640 x 480 head camera with fx = fy = 270, its optical centre 0.33 m above the floor, pitched
down, in a room of axis-aligned boxes with near obstacles (a box 0.35-0.55 m ahead, a stool
0.8-1.0 m, a table on thin legs 1.3-2.1 m, a crate, walls). Rendered analytically (ray-box
slabs, no Open3D) with exact z-depth and surface ids, and the evaluation's texture: a smooth
multi-octave sum of oriented sinusoids (wavelengths 30 cm to 1 cm) around each surface's
colour, which SIFT matches everywhere (the blocky hash texture of ``tests.synth.scene`` is flat
inside its blocks). Pixel centres at integers (OpenCV's, the analysis's ``rays``)."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from tests.synth.rooms import camera

W, H, F = 640, 480, 270.0
CX, CY = 320.0, 240.0
HEAD_M = 0.33
HUMANOID = {"fx": F, "fy": F, "cx": CX, "cy": CY, "width": W, "height": H}
K = np.array([[F, 0.0, CX], [0.0, F, CY], [0.0, 0.0, 1.0]])
FLOOR, WALLS = 0, 1  # surface ids; the obstacles are 2 and up

_PH = np.random.default_rng(12345).uniform(0, 2 * np.pi, (6, 3, 2))
_DIR = np.random.default_rng(54321).normal(size=(6, 3, 3))


@dataclass
class Room:
    """Boxes ``(lo, hi, colour, surface id)``; ``flat`` ids take their colour alone."""

    boxes: list[tuple[NDArray[np.float64], NDArray[np.float64], tuple[int, int, int], int]] = \
        field(default_factory=list)
    flat: set[int] = field(default_factory=set)

    def add(self, lo: tuple[float, float, float], hi: tuple[float, float, float],
            colour: tuple[int, int, int], sid: int) -> Room:
        self.boxes.append((np.array(lo, np.float64), np.array(hi, np.float64), colour, sid))
        return self


def ainex_room(flat_floor: bool = False) -> Room:
    """The humanoid's room: the camera stands at the origin looking along +x."""
    r = Room(flat={FLOOR} if flat_floor else set())
    r.add((-4.0, -4.0, -0.05), (6.0, 4.0, 0.0), (140, 110, 80), FLOOR)
    for lo, hi in (((3.0, -4.0, 0.0), (3.1, 4.0, 2.6)), ((-4.0, 2.5, 0.0), (6.0, 2.6, 2.6)),
                   ((-4.0, -2.6, 0.0), (6.0, -2.5, 2.6))):
        r.add(lo, hi, (190, 185, 170), WALLS)
    r.add((0.35, -0.05, 0.0), (0.55, 0.2, 0.2), (200, 60, 50), 2)  # a box underfoot
    r.add((0.8, -0.45, 0.0), (1.0, -0.25, 0.45), (60, 150, 70), 3)  # a stool
    r.add((1.3, 0.2, 0.70), (2.1, 0.8, 0.74), (120, 80, 160), 4)  # a table top ...
    for x, y in ((1.3, 0.2), (2.06, 0.2), (1.3, 0.76), (2.06, 0.76)):
        r.add((x, y, 0.0), (x + 0.04, y + 0.04, 0.70), (120, 80, 160), 4)  # ... on thin legs
    r.add((1.6, -0.95, 0.0), (2.0, -0.55, 0.6), (70, 90, 200), 5)  # a crate
    return r


def far_wall() -> Room:
    """A textured wall 12 m ahead and nothing else: every match has little parallax."""
    return Room().add((12.0, -40.0, -30.0), (12.5, 40.0, 30.0), (150, 150, 150), WALLS)


def texture(p: NDArray[np.float64], base: NDArray[np.float64], sid: int) -> NDArray[np.float64]:
    """The evaluation's texture: per surface a different phase, ±25 % around its colour."""
    f = np.zeros((len(p), 3))
    amp = 0.0
    for o, lam in enumerate((0.30, 0.13, 0.06, 0.03, 0.015, 0.01)):
        a = 1.0 / (1 + o) ** 0.6
        for j in range(3):
            d = _DIR[o, j] / np.linalg.norm(_DIR[o, j])
            ph = _PH[o, j, 0] + sid * 1.7
            f[:, j] += a * np.sin(2 * np.pi * (p @ d) / lam + ph
                                  + np.sin(2 * np.pi * (p @ d[::-1]) / (3 * lam) + _PH[o, j, 1]))
        amp += a
    g = f.mean(1, keepdims=True) * 0.8 + f * 0.2
    out: NDArray[np.float64] = np.clip(base[None, :] * (1 + 0.25 * g / amp * 2.2), 0, 255)
    return out


@dataclass
class Shot:
    """A rendered view: its JPEG, the RGB the JPEG decodes to, the exact z-depth (0: nothing)
    and surface ids (-1: nothing), and the camera's pose (camera-to-room, optical axes)."""

    jpeg: bytes
    rgb: NDArray[np.uint8]
    depth: NDArray[np.float32]
    ids: NDArray[np.int32]
    T: NDArray[np.float64]


def render(room: Room, T: NDArray[np.float64], seed: int = 0) -> Shot:
    """The view of camera ``T``: the nearest box along each pixel's ray, textured, blurred by
    0.5 px, noise sigma 2.5, JPEG 92."""
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    rays = np.stack([(u - CX) / F, (v - CY) / F, np.ones_like(u)], -1).reshape(-1, 3)
    d = rays @ T[:3, :3].T
    C = T[:3, 3]
    best = np.full(len(d), np.inf)
    sid = np.full(len(d), -1, np.int32)
    which = np.full(len(d), -1, np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        for k, (lo, hi, _, s) in enumerate(room.boxes):
            t1, t2 = (lo - C) / d, (hi - C) / d
            tmin = np.nanmax(np.minimum(t1, t2), axis=1)
            tmax = np.nanmin(np.maximum(t1, t2), axis=1)
            hit = (tmax >= np.maximum(tmin, 0)) & (tmin > 1e-6) & (tmin < best)
            best = np.where(hit, tmin, best)
            sid = np.where(hit, s, sid)
            which = np.where(hit, k, which)
    seen = np.isfinite(best)
    pts = C + d * np.where(seen, best, 0.0)[:, None]
    rgb = np.zeros((len(d), 3))
    for k, (_, _, colour, s) in enumerate(room.boxes):
        m = which == k
        if m.any():
            base = np.asarray(colour, np.float64)
            rgb[m] = base if s in room.flat else texture(pts[m], base, s)
    img = cv2.GaussianBlur(rgb.reshape(H, W, 3).astype(np.float32), (0, 0), 0.5)
    img += np.random.default_rng(seed).normal(0, 2.5, img.shape)
    buf = io.BytesIO()
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(buf, format="JPEG", quality=92)
    jpeg = buf.getvalue()
    return Shot(jpeg, np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB")),
                np.where(seen, best, 0.0).reshape(H, W).astype(np.float32),
                sid.reshape(H, W), T)


def head(x: float = 0.0, y: float = 0.0, yaw_deg: float = 0.0, pitch_deg: float = -25.0,
         z: float = HEAD_M) -> NDArray[np.float64]:
    """The head camera at (x, y, z) looking along ``yaw`` (CCW from +x), pitched down."""
    return camera(x, y, z, yaw_deg, pitch_deg)


def panned(pan_deg: float, neck_m: float = 0.057, pitch_deg: float = -25.0
           ) -> NDArray[np.float64]:
    """The head panned by ``pan_deg`` about a vertical axis ``neck_m`` behind the optical
    centre of ``head()`` (the client's simulator head: 5.7 cm)."""
    a = math.radians(pan_deg)
    return camera(-neck_m + neck_m * math.cos(a), neck_m * math.sin(a), HEAD_M, pan_deg,
                  pitch_deg)


def relative(T_ref: NDArray[np.float64], T_other: NDArray[np.float64]) -> NDArray[np.float64]:
    """``T_ref_frame``: the other camera's pose in the reference's optical frame."""
    out: NDArray[np.float64] = np.linalg.inv(T_ref) @ T_other
    return out


def model_depth(z: NDArray[np.float32]) -> NDArray[np.float32]:
    """A monocular model's depth of a view: the truth in the model's own metres (0.8 x), its
    shape stretched in ln z (z^0.9) and bent by a smooth ±6 % across the image, as MoGe's is."""
    h, w = z.shape
    yy, xx = np.mgrid[0:h, 0:w]
    bend = 1.0 + 0.06 * np.sin(xx / 90.0) * np.cos(yy / 120.0)
    out = np.where(z > 0, 0.8 * np.power(np.maximum(z, 1e-6), 0.9) * bend, 0.0)
    return out.astype(np.float32)
