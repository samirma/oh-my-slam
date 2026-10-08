"""Small value types shared by every package.

Conventions (C9): camera frames use OpenCV axes (x right, y down, z forward); the map frame is
right-handed, metric and gravity-aligned with z up. A :class:`Pose` is always camera-to-parent
(``T_parent_cam``): it maps camera coordinates into the parent frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

IntrinsicsSource = Literal["exif", "model", "colmap", "given"]

FloatArray = NDArray[np.floating[Any]]


@dataclass(frozen=True)
class Intrinsics:
    """Intrinsics in pixels for an image of ``width`` x ``height``: a pinhole, and the radial
    distortion ``k`` of COLMAP's division model (``SIMPLE_DIVISION``; 0: none). A pixel ``d``
    (from the principal point, over the focal length) is the ray ``(d / (1 + k |d|²), 1)``, a ray
    ``(u, 1)`` is seen at ``2 u / (1 + sqrt(1 - 4 k |u|²))``; ``k`` is the same at any image
    size."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    source: IntrinsicsSource = "model"
    k: float = 0.0

    def K(self) -> NDArray[np.float64]:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )

    def resized(self, width: int, height: int) -> Intrinsics:
        """Intrinsics for the same camera after resizing the image to ``width`` x ``height``."""
        sx, sy = width / self.width, height / self.height
        return Intrinsics(
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            width=width,
            height=height,
            source=self.source,
            k=self.k,
        )

    def with_source(self, source: IntrinsicsSource) -> Intrinsics:
        return Intrinsics(self.fx, self.fy, self.cx, self.cy, self.width, self.height, source,
                          self.k)

    def pinhole(self) -> Intrinsics:
        """The camera of the undistorted image: a pinhole of the same size and principal point
        whose focal length (``undistorted_scale`` x the lens's) holds the whole lens; the camera
        itself without distortion."""
        s = self.undistorted_scale
        return Intrinsics(self.fx * s, self.fy * s, self.cx, self.cy, self.width, self.height,
                          self.source)

    @property
    def undistorted_scale(self) -> float:
        """Focal length of ``pinhole`` over the lens's: barrel distortion (``k`` < 0) shows a ray
        ``(u, 1)`` at ``d = (1 + k |d|²) u``, so the pinhole that keeps the image's corners — its
        farthest pixels from the principal point — needs ``1 + k`` times their largest squared
        distance (at least 0.1: a lens past 90°); 1 without distortion or with pincushion
        distortion, whose rays all lie inside the image's."""
        reach = max((x / self.fx) ** 2 + (y / self.fy) ** 2
                    for x in (self.cx, self.width - self.cx) for y in (self.cy, self.height - self.cy))
        return float(np.clip(1.0 + self.k * reach, 0.1, 1.0))

    def rays(self, uv: FloatArray) -> NDArray[np.float64]:
        """The rays ``(x / z, y / z)`` the image pixels ``uv`` (N, 2) see."""
        d = (np.asarray(uv, np.float64).reshape(-1, 2) - [self.cx, self.cy]) / [self.fx, self.fy]
        return np.asarray(d / (1.0 + self.k * np.sum(d * d, axis=1, keepdims=True)))

    def pinhole_pixels(self, uv: FloatArray) -> NDArray[np.float64]:
        """Where the undistorted image (``pinhole``) shows the image pixels ``uv`` (N, 2)."""
        uv = np.asarray(uv, np.float64).reshape(-1, 2)
        if not self.k:
            return uv
        p = self.pinhole()
        return np.asarray(self.rays(uv) * [p.fx, p.fy] + [self.cx, self.cy])

    def image_pixels(self, uv: FloatArray) -> NDArray[np.float64]:
        """Where the image shows what the undistorted image (``pinhole``) shows at ``uv`` (N, 2);
        nan for a ray beyond the division model's reach."""
        uv = np.asarray(uv, np.float64).reshape(-1, 2)
        if not self.k:
            return uv
        p = self.pinhole()
        u = (uv - [self.cx, self.cy]) / [p.fx, p.fy]
        disc = 1.0 - 4.0 * self.k * np.sum(u * u, axis=1, keepdims=True)
        with np.errstate(invalid="ignore"):
            d = 2.0 * u / (1.0 + np.sqrt(disc))
        return np.asarray(np.where(disc > 0, d * [self.fx, self.fy] + [self.cx, self.cy],
                                   np.nan))

    def opencv_distortion(self) -> list[float]:
        """OpenCV distortion coefficients (k1, k2, p1, p2, k3, k4, k5, k6: its rational model)
        that bend the rays (``rays``) as the division model does over the image (least squares
        over the radii up to the image corners); without distortion, OpenCV's plain model at zero
        (k1, k2, p1, p2, k3)."""
        if not self.k:
            return [0.0] * 5
        half = np.hypot(max(self.cx, self.width - self.cx), max(self.cy, self.height - self.cy))
        d = np.linspace(0.0, half / min(self.fx, self.fy), 400)[1:]
        s = (d / (1.0 + self.k * d * d)) ** 2  # the undistorted radius², over which OpenCV fits
        g = 2.0 / (1.0 + np.sqrt(1.0 - 4.0 * self.k * s))  # distorted / undistorted radius
        A = np.column_stack([s, s**2, s**3, -g * s, -g * s**2, -g * s**3])
        c = np.linalg.lstsq(A, g - 1.0, rcond=None)[0]
        return [float(c[0]), float(c[1]), 0.0, 0.0, float(c[2]), float(c[3]), float(c[4]),
                float(c[5])]

    @property
    def fov_x_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan(self.width / (2.0 * self.fx))))

    def to_dict(self) -> dict[str, Any]:
        d = {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
            "source": self.source,
        }
        if self.k:
            d["k"] = self.k
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Intrinsics:
        return Intrinsics(
            fx=float(d["fx"]),
            fy=float(d["fy"]),
            cx=float(d["cx"]),
            cy=float(d["cy"]),
            width=int(d["width"]),
            height=int(d["height"]),
            source=d.get("source", "model"),
            k=float(d.get("k", 0.0)),
        )


@dataclass(frozen=True)
class Pose:
    """Rigid transform ``T_parent_cam``: ``x_parent = R @ x_cam + t``."""

    R: NDArray[np.float64] = field(default_factory=lambda: np.eye(3))
    t: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))

    @staticmethod
    def identity() -> Pose:
        return Pose(np.eye(3), np.zeros(3))

    @staticmethod
    def from_matrix(T: FloatArray) -> Pose:
        M = np.asarray(T, dtype=np.float64)
        return Pose(np.array(M[:3, :3], dtype=np.float64), np.array(M[:3, 3], dtype=np.float64))

    def matrix(self) -> NDArray[np.float64]:
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    def inverse(self) -> Pose:
        Rt = self.R.T
        return Pose(Rt, -Rt @ self.t)

    def compose(self, other: Pose) -> Pose:
        """``self @ other`` (apply ``other`` first)."""
        return Pose(self.R @ other.R, self.R @ other.t + self.t)

    def apply(self, points: FloatArray) -> NDArray[np.float64]:
        pts = np.asarray(points, dtype=np.float64)
        return pts @ self.R.T + self.t

    @property
    def center(self) -> NDArray[np.float64]:
        """Camera centre in the parent frame."""
        return self.t.copy()

    def to_dict(self) -> dict[str, Any]:
        from oh_my_slam.core.geometry import rot_to_quat

        return {"quaternion_xyzw": rot_to_quat(self.R).tolist(), "translation": self.t.tolist()}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Pose:
        from oh_my_slam.core.geometry import quat_to_rot

        return Pose(
            quat_to_rot(np.asarray(d["quaternion_xyzw"], dtype=np.float64)),
            np.asarray(d["translation"], dtype=np.float64),
        )
