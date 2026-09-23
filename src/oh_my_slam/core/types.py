"""Small value types shared by every package.

Conventions (C9): camera frames use OpenCV axes (x right, y down, z forward); the map frame is
right-handed, metric and gravity-aligned with z up. A :class:`Pose` is always camera-to-parent
(``T_parent_cam``): it maps camera coordinates into the parent frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

IntrinsicsSource = Literal["exif", "model", "colmap", "given"]

FloatArray = NDArray[np.floating[Any]]


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics in pixels for an image of ``width`` x ``height``."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    source: IntrinsicsSource = "model"

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
        )

    def with_source(self, source: IntrinsicsSource) -> Intrinsics:
        return Intrinsics(self.fx, self.fy, self.cx, self.cy, self.width, self.height, source)

    @property
    def fov_x_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan(self.width / (2.0 * self.fx))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
            "source": self.source,
        }

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


@dataclass(frozen=True)
class FrameRef:
    """One input frame: where it came from and when it was captured (video) if known."""

    index: int
    name: str
    image_path: Path
    timestamp: float | None = None
