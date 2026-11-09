"""The requests' cameras (``Camera``) and the pinhole the mapper sees.

A camera has intrinsics ``fx, fy, cx, cy`` for its image size, optional OpenCV distortion, an
optional ignore mask (non-zero: the robot's own body), the metres per unit of its depth PNGs and
an optional height above the floor. The mapper models a pinhole with one focal length and the
principal point at the centre. A frame whose camera is not already that pinhole is therefore
resampled once to the camera's *canonical pinhole*: same size, focal length ``sqrt(fx fy)``,
principal point at the image centre, no distortion, no rotation. The optical frame is unchanged,
so the poses the mapper estimates are the camera's own. Its depth PNG and ignore mask are
resampled with it (nearest neighbour). Ignored pixels are filled from their surroundings
(``cv2.inpaint``) so that no feature is extracted on the robot's body."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from oh_my_slam.core.types import Intrinsics
from oh_my_slam.service.errors import invalid

DEFAULT_DEPTH_SCALE_M = 0.001
INPAINT_RADIUS_PX = 5


@dataclass(frozen=True)
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: tuple[float, ...] = ()
    depth_scale_m: float = DEFAULT_DEPTH_SCALE_M
    height_above_floor_m: float | None = None
    height_sigma_m: float | None = None

    @staticmethod
    def from_json(d: dict[str, Any]) -> Camera:
        dist = d.get("distortion") or {}
        return Camera(float(d["fx"]), float(d["fy"]), float(d["cx"]), float(d["cy"]),
                      int(d["width"]), int(d["height"]),
                      tuple(float(c) for c in dist.get("coefficients", ())),
                      float(d.get("depth_scale_m", DEFAULT_DEPTH_SCALE_M)),
                      d.get("height_above_floor_m"), d.get("height_sigma_m"))

    @property
    def focal(self) -> float:
        return math.sqrt(self.fx * self.fy)

    @property
    def is_canonical(self) -> bool:
        """Whether the camera already is its canonical pinhole (no resampling)."""
        return (not any(self.distortion) and abs(self.fx - self.fy) < 1e-6 * self.fx
                and abs(self.cx - self.width / 2.0) < 1e-3 and abs(self.cy - self.height / 2.0) < 1e-3)

    def canonical(self) -> Intrinsics:
        """The canonical pinhole's intrinsics (``given``)."""
        f = self.focal
        return Intrinsics(f, f, self.width / 2.0, self.height / 2.0, self.width, self.height,
                          "given")

    def _maps(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        K = np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])
        dist = np.array(self.distortion or (0.0, 0.0, 0.0, 0.0), np.float64)
        m1, m2 = cv2.initUndistortRectifyMap(K, dist, None, self.canonical().K(),
                                             (self.width, self.height), cv2.CV_32FC1)
        return np.asarray(m1, np.float32), np.asarray(m2, np.float32)

    def rectify(self, img: NDArray[Any], nearest: bool = False) -> NDArray[Any]:
        """``img`` (this camera's size) as its canonical pinhole sees it."""
        if self.is_canonical:
            return img
        m1, m2 = self._maps()
        out: NDArray[Any] = cv2.remap(img, m1, m2,
                                      cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE if not nearest
                                      else cv2.BORDER_CONSTANT, borderValue=0)
        return out


def same_camera(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Whether two ``Camera`` bodies describe the same camera (the ignore mask's part name may
    differ: its content is compared by the caller)."""
    def norm(d: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in d.items() if k != "ignore_mask" and v is not None}
        out.setdefault("depth_scale_m", DEFAULT_DEPTH_SCALE_M)
        if not (out.get("distortion") or {}).get("coefficients") or not any(
                out["distortion"]["coefficients"]):
            out.pop("distortion", None)
        return out
    return norm(a) == norm(b)


def decode_rgb(data: bytes, field: str) -> NDArray[np.uint8]:
    """An 8-bit JPEG or PNG as RGB pixels, as stored (EXIF orientation is not applied: the
    camera's intrinsics describe the stored pixels)."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in ("JPEG", "PNG"):
                raise invalid(field, f"{field}: a JPEG or PNG image is expected, got {im.format}")
            return np.asarray(im.convert("RGB"))
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise invalid(field, f"{field}: cannot decode the image ({exc})") from exc


def decode_png16(data: bytes, field: str) -> NDArray[np.uint16]:
    """A 16-bit single-channel PNG (depth)."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt, mode = im.format, im.mode
            arr = np.asarray(im)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise invalid(field, f"{field}: cannot decode the PNG ({exc})") from exc
    if fmt != "PNG" or arr.ndim != 2 or mode not in ("I;16", "I;16B", "I;16L", "I"):
        raise invalid(field, f"{field}: a 16-bit single-channel PNG is expected "
                      f"(got {fmt} {mode})")
    return arr.astype(np.uint16)


def decode_mask(data: bytes, field: str) -> NDArray[np.bool_]:
    """An 8-bit PNG mask: non-zero pixels are ignored."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            arr = np.asarray(im.convert("L"))
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise invalid(field, f"{field}: cannot decode the mask ({exc})") from exc
    if fmt != "PNG":
        raise invalid(field, f"{field}: an 8-bit PNG mask is expected (got {fmt})")
    out: NDArray[np.bool_] = arr > 0
    return out


def mapper_image(rgb: NDArray[np.uint8], cam: Camera, mask: NDArray[np.bool_] | None
                 ) -> NDArray[np.uint8]:
    """The frame as the mapper gets it: rectified to the canonical pinhole, the ignored pixels
    filled from their surroundings."""
    out = cam.rectify(rgb)
    if mask is not None and mask.any():
        m = cam.rectify(mask.astype(np.uint8), nearest=True)
        out = cv2.inpaint(np.ascontiguousarray(out), m, INPAINT_RADIUS_PX, cv2.INPAINT_TELEA)
    return out


def mapper_depth(depth: NDArray[np.uint16], cam: Camera, mask: NDArray[np.bool_] | None
                 ) -> NDArray[np.float32]:
    """A depth PNG in metres on the canonical pinhole (0: no value; ignored pixels have none)."""
    d = cam.rectify(depth, nearest=True).astype(np.float32) * cam.depth_scale_m
    if mask is not None:
        d[cam.rectify(mask.astype(np.uint8), nearest=True) > 0] = 0.0
    out: NDArray[np.float32] = d
    return out
