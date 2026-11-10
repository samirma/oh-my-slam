"""Per-frame image measurements behind the rejection reasons (api.md §4; thresholds in README,
"Per-input status").

Every frame of an update is measured before it reaches the mapper:

* ``features`` — SIFT keypoints on the image scaled to at most ``WORK_SIDE`` px;
* ``texture`` — the share of pixels whose intensity gradient exceeds ``TEXTURE_GRADIENT`` grey
  levels per pixel, at ``TEXTURE_SIDE`` px: a bare wall or ceiling has almost none;
* ``sharpness`` — the variance of the Laplacian at the working size over its variance at a
  quarter of it. Blur removes the fine scale and keeps the coarse one, so the ratio does not
  depend on the scene's content as the plain variance of the Laplacian does;
* the median depth, from the frame's depth PNG, or from the monocular depth model for frames
  whose features or texture are low (``suspicious``).

A frame is rejected before mapping when it is ``too_close``, ``plain``, ``blur`` or has
``few_features``, in that order of precedence: such frames registered wrongly, or warped the
map, in the field."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

WORK_SIDE = 1024
TEXTURE_SIDE = 512
TEXTURE_GRADIENT = 4.0  # grey levels per pixel (Sobel / 8)
PLAIN_TEXTURE = 0.01  # bare walls 0.0003-0.0024; the 128 reference frames >= 0.0395
BLUR_SHARPNESS = 0.035  # the reference frames >= 0.12; a Gaussian blur of >= 1.2 px < 0.035
FEW_FEATURES = 20  # the reference frames >= 27 (a bare ceiling with a door frame)
SUSPECT_FEATURES = 150  # below this (or a low texture) the median depth is measured
SUSPECT_TEXTURE = 0.05
TOO_CLOSE_M = 0.30  # median depth along the optical axis


@dataclass(frozen=True)
class Measures:
    features: int
    texture: float
    sharpness: float

    @property
    def suspicious(self) -> bool:
        """Whether the median depth is worth measuring (``too_close`` is likely)."""
        return self.features < SUSPECT_FEATURES or self.texture < SUSPECT_TEXTURE


def _scaled(gray: NDArray[np.uint8], side: int) -> NDArray[np.uint8]:
    h, w = gray.shape
    s = min(1.0, side / max(h, w))
    if s >= 1.0:
        return gray
    return np.asarray(cv2.resize(gray, (max(1, round(w * s)), max(1, round(h * s))),
                                 interpolation=cv2.INTER_AREA), np.uint8)


def measure(rgb: NDArray[np.uint8], ignore: NDArray[np.bool_] | None = None) -> Measures:
    """The measurements of an RGB frame; ``ignore`` pixels (the robot's body) take no part."""
    gray = np.asarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), np.uint8)
    work = _scaled(gray, WORK_SIDE)
    keep = None
    if ignore is not None and ignore.any():
        keep = (cv2.resize((~ignore).astype(np.uint8), (work.shape[1], work.shape[0]),
                           interpolation=cv2.INTER_NEAREST) > 0)
    sift: Any = cv2.SIFT_create()  # type: ignore[attr-defined]
    kps = sift.detect(work, None if keep is None else keep.astype(np.uint8) * 255)
    tex = _scaled(gray, TEXTURE_SIDE).astype(np.float32)
    mag = np.hypot(cv2.Sobel(tex, cv2.CV_32F, 1, 0), cv2.Sobel(tex, cv2.CV_32F, 0, 1)) / 8.0
    if keep is not None:
        k = cv2.resize(keep.astype(np.uint8), (tex.shape[1], tex.shape[0]),
                       interpolation=cv2.INTER_NEAREST) > 0
        mag = mag[k] if k.any() else mag.ravel()[:0]
    texture = float((mag > TEXTURE_GRADIENT).mean()) if mag.size else 0.0
    fine = float(cv2.Laplacian(work.astype(np.float32), cv2.CV_32F).var())
    small = cv2.resize(work, (max(1, work.shape[1] // 4), max(1, work.shape[0] // 4)),
                       interpolation=cv2.INTER_AREA)
    coarse = float(cv2.Laplacian(small.astype(np.float32), cv2.CV_32F).var())
    return Measures(len(kps), texture, fine / coarse if coarse > 1e-6 else 0.0)


def median_depth(depth_m: NDArray[np.float32]) -> float | None:
    """The median of the valid (positive, finite) depths, or None without any."""
    v = depth_m[np.isfinite(depth_m) & (depth_m > 0)]
    return float(np.median(v)) if v.size else None


def reason(m: Measures, depth_median_m: float | None) -> str | None:
    """Why a frame is rejected before mapping, or None."""
    if depth_median_m is not None and depth_median_m < TOO_CLOSE_M:
        return "too_close"
    if m.texture < PLAIN_TEXTURE:
        return "plain"
    if m.sharpness < BLUR_SHARPNESS:
        return "blur"
    if m.features < FEW_FEATURES:
        return "few_features"
    return None


def unregistered_reason(m: Measures) -> str:
    """Why a frame the mapper did not register was rejected: ``few_features`` when it has few,
    else ``no_overlap`` (no verified link to the map)."""
    return "few_features" if m.features < SUSPECT_FEATURES else "no_overlap"
