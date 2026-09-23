"""Image loading (EXIF orientation, HEIC) and EXIF-derived intrinsics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import ExifTags, Image, ImageOps

from oh_my_slam.core.errors import InputError
from oh_my_slam.core.types import Intrinsics

IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}
)
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"})

# Diagonal of a 36 x 24 mm full-frame sensor.
FULL_FRAME_DIAGONAL_MM = math.hypot(36.0, 24.0)

_heif_registered = False


def _register_heif() -> None:
    global _heif_registered
    if not _heif_registered:
        import pillow_heif

        pillow_heif.register_heif_opener()
        _heif_registered = True


def is_image_file(path: Path) -> bool:
    p = Path(path)
    return p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_SUFFIXES


def is_video_file(path: Path) -> bool:
    p = Path(path)
    return p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES


def open_image(path: Path) -> Image.Image:
    """Open an image with EXIF orientation applied (not yet converted to RGB)."""
    path = Path(path)
    if not path.is_file():
        raise InputError(f"image not found: {path}")
    if path.suffix.lower() in {".heic", ".heif"}:
        _register_heif()
    try:
        img = Image.open(path)
        img.load()
    except Exception as exc:  # PIL raises many types
        raise InputError(f"cannot read image {path}: {exc}") from exc
    return ImageOps.exif_transpose(img) or img


def load_rgb(path: Path, max_side: int | None = None) -> NDArray[np.uint8]:
    """RGB uint8 (H, W, 3), upright, optionally downscaled so the long side is <= max_side."""
    img = open_image(path).convert("RGB")
    if max_side is not None:
        img = resize_to_max_side(img, max_side)
    return np.asarray(img, dtype=np.uint8)


def resize_to_max_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    scale = max_side / max(w, h)
    if scale >= 1.0:
        return img
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return img.resize(size, Image.Resampling.LANCZOS)


def scaled_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    scale = max_side / max(width, height)
    if scale >= 1.0:
        return width, height
    return max(1, round(width * scale)), max(1, round(height * scale))


def upright_size(path: Path) -> tuple[int, int]:
    """(width, height) after EXIF orientation, without decoding pixels where possible."""
    path = Path(path)
    if path.suffix.lower() in {".heic", ".heif"}:
        _register_heif()
    with Image.open(path) as img:
        w, h = img.size
        orientation = img.getexif().get(ExifTags.Base.Orientation, 1)
    if orientation in (5, 6, 7, 8):
        w, h = h, w
    return w, h


def _exif_dict(img: Image.Image) -> dict[str, Any]:
    exif = img.getexif()
    out: dict[str, Any] = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif.items()}
    try:
        sub = exif.get_ifd(ExifTags.IFD.Exif)
        out.update({ExifTags.TAGS.get(k, str(k)): v for k, v in sub.items()})
    except Exception:
        pass
    return out


def _as_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return f if math.isfinite(f) and f > 0 else None


def exif_focal_px(path: Path) -> float | None:
    """Focal length in pixels of the upright image, from EXIF, or None.

    Priority: 35 mm-equivalent focal length with the diagonal formula
    ``f_px = f35 * diag_px / 43.27``; else FocalLength with FocalPlaneXResolution.
    """
    path = Path(path)
    if path.suffix.lower() in {".heic", ".heif"}:
        _register_heif()
    with Image.open(path) as img:
        tags = _exif_dict(img)
        raw_w, raw_h = img.size
    diag_px = math.hypot(raw_w, raw_h)
    f35 = _as_float(tags.get("FocalLengthIn35mmFilm"))
    if f35 is not None:
        return f35 * diag_px / FULL_FRAME_DIAGONAL_MM
    f_mm = _as_float(tags.get("FocalLength"))
    res = _as_float(tags.get("FocalPlaneXResolution"))
    unit = tags.get("FocalPlaneResolutionUnit", 2)
    unit_mm = {2: 25.4, 3: 10.0, 4: 1.0, 5: 0.001}.get(int(unit) if unit else 2)
    if f_mm is not None and res is not None and unit_mm is not None:
        f_px = f_mm * res / unit_mm
        # Plausibility: horizontal FOV between 10 and 150 degrees.
        fov = 2 * math.degrees(math.atan(raw_w / (2 * f_px)))
        if 10.0 <= fov <= 150.0:
            return f_px
    return None


def exif_intrinsics(path: Path) -> Intrinsics | None:
    """Intrinsics of the upright image from EXIF (principal point at the centre), or None."""
    f = exif_focal_px(path)
    if f is None:
        return None
    w, h = upright_size(path)
    return Intrinsics(fx=f, fy=f, cx=w / 2.0, cy=h / 2.0, width=w, height=h, source="exif")


def save_jpeg(rgb: NDArray[np.uint8], path: Path, quality: int = 95) -> None:
    import io

    from oh_my_slam.core.atomic import atomic_write_bytes

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    atomic_write_bytes(path, buf.getvalue())


def save_png(array: NDArray[Any], path: Path) -> None:
    """Lossless PNG without colour profile or gamma chunk."""
    from oh_my_slam.core.atomic import atomic_write_bytes

    atomic_write_bytes(path, png_bytes(array))


def png_bytes(array: NDArray[Any]) -> bytes:
    import io

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(array)).save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def load_png(path: Path) -> NDArray[Any]:
    with Image.open(path) as img:
        return np.asarray(img)
