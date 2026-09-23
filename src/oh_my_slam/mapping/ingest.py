"""Mapper inputs: ``-a`` expansion (image files, image folders, or exactly one video), keyframe
sampling and naming, capture timestamps."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import ExifTags, Image

from oh_my_slam.core.errors import InputError, UsageError
from oh_my_slam.core.images import (
    IMAGE_SUFFIXES,
    VIDEO_SUFFIXES,
    exif_intrinsics,
    is_image_file,
    load_rgb,
    save_jpeg,
)
from oh_my_slam.core.types import Intrinsics

DEFAULT_FPS = 2.0


@dataclass
class InputSpec:
    kind: str  # "images" | "video"
    images: list[Path]
    video: Path | None = None


@dataclass
class Keyframe:
    name: str  # fNNNNNN
    index: int
    path: Path  # JPEG written into the map staging area
    source: str
    timestamp: float | None
    exif: Intrinsics | None  # intrinsics from the original file's EXIF, if any


def _expand_dir(d: Path) -> list[Path]:
    return sorted((p for p in d.iterdir() if is_image_file(p)), key=lambda p: p.name)


def resolve_inputs(args: list[Path]) -> InputSpec:
    """Files and folders of images (folders sorted by name, hidden/non-image skipped), or
    exactly one video."""
    if not args:
        raise UsageError("-a needs at least one image, image folder or video")
    videos = [a for a in args if a.is_file() and a.suffix.lower() in VIDEO_SUFFIXES]
    if videos:
        if len(args) != 1:
            raise UsageError("-a takes exactly one video, or images/folders (not both)")
        return InputSpec("video", [], videos[0])
    images: list[Path] = []
    for a in args:
        if not a.exists():
            raise InputError(f"input not found: {a}")
        if a.is_dir():
            found = _expand_dir(a)
            if not found:
                raise InputError(f"no images in folder {a}")
            images.extend(found)
        elif a.suffix.lower() in IMAGE_SUFFIXES and not a.name.startswith("."):
            images.append(a)
        else:
            raise InputError(f"unsupported input (not an image or video): {a}")
    return InputSpec("images", images)


def _exif_time(path: Path) -> float | None:
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            sub = exif.get_ifd(ExifTags.IFD.Exif)
            raw = sub.get(0x9003) or exif.get(0x0132)  # DateTimeOriginal / DateTime
        if not raw:
            return None
        return datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S").timestamp()
    except Exception:
        return None


def _write_image_keyframe(src: Path, dst: Path) -> None:
    """Upright JPEG copy: byte copy for upright JPEGs (keeps EXIF), re-encode otherwise."""
    with Image.open(src) as img:
        orientation = img.getexif().get(ExifTags.Base.Orientation, 1)
        fmt = img.format
    dst.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "JPEG" and orientation in (1, None):
        shutil.copyfile(src, dst)
    else:
        save_jpeg(load_rgb(src), dst, quality=95)


def keyframes(spec: InputSpec, fps: float, frames_dir: Path, start_index: int
              ) -> Iterator[Keyframe]:
    """Write keyframes as ``frames_dir/fNNNNNN.jpg`` and yield them in capture order."""
    idx = start_index
    if spec.kind == "video":
        from oh_my_slam.core.video import sample_frames

        assert spec.video is not None
        for f in sample_frames(spec.video, fps):
            name = f"f{idx:06d}"
            path = frames_dir / f"{name}.jpg"
            save_jpeg(f.rgb, path, quality=95)
            yield Keyframe(name, idx, path, f"{spec.video}@{f.timestamp:.3f}", f.timestamp, None)
            idx += 1
        return
    for src in spec.images:
        name = f"f{idx:06d}"
        path = frames_dir / f"{name}.jpg"
        _write_image_keyframe(src, path)
        yield Keyframe(name, idx, path, str(src), _exif_time(src), exif_intrinsics(src))
        idx += 1
