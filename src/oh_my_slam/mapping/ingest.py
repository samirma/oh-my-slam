"""Mapper inputs: ``-i`` resolution (image files, or exactly one video), keyframe sampling and
naming."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from PIL import ExifTags

from oh_my_slam.core.constants import DEFAULT_FPS as DEFAULT_FPS
from oh_my_slam.core.errors import InputError, UsageError
from oh_my_slam.core.images import (
    IMAGE_SUFFIXES,
    VIDEO_SUFFIXES,
    exif_intrinsics,
    load_rgb,
    open_header,
    save_jpeg,
)
from oh_my_slam.core.types import Intrinsics
from oh_my_slam.mapping.store import frame_name


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
    exif: Intrinsics | None  # intrinsics from the original file's EXIF, if any


def resolve_inputs(args: list[Path]) -> InputSpec:
    """Image files, in the order given, or exactly one video (spec §2.3: image(s) or a video)."""
    if not args:
        raise UsageError("-i needs at least one image or a video")
    videos = [a for a in args if a.is_file() and a.suffix.lower() in VIDEO_SUFFIXES]
    if videos:
        if len(args) != 1:
            raise UsageError("-i takes exactly one video, or images (not both)")
        return InputSpec("video", [], videos[0])
    images: list[Path] = []
    for a in args:
        if not a.exists():
            raise InputError(f"input not found: {a}")
        if a.is_dir():
            raise InputError(f"{a} is a folder; -i takes image files (e.g. {a}/*.jpg) or a video")
        if a.suffix.lower() in IMAGE_SUFFIXES and not a.name.startswith("."):
            images.append(a)
        else:
            raise InputError(f"unsupported input (not an image or video): {a}")
    return InputSpec("images", images)


def write_upright_jpeg(src: Path, dst: Path) -> None:
    """Upright JPEG copy: byte copy for upright JPEGs (keeps EXIF), re-encode otherwise."""
    with open_header(src) as img:
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
            name = frame_name(idx)
            path = frames_dir / f"{name}.jpg"
            save_jpeg(f.rgb, path, quality=95)
            yield Keyframe(name, idx, path, f"{spec.video}@{f.timestamp:.3f}", None)
            idx += 1
        return
    for src in spec.images:
        name = frame_name(idx)
        path = frames_dir / f"{name}.jpg"
        write_upright_jpeg(src, path)
        yield Keyframe(name, idx, path, str(src), exif_intrinsics(src))
        idx += 1
