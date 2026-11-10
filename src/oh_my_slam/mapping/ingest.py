"""Mapper inputs: the image files of an update, in order, written as named keyframes. There is no
video input (high_level_spec.md §13): the client samples its own frames."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from PIL import ExifTags

from oh_my_slam.core.errors import InputError, UsageError
from oh_my_slam.core.images import (
    IMAGE_SUFFIXES,
    exif_intrinsics,
    load_rgb,
    open_header,
    save_jpeg,
)
from oh_my_slam.core.types import Intrinsics
from oh_my_slam.mapping.store import frame_name


@dataclass
class InputSpec:
    kind: str  # "images" (an update's record names its kind; maps of earlier tools: "video")
    images: list[Path]


@dataclass
class Keyframe:
    name: str  # fNNNNNN
    index: int
    path: Path  # JPEG written into the map staging area
    source: str
    exif: Intrinsics | None  # intrinsics from the original file's EXIF, if any


def resolve_inputs(args: list[Path]) -> InputSpec:
    """Image files, in the order given."""
    if not args:
        raise UsageError("an update needs at least one image")
    images: list[Path] = []
    for a in args:
        if not a.exists():
            raise InputError(f"input not found: {a}")
        if a.is_dir():
            raise InputError(f"{a} is a folder; an update takes image files (e.g. {a}/*.jpg)")
        if a.suffix.lower() in IMAGE_SUFFIXES:
            images.append(a)
        else:
            raise InputError(f"unsupported input (not an image): {a}")
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


def keyframes(spec: InputSpec, frames_dir: Path, start_index: int,
              intrinsics: dict[Path, Intrinsics] | None = None) -> Iterator[Keyframe]:
    """Write keyframes as ``frames_dir/fNNNNNN.jpg`` and yield them in capture order. An image
    that ``intrinsics`` names carries those (a calibrated camera, ``oh_my_slam.service``) instead
    of its EXIF ones: the mapper holds them as it holds an EXIF focal length."""
    idx = start_index
    for src in spec.images:
        name = frame_name(idx)
        path = frames_dir / f"{name}.jpg"
        write_upright_jpeg(src, path)
        given = (intrinsics or {}).get(src)
        yield Keyframe(name, idx, path, str(src), exif_intrinsics(src) if given is None else given)
        idx += 1
