"""The image a request hands the server, at the size the server reads it.

The server reads each request's image from disk and downscales it on its single device thread
(``core.images.load_rgb``: decode, EXIF orientation, RGB, LANCZOS down to the long side its model
reads), so that work queues with the models. For a 12 MP photo it costs more than some of the
models, and it is repeated for geometry, gravity and both detection passes. The client therefore
decodes each image once and sends it already at the size each request reads, as an uncompressed
BMP that decodes in about a millisecond and that ``load_rgb`` leaves as it is (its long side is the
side asked for, and ``load_rgb`` never enlarges): the array the model reads is the same, pixel for
pixel, and so are its results. What the server reports in pixels of the image it read is converted
back by ``InferenceClient``.
"""

from __future__ import annotations

import os
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from oh_my_slam.core import paths
from oh_my_slam.core.images import open_image, resize_to_max_side, size_at_max_side, upright_size

KEEP_DECODED = 6  # images kept: the keyframes in flight and their requests
# The long sides the server reads an image at: gravity; the mapper's keyframe grid and the coarse
# detection pass; detection, the single-image grid and multi-view poses. An image is decoded once
# and kept at these sides only (a few MB, where a 12 MP photo decoded takes 36 MB).
READ_SIDES = (640, 768, 1024)


@dataclass
class _Entry:
    lock: threading.Lock = field(default_factory=threading.Lock)
    sides: dict[int, Image.Image] = field(default_factory=dict)


class _Decoded:
    """The last few sources at the sides the server reads them (thread-safe; an image is decoded
    once however many threads ask for it at the ``READ_SIDES``)."""

    def __init__(self, keep: int = KEEP_DECODED) -> None:
        self.keep = keep
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple[str, int, int], _Entry] = OrderedDict()

    def _entry(self, path: Path) -> _Entry:
        st = path.stat()
        key = (str(path.resolve()), st.st_mtime_ns, st.st_size)
        with self._lock:
            e = self._entries.get(key)
            if e is None:
                e = self._entries[key] = _Entry()
                while len(self._entries) > self.keep:
                    self._entries.popitem(last=False)
            else:
                self._entries.move_to_end(key)
            return e

    def at_side(self, path: Path, side: int) -> Image.Image:
        """``load_rgb(path, side)`` as a PIL image."""
        e = self._entry(path)
        with e.lock:
            img = e.sides.get(side)
            if img is None:
                full = open_image(path).convert("RGB")
                for s in (side, *READ_SIDES):
                    if s not in e.sides:
                        e.sides[s] = resize_to_max_side(full, s)
                img = e.sides[side]
            return img


_decoded = _Decoded()


def request_rgb(path: Path, side: int) -> NDArray[np.uint8]:
    """``core.images.load_rgb(path, side)``, from the decoded image the requests share."""
    return np.asarray(_decoded.at_side(Path(path), side), dtype=np.uint8)


def remember_rgb(path: Path, side: int, rgb: NDArray[np.uint8]) -> None:
    """``rgb`` is ``request_rgb(path, side)`` as an earlier request read it (the mapper's focal
    re-run has every keyframe's): requests at that side send it without decoding the image."""
    e = _decoded._entry(Path(path))
    with e.lock:
        if side not in e.sides:
            e.sides[side] = Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8))


@dataclass(frozen=True)
class Sent:
    """The file a request hands the server for an image, and how its pixels relate to the
    original's: ``scale`` = (width, height) of the file over the original's (upright), as the
    server would compute them from the original; 1 when the original itself is sent."""

    path: Path
    scale: tuple[float, float] = (1.0, 1.0)

    @property
    def downscaled(self) -> bool:
        return self.scale != (1.0, 1.0)


@contextmanager
def request_image(path: Path, side: int) -> Iterator[Sent]:
    """The image to send the server for ``path`` read at long side ``side``: ``path`` itself when
    the server would not downscale it (or cannot read it: it reports why), else a temporary BMP of
    ``load_rgb(path, side)``, removed when the request is done."""
    path = Path(path)
    try:
        w, h = upright_size(path)
        sw, sh = size_at_max_side(w, h, side)
        img = None if (sw, sh) == (w, h) else _decoded.at_side(path, side)
    except Exception:  # a missing or unreadable image: the server's error names it
        img = None
    if img is None:
        yield Sent(path)
        return
    fd, tmp = tempfile.mkstemp(prefix="request-", suffix=".bmp", dir=paths.scratch_dir())
    try:
        with os.fdopen(fd, "wb") as f:
            img.save(f, format="BMP")
        yield Sent(Path(tmp), (sw / w, sh / h))
    finally:
        Path(tmp).unlink(missing_ok=True)
