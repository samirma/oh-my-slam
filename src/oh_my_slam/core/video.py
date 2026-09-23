"""Video frame sampling: the sharpest frame in each 1/fps time slot, upright.

PyAV is imported lazily (it bundles FFmpeg dylibs that duplicate OpenCV's; importing it only
when a video is actually read keeps the two apart in every other command).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.errors import InputError

_SHARPNESS_WIDTH = 480


@dataclass
class SampledFrame:
    slot: int
    timestamp: float
    rgb: NDArray[np.uint8]
    sharpness: float


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float
    rotation: int  # degrees clockwise to display upright (0, 90, 180, 270)


def laplacian_variance(gray: NDArray[Any]) -> float:
    """Sharpness score: variance of the 4-neighbour Laplacian."""
    g = np.asarray(gray, dtype=np.float32)
    lap = (
        -4.0 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    )
    return float(lap.var())


def _rotation_of(stream: Any, frame: Any | None = None) -> int:
    """Clockwise display rotation in degrees from frame side data or stream metadata."""
    angle: float | None = None
    if frame is not None:
        rot = getattr(frame, "rotation", None)
        if rot is not None:
            angle = float(rot)
    if angle is None:
        meta = getattr(stream, "metadata", {}) or {}
        if "rotate" in meta:
            try:
                angle = -float(meta["rotate"])
            except ValueError:
                angle = None
    if angle is None:
        return 0
    # PyAV reports the display-matrix angle counter-clockwise; convert to clockwise.
    return round(-angle / 90.0) % 4 * 90


def upright(rgb: NDArray[np.uint8], rotation_cw: int) -> NDArray[np.uint8]:
    k = (rotation_cw // 90) % 4
    if k == 0:
        return rgb
    return np.ascontiguousarray(np.rot90(rgb, k=-k))


def probe(path: Path) -> VideoInfo:
    import av

    path = Path(path)
    if not path.is_file():
        raise InputError(f"video not found: {path}")
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate or stream.guessed_rate or 30.0)
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = container.duration / 1_000_000.0
            else:
                duration = 0.0
            first = next(container.decode(stream), None)
            rotation = _rotation_of(stream, first)
            w, h = stream.codec_context.width, stream.codec_context.height
    except InputError:
        raise
    except Exception as exc:
        raise InputError(f"cannot read video {path}: {exc}") from exc
    if rotation in (90, 270):
        w, h = h, w
    return VideoInfo(width=w, height=h, fps=fps, duration=duration, rotation=rotation)


def sample_frames(path: Path, fps: float) -> Iterator[SampledFrame]:
    """Yield the sharpest upright frame of each ``1/fps`` slot, in time order."""
    import av

    if fps <= 0:
        raise InputError("-fps must be positive")
    path = Path(path)
    if not path.is_file():
        raise InputError(f"video not found: {path}")
    try:
        container = av.open(str(path))
    except Exception as exc:
        raise InputError(f"cannot read video {path}: {exc}") from exc
    with container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rotation: int | None = None
        best: tuple[float, float, Any] | None = None  # (sharpness, t, frame)
        current_slot: int | None = None
        t0: float | None = None
        src_w = stream.codec_context.width
        sw = min(_SHARPNESS_WIDTH, src_w)
        sh = max(2, round(stream.codec_context.height * sw / src_w))
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                continue
            t = float(frame.pts * frame.time_base)
            if t0 is None:
                t0 = t
            t -= t0
            if rotation is None:
                rotation = _rotation_of(stream, frame)
            slot = int(np.floor(t * fps + 1e-9))
            if current_slot is not None and slot != current_slot and best is not None:
                yield _finish(current_slot, best, rotation)
                best = None
            current_slot = slot
            gray = frame.reformat(width=sw, height=sh, format="gray").to_ndarray()
            score = laplacian_variance(gray)
            if best is None or score > best[0]:
                best = (score, t, frame)
        if current_slot is not None and best is not None:
            yield _finish(current_slot, best, rotation or 0)


def _finish(slot: int, best: tuple[float, float, Any], rotation: int) -> SampledFrame:
    score, t, frame = best
    rgb = frame.to_ndarray(format="rgb24")
    return SampledFrame(slot=slot, timestamp=t, rgb=upright(rgb, rotation), sharpness=score)
