"""Video sampling (``mapper.sh -fps``) on what real files contain besides clean frames: display
rotation from frame side data or the older ``rotate`` stream tag (or a malformed one), frames
without a timestamp, and videos without any usable frame. PyAV is replaced by a double that hands
out such frames (``core.video`` imports it lazily)."""

from __future__ import annotations

import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oh_my_slam.core import video
from tests.fakes.fake_torch import installed


def _stream(metadata: dict[str, str] | None = None, w: int = 8, h: int = 6) -> Any:
    return types.SimpleNamespace(metadata=metadata, thread_type=None,
                                 codec_context=types.SimpleNamespace(width=w, height=h))


@pytest.mark.parametrize(("frame_rotation", "metadata", "clockwise"), [
    (None, {"rotate": "90"}, 90),  # the old tag is clockwise already
    (None, {"rotate": "270"}, 270),
    (90.0, {"rotate": "270"}, 270),  # side data wins: 90 counter-clockwise
    (-90.0, None, 90),
    (None, {"rotate": "sideways"}, 0),  # malformed: not rotated
    (None, {}, 0),
    (None, None, 0),
])
def test_rotation_from_side_data_or_the_rotate_tag(frame_rotation: float | None,
                                                   metadata: dict[str, str] | None,
                                                   clockwise: int) -> None:
    frame = types.SimpleNamespace(rotation=frame_rotation)
    assert video._rotation_of(_stream(metadata), frame) == clockwise
    if frame_rotation is None:  # no frame at all: the stream's tag alone
        assert video._rotation_of(_stream(metadata)) == clockwise


class _Frame:
    def __init__(self, t: float | None, sharpness: float, tag: int, time_base: float | None = 0.01
                 ) -> None:
        self.pts = None if t is None else round(t / 0.01)
        self.time_base = time_base
        self.rotation = None
        rng = np.random.default_rng(tag)
        self.gray = (rng.random((6, 8)) * sharpness).astype(np.float32)
        self.rgb = np.zeros((6, 8, 3), np.uint8)
        self.rgb[0, 0] = tag  # identifies the frame
        self.rgb[5, 0] = 255  # bottom-left marker

    def reformat(self, width: int, height: int, format: str) -> Any:
        assert (width, height, format) == (8, 6, "gray")
        return types.SimpleNamespace(to_ndarray=lambda: self.gray)

    def to_ndarray(self, format: str) -> np.ndarray:
        assert format == "rgb24"
        return self.rgb


class _Container:
    def __init__(self, stream: Any, frames: list[_Frame]) -> None:
        self.streams = types.SimpleNamespace(video=[stream])
        self.frames = frames

    def __enter__(self) -> _Container:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def decode(self, stream: Any) -> Iterator[_Frame]:
        assert stream.thread_type == "AUTO"
        yield from self.frames


def _sample(tmp_path: Path, container: _Container, fps: float) -> list[video.SampledFrame]:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\x00")
    av = types.ModuleType("av")
    av.open = lambda p: container  # type: ignore[attr-defined]
    with installed({"av": av}):
        return list(video.sample_frames(path, fps))


def test_frames_without_a_timestamp_are_skipped(tmp_path: Path) -> None:
    frames = [_Frame(None, 100.0, 1), _Frame(5.0, 100.0, 2, time_base=None),
              _Frame(10.0, 1.0, 3), _Frame(10.2, 50.0, 4), _Frame(10.4, 2.0, 5),
              _Frame(10.6, 1.0, 6)]
    out = _sample(tmp_path, _Container(_stream({"rotate": "90"}), frames), fps=2)
    # time starts at the first timestamped frame; the sharpest of each half second is kept
    assert [(f.slot, int(f.rgb[0, -1, 0])) for f in out] == [(0, 4), (1, 6)]
    assert out[0].timestamp == pytest.approx(0.2) and out[1].timestamp == pytest.approx(0.6)
    # upright by the stream's rotate tag (90 clockwise): the bottom-left marker goes top-left
    assert out[0].rgb.shape == (8, 6, 3) and out[0].rgb[0, 0, 0] == 255


def test_a_video_without_usable_frames_gives_none(tmp_path: Path) -> None:
    frames = [_Frame(None, 10.0, 1), _Frame(None, 10.0, 2)]
    assert _sample(tmp_path, _Container(_stream(), frames), fps=1) == []
    assert _sample(tmp_path, _Container(_stream(), []), fps=1) == []
