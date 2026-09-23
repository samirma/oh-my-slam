from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from oh_my_slam.core import video
from oh_my_slam.core.errors import InputError


def _make_clip(path: Path, seconds: float = 3.0, fps: int = 10, size: tuple[int, int] = (64, 48),
               rotate: int | None = None) -> None:
    """Frames whose top-left pixel encodes the frame index; odd frames blurred (less sharp)."""
    import av

    w, h = size
    with av.open(str(path), "w") as out:
        stream = out.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        stream.bit_rate = 4_000_000
        rng = np.random.default_rng(0)
        for i in range(int(seconds * fps)):
            img = np.full((h, w, 3), 128, np.uint8)
            if i % 2 == 0:
                img = rng.integers(0, 256, size=(h, w, 3)).astype(np.uint8)  # sharp noise
            img[:8, :8] = (i * 8) % 256
            img[-8:, :8] = (255, 0, 0)  # marker in the bottom-left corner
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for pkt in stream.encode(frame):
                out.mux(pkt)
        for pkt in stream.encode():
            out.mux(pkt)
    if rotate is not None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            pytest.skip("ffmpeg not available")
        tmp = path.with_name("rot_" + path.name)
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-display_rotation", str(rotate), "-i", str(path),
             "-c", "copy", str(tmp)],
            check=True,
        )
        tmp.replace(path)


def test_laplacian_variance_orders_sharpness(rng: np.random.Generator) -> None:
    sharp = rng.integers(0, 256, size=(40, 40)).astype(np.float32)
    flat = np.full((40, 40), 100.0)
    assert video.laplacian_variance(sharp) > video.laplacian_variance(flat) == 0.0


def test_sample_frames_one_per_slot_sharpest(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    _make_clip(clip, seconds=3.0, fps=10)
    info = video.probe(clip)
    assert (info.width, info.height, info.rotation) == (64, 48, 0)
    assert info.fps == pytest.approx(10)
    frames = list(video.sample_frames(clip, fps=2))
    assert [f.slot for f in frames] == list(range(6))
    for f in frames:
        # timestamp within its slot of the 1/fps grid (AC8)
        assert f.slot / 2 <= f.timestamp < (f.slot + 1) / 2 + 1e-6
        # sharpest = an even (noise) frame
        idx = round(float(f.rgb[:8, :8].mean()) / 8)
        assert idx % 2 == 0
        assert f.rgb.shape == (48, 64, 3)


def test_rotation_metadata_yields_upright_frames(tmp_path: Path) -> None:
    clip = tmp_path / "portrait.mp4"
    # display_rotation 90 = counter-clockwise: stored landscape, shown portrait.
    _make_clip(clip, seconds=1.0, fps=10, rotate=90)
    info = video.probe(clip)
    # ffmpeg's display rotation is counter-clockwise: 90 CCW == 270 clockwise.
    assert info.rotation == 270
    assert (info.width, info.height) == (48, 64)
    frames = list(video.sample_frames(clip, fps=2))
    assert frames and frames[0].rgb.shape == (64, 48, 3)
    rgb = frames[0].rgb
    # Rotating CCW by 90 moves the stored bottom-left red marker to the bottom-right.
    corner = rgb[-6:, -6:].reshape(-1, 3).mean(0)
    assert corner[0] > 150 and corner[1] < 100


def test_upright_rotations() -> None:
    a = np.arange(6, dtype=np.uint8).reshape(2, 3, 1).repeat(3, axis=2)
    assert video.upright(a, 0) is a
    assert video.upright(a, 90).shape == (3, 2, 3)
    # clockwise: first row becomes last column
    np.testing.assert_array_equal(video.upright(a, 90)[:, -1, 0], a[0, :, 0])
    np.testing.assert_array_equal(video.upright(a, 180)[..., 0], a[::-1, ::-1, 0])


def test_video_errors(tmp_path: Path) -> None:
    with pytest.raises(InputError):
        list(video.sample_frames(tmp_path / "missing.mp4", 2))
    with pytest.raises(InputError):
        video.probe(tmp_path / "missing.mp4")
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"garbage")
    with pytest.raises(InputError):
        list(video.sample_frames(bad, 2))
    with pytest.raises(InputError):
        video.probe(bad)
    with pytest.raises(InputError):
        list(video.sample_frames(bad, 0))
