from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from oh_my_slam.core import atomic, images, paths, rle
from oh_my_slam.core.errors import ExitCode, InputError, ServerUnavailableError
from oh_my_slam.core.ply import PointCloud, parse_ply, ply_bytes, read_ply, write_ply

# --- PLY ------------------------------------------------------------------------------------------


def test_ply_roundtrip_with_and_without_label(tmp_path: Path, rng: np.random.Generator) -> None:
    xyz = rng.normal(size=(100, 3)).astype(np.float32)
    rgb = rng.integers(0, 256, size=(100, 3)).astype(np.uint8)
    cloud = PointCloud(xyz, rgb)
    data = ply_bytes(cloud, comment="test")
    assert data.startswith(b"ply\nformat binary_little_endian 1.0\n")
    back = parse_ply(data)
    np.testing.assert_array_equal(back.xyz, xyz)
    np.testing.assert_array_equal(back.rgb, rgb)
    assert back.label is None
    labelled = PointCloud(xyz, rgb, np.arange(100))
    write_ply(tmp_path / "a.ply", labelled)
    back = read_ply(tmp_path / "a.ply")
    assert back.label is not None
    np.testing.assert_array_equal(back.label, np.arange(100))
    # header size is exact: payload = 15 bytes per point + 4 for the label
    assert len(ply_bytes(labelled)) - ply_bytes(labelled).find(b"end_header\n") - 11 == 100 * 19


def test_ply_validation_errors(rng: np.random.Generator) -> None:
    with pytest.raises(ValueError):
        PointCloud(np.zeros((3, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        PointCloud(np.zeros((3, 3)), np.zeros((3, 3)), np.zeros(2))
    with pytest.raises(ValueError):
        parse_ply(b"not a ply")
    with pytest.raises(ValueError):
        parse_ply(b"ply\nformat ascii 1.0\nend_header\n")


def test_pointcloud_concat_and_subset() -> None:
    a = PointCloud(np.zeros((2, 3)), np.zeros((2, 3)), np.array([1, 2]))
    b = PointCloud(np.ones((3, 3)), np.ones((3, 3)), np.array([3, 4, 5]))
    c = PointCloud.concat([a, b])
    assert len(c) == 5 and c.label is not None and c.label.tolist() == [1, 2, 3, 4, 5]
    assert len(c.subset(np.array([0, 4]))) == 2
    assert len(PointCloud.concat([])) == 0


# --- RLE ------------------------------------------------------------------------------------------


def test_rle_roundtrip(rng: np.random.Generator) -> None:
    for shape in [(1, 1), (7, 5), (48, 64)]:
        for density in (0.0, 0.3, 1.0):
            m = rng.random(shape) < density
            enc = rle.encode(m)
            assert isinstance(enc["counts"], str)
            np.testing.assert_array_equal(rle.decode(enc), m)
            assert rle.area(enc) == int(m.sum())


def test_rle_known_coco_string() -> None:
    # A 3x3 mask with the centre pixel set: F-order runs [4, 1, 4].
    m = np.zeros((3, 3), bool)
    m[1, 1] = True
    assert rle.encode_counts(m) == [4, 1, 4]
    assert rle.string_to_counts(rle.counts_to_string([4, 1, 4])) == [4, 1, 4]
    big = [1000, 20000, 5, 3, 70000]
    assert rle.string_to_counts(rle.counts_to_string(big)) == big
    with pytest.raises(ValueError):
        rle.decode_counts([1, 2], 2, 2)
    assert rle.decode({"size": [2, 2], "counts": [0, 4]}).all()


# --- atomic / paths / errors -----------------------------------------------------------------------


def test_atomic_writes(tmp_path: Path) -> None:
    p = tmp_path / "sub" / "x.bin"
    atomic.atomic_write_bytes(p, b"abc")
    assert p.read_bytes() == b"abc"
    atomic.atomic_write_json(tmp_path / "x.json", {"a": 1})
    assert (tmp_path / "x.json").read_text() == '{"a":1}\n'
    atomic.atomic_write_text(tmp_path / "t.txt", "hé")
    assert (tmp_path / "t.txt").read_text("utf-8") == "hé"
    atomic.atomic_save_npy(tmp_path / "a.npy", np.arange(3))
    assert np.load(tmp_path / "a.npy").tolist() == [0, 1, 2]
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]


def test_replace_dir(tmp_path: Path) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    (src / "a").write_text("new")
    dst.mkdir()
    (dst / "a").write_text("old")
    atomic.replace_dir(src, dst)
    assert (dst / "a").read_text() == "new" and not src.exists()
    src.mkdir()
    atomic.replace_dir(src, tmp_path / "fresh")
    assert (tmp_path / "fresh").is_dir()
    atomic.fsync_dir(tmp_path)


def test_paths_short_socket(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    long_dir = tmp_path / ("x" * 90)
    monkeypatch.setenv("OH_MY_SLAM_RUNTIME_DIR", str(long_dir))
    sock = paths.socket_path()
    assert len(str(sock)) <= 100 and sock.name == "srv.sock"
    monkeypatch.setenv("OH_MY_SLAM_RUNTIME_DIR", "/tmp/oms-short-test")
    assert paths.socket_path() == Path("/tmp/oms-short-test/srv.sock")
    assert paths.state_file().name == "server.json"
    assert paths.lock_file().name == "server.lock"
    assert paths.server_log().name == "server.log"
    assert paths.scratch_dir().is_dir()
    monkeypatch.setenv("OH_MY_SLAM_WEIGHTS_DIR", str(tmp_path / "w"))
    assert paths.weights_dir().is_dir()


def test_error_codes() -> None:
    err = ServerUnavailableError("no socket")
    assert err.exit_code == ExitCode.SERVER_UNAVAILABLE == 3
    assert "./start_inference_server.sh" in str(err)
    assert InputError("x").exit_code == 2


# --- images ---------------------------------------------------------------------------------------


def _jpeg_with_exif(path: Path, size: tuple[int, int], orientation: int = 1, **tags: object) -> None:
    img = Image.new("RGB", size, (10, 20, 30))
    img.putpixel((0, 0), (255, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = orientation
    sub = exif.get_ifd(0x8769)
    ids = {"FocalLengthIn35mmFilm": 0xA405, "FocalLength": 0x920A, "FocalPlaneXResolution": 0xA20E,
           "FocalPlaneResolutionUnit": 0xA210}
    for k, v in tags.items():
        sub[ids[k]] = v
    img.save(path, exif=exif, quality=100)


def test_exif_intrinsics_35mm(tmp_path: Path) -> None:
    p = tmp_path / "a.jpg"
    _jpeg_with_exif(p, (400, 300), FocalLengthIn35mmFilm=23)
    intr = images.exif_intrinsics(p)
    assert intr is not None and intr.source == "exif"
    assert intr.fx == pytest.approx(23 * 500 / 43.2666, rel=1e-3)
    assert (intr.cx, intr.cy, intr.width, intr.height) == (200, 150, 400, 300)


def test_exif_intrinsics_focal_plane_and_missing(tmp_path: Path) -> None:
    p = tmp_path / "b.jpg"
    # 4 mm lens, 1000 px/mm focal plane resolution (unit 4 = mm) -> 4000 px? implausible FOV for
    # a 400 px image (5.7 deg) -> rejected; use 0.25 mm/px equivalent instead.
    _jpeg_with_exif(p, (400, 300), FocalLength=4.0, FocalPlaneXResolution=100.0,
                    FocalPlaneResolutionUnit=4)
    intr = images.exif_intrinsics(p)
    assert intr is not None and intr.fx == pytest.approx(400.0)
    q = tmp_path / "c.jpg"
    _jpeg_with_exif(q, (400, 300))
    assert images.exif_intrinsics(q) is None


def test_orientation_applied(tmp_path: Path) -> None:
    p = tmp_path / "rot.jpg"
    _jpeg_with_exif(p, (40, 30), orientation=6, FocalLengthIn35mmFilm=28)
    rgb = images.load_rgb(p)
    assert rgb.shape == (40, 30, 3)
    assert images.upright_size(p) == (30, 40)
    intr = images.exif_intrinsics(p)
    assert intr is not None and (intr.width, intr.height) == (30, 40)
    small = images.load_rgb(p, max_side=20)
    assert max(small.shape[:2]) == 20
    assert images.scaled_size(4000, 3000, 1024) == (1024, 768)
    assert images.scaled_size(100, 50, 1024) == (100, 50)


def test_image_errors_and_types(tmp_path: Path) -> None:
    with pytest.raises(InputError):
        images.load_rgb(tmp_path / "missing.jpg")
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"nope")
    with pytest.raises(InputError):
        images.load_rgb(bad)
    ok = tmp_path / "x.png"
    images.save_png(np.zeros((4, 4, 3), np.uint8), ok)
    assert images.is_image_file(ok)
    hidden = tmp_path / ".DS_Store"
    hidden.write_bytes(b"")
    assert not images.is_image_file(hidden)
    assert not images.is_image_file(tmp_path / "x.txt")
    images.save_jpeg(np.zeros((4, 4, 3), np.uint8), tmp_path / "y.jpg")
    assert images.load_rgb(tmp_path / "y.jpg").shape == (4, 4, 3)
    png = images.png_bytes(np.full((2, 2, 3), 7, np.uint8))
    assert b"iCCP" not in png and b"gAMA" not in png and b"sRGB" not in png
    assert images.load_png(ok).shape == (4, 4, 3)


def test_heic_roundtrip(tmp_path: Path) -> None:
    pillow_heif = pytest.importorskip("pillow_heif")
    pillow_heif.register_heif_opener()
    p = tmp_path / "x.heic"
    buf = io.BytesIO()
    try:
        Image.new("RGB", (32, 16), (200, 100, 50)).save(buf, format="HEIF")
    except Exception as exc:  # encoder may be unavailable
        pytest.skip(f"HEIF encoder unavailable: {exc}")
    p.write_bytes(buf.getvalue())
    assert images.load_rgb(p).shape == (16, 32, 3)
