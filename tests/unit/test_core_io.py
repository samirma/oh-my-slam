from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from oh_my_slam.core import atomic, images, paths, rle
from oh_my_slam.core.errors import ExitCode, InputError, ServerUnavailableError
from oh_my_slam.core.ply import PointCloud, parse_header, parse_ply, ply_bytes, read_ply

# --- PLY ------------------------------------------------------------------------------------------


def test_ply_roundtrip_with_and_without_label(tmp_path: Path, rng: np.random.Generator) -> None:
    xyz = rng.normal(size=(100, 3)).astype(np.float32)
    rgb = rng.integers(0, 256, size=(100, 3)).astype(np.uint8)
    cloud = PointCloud(xyz, rgb)
    data = ply_bytes(cloud, comments=["test"])
    assert data.startswith(b"ply\nformat binary_little_endian 1.0\ncomment test\n")
    back = parse_ply(data)
    np.testing.assert_array_equal(back.xyz, xyz)
    np.testing.assert_array_equal(back.rgb, rgb)
    assert back.label is None
    labelled = PointCloud(xyz, rgb, np.arange(100))
    (tmp_path / "a.ply").write_bytes(ply_bytes(labelled))
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
        parse_ply(b"ply\nformat ascii 1.0\nend_header\n")  # no vertex element
    with pytest.raises(ValueError):
        parse_ply(b"ply\nformat binary_big_endian 1.0\nelement vertex 0\nend_header\n")
    with pytest.raises(ValueError):
        PointCloud(np.zeros((3, 3)), normals=np.zeros((2, 3)))
    with pytest.raises(ValueError):
        ply_bytes(PointCloud(np.zeros((1, 3))), comments=["two\nlines"])
    with pytest.raises(ValueError):
        ply_bytes(PointCloud(np.zeros((1, 3))), encoding="utf8")


def _cloud(rng: np.random.Generator, n: int, *, rgb: bool, label: bool, normals: bool
           ) -> PointCloud:
    nrm = rng.normal(size=(n, 3))
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    return PointCloud(rng.normal(size=(n, 3)).astype(np.float32),
                      rng.integers(0, 256, (n, 3)).astype(np.uint8) if rgb else None,
                      rng.integers(0, 40, n).astype(np.int32) if label else None,
                      nrm.astype(np.float32) if normals else None)


@pytest.mark.parametrize("encoding", ["binary", "ascii"])
@pytest.mark.parametrize(("rgb", "label", "normals"),
                         [(True, False, False), (False, False, False), (True, True, True),
                          (False, True, True), (True, False, True)])
def test_ply_writer_reader_roundtrip(rng: np.random.Generator, encoding: str, rgb: bool,
                                     label: bool, normals: bool) -> None:
    cloud = _cloud(rng, 57, rgb=rgb, label=label, normals=normals)
    data = ply_bytes(cloud, encoding=encoding, comments=["first", "attributes color=rgb"])
    fmt = b"binary_little_endian 1.0" if encoding == "binary" else b"ascii 1.0"
    assert data.startswith(b"ply\nformat " + fmt + b"\ncomment first\ncomment attributes")
    header = parse_header(data)
    assert header.comments == ["first", "attributes color=rgb"] and header.count == 57
    names = [n for n, _ in header.fields]
    expected = ["x", "y", "z"] + (["nx", "ny", "nz"] if normals else []) \
        + (["red", "green", "blue"] if rgb else []) + (["label"] if label else [])
    assert names == expected
    head = data[:header.body_offset].decode()
    assert ("property uchar red" in head) == rgb and ("property int label" in head) == label
    assert ("property float nx" in head) == normals
    back = parse_ply(data)
    np.testing.assert_array_equal(back.xyz, cloud.xyz)  # exact, also in ASCII (%.9g)
    for a, b in ((back.rgb, cloud.rgb), (back.label, cloud.label), (back.normals, cloud.normals)):
        assert (a is None) == (b is None)
        if a is not None:
            np.testing.assert_array_equal(a, b)
    if encoding == "binary":
        per = 12 + 12 * normals + 3 * rgb + 4 * label
        assert len(data) == header.body_offset + 57 * per
    else:
        assert data[header.body_offset:].count(b"\n") == 57
    # deterministic: the same cloud gives the same bytes
    assert ply_bytes(cloud, encoding=encoding, comments=["first", "attributes color=rgb"]) == data


def test_ply_empty_cloud_both_encodings() -> None:
    for enc in ("binary", "ascii"):
        data = ply_bytes(PointCloud(np.zeros((0, 3)), np.zeros((0, 3))), encoding=enc)
        back = parse_ply(data)
        assert len(back) == 0 and back.rgb is not None and back.label is None


@pytest.mark.parametrize(("rgb", "label", "normals"),
                         [(True, True, True), (False, False, False), (True, False, False)])
def test_read_ply_reads_a_binary_body_in_chunks(tmp_path: Path, rng: np.random.Generator,
                                                monkeypatch: pytest.MonkeyPatch, rgb: bool,
                                                label: bool, normals: bool) -> None:
    """``read_ply`` streams a binary body into the cloud's arrays (a large map cloud is never
    held twice); the values are exactly those of ``parse_ply``, chunk boundaries included."""
    import oh_my_slam.core.ply as ply

    monkeypatch.setattr(ply, "READ_CHUNK_BYTES", 100)  # a few vertices per chunk
    cloud = _cloud(rng, 57, rgb=rgb, label=label, normals=normals)
    long_header = [f"comment line {i:05d} " + "x" * 60 for i in range(1500)]  # > one header read
    for enc, comments in (("binary", ["c"]), ("binary", long_header), ("ascii", ["c"])):
        data = ply_bytes(cloud, encoding=enc, comments=comments)
        (tmp_path / "c.ply").write_bytes(data)
        back, ref = read_ply(tmp_path / "c.ply"), parse_ply(data)
        for a, b in ((back.xyz, ref.xyz), (back.rgb, ref.rgb), (back.label, ref.label),
                     (back.normals, ref.normals)):
            assert (a is None) == (b is None)
            if a is not None:
                assert a.dtype == b.dtype
                np.testing.assert_array_equal(a, b)
    empty = ply_bytes(PointCloud(np.zeros((0, 3)), np.zeros((0, 3))))
    (tmp_path / "e.ply").write_bytes(empty)
    assert len(read_ply(tmp_path / "e.ply")) == 0
    (tmp_path / "t.ply").write_bytes(ply_bytes(cloud)[:-5])
    with pytest.raises(ValueError, match="shorter than its 57 vertices"):
        read_ply(tmp_path / "t.ply")
    (tmp_path / "n.ply").write_bytes(b"not a ply")
    with pytest.raises(ValueError, match="not a PLY file"):
        read_ply(tmp_path / "n.ply")


def test_pointcloud_subset() -> None:
    c = PointCloud(np.zeros((5, 3)), np.zeros((5, 3)), np.array([1, 2, 3, 4, 5]),
                   np.arange(15).reshape(5, 3))
    sub = c.subset(np.array([0, 4]))
    assert len(sub) == 2 and sub.label is not None and sub.label.tolist() == [1, 5]
    assert sub.normals is not None and sub.normals[1].tolist() == [12, 13, 14]
    assert PointCloud(np.zeros((2, 3))).subset(np.array([1])).rgb is None


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


def test_image_errors_and_types(tmp_path: Path) -> None:
    with pytest.raises(InputError):
        images.load_rgb(tmp_path / "missing.jpg")
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"nope")
    with pytest.raises(InputError):
        images.load_rgb(bad)
    ok = tmp_path / "x.png"
    ok.write_bytes(images.png_bytes(np.zeros((4, 4, 3), np.uint8)))
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
