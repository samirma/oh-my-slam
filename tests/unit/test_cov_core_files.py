"""``core`` file handling at its edges: crash-safe writes that fail, write targets that cannot be
written, PLY headers the reader refuses or skips, empty RLE masks, EXIF that cannot be used, and
the small pose / similarity helpers."""

from __future__ import annotations

import os
import struct
import subprocess
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from oh_my_slam.core import atomic, images, rle
from oh_my_slam.core.errors import ExitCode, UsageError
from oh_my_slam.core.geometry import Sim3, rot_z
from oh_my_slam.core.ply import parse_ply, read_ply
from oh_my_slam.core.types import Pose

# --- crash-safe writes -------------------------------------------------------------------------


def _leftovers(folder: Path) -> list[str]:
    return sorted(f for f in os.listdir(folder) if f.startswith("."))


def test_a_failed_write_keeps_the_previous_file_and_leaves_no_temporary(tmp_path: Path) -> None:
    target = tmp_path / "scene.json"
    atomic.atomic_write_text(target, "previous")
    with pytest.raises(TypeError):
        atomic.atomic_write_bytes(target, "not bytes")  # type: ignore[arg-type]
    assert target.read_text() == "previous" and _leftovers(tmp_path) == []


def test_a_failed_array_save_keeps_the_previous_file_and_leaves_no_temporary(
        tmp_path: Path) -> None:
    target = tmp_path / "depth.npy"
    atomic.atomic_save_npy(target, np.arange(3))
    with pytest.raises(ValueError):  # object arrays would need pickling, which is refused
        atomic.atomic_save_npy(target, np.array([{"a": 1}], dtype=object))
    assert np.load(target).tolist() == [0, 1, 2] and _leftovers(tmp_path) == []


def test_published_files_get_the_usual_mode_not_mkstemps_0600(tmp_path: Path) -> None:
    atomic.atomic_write_bytes(tmp_path / "a.bin", b"x")
    atomic.atomic_save_npy(tmp_path / "a.npy", np.zeros(2))
    umask = os.umask(0)
    os.umask(umask)
    for name in ("a.bin", "a.npy"):
        assert (tmp_path / name).stat().st_mode & 0o777 == 0o666 & ~umask


def test_a_folder_that_may_not_be_written_is_a_usage_error(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        with pytest.raises(UsageError, match="Permission denied") as err:
            atomic.check_dir(locked / "out" / "deeper", "-d")
        assert err.value.exit_code == ExitCode.USAGE and str(locked / "out") in str(err.value)
        with pytest.raises(UsageError, match="cannot write there"):
            atomic.check_file(locked / "x.ply", "-o")
    finally:
        locked.chmod(0o700)


def test_a_special_file_is_written_in_place_unless_read_only(tmp_path: Path) -> None:
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo, 0o600)
    atomic.check_file(fifo, "-o")  # writable: accepted, its folder is not checked
    fifo.chmod(0o400)
    for check in (atomic.check_file, atomic.preflight_file):
        with pytest.raises(UsageError, match=r"-o .*pipe: cannot write there \(permission denied\)"):
            check(fifo, "-o")


def test_a_copy_falls_back_to_a_plain_copy_without_clones(tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "src.jpg"
    src.write_bytes(b"\xff\xd8 jpeg bytes")
    atomic.clone_file(src, tmp_path / "clone.jpg")  # APFS: a clone
    assert (tmp_path / "clone.jpg").read_bytes() == src.read_bytes()
    calls: list[list[str]] = []

    def no_clones(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(cmd)  # a filesystem without clones: cp -c fails
        return subprocess.CompletedProcess(cmd, 1, b"", b"cp: clonefile failed")

    monkeypatch.setattr(atomic.subprocess, "run", no_clones)
    atomic.clone_file(src, tmp_path / "copy.jpg")
    assert calls == [["cp", "-c", str(src), str(tmp_path / "copy.jpg")]]
    assert (tmp_path / "copy.jpg").read_bytes() == src.read_bytes()


# --- PLY -----------------------------------------------------------------------------------------


def _ply(header: str, body: bytes = b"") -> bytes:
    return header.encode("ascii") + body


def test_the_first_element_must_be_the_vertices() -> None:
    data = _ply("ply\nformat ascii 1.0\nelement face 0\nproperty list uchar int vertex_indices\n"
                "element vertex 1\nproperty float x\nend_header\n1\n")
    with pytest.raises(ValueError, match="the first PLY element must be vertex"):
        parse_ply(data)


def test_list_properties_on_vertices_are_refused() -> None:
    data = _ply("ply\nformat ascii 1.0\nelement vertex 1\nproperty list uchar int idx\n"
                "end_header\n0\n")
    with pytest.raises(ValueError, match="list properties are not supported on vertices"):
        parse_ply(data)


def test_elements_after_the_vertices_are_ignored(tmp_path: Path) -> None:
    """A mesh PLY: its faces (and their list property) follow the vertices and are skipped."""
    header = ("ply\nformat binary_little_endian 1.0\ncomment a mesh\nelement vertex 2\n"
              "property float x\nproperty float y\nproperty float z\n"
              "element face 1\nproperty list uchar int vertex_indices\nend_header\n")
    body = struct.pack("<6f", 1, 2, 3, 4, 5, 6) + struct.pack("<B3i", 3, 0, 1, 1)
    path = tmp_path / "mesh.ply"
    path.write_bytes(_ply(header, body))
    for cloud in (parse_ply(path.read_bytes()), read_ply(path)):
        assert cloud.xyz.tolist() == [[1, 2, 3], [4, 5, 6]] and cloud.rgb is None


def test_a_binary_ply_without_positions_is_refused(tmp_path: Path) -> None:
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    path = tmp_path / "colours.ply"
    path.write_bytes(_ply(header, bytes([1, 2, 3])))
    with pytest.raises(ValueError, match="PLY vertices have no x y z"):
        read_ply(path)


# --- RLE -----------------------------------------------------------------------------------------


def test_an_empty_mask_round_trips() -> None:
    empty = np.zeros((0, 5), bool)
    assert rle.encode_counts(empty) == []
    enc = rle.encode(empty)
    assert enc == {"size": [0, 5], "counts": ""}
    assert rle.decode(enc).shape == (0, 5) and rle.area(enc) == 0


def test_area_of_uncompressed_counts() -> None:
    mask = np.zeros((4, 5), bool)
    mask[1:3, 1:4] = True
    counts = rle.encode_counts(mask)
    assert rle.area({"size": [4, 5], "counts": counts}) == 6 == rle.area(rle.encode(mask))


# --- EXIF ----------------------------------------------------------------------------------------


def test_an_implausible_exif_focal_length_is_ignored(tmp_path: Path) -> None:
    img = Image.new("RGB", (400, 300))
    exif = Image.Exif()
    sub = exif.get_ifd(0x8769)
    # a 4 mm lens on a 1000 px/mm focal plane: 4000 px, a 5.7 degree field of view for 400 px
    sub[0x920A], sub[0xA20E], sub[0xA210] = 4.0, 1000.0, 4
    path = tmp_path / "tele.jpg"
    img.save(path, exif=exif)
    assert images.exif_focal_px(path) is None and images.exif_intrinsics(path) is None


def test_a_broken_exif_sub_directory_keeps_the_main_tags() -> None:
    exif = Image.Exif()
    exif[0x0110] = "Camera"  # Model
    exif[0x8769] = 26  # an Exif sub-IFD pointer Pillow cannot follow here (it raises)
    with pytest.raises(AttributeError):
        exif.get_ifd(0x8769)
    tags = images._exif_dict(types.SimpleNamespace(getexif=lambda: exif))  # type: ignore[arg-type]
    assert tags["Model"] == "Camera" and tags["ExifOffset"] == 26


# --- poses and similarities ----------------------------------------------------------------------


def test_the_camera_centre_is_the_translation_a_copy() -> None:
    pose = Pose(rot_z(0.4), np.array([1.0, 2.0, 3.0]))
    c = pose.center
    assert c.tolist() == [1.0, 2.0, 3.0]
    c[0] = 9.0
    assert pose.t[0] == 1.0
    np.testing.assert_allclose(pose.apply(np.zeros((1, 3)))[0], pose.center)


def test_sim3_identity_and_matrix(rng: np.random.Generator) -> None:
    pts = rng.normal(size=(5, 3))
    ident = Sim3.identity()
    np.testing.assert_array_equal(ident.apply(pts), pts)
    np.testing.assert_array_equal(ident.matrix(), np.eye(4))
    sim = Sim3(2.5, rot_z(0.7), np.array([1.0, -2.0, 0.5]))
    homogeneous = np.c_[pts, np.ones(5)] @ sim.matrix().T
    np.testing.assert_allclose(homogeneous[:, :3], sim.apply(pts))
    assert homogeneous[:, 3].tolist() == [1.0] * 5
    np.testing.assert_allclose(sim.compose(sim.inverse()).matrix(), np.eye(4), atol=1e-12)
