"""Command-line contracts on folders and files, through the shell scripts: a non-empty folder that
is not a map is refused and left untouched (spec 2.3, a `.git`-only folder included), and an
unwritable ``-o`` / ``-d`` target is a usage error (exit 2) before the server is contacted or any
work starts."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from oh_my_slam.core.atomic import preflight_dir, preflight_file
from oh_my_slam.core.errors import UsageError
from oh_my_slam.core.log import PayloadWriter
from tests.unit.test_view_cli import minimal_map, sh


def snapshot(root: Path) -> list[tuple[str, bytes | None]]:
    """Every entry of a folder, hidden ones included, with its bytes."""
    return [(str(p.relative_to(root)), p.read_bytes() if p.is_file() else None)
            for p in sorted(root.rglob("*"))]


@pytest.fixture(scope="module")
def image(tmp_path_factory: pytest.TempPathFactory) -> Path:
    p = tmp_path_factory.mktemp("img") / "photo.jpg"
    rng = np.random.default_rng(5)
    Image.fromarray(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)).save(p, quality=95)
    return p


@pytest.fixture
def stub_server() -> Iterator[None]:
    res = sh("start_inference_server.sh", "--stub")
    assert res.returncode == 0, res.stderr.decode()
    try:
        yield
    finally:
        sh("start_inference_server.sh", "--stop")


@pytest.fixture
def not_a_folder(tmp_path: Path) -> Path:
    """A regular file, so nothing can be created below it, whoever runs the test."""
    f = tmp_path / "plain_file"
    f.write_text("x")
    return f


def _non_map_folders(tmp_path: Path) -> dict[str, Path]:
    files = tmp_path / "files"
    files.mkdir()
    (files / "a.txt").write_text("x")
    dotgit = tmp_path / "dotgit"
    (dotgit / ".git" / "objects").mkdir(parents=True)
    (dotgit / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    (hidden / ".env").write_text("SECRET=1\n")
    return {"visible file": files, "only .git": dotgit, "only a hidden file": hidden}


def test_mapper_refuses_a_non_empty_folder_that_is_not_a_map(stub_server: None, tmp_path: Path,
                                                             image: Path) -> None:
    """Exit 4, stdout empty, folder byte-identical and no lock file left in it."""
    for name, folder in _non_map_folders(tmp_path).items():
        before = snapshot(folder)
        res = sh("mapper.sh", "update", "-i", str(image), "-m", str(folder))
        assert res.returncode == 4, (name, res.stderr.decode())
        assert res.stdout == b"" and b"not empty and not a map" in res.stderr, name
        assert snapshot(folder) == before, name


def test_mapper_refuses_a_non_map_folder_with_the_server_down(tmp_path: Path,
                                                               image: Path) -> None:
    """No server runs here: the refusal (exit 4) comes before the server is contacted, and the
    folder is untouched (no lock file). A new or empty folder still needs the server (exit 3)."""
    for name, folder in _non_map_folders(tmp_path).items():
        before = snapshot(folder)
        res = sh("mapper.sh", "update", "-i", str(image), "-m", str(folder))
        assert res.returncode == 4, (name, res.stderr.decode())
        assert res.stdout == b"" and b"not empty and not a map" in res.stderr, name
        assert snapshot(folder) == before, name
    empty = tmp_path / "empty"
    empty.mkdir()
    for folder in (empty, tmp_path / "new"):
        res = sh("mapper.sh", "update", "-i", str(image), "-m", str(folder))
        assert res.returncode == 3, res.stderr.decode()


@pytest.mark.parametrize("args", [
    ["segment.sh", "-i", "{image}", "-o", "{bad}/out.json"],
    ["segment.sh", "-i", "{image}", "-d", "{bad}/out"],
    ["segment.sh", "-m", "{map}", "-o", "{bad}/out.json"],
    ["segment.sh", "-m", "{map}", "-d", "{bad}/out"],
    ["reconstruct.sh", "-i", "{image}", "-o", "{bad}/out.json"],
    ["mapper.sh", "update", "-i", "{image}", "-m", "{fresh}", "-o", "{bad}/out.json"],
])
def test_unwritable_output_is_a_usage_error_before_any_work(
        args: list[str], tmp_path: Path, image: Path, not_a_folder: Path) -> None:
    """No server runs here: exit 2 (not 3) proves the target is checked before the server is
    contacted. A map is not created, and an existing one is not touched."""
    map_dir = minimal_map(tmp_path / "map")
    fresh = tmp_path / "fresh"
    before = snapshot(map_dir)
    argv = [a.format(image=image, bad=not_a_folder, map=map_dir, fresh=fresh) for a in args]
    res = sh(*argv)
    assert res.returncode == 2, res.stderr.decode()
    assert res.stdout == b"" and str(not_a_folder) in res.stderr.decode()
    assert not fresh.exists()
    assert snapshot(map_dir) == before


def test_preflight_creates_missing_parents_and_names_the_path(tmp_path: Path,
                                                              not_a_folder: Path) -> None:
    target = tmp_path / "a" / "b" / "scene.json"
    preflight_file(target, "-o")
    assert target.parent.is_dir() and not target.exists()  # parents created, nothing written
    assert list(target.parent.iterdir()) == []  # the probe file is gone
    preflight_dir(tmp_path / "c" / "d", "-d")
    assert (tmp_path / "c" / "d").is_dir()
    with pytest.raises(UsageError, match="plain_file"):
        preflight_dir(not_a_folder / "x", "-d")
    with pytest.raises(UsageError, match="is a folder"):
        preflight_file(tmp_path, "-o")
    with pytest.raises(UsageError, match="plain_file"):
        PayloadWriter(path=not_a_folder / "x" / "y.json")


def test_output_to_a_device_or_fifo_is_written_in_place(tmp_path: Path) -> None:
    """``-o /dev/null`` and a named pipe cannot be replaced by a rename: they are written to."""
    preflight_file(Path(os.devnull), "-o")
    writer = PayloadWriter(path=Path(os.devnull))
    writer.write_bytes(b"{}\n")
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    reader = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
    try:
        PayloadWriter(path=fifo).write_bytes(b"{}\n")
        assert os.read(reader, 16) == b"{}\n"
    finally:
        os.close(reader)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pipe"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write anywhere")
def test_preflight_rejects_a_read_only_folder(tmp_path: Path) -> None:
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        with pytest.raises(UsageError, match="cannot write"):
            preflight_file(ro / "out.json", "-o")
    finally:
        ro.chmod(0o700)

