"""The capture file-name grammars of the capture sequences (spec §5), on the real names: the 79
of examples/ainex-captures and the 27 of examples/camera."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from oh_my_slam.tools.evaluate.names import (
    PanCapture,
    captures_in,
    level_siblings,
    mid_siblings,
    pan_captures_in,
    pan_positions,
    parse_capture,
    parse_pan_capture,
    same_heading_pairs,
    tilt_pairs,
    wrap_deg,
)

SEQUENCE = Path(__file__).resolve().parents[2] / "examples" / "ainex-captures"
CAMERA = Path(__file__).resolve().parents[2] / "examples" / "camera"


@pytest.fixture(scope="module")
def captures() -> list:
    return captures_in(SEQUENCE)


def test_all_79_real_names_parse(captures: list) -> None:
    assert len(captures) == 79
    assert [c.index for c in captures] == list(range(1, 80))
    assert Counter(c.tilt for c in captures) == {"level": 48, "up": 16, "down": 15}


@pytest.mark.parametrize(("index", "yaw", "heading"), [
    (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 15, 15), (8, 30, 30), (11, 60, 60), (23, 180, -180),
    (26, 210, -150), (29, 200, -160), (40, 90, 90), (53, 0, 0), (56, -10, -10), (78, -150, -150),
    (79, -150, -150),
])
def test_known_commanded_yaws(captures: list, index: int, yaw: float, heading: float) -> None:
    c = captures[index - 1]
    assert c.index == index
    assert c.yaw_deg == yaw and wrap_deg(c.yaw_deg) == heading


def test_every_motion_form() -> None:
    cases = {
        "001_bootstrap_level.jpg": ("bootstrap", 0.0, "level"),
        "002_bootstrap_side1_level.jpg": ("bootstrap_side1", 0.0, "level"),
        "003_bootstrap_side2_level.jpg": ("bootstrap_side2", 0.0, "level"),
        "005_bootstrap_left015_up.jpg": ("bootstrap_left015", 15.0, "up"),
        "008_bootstrap_left030_side_level.jpg": ("bootstrap_left030_side", 30.0, "level"),
        "013_left_060_down.jpg": ("left_060", 60.0, "down"),
        "044_right_to_060_up.jpg": ("right_to_060", 60.0, "up"),
        "064_right_060_up.jpg": ("right_060", -60.0, "up"),
    }
    for name, (motion, yaw, tilt) in cases.items():
        c = parse_capture(name)
        assert (c.motion, c.yaw_deg, c.tilt) == (motion, yaw, tilt), name


@pytest.mark.parametrize("name", [
    "001_bootstrap.jpg", "01_bootstrap_level.jpg", "001_left_60_level.jpg",
    "001_right_to_060_sideways.jpg", "001_bootstrap_side3_level.jpg", "001_forward_010_level.jpg",
    "001_left_060_level.png",
])
def test_names_outside_the_grammar_are_refused(name: str) -> None:
    with pytest.raises(ValueError, match="not a capture name"):
        parse_capture(name)


def test_hidden_files_and_non_images_are_ignored(tmp_path: Path) -> None:
    for name in ("002_left_015_level.jpg", "001_bootstrap_level.jpg", ".DS_Store", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    assert [c.index for c in captures_in(tmp_path)] == [1, 2]
    (tmp_path / "frame.jpg").write_bytes(b"x")
    with pytest.raises(ValueError):
        captures_in(tmp_path)


def test_tilted_frames_pair_with_the_level_frame_of_the_same_motion(captures: list) -> None:
    sib = {k: v.index for k, v in level_siblings(captures).items()}
    assert len(sib) == 31  # every up/down frame has one
    assert sib["009_bootstrap_left030_up.jpg"] == 7  # not the side-stepped 008
    assert sib["044_right_to_060_up.jpg"] == 43  # not 011 (left_060), a different motion
    assert sib["079_right_150_up.jpg"] == 78


def test_same_heading_pairs_of_the_spec(captures: list) -> None:
    pairs = [(a.index, b.index) for a, b in same_heading_pairs(captures)]
    assert pairs == [(1, 53), (26, 78)]


@pytest.mark.parametrize(("angle", "wrapped"), [
    (0, 0), (190, -170), (-190, 170), (180, -180), (-180, -180), (540, -180), (359, -1),
])
def test_wrap(angle: float, wrapped: float) -> None:
    assert wrap_deg(angle) == pytest.approx(wrapped)


# -- examples/camera ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pan_captures() -> list:
    return pan_captures_in(CAMERA)


def test_all_27_real_camera_names_parse(pan_captures: list) -> None:
    """img_NNN_pPP_<tilt>.jpg: 9 pan positions in pan order, the three tilts of each consecutive
    in capture order (the folder's .DS_Store, when the Finder left one, is not a frame)."""
    assert len(pan_captures) == 27
    assert [c.index for c in pan_captures] == list(range(7, 34))
    groups = pan_positions(pan_captures)
    assert list(groups) == list(range(3, 12))
    for pan, group in groups.items():
        assert sorted(c.tilt for c in group) == ["down", "mid", "up"], pan
        first = group[0].index
        assert [c.index for c in group] == [first, first + 1, first + 2], pan
    assert [c.pan for c in pan_captures] == sorted(c.pan for c in pan_captures)
    assert Counter(c.tilt for c in pan_captures) == {"down": 9, "mid": 9, "up": 9}


def test_camera_name_forms() -> None:
    assert parse_pan_capture("img_007_p03_down.jpg") == PanCapture("img_007_p03_down.jpg", 7, 3,
                                                                   "down")
    c = parse_pan_capture("IMG_021_P07_UP.JPEG")
    assert (c.name, c.index, c.pan, c.tilt) == ("IMG_021_P07_UP.JPEG", 21, 7, "up")


@pytest.mark.parametrize("name", [
    "img_007_p03_level.jpg", "007_p03_mid.jpg", "img_7_p03_mid.jpg", "img_007_p3_mid.jpg",
    "img_007_03_mid.jpg", "img_007_p03_mid.png", "001_bootstrap_level.jpg",
])
def test_camera_names_outside_the_grammar_are_refused(name: str) -> None:
    with pytest.raises(ValueError, match="not a pan-tilt capture name"):
        parse_pan_capture(name)


def test_camera_hidden_files_and_non_images_are_ignored(tmp_path: Path) -> None:
    for name in ("img_002_p01_up.jpg", "img_001_p01_mid.jpg", ".DS_Store", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    assert [c.index for c in pan_captures_in(tmp_path)] == [1, 2]
    (tmp_path / "001_bootstrap_level.jpg").write_bytes(b"x")  # the other sequence's grammar
    with pytest.raises(ValueError, match="not a pan-tilt capture name"):
        pan_captures_in(tmp_path)


def test_tilted_frames_pair_with_the_mid_frame_of_their_pan_position(pan_captures: list) -> None:
    sib = {k: v.name for k, v in mid_siblings(pan_captures).items()}
    assert len(sib) == 18  # every up/down frame has one
    assert sib["img_007_p03_down.jpg"] == "img_008_p03_mid.jpg"
    assert sib["img_010_p04_up.jpg"] == "img_011_p04_mid.jpg"  # the tilts run up-mid-down here
    assert sib["img_033_p11_up.jpg"] == "img_032_p11_mid.jpg"
    lone = [parse_pan_capture("img_001_p01_up.jpg"), parse_pan_capture("img_002_p02_mid.jpg")]
    assert mid_siblings(lone) == {}  # no mid frame at p01


def test_the_tilts_of_a_pan_position_are_its_same_heading_pairs(pan_captures: list) -> None:
    pairs = [(a.index, b.index) for a, b in tilt_pairs(pan_captures)]
    assert len(pairs) == 27  # three per pan position
    assert pairs[:3] == [(7, 8), (7, 9), (8, 9)]
    assert all(a < b and b - a <= 2 for a, b in pairs)
