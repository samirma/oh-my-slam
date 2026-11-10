"""The bird's-eye image (``mapping.birdseye``, the later versioned ``GET birdseye.png``),
offline: its scale and pixel grid, the ceiling it leaves out, the highest point per pixel, the
OBB outlines in the objects' exact display colours, the camera markers and its text metadata,
on images drawn from a synthetic room."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from oh_my_slam.core.ply import PointCloud
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.mapping import birdseye as be
from oh_my_slam.mapping.store import FrameRecord
from oh_my_slam.segmentation.api import SceneObject
from oh_my_slam.segmentation.colors import color_for_id
from oh_my_slam.segmentation.obb import OBB

K = Intrinsics(100.0, 100.0, 80.0, 60.0, 160, 120, "given")
FLOOR, WALL, CEILING = (90, 90, 90), (60, 120, 200), (200, 30, 30)


def cam(yaw: float, pitch: float = 0.0, t: tuple[float, float, float] = (0, 0, 0)) -> Pose:
    """Camera-to-map pose (OpenCV camera axes, z-up map) looking at ``yaw`` / ``pitch``."""
    y, p = np.radians(yaw), np.radians(pitch)
    f = np.array([np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), np.sin(p)])
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return Pose(np.stack([right, np.cross(f, right), f], axis=1), np.asarray(t, float))


def room(ceiling: bool = True, step: float = 0.04) -> tuple[np.ndarray, np.ndarray]:
    """A 4 x 4 m room, floor at -1.5 m, walls up to 1.0 m, the cameras at 0: its points and
    colours."""
    g = np.arange(-2.0, 2.0 + 1e-9, step)
    u, v = np.meshgrid(g, g)
    h = np.arange(-1.5, 1.0, step)
    a, z = np.meshgrid(g, h)
    parts = [(np.column_stack([u.ravel(), v.ravel(), np.full(u.size, -1.5)]), FLOOR)]
    for x in (-2.0, 2.0):
        parts.append((np.column_stack([np.full(a.size, x), a.ravel(), z.ravel()]), WALL))
        parts.append((np.column_stack([a.ravel(), np.full(a.size, x), z.ravel()]), WALL))
    if ceiling:
        parts.append((np.column_stack([u.ravel(), v.ravel(), np.full(u.size, 1.0)]), CEILING))
    xyz = np.concatenate([p for p, _ in parts])
    rgb = np.concatenate([np.tile(c, (len(p), 1)) for p, c in parts]).astype(np.uint8)
    return xyz, rgb


def record(index: int, pose: Pose, update_id: int = 1) -> FrameRecord:
    return FrameRecord(index, f"f{index:06d}", f"frames/f{index:06d}.jpg", f"/in/{index}.jpg", 1,
                       K.width, K.height, K, pose, K.width, K.height, update_id=update_id)


def obj(oid: int, centre: tuple[float, float], size: tuple[float, float], yaw: float = 0.0
        ) -> SceneObject:
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return SceneObject(oid, "box", 0.9, OBB(np.array([*centre, -1.0]), R,
                                           np.array([*size, 1.0])), 10, 10)


OBJECTS = [obj(1, (0.8, -0.6), (0.6, 0.4), 30.0), obj(2, (-1.2, 1.9), (0.5, 0.2)),
           obj(23, (1.5, 1.2), (0.3, 0.3))]
FRAMES = [record(0, cam(0.0)), record(1, cam(90.0, t=(0.3, 0.1, 0.0))),
          record(2, cam(200.0, t=(-0.4, -0.3, 0.05)), update_id=2)]


@pytest.fixture
def scene(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):  # type: ignore[no-untyped-def]
    """A map reader of the synthetic room (its points, objects and keyframes), read through the
    code of mapping.export as a persisted map is."""
    def make(ceiling: bool = True, objects: list[SceneObject] = OBJECTS,
             frames: list[FrameRecord] = FRAMES, points: bool = True) -> Any:
        xyz, rgb = room(ceiling) if points else (np.zeros((0, 3)), np.zeros((0, 3), np.uint8))
        monkeypatch.setattr(be.export, "map_objects", lambda reader: (None, objects))
        monkeypatch.setattr(be.export, "map_cloud", lambda reader: PointCloud(
            xyz, rgb, np.zeros(len(xyz), np.int32)))
        return SimpleNamespace(root=tmp_path / "room", frames=frames,
                               meta={"next_frame_index": len(frames)})
    return make


def located(k: int, pose: Pose) -> be.Camera:
    """A located camera as the image draws it, tagged with its image's number (``k + 1``)."""
    t = pose.t.tolist()
    return be.Camera(f"located_{k}", "located", pose,
                     {"located": True, "inliers": 40, "translation": t}, str(k + 1))


def keyframes(frames: list[FrameRecord] = FRAMES) -> list[be.Camera]:
    return [be.Camera(fr.name, "keyframe", fr.T_map_cam,
                      {"keyframe": fr.name, "frame": fr.index, "source": fr.source,
                       "translation": fr.T_map_cam.t.tolist()}) for fr in frames]


def rgb_of(png: bytes) -> tuple[np.ndarray, dict[str, str]]:
    with Image.open(io.BytesIO(png)) as img:
        return np.asarray(img.convert("RGB")), dict(img.text)  # type: ignore[attr-defined]


def pixel(text: dict[str, str], x: float, y: float) -> tuple[int, int]:
    M = np.asarray(json.loads(text["map_to_pixel"]))
    u, v = M @ [x, y, 1.0]
    return int(np.floor(v)), int(np.floor(u))  # row, column


def colours(rgb: np.ndarray) -> set[tuple[int, ...]]:
    return {tuple(int(c) for c in px) for px in np.unique(rgb.reshape(-1, 3), axis=0)}


# -- scale and grid --------------------------------------------------------------------------------


def test_the_scale_is_a_1_2_5_step_from_1_cm_up() -> None:
    assert be.nice_scale(0.0) == 0.01 and be.nice_scale(20.0) == 0.01  # 2000 px at 1 cm
    assert be.nice_scale(20.5) == 0.02 and be.nice_scale(90.0) == 0.05
    assert be.nice_scale(150.0) == 0.1 and be.nice_scale(3000.0) == 2.0


def test_the_grid_covers_the_points_with_its_margin_and_maps_both_ways() -> None:
    xy = np.array([[-1.234, 0.5], [2.0, -0.75], [0.004, 0.001]])
    g = be.grid_for(xy)
    assert g.s == 0.01 and (g.x0, g.y0) == (-1.48, 0.75)  # aligned to the scale
    uv = g.pixels(xy)
    m = be.MARGIN_PX
    # inside, with the margin around (a point on a cell's lower edge is drawn in the row below)
    assert np.all(uv >= m) and np.all(uv[:, 0] < g.width - m) and np.all(uv[:, 1] <= g.height - m)
    assert np.all(np.floor(uv) < [g.width, g.height])
    to_map = np.vstack([g.pixel_to_map(), [0, 0, 1]])
    to_px = np.vstack([g.map_to_pixel(), [0, 0, 1]])
    np.testing.assert_allclose(to_map @ to_px, np.eye(3), atol=1e-12)
    one = be.grid_for(np.array([[5.0, 5.0]]))  # a single point: one pixel and the margins
    assert (one.width, one.height) == (1 + 2 * m, 1 + 2 * m)
    assert be.Grid(0.0, 0.0, 0.5, 1, 1).map_to_pixel() == [[2.0, 0.0, 0.0], [0.0, -2.0, 0.0]]


# -- the ceiling -------------------------------------------------------------------------------------


def test_a_ceiling_above_the_cameras_is_recognised() -> None:
    xyz, _ = room()
    cut = be.ceiling_cut(xyz, 0.05)
    assert cut is not None and 0.6 < cut < 1.0  # under the ceiling, above the cameras
    assert np.all(xyz[xyz[:, 2] >= cut][:, 2] > 0.5)  # only the walls' tops go with it
    no_ceiling, _ = room(ceiling=False)
    assert be.ceiling_cut(no_ceiling, 0.05) is None  # walls alone: no dense layer
    assert be.ceiling_cut(xyz, None) is None  # no camera: nothing is known to be above
    assert be.ceiling_cut(np.zeros((0, 3)), 0.0) is None
    assert be.ceiling_cut(xyz, 2.0) is None  # nothing above the cameras
    assert be.ceiling_cut(xyz, 0.98 - be.CEILING_CLEARANCE_M) is None  # fewer bins than a layer
    # a table top above the cameras (a robot's view): dense, but not wide enough to be a ceiling
    g = np.arange(-0.3, 0.3, 0.01)
    u, v = np.meshgrid(g, g)
    table = np.column_stack([u.ravel(), v.ravel(), np.full(u.size, 0.7)])
    assert be.ceiling_cut(np.concatenate([no_ceiling, table]), 0.05) is None


def test_each_pixel_shows_its_highest_point() -> None:
    g = be.Grid(0.0, 1.0, 0.1, 10, 10)
    xyz = np.array([[0.05, 0.95, 0.0], [0.06, 0.94, 1.0], [0.07, 0.93, 0.5],
                    [0.55, 0.55, 2.0], [0.56, 0.56, 2.0]])
    rgb = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]])
    pix, col = be.highest_rgb(g, xyz, rgb)
    assert dict(zip(pix.tolist(), col.tolist(), strict=True)) == {0: [2, 2, 2], 45: [5, 5, 5]}


def test_the_heading_is_the_viewing_direction_on_the_floor() -> None:
    np.testing.assert_allclose(be.heading(cam(90.0, pitch=30.0)), [0.0, 1.0], atol=1e-12)
    down = cam(0.0, pitch=-90.0)  # straight down: the image's top tells the way it faces
    np.testing.assert_allclose(be.heading(down), [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(be.heading(Pose.identity()), [0.0, -1.0], atol=1e-12)


def test_the_footprint_is_the_upright_box_s_rectangle() -> None:
    o = obj(5, (1.0, 2.0), (0.4, 0.2), 90.0)
    np.testing.assert_allclose(be.footprint(o), [[1.1, 1.8], [1.1, 2.2], [0.9, 2.2], [0.9, 1.8]],
                               atol=1e-12)


# -- the image ---------------------------------------------------------------------------------------


def test_the_image_draws_the_cloud_and_objects_under_the_cameras(scene: Any) -> None:
    reader = scene()
    cams = [*keyframes(), located(0, cam(45.0, t=(1.0, 1.0, 0.0)))]
    png = be.birdseye_png(reader, cams)
    assert png == be.birdseye_png(reader, cams)  # deterministic
    rgb, text = rgb_of(png)
    with Image.open(io.BytesIO(png)) as img:
        assert img.mode == "RGB" and img.size == (rgb.shape[1], rgb.shape[0])
    assert set(text) <= {"Description", *be.METADATA} and "layers" not in text
    assert json.loads(text["metres_per_pixel"]) == 0.01
    assert "orthographic projection along the map's +z" in text["Description"]
    assert "showing its point cloud and its objects' OBBs" in text["Description"]
    assert text["map"].endswith("room")
    cut = json.loads(text["ceiling_cut_z"])
    assert 0.6 < cut < 1.0
    seen = colours(rgb)
    assert CEILING not in seen and FLOOR in seen and WALL in seen  # the floor plan, no ceiling
    assert all(color_for_id(o.id) in seen for o in OBJECTS)  # every outline, exact colours
    meta = json.loads(text["cameras"])
    assert list(meta) == ["f000000", "f000001", "f000002", "located_0"]
    assert meta["located_0"]["located"] and meta["located_0"]["inliers"] == 40
    assert meta["f000001"]["frame"] == 1 and meta["f000001"]["source"] == "/in/1.jpg"
    for key, c in (("located_0", be.CAMERA_FILL), ("f000000", be.KEYFRAME_FILL)):
        t = meta[key]["translation"]
        assert tuple(rgb[pixel(text, t[0], t[1])]) == c, key
    # the legend: every mark drawn, in the order it is drawn, with its shape and colours
    legend = json.loads(text["legend"])
    assert [m["mark"] for m in legend] == ["cloud", "obb", "keyframe", "located"]
    assert [m["shape"] for m in legend] == ["points", "outline", "disc", "triangle"]
    assert legend[2]["fill"] == "#c9d1dc" and legend[3]["fill"] == "#ffffff"
    assert all(m["text"] for m in legend)
    alone = rgb_of(be.birdseye_png(reader, cams[-1:]))[1]
    assert list(json.loads(alone["cameras"])) == ["located_0"]
    assert [m["mark"] for m in json.loads(alone["legend"])] == ["cloud", "obb", "located"]
    assert alone["pixel_to_map"] == text["pixel_to_map"]  # one grid for both


def test_an_empty_map_still_shows_its_cameras(scene: Any) -> None:
    """No point, object or keyframe: the image covers the cameras alone, no ceiling is
    sought."""
    reader = scene(objects=[], frames=[], points=False)
    be_cam = be.Camera("located_0", "located", cam(10.0, t=(3.0, -2.0, 0.0)), {})
    rgb, text = rgb_of(be.birdseye_png(reader, [be_cam]))
    assert json.loads(text["ceiling_cut_z"]) is None
    assert rgb.shape[:2] == (1 + 2 * be.MARGIN_PX, 1 + 2 * be.MARGIN_PX)
    assert tuple(rgb[pixel(text, 3.0, -2.0)]) == be.CAMERA_FILL


def test_located_cameras_are_tagged_with_their_inputs_numbers_apart(
        scene: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each located camera is tagged with the number of its image among the inputs (located_<k>:
    k + 1), black on white, where no other camera's tag is — three cameras a few centimetres
    apart keep three tags apart — and the description says what the image shows, in words."""
    reader = scene()
    cams = [located(k, cam(10.0 * k, t=(0.5 + 0.03 * k, 0.5, 0.0))) for k in range(3)]
    placed: list[tuple[int, int, int, int]] = []
    tag_box = be.tag_box
    monkeypatch.setattr(be, "tag_box", lambda *a: placed.append(tag_box(*a)) or placed[-1])
    tags: list[str] = []
    draw_tag = be._draw_tag
    monkeypatch.setattr(be, "_draw_tag", lambda d, g, c, t: (tags.append(c.tag), draw_tag(d, g, c, t)))
    rgb, text = rgb_of(be.birdseye_png(reader, cams))
    assert tags == ["1", "2", "3"] and len(placed) == 3
    assert not any(a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]
                   for i, a in enumerate(placed) for b in placed[i + 1:])  # none over another
    for x0, y0, x1, y1 in placed:  # each a white tag with a black edge
        assert tuple(rgb[y0, x0]) == be.CAMERA_EDGE and be.CAMERA_FILL in colours(
            rgb[y0 + 1:y1, x0 + 1:x1])
    assert text["Description"].startswith("Bird's-eye image of the map room, seen from straight "
                                          "above (an orthographic projection along the map's +z")
    assert "under the cameras" in text["Description"]
    assert "pixel_to_map" not in text["Description"] and ";" not in text["Description"]
    # the keyframes have no tag
    tags.clear()
    be.birdseye_png(reader, keyframes())
    assert tags == []
    # with nowhere free, a tag takes its first place, inside the image
    assert tag_box(np.array([5.0, 5.0]), 6, 6, [(0, 0, 999, 999)], 40, 40) == (2, 15, 7, 20)


def test_a_tag_stays_inside_the_image(scene: Any) -> None:
    """An object at the top edge has its id above it clamped into the image, drawn in its
    colour."""
    reader = scene(points=False)
    rgb, text = rgb_of(be.birdseye_png(reader, []))
    top = rgb[:be.MARGIN_PX + 5]
    assert np.any(np.all(top == color_for_id(2), axis=-1))  # object 2 lies at y = 1.9..2.0


def test_a_tag_never_hides_another_object_s_outline(scene: Any) -> None:
    """A small object right where a neighbour's id tag goes (a cluttered desk) keeps its whole
    outline: the tags lie under every outline."""
    big = obj(1, (0.0, 0.0), (0.6, 0.4))
    small = obj(7, (0.0, 0.29), (0.05, 0.04))  # under the tag above object 1's footprint
    reader = scene(objects=[big, small], points=False)
    rgb, text = rgb_of(be.birdseye_png(reader, []))
    for x, y in ((-0.025, 0.29), (0.025, 0.29), (0.0, 0.27), (0.0, 0.31)):  # its four edges
        r, c = pixel(text, x, y)
        window = rgb[r - 1:r + 2, c - 1:c + 2].reshape(-1, 3)
        assert any(tuple(int(v) for v in px) == color_for_id(7) for px in window), (x, y)
