"""The bird's-eye image of ``mapper.sh locate -f png`` (specs/mapper.md "Bird's-eye image"),
offline: its scale and pixel grid, the ceiling it leaves out, the highest point per pixel, the
OBB outlines in the colour contract's exact colours, the camera markers, its text metadata — and
the evaluator's check of a bird's-eye image against the JSON result of the same call (spec §5
"Contracts"), on images drawn by the command's own code from a synthetic room."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from oh_my_slam.core.cloud_attrs import CloudAttrs
from oh_my_slam.core.ply import PointCloud
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.mapping import birdseye as be
from oh_my_slam.mapping import locate as lo
from oh_my_slam.mapping.store import FrameRecord
from oh_my_slam.segmentation.api import SceneObject
from oh_my_slam.segmentation.colors import color_for_id, color_hex_for_id
from oh_my_slam.segmentation.obb import OBB
from oh_my_slam.tools.evaluate.contracts import birdseye_problems, doc_cameras
from tests.unit.test_evaluate_metrics import cam

K = Intrinsics(100.0, 100.0, 80.0, 60.0, 160, 120, "given")
FLOOR, WALL, CEILING = (90, 90, 90), (60, 120, 200), (200, 30, 30)


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


def located(k: int, pose: Pose) -> lo.ResultCamera:
    return lo.located_cameras(SimpleNamespace(frames=FRAMES, meta={}), [
        lo.Located(Path(f"/q/{k}.jpg"), k, pose, K, inliers=40)])[0]


def rgb_of(png: bytes) -> tuple[np.ndarray, dict[str, str]]:
    with Image.open(io.BytesIO(png)) as img:
        return np.asarray(img.convert("RGB")), dict(img.text)  # type: ignore[attr-defined]


def pixel(text: dict[str, str], x: float, y: float) -> tuple[int, int]:
    M = np.asarray(json.loads(text["map_to_pixel"]))
    u, v = M @ [x, y, 1.0]
    return int(np.floor(v)), int(np.floor(u))  # row, column


def colours(rgb: np.ndarray) -> set[tuple[int, ...]]:
    return {tuple(int(c) for c in px) for px in np.unique(rgb.reshape(-1, 3), axis=0)}


def png_with(png: bytes, rgb: np.ndarray | None = None, **text: Any) -> bytes:
    """``png`` with other pixels and text metadata (a key set to None is left out)."""
    old_rgb, old = rgb_of(png)
    info = PngImagePlugin.PngInfo()
    for k, v in {**old, **text}.items():
        if v is not None:
            info.add_text(k, v if isinstance(v, str) else json.dumps(v))
    buf = io.BytesIO()
    Image.fromarray(old_rgb if rgb is None else rgb).save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


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


def test_the_image_draws_the_layers_under_the_cameras(scene: Any) -> None:
    reader = scene()
    cams = [located(0, cam(45.0, t=(1.0, 1.0, 0.0))), lo.last_known_camera(reader)]
    png = lo.png_payload(reader, cams, "full", ("cloud", "obb"))
    assert png == lo.png_payload(reader, cams, "full", ("cloud", "obb"))  # deterministic
    rgb, text = rgb_of(png)
    with Image.open(io.BytesIO(png)) as img:
        assert img.mode == "RGB" and img.size == (rgb.shape[1], rgb.shape[0])
    assert set(text) == {"Description", *be.METADATA}
    assert text["layers"] == "cloud,obb" and json.loads(text["metres_per_pixel"]) == 0.01
    assert "orthographic projection along the map's +z" in text["Description"]
    assert text["map"].endswith("room")
    cut = json.loads(text["ceiling_cut_z"])
    assert 0.6 < cut < 1.0
    seen = colours(rgb)
    assert CEILING not in seen and FLOOR in seen and WALL in seen  # the floor plan, no ceiling
    assert all(color_for_id(o.id) in seen for o in OBJECTS)  # every outline, exact colours
    # the cameras' metadata: their JSON representation, the keyframes' too (-t full)
    meta = json.loads(text["cameras"])
    assert list(meta) == ["f000000", "f000001", "f000002", "located_0", "last_known"]
    assert meta["last_known"]["last_known"] and meta["last_known"]["keyframe"] == "f000002"
    assert meta["located_0"]["located"] and meta["located_0"]["inliers"] == 40
    assert meta["f000001"]["frame"] == 1 and meta["f000001"]["source"] == "/in/1.jpg"
    for key, c in (("located_0", be.CAMERA_FILL), ("last_known", be.CAMERA_FILL),
                   ("f000000", be.KEYFRAME_FILL)):
        t = meta[key]["transform_src_to_dst"]["translation"]
        assert tuple(rgb[pixel(text, t[0], t[1])]) == c, key
    # the legend: every mark drawn, in the order it is drawn, with its shape and colours
    legend = json.loads(text["legend"])
    assert [m["mark"] for m in legend] == ["cloud", "obb", "keyframe", "last_known", "located"]
    assert [m["shape"] for m in legend] == ["points", "outline", "disc", "diamond", "triangle"]
    assert legend[2]["fill"] == "#c9d1dc" and legend[4]["fill"] == "#ffffff"
    assert all(m["text"] for m in legend)
    single = lo.png_payload(reader, cams, "single", ("cloud", "obb"))
    assert list(json.loads(rgb_of(single)[1]["cameras"])) == ["located_0", "last_known"]
    assert [m["mark"] for m in json.loads(rgb_of(single)[1]["legend"])] == [
        "cloud", "obb", "last_known", "located"]  # no keyframe drawn
    assert rgb_of(single)[1]["pixel_to_map"] == text["pixel_to_map"]  # one grid for both


def test_one_layer_or_the_other(scene: Any) -> None:
    reader = scene()
    cams = [lo.last_known_camera(reader)]
    obb = rgb_of(lo.png_payload(reader, cams, "single", ("obb",)))[0]
    allowed = {be.BACKGROUND, be.CAMERA_FILL, be.CAMERA_EDGE, *(color_for_id(o.id) for o in OBJECTS)}
    assert colours(obb) == allowed  # opaque, not anti-aliased: no other colour
    cloud, text = rgb_of(lo.png_payload(reader, cams, "single", ("cloud",)))
    assert text["layers"] == "cloud"
    assert [m["mark"] for m in json.loads(text["legend"])] == ["cloud", "last_known"]
    assert not any(color_for_id(o.id) in colours(cloud) for o in OBJECTS)
    for bad in ((), ("roof",), ("cloud", "walls")):
        with pytest.raises(ValueError, match="layers must be some of"):
            be.birdseye_png(reader, [], bad)


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
    rgb, text = rgb_of(lo.png_payload(reader, cams, "single", ("obb",)))
    assert tags == ["1", "2", "3"] and len(placed) == 3
    assert not any(a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]
                   for i, a in enumerate(placed) for b in placed[i + 1:])  # none over another
    for x0, y0, x1, y1 in placed:  # each a white tag with a black edge
        assert tuple(rgb[y0, x0]) == be.CAMERA_EDGE and be.CAMERA_FILL in colours(
            rgb[y0 + 1:y1, x0 + 1:x1])
    assert colours(rgb) <= {be.BACKGROUND, be.CAMERA_FILL, be.CAMERA_EDGE,
                            *(color_for_id(o.id) for o in OBJECTS)}  # opaque, exact colours
    assert text["Description"].startswith("Bird's-eye image of the map room, seen from straight "
                                          "above (an orthographic projection along the map's +z")
    assert "showing its objects' OBBs under the cameras" in text["Description"]
    assert "pixel_to_map" not in text["Description"] and ";" not in text["Description"]
    # the last known camera and the keyframes have no tag
    tags.clear()
    lo.png_payload(reader, [lo.last_known_camera(reader)], "full", ("obb",))
    assert tags == []
    # with nowhere free, a tag takes its first place, inside the image
    assert tag_box(np.array([5.0, 5.0]), 6, 6, [(0, 0, 999, 999)], 40, 40) == (2, 15, 7, 20)


def test_a_tag_stays_inside_the_image(scene: Any) -> None:
    """An object at the top edge has its id above it clamped into the image, drawn in its
    colour."""
    reader = scene()
    rgb, text = rgb_of(lo.png_payload(reader, [], "single", ("obb",)))
    top = rgb[:be.MARGIN_PX + 5]
    assert np.any(np.all(top == color_for_id(2), axis=-1))  # object 2 lies at y = 1.9..2.0


# -- the evaluator's check -----------------------------------------------------------------------------


@pytest.fixture
def pair(scene: Any) -> tuple[bytes, dict[str, Any], bytes]:
    """(bird's-eye image, the JSON result of the same call, the image of the OBB layer alone),
    -t full, by the command's own code."""
    reader = scene()
    cams = [located(0, cam(45.0, t=(1.0, 1.0, 0.0))), located(2, cam(-30.0, t=(-1.0, 0.5, 0.0)))]
    doc = json.loads(lo.payload(reader, cams, "full", "json", CloudAttrs(), ()))
    png = lo.payload(reader, cams, "full", "png", CloudAttrs(), ("cloud", "obb"))
    obb = lo.payload(reader, cams, "full", "png", CloudAttrs(), ("obb",))
    return png, doc, obb


def test_an_image_agrees_with_the_json_of_the_same_call(pair: Any) -> None:
    png, doc, obb = pair
    assert set(doc_cameras(doc)) == {"located_0", "located_2", "f000000", "f000001", "f000002"}
    assert birdseye_problems(png, doc) == [] and birdseye_problems(obb, doc) == []
    cloud = png_with(png, rgb=rgb_of(png)[0], layers="cloud")  # no outline is looked for
    hidden = rgb_of(png)[0].copy()
    hidden[np.all(hidden == color_for_id(23), axis=-1)] = be.BACKGROUND
    assert birdseye_problems(png_with(cloud, rgb=hidden), doc) == []
    # a -t single result lists no object: the palette is not checked against it
    single = json.loads(json.dumps(doc))
    single["openlabel"]["metadata"]["scope"] = "single"
    single["openlabel"]["objects"] = {}
    assert birdseye_problems(obb, single) == []


def test_the_cameras_of_a_json_result_by_their_image_keys() -> None:
    """Located and last known cameras by their transform's source, keyframes by their name; a
    frame with no pose (a single image's) or a pose of no such camera is none."""
    T = {"quaternion": [0, 0, 0, 1], "translation": [1.0, 2.0, 3.0]}

    def fr(src: str, **props: Any) -> dict[str, Any]:
        return {"frame_properties": {"transforms": {f"{src}_to_map": {
            "src": src, "dst": "map", "transform_src_to_dst": T}}, **props}}

    doc = {"openlabel": {"frames": {"0": {"frame_properties": {"timestamp": 0}},
                                    "1": fr("camera_1"), "2": fr("camera_1", keyframe="f000002"),
                                    "3": fr("located_0", located=True),
                                    "4": fr("last_known", last_known=True, keyframe="f000002")}}}
    assert {k: (kind, t.tolist()) for k, (kind, t) in doc_cameras(doc).items()} == {
        "f000002": ("keyframe", [1, 2, 3]), "located_0": ("located", [1, 2, 3]),
        "last_known": ("last_known", [1, 2, 3])}
    assert doc_cameras({}) == {}


def test_what_disagrees_is_told(pair: Any) -> None:
    png, doc, obb = pair
    assert "PNG signature" in birdseye_problems(b"{}", doc)[0]
    grey = io.BytesIO()
    Image.new("L", (4, 4)).save(grey, format="PNG")
    assert birdseye_problems(grey.getvalue(), doc) == ["a L image, not 8-bit RGB"]
    assert "does not map pixels" in birdseye_problems(png_with(png, cameras=None), doc)[0]
    assert "does not map pixels" in birdseye_problems(png_with(png, cameras=[1]), doc)[0]
    _, text = rgb_of(png)
    M = json.loads(text["map_to_pixel"])
    shifted = [[M[0][0], 0, M[0][2] + 50], M[1]]
    out = birdseye_problems(png_with(png, map_to_pixel=shifted), doc)
    assert out[0] == "map_to_pixel is not the inverse of pixel_to_map"
    assert any("is not drawn at pixel" in p for p in out)  # every marker is elsewhere now
    cams = json.loads(text["cameras"])
    moved = {**cams, "located_2": {**cams["located_2"], "transform_src_to_dst": {
        "quaternion": [0, 0, 0, 1], "translation": [9.0, 9.0, 0.0]}}}
    del moved["f000001"]
    moved["located_9"] = {}
    out = birdseye_problems(png_with(png, cameras=moved), doc)
    assert out[0] == ("cameras ['f000001', 'located_9'] are not in both the image's metadata and "
                      "the JSON result")
    assert "camera located_2: its metadata position is not the JSON's" in out[1]
    no_pose = {**cams, "located_0": {"transform_src_to_dst": {"quaternion": [0, 0, 0, 1]}}}
    assert "camera located_0" in birdseye_problems(png_with(png, cameras=no_pose), doc)[0]
    rgb, _ = rgb_of(png)
    erased = rgb.copy()
    r, c = pixel(text, 1.0, 1.0)
    erased[r - 3:r + 4, c - 3:c + 4] = be.BACKGROUND
    (gone,) = birdseye_problems(png_with(png, rgb=erased), doc)
    assert gone.startswith(f"camera located_0 (located) is not drawn at pixel ({c}.")
    hidden = rgb.copy()
    hidden[np.all(hidden == color_for_id(23), axis=-1)] = be.BACKGROUND
    assert birdseye_problems(png_with(png, rgb=hidden), doc) == [
        f"object 23: its footprint is outlined in {color_hex_for_id(23)} at 0 of "
        f"{len(edge_samples(text, doc, 23))} points of its edges"]
    blended = rgb_of(obb)[0].copy()
    blended[0, 0] = (17, 18, 19)
    assert birdseye_problems(png_with(obb, rgb=blended), doc) == [
        "1 colours are neither the background, an object's nor a marker's (blended?): "
        "['#111213']"]
    flat = json.loads(json.dumps(doc))
    flat["openlabel"]["objects"]["23"]["object_data"].pop("cuboid")
    assert "object 23 has no cuboid" in birdseye_problems(png, flat)


def edge_samples(text: dict[str, str], doc: dict[str, Any], oid: int) -> list[Any]:
    """The points the check samples along an object's footprint edges (one per pixel)."""
    from oh_my_slam.tools.evaluate.scene import doc_objects

    o = next(o for o in doc_objects(doc) if o.id == oid)
    box = o.obb()
    assert box is not None
    hw, hd = box.size[0] / 2, box.size[1] / 2
    corners = box.center[:2] + np.array([[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd]]) \
        @ box.R[:2, :2].T
    M = np.asarray(json.loads(text["map_to_pixel"]))
    px = (M[:, :2] @ corners.T).T + M[:, 2]
    return [k for a, b in zip(px, np.roll(px, -1, axis=0), strict=True)
            for k in range(max(int(np.ceil(np.hypot(*(b - a)))), 1))]
