"""Objects, surfaces and relations of an anchored map (``mapping.anchored_objects``) on a
ray-cast table scene (``tests.synth.table``) with exact depth and exact masks: the surface
heights exactly, ``supported_by`` and ``inside_of``, the observed colour, latest wins (struck,
moved), carving, stable ids, persistence."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oh_my_slam.mapping import anchored_objects as ao
from oh_my_slam.mapping.anchored import Cloud, edge_free
from oh_my_slam.mapping.anchored_objects import Box, MapObject, ObjectMap, Sighting, View
from oh_my_slam.reconstruction.pointcloud import pixel_points
from oh_my_slam.segmentation.api import OBB, Detection
from tests.synth.table import K_TABLE, TOP, Item, TableScene, container, default_items, look_at

EYES = [(0.6, -0.9, 1.5), (1.9, 0.4, 1.5), (0.6, 1.8, 1.5), (-0.7, 0.4, 1.5), (1.6, -0.6, 1.3)]
TARGET = (0.6, 0.4, 0.75)


def detections(scene: TableScene, idx: np.ndarray, score: float = 0.9) -> list[Detection]:
    out = []
    for j, it in enumerate(scene.items):
        m = idx == j
        if m.sum() < 64:
            continue
        r, c = np.nonzero(m)
        out.append(Detection(it.label, score, "test", m,
                             (float(c.min()), float(r.min()), float(c.max()), float(r.max()))))
    return out


def view(kid: str, eye: tuple[float, float, float], scene: TableScene, cloud: Cloud | None = None,
         detect: bool = True, target: tuple[float, float, float] = TARGET) -> View:
    T = look_at(eye, target)
    rgb, z, idx = scene.render(T)
    if cloud is not None:
        ok = edge_free(z)
        lat = np.zeros_like(ok)
        lat[::2, ::2] = True
        pts, fl = pixel_points(z, K_TABLE, ok & lat)
        cloud.add(pts @ T[:3, :3].T + T[:3, 3], rgb.reshape(-1, 3)[fl].astype(float))
    return View(kid, T, 0.01, K_TABLE, z, rgb, detections(scene, idx) if detect else [])


@functools.lru_cache(maxsize=1)
def _first() -> tuple[dict[str, Any], Any]:
    import tempfile

    root = Path(tempfile.mkdtemp())
    cloud = Cloud()
    sc = TableScene()
    om = ObjectMap(root)
    views = [view(f"k{i}", e, sc, cloud) for i, e in enumerate(EYES)]
    assert om.update(views, 1, 0.0) == []
    om.finish(cloud.points(), 1.5, 1)
    om.save()
    return om.records(), (root, cloud)


def by_label(rec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {o["label"]: o for o in rec["objects"]}


def test_surfaces_heights_exactly() -> None:
    rec, _ = _first()
    hs = [s["height_m"] for s in rec["surfaces"]]
    assert len(hs) == 2
    assert hs[0] == pytest.approx(0.0, abs=0.005) and hs[1] == pytest.approx(TOP, abs=0.005)
    top = rec["surfaces"][1]
    assert top["area_m2"] == pytest.approx(1.2 * 0.8, rel=0.25)
    xs = [p[0] for p in top["polygon_m"]]
    assert min(xs) == pytest.approx(0.0, abs=0.08) and max(xs) == pytest.approx(1.2, abs=0.08)


def test_objects_and_relations() -> None:
    rec, _ = _first()
    o = by_label(rec)
    assert set(o) == {"table", "box", "basket", "apple"}
    floor, top = rec["surfaces"]
    assert o["table"]["supported_by"] == {"kind": "surface", "id": floor["id"],
                                          "verified_version": 1}
    assert o["box"]["supported_by"]["kind"] == "surface"
    assert o["box"]["supported_by"]["id"] == top["id"]
    assert o["basket"]["supported_by"]["id"] == top["id"]
    assert o["apple"]["inside_of"] == {"id": o["basket"]["id"], "verified_version": 1}
    assert o["apple"]["supported_by"] == {"kind": "object", "id": o["basket"]["id"],
                                          "verified_version": 1}
    # boxes rest on their support: the box's bottom is the table top
    box = o["box"]
    assert box["centre_m"][2] - box["size_m"][2] / 2 == pytest.approx(TOP, abs=0.01)
    assert box["size_m"][:2] == pytest.approx([0.2, 0.15], abs=0.02)
    assert box["centre_m"][:2] == pytest.approx([0.25, 0.225], abs=0.02)
    for x in o.values():
        assert x["state"] == "confirmed" and x["moved_from_m"] is None
        assert x["first_seen_version"] == x["last_seen_version"] == 1
        assert x["extent_seen"] == {"x": "full", "y": "full", "z": "full"}
        assert 0.005 <= x["sigma_centre_m"] < 0.05
        assert x["seen_from"]


def test_mean_rgb_is_the_observed_colour() -> None:
    rec, _ = _first()
    o = by_label(rec)
    assert o["box"]["mean_rgb"] == [200, 40, 40]
    assert o["basket"]["mean_rgb"] == [40, 60, 200]
    assert o["apple"]["mean_rgb"] == [60, 190, 50]


def test_struck_moved_carved_and_kept_ids(tmp_path: Path) -> None:
    rec1, (root, cloud) = _first()
    import shutil

    shutil.copytree(root, tmp_path / "m")
    items = default_items()
    items[3].boxes = []  # the apple taken away
    items[1].boxes = [((0.2, 0.5, TOP), (0.4, 0.65, TOP + 0.12))]  # the box moved
    sc = TableScene(items=items)
    om = ObjectMap.load(tmp_path / "m")
    c2 = Cloud(cloud.keys.copy(), cloud.xyz.copy(), cloud.rgb.copy(), cloud.count.copy())
    struck = om.update([view(f"n{i}", e, sc, c2) for i, e in enumerate(EYES[:3])], 2, 0.0)
    assert len(struck) == 1
    drop = om.carve(struck, c2.points())
    assert drop.sum() > 0
    apple1 = by_label(rec1)["apple"]
    assert struck[0].box.contains(c2.points()[drop], ao.CARVE_GROW_M + 1e-6).all()
    c2.drop(drop)
    om.finish(c2.points(), 1.5, 2)
    rec2 = om.records()
    o1, o2 = by_label(rec1), by_label(rec2)
    assert {k: v["id"] for k, v in o1.items()} == {k: v["id"] for k, v in o2.items()}
    assert o2["apple"]["state"] == "struck" and o2["apple"]["last_seen_version"] == 1
    assert o2["apple"]["centre_m"] == apple1["centre_m"]
    assert o2["box"]["state"] == "moved"
    assert o2["box"]["moved_from_m"] == pytest.approx(o1["box"]["centre_m"], abs=1e-3)
    assert o2["box"]["centre_m"][:2] == pytest.approx([0.3, 0.575], abs=0.02)
    assert o2["table"]["state"] == "confirmed" and o2["table"]["last_seen_version"] == 2
    assert [s["id"] for s in rec2["surfaces"]] == [s["id"] for s in rec1["surfaces"]]
    om.save()
    # the next update removes the struck object; a moved object seen again is confirmed
    om3 = ObjectMap.load(tmp_path / "m")
    om3.update([view("q0", EYES[3], sc)], 3, 0.0)
    om3.finish(c2.points(), 1.5, 3)
    o3 = by_label(om3.records())
    assert "apple" not in o3
    assert o3["box"]["state"] == "confirmed" and o3["box"]["moved_from_m"] is None


def test_relations_keep_their_verified_version(tmp_path: Path) -> None:
    rec1, (root, cloud) = _first()
    import shutil

    shutil.copytree(root, tmp_path / "m")
    om = ObjectMap.load(tmp_path / "m")
    # a keyframe looking away: nothing observed, the relations hold as verified in version 1
    om.update([view("away", (0.6, 0.4, 1.5), TableScene(), target=(-3.0, 0.4, 1.2))], 2, 0.0)
    om.finish(cloud.points(), 1.5, 2)
    o = by_label(om.records())
    assert o["apple"]["inside_of"]["verified_version"] == 1
    assert o["apple"]["supported_by"]["verified_version"] == 1
    assert o["box"]["supported_by"]["verified_version"] == 2  # surfaces are measured again
    assert o["apple"]["last_seen_version"] == 1


def test_a_candidate_seen_gone_is_dropped(tmp_path: Path) -> None:
    sc = TableScene(items=[Item("box", [((0.15, 0.15, TOP), (0.35, 0.3, TOP + 0.12))])])
    om = ObjectMap(tmp_path)
    v1 = view("a", EYES[0], sc)
    empty = TableScene(items=[])
    later = [view("b", EYES[1], empty), view("c", EYES[2], empty)]
    om.update([v1, *later], 1, 0.0)
    assert om.objects == [] and om.records()["objects"] == []


def test_a_miss_keeps_a_single_detection_unlisted(tmp_path: Path) -> None:
    sc = TableScene(items=[Item("box", [((0.15, 0.15, TOP), (0.35, 0.3, TOP + 0.12))])])
    om = ObjectMap(tmp_path)
    om.update([view("a", EYES[0], sc), view("b", EYES[1], sc, detect=False)], 1, 0.0)
    assert len(om.objects) == 1 and not om.objects[0].published
    assert om.objects[0].misses == 1 and om.records()["objects"] == []
    # a second detection lists it
    om.update([view("c", EYES[2], sc)], 2, 0.0)
    assert om.objects[0].published and om.records()["objects"][0]["first_seen_version"] == 1


def test_the_floor_named_as_an_object_is_dropped(tmp_path: Path) -> None:
    sc = TableScene(items=[])
    T = look_at(EYES[0], TARGET)
    rgb, z, idx = sc.render(T)
    floor = (idx == -1) & (z > 0)
    pts, _ = pixel_points(z, K_TABLE, np.ones_like(floor))
    zmap = (pts @ T[:3, :3].T + T[:3, 3])[:, 2].reshape(z.shape)
    m = floor & (np.abs(zmap) < 0.01)
    m[:, :400] = False  # a part of the floor (a mask over most of the view is background)
    r, c = np.nonzero(m)
    det = Detection("bed", 0.8, "test", m, (c.min(), r.min(), c.max(), r.max()))
    v = View("a", T, 0.01, K_TABLE, z, rgb, [det])
    assert ao.observe(v, 1, 0.0) == []
    assert len(ao.observe(v, 1, None)) == 1
    # the floor band is max(0.06 m, 3 floor sigmas) of the floor estimate (map.md §5): the floor
    # still, with the estimate 4.5 cm off; not floor 8 cm off on a well-known floor; floor again
    # when the floor's height is known to 3 cm only
    assert ao.observe(v, 1, 0.045) == []
    assert len(ao.observe(v, 1, 0.08)) == 1
    assert ao.observe(v, 1, 0.08, floor_sigma=0.03) == []
    # a detection naming an opening is no object (map.md §5)
    v.dets = [Detection("doorway", 0.8, "test", m, det.box)]
    assert ao.observe(v, 1, None) == []


def test_flickering_labels_and_ignore_masks(tmp_path: Path) -> None:
    sc = TableScene(items=[Item("cabinet", [((0.15, 0.15, 0.0), (0.55, 0.45, 0.6))])])
    om = ObjectMap(tmp_path)
    v1 = view("a", EYES[0], sc)
    v2 = view("b", EYES[1], sc)
    v2.dets = [Detection("wardrobe", d.score, d.source, d.mask, d.box) for d in v2.dets]
    v3 = view("c", EYES[0], sc)  # another label holds only what it overlaps both ways
    v3.dets = [Detection("plant", d.score, d.source, d.mask, d.box) for d in v3.dets]
    v3.ignore = np.zeros(v3.depth.shape, bool)
    v3.ignore[:5] = True
    om.update([v1, v2, v3], 1, 0.0)
    assert len(om.objects) == 1
    o = om.records()["objects"][0]
    assert o["label"] in ("cabinet", "wardrobe", "plant")
    assert set(o["detected_as"]) == {"cabinet", "wardrobe", "plant"}


def test_cut_masks_leave_extents_partial(tmp_path: Path) -> None:
    sc = TableScene(items=[Item("shelf", [((-1.0, 2.3, 0.8), (1.0, 2.5, 1.2))])])
    om = ObjectMap(tmp_path)
    om.update([view("a", (0.0, 1.2, 1.0), sc, target=(0.0, 3.0, 1.0))], 1, 0.0)
    assert om.records()["objects"] == []  # one sighting cut by the border: not listed yet
    o = om.objects[0]
    assert ao.extent_seen(o) == {"x": "partial", "y": "partial", "z": "partial"}
    assert ao.sigma_centre(o) >= ao.MIN_SIGMA_M


# -- pieces ---------------------------------------------------------------------------------------


def test_box_of_an_obb_and_footprint() -> None:
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    b = Box.of(OBB(np.zeros(3), R, np.array([0.2, 0.5, 0.1])))
    assert b.size.tolist() == [0.5, 0.2, 0.1] and b.yaw_deg == pytest.approx(0.0, abs=1e-9) or \
        b.yaw_deg in (90.0, 0.0)
    b2 = Box.of(OBB(np.zeros(3), np.eye(3) @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]]),
                    np.array([0.5, 0.2, 0.1])))
    assert b2.yaw_deg == 90.0
    assert len(b.footprint()) == 4
    assert Box.from_json(b.to_json()).to_json() == b.to_json()
    assert b.footprint_contains(np.array([0.1, 0.0])) and not b.footprint_contains(
        np.array([0.0, 0.3]))


def test_match_and_affinity() -> None:
    o = MapObject(5, {"cup": 1.0}, box=Box(np.array([0.0, 0.0, 0.05]), np.array([0.08, 0.08, 0.1]),
                                            0.0))
    o.points = np.random.default_rng(0).uniform(-0.04, 0.04, (50, 3)).astype(np.float32)
    o.points[:, 2] += 0.05
    near = o.points.astype(np.float64) + [0.1, 0.0, 0.0]
    far = near + [1.0, 0.0, 0.0]
    assert ao.affinity("mug", near, near.mean(0), o) > ao.MATCH_MIN
    assert ao.affinity("chair", near, near.mean(0), o) == 0.0  # partly inside, other kind
    assert ao.affinity("cup", far, far.mean(0), o) == 0.0
    assert ao.affinity("chair", o.points.astype(np.float64), np.zeros(3), o) >= ao.MATCH_ANY
    assert ao.affinity("cup", np.zeros((0, 3)), np.zeros(3), o) == 0.0
    o2 = MapObject(6, {"cup": 1.0}, box=o.box)
    assert ao.affinity("cup", near, near.mean(0), o2) > 0  # no points: the box and centres
    assert ao.match([("cup", near, near.mean(0)), ("cup", near, near.mean(0)),
                     ("cup", far, far.mean(0))], [o]) == [5, None, None]
    assert ao.match([], [o]) == []
    assert ao._extent(np.zeros((0, 3))) == 0.0


def test_robust_rgb_and_thin() -> None:
    rgb = np.zeros((20, 20, 3), np.uint8)
    rgb[...] = (10, 20, 30)
    mask = np.zeros((20, 20), bool)
    assert ao.robust_rgb(rgb, mask) is None
    mask[2:4, 2:4] = True  # too small to erode: the mask itself
    assert ao.robust_rgb(rgb, mask) == [10.0, 20.0, 30.0]
    assert ao._thin(np.zeros((0, 3))).shape == (0, 3)
    pts = np.random.default_rng(1).uniform(0, 1, (5000, 3))
    assert len(ao._thin(pts, 100)) == 100


def test_surface_ids_and_polygons() -> None:
    s = ao.Surface(0.75, [[0, 0], [1, 0], [1, 1], [0, 1]], 1.0, [0.5, 0.5], [0, 0, 1, 1])
    reg, nxt = ao.assign_surface_ids([s], [], 1)
    assert s.id == 1 and nxt == 2
    a = ao.Surface(0.76, s.polygon, 1.0, [0.5, 0.5], [0, 0, 1, 1])
    b = ao.Surface(0.76, s.polygon, 1.0, [0.52, 0.5], [0, 0, 1, 1])  # same place twice
    c = ao.Surface(1.2, s.polygon, 1.0, [0.5, 0.5], [0, 0, 1, 1])
    reg2, nxt2 = ao.assign_surface_ids([a, b, c], reg, nxt)
    assert (a.id, b.id, c.id) == (1, 2, 3) and nxt2 == 4 and len(reg2) == 3
    assert ao._iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0.0
    assert s.contains(np.array([0.5, 0.5])) and not s.contains(np.array([2.0, 2.0]))
    # degenerate outlines: one cell, a line of cells
    one = np.zeros((1, 1), bool)
    one[0, 0] = True
    assert len(ao._polygon(one, 0, 0)) >= 3
    line = np.ones((1, 4), bool)
    assert len(ao._polygon(line, 0, 0)) >= 3
    assert ao.find_surfaces(np.zeros((10, 3))) == []
    # walls only: no flat cell
    y = np.linspace(0, 2, 400)
    wall = np.stack([np.zeros_like(y), np.linspace(0, 2, 400)[::-1], y], 1)
    assert ao.find_surfaces(wall) == []


def test_find_surfaces_lines_and_tops() -> None:
    rng = np.random.default_rng(2)
    top = np.column_stack([rng.uniform(0, 1, 4000), rng.uniform(0, 1, 4000),
                           np.full(4000, 0.5)])
    strip = np.column_stack([rng.uniform(0, 3, 3000), rng.uniform(2.0, 2.04, 3000),
                             np.full(3000, 0.3)])  # a ledge one cell wide: no plane
    small = np.column_stack([rng.uniform(3, 3.1, 300), rng.uniform(3, 3.1, 300),
                             np.full(300, 0.9)])  # under the least area
    out = ao.find_surfaces(np.vstack([top, strip, small]))
    assert [round(s.height_m, 3) for s in out] == [0.5]
    assert ao.find_surfaces(np.vstack([top, strip]), top_z=0.4) == []
    sparse = top[:150]
    sparse = np.vstack([sparse, rng.uniform(5, 9, (100, 3))])
    assert ao.find_surfaces(sparse) == []


def test_judge_cases() -> None:
    sc = TableScene(items=[Item("box", [((0.15, 0.15, TOP), (0.35, 0.3, TOP + 0.12))])])
    v = view("a", EYES[0], sc)
    o = MapObject(1, {"box": 1.0})
    assert ao.judge(o, v) is None  # no box
    o.box = Box(np.array([0.25, 0.225, TOP + 0.06]), np.array([0.2, 0.15, 0.12]), 0.0)
    o.points = np.zeros((5, 3), np.float32) + o.box.centre.astype(np.float32)
    assert ao.judge(o, v) is None  # too few points
    xy = np.mgrid[0.16:0.34:0.01, 0.16:0.29:0.01].reshape(2, -1).T
    g = np.column_stack([xy, np.full(len(xy), TOP + 0.12)])  # its top, as the keyframes saw it
    o.points = g.astype(np.float32)
    assert ao.judge(o, v)[0] == "in_place"  # type: ignore[index]
    away = view("b", (0.6, 0.4, 1.5), sc, target=(-3.0, 0.4, 1.2))
    assert ao.judge(o, away) is None  # out of view
    gone = view("c", EYES[0], TableScene(items=[]))
    assert ao.judge(o, gone)[0] == "gone"  # type: ignore[index]
    hidden = TableScene(items=[Item("wall", [((0.0, -0.4, 0.0), (1.2, -0.38, 2.0))]),
                               Item("box", [((0.15, 0.15, TOP), (0.35, 0.3, TOP + 0.12))])])
    assert ao.judge(o, view("d", EYES[0], hidden)) is None  # hidden behind a nearer surface
    v.ignore = np.ones(v.depth.shape, bool)
    assert ao.judge(o, v) is None  # all ignored
    v.ignore = None
    # half seen through: no verdict
    half = view("e", EYES[0], sc)
    half.depth = half.depth.copy()
    half.depth[:, ::2] += 0.5
    assert ao.judge(o, half) is None
    far = MapObject(2, {"box": 1.0}, box=Box(np.array([0.25, 9.0, 0.8]), np.array([0.2, 0.2, 0.2]),
                                              0.0))
    far.points = (g + [0.0, 8.8, 0.0]).astype(np.float32)
    assert ao.judge(far, view("f", (0.25, 0.0, 0.85), sc, target=(0.25, 9.0, 0.8))) is None


def test_summaries_without_sightings() -> None:
    o = MapObject(3, {"x": 1.0})
    assert ao.mean_rgb(o) == [128.0, 128.0, 128.0]
    assert ao.sigma_centre(o) == 0.0
    assert ao.extent_seen(o) == {"x": "partial", "y": "partial", "z": "partial"}
    s = Sighting("k", 1, [0, 0, 0], 1.0, 100, [False, False, False, True], [0.0, 1.0, 0.0],
                 [0, -1, 0], 0.01, [1.0, 2.0, 3.0], "x", 0.5)
    o.box = Box(np.zeros(3), np.ones(3), 0.0)
    o.sightings = [s]
    o.supported_by = {"kind": "surface", "id": 1, "verified_version": 1}
    assert ao.extent_seen(o)["z"] == "partial"  # its bottom cut: the height not seen
    from_above = Sighting("k3", 1, [0, 0, 0], 1.0, 100, [False] * 4, [0.0, 0.0, -1.0],
                          [0, 0, 1], 0.01, [1.0, 2.0, 3.0], "x", 0.5)
    o.sightings = [s, from_above]
    assert ao.extent_seen(o)["z"] == "partial"  # seen from above: the top alone
    side = Sighting("k4", 1, [0, 0, 0], 1.0, 100, [False] * 4, [0.0, 1.0, 0.0],
                    [0, -1, 0], 0.01, [1.0, 2.0, 3.0], "x", 0.5)
    o.sightings = [s, side]
    assert ao.extent_seen(o)["z"] == "full"
    o.sightings = [s]
    assert Sighting.from_json(s.to_json()) == s
    back = Sighting("k2", 1, [0, 0, 0], 1.0, 100, [True, False, False, False], [0.0, -1.0, 0.0],
                    [0, 1, 0], 0.01, [1.0, 2.0, 3.0], "x", 0.5)
    o.sightings = [s, back]
    o.supported_by = None
    assert ao.extent_seen(o)["y"] == "full"  # seen from both ends
    o.points = np.zeros((0, 3), np.float32)
    o.source = np.zeros(0, np.int32)
    ObjectMap(Path("."))._fit(o, None)  # nothing to fit: the box stays
    assert o.box is not None


def test_moves_need_a_like_object(tmp_path: Path) -> None:
    om = ObjectMap(tmp_path)

    def obj(oid: int, label: str, c: list[float], size: list[float], rgb: list[float]) -> MapObject:
        o = MapObject(oid, {label: 1.0}, box=Box(np.array(c), np.array(size), 0.0), published=True)
        o.sightings = [Sighting("k", 1, c, 1.0, 100, [False] * 4, [1.0, 0, 0], [0, 0, 0], 0.01,
                                rgb, label, 0.9)]
        return o

    old = obj(1, "cup", [0, 0, 0.8], [0.08, 0.08, 0.1], [200, 0, 0])
    nobox = MapObject(9, {"cup": 1.0})
    cands = [obj(2, "cup", [0.02, 0, 0.8], [0.08, 0.08, 0.1], [200, 0, 0]),  # too near
             obj(3, "chair", [1, 0, 0.8], [0.08, 0.08, 0.1], [200, 0, 0]),  # other kind
             obj(4, "cup", [1, 0, 0.8], [0.3, 0.3, 0.3], [200, 0, 0]),  # too big
             obj(5, "cup", [1, 0, 0.8], [0.08, 0.08, 0.1], [0, 200, 0]),  # other colour
             obj(6, "cup", [1, 1, 0.8], [0.08, 0.08, 0.1], [190, 0, 0]),
             obj(7, "cup", [2, 1, 0.8], [0.08, 0.08, 0.1], [200, 0, 0]),  # the closest colour
             obj(8, "cup", [3, 1, 0.8], [0.08, 0.08, 0.1], [180, 0, 0])]
    om.objects = [old, nobox, *cands]
    om._moves([nobox, old], {2, 3, 4, 5, 6, 7, 8}, 2)
    assert old.state == "moved" and old.box.centre.tolist() == [2, 1, 0.8]  # type: ignore[union-attr]
    assert 7 not in {o.id for o in om.objects}
    lone = obj(10, "lamp", [5, 5, 1], [0.2, 0.2, 0.5], [9, 9, 9])
    om._moves([lone], set(), 2)
    assert lone.state == "confirmed"


def test_relations_prefer_object_tops_and_containers(tmp_path: Path) -> None:
    om = ObjectMap(tmp_path)

    def obj(oid: int, label: str, c: list[float], size: list[float]) -> MapObject:
        return MapObject(oid, {label: 1.0}, box=Box(np.array(c), np.array(size), 0.0),
                         published=True, last_seen_version=1)

    book = obj(1, "book", [0, 0, 0.765], [0.3, 0.2, 0.03])
    cup = obj(2, "cup", [0, 0, 0.83], [0.08, 0.08, 0.1])  # on the book, above the table top
    tray = obj(3, "box", [2, 0, 0.8], [0.4, 0.4, 0.1])
    big = obj(4, "box", [2, 0, 0.8], [0.6, 0.6, 0.2])
    pen = obj(5, "pen", [2, 0, 0.77], [0.1, 0.02, 0.02])
    om.surfaces = [ao.Surface(0.75, [[-1, -1], [3, -1], [3, 1], [-1, 1]], 8.0, [1, 0],
                              [-1, -1, 3, 1], 7)]
    om.objects = [book, cup, tray, big, pen]
    om.finish(np.zeros((0, 3)), None, 1)  # no cloud: no surfaces measured
    assert om.surfaces == []
    om.surfaces = [ao.Surface(0.75, [[-1, -1], [3, -1], [3, 1], [-1, 1]], 8.0, [1, 0],
                              [-1, -1, 3, 1], 7)]
    for o in om.live():
        om._relate(o, om.live(), 1)
    assert cup.supported_by == {"kind": "object", "id": 1, "verified_version": 1}
    assert book.supported_by == {"kind": "surface", "id": 7, "verified_version": 1}
    assert pen.inside_of == {"id": 3, "verified_version": 1}  # the smaller container
    assert tray.inside_of == {"id": 4, "verified_version": 1}
    # seen again later without change: the relation keeps its version; changed: the new one
    for o in om.objects:
        o.last_seen_version = 1
    for o in om.live():
        om._relate(o, om.live(), 2)
    assert cup.supported_by["verified_version"] == 1  # type: ignore[index]
    assert pen.inside_of["verified_version"] == 1  # type: ignore[index]
    tray.box = Box(np.array([5.0, 0, 0.8]), np.array([0.4, 0.4, 0.1]), 0.0)
    for o in om.live():
        om._relate(o, om.live(), 3)
    assert pen.inside_of == {"id": 4, "verified_version": 3}
    om.surfaces = []
    book.box = Box(np.array([0.0, 0, 1.765]), np.array([0.3, 0.2, 0.03]), 0.0)
    om._relate(cup, om.live(), 4)
    assert cup.supported_by is None


def test_load_without_state_and_save_roundtrip(tmp_path: Path) -> None:
    om = ObjectMap.load(tmp_path / "none")
    assert om.objects == [] and om.next_id == 1
    rec, (root, _) = _first()
    om = ObjectMap.load(root)
    assert om.records() == rec
    (root / ao.OBJECTS_NPZ).rename(root / "x.npz")
    try:
        bare = ObjectMap.load(root)
        assert all(len(o.points) == 0 for o in bare.objects)
    finally:
        (root / "x.npz").rename(root / ao.OBJECTS_NPZ)


def test_containers_interior(tmp_path: Path) -> None:
    sc = TableScene(items=[default_items()[0], Item("bowl", container(0.4, 0.2, TOP, 0.4, 0.15)),
                           Item("egg", [((0.57, 0.37, TOP + 0.01), (0.63, 0.43, TOP + 0.06))])])
    om = ObjectMap(tmp_path)
    cloud = Cloud()
    om.update([view(f"k{i}", e, sc, cloud) for i, e in enumerate(EYES[:3])], 1, 0.0)
    om.finish(cloud.points(), 1.5, 1)
    o = by_label(om.records())
    assert o["egg"]["inside_of"]["id"] == o["bowl"]["id"]
    assert o["bowl"]["supported_by"]["kind"] == "surface"


def test_observations_trimmed_to_nothing_are_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    sc = TableScene(items=[Item("box", [((0.15, 0.15, TOP), (0.35, 0.3, TOP + 0.12))])])
    v = view("a", EYES[0], sc)
    assert len(ao.observe(v, 1, None)) == 1

    def bare(inst: Any, *a: Any) -> Any:
        from oh_my_slam.segmentation.api import LiftedInstance

        lifted = type(inst.lifted)(inst.lifted.points[:3], inst.lifted.pixels[:3],
                                   inst.lifted.through)
        return LiftedInstance(inst.detection, inst.mask, lifted)

    monkeypatch.setattr(ao, "trim_support", bare)
    assert ao.observe(v, 1, None) == []


def test_carving_skips_boxes_without_points() -> None:
    sc = TableScene(items=[])
    v = view("a", EYES[0], sc)
    s = ao.Struck(Box(np.array([9.0, 9.0, 9.0]), np.array([0.1, 0.1, 0.1]), 0.0), [v])
    assert not ObjectMap(Path(".")).carve([s], np.zeros((5, 3))).any()


def test_spread_counts_while_an_axis_is_partial() -> None:
    o = MapObject(1, {"shelf": 1.0}, box=Box(np.zeros(3), np.array([2.0, 0.3, 1.0]), 0.0))
    cut = [True, True, False, False]
    o.sightings = [Sighting("a", 1, [0.0, 0, 0], 2.0, 100, cut, [0.0, 1.0, 0.0], [0, -2, 0],
                            0.01, [1.0, 1, 1], "shelf", 0.9),
                   Sighting("b", 1, [0.4, 0, 0], 2.0, 100, cut, [0.0, 1.0, 0.0], [0, -2, 0],
                            0.01, [1.0, 1, 1], "shelf", 0.9)]
    assert ao.sigma_centre(o) > 0.1


def test_find_surfaces_without_flat_cells() -> None:
    g = np.mgrid[0:20, 0:20].reshape(2, -1).T * 0.3  # one point per cell
    pts = np.column_stack([g, np.zeros(len(g))])
    assert ao.find_surfaces(pts) == []


def test_a_plateau_of_two_bins_is_one_surface() -> None:
    xy = np.mgrid[0:50, 0:50].reshape(2, -1).T * 0.02 + 0.01
    xy = np.vstack([xy, xy])
    z = np.repeat([0.4999, 0.5001], 2500)  # either side of a bin edge, in every cell
    out = ao.find_surfaces(np.column_stack([xy, z]))
    assert len(out) == 1 and out[0].height_m == pytest.approx(0.5, abs=0.001)


def test_a_surface_elsewhere_at_the_same_height_takes_a_new_id() -> None:
    s = ao.Surface(0.75, [[0, 0], [1, 0], [1, 1], [0, 1]], 1.0, [0.5, 0.5], [0, 0, 1, 1])
    known = [{"id": 4, "h": 0.75, "c": [5.0, 5.0], "bbox": [4.5, 4.5, 5.5, 5.5]}]
    reg, nxt = ao.assign_surface_ids([s], known, 5)
    assert s.id == 5 and nxt == 6 and [e["id"] for e in reg] == [4, 5]


def test_yaw_stays_in_its_range() -> None:
    assert ao.yaw_deg(-90.0) == 90.0 and ao.yaw_deg(-89.99996) == 90.0
    assert ao.yaw_deg(270.0) == 90.0 and ao.yaw_deg(10.0) == 10.0


def test_seen_twice_then_gone_in_one_update_is_struck(tmp_path: Path) -> None:
    sc = TableScene(items=[Item("box", [((0.15, 0.15, TOP), (0.35, 0.3, TOP + 0.12))])])
    empty = TableScene(items=[])
    om = ObjectMap(tmp_path)
    struck = om.update([view("a", EYES[0], sc), view("b", EYES[1], sc),
                        view("c", EYES[2], empty), view("d", EYES[3], empty)], 1, 0.0)
    assert len(struck) == 1
    assert [(o["label"], o["state"]) for o in om.records()["objects"]] == [("box", "struck")]


def test_a_contradicted_object_is_judged_by_its_detection_only(tmp_path: Path) -> None:
    sc = TableScene(items=[Item("box", [((0.15, 0.15, TOP), (0.35, 0.3, TOP + 0.12))])])
    empty = TableScene(items=[])
    om = ObjectMap(tmp_path)
    # seen, its place seen through (the depth looks past it), seen again: unjudgeable by depth
    om.update([view("a", EYES[0], sc), view("b", EYES[1], sc), view("c", EYES[2], empty),
               view("d", EYES[3], sc)], 1, 0.0)
    o = om.objects[0]
    assert o.contradicted == 1 and ao.see_through(o) and o.state == "confirmed"
    # one keyframe that does not detect it never strikes it
    assert om.update([view("e", EYES[0], empty)], 2, 0.0) == []
    assert om.objects[0].state == "confirmed"
    assert ao.judge(o, view("f", EYES[1], empty)) == ("gone", ao.ABSENT_SHARE)
    # hidden behind a nearer surface, or too small to be detected: no verdict
    wall = TableScene(items=[Item("wall", [((0.0, -0.4, 0.0), (1.2, -0.38, 2.0))])])
    assert ao.judge(o, view("g", EYES[0], wall)) is None
    tiny = MapObject(9, {"box": 1.0}, box=o.box, sightings=[
        Sighting("k", 1, [0, 0, 0], 0.1, 1, [False] * 4, [1.0, 0, 0], [0, 0, 0], 0.01,
                 [1.0, 1, 1], "box", 0.9)], points=o.points, contradicted=1)
    assert ao.judge(tiny, view("h", EYES[1], empty)) is None
    # two keyframes that do not detect it strike it
    assert len(om.update([view("i", EYES[1], empty), view("j", EYES[2], empty)], 3, 0.0)) == 1


def test_a_place_detected_as_something_else_is_not_judged() -> None:
    sc = TableScene(items=[Item("box", [((0.15, 0.15, TOP), (0.35, 0.3, TOP + 0.12))])])
    v = view("a", EYES[0], sc)
    obs = ao.observe(v, 1, None)
    o = MapObject(1, {"box": 1.0}, sightings=[obs[0].sighting],
                  points=obs[0].points.astype(np.float32))
    o.box = Box.of(__import__("oh_my_slam.segmentation.api", fromlist=["x"]).fit_object_obb(
        obs[0].points, "box", ao.UP, None))
    gone = view("b", EYES[0], TableScene(items=[]))
    assert ao.judge(o, gone)[0] == "gone"  # type: ignore[index]
    gone.dets = [Detection("globe", 0.8, "test", v.dets[0].mask, v.dets[0].box)]
    assert ao.judge(o, gone) is None
    big = np.ones_like(v.dets[0].mask)
    gone.dets = [Detection("wall art", 0.8, "test", big, (0.0, 0.0, 640.0, 480.0))]
    assert ao.judge(o, gone)[0] == "gone"  # type: ignore[index]
    assert not ao._detected_there(gone, np.zeros(0), np.zeros(0), 10.0)


def test_see_through_evidence_inside_the_masks() -> None:
    def sighting(behind: float, mask_range: float) -> Sighting:
        return Sighting("k", 1, [0, 0, 0], 1.0, 100, [False] * 4, [1.0, 0, 0], [-1, 0, 0],
                        0.01, [1.0, 1, 1], "globe", 0.9, behind=behind, mask_range=mask_range)

    o = MapObject(1, {"globe": 1.0}, box=Box(np.zeros(3), np.full(3, 0.2), 0.0))
    o.sightings = [sighting(0.5, 1.0), sighting(0.0, 1.0)]
    assert not ao.see_through(o)  # one sighting alone
    o.sightings = [sighting(0.5, 1.0), sighting(0.0, 2.0), sighting(0.0, 1.0)]
    assert ao.see_through(o)  # its mask's depth beyond its points, and beyond its box
    assert not ao.see_through(MapObject(2, {"globe": 1.0}))
    sc = TableScene(items=[Item("box", [((0.15, 0.15, TOP), (0.35, 0.3, TOP + 0.12))])])
    v = view("a", EYES[0], sc)
    m = np.zeros(v.depth.shape, bool)
    m[200:203, 300:303] = True  # too thin to erode: the mask itself
    pts = np.array([[0.25, 0.225, TOP + 0.12]])
    share, rng = ao._behind(m, v.depth, v.depth > 0, v, pts)
    assert 0.0 <= share <= 1.0 and rng > 0

@pytest.mark.parametrize(("rel", "struck"), [(ao.SENSOR_DEPTH_REL, True),
                                             (ao.MONO_DEPTH_REL, False)])
def test_one_keyframe_strikes_only_with_a_depth_image(tmp_path: Path, rel: float,
                                                      struck: bool) -> None:
    sc = TableScene(items=[Item("box", [((0.15, 0.15, TOP), (0.35, 0.3, TOP + 0.12))])])
    om = ObjectMap(tmp_path)
    om.update([view("a", EYES[0], sc), view("b", EYES[1], sc)], 1, 0.0)
    gone = view("c", EYES[2], TableScene(items=[]))
    gone.depth_rel = rel
    assert bool(om.update([gone], 2, 0.0)) is struck


def test_specific_labels_win_over_generic_ones() -> None:
    o = MapObject(1, {"ball": 2.0, "apple": 0.6, "fruit": 0.5})
    assert o.label == "apple"
    o = MapObject(1, {"ball": 4.0, "apple": 0.6})
    assert o.label == "ball"  # too little evidence for the specific label
    assert MapObject(1, {"ball": 1.0}).label == "ball"
    assert MapObject(1, {"cup": 1.0, "apple": 0.5}).label == "cup"
    assert ao.alike("ball", "tomato") and ao.alike("egg", "fruit") and not ao.alike("egg", "cup")

ABOVE = [(0.42, 0.05, 1.5), (0.40, 0.15, 1.55), (0.45, 0.1, 1.45)]
OBLIQUE = [(0.0, 0.0, 1.45), (0.1, -0.3, 1.5), (0.0, 0.3, 1.4), (0.2, 0.0, 1.55)]


@pytest.mark.parametrize("eyes", [OBLIQUE, ABOVE], ids=["oblique", "above"])
def test_worktop_support_heights_with_noisy_depth(tmp_path: Path,
                                                  eyes: list[tuple[float, float, float]]
                                                  ) -> None:
    """An arm's worktop at 0.9 m at 640 x 360 with 1 % + 3 mm depth noise, beside a lower
    counter 9 cm down, a rug detected on the worktop: the worktop's height within 1 cm; the
    items 3-8 cm tall rest on it with their full heights, also when seen only from above (their
    height then ``partial``); a tomato on a plate; the rug neither listed nor a support."""
    from oh_my_slam.core.types import Intrinsics
    from tests.synth.room import Room

    K = Intrinsics(320.0, 320.0, 320.0, 180.0, 640, 360)
    w = 0.9
    items = [Item("counter", [((-0.2, -0.4, 0.0), (1.0, 0.6, w))]),
             Item("cabinet", [((1.0, -0.4, 0.0), (1.6, 0.6, w - 0.09))]),
             Item("rug", [((0.50, -0.20, w), (0.68, -0.02, w + 0.002))], (90, 60, 40)),
             Item("apple", [((0.30, 0.00, w), (0.37, 0.07, w + 0.07))], (200, 30, 30)),
             Item("egg", [((0.45, 0.10, w), (0.49, 0.15, w + 0.045))], (230, 220, 200)),
             Item("plate", [((0.20, 0.20, w), (0.42, 0.42, w + 0.02))], (240, 240, 240)),
             Item("mug", [((0.55, -0.15, w + 0.002), (0.63, -0.07, w + 0.08))],
                  (40, 90, 200)),
             Item("tomato", [((0.28, 0.27, w + 0.02), (0.34, 0.33, w + 0.07))], (220, 50, 20))]
    sc = TableScene(items=items, room=Room(boxes=[]))
    rng = np.random.default_rng(0)
    cloud = Cloud()
    views = []
    for i, eye in enumerate(eyes):
        T = look_at(eye, (0.42, 0.1, w)) if eye[0] != 0.42 else look_at(eye, (0.42, 0.06, w))
        rgb, z, idx = sc.render(T, K)
        z = (z * (1 + 0.01 * rng.standard_normal(z.shape))
             + 0.003 * rng.standard_normal(z.shape)).astype(np.float32)
        ok = edge_free(z)
        lat = np.zeros_like(ok)
        lat[::2, ::2] = True
        pts, fl = pixel_points(z, K, ok & lat)
        cloud.add(pts @ T[:3, :3].T + T[:3, 3], rgb.reshape(-1, 3)[fl].astype(float))
        views.append(View(f"k{i}", T, 0.005, K, z, rgb, detections(sc, idx), None,
                          ao.SENSOR_DEPTH_REL))
    om = ObjectMap(tmp_path)
    om.update(views, 1, w)
    om.finish(cloud.points(), 1.6, 1, np.array([v.centre for v in views]))
    rec = om.records()
    top = next(s for s in rec["surfaces"] if abs(s["height_m"] - w) < 0.05)
    assert top["height_m"] == pytest.approx(w, abs=0.01)
    o = by_label(rec)
    assert "rug" not in o
    heights = {"apple": 0.07, "egg": 0.045, "plate": 0.02, "mug": 0.08}
    for name, h in heights.items():
        assert o[name]["supported_by"]["id"] == top["id"], name
        bottom = o[name]["centre_m"][2] - o[name]["size_m"][2] / 2
        assert bottom == pytest.approx(w, abs=0.01), name
        assert o[name]["size_m"][2] == pytest.approx(h, abs=0.02), name
    assert o["tomato"]["supported_by"] == {"kind": "object", "id": o["plate"]["id"],
                                           "verified_version": 1}
    if eyes is ABOVE:
        assert o["egg"]["extent_seen"]["z"] == "partial"


def _edge_view(sc: TableScene, kid: str) -> View:
    """A keyframe that has the apple cut by its left border."""
    eye = (0.25, -0.6, 1.3)
    for dx in np.linspace(0.7, 1.2, 51):
        v = view(kid, eye, sc, target=(0.25 + dx, 0.3, TOP))
        d = next((d for d in v.dets if d.label == "apple"), None)
        if d is not None and d.mask[:, 0].any() and d.mask.sum() > 150:
            return v
    raise AssertionError("no edge view")


def test_a_fragment_at_the_border_is_the_object(tmp_path: Path) -> None:
    sc = TableScene(items=[default_items()[0],
                           Item("apple", [((0.21, 0.26, TOP), (0.29, 0.34, TOP + 0.08))],
                                (200, 30, 30))])
    whole = view("a", EYES[0], sc)
    edge = _edge_view(sc, "b")
    edge.dets = [Detection("orange" if d.label == "apple" else d.label, 0.95, d.source, d.mask,
                           d.box) for d in edge.dets]
    om = ObjectMap(tmp_path)
    om.update([whole, edge], 1, 0.0)
    labels = [o["label"] for o in om.records()["objects"]]
    assert labels.count("apple") == 1 and "orange" not in labels
    apple = next(o for o in om.objects if o.label == "apple")
    assert len(apple.sightings) == 2 and apple.sightings[1].truncated
    # a fragment alone is not listed until another view confirms it
    om2 = ObjectMap(tmp_path / "two")
    om2.update([_edge_view(sc, "c")], 1, 0.0)
    assert not [o for o in om2.records()["objects"] if o["label"] == "apple"]


def test_produce_goes_together() -> None:
    assert ao.alike("apple", "orange") and ao.alike("tomato", "peach")
    assert not ao.alike("apple", "cup")


def test_tiny_fragments_are_not_listed() -> None:
    o = MapObject(1, {"wire": 1.0}, published=True,
                  box=Box(np.zeros(3), np.array([0.02, 0.01, 0.01]), 0.0))
    assert not ao.listed(o)
    o.box = Box(np.zeros(3), np.array([0.04, 0.01, 0.01]), 0.0)
    assert ao.listed(o)


def test_a_door_is_listed_only_as_a_leaf() -> None:
    """A door object is a leaf (map.md §5): thinner than 0.20 m, vertical, at least 0.6 m tall."""
    o = MapObject(1, {"door": 1.0}, published=True,
                  box=Box(np.zeros(3), np.array([0.9, 0.17, 2.0]), 0.0))
    assert o.label == "door" and ao.listed(o)  # an open leaf seen at an angle
    o.box = Box(np.zeros(3), np.array([0.9, 0.4, 2.0]), 0.0)  # a block: no leaf
    assert not ao.listed(o)
    o.box = Box(np.zeros(3), np.array([0.9, 0.05, 0.4]), 0.0)  # too low
    assert not ao.listed(o)


def test_background_and_robot_masks_are_dropped() -> None:
    sc = TableScene(items=[Item("box", [((0.15, 0.15, TOP), (0.35, 0.3, TOP + 0.12))])])
    v = view("a", EYES[0], sc)
    whole = np.ones(v.depth.shape, bool)
    v.dets = [Detection("tree", 0.4, "test", whole, (0.0, 0.0, 640.0, 480.0))]
    assert ao.observe(v, 1, None) == []
    near = view("b", (0.25, 0.225, TOP + 0.6), sc, target=(0.25, 0.225, TOP))
    near.depth = np.full_like(near.depth, 0.08)
    near.dets = [Detection("box", 0.9, "test", near.dets[0].mask, near.dets[0].box)]
    assert ao.observe(near, 1, None) == []


def test_a_low_score_detection_needs_a_second_view(tmp_path: Path) -> None:
    sc = TableScene(items=[Item("plate", [((0.15, 0.15, TOP), (0.35, 0.3, TOP + 0.02))])])
    om = ObjectMap(tmp_path)
    v = view("a", EYES[0], sc)
    v.dets = [Detection(d.label, 0.3, d.source, d.mask, d.box) for d in v.dets]
    om.update([v], 1, 0.0)
    assert om.records()["objects"] == []
    w = view("b", EYES[1], sc)
    w.dets = [Detection(d.label, 0.25, d.source, d.mask, d.box) for d in w.dets]
    om.update([w], 2, 0.0)
    assert [o["label"] for o in om.records()["objects"]] == ["plate"]


def test_detection_votes_over_the_frames_that_see_the_centre() -> None:
    """map.md §5: per label, the frames that gave it over the confirmed frames whose frustum
    holds the box centre (detected or not), highest share first."""
    o = MapObject(3, {"cup": 1.0, "mug": 0.5})

    def sight(kid: str, label: str) -> Sighting:
        return Sighting(kid, 1, [0, 0, 0], 1.0, 100, [False] * 4, [1.0, 0.0, 0.0], [-1, 0, 0],
                        0.01, [1.0, 2.0, 3.0], label, 0.5)

    o.sightings = [sight("a", "cup"), sight("b", "cup"), sight("b", "cup"), sight("c", "mug")]
    assert ao.detection_votes(o, []) == [{"label": "cup", "share": 0.6667},
                                         {"label": "mug", "share": 0.3333}]
    o.box = Box(np.zeros(3), np.ones(3) * 0.1, 0.0)
    looking = look_at((-1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    away = look_at((-1.0, 0.0, 0.0), (-2.0, 0.0, 0.0))
    side = look_at((-1.0, 0.0, 0.0), (-1.0, 3.0, 0.0))  # the centre left of its image
    cams = [("d", looking, K_TABLE), ("e", away, K_TABLE), ("a", looking, K_TABLE),
            ("f", side, K_TABLE)]
    assert ao.in_view(np.zeros(3), looking, K_TABLE) and not ao.in_view(np.zeros(3), away,
                                                                         K_TABLE)
    assert not ao.in_view(np.zeros(3), side, K_TABLE)
    votes = ao.detection_votes(o, cams)  # seen by a, b, c, d: 4 frames
    assert votes == [{"label": "cup", "share": 0.5}, {"label": "mug", "share": 0.25}]
    assert ao.detection_votes(MapObject(4, {}), cams) == []


def test_records_carry_detection_votes() -> None:
    rec, _ = _first()
    for o in rec["objects"]:
        votes = o["detection_votes"]
        assert votes and sum(v["share"] for v in votes) <= 1.0 + 1e-9
        assert votes == sorted(votes, key=lambda v: (-v["share"], v["label"]))
