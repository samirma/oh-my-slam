"""Object quality: confirmation from evidence, label-flicker de-duplication, boxes fitted to the
sightings that agree, floor grounding only where the evidence supports it, and point counts from
the map cloud."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.mapping import objects as mo
from oh_my_slam.mapping.objects import MapObject, ObjectState, Sighting
from oh_my_slam.schema import openlabel as ol
from oh_my_slam.segmentation.api import SceneObject, fit_object_obb
from oh_my_slam.segmentation.obb import OBB
from oh_my_slam.segmentation.scene import object_entry
from oh_my_slam.tools.evaluate.scene import doc_objects
from oh_my_slam.tools.evaluate.segmentation import paired_labels

UP = np.array([0.0, 0.0, 1.0])
K = Intrinsics(260.0, 260.0, 160.0, 120.0, 320, 240)


def slab(rng: np.random.Generator, size: tuple[float, float, float],
         center: tuple[float, float, float], n: int = 3000) -> np.ndarray:
    """Points filling a box (a visible surface patch when one side is thin)."""
    return rng.uniform(-0.5, 0.5, (n, 3)) * np.asarray(size) + np.asarray(center)


def sighting(frame: int, pts: np.ndarray, border: float = 0.0) -> Sighting:
    lo, hi = np.percentile(pts, [2, 98], axis=0)
    c = pts.mean(0)
    return Sighting(frame, len(pts), border, (float(c[0]), float(c[1]), float(c[2])),
                    (float(lo[0]), float(lo[1]), float(lo[2])),
                    (float(hi[0]), float(hi[1]), float(hi[2])))


def map_object(oid: int, label: str, pts: np.ndarray, frames: list[int], score: float = 0.7,
               sightings: list[Sighting] | None = None) -> MapObject:
    o = MapObject(oid, label, {label: score * len(frames)}, [score] * len(frames),
                  mo.canonical_points(pts), frames=sorted(frames), obs_depth=2.0)
    o.sightings = sorted(sightings or [], key=Sighting.key)
    mo.refit(o, None)
    return o


def bottom(b: OBB) -> float:
    return float(b.center[2] - b.size[2] / 2)


# --- floor grounding (segmentation) --------------------------------------------------------------


def test_grounding_only_where_the_evidence_supports_it() -> None:
    rng = np.random.default_rng(0)
    floor = 0.0
    # a chair's backrest showing 0.3 m above a table: its lower part is hidden -> grounded
    b = fit_object_obb(slab(rng, (0.45, 0.05, 0.3), (0, 0, 0.65)), "chair", UP, floor)
    assert bottom(b) == pytest.approx(floor, abs=1e-6) and b.size[2] == pytest.approx(0.8, abs=0.03)
    # a table seen from above shows only its top: grounded whatever the top's thickness
    b = fit_object_obb(slab(rng, (1.0, 0.6, 0.04), (0, 0, 0.73)), "dining table", UP, floor)
    assert bottom(b) == pytest.approx(floor, abs=1e-6)
    # a 5 cm print lying on a 0.9 m counter, detected as a person: never extended to the floor
    b = fit_object_obb(slab(rng, (0.3, 0.2, 0.05), (0, 0, 0.93)), "person", UP, floor)
    assert b.size[2] < 0.1 and bottom(b) > 0.85
    # ... nor a tiny fragment of a floor-standing class
    b = fit_object_obb(slab(rng, (0.08, 0.05, 0.06), (0, 0, 0.5)), "chair", UP, floor)
    assert b.size[2] < 0.1
    # a plant on a table is not closed down to the floor; one trimmed at the floor contact is
    b = fit_object_obb(slab(rng, (0.3, 0.3, 0.4), (0, 0, 0.95)), "potted plant", UP, floor)
    assert bottom(b) > 0.7
    b = fit_object_obb(slab(rng, (0.3, 0.3, 0.4), (0, 0, 0.3)), "potted plant", UP, floor)
    assert bottom(b) == pytest.approx(floor, abs=1e-6)
    # a door whose lower 0.4 m the keyframes that detected it never saw (behind a kitchen island)
    b = fit_object_obb(slab(rng, (0.85, 0.05, 1.5), (0, 0, 1.15)), "door", UP, floor)
    assert bottom(b) == pytest.approx(floor, abs=1e-6) and b.size[2] == pytest.approx(1.9, abs=0.05)
    # classes that are not floor-standing keep their visible box; no floor, no grounding
    b = fit_object_obb(slab(rng, (0.07, 0.07, 0.25), (0, 0, 0.3)), "bottle", UP, floor)
    assert bottom(b) > 0.15
    b = fit_object_obb(slab(rng, (0.45, 0.05, 0.3), (0, 0, 0.65)), "chair", UP, None)
    assert b.size[2] == pytest.approx(0.3, abs=0.03)


# --- confirmation --------------------------------------------------------------------------------


def test_confirmation_needs_two_reliable_keyframes_when_others_had_it_in_view() -> None:
    rng = np.random.default_rng(1)
    pts = slab(rng, (0.2, 0.2, 0.2), (0, 0, 2.0))
    one = map_object(1, "cup", pts, [3], sightings=[sighting(3, pts)])
    one.views_in_frustum = 1  # the only keyframe that could see it (a single image)
    mo.confirm(one)
    assert one.confirmed
    one.views_in_frustum = 6  # five other keyframes had it in view and did not detect it
    mo.confirm(one)
    assert not one.confirmed
    two = map_object(2, "cup", pts, [3, 7], sightings=[sighting(3, pts), sighting(7, pts)])
    two.views_in_frustum = 9
    mo.confirm(two)
    assert two.confirmed
    # a detection mostly in the image-border band (depth unreliable, object cut off) is weak
    border = map_object(3, "cup", pts, [3, 7],
                        sightings=[sighting(3, pts), sighting(7, pts, border=0.8)])
    border.views_in_frustum = 9
    mo.confirm(border)
    assert not border.confirmed
    # objects of maps written before sightings were recorded: every detection counts
    legacy = map_object(4, "cup", pts, [3, 7])
    legacy.views_in_frustum = 9
    mo.confirm(legacy)
    assert legacy.confirmed


def test_views_count_keyframes_with_the_object_in_view_ignoring_occlusion() -> None:
    rng = np.random.default_rng(2)
    pts = slab(rng, (0.3, 0.3, 0.3), (0, 0, 2.0))  # 2 m in front of a camera at the origin
    o = map_object(1, "box", pts, [0])
    ahead = Pose(np.eye(3), np.zeros(3))
    behind = Pose(np.diag([-1.0, 1.0, -1.0]), np.zeros(3))  # looking the other way
    recs = [SimpleNamespace(index=i, K_grid=K, T_map_cam=T)
            for i, T in enumerate([ahead, ahead, behind, ahead])]
    assert mo.count_views(o, recs) == 3  # the detecting keyframe and two others facing it
    o.frames = [2]  # a keyframe that detected it counts even if its points project elsewhere
    assert mo.count_views(o, recs) == 4
    mask = np.zeros((100, 100), bool)
    mask[:20, 40:60] = True  # 8 of the 20 rows lie in the 8-px border band
    assert mo.border_share(mask) == pytest.approx(0.4)
    assert mo.border_share(np.zeros((10, 10), bool)) == 0.0


# --- label-flicker de-duplication ----------------------------------------------------------------


def test_objects_of_different_labels_in_the_same_space_are_merged() -> None:
    rng = np.random.default_rng(3)
    door = slab(rng, (0.85, 0.04, 2.0), (2.0, 0.0, 1.0), n=6000)
    a = map_object(4, "door", door, [0, 1, 2, 3, 6], score=0.85)
    b = map_object(9, "wardrobe", door[::2] + rng.normal(0, 0.01, door[::2].shape), [5, 8],
                   score=0.6)
    st = ObjectState([a, b], 20)
    alias: dict[int, int] = {}
    assert mo._merge(st, {9}, alias) == 1 and alias == {9: 4}
    (kept,) = st.objects
    assert kept.id == 4 and kept.label == "door"  # lower id; the label with the most evidence
    assert kept.frames == [0, 1, 2, 3, 5, 6, 8] and kept.label_votes["wardrobe"] > 0
    assert kept.scene_object().labels == ("door", "wardrobe")
    # a keyframe that detected both saw two things there: never merged
    a = map_object(4, "door", door, [0, 1, 2, 3, 6], score=0.85)
    b = map_object(9, "wardrobe", door[::2], [5, 6], score=0.6)
    st = ObjectState([a, b], 20)
    assert mo._merge(st, {9}, {}) == 0 and len(st.objects) == 2
    # an item resting on a larger object (a book on a desk top) is not the desk
    desk = slab(rng, (1.4, 0.7, 0.03), (0.0, 0.0, 0.75), n=8000)
    book = slab(rng, (0.3, 0.22, 0.03), (0.1, 0.0, 0.78))
    st = ObjectState([map_object(2, "desk", desk, [0, 1, 2]), map_object(5, "book", book, [4])],
                     20)
    assert mo._merge(st, {5}, {}) == 0 and len(st.objects) == 2


# --- boxes from agreeing sightings ---------------------------------------------------------------


def test_box_fitted_to_the_sightings_that_agree() -> None:
    rng = np.random.default_rng(4)
    # a 10 cm kettle whose monocular depth put two of its six sightings 0.5-1 m further away
    parts = [slab(rng, (0.1, 0.1, 0.12), (2.0 + dx, 0.0, 0.9), n=300)
             for dx in (0.0, 0.02, -0.02, 0.01, 0.5, 1.0)]
    k = map_object(1, "kettle", np.concatenate(parts), list(range(6)),
                   sightings=[sighting(i, p) for i, p in enumerate(parts)])
    assert k.obb is not None and k.obb.size[0] < 0.2  # not the 1.1 m streak of all points
    assert len(mo.agreeing_sightings(k)) == 4
    # a counter seen piecewise: overlapping partial views all agree, the box spans them
    parts = [slab(rng, (1.0, 0.6, 0.05), (x, 0.0, 0.9), n=2000) for x in (0.0, 0.6, 1.2)]
    c = map_object(2, "counter", np.concatenate(parts), [0, 1, 2],
                   sightings=[sighting(i, p) for i, p in enumerate(parts)])
    assert len(mo.agreeing_sightings(c)) == 3
    assert c.obb is not None and c.obb.size[0] == pytest.approx(2.1, abs=0.1)
    # sightings persist with the object
    back = MapObject.from_dict(k.to_dict(), k.points)
    assert back.sightings == k.sightings


def sourced(oid: int, label: str, parts: list[tuple[np.ndarray, float, bool]]) -> MapObject:
    """An object from detections (points, border share, keyframe placed confidently), their
    points recorded with their sources, as ``MapObject.add`` does."""
    o = MapObject(oid, label, {label: 0.8}, [0.8], np.zeros((0, 3), np.float32),
                  frames=list(range(len(parts))), obs_depth=0.7)
    for p, border, confident in parts:
        o.add_points(p, mo.source_of(border <= mo.BORDER_EVIDENCE, confident))
    o.sightings = sorted((replace(sighting(f, p, border), confident=conf)
                          for f, (p, border, conf) in enumerate(parts)), key=Sighting.key)
    mo.refit(o, None)
    return o


def test_box_shaped_by_the_sightings_placed_confidently() -> None:
    """A wallet on a sill seen by four confidently placed keyframes and by three of a later update
    whose poses are uncertain (low confidence) and place it up to 11 cm off, onto the item beside
    it: the box is the wallet's, from the confident sightings; their points alone shape it (the
    others' points that fall in their bounds too). The uncertain ones still shape the box of an
    object that nothing else saw, and a detection cut by the image border shapes it only while
    the reliable ones are not the majority."""
    rng = np.random.default_rng(8)

    def wallet(dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
        return slab(rng, (0.11, 0.08, 0.02), (0.35 + dx, 0.45 + dy, -0.37), n=800)

    good = [(wallet(), 0.0, True) for _ in range(4)]
    off = [(wallet(-0.08, 0.08), 0.0, False), (wallet(0.0, -0.05), 0.2, False),
           (wallet(-0.06, 0.03), 0.0, False)]
    w = sourced(8, "wallet", good + off)
    assert w.obb is not None and w.obb.size[0] == pytest.approx(0.11, abs=0.015)
    assert w.obb.size[1] == pytest.approx(0.08, abs=0.015)
    assert mo.shaping_sightings(w) == [s for s in w.sightings if s.confident]
    # all confident: the offset copies agree within the depth noise and widen the box
    wide = sourced(8, "wallet", good + [(p, b, True) for p, b, _ in off])
    assert wide.obb is not None and wide.obb.size[0] > 0.15
    # seen only by uncertain keyframes: they are all there is
    alone = sourced(9, "wallet", off[:1] + [(wallet(-0.08, 0.08), 0.0, False)])
    assert alone.obb is not None and alone.obb.size[0] == pytest.approx(0.11, abs=0.02)
    # a globe cut by the image border in one of five keyframes, placed 8 cm aside: left out
    ball = [(slab(rng, (0.09, 0.08, 0.16), (0.07, 0.62, -0.3), n=800), 0.0, True)
            for _ in range(4)]
    cut = (slab(rng, (0.06, 0.05, 0.16), (0.15, 0.55, -0.3), n=500), 1.0, True)
    g = sourced(14, "globe", [*ball, cut])
    assert g.obb is not None and g.obb.size[0] < 0.12
    # a counter seen piecewise, mostly by views cut by the border: every piece counts
    parts = [(slab(rng, (1.0, 0.6, 0.05), (x, 0.0, 0.9), n=2000), b, True)
             for x, b in ((0.0, 0.0), (0.6, 0.8), (1.2, 0.8))]
    c = sourced(2, "counter", parts)
    assert c.obb is not None and c.obb.size[0] == pytest.approx(2.1, abs=0.1)


def test_point_sources_persist(tmp_path: Any) -> None:
    rng = np.random.default_rng(9)
    o = sourced(3, "cup", [(slab(rng, (0.1, 0.1, 0.1), (1, 0, 0)), 0.0, True),
                           (slab(rng, (0.1, 0.1, 0.1), (1.2, 0, 0)), 0.0, False)])
    assert set(np.unique(o.point_sources()).tolist()) >= {mo.SRC_RELIABLE, mo.SRC_LOW_CONFIDENCE}
    files: dict[str, Any] = {}
    tx = SimpleNamespace(write_json=lambda rel, obj: files.__setitem__(rel, obj))
    mo.save_state(tx, ObjectState([o], 10))
    (tmp_path / "objects").mkdir()
    (tmp_path / mo.OBJECTS_JSON).write_text(json.dumps(files[mo.OBJECTS_JSON]))
    np.save(tmp_path / mo.points_file(3), o.points)
    np.save(tmp_path / mo.sources_file(3), o.point_sources())
    (back,) = mo.load_state(lambda rel: tmp_path / rel, {}).objects
    np.testing.assert_array_equal(back.point_sources(), o.point_sources())
    assert back.sightings == o.sightings and not all(s.confident for s in back.sightings)
    np.testing.assert_array_equal(mo.fit_points(back), mo.fit_points(o))
    # maps written before sources and confidence were recorded: every point, every sighting counts
    (tmp_path / mo.sources_file(3)).unlink()
    (legacy,) = mo.load_state(lambda rel: tmp_path / rel, {}).objects
    assert (legacy.point_sources() == mo.SRC_ANY).all()
    assert Sighting.from_list(o.sightings[0].to_list()[:12]).confident


def test_support_bled_onto_is_trimmed_from_a_mask() -> None:
    """A window's mask that ran onto a strip of the windowsill in front of it: the strip is
    trimmed from the mask and from the lifted points; a laptop's base (half of it) and a flat
    keyboard are left whole."""
    from oh_my_slam.segmentation.api import LiftedInstance, trim_support
    from oh_my_slam.segmentation.detect import Detection
    from oh_my_slam.segmentation.lift import lift_mask

    p = np.radians(25.0)  # the camera at the origin looks along +x, 25° down
    f = np.array([np.cos(p), 0.0, -np.sin(p)])
    right = np.array([0.0, -1.0, 0.0])
    pose = Pose(np.stack([right, np.cross(f, right), f], axis=1), np.zeros(3))
    v, u = np.mgrid[0:240, 0:320]
    rays = np.stack([(u - K.cx) / K.fx, (v - K.cy) / K.fy, np.ones(u.shape)], -1) @ pose.R.T

    def scene(vertical: tuple[float, float, float], horizontal: tuple[float, float, float],
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Depth of a vertical plane at x = ``vertical[0]`` (z from its [1] to [2]) above a
        horizontal one at z = ``horizontal[0]`` (x from [1] to [2]), |y| <= 0.4; the masks of
        each."""
        with np.errstate(divide="ignore", invalid="ignore"):
            tv = vertical[0] / rays[..., 0]
            zv = tv * rays[..., 2]
            onv = (tv > 0) & (np.abs(tv * rays[..., 1]) <= 0.4) & (zv >= vertical[1]) \
                & (zv <= vertical[2])
            th = horizontal[0] / rays[..., 2]
            xh = th * rays[..., 0]
            onh = (th > 0) & (np.abs(th * rays[..., 1]) <= 0.4) & (xh >= horizontal[1]) \
                & (xh <= horizontal[2])
        depth = np.where(onv, tv, np.where(onh, th, 3.0))
        onh &= ~onv
        return depth.astype(np.float32), onv, onh

    def instance(mask: np.ndarray, depth: np.ndarray, label: str) -> LiftedInstance:
        lifted = lift_mask(mask, depth, K, None, pose)
        return LiftedInstance(Detection(label, 0.8, "yoloe", mask, (0, 0, 1, 1)), mask, lifted)

    depth, window, sill = scene((0.8, -0.35, 0.3), (-0.35, 0.5, 0.8))
    # the mask covers the sill up to 0.18 m in front of the glass
    xs = np.where(sill, depth * rays[..., 0], np.inf)
    bled = window | (sill & (xs >= 0.62))
    inst = instance(bled, depth, "window")
    out = trim_support(inst, depth, K, np.ones(depth.shape, bool), pose)
    # what is left of the strip lies against the glass (within the footprint's cell)
    assert (out.mask & window).sum() == window.sum() and (xs[out.mask & sill] > 0.75).all()
    assert (bled & sill & ~out.mask).sum() > 0.8 * (bled & sill).sum()
    x = out.lifted.points[:, 0]
    assert np.percentile(x, 2) > 0.77 and np.percentile(inst.lifted.points[:, 0], 2) < 0.66
    # a laptop: its base (half of what the mask covers) is the laptop
    depth, screen, base = scene((0.8, -0.35, -0.15), (-0.35, 0.6, 0.8))
    lap = instance(screen | base, depth, "laptop")
    assert trim_support(lap, depth, K, np.ones(depth.shape, bool), pose) is lap
    # a flat keyboard: nothing stands above its bottom band
    kb = instance(base, depth, "keyboard")
    assert trim_support(kb, depth, K, np.ones(depth.shape, bool), pose) is kb


# --- point counts --------------------------------------------------------------------------------


def test_point_count_is_the_objects_points_in_the_map_cloud() -> None:
    rng = np.random.default_rng(5)
    big = map_object(3, "refrigerator", slab(rng, (0.8, 0.7, 1.8), (0, 0, 0.9), n=200_000), [0])
    small = map_object(7, "cup", slab(rng, (0.1, 0.1, 0.1), (1, 0, 0.9)), [0])
    assert len(big.points) == mo.POINT_CAP  # the stored sample is capped ...
    assert big.point_count == mo.POINT_CAP  # ... and stands in until the cloud is counted
    written: dict[str, Any] = {}
    tx = SimpleNamespace(write_json=lambda rel, obj: written.__setitem__(rel, obj))
    st = ObjectState([big, small], 10)
    labels = np.array([0] * 5 + [3] * 45_000 + [7] * 120 + [9] * 4)  # 9: another, removed id
    mo.set_cloud_counts(tx, st, labels)
    assert big.point_count == 45_000 and small.point_count == 120
    assert big.scene_object().point_count == 45_000
    stored = {d["id"]: d for d in written[mo.OBJECTS_JSON]["objects"]}
    assert stored[3]["point_count"] == 45_000
    assert MapObject.from_dict(stored[3], big.points).point_count == 45_000


# --- export and evaluation of flickering labels ---------------------------------------------------


def test_detected_as_labels_are_exported_and_paired() -> None:
    box = OBB(np.zeros(3), np.eye(3), np.ones(3))
    plain = SceneObject(1, "cup", 0.8, box, 10, 10)
    assert "detected_as" not in json_names(object_entry(plain, "map"))
    merged = SceneObject(2, "door", 0.8, box, 10, 10, frames=[0, 5],
                         labels=("door", "wardrobe"))
    entry = object_entry(merged, "map")
    assert entry["object_data"]["vec"][0]["name"] == "color"  # the colour stays first
    doc = {"openlabel": {"objects": {"1": object_entry(plain, "map"), "2": entry}}}
    objs = doc_objects(doc)
    assert objs[1].labels == ("door", "wardrobe") and objs[0].labels == ()
    # frame 5 detected the door as a wardrobe: the map object is backed by that detection
    assert paired_labels([objs[1].labels], ["wardrobe"]) == 1
    assert paired_labels(["door"], ["wardrobe"]) == 0
    assert paired_labels([("door", "wardrobe"), "cup"], ["mug", "door"]) == 2
    assert ol.vec("detected_as", ["a", "b"]) == {"name": "detected_as", "val": ["a", "b"]}


def json_names(entry: dict[str, Any]) -> set[str]:
    return {v["name"] for v in entry["object_data"].get("vec", [])}
