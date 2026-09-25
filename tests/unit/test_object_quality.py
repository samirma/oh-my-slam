"""Object quality: confirmation from evidence, label-flicker de-duplication, boxes fitted to the
sightings that agree, floor grounding only where the evidence supports it, and point counts from
the map cloud."""

from __future__ import annotations

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
