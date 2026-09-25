"""Evaluator metrics on synthetic data: yaw alignment and wrap, pitch direction, registration,
same-heading depth agreement, one-update vs split object stability, segmentation vs map."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from oh_my_slam.core.geometry import rot_z
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.mapping.store import FrameRecord
from oh_my_slam.schema import openlabel as ol
from oh_my_slam.tools.evaluate.mapquality import (
    agreement_metrics,
    match_objects,
    split_alignment,
    stability_metrics,
)
from oh_my_slam.tools.evaluate.metrics import Metrics
from oh_my_slam.tools.evaluate.names import captures_in
from oh_my_slam.tools.evaluate.poses import capture_poses, capture_sources, pose_metrics
from oh_my_slam.tools.evaluate.scene import DocObject, pitch_deg, yaw_deg
from oh_my_slam.tools.evaluate.segmentation import (
    MAP_CONSISTENCY,
    MAP_CONSISTENCY_METRICS,
    map_consistency,
    paired_labels,
)

SEQUENCE = Path(__file__).resolve().parents[2] / "examples" / "ainex-captures"
K = Intrinsics(100.0, 100.0, 80.0, 60.0, 160, 120, "given")


def cam(yaw: float, pitch: float = 0.0, t: tuple[float, float, float] = (0, 0, 0)) -> Pose:
    """Camera-to-map pose (OpenCV camera axes, z-up map) looking at ``yaw`` / ``pitch``."""
    y, p = np.radians(yaw), np.radians(pitch)
    f = np.array([np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), np.sin(p)])
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return Pose(np.stack([right, np.cross(f, right), f], axis=1), np.asarray(t, float))


@pytest.fixture(scope="module")
def captures() -> list:
    return captures_in(SEQUENCE)


def commanded_poses(captures: list, offset: float = 0.0) -> dict[str, Pose]:
    """Every capture at its commanded yaw (+ offset), tilted frames pitched ±10°."""
    tilt = {"level": 0.0, "up": 10.0, "down": -10.0}
    return {c.name: cam(c.yaw_deg + offset, tilt[c.tilt], (0.01 * c.index, 0, 0))
            for c in captures}


def test_yaw_and_pitch_of_a_pose() -> None:
    p = cam(130.0, -20.0)
    assert yaw_deg(p.R) == pytest.approx(130.0)
    assert pitch_deg(p.R) == pytest.approx(-20.0)
    assert np.linalg.det(p.R) == pytest.approx(1.0)


def test_poses_decode_from_the_mapper_openlabel_encoding(captures: list, tmp_path: Path) -> None:
    """Scalar-last quaternions of ``camera_*_to_map`` round-trip; frames link to captures through
    the map's keyframe records when the scene does not carry the source."""
    names = [c.name for c in captures[:4]]
    poses = {n: cam(20.0 * i, 5.0 * i, (i, -i, 0.5)) for i, n in enumerate(names)}
    frames = {str(i): ol.frame(float(i), {"camera_0": f"frames/f{i:06d}.jpg"},
                               {"camera_0_to_map": ol.transform("camera_0", "map", poses[n])},
                               keyframe=f"f{i:06d}") for i, n in enumerate(names)}
    doc = ol.document(ol.metadata("m"), {}, coordinate_systems={
        "map": ol.map_cs(["camera_0"]), "camera_0": ol.sensor_cs("map")}, frames=frames)
    write_map(tmp_path, [record(i, n, poses[n]) for i, n in enumerate(names)])
    got = capture_poses(doc, tmp_path)
    assert set(got) == set(names)
    for n in names:
        assert np.allclose(got[n].R, poses[n].R, atol=1e-6)
        assert np.allclose(got[n].t, poses[n].t)
    frames["0"]["frame_properties"]["source"] = "/elsewhere/renamed.jpg"
    assert capture_sources(doc, tmp_path)[0] == "renamed.jpg"


def test_yaw_is_measured_relative_to_frame_001_and_wrapped(captures: list) -> None:
    m = Metrics()
    rows = pose_metrics(m, "pose.t", commanded_poses(captures, offset=-97.0), captures)
    v = {k: x.value for k, x in m.items.items()}
    assert v["pose.t.registered_fraction"] == 1.0
    assert v["pose.t.yaw_err_median_deg"] == pytest.approx(0.0, abs=1e-6)
    assert v["pose.t.yaw_err_max_deg"] == pytest.approx(0.0, abs=1e-6)  # 026 +210 ≡ 078 -150
    assert v["pose.t.same_heading_yaw_diff_max_deg"] == pytest.approx(0.0, abs=1e-6)
    assert v["pose.t.pitch_direction_fraction"] == 1.0
    assert v["pose.t.centre_radius_m"] == pytest.approx(0.39)
    assert rows[25]["commanded_yaw_deg"] == 210 and rows[25]["yaw_deg"] == pytest.approx(-150)


def test_an_offset_reference_frame_shifts_every_error(captures: list) -> None:
    poses = commanded_poses(captures)
    poses[captures[0].name] = cam(12.0)  # frame 001 itself is 12° off
    poses[captures[52].name] = cam(3.0)  # 053 returns to 3° instead of 0°
    m = Metrics()
    pose_metrics(m, "p", poses, captures)
    assert m.items["p.yaw_err_median_deg"].value == pytest.approx(12.0)
    assert m.items["p.yaw_err_max_deg"].value == pytest.approx(12.0)
    assert m.items["p.same_heading_yaw_diff_max_deg"].value == pytest.approx(9.0)


def test_pitch_direction_is_the_sign_against_the_level_sibling(captures: list) -> None:
    poses = commanded_poses(captures)
    poses["005_bootstrap_left015_up.jpg"] = cam(15.0, -1.0)  # "up" but pitched down
    poses["078_right_150_level.jpg"] = cam(-150.0, 20.0)  # 079 (up, +10°) is now below it
    m = Metrics()
    pose_metrics(m, "p", poses, captures)
    frac = m.items["p.pitch_direction_fraction"]
    assert frac.value == pytest.approx(29 / 31)
    assert sorted(frac.detail["wrong"]) == ["005_bootstrap_left015_up.jpg",
                                            "079_right_150_up.jpg"]


def test_unregistered_frames(captures: list) -> None:
    poses = commanded_poses(captures)
    for c in captures[40:48]:
        del poses[c.name]
    m = Metrics()
    pose_metrics(m, "p", poses, captures)
    assert m.items["p.registered_fraction"].value == pytest.approx(71 / 79)
    # 044/045 are gone, 049/050 lost their level frame 048
    assert m.items["p.pitch_direction_fraction"].detail["evaluated"] == 31 - 4
    del poses[captures[0].name]
    m = Metrics()
    pose_metrics(m, "p", poses, captures)
    assert m.items["p.yaw_err_median_deg"].value is None
    assert "not registered" in (m.items["p.yaw_err_median_deg"].error or "")


# -- map quality -------------------------------------------------------------------------------------


def record(index: int, source: str, pose: Pose) -> FrameRecord:
    return FrameRecord(index, f"f{index:06d}", f"frames/f{index:06d}.jpg", f"/in/{source}", 0,
                       K.width, K.height, K, pose, K.width, K.height)


def write_map(root: Path, records: list[FrameRecord],
              depths: dict[str, np.ndarray] | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "map.json").write_text(json.dumps({"format_version": 1, "update_count": 1}))
    (root / "frames.json").write_text(json.dumps({"frames": [r.to_dict() for r in records]}))
    for r in records:
        d = root / "per_frame" / r.name
        d.mkdir(parents=True, exist_ok=True)
        depth = (depths or {}).get(r.name, np.full((K.height, K.width), 2.0))
        np.save(d / "depth.npy", depth.astype(np.float16))


def test_same_heading_depth_agreement(captures: list, tmp_path: Path) -> None:
    """053 sees the wall 3 % farther than 001 from the same pose; 026/078 agree exactly."""
    names = ["001_bootstrap_level.jpg", "053_right_to_000_level.jpg", "026_left_210_level.jpg",
             "078_right_150_level.jpg"]
    recs = [record(i, n, cam(0.0 if i < 2 else 210.0)) for i, n in enumerate(names)]
    write_map(tmp_path, recs, {"f000001": np.full((K.height, K.width), 2.0 * 1.03)})
    m = Metrics()
    agreement_metrics(m, "map.t", tmp_path, captures)
    assert m.items["map.t.frame_agreement_median_pct"].value == pytest.approx(2.91, abs=0.05)
    assert m.items["map.t.frame_agreement_p90_pct"].value == pytest.approx(2.91, abs=0.05)
    pairs = m.items["map.t.frame_agreement_median_pct"].detail
    assert pairs["026_left_210_level.jpg~078_right_150_level.jpg"]["median_pct"] == 0.0


def test_depth_agreement_needs_a_registered_pair(captures: list, tmp_path: Path) -> None:
    write_map(tmp_path, [record(0, "001_bootstrap_level.jpg", cam(0.0))])
    m = Metrics()
    agreement_metrics(m, "map.t", tmp_path, captures)
    assert m.items["map.t.frame_agreement_median_pct"].value is None
    assert "not registered" in (m.items["map.t.frame_agreement_median_pct"].error or "")


def box(oid: int, label: str, centre: tuple[float, float, float],
        size: tuple[float, float, float] = (0.6, 0.4, 0.9), yaw: float = 0.0) -> DocObject:
    return DocObject(oid, label, 0.9, None, None,
                     tuple(ol.cuboid_val(np.array(centre), rot_z(np.radians(yaw)),
                                         np.array(size))))


def test_split_map_objects_are_matched_after_pose_alignment() -> None:
    single = [box(1, "chair", (2, 0, 0.45)), box(2, "sofa", (0, 3, 0.4), (2.0, 0.9, 0.8)),
              box(3, "cup", (1, 1, 0.8), (0.1, 0.1, 0.1)), box(4, "lamp", (-2, -2, 1.0))]
    # the split map's frame is rotated 30° and shifted; its sofa is a "couch", its lamp missing
    T = Pose(rot_z(np.radians(30.0)), np.array([0.2, -0.1, 0.0]))
    inv = T.inverse()

    def moved(o: DocObject, oid: int, label: str) -> DocObject:
        b = o.obb()
        assert b is not None
        b2 = b.transformed(inv)
        return DocObject(oid, label, 0.9, None, None,
                         tuple(ol.cuboid_val(b2.center, b2.R, b2.size)))

    split = [moved(single[0], 1, "chair"), moved(single[1], 7, "couch"),
             moved(single[2], 3, "cup")]
    shared = [cam(a) for a in (0, 40, 80)]
    T_est = split_alignment({str(i): p for i, p in enumerate(shared)},
                            {str(i): inv.compose(p) for i, p in enumerate(shared)})
    assert np.allclose(T_est.R, T.R) and np.allclose(T_est.t, T.t)
    m = Metrics()
    rows = stability_metrics(m, "s", single, split, T_est)
    v = {k: x.value for k, x in m.items.items()}
    assert v["s.matched_fraction"] == pytest.approx(3 / 4)
    assert v["s.label_agreement"] == 1.0  # sofa ~ couch
    assert v["s.id_agreement"] == pytest.approx(2 / 3)
    assert v["s.centre_delta_median_m"] == pytest.approx(0.0, abs=1e-6)
    assert v["s.extent_delta_median_rel"] == pytest.approx(0.0, abs=1e-6)
    assert v["s.obb_iou_median"] > 0.95
    assert {(r["single_id"], r["split_id"]) for r in rows} == {(1, 1), (2, 7), (3, 3)}


def test_overlapping_objects_of_different_labels_are_not_swapped() -> None:
    """A desk and the carpet under it, both in both maps with the same ids, but the split map's
    boxes are shifted so that each overlaps the other label's box of the one-update map more than
    its own. Labels break that tie; an object whose label changed still pairs with its box."""
    def pair_ids(a: list[DocObject], b: list[DocObject]) -> set[tuple[int, int]]:
        ab = [(o, o.obb()) for o in a]
        bb = [(o, o.obb()) for o in b]
        return {(a[i].id, b[j].id) for i, j, _, _ in match_objects(ab, bb)}  # type: ignore[arg-type]

    single = [box(2, "desk", (0.0, 0, 0)), box(35, "carpet", (0.2, 0, 0))]
    split = [box(2, "desk", (0.15, 0, 0)), box(35, "carpet", (0.05, 0, 0))]
    assert pair_ids(single, split) == {(2, 2), (35, 35)}
    relabelled = [box(2, "table", (0.15, 0, 0)), box(35, "carpet", (0.05, 0, 0))]
    assert pair_ids(single, relabelled) == {(2, 2), (35, 35)}
    # geometry alone decides between two objects whose labels both changed
    both = [box(2, "lamp", (0.02, 0, 0)), box(35, "cup", (0.18, 0, 0))]
    assert pair_ids(single, both) == {(2, 2), (35, 35)}


def test_far_apart_boxes_are_not_matched() -> None:
    a = [(o, o.obb()) for o in [box(1, "chair", (0, 0, 0))]]
    b = [(o, o.obb()) for o in [box(1, "chair", (3, 0, 0))]]
    assert match_objects(a, b) == []  # type: ignore[arg-type]
    near = [(o, o.obb()) for o in [box(5, "chair", (0.2, 0, 0))]]
    (i, j, iou, d), = match_objects(a, near)  # type: ignore[arg-type]
    assert (i, j) == (0, 0) and d == pytest.approx(0.2) and 0.3 < iou < 0.6


# -- segmentation ------------------------------------------------------------------------------------


def test_label_pairing_prefers_identical_labels() -> None:
    assert paired_labels(["chair", "chair", "sofa"], ["chair", "couch", "table"]) == 2
    assert paired_labels(["couch", "sofa"], ["sofa"]) == 1
    assert paired_labels([], ["cup"]) == 0


def test_segmentation_is_consistent_with_the_objects_the_map_observed_per_frame() -> None:
    map_objs = [DocObject(1, "chair", 0.9, None, None, None, frozenset({0, 1})),
                DocObject(2, "sofa", 0.9, None, None, None, frozenset({1})),
                DocObject(3, "plant", 0.9, None, None, None, frozenset({5}))]
    frames = {"a.jpg": [DocObject(1, "chair", 0.8, None, None, None)],
              "b.jpg": [DocObject(1, "couch", 0.7, None, None, None),
                        DocObject(2, "cup", 0.6, None, None, None)]}
    m = Metrics()
    rows = map_consistency(m, MAP_CONSISTENCY, frames, map_objs,
                           {0: "a.jpg", 1: "b.jpg", 5: "c.jpg"})
    # frame a: chair ↔ chair; frame b: {chair, sofa} vs {couch, cup} → sofa ~ couch
    assert set(m.items) == {f"{MAP_CONSISTENCY}.{k}" for k in MAP_CONSISTENCY_METRICS}
    assert m.items["seg.map_consistency.map_objects_detected"].value == pytest.approx(2 / 3)
    assert m.items["seg.map_consistency.detections_in_map"].value == pytest.approx(2 / 3)
    assert [r["paired"] for r in rows] == [1, 1]
