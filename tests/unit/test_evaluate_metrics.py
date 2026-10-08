"""Evaluator metrics on synthetic data: yaw alignment and wrap, pitch direction, registration,
the pan order and tilt agreement of a pan-tilt sequence, same-heading and all-pairs depth
agreement, one-update vs split object stability (label agreement on a label-blind pairing),
near-duplicate objects, detected mask points and cloud points outside their object's box,
segmentation vs map."""

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
    FAR_GAP,
    PAIRS_MAX_ANGLE_DEG,
    Published,
    agreement_metrics,
    match_objects,
    overlapping_pairs,
    split_alignment,
    stability_metrics,
)
from oh_my_slam.tools.evaluate.metrics import Metrics
from oh_my_slam.tools.evaluate.names import (
    captures_in,
    pan_captures_in,
    same_heading_pairs,
    tilt_pairs,
)
from oh_my_slam.tools.evaluate.poses import (
    capture_poses,
    capture_sources,
    pan_pose_metrics,
    pose_metrics,
)
from oh_my_slam.tools.evaluate.scene import DocObject, pitch_deg, yaw_deg
from oh_my_slam.tools.evaluate.segmentation import detection_row, paired_labels

SEQUENCE = Path(__file__).resolve().parents[2] / "examples" / "ainex-captures"
CAMERA = Path(__file__).resolve().parents[2] / "examples" / "camera"
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


# -- a pan-tilt sequence (examples/camera) -----------------------------------------------------------

STEP = 20.0  # the pan step the names do not record
TILT = {"mid": 0.0, "up": 15.0, "down": -15.0}


@pytest.fixture(scope="module")
def pan_captures() -> list:
    return pan_captures_in(CAMERA)


def pan_poses(captures: list, offset: float = 0.0) -> dict[str, Pose]:
    """Every capture at its pan position's heading (turning left by ``STEP`` per position, plus
    ``offset``), pitched by its tilt."""
    return {c.name: cam(STEP * (c.pan - 3) + offset, TILT[c.tilt]) for c in captures}


def test_a_camera_turning_left_with_level_tilts(pan_captures: list) -> None:
    """The headings follow the pan order across ±180° (offset 110°: p06 is at 170°, p07 at
    -170°), the tilts of a pan position agree, and up / down are pitched that way from mid."""
    m = Metrics()
    rows = pan_pose_metrics(m, "pose.c", pan_poses(pan_captures, offset=110.0), pan_captures)
    v = {k: x.value for k, x in m.items.items()}
    assert v == pytest.approx({
        "pose.c.registered_fraction": 1.0, "pose.c.pan_order_fraction": 1.0,
        "pose.c.tilt_yaw_diff_median_deg": 0.0, "pose.c.tilt_yaw_diff_max_deg": 0.0,
        "pose.c.pitch_direction_fraction": 1.0}, abs=1e-6)
    order = m.items["pose.c.pan_order_fraction"].detail
    assert order["steps_deg"]["p06~p07"] == pytest.approx(STEP) and order["wrong"] == []
    assert len(order["steps_deg"]) == 8
    assert len(m.items["pose.c.tilt_yaw_diff_max_deg"].detail) == 27
    assert m.items["pose.c.pitch_direction_fraction"].detail == {"evaluated": 18, "wrong": []}
    first, up = rows[0], rows[2]
    assert (first["capture"], first["pan"], first["tilt"]) == ("img_007_p03_down.jpg", 3, "down")
    assert first["yaw_deg"] == 0.0 and first["pitch_delta_deg"] == pytest.approx(-15.0)
    assert up["pitch_ok"] and rows[-1]["yaw_deg"] == pytest.approx(8 * STEP)


def test_a_step_to_the_right_and_disagreeing_tilts(pan_captures: list) -> None:
    poses = pan_poses(pan_captures)
    for c in pan_captures:
        if c.pan == 6:  # p06 is right of p05: the camera turned back (p07 is left of both)
            poses[c.name] = cam(STEP * 1.5, TILT[c.tilt])
    poses["img_013_p05_down.jpg"] = cam(STEP * 2 + 4.0, -15.0)  # 4° off its pan position
    poses["img_015_p05_up.jpg"] = cam(STEP * 2, -2.0)  # "up" but pitched below mid
    m = Metrics()
    pan_pose_metrics(m, "p", poses, pan_captures)
    order = m.items["p.pan_order_fraction"]
    assert order.value == pytest.approx(7 / 8) and order.detail["wrong"] == ["p05~p06"]
    assert m.items["p.tilt_yaw_diff_max_deg"].value == pytest.approx(4.0)
    assert m.items["p.tilt_yaw_diff_median_deg"].value == pytest.approx(0.0, abs=1e-6)
    pitch = m.items["p.pitch_direction_fraction"]
    assert pitch.value == pytest.approx(17 / 18) and pitch.detail["wrong"] == [
        "img_015_p05_up.jpg"]


def test_unregistered_camera_frames(pan_captures: list) -> None:
    poses = pan_poses(pan_captures)
    del poses["img_008_p03_mid.jpg"]  # p03's up and down lose their mid frame
    for c in pan_captures[3:6]:  # p04 entirely
        del poses[c.name]
    m = Metrics()
    pan_pose_metrics(m, "p", poses, pan_captures)
    reg = m.items["p.registered_fraction"]
    assert reg.value == pytest.approx(23 / 27) and len(reg.detail["missing"]) == 4
    assert m.items["p.pan_order_fraction"].detail["steps_deg"]["p03~p05"] == pytest.approx(
        2 * STEP)  # the step over the missing pan position
    assert len(m.items["p.tilt_yaw_diff_max_deg"].detail) == 27 - 3 - 2  # p04, and p03 but one
    assert m.items["p.pitch_direction_fraction"].detail["evaluated"] == 18 - 2 - 2
    m = Metrics()
    pan_pose_metrics(m, "p", {"img_008_p03_mid.jpg": cam(0.0)}, pan_captures)
    assert m.items["p.registered_fraction"].value == pytest.approx(1 / 27)
    errors = {k: x.error for k, x in m.items.items() if x.value is None}
    assert errors == {
        "p.pan_order_fraction": "fewer than two pan positions registered",
        "p.tilt_yaw_diff_median_deg": "no pan position has two registered tilts",
        "p.tilt_yaw_diff_max_deg": "no pan position has two registered tilts",
        "p.pitch_direction_fraction": "no up/down capture with a registered mid sibling"}
    m = Metrics()
    pan_pose_metrics(m, "p", {}, [])
    assert m.items["p.registered_fraction"].error == "no captures"


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
    agreement_metrics(m, "map.t", tmp_path, same_heading_pairs(captures))
    assert m.items["map.t.frame_agreement_median_pct"].value == pytest.approx(2.91, abs=0.05)
    assert m.items["map.t.frame_agreement_p90_pct"].value == pytest.approx(2.91, abs=0.05)
    pairs = m.items["map.t.frame_agreement_median_pct"].detail
    assert pairs["026_left_210_level.jpg~078_right_150_level.jpg"]["median_pct"] == 0.0


def test_the_tilts_of_a_pan_position_are_judged_like_overlapping_pairs(
        pan_captures: list, tmp_path: Path) -> None:
    """The camera's same-heading pairs are the tilts of one pan position, judged like the
    overlapping pairs over them alone (user ruling 2026-10-08): median, p90, share above 10 % and
    worst of the pairs' medians. p03's up frame sees the sphere 4 % farther than its other tilts;
    p04's up frame 6 % and its down frame 12 % farther than its mid frame."""
    tilt = {"down": -10.0, "mid": 0.0, "up": 10.0}
    farther = {"img_009_p03_up.jpg": 1.04, "img_010_p04_up.jpg": 1.06,
               "img_012_p04_down.jpg": 1.12}
    caps = pan_captures[:6]  # p03: down, mid, up; p04: up, mid, down
    recs = [record(i, c.name, cam(30.0 * (c.pan - 3), tilt[c.tilt])) for i, c in enumerate(caps)]
    write_map(tmp_path, recs, {r.name: sphere_depth() * farther.get(c.name, 1.0)
                               for r, c in zip(recs, caps, strict=True)})
    m = Metrics()
    agreement_metrics(m, "map.t", tmp_path, tilt_pairs(caps), tilts=True)
    v = {k: x.value for k, x in m.items.items()}
    detail = m.items["map.t.frame_agreement_tilt_median_pct"].detail
    pairs = detail["same_heading"]
    assert list(pairs) == ["img_007_p03_down.jpg~img_008_p03_mid.jpg",
                           "img_007_p03_down.jpg~img_009_p03_up.jpg",
                           "img_008_p03_mid.jpg~img_009_p03_up.jpg",
                           "img_010_p04_up.jpg~img_011_p04_mid.jpg",
                           "img_010_p04_up.jpg~img_012_p04_down.jpg",
                           "img_011_p04_mid.jpg~img_012_p04_down.jpg"]
    # |z_i→j / z_j - 1|, the earlier capture i: 0, 3.85, 3.85 (p03); 6.0, 5.36, 10.71 (p04)
    medians = [pairs[k]["median_pct"] for k in pairs]
    assert medians == pytest.approx([0.0, 3.85, 3.85, 6.0, 5.36, 10.71], abs=0.2)
    assert detail["pairs"] == 6
    assert v["map.t.frame_agreement_tilt_median_pct"] == pytest.approx((3.85 + 5.36) / 2, abs=0.2)
    assert v["map.t.frame_agreement_tilt_p90_pct"] == pytest.approx((6.0 + 10.71) / 2, abs=0.2)
    assert v["map.t.frame_agreement_tilt_over10_pct"] == pytest.approx(100 / 6, abs=0.01)
    assert v["map.t.frame_agreement_tilt_max_pct"] == pytest.approx(10.71, abs=0.2)
    # the worst pair's median and p90 judge ainex's revisits only
    assert "map.t.frame_agreement_median_pct" not in v and "map.t.frame_agreement_p90_pct" not in v
    assert v["map.t.frame_agreement_pairs_max_pct"] is not None


def test_depth_agreement_needs_a_registered_pair(captures: list, tmp_path: Path) -> None:
    write_map(tmp_path, [record(0, "001_bootstrap_level.jpg", cam(0.0))])
    m = Metrics()
    agreement_metrics(m, "map.t", tmp_path, same_heading_pairs(captures))
    assert m.items["map.t.frame_agreement_median_pct"].value is None
    assert "not registered" in (m.items["map.t.frame_agreement_median_pct"].error or "")


def sphere_depth(radius: float = 2.0) -> np.ndarray:
    """z-depth of a sphere around the camera centre: every rotation of the camera agrees."""
    v, u = np.mgrid[0:K.height, 0:K.width]
    x, y = (u - K.cx) / K.fx, (v - K.cy) / K.fy
    return (radius / np.sqrt(1.0 + x * x + y * y)).astype(np.float32)


def test_all_overlapping_pairs_agreement(captures: list, tmp_path: Path) -> None:
    """Two laps of a head turning in place (12° steps, the second lap offset by 5°) inside a
    sphere; keyframes 50-52 place it 15 % further (a depth that drifted over a few keyframes).
    Every pair less than 45° apart is compared, whatever its distance in capture order: sequence
    neighbours and loop closures; the detail splits them."""
    yaws = [12.0 * k for k in range(30)] + [12.0 * k + 5.0 for k in range(30)]
    recs = [record(i, captures[i % len(captures)].name, cam(y)) for i, y in enumerate(yaws)]

    def off(r: FrameRecord) -> bool:
        return 50 <= r.index < 53

    depths = {r.name: sphere_depth() * (1.15 if off(r) else 1.0) for r in recs}
    write_map(tmp_path, recs, depths)
    pairs = overlapping_pairs(recs)
    assert (recs[0], recs[1]) in pairs and (recs[0], recs[3]) in pairs  # neighbours, 12-36°
    assert (recs[0], recs[4]) not in pairs  # 48° apart
    assert (recs[0], recs[29]) in pairs  # the first lap closes on itself (12° apart)
    assert PAIRS_MAX_ANGLE_DEG == 45.0
    m = Metrics()
    worst = agreement_metrics(m, "map.t", tmp_path, same_heading_pairs(captures))
    drifted = [(a, b) for a, b in pairs if off(a) != off(b)]
    median = m.items["map.t.frame_agreement_pairs_median_pct"]
    detail = median.detail
    assert detail["pairs"] == len(pairs) and 0 < len(drifted) < len(pairs) / 10
    assert median.value == pytest.approx(0.0, abs=0.2)  # float16 depth storage
    over = m.items["map.t.frame_agreement_pairs_over10_pct"].value
    assert over == pytest.approx(100.0 * len(drifted) / len(pairs), abs=0.01)
    # a drifted keyframe compared into an undrifted one: 15 %; the reverse: |1/1.15 - 1| = 13 %
    assert {round(r["median_pct"]) for r in worst} == {15, 13}
    assert worst[0]["median_pct"] == pytest.approx(15.0, abs=0.1)
    assert m.items["map.t.frame_agreement_pairs_max_pct"].value == pytest.approx(15.0, abs=0.1)
    assert m.items["map.t.frame_agreement_pairs_p90_pct"].value == pytest.approx(0.0, abs=0.2)
    near = [(a, b) for a, b in pairs if b.index - a.index <= FAR_GAP]
    assert detail["neighbours"]["pairs"] == len(near)
    assert detail["far"]["pairs"] == len(pairs) - len(near)
    assert detail["neighbours"]["over10_pct"] == pytest.approx(
        100.0 * sum(1 for p in drifted if p in near) / len(near), abs=0.01)
    assert {"gap", "angle_deg"} <= set(worst[0])
    assert set(detail["same_heading"]) >= {"001_bootstrap_level.jpg~053_right_to_000_level.jpg"}


def test_a_pair_that_disagrees_everywhere_counts_as_grossly_inconsistent(
        captures: list, tmp_path: Path) -> None:
    recs = [record(0, captures[0].name, cam(0.0)), record(1, captures[1].name, cam(5.0))]
    write_map(tmp_path, recs, {"f000001": sphere_depth() * 1.6, "f000000": sphere_depth()})
    m = Metrics()
    agreement_metrics(m, "map.t", tmp_path, same_heading_pairs(captures))
    assert m.items["map.t.frame_agreement_pairs_max_pct"].value == pytest.approx(30.0)


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
    assert m.items["s.matched_fraction"].detail["extra_published"] == []
    assert v["s.unexcused_extra"] == 0


def test_only_extra_objects_an_earlier_update_published_are_not_counted() -> None:
    """mapper.md: the split map may keep an object an earlier update published that no later
    image contradicts. matched_fraction is the share of the one-update map's objects the split
    map has; an extra object an earlier update published (that id, label and place, carried into
    the split map's frame) is listed, any other counts in unexcused_extra."""
    single = [box(1, "chair", (2, 0, 0.45)), box(2, "lamp", (-2, -2, 1.0))]
    split = [box(1, "chair", (2, 0, 0.45)), box(2, "lamp", (-2, -2, 1.0)),
             box(9, "box", (4, 4, 0.2)), box(10, "bag", (-4, 3, 0.2))]
    # the earlier update's frame is 0.5 m off along x: its box 9 is the split map's box 9
    T = Pose(np.eye(3), np.array([0.5, 0.0, 0.0]))
    earlier = [Published([box(9, "box", (3.5, 4, 0.2)), box(10, "bag", (0, 0, 0.2))], T)]
    m = Metrics()
    stability_metrics(m, "s", single, split, Pose.identity(), earlier)
    mf = m.items["s.matched_fraction"]
    assert mf.value == 1.0
    assert mf.detail["extra_published"] == [{"id": 9, "label": "box"}]
    assert mf.detail["extra_unexcused"] == [{"id": 10, "label": "bag"}]  # its id, elsewhere
    assert m.items["s.unexcused_extra"].value == 1
    m = Metrics()
    stability_metrics(m, "s", single, split, Pose.identity())  # nothing published earlier
    assert m.items["s.unexcused_extra"].value == 2
    m = Metrics()
    stability_metrics(m, "s", single, split[:1], Pose.identity())  # one of its objects is missing
    assert m.items["s.matched_fraction"].value == 0.5
    m = Metrics()
    stability_metrics(m, "s", [], split, Pose.identity())
    assert m.items["s.matched_fraction"].value is None


def test_an_id_differs_only_where_an_earlier_update_published_it_for_that_object() -> None:
    """mapper.md: the split map's id may differ from the one-update map's where an earlier update
    published it for this object. Ids two objects swapped are no such allowance."""
    single = [box(7, "chair", (2, 0, 0.45)), box(8, "table", (0, 3, 0.4))]
    earlier = [Published([box(1, "chair", (2, 0, 0.45)), box(2, "table", (0, 3, 0.4))],
                         Pose.identity())]
    kept = [box(1, "chair", (2, 0, 0.45)), box(2, "table", (0, 3, 0.4))]
    m = Metrics()
    stability_metrics(m, "s", single, kept, Pose.identity(), earlier)
    ids = m.items["s.id_agreement"]
    assert ids.value == 1.0 and ids.detail == {"same_id": 0.0,
                                               "published_earlier": [[7, 1], [8, 2]]}
    swapped = [box(2, "chair", (2, 0, 0.45)), box(1, "table", (0, 3, 0.4))]
    m = Metrics()
    stability_metrics(m, "s", single, swapped, Pose.identity(), earlier)
    ids = m.items["s.id_agreement"]
    assert ids.value == 0.0 and ids.detail["published_earlier"] == []
    # the same label, but the earlier update published that id for an object elsewhere
    moved = [Published([box(1, "chair", (5, 5, 0.45)), box(2, "table", (0, 3, 0.4))],
                       Pose.identity())]
    m = Metrics()
    stability_metrics(m, "s", single, kept, Pose.identity(), moved)
    assert m.items["s.id_agreement"].detail["published_earlier"] == [[8, 2]]


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


def test_label_agreement_is_measured_on_a_label_blind_pairing() -> None:
    """Ids and boxes are compared on a label-aware pairing (the desk pairs with the desk), but the
    labels are compared on pairs that geometry alone chose: when each split box lies on the
    other label's box, the labels disagree there, whatever the label-aware pairing says."""
    single = [box(2, "desk", (0.0, 0, 0)), box(35, "carpet", (0.2, 0, 0))]
    split = [box(2, "desk", (0.15, 0, 0)), box(35, "carpet", (0.05, 0, 0))]
    m = Metrics()
    rows = stability_metrics(m, "s", single, split, Pose.identity())
    assert {(r["single_id"], r["split_id"]) for r in rows} == {(2, 2), (35, 35)}
    agreement = m.items["s.label_agreement"]
    assert m.items["s.id_agreement"].value == 1.0
    assert agreement.value == 0.0 and agreement.detail["label_aware"] == 1.0
    assert sorted(agreement.detail["disagreeing"]) == [[2, "desk", 35, "carpet"],
                                                       [35, "carpet", 2, "desk"]]
    # boxes that stay in place agree however they are paired
    m = Metrics()
    stability_metrics(m, "s", single, [box(2, "table", (0.01, 0, 0)),
                                       box(35, "rug", (0.21, 0, 0))], Pose.identity())
    assert m.items["s.label_agreement"].value == 1.0


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


def test_detections_labels_and_scores_per_frame() -> None:
    objs = [DocObject(1, "chair", 0.9, None, None, None), DocObject(2, "chair", 0.6, None, None,
                                                                     None),
            DocObject(3, "sofa", None, None, None, None)]
    assert detection_row("a.jpg", objs) == {
        "image": "a.jpg", "objects": 3, "labels": {"chair": 2, "sofa": 1}, "min_score": 0.6,
        "median_score": 0.75, "scores": [0.9, 0.6]}
    assert detection_row("b.jpg", [])["min_score"] is None
