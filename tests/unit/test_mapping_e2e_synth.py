"""Synthetic end-to-end mapping (AC7, AC10–AC13): real COLMAP on rendered rooms, fake inference.

Update A maps a room with three boxes; update B re-observes it (identity kept, OBBs refined);
update C sees the room after a box was removed, one moved and one added (latest wins)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from oh_my_slam.core.cloud_attrs import CloudScope, parse_cloud_attrs
from oh_my_slam.core.errors import MapLockedError, NotAMapError, RegistrationError
from oh_my_slam.core.geometry import rot_z
from oh_my_slam.core.ply import parse_header, parse_ply
from oh_my_slam.mapping import store
from oh_my_slam.mapping.api import update
from oh_my_slam.mapping.export import scene_bytes
from oh_my_slam.schema.validate import validation_errors
from oh_my_slam.segmentation.colors import color_hex_for_id, hex_to_rgb
from oh_my_slam.segmentation.obb import OBB, obb_iou_upright
from tests.fakes.client import FakeClient
from tests.mapsnap import full_tree_hash
from tests.synth.mapping import add_frames, mapping_room, ring
from tests.synth.scene import Box, Room

pytestmark = pytest.mark.skipif(shutil.which("colmap") is None, reason="needs Homebrew colmap")


def quiet(msg: str) -> None:
    pass


def truth_obb(box: Box) -> OBB:
    return OBB(np.asarray(box.center, float), rot_z(box.yaw), np.asarray(box.size, float))


def objects_by_label(doc: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for oid, o in doc["openlabel"]["objects"].items():
        o = dict(o, id=int(oid))
        out.setdefault(o["type"], []).append(o)
    return out


def cuboid_obb(o: dict) -> OBB:
    from oh_my_slam.core.geometry import quat_to_rot

    v = o["object_data"]["cuboid"][0]["val"]
    return OBB(np.array(v[:3]), quat_to_rot(np.array(v[3:7])), np.array(v[7:10]))


def map_to_world(doc: dict, frames_true: dict[str, object]) -> object:
    """Similarity map → synthetic world from the keyframe camera centres (Umeyama)."""
    from oh_my_slam.core.geometry import umeyama

    src, dst = [], []
    for fr in doc["openlabel"]["frames"].values():
        name = fr["frame_properties"]["keyframe"]
        if name in frames_true:
            src.append(fr["frame_properties"]["transforms"]["camera_1_to_map"][
                "transform_src_to_dst"]["translation"] if "camera_1_to_map" in fr[
                "frame_properties"]["transforms"] else next(iter(fr["frame_properties"][
                    "transforms"].values()))["transform_src_to_dst"]["translation"])
            dst.append(frames_true[name])
    return umeyama(np.array(src), np.array(dst), with_scale=True)


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    base = tmp_path_factory.mktemp("synthmap")
    client = FakeClient()
    room_a = mapping_room()
    poses_a = ring(14)
    imgs_a = add_frames(client, room_a, poses_a, base / "a", "a", depth_noise=0.03, seed=1)
    poses_b = ring(10, start=0.3)
    imgs_b = add_frames(client, room_a, poses_b, base / "b", "b", depth_noise=0.03, seed=2)
    boxes_c = [
        Box(np.array([0.7, -0.1, 0.4]), np.array([0.9, 0.5, 0.8]), 0.3, (220, 40, 40), "cabinet"),
        room_a.boxes[2],  # sofa unchanged
        Box(np.array([-0.6, 0.9, 0.35]), np.array([0.5, 0.5, 0.7]), 0.0, (230, 200, 40),
            "chair"),
    ]
    room_c = Room(boxes=boxes_c)
    poses_c = ring(14, start=0.15)
    imgs_c = add_frames(client, room_c, poses_c, base / "c", "c", depth_noise=0.03, seed=3)
    return {"base": base, "client": client, "room_a": room_a, "room_c": room_c,
            "imgs": (imgs_a, imgs_b, imgs_c), "poses": (poses_a, poses_b, poses_c)}


def test_update_sequence(world) -> None:  # type: ignore[no-untyped-def]
    base, client = world["base"], world["client"]
    imgs_a, imgs_b, imgs_c = world["imgs"]
    poses_a, poses_b, poses_c = world["poses"]
    mdir = base / "map"

    # --- update A: create -----------------------------------------------------------------------
    res = update(mdir, sorted((base / "a").glob("*.png")), mode="full", client=client, progress=quiet)
    doc_a = json.loads(res.payload)
    assert validation_errors(doc_a) == []
    assert len(res.new_frames) >= 13  # >= 90 % registered
    assert (mdir / "map.json").exists() and not (mdir / "mesh").exists()
    for rel in ("frames.json", "cloud.ply", "cloud_objects.npy", "objects.json",
                "sfm/database.db", "sfm/model"):
        assert (mdir / rel).exists(), rel
    assert len(doc_a["openlabel"]["frames"]) == len(res.new_frames)
    # per-stage timings (R44): in the result and, up to the commit, in the update history
    expected_stages = {"setup", "ingest", "inference", "features_matching", "sfm", "map_frame",
                       "depth_alignment", "persist_frames", "validity", "objects", "cloud",
                       "export"}
    assert expected_stages | {"commit"} <= set(res.timings["stages_s"])
    assert sum(res.timings["stages_s"].values()) <= res.timings["total_s"] + 0.01
    assert res.timings["counts"]["keyframes_sampled"] == len(imgs_a)
    assert res.timings["counts"]["keyframes_registered"] == len(res.new_frames)
    assert {"geometry", "gravity", "segmentation", "lift"} <= set(res.timings["parts"])
    hist = json.loads((mdir / "map.json").read_text())["updates"][-1]["timings"]
    assert expected_stages <= set(hist["stages_s"]) and "commit" not in hist["stages_s"]
    for fr in doc_a["openlabel"]["frames"].values():
        assert any(k.endswith("_to_map") for k in fr["frame_properties"]["transforms"])
    by = objects_by_label(doc_a)
    for label in ("cabinet", "box", "sofa"):
        assert len(by.get(label, [])) == 1, (label, by.keys())
    truth_a = {n: p.t for n, p in zip([f"f{i:06d}" for i in range(len(poses_a))], poses_a,
                                      strict=True)}
    sim = map_to_world(doc_a, truth_a)
    assert sim.s == pytest.approx(1.0, rel=0.1)  # metric scale within 10 %
    ious_a = {}
    for label, box in zip(("cabinet", "box", "sofa"), world["room_a"].boxes, strict=True):
        o = by[label][0]
        est = cuboid_obb(o)
        est_w = OBB(sim.apply(est.center[None])[0], sim.R @ est.R, est.size * sim.s)
        ious_a[label] = obb_iou_upright(est_w, truth_obb(box))
        assert ious_a[label] >= 0.5, (label, ious_a[label], est_w.center, est_w.size, box.center,
                                      box.size, sim.s)
        assert o["object_data"]["text"][0]["val"] == color_hex_for_id(o["id"])
    ids_a = {label: by[label][0]["id"] for label in ("cabinet", "box", "sofa")}

    # --- update B: same room again, -t single -------------------------------------------------
    res_b = update(mdir, sorted((base / "b").glob("*.png")), mode="single", client=client, progress=quiet)
    doc_b = json.loads(res_b.payload)
    assert validation_errors(doc_b) == []
    names_b = {fr["frame_properties"]["keyframe"] for fr in doc_b["openlabel"]["frames"].values()}
    assert names_b == set(res_b.new_frames)  # -t single lists exactly the new frames
    full = json.loads(scene_bytes(store.MapReader(mdir)))
    by_b = objects_by_label(full)
    for label, oid in ids_a.items():
        assert [o["id"] for o in by_b[label]] == [oid]  # identity and colour kept
        est = cuboid_obb(by_b[label][0])
        box = world["room_a"].boxes[("cabinet", "box", "sofa").index(label)]
        est_w = OBB(sim.apply(est.center[None])[0], sim.R @ est.R, est.size * sim.s)
        assert obb_iou_upright(est_w, truth_obb(box)) >= min(0.5, ious_a[label] - 0.05)
    single_ids = {int(k) for k in doc_b["openlabel"]["objects"]}
    observed = {int(it["object_id"]) for n in res_b.new_frames for it in json.loads(
        (mdir / store.frame_file(n, "instances.json")).read_text())["instances"]}
    # exactly the map's objects the new frames observed
    assert single_ids == observed & {o["id"] for v in by_b.values() for o in v} != set()

    # --- update C: removed / moved / added (PLY payload with point-cloud attributes) ------------
    attrs = parse_cloud_attrs("color=segment,label=on,normals=on,voxel=0.03", CloudScope.MAP)
    res_c = update(mdir, sorted((base / "c").glob("*.png")), mode="full", fmt="ply", attrs=attrs, client=client,
                   progress=quiet)
    cloud_c = parse_ply(res_c.payload)
    assert parse_header(res_c.payload).comments[-1] == \
        f"attributes {attrs.describe(CloudScope.MAP)}"
    full_c = json.loads(scene_bytes(store.MapReader(mdir)))
    by_c = objects_by_label(full_c)
    # colour contract on the map cloud: every labelled point has its object's colour, the others
    # are grey, and only objects of the scene appear; normals are unit vectors
    assert cloud_c.label is not None and cloud_c.rgb is not None and cloud_c.normals is not None
    scene_ids = {int(k) for k in full_c["openlabel"]["objects"]}
    assert set(np.unique(cloud_c.label)) - {0} <= scene_ids and (cloud_c.label > 0).any()
    for oid in set(np.unique(cloud_c.label)) - {0}:
        np.testing.assert_array_equal(np.unique(cloud_c.rgb[cloud_c.label == oid], axis=0),
                                      [hex_to_rgb(color_hex_for_id(int(oid)))])
    np.testing.assert_array_equal(np.unique(cloud_c.rgb[cloud_c.label == 0], axis=0),
                                  [(128, 128, 128)])
    np.testing.assert_allclose(np.linalg.norm(cloud_c.normals, axis=1), 1.0, atol=1e-4)
    keys = np.floor(cloud_c.xyz.astype(np.float64) / 0.03).astype(np.int64)
    assert len(np.unique(keys, axis=0)) == len(cloud_c)  # one point per voxel
    all_ids = [o["id"] for v in by_c.values() for o in v]
    assert "box" not in by_c  # removed box gone
    assert ids_a["sofa"] in [o["id"] for o in by_c["sofa"]]  # unchanged object kept
    assert "chair" in by_c  # added
    cab_ids = [o["id"] for o in by_c.get("cabinet", [])]
    assert ids_a["cabinet"] not in cab_ids and len(cab_ids) == 1  # moved → new id
    assert max(all_ids) >= max(ids_a.values())
    assert len(all_ids) == len(set(all_ids))
    meta = json.loads((mdir / "map.json").read_text())
    assert meta["update_count"] == 3 and meta["next_object_id"] > max(all_ids)


def test_folder_rules_and_errors(world, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    client = world["client"]
    other = tmp_path / "notamap"
    other.mkdir()
    (other / "file.txt").write_text("x")
    with pytest.raises(NotAMapError):
        update(other, sorted((world["base"] / "a").glob("*.png")), client=client, progress=quiet)
    assert sorted(p.name for p in other.iterdir()) == ["file.txt"]
    hidden_only = tmp_path / "hidden"
    hidden_only.mkdir()
    (hidden_only / ".DS_Store").write_text("")
    assert store.classify(hidden_only) == "empty"
    # a second concurrent update is refused immediately
    locked = tmp_path / "locked"
    with store.MapTransaction(locked):
        with pytest.raises(MapLockedError):
            update(locked, sorted((world["base"] / "a").glob("*.png")), client=client, progress=quiet)


def test_non_overlapping_update_is_rejected(world, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """An image of a different scene cannot be registered: exit 5, map byte-identical."""
    from tests.synth.mapping import add_frames as add

    client = world["client"]
    mdir = tmp_path / "m"
    update(mdir, sorted((world["base"] / "a").glob("*.png")), client=client, progress=quiet)
    before = full_tree_hash(mdir)
    other_room = Room(size=(8.0, 7.0, 3.0), boxes=[], floor_color=(40, 90, 160),
                      wall_color=(90, 160, 60))
    imgs = add(client, other_room, ring(1, radius=3.0), tmp_path / "x", "x")
    with pytest.raises(RegistrationError):
        update(mdir, imgs, client=client, progress=quiet)
    assert full_tree_hash(mdir) == before


@pytest.mark.parametrize("f35_later", [28, 27])
def test_photos_in_a_later_update_keep_refined_intrinsics(tmp_path: Path, f35_later: int) -> None:
    """Office split (OS-2): photos with EXIF mapped 7 + 7. The EXIF focal is 8 % off (a phone's
    35 mm-equivalent tag); the first update refines it. The second update's photos, of the same
    device (same EXIF focal), share the map's camera — its refined focal length, not a second
    camera left at the EXIF prior; of another zoom (another EXIF focal) they get their own camera,
    whose focal length the extension refines and the map keeps. Either way they are posed as
    accurately as the first update's."""
    from PIL import Image

    from oh_my_slam.core.types import Pose
    from oh_my_slam.mapping.frame import similarity_by_poses, transform_pose
    from tests.synth.mapping import K as K_TRUE

    client = FakeClient()
    poses = ring(14, span=1.3 * np.pi)
    imgs = add_frames(client, mapping_room(), poses, tmp_path / "in", "p", depth_noise=0.03,
                      seed=4)
    f35 = 28  # FocalLengthIn35mmFilm: 28 * 500 px / 43.27 mm = 323.5 px, true 300 px
    jpgs = []
    for k, src in enumerate(imgs):
        dst = src.with_suffix(".jpg")
        exif = Image.Exif()
        exif.get_ifd(0x8769)[0xA405] = f35 if k < 7 else f35_later
        Image.open(src).convert("RGB").save(dst, quality=98, exif=exif)
        client.frames[str(dst.resolve())] = client.frames[str(src.resolve())]
        jpgs.append(dst)
    mdir = tmp_path / "map"
    update(mdir, jpgs[:7], client=client, progress=quiet)
    update(mdir, jpgs[7:], client=client, progress=quiet)
    frames = json.loads((mdir / "frames.json").read_text())["frames"]
    assert len(frames) == 14
    cameras = {f["camera_id"] for f in frames}
    assert len(cameras) == (1 if f35_later == f35 else 2)
    for f in frames:  # refined, not the EXIF prior (323.5 or 312.0 px)
        assert abs(f["K"]["fx"] - K_TRUE.fx) < 0.02 * K_TRUE.fx, (f["name"], f["K"]["fx"])
    assert not any(f["low_confidence"] for f in frames)

    def pose(f: dict) -> Pose:
        from oh_my_slam.core.geometry import quat_to_rot

        T = f["T_map_cam"]
        return Pose(quat_to_rot(np.array(T["quaternion_xyzw"])), np.array(T["translation"]))

    first = [f for f in frames if f["update_id"] == 1]
    truth = {f"f{i:06d}": p for i, p in enumerate(poses)}
    sim = similarity_by_poses([pose(f) for f in first], [truth[f["name"]] for f in first])
    for f in frames:
        T, ref = transform_pose(sim, pose(f)), truth[f["name"]]
        rot = np.degrees(np.arccos(np.clip((np.trace(T.R.T @ ref.R) - 1) / 2, -1, 1)))
        assert np.linalg.norm(T.t - ref.t) < 0.03 and rot < 0.5, (f["name"], f["update_id"],
                                                                  T.t - ref.t, rot)


def test_one_update_and_split_updates_agree(world, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """§5 stability: the same 14 frames mapped in one update and in two (7 + 7) give the same
    objects, labels, ids and colours, and boxes within the tolerance that two SfM runs of the
    same frames (7 of which fix the metric scale in the split case) allow."""
    client = world["client"]
    imgs = sorted((world["base"] / "a").iterdir())
    one = json.loads(update(tmp_path / "one", imgs, client=client, progress=quiet).payload)
    update(tmp_path / "two", imgs[:7], client=client, progress=quiet)
    two = json.loads(update(tmp_path / "two", imgs[7:], client=client, progress=quiet).payload)
    a, b = objects_by_label(one), objects_by_label(two)
    assert sorted(a) == sorted(b) == ["box", "cabinet", "sofa"], (a.keys(), b.keys())
    for label in a:
        (oa,), (ob,) = a[label], b[label]
        assert oa["id"] == ob["id"], label
        assert oa["object_data"]["text"] == ob["object_data"]["text"]  # colour
        iou = obb_iou_upright(cuboid_obb(oa), cuboid_obb(ob))
        centre = np.linalg.norm(cuboid_obb(oa).center - cuboid_obb(ob).center)
        assert iou > 0.5 and centre < 0.15, (label, iou, centre)
