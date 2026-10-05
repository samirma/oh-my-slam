"""``mapper.sh locate`` (spec 2.3): real COLMAP on rendered rooms, fake inference.

A map of ring a (14 views); held-out views of ring b are located and checked against the truth;
an image of another room is reported as unlocalisable; the map folder never changes."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from oh_my_slam.core.cloud_attrs import CloudAttrs, CloudScope, parse_cloud_attrs
from oh_my_slam.core.errors import InputError, NotAMapError, ServerUnavailableError, UsageError
from oh_my_slam.core.ply import parse_header, parse_ply
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.mapping import store
from oh_my_slam.mapping.api import update
from oh_my_slam.mapping.export import map_cloud, scene_bytes
from oh_my_slam.mapping.frame import similarity_by_poses, transform_pose
from oh_my_slam.mapping.locate import (
    LOCATED_CS,
    LocateResult,
    check_output,
    locate,
    open_map,
    resolve_images,
    visible_points,
)
from oh_my_slam.schema.validate import validation_errors
from oh_my_slam.segmentation.cloud import MAP_FRAME
from tests.fakes.client import FakeClient, FakeFrame
from tests.mapsnap import snapshot
from tests.synth.mapping import add_frames, mapping_room, ring
from tests.synth.scene import Room

REPO = Path(__file__).resolve().parents[2]
needs_colmap = pytest.mark.skipif(shutil.which("colmap") is None, reason="needs Homebrew colmap")


def quiet(msg: str) -> None:
    pass


def rot_deg(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1.0, 1.0))))


def located_frames(doc: dict) -> dict[str, dict]:
    return {k: f["frame_properties"] for k, f in doc["openlabel"]["frames"].items()
            if f["frame_properties"].get("located")}


def pose_of(props: dict, key: str) -> Pose:
    from oh_my_slam.core.geometry import quat_to_rot

    tr = props["transforms"][f"{key}_to_map"]["transform_src_to_dst"]
    return Pose(quat_to_rot(np.array(tr["quaternion"])), np.array(tr["translation"]))


def run(mdir: Path, images: list[Path], **kw: object) -> LocateResult:
    """``locate`` as the CLI calls it: resolved images, opened map."""
    return locate(open_map(mdir), resolve_images(images), progress=quiet, **kw)  # type: ignore[arg-type]


def sh(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run([str(REPO / "mapper.sh"), *args], capture_output=True, timeout=300,
                          env=os.environ.copy())


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    base = tmp_path_factory.mktemp("locate")
    client = FakeClient()
    room = mapping_room()
    poses_a, poses_b = ring(14), ring(10, start=0.3)
    imgs_a = add_frames(client, room, poses_a, base / "a", "a", depth_noise=0.03, seed=1)
    imgs_b = add_frames(client, room, poses_b, base / "b", "b", depth_noise=0.03, seed=2)
    other_room = Room(size=(8.0, 7.0, 3.0), boxes=[], floor_color=(40, 90, 160),
                      wall_color=(90, 160, 60))
    other = add_frames(client, other_room, ring(1, radius=3.0), base / "x", "x")[0]
    mdir = base / "map"
    res = update(mdir, imgs_a, mode="full", client=client, progress=quiet)
    reader = store.MapReader(mdir)
    truth = {f"f{i:06d}": p for i, p in enumerate(poses_a)}
    sim = similarity_by_poses([r.T_map_cam for r in reader.frames],
                              [truth[r.name] for r in reader.frames])  # map -> world
    return SimpleNamespace(base=base, client=client, map=mdir, update_doc=json.loads(res.payload),
                           imgs_b=imgs_b, poses_b=poses_b, other=other, sim=sim)


# ------------------------------------------------------------------------------------------------
# poses, scopes and formats (COLMAP)


@needs_colmap
def test_located_poses_match_the_truth_and_the_map_is_untouched(world) -> None:  # type: ignore[no-untyped-def]
    before, hashed = snapshot(world.map), store.full_tree_hash(world.map)
    res = run(world.map, world.imgs_b)  # defaults: json, single
    assert snapshot(world.map) == before and store.full_tree_hash(world.map) == hashed
    assert not (world.map / store.STAGING).exists()
    assert [r.located for r in res.results] == [True] * len(world.imgs_b), \
        [r.reason for r in res.results]
    doc = json.loads(res.payload)
    assert validation_errors(doc) == []
    root = doc["openlabel"]
    assert root["objects"] == {} and root["metadata"]["scope"] == "single"
    frames = located_frames(doc)
    assert len(frames) == len(root["frames"]) == len(world.imgs_b)
    reader = store.MapReader(world.map)
    base = max(r.index for r in reader.frames) + 1
    for k, (img, truth) in enumerate(zip(world.imgs_b, world.poses_b, strict=True)):
        props = frames[str(base + k)]
        key = f"{LOCATED_CS}{k}"
        assert props["image"] == str(img) and props["streams"][key]["uri"] == str(img)
        assert props["timestamp"] == float(base + k)  # as a keyframe's: its frame key
        T = pose_of(props, key)
        r = res.results[k]
        np.testing.assert_allclose(T.t, r.T_map_cam.t, atol=1e-5)  # type: ignore[union-attr]
        W = transform_pose(world.sim, T)
        assert np.linalg.norm(W.t - truth.t) < 0.03 and rot_deg(W.R, truth.R) < 0.5, \
            (img.name, W.t - truth.t, rot_deg(W.R, truth.R))
        assert root["coordinate_systems"][key]["parent"] == "map"
        fx = root["streams"][key]["stream_properties"]["intrinsics_pinhole"]["camera_matrix"][0]
        assert abs(fx - 300.0) < 0.03 * 300.0  # the map camera's refined focal (true 300 px)
    assert {"setup", "features_matching", "pose", "export"} <= set(res.timings["stages_s"])
    assert res.timings["counts"]["located"] == len(world.imgs_b)


@needs_colmap
def test_another_camera_gets_its_focal_length_estimated(world, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """A query of another image size shares no map camera: its focal length is estimated with
    the pose (here 360 px at 480 x 360, against the map's 300 px at 400 x 300)."""
    from oh_my_slam.core.images import png_bytes
    from tests.synth.mapping import mapping_room
    from tests.synth.scene import render

    K2 = Intrinsics(360.0, 360.0, 240.0, 180.0, 480, 360)
    truth = ring(1, start=1.0)[0]
    img = tmp_path / "other_camera.png"
    img.write_bytes(png_bytes(render(mapping_room(), truth, K2).rgb))
    (r,) = run(world.map, [img]).results
    assert r.located, r.reason
    assert r.K is not None and abs(r.K.fx - 360.0) < 0.03 * 360.0 and r.K.source == "colmap"
    W = transform_pose(world.sim, r.T_map_cam)  # type: ignore[arg-type]
    assert np.linalg.norm(W.t - truth.t) < 0.05 and rot_deg(W.R, truth.R) < 1.0, \
        (W.t - truth.t, rot_deg(W.R, truth.R))


@needs_colmap
def test_full_scope_is_the_update_document_plus_the_located_cameras(world) -> None:  # type: ignore[no-untyped-def]
    imgs = [world.imgs_b[0], world.other, world.imgs_b[5]]
    res = run(world.map, imgs, mode="full")
    assert [r.located for r in res.results] == [True, False, True]
    assert "not enough overlap" in res.results[1].reason
    doc = json.loads(res.payload)
    assert validation_errors(doc) == []
    root = doc["openlabel"]
    located = located_frames(doc)
    assert sorted(p["image"] for p in located.values()) == sorted(map(str, imgs[::2]))
    keys = [f"{LOCATED_CS}0", f"{LOCATED_CS}2"]
    assert set(root["coordinate_systems"]) - set(world.update_doc["openlabel"][
        "coordinate_systems"]) == set(keys)
    # without the located cameras it is exactly the map as update -t full returned it
    for k in located:
        del root["frames"][k]
    for key in keys:
        del root["coordinate_systems"][key]
        del root["streams"][key]
        root["coordinate_systems"]["map"]["children"].remove(key)
    from oh_my_slam.schema.openlabel import frame_intervals

    root["frame_intervals"] = frame_intervals([int(k) for k in root["frames"]])
    assert doc == world.update_doc
    assert doc == json.loads(scene_bytes(store.MapReader(world.map)))
    assert set(located) & set(world.update_doc["openlabel"]["frames"]) == set()


@needs_colmap
def test_ply_poses_in_the_header_and_visible_points(world) -> None:  # type: ignore[no-untyped-def]
    from oh_my_slam.schema import openlabel as ol

    imgs = [world.imgs_b[0], world.other, world.imgs_b[1]]  # the second cannot be located
    js = run(world.map, imgs)
    doc = json.loads(js.payload)
    single = run(world.map, imgs, fmt="ply")
    full = run(world.map, imgs, fmt="ply", mode="full")
    attrs = CloudAttrs()
    for res in (single, full):
        comments = parse_header(res.payload).comments
        assert comments[:2] == [MAP_FRAME, f"attributes {attrs.describe(CloudScope.MAP)}"]
        assert len(comments) == 2 + len(imgs)  # one line per input image
        for k, line in enumerate(comments[2:]):
            key, payload = line.split(" ", 1)
            d = json.loads(payload)
            assert key == f"{LOCATED_CS}{k}" and d["image"] == str(imgs[k])
            if k == 1:
                assert d == {"image": str(world.other), "located": False}  # no pose
                continue
            r = js.results[k]
            assert d["located"] and d["transform_src_to_dst"] == ol.transform_data(r.T_map_cam)  # type: ignore[arg-type]
            # the representation of the JSON result: its frame transform and stream properties
            frame = next(f["frame_properties"] for f in doc["openlabel"]["frames"].values()
                         if f["frame_properties"]["image"] == str(imgs[k]))
            assert d["transform_src_to_dst"] == frame["transforms"][f"{key}_to_map"][
                "transform_src_to_dst"]
            assert d["stream_properties"] == doc["openlabel"]["streams"][key]["stream_properties"]
    whole = map_cloud(store.MapReader(world.map))
    c_full, c_single = parse_ply(full.payload), parse_ply(single.payload)
    assert len(c_full) == len(whole)
    np.testing.assert_allclose(c_full.xyz, whole.xyz, atol=1e-6)
    rows = {tuple(p) for p in np.round(c_full.xyz, 5)}
    assert 0 < len(c_single) < len(c_full)
    assert all(tuple(p) in rows for p in np.round(c_single.xyz, 5))
    # -p applies to the map-scope attributes
    seg = parse_cloud_attrs("color=segment,label=on,voxel=0.05", CloudScope.MAP)
    voxel = parse_ply(run(world.map, imgs, fmt="ply", attrs=seg).payload)
    assert voxel.label is not None and len(voxel) < len(c_single)


@needs_colmap
def test_only_unlocalisable_images_are_an_input_error(world) -> None:  # type: ignore[no-untyped-def]
    before = snapshot(world.map)
    with pytest.raises(InputError, match="none of the images"):
        run(world.map, [world.other])
    res = sh("locate", "-i", str(world.other), "-m", str(world.map))
    assert res.returncode == 2 and res.stdout == b""
    assert str(world.other).encode() in res.stderr and b"not located" in res.stderr
    assert snapshot(world.map) == before


@needs_colmap
def test_cli_stdout_output_file_and_errors(world, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    before = snapshot(world.map)
    img = str(world.imgs_b[1])
    res = sh("locate", "-i", img, "-m", str(world.map))
    assert res.returncode == 0, res.stderr.decode()
    doc = json.loads(res.stdout)
    assert validation_errors(doc) == [] and len(located_frames(doc)) == 1
    assert b"timings: total" in res.stderr
    out = tmp_path / "res" / "located.ply"
    res = sh("locate", "-i", img, "-m", str(world.map), "-f", "ply", "-t", "full", "-o", str(out),
             "-p", "normals=on")
    assert res.returncode == 0 and res.stdout == b"", res.stderr.decode()
    assert parse_ply(out.read_bytes()).normals is not None
    inside = sh("locate", "-i", img, "-m", str(world.map), "-o", str(world.map / "x.json"))
    assert inside.returncode == 2 and b"inside the map" in inside.stderr
    other = tmp_path / "other"
    other.mkdir()
    (other / "notes.txt").write_text("x")
    assert sh("locate", "-i", img, "-m", str(other)).returncode == 4  # not a map
    assert snapshot(world.map) == before


@needs_colmap
def test_an_update_committing_during_locate_gives_a_consistent_result(world, tmp_path: Path,  # type: ignore[no-untyped-def]
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """An update commits between feature matching and pose: locate starts again on the new map,
    and its -t full result is the new map's scene (plus the located cameras), never a mix."""
    from oh_my_slam.mapping import locate as lmod

    mdir = tmp_path / "map"
    shutil.copytree(world.map, mdir)
    before = json.loads(scene_bytes(store.MapReader(mdir)))
    real = lmod._MapPoints
    commits = []

    class CommitFirst(real):  # type: ignore[misc, valid-type]
        def __init__(self, reader: store.MapReader) -> None:
            if not commits:  # the first attempt: a concurrent update commits now
                commits.append(update(mdir, world.imgs_b[6:8], client=world.client,
                                      progress=quiet))
            super().__init__(reader)

    monkeypatch.setattr(lmod, "_MapPoints", CommitFirst)
    res = run(mdir, world.imgs_b[:2], mode="full")
    assert len(commits) == 1 and all(r.located for r in res.results)
    doc = json.loads(res.payload)
    after = json.loads(scene_bytes(store.MapReader(mdir)))
    assert after != before
    root = doc["openlabel"]
    for k in located_frames(doc):
        del root["frames"][k]
    for key in (f"{LOCATED_CS}0", f"{LOCATED_CS}1"):
        del root["coordinate_systems"][key]
        del root["streams"][key]
        root["coordinate_systems"]["map"]["children"].remove(key)
    from oh_my_slam.schema.openlabel import frame_intervals

    root["frame_intervals"] = frame_intervals([int(k) for k in root["frames"]])
    assert doc == after
    # located frames are keyed past the keyframes the update added
    assert min(int(k) for k in located_frames(json.loads(res.payload))) > max(
        int(k) for k in after["openlabel"]["frames"])


@needs_colmap
def test_a_map_that_keeps_changing_is_an_input_error(world, tmp_path: Path,  # type: ignore[no-untyped-def]
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    from oh_my_slam.mapping import locate as lmod

    mdir = tmp_path / "map"
    shutil.copytree(world.map, mdir)
    reader = open_map(mdir)
    calls = iter(range(100))
    monkeypatch.setattr(lmod, "map_identity", lambda root: (b"", bytes([next(calls)])))
    with pytest.raises(InputError, match="kept changing"):
        locate(reader, world.imgs_b[:1], progress=quiet)


# ------------------------------------------------------------------------------------------------
# a one-keyframe map: no sfm/, 2D-3D from the keyframe's stored depth


@needs_colmap
def test_one_keyframe_map(tmp_path: Path) -> None:
    client = FakeClient()
    room = mapping_room()
    key_pose = ring(1)[0]
    add_frames(client, room, [key_pose], tmp_path / "k", "k")
    mdir = tmp_path / "map"
    update(mdir, sorted((tmp_path / "k").iterdir()), client=client, progress=quiet)
    assert not (mdir / "sfm").exists()
    reader = store.MapReader(mdir)
    sim = similarity_by_poses([reader.frames[0].T_map_cam], [key_pose], with_scale=False)
    poses = ring(2, start=0.08, span=0.16)
    imgs = add_frames(client, room, poses, tmp_path / "q", "q")
    before = snapshot(mdir)
    res = run(mdir, imgs, mode="full")
    assert snapshot(mdir) == before
    assert all(r.located for r in res.results), [r.reason for r in res.results]
    assert validation_errors(json.loads(res.payload)) == []
    for r, truth in zip(res.results, poses, strict=True):
        W = transform_pose(sim, r.T_map_cam)  # type: ignore[arg-type]
        assert np.linalg.norm(W.t - truth.t) < 0.05 and rot_deg(W.R, truth.R) < 1.0, \
            (W.t - truth.t, rot_deg(W.R, truth.R))


# ------------------------------------------------------------------------------------------------
# an SfM map: model points first, the scaled depth for the rest


def _model_only(lmod):  # type: ignore[no-untyped-def]
    """``_MapPoints`` as before the depth fill: model points only for a keyframe the model
    holds."""
    class ModelOnly(lmod._MapPoints):  # type: ignore[misc, name-defined]
        def lookup_split(self, m):  # type: ignore[no-untyped-def]
            model = self._model_points(m.keyframe)
            if model is None:
                return super().lookup_split(m)
            xyz, has, _ = model
            ok = m.idx_k < len(has)
            idx = np.where(ok, m.idx_k, 0)
            return xyz[idx], ok & has[idx], np.zeros(len(ok), bool)
    return ModelOnly


@needs_colmap
def test_the_depth_fill_locates_no_worse_than_the_model_points(world,  # type: ignore[no-untyped-def]
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """On an SfM-posed ring map the union (model points, then depth) locates every held-out view
    the model points alone locate, as accurately: a pose the model points alone give with
    ``MODEL_FIRST_MIN_INLIERS`` inliers is kept as it is, the others stay within the accuracy
    target, and the median errors are no worse."""
    from oh_my_slam.mapping import locate as lmod

    union = run(world.map, world.imgs_b).results
    monkeypatch.setattr(lmod, "_MapPoints", _model_only(lmod))
    alone = run(world.map, world.imgs_b).results
    assert all(r.located for r in union), [r.reason for r in union]

    def err(r: object) -> tuple[float, float]:
        W = transform_pose(world.sim, r.T_map_cam)  # type: ignore[attr-defined]
        return float(np.linalg.norm(W.t - world.poses_b[r.index].t)), \
            rot_deg(W.R, world.poses_b[r.index].R)  # type: ignore[attr-defined]

    same = 0
    for u, a in zip(union, alone, strict=True):
        assert err(u)[0] < 0.03 and err(u)[1] < 0.5, (u.image.name, err(u))
        if a.located and a.inliers >= lmod.MODEL_FIRST_MIN_INLIERS:
            np.testing.assert_allclose(u.T_map_cam.matrix(), a.T_map_cam.matrix())  # type: ignore[union-attr]
            same += 1
    assert same > 0
    both = [(err(u), err(a)) for u, a in zip(union, alone, strict=True) if a.located]
    assert np.median([e[0][0] for e in both]) <= np.median([e[1][0] for e in both]) + 0.005
    assert np.median([e[0][1] for e in both]) <= np.median([e[1][1] for e in both]) + 0.05


@needs_colmap
def test_depth_fill_is_scaled_to_the_sfm_points(world, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """A keyframe's depth points take the median model/depth ratio of its keypoints with both; a
    keyframe whose depth is off by more than 2x gives none."""
    from oh_my_slam.mapping import locate as lmod

    mdir = tmp_path / "map"
    shutil.copytree(world.map, mdir)
    before = lmod._MapPoints(store.MapReader(mdir))
    fr0, fr1 = before.reader.frames[:2]
    s0 = before.depth_scale(fr0)
    assert s0 is not None and abs(s0 - 1.0) < 0.06  # the map's depth fits its SfM points

    def no_model_match(points, fr):  # type: ignore[no-untyped-def]
        model = points._model_points(f"{fr.name}.jpg")
        assert model is not None
        _, has, uv = model
        idx = np.flatnonzero(~has)[:200]
        assert len(idx) > 50
        return lmod._Match(f"{fr.name}.jpg", idx, idx, uv[idx], uv[idx])

    xyz0, ok0 = before.lookup(no_model_match(before, fr0))
    assert ok0.sum() > 50
    for fr, f in ((fr0, 0.8), (fr1, 0.24)):
        p = mdir / store.frame_file(fr.name, "depth.npy")
        np.save(p, (np.load(p).astype(np.float32) * f).astype(np.float16))
    after = lmod._MapPoints(store.MapReader(mdir))
    s = after.depth_scale(fr0)
    assert s is not None and abs(s - s0 / 0.8) < 0.01 * s0 / 0.8
    xyz, ok = after.lookup(no_model_match(after, fr0))
    np.testing.assert_array_equal(ok, ok0)
    np.testing.assert_allclose(xyz[ok], xyz0[ok0], atol=0.01)  # the same points as before
    assert after.depth_scale(fr1) is None
    m1 = no_model_match(after, fr1)
    assert not after.lookup(m1)[1].any()  # no depth points
    _, has1, uv1 = after._model_points(f"{fr1.name}.jpg")  # type: ignore[misc]
    idx = np.flatnonzero(has1)[:50]
    assert after.lookup(lmod._Match(m1.keyframe, idx, idx, uv1[idx], uv1[idx]))[1].all()


# ------------------------------------------------------------------------------------------------
# a rotation-dominant map (multi-view poses): 2D-3D from the keyframes' stored depth


@needs_colmap
def test_held_out_views_of_a_rotation_dominant_map(tmp_path: Path) -> None:
    """A head turning in place is posed by the multi-view fallback, whose ``sfm/model`` points are
    triangulated from near-zero baselines without bundle adjustment: held-out headings between the
    keyframes are located from the keyframes' stored depth (``model_points_trusted``), not from
    those few unreliable points."""
    from oh_my_slam.mapping import locate as lmod
    from tests.unit.test_mapping_e2e_rotation import turning, yaw

    client = FakeClient(mv_noise=(3.0, 0.15))
    room = mapping_room()
    first = turning(20, 0.0, 12.0)
    add_frames(client, room, first, tmp_path / "a", "a", depth_noise=0.03, seed=4)
    held = turning(19, 6.0, 12.0)  # half-way between consecutive keyframes
    imgs = add_frames(client, room, held, tmp_path / "q", "q", depth_noise=0.03, seed=6)
    mdir = tmp_path / "map"
    msgs: list[str] = []
    update(mdir, sorted((tmp_path / "a").glob("*.png")), client=client, progress=msgs.append)
    assert any("multi-view fallback (rotation-dominant" in m for m in msgs), msgs
    reader = store.MapReader(mdir)
    assert {fr.pose_source for fr in reader.frames} == {"multiview"}
    assert not any(lmod.model_points_trusted(fr) for fr in reader.frames)
    before = snapshot(mdir)
    res = run(mdir, imgs)
    assert snapshot(mdir) == before
    assert [r.located for r in res.results] == [True] * len(imgs), [r.reason for r in res.results]
    # truth: the map frame is the first keyframe's up to a rigid transform (metric depth)
    sim = similarity_by_poses([fr.T_map_cam for fr in reader.frames], first, with_scale=False)
    centre = np.mean([transform_pose(sim, fr.T_map_cam).t for fr in reader.frames], axis=0)
    y0 = yaw(reader.frames[0].T_map_cam)
    for r, truth in zip(res.results, held, strict=True):
        assert r.T_map_cam is not None
        W = transform_pose(sim, r.T_map_cam)
        dyaw = (yaw(r.T_map_cam) - y0 - (yaw(truth) - yaw(first[0])) + 180) % 360 - 180
        assert abs(dyaw) < 1.0 and rot_deg(W.R, truth.R) < 1.5, (r.image.name, dyaw)
        assert np.linalg.norm(W.t - centre) < 0.15, (r.image.name, W.t - centre)


# ------------------------------------------------------------------------------------------------
# offline rules (no COLMAP)


def test_inputs_map_folder_and_output_rules(tmp_path: Path) -> None:
    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    assert resolve_images([img]) == [img]
    with pytest.raises(UsageError, match="video"):
        resolve_images([img, video])
    with pytest.raises(InputError, match="not found"):
        resolve_images([tmp_path / "missing.jpg"])
    with pytest.raises(InputError, match="not an image"):
        resolve_images([tmp_path])  # a folder is not an image
    with pytest.raises(UsageError):
        resolve_images([])
    with pytest.raises(InputError, match="existing map"):
        open_map(tmp_path / "nothing")
    assert not (tmp_path / "nothing").exists()  # not created
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(InputError, match="existing map"):
        open_map(empty)
    assert list(empty.iterdir()) == []
    other = tmp_path / "other"
    other.mkdir()
    (other / "f.txt").write_text("x")
    with pytest.raises(NotAMapError):
        open_map(other)
    with pytest.raises(UsageError, match="inside the map"):
        check_output(other, other / "sub" / "r.json")
    check_output(other, tmp_path / "r.json")
    check_output(other, None)
    with pytest.raises(UsageError):
        locate(None, [img], mode="both")  # type: ignore[arg-type]


def test_visible_points_frustum_and_occlusion() -> None:
    K = Intrinsics(100.0, 100.0, 50.0, 50.0, 100, 100)
    T = Pose.identity()  # camera at the origin looking along +z
    side = np.linspace(-0.9, 0.9, 60)  # 1.5 px apart: a map cloud seen from nearby has gaps
    wall = np.array([[x, y, 2.0] for x in side for y in side])
    behind_wall = np.array([[0.1, 0.1, 4.0]])
    on_wall = np.array([[0.0, 0.0, 2.05]])  # within the z-buffer tolerance
    outside = np.array([[5.0, 0.0, 2.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.01]])
    xyz = np.vstack([wall, behind_wall, on_wall, outside])
    vis = visible_points(xyz, [(T, K)])
    assert vis[:len(wall)].all()
    assert not vis[len(wall)] and vis[len(wall) + 1] and not vis[-3:].any()
    # a second camera behind the wall's back sees the far point
    T2 = Pose(np.diag([-1.0, 1.0, -1.0]), np.array([0.1, 0.1, 6.0]))
    vis2 = visible_points(xyz, [(T, K), (T2, K)])
    assert vis2[len(wall)]
    assert not visible_points(xyz, []).any()


def _fake_reader(n: int, tmp_path: Path) -> SimpleNamespace:
    K = Intrinsics(300.0, 300.0, 200.0, 150.0, 400, 300)
    frames = [store.FrameRecord(i, f"f{i:06d}", f"frames/f{i:06d}.jpg", "", 1, 400, 300, K,
                                Pose.identity(), 400, 300) for i in range(n)]
    desc = {fr.name: np.eye(8)[i % 8] for i, fr in enumerate(frames)}
    return SimpleNamespace(frames=frames, descriptor=lambda fr: desc[fr.name])


def test_a_vanished_file_retries_only_when_the_map_changed(tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """``_locate``: a FileNotFoundError while the map changed (a commit moved the file) starts
    again and the next attempt's result is returned, its warnings and counts once; with the map
    unchanged the error is a real one and propagates."""
    from oh_my_slam.core import timing
    from oh_my_slam.mapping import locate as lmod

    reader = SimpleNamespace(root=tmp_path, frames=[], meta={})
    ok = [lmod.Located(Path("a.jpg"), 0, Pose.identity(), Intrinsics(1, 1, 0, 0, 2, 2)),
          lmod.Located(Path("b.jpg"), 1, reason="far away")]
    attempts: list[int] = []
    identity = iter([b"1", b"2", b"2", b"2"])

    def once(*args: object) -> tuple[bytes, list]:
        attempts.append(1)
        if len(attempts) == 1:
            raise FileNotFoundError("sfm/model/images.bin")
        return b"doc", ok

    monkeypatch.setattr(lmod, "_locate_once", once)
    monkeypatch.setattr(lmod, "_stale", lambda r: False)
    monkeypatch.setattr(lmod, "open_map", lambda root: reader)
    monkeypatch.setattr(lmod, "map_identity", lambda root: (next(identity), None))
    with timing.collect() as tm:
        out = lmod._locate(reader, [], "single", "json", CloudAttrs(), None, quiet)  # type: ignore[arg-type]
    assert out == (b"doc", ok) and len(attempts) == 2
    counts = tm.to_dict()["counts"]
    assert counts["attempts"] == 2 and counts["located"] == 1 and counts["images"] == 2

    attempts.clear()
    monkeypatch.setattr(lmod, "map_identity", lambda root: (b"same", None))
    with pytest.raises(FileNotFoundError):
        lmod._locate(reader, [], "single", "json", CloudAttrs(), None, quiet)  # type: ignore[arg-type]
    assert len(attempts) == 1

    # an attempt that located nothing fails only once its map state is confirmed
    monkeypatch.setattr(lmod, "_locate_once", lambda *a: (None, ok[1:]))
    with pytest.raises(InputError, match="none of the images"):
        lmod._locate(reader, [], "single", "json", CloudAttrs(), None, quiet)  # type: ignore[arg-type]


def test_pairs_exhaustive_for_small_maps_retrieval_for_large(tmp_path: Path) -> None:
    from oh_my_slam.mapping import locate as lmod
    from oh_my_slam.mapping.api import RETRIEVAL_TOP_K, UPDATE_EXHAUSTIVE_MAX

    q_path = tmp_path / "q.png"
    client = FakeClient()
    client.add(q_path, np.zeros((300, 400, 3), np.uint8),
               FakeFrame(np.ones((300, 400), np.float32), Intrinsics(300, 300, 200, 150, 400, 300),
                         np.array([0, -1.0, 0]), descriptor=np.eye(8)[3]))
    q = lmod._Query(q_path, 0, "locate_000000.jpg", (400, 300), None)

    def db(n: int) -> dict:
        d = {f"f{i:06d}.jpg": lmod._DbImage(i + 1, 1) for i in range(n)}
        d[q.name] = lmod._DbImage(n + 1, 2)
        return d

    small = _fake_reader(UPDATE_EXHAUSTIVE_MAX, tmp_path)
    pairs = lmod._pairs(small, [q], db(UPDATE_EXHAUSTIVE_MAX), None)  # no server needed
    assert len(pairs) == UPDATE_EXHAUSTIVE_MAX
    large = _fake_reader(UPDATE_EXHAUSTIVE_MAX + 1, tmp_path)
    n = UPDATE_EXHAUSTIVE_MAX + 1
    pairs = lmod._pairs(large, [q], db(n), client)
    assert len(pairs) == RETRIEVAL_TOP_K
    best = {i + 1 for i in range(n) if i % 8 == 3}  # the keyframes whose descriptor is the query's
    assert best <= {a for a, _ in pairs}
    assert client.calls["geometry"] == 1 and client.calls["gravity"] == 0
    with pytest.raises(ServerUnavailableError):  # no server in the test session: exit 3
        lmod._pairs(large, [q], db(n), None)


# ------------------------------------------------------------------------------------------------
# the epipolar gate in both regimes (analytic head, no COLMAP)


def _gate_world(steps: dict[str, np.ndarray], pose_source: str):  # type: ignore[no-untyped-def]
    """Keyframes of an analytic head (``tests.synth.turning``) at 15° steps, each moved by
    ``steps`` (none: turning in place), the query half-way between two of them; matches, the
    keyframes' true depth as ``_MapPoints`` would give them, and the 2D-3D correspondences of
    the query (``uv``, ``xyz``) with the inliers of its true pose."""
    from oh_my_slam.mapping import locate as lmod
    from tests.synth.turning import K, head_pose, turning_rig

    poses = {f"f{k:06d}": head_pose(15.0 * k, step=steps.get(f"f{k:06d}")) for k in range(7)}
    truth = head_pose(37.5)
    rig = turning_rig({**poses, "q": truth}, seed=7)
    frames = {f"{n}.jpg": store.FrameRecord(k, n, f"frames/{n}.jpg", "", 1, K.width, K.height, K,
                                            T, K.width, K.height, pose_source=pose_source)
              for k, (n, T) in enumerate(poses.items())}
    matches = []
    for p in rig.pairs:
        if "q" in (p.a, p.b):
            kf, uq, uk = (p.b, p.uv_a, p.uv_b) if p.a == "q" else (p.a, p.uv_b, p.uv_a)
            idx = np.arange(len(uq))
            matches.append(lmod._Match(f"{kf}.jpg", idx, idx, uq, uk))
    depth = {f"{n}.jpg": rig.depth[n] for n in poses}

    def depth_pts(fr: store.FrameRecord, uv: np.ndarray):  # type: ignore[no-untyped-def]
        d = depth[f"{fr.name}.jpg"]
        return lmod.depth_points(d, np.ones(d.shape, bool), fr, uv)

    points = SimpleNamespace(frames=frames, _depth_points=depth_pts)
    uv = np.vstack([m.uv_q for m in matches])
    xyz = np.vstack([depth_pts(frames[m.keyframe], m.uv_k)[0] for m in matches])
    w = SimpleNamespace(lmod=lmod, K=K, truth=truth, matches=matches, points=points, uv=uv,
                        xyz=xyz, inliers=lmod.count_inliers(truth, K, uv, xyz))
    assert w.inliers > 0.9 * len(uv)
    return w


def _gate(w, T: Pose, inliers: int | None = None):  # type: ignore[no-untyped-def]
    """``epipolar_gate`` of pose ``T``, solved from ``w``'s correspondences with ``inliers``
    (default: as many as ``T`` reprojects)."""
    n = w.lmod.count_inliers(T, w.K, w.uv, w.xyz) if inliers is None else inliers
    return w.lmod.epipolar_gate(T, w.K, w.matches, w.points, w.uv, w.xyz, n)


def _worst_shift(lmod, K, truth, matches, frames, size: float) -> Pose:  # type: ignore[no-untyped-def]
    """``truth`` moved by ``size`` along the axis that contradicts the matches most."""
    cands = [Pose(truth.R, truth.t + size * d) for d in np.vstack([np.eye(3), -np.eye(3)])]
    return max(cands, key=lambda T: lmod.match_residual_deg(T, K, matches, frames)[0])


def test_gate_refines_a_centre_error_seen_from_the_keyframes_spot() -> None:
    """Turning in place (multi-view keyframes): a centre 3 cm off — the depth's scale error —
    contradicts the matches by more than the limit; refined against the keyframe poses it is
    accepted, as accurate as the truth allows. A rotation 3° off is never accepted as such."""
    from scipy.spatial.transform import Rotation

    from tests.synth.turning import rot_err_deg

    w = _gate_world({}, "multiview")
    lmod, K, truth, matches, points = w.lmod, w.K, w.truth, w.matches, w.points
    T = _worst_shift(lmod, K, truth, matches, points.frames, 0.03)
    med, n = lmod.match_residual_deg(T, K, matches, points.frames)
    assert n >= lmod.EPIPOLAR_MIN_MATCHES and med > lmod.MAX_EPIPOLAR_DEG
    assert lmod.rotation_dominant(T, matches, points)
    T2, med2, ok = _gate(w, T)
    assert ok and med2 <= lmod.MAX_EPIPOLAR_DEG, med2
    assert rot_err_deg(T2, truth) < 0.15 and np.linalg.norm(T2.t - truth.t) < 0.03
    # the truth itself passes untouched
    assert _gate(w, truth)[0] is truth
    # a rotation error the limit rejects is not made acceptable by the refinement (a turn about
    # the vertical, along a horizontal baseline, barely moves the epipolar lines: the limit
    # itself cannot see it there)
    rejected = 0
    for axis in np.eye(3):
        wrong = Pose(Rotation.from_rotvec(np.radians(3.0) * axis).as_matrix() @ truth.R, truth.t)
        if lmod.match_residual_deg(wrong, K, matches, points.frames)[0] <= lmod.MAX_EPIPOLAR_DEG:
            continue
        rejected += 1
        T3, med3, ok3 = _gate(w, wrong, w.inliers)
        assert not ok3 or rot_err_deg(T3, truth) < 0.25, (axis, med3, rot_err_deg(T3, truth))
    assert rejected >= 2


def test_gate_rejects_a_centre_far_off_seen_from_the_keyframes_spot(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Turning in place: a centre 0.5 or 1 m off is not a depth scale error. Where the limit
    rejects it (along some directions the epipolar lines barely move: the limit itself cannot
    see it there, and only the 2D-3D fit that produced the pose can), the refinement may bring
    its epipolar distance under the limit, but the pose stays rejected by the bound on the
    centre's move; that bound lifted, a refined pose is accepted only where it went back to the
    truth, the others by the 2D-3D inliers they lose."""
    w = _gate_world({}, "multiview")
    lmod = w.lmod
    dirs = [np.subtract(d, 1.0) for d in np.ndindex(3, 3, 3) if d != (1, 1, 1)]  # 26 directions
    far = [Pose(w.truth.R, w.truth.t + size * d / np.linalg.norm(d))
           for size in (0.5, 1.0) for d in dirs]
    far = [T for T in far
           if lmod.match_residual_deg(T, w.K, w.matches, w.points.frames)[0] > lmod.MAX_EPIPOLAR_DEG]
    assert len(far) >= 10
    for T in far:
        T2, _, ok = _gate(w, T, w.inliers)
        assert not ok and T2 is T, T.t - w.truth.t
    # that bound lifted: a refined pose that went back to the truth is right; any other loses the
    # 2D-3D inliers of the pose's own fit
    from tests.synth.turning import rot_err_deg

    monkeypatch.setattr(lmod, "REFINE_MAX_MOVE_M", 10.0)
    lost = 0
    for T in far:
        T2, _, ok = _gate(w, T, w.inliers)
        if ok:
            assert np.linalg.norm(T2.t - w.truth.t) < 0.05 and rot_err_deg(T2, w.truth) < 0.5
        else:
            lost += 1
    assert lost >= 1
    assert _gate(w, w.truth, w.inliers)[2]


def test_gate_keeps_the_limit_for_well_baselined_keyframes() -> None:
    """SfM keyframes metres apart: a pose above the limit is rejected as it is, not refined."""
    rng = np.random.default_rng(3)
    steps = {f"f{k:06d}": np.array([*rng.uniform(-1.2, 1.2, 2), 0.0]) for k in range(7)}
    w = _gate_world(steps, "sfm-global")
    lmod, K, truth, matches, points = w.lmod, w.K, w.truth, w.matches, w.points
    T = _worst_shift(lmod, K, truth, matches, points.frames, 0.15)
    med, n = lmod.match_residual_deg(T, K, matches, points.frames)
    assert n >= lmod.EPIPOLAR_MIN_MATCHES and med > lmod.MAX_EPIPOLAR_DEG
    assert not lmod.rotation_dominant(T, matches, points)
    T2, med2, ok = _gate(w, T)
    assert not ok and T2 is T and med2 == med
    assert _gate(w, truth)[2]
