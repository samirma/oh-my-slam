"""``mapper.sh locate`` (spec 2.3): real COLMAP on rendered rooms, fake inference.

A map of ring a (14 views); held-out views of ring b are located and checked against the truth;
an image of another room is reported as unlocalisable; the map folder never changes."""

from __future__ import annotations

import hashlib
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
    check_output,
    locate,
    open_map,
    resolve_images,
    visible_points,
)
from oh_my_slam.schema.validate import validation_errors
from oh_my_slam.segmentation.cloud import MAP_FRAME
from tests.fakes.client import FakeClient, FakeFrame
from tests.synth.mapping import add_frames, mapping_room, ring
from tests.synth.scene import Room

REPO = Path(__file__).resolve().parents[2]
needs_colmap = pytest.mark.skipif(shutil.which("colmap") is None, reason="needs Homebrew colmap")


def quiet(msg: str) -> None:
    pass


def snapshot(root: Path) -> dict[str, tuple[bool, int, int, str]]:
    """Every entry under ``root``, hidden ones included (``.lock``, ``.staging``): kind, size,
    mtime and content hash — ``store.full_tree_hash`` skips hidden entries."""
    out = {}
    for p in sorted(root.rglob("*")):
        st = p.stat()
        digest = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else ""
        out[str(p.relative_to(root))] = (p.is_dir(), st.st_size, st.st_mtime_ns, digest)
    return out


def rot_deg(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1.0, 1.0))))


def located_frames(doc: dict) -> dict[str, dict]:
    return {k: f["frame_properties"] for k, f in doc["openlabel"]["frames"].items()
            if f["frame_properties"].get("located")}


def pose_of(props: dict, key: str) -> Pose:
    from oh_my_slam.core.geometry import quat_to_rot

    tr = props["transforms"][f"{key}_to_map"]["transform_src_to_dst"]
    return Pose(quat_to_rot(np.array(tr["quaternion"])), np.array(tr["translation"]))


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
    res = locate(world.map, world.imgs_b, progress=quiet)  # defaults: json, single
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
    (r,) = locate(world.map, [img], progress=quiet).results
    assert r.located, r.reason
    assert r.K is not None and abs(r.K.fx - 360.0) < 0.03 * 360.0 and r.K.source == "colmap"
    W = transform_pose(world.sim, r.T_map_cam)  # type: ignore[arg-type]
    assert np.linalg.norm(W.t - truth.t) < 0.05 and rot_deg(W.R, truth.R) < 1.0, \
        (W.t - truth.t, rot_deg(W.R, truth.R))


@needs_colmap
def test_full_scope_is_the_update_document_plus_the_located_cameras(world) -> None:  # type: ignore[no-untyped-def]
    imgs = [world.imgs_b[0], world.other, world.imgs_b[5]]
    res = locate(world.map, imgs, mode="full", progress=quiet)
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
    imgs = world.imgs_b[:2]
    js = locate(world.map, imgs, progress=quiet)
    single = locate(world.map, imgs, fmt="ply", progress=quiet)
    full = locate(world.map, imgs, fmt="ply", mode="full", progress=quiet)
    attrs = CloudAttrs()
    for res in (single, full):
        comments = parse_header(res.payload).comments
        assert comments[:2] == [MAP_FRAME, f"attributes {attrs.describe(CloudScope.MAP)}"]
        assert len(comments) == 2 + len(imgs)  # one line per input image
        for k, line in enumerate(comments[2:]):
            key, payload = line.split(" ", 1)
            d = json.loads(payload)
            assert key == f"{LOCATED_CS}{k}" and d["image"] == str(imgs[k]) and d["located"]
            r = js.results[k]
            np.testing.assert_allclose(d["T_map_cam"]["translation"], r.T_map_cam.t, atol=1e-5)  # type: ignore[union-attr]
            assert d["K"]["width"] == 400 and d["K"]["height"] == 300
    whole = map_cloud(store.MapReader(world.map))
    c_full, c_single = parse_ply(full.payload), parse_ply(single.payload)
    assert len(c_full) == len(whole)
    np.testing.assert_allclose(c_full.xyz, whole.xyz, atol=1e-6)
    rows = {tuple(p) for p in np.round(c_full.xyz, 5)}
    assert 0 < len(c_single) < len(c_full)
    assert all(tuple(p) in rows for p in np.round(c_single.xyz, 5))
    # -p applies to the map-scope attributes
    seg = parse_cloud_attrs("color=segment,label=on,voxel=0.05", CloudScope.MAP)
    voxel = parse_ply(locate(world.map, imgs, fmt="ply", attrs=seg, progress=quiet).payload)
    assert voxel.label is not None and len(voxel) < len(c_single)


@needs_colmap
def test_only_unlocalisable_images_are_an_input_error(world) -> None:  # type: ignore[no-untyped-def]
    before = snapshot(world.map)
    with pytest.raises(InputError, match="none of the images"):
        locate(world.map, [world.other], progress=quiet)
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
    res = locate(mdir, imgs, mode="full", progress=quiet)
    assert snapshot(mdir) == before
    assert all(r.located for r in res.results), [r.reason for r in res.results]
    assert validation_errors(json.loads(res.payload)) == []
    for r, truth in zip(res.results, poses, strict=True):
        W = transform_pose(sim, r.T_map_cam)  # type: ignore[arg-type]
        assert np.linalg.norm(W.t - truth.t) < 0.05 and rot_deg(W.R, truth.R) < 1.0, \
            (W.t - truth.t, rot_deg(W.R, truth.R))


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
        locate(other, [img], mode="both")


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
