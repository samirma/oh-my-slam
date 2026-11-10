"""Locate (``mapping.locate``): real COLMAP on rendered rooms, fake inference.

A map of ring a (14 views); held-out views of ring b are located and checked against the truth;
an image of another room is reported as unlocalisable; the map folder never changes."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from oh_my_slam.core.errors import InputError, NotAMapError, ServerUnavailableError, UsageError
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.mapping import store
from oh_my_slam.mapping.api import update
from oh_my_slam.mapping.frame import similarity_by_poses, transform_pose
from oh_my_slam.mapping.locate import LocateResult, locate, open_map, resolve_images
from tests.fakes.client import FakeClient, FakeFrame
from tests.mapsnap import full_tree_hash, snapshot, with_committed_overlay
from tests.synth.mapping import add_frames, mapping_room, ring
from tests.synth.scene import Room

REPO = Path(__file__).resolve().parents[2]
needs_colmap = pytest.mark.skipif(shutil.which("colmap") is None, reason="needs Homebrew colmap")


def quiet(msg: str) -> None:
    pass


def rot_deg(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1.0, 1.0))))


def run(mdir: Path, images: list[Path]) -> LocateResult:
    """``locate`` on resolved images in an opened map."""
    return locate(open_map(mdir), resolve_images(images), progress=quiet)


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
    update(mdir, imgs_a, client=client, progress=quiet)
    reader = store.MapReader(mdir)
    truth = {f"f{i:06d}": p for i, p in enumerate(poses_a)}
    sim = similarity_by_poses([r.T_map_cam for r in reader.frames],
                              [truth[r.name] for r in reader.frames])  # map -> world
    return SimpleNamespace(base=base, client=client, map=mdir, imgs_b=imgs_b, poses_b=poses_b, other=other, sim=sim)


# ------------------------------------------------------------------------------------------------
# poses (COLMAP)


@needs_colmap
def test_located_poses_match_the_truth_and_the_map_is_untouched(world) -> None:  # type: ignore[no-untyped-def]
    before, hashed = snapshot(world.map), full_tree_hash(world.map)
    res = run(world.map, world.imgs_b)
    assert snapshot(world.map) == before and full_tree_hash(world.map) == hashed
    assert not (world.map / store.STAGING).exists()
    assert [r.located for r in res.results] == [True] * len(world.imgs_b), \
        [r.reason for r in res.results]
    for k, (img, truth) in enumerate(zip(world.imgs_b, world.poses_b, strict=True)):
        r = res.results[k]
        assert r.image == img and r.index == k and r.inliers > 0
        W = transform_pose(world.sim, r.T_map_cam)  # type: ignore[arg-type]
        assert np.linalg.norm(W.t - truth.t) < 0.03 and rot_deg(W.R, truth.R) < 0.5, \
            (img.name, W.t - truth.t, rot_deg(W.R, truth.R))
        assert r.K is not None and abs(r.K.fx - 300.0) < 0.03 * 300.0  # the map camera's refined focal (true 300 px)
    assert {"setup", "features_matching", "pose"} <= set(res.timings["stages_s"])
    assert res.timings["counts"]["located"] == len(world.imgs_b)


@needs_colmap
def test_locate_leaves_a_committed_staging_overlay_unchanged(  # type: ignore[no-untyped-def]
        world, tmp_path: Path) -> None:
    """Spec §2.3, "nothing in the map folder (including ``.staging/``) changes": a map whose last
    update was killed after its commit point is read through the overlay; ``locate`` neither
    rolls the committed update forward nor discards it."""
    mdir = with_committed_overlay(world.map, tmp_path / "overlaid")
    before = snapshot(mdir)
    res = run(mdir, world.imgs_b[:2])
    assert all(r.located for r in res.results), [r.reason for r in res.results]
    assert snapshot(mdir) == before
    assert (mdir / store.STAGING / store.COMMIT).exists()


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
def test_an_image_of_another_room_is_not_located(world) -> None:  # type: ignore[no-untyped-def]
    """An image the map cannot place is a result with a reason, not an error, whether the other
    images are located or not; the map is untouched."""
    before = snapshot(world.map)
    imgs = [world.imgs_b[0], world.other, world.imgs_b[5]]
    res = run(world.map, imgs)
    assert [r.located for r in res.results] == [True, False, True]
    assert "not enough overlap" in res.results[1].reason and res.results[1].T_map_cam is None
    (alone,) = run(world.map, [world.other]).results
    assert not alone.located and alone.reason
    assert snapshot(world.map) == before


@needs_colmap
def test_an_update_committing_during_locate_gives_a_consistent_result(world, tmp_path: Path,  # type: ignore[no-untyped-def]
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """An update commits between feature matching and pose: locate starts again on the new map,
    and its result comes from the new map, never a mix."""
    from oh_my_slam.mapping import locate as lmod

    mdir = tmp_path / "map"
    shutil.copytree(world.map, mdir)
    real = lmod._MapPoints
    commits = []

    class CommitFirst(real):  # type: ignore[misc, valid-type]
        def __init__(self, reader: store.MapReader) -> None:
            if not commits:  # the first attempt: a concurrent update commits now
                commits.append(update(mdir, world.imgs_b[6:8], client=world.client,
                                      progress=quiet))
            super().__init__(reader)

    monkeypatch.setattr(lmod, "_MapPoints", CommitFirst)
    res = run(mdir, world.imgs_b[:2])
    assert len(commits) == 1 and all(r.located for r in res.results)
    assert res.timings["counts"]["attempts"] == 2
    assert res.timings["counts"]["map_frames"] == len(store.MapReader(mdir).frames) > len(
        store.MapReader(world.map).frames)


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
    res = run(mdir, imgs)
    assert snapshot(mdir) == before
    assert all(r.located for r in res.results), [r.reason for r in res.results]
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


def test_inputs_and_map_folder_rules(tmp_path: Path) -> None:
    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    assert resolve_images([img]) == [img]
    hidden = tmp_path / ".hidden.jpg"  # named explicitly: hidden files are no exception
    hidden.write_bytes(b"x")
    assert resolve_images([hidden, img]) == [hidden, img]
    with pytest.raises(InputError, match="not an image"):
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
    images = [Path("a.jpg"), Path("b.jpg")]
    ok = [lmod.Located(Path("a.jpg"), 0, Pose.identity(), Intrinsics(1, 1, 0, 0, 2, 2)),
          lmod.Located(Path("b.jpg"), 1, reason="far away")]
    attempts: list[int] = []
    identity = iter([b"1", b"2", b"2", b"2"])

    def once(*args: object) -> list:
        attempts.append(1)
        if len(attempts) == 1:
            raise FileNotFoundError("sfm/model/images.bin")
        return ok

    monkeypatch.setattr(lmod, "_register", once)
    monkeypatch.setattr(lmod, "_stale", lambda r: False)
    monkeypatch.setattr(lmod, "open_map", lambda root: reader)
    monkeypatch.setattr(lmod, "map_identity", lambda root: (next(identity), None))
    with timing.collect() as tm:
        out = lmod._locate(reader, images, None, quiet)  # type: ignore[arg-type]
    assert out == ok and len(attempts) == 2
    counts = tm.to_dict()["counts"]
    assert counts["attempts"] == 2 and counts["located"] == 1 and counts["images"] == 2

    attempts.clear()
    monkeypatch.setattr(lmod, "map_identity", lambda root: (b"same", None))
    with pytest.raises(FileNotFoundError):
        lmod._locate(reader, images, None, quiet)  # type: ignore[arg-type]
    assert len(attempts) == 1

    # an attempt that located nothing is a result too
    monkeypatch.setattr(lmod, "_register", lambda *a: ok[1:])
    assert lmod._locate(reader, images, None, quiet) == ok[1:]  # type: ignore[arg-type]


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
    assert lmod._viewpoint(T, matches, points)[0]
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
    assert not lmod._viewpoint(T, matches, points)[0]
    T2, med2, ok = _gate(w, T)
    assert not ok and T2 is T and med2 == med
    assert _gate(w, truth)[2]


# ------------------------------------------------------------------------------------------------
# a query placed by its rotation alone, from the keyframes' spot (analytic head, no COLMAP)


def _turn_world(steps: dict[str, np.ndarray], pose_source: str, keep: int | None = None,
                outliers: int = 0):  # type: ignore[no-untyped-def]
    """``_gate_world``'s head with ``keep`` matches per keyframe (evenly), ``outliers`` of them
    (each keyframe's last) moved to random pixels, and keyframes whose depth is valid everywhere
    (``median_depth``)."""
    w = _gate_world(steps, pose_source)
    rng = np.random.default_rng(11)
    out = []
    for m in w.matches:
        sel = np.linspace(0, len(m.uv_q) - 1, keep or len(m.uv_q)).astype(int)
        uq = m.uv_q[sel].copy()
        if outliers:
            uq[-outliers:] = rng.uniform([0, 0], [w.K.width, w.K.height], (outliers, 2))
        out.append(w.lmod._Match(m.keyframe, m.idx_q[sel], m.idx_k[sel], uq, m.uv_k[sel]))
    w.matches = out
    w.points.median_depth = lambda fr: 2.0
    w.query = w.lmod._Query(Path("/q/turned.jpg"), 0, "locate_000000.jpg", (w.K.width, w.K.height),
                            None)
    return w


def test_a_query_without_depth_is_turned_from_the_keyframes_spot() -> None:
    """Keyframes of a head turning in place (multi-view): a query whose matches have no depth is
    placed by its rotation alone, at the keyframes' centre — within half a degree of the truth
    (its camera is 5 cm in front of the pivot: the near matches' parallax is no rotation's), and
    despite a share of outliers."""
    from tests.synth.turning import rot_err_deg

    w = _turn_world({}, "multiview", keep=12, outliers=2)
    lmod = w.lmod
    centre = lmod.one_spot(w.matches, w.points)
    assert centre is not None
    r = lmod._located_by_turn(w.query, w.K, w.matches, w.points)
    assert r is not None and r.located and r.K is w.K
    assert rot_err_deg(r.T_map_cam, w.truth) < 0.5
    assert np.allclose(r.T_map_cam.t, centre)
    total = sum(len(m.idx_q) for m in w.matches)
    assert lmod.MIN_INLIERS <= r.inliers <= total - 2 * len(w.matches)


def test_a_turn_needs_keyframes_from_one_spot_and_agreeing_matches() -> None:
    """No turn for keyframes metres apart (SfM poses of a walk), for too few matches, or where
    most of the matches disagree with the rotation."""
    rng = np.random.default_rng(3)
    steps = {f"f{k:06d}": np.array([*rng.uniform(-1.2, 1.2, 2), 0.0]) for k in range(7)}
    w = _turn_world(steps, "sfm-global")
    lmod = w.lmod
    assert lmod.one_spot(w.matches, w.points) is None
    assert lmod._located_by_turn(w.query, w.K, w.matches, w.points) is None
    # SfM keyframes within a fraction of the scene depth of each other: one spot after all
    near = {n: 0.002 * s for n, s in steps.items()}
    assert lmod.one_spot(_turn_world(near, "sfm-global").matches, w.points) is None  # their poses
    w = _turn_world(near, "sfm-global")
    assert lmod.one_spot(w.matches, w.points) is not None
    few = _turn_world({}, "multiview", keep=2)
    assert sum(len(m.idx_q) for m in few.matches) < lmod.MIN_INLIERS
    assert lmod._located_by_turn(few.query, few.K, few.matches, few.points) is None
    noisy = _turn_world({}, "multiview", keep=8, outliers=7)
    assert lmod._located_by_turn(noisy.query, noisy.K, noisy.matches, noisy.points) is None
    # keyframes without any depth say nothing about where they were taken from
    noisy.points.median_depth = lambda fr: None
    assert lmod.one_spot(noisy.matches, noisy.points) is None


def test_a_turn_is_tried_only_after_the_points_and_for_a_map_camera(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """``_locate_one``: the 2D-3D pose first; the turn only when it fails and the query shares a
    map camera; the points' reason stays when the turn fails too."""
    w = _turn_world({}, "multiview", keep=6)
    lmod = w.lmod
    seen: list[int] = []

    def no_points(q, K, refine, matches, points):  # type: ignore[no-untyped-def]
        seen.append(len(matches))
        return lmod.Located(q.image, q.index, reason="too few")

    monkeypatch.setattr(lmod, "_located_by_points", no_points)
    assert lmod._locate_one(w.query, w.K, False, w.matches, w.points).located
    assert seen == [len(w.matches)]
    assert lmod._locate_one(w.query, w.K, True, w.matches, w.points).reason == "too few"
    monkeypatch.setattr(lmod, "_located_by_turn", lambda *a: None)
    assert lmod._locate_one(w.query, w.K, False, w.matches, w.points).reason == "too few"
    found = lmod.Located(w.query.image, 0, w.truth, w.K, 40)
    monkeypatch.setattr(lmod, "_located_by_points", lambda *a: found)
    assert lmod._locate_one(w.query, w.K, False, w.matches, w.points) is found


def test_turn_ransac_with_a_single_agreeing_pair() -> None:
    """Fewer than two inliers: the best two-match rotation and its inliers as they are."""
    from oh_my_slam.mapping import locate as lmod

    a = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    b = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    R, inl = lmod.turn_ransac(a, b)
    assert inl.sum() == 2 and np.allclose(R @ a.T, b.T)  # turned onto each other exactly
    # rays 90° apart onto rays 10° apart: no rotation keeps both (the best fit splits the error)
    c = np.array([[0.0, 0.0, 1.0], [np.sin(np.radians(10.0)), 0.0, np.cos(np.radians(10.0))]])
    R, inl = lmod.turn_ransac(a, c)
    assert inl.sum() < 2 and R.shape == (3, 3)


def test_a_keyframe_s_median_depth_skips_invalid_cells() -> None:
    from oh_my_slam.mapping import locate as lmod

    pts = lmod._MapPoints.__new__(lmod._MapPoints)
    fr = SimpleNamespace(name="f000001")
    d = np.array([[1.0, 2.0], [3.0, 0.0]], np.float32)
    pts._depth = {"f000001": (d, np.array([[True, True], [False, True]]))}
    pts._depth_points = lambda fr, uv: None  # type: ignore[method-assign]
    assert pts.median_depth(fr) == 1.5  # type: ignore[arg-type]
    pts._depth = {"f000001": (d, np.zeros((2, 2), bool))}
    assert pts.median_depth(fr) is None  # type: ignore[arg-type]
