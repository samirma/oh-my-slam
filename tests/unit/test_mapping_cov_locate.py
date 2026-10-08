"""``mapper.sh locate`` (``mapping.locate``) at its edges: the query's camera, retrieval without
descriptors, the verified matches of the scratch database, the 2D-3D correspondences of keyframes
the model holds only partly, poses that cannot be solved or that the epipolar gate rejects, and a
map whose ``map.json`` vanished while it was read."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from oh_my_slam.core.errors import InputError
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.mapping import locate as lmod
from oh_my_slam.mapping import store
from oh_my_slam.mapping.sfm import shared_camera
from oh_my_slam.schema import openlabel as ol
from tests.synth.scene import look_at

K = Intrinsics(500.0, 500.0, 320.0, 240.0, 640, 480, "colmap")


def frame(i: int, size: tuple[int, int] = (640, 480), source: str = "sfm-global",
          pose: Pose | None = None) -> store.FrameRecord:
    return store.FrameRecord(i, f"f{i:06d}", f"frames/f{i:06d}.jpg", "", 1, *size, K,
                             pose or Pose.identity(), 64, 48, pose_source=source)


def test_a_query_with_exif_takes_a_map_camera_of_its_focal_prior() -> None:
    frames = [frame(0), frame(1), frame(2, (800, 600))]
    db = {0: 3, 1: 5, 2: 9}  # keyframe -> its camera in the scratch database

    def camera(fr: store.FrameRecord) -> int:
        return db[fr.index]

    prior, _ = shared_camera(frames, (640, 480), True, 510.0, camera)
    assert (prior.focal, prior.existing_id, prior.same_focal_ids) == (510.0, None, (5, 3))
    plain, latest = shared_camera(frames, (640, 480), False, None, camera)
    assert (plain.focal, plain.existing_id, latest) == (None, 5, frames[1])  # the latest one's
    mixed, _ = shared_camera(frames, (640, 480), None, 500.0, camera)
    assert (mixed.existing_id, mixed.same_focal_ids) == (None, ())  # with and without EXIF
    alone, none = shared_camera(frames, (320, 240), False)
    assert (alone.existing_id, none) == (None, None)  # no keyframe of that size


def test_retrieval_pairs_only_keyframes_with_a_descriptor(monkeypatch: pytest.MonkeyPatch
                                                          ) -> None:
    from oh_my_slam.core.constants import UPDATE_EXHAUSTIVE_MAX

    n = UPDATE_EXHAUSTIVE_MAX + 1
    frames = [frame(i) for i in range(n)]
    desc = {fr.name: (np.eye(8)[i % 8] if i % 3 else None) for i, fr in enumerate(frames)}
    reader = SimpleNamespace(frames=frames, descriptor=lambda fr: desc[fr.name])
    q = lmod._Query(Path("q.jpg"), 0, "locate_000000.jpg", (640, 480), None)
    db = {f"f{i:06d}.jpg": lmod._DbImage(i + 1, 1) for i in range(n)}
    db[q.name] = lmod._DbImage(n + 1, 2)
    monkeypatch.setattr(lmod, "query_descriptors",
                        lambda qs, client: np.eye(8, dtype=np.float32)[[2]])
    pairs = lmod._pairs(reader, [q], db, None)  # type: ignore[arg-type]
    assert pairs and all(a % 3 != 1 for a, _ in pairs)  # image id i + 1: keyframe i % 3 != 0
    assert {a for a, _ in pairs} >= {i + 1 for i in range(n) if i % 8 == 2 and i % 3}


def test_a_server_without_a_descriptor_is_an_input_error(monkeypatch: pytest.MonkeyPatch
                                                          ) -> None:
    from oh_my_slam.reconstruction import api as rapi

    monkeypatch.setattr(rapi, "reconstruct_image",
                        lambda *a, **k: SimpleNamespace(descriptor=None))
    q = lmod._Query(Path("q.jpg"), 0, "locate_000000.jpg", (640, 480), None)
    with pytest.raises(InputError, match="no retrieval descriptor"):
        lmod.query_descriptors([q], client=object())


def test_query_matches_keep_verified_pairs_with_keyframes_whichever_id_is_first(
        tmp_path: Path) -> None:
    import pycolmap

    path = tmp_path / "db.db"
    d = pycolmap.Database.open(str(path))
    try:
        cam = d.write_camera(pycolmap.Camera.create_from_model_id(
            0, pycolmap.CameraModelId.SIMPLE_PINHOLE, 500.0, 640, 480))
        ids = {n: d.write_image(pycolmap.Image(name=n, camera_id=cam))
               for n in ("f000000.jpg", "locate_000000.jpg", "f000001.jpg", "f000002.jpg")}
        for k, i in enumerate(ids.values()):
            d.write_keypoints(i, np.arange(80, dtype=np.float32).reshape(40, 2) + 1000 * k)

        def pair(a: str, b: str, inliers: int, config: int = 2) -> None:
            g = pycolmap.TwoViewGeometry()
            g.config = config
            g.inlier_matches = np.stack([np.arange(inliers), np.arange(inliers)[::-1]],
                                        1).astype(np.uint32)
            d.write_two_view_geometry(ids[a], ids[b], g)

        pair("f000000.jpg", "locate_000000.jpg", 20)  # the keyframe's id first
        pair("locate_000000.jpg", "f000001.jpg", 25)  # the query's id first
        pair("locate_000000.jpg", "f000002.jpg", 30, config=1)  # degenerate: not verified
        pair("f000000.jpg", "f000001.jpg", 40)  # two keyframes
    finally:
        d.close()
    q = lmod._Query(Path("q.jpg"), 0, "locate_000000.jpg", (640, 480), None)
    db = {n: lmod._DbImage(i, 1) for n, i in ids.items()}
    out = lmod._query_matches(path, [q], db)
    m0, m1 = out[q.name]
    assert (m0.keyframe, len(m0.idx_q), m1.keyframe, len(m1.idx_q)) == (
        "f000000.jpg", 20, "f000001.jpg", 25)
    # the keyframe's keypoint indices on its side, the query's on the other
    assert m0.idx_k[0] == 0 and m0.idx_q[0] == 19
    np.testing.assert_array_equal(m0.uv_q[0], [1000 + 38.0, 1000 + 39.0])
    assert m1.idx_q[0] == 0 and m1.idx_k[0] == 24
    np.testing.assert_array_equal(m1.uv_k[0], [2000 + 48.0, 2000 + 49.0])


def test_query_matches_of_a_lens_are_where_its_pinhole_sees_them(tmp_path: Path) -> None:
    """A map camera with distortion (``panorama.PairMatches``): the matches where its pinhole
    sees the keypoints, where the keyframe's depth (on the pinhole's grid) is read and unprojected
    along the pinhole's rays."""
    import pycolmap

    from oh_my_slam.mapping.panorama import pinhole_pixels

    path = tmp_path / "db.db"
    d = pycolmap.Database.open(str(path))
    try:
        lens = pycolmap.Camera.create_from_model_id(
            0, pycolmap.CameraModelId.SIMPLE_DIVISION, 500.0, 640, 480)
        lens.params = [500.0, 320.0, 240.0, -0.3]
        cam = d.write_camera(lens)
        ids = {n: d.write_image(pycolmap.Image(name=n, camera_id=cam))
               for n in ("f000000.jpg", "locate_000000.jpg")}
        kp = np.stack([np.linspace(10, 630, 30), np.linspace(10, 470, 30)], 1)
        for i in ids.values():
            d.write_keypoints(i, kp.astype(np.float32))
        g = pycolmap.TwoViewGeometry()
        g.config = 2
        g.inlier_matches = np.stack([np.arange(30)] * 2, 1).astype(np.uint32)
        d.write_two_view_geometry(ids["f000000.jpg"], ids["locate_000000.jpg"], g)
        (lens,) = d.read_all_cameras()
    finally:
        d.close()
    q = lmod._Query(Path("q.jpg"), 0, "locate_000000.jpg", (640, 480), None)
    db = {n: lmod._DbImage(i, cam) for n, i in ids.items()}
    (m,) = lmod._query_matches(path, [q], db)[q.name]
    pin = pinhole_pixels(lens, kp)
    np.testing.assert_allclose(m.uv_q, pin, rtol=1e-6)
    np.testing.assert_allclose(m.uv_k, pin, rtol=1e-6)
    K = Intrinsics(500.0, 500.0, 320.0, 240.0, 640, 480, "colmap", -0.3)
    fr = store.FrameRecord(0, "f000000", "frames/f000000.jpg", "", cam, 640, 480, K,
                           Pose.identity(), 64, 48)
    depth = np.full((48, 64), 2.0, np.float32)
    xyz, ok = lmod.depth_points(depth, np.ones((48, 64), bool), fr, m.uv_k)
    # the grid is the undistorted image's, which holds the whole lens: every keypoint is on it
    at = K.pinhole_pixels(kp)
    inside = (at >= 0).all(axis=1) & (at < [640, 480]).all(axis=1)
    assert inside.all() and np.array_equal(ok, inside)
    np.testing.assert_allclose(xyz[ok, :2], (pin[ok] - [320.0, 240.0]) / 500.0 * 2.0, atol=1e-6)


# --- 2D-3D correspondences ----------------------------------------------------------------------


def _model(points_of: dict[str, int], deregister: str | None = None) -> Any:
    """A pycolmap model of keyframes ``f<i>.jpg`` at the origin, each with that many keypoints
    on triangulated points 4 m ahead (point j is seen by every keyframe with more than j)."""
    import pycolmap

    rec = pycolmap.Reconstruction()
    rec.add_camera_with_trivial_rig(pycolmap.Camera.create_from_model_id(
        1, pycolmap.CameraModelId.SIMPLE_PINHOLE, 500.0, 640, 480))
    n = max(points_of.values())
    uv = np.column_stack([100.0 + 10 * np.arange(n), np.full(n, 240.0)])
    for k, name in enumerate(sorted(points_of)):
        rec.add_image_with_trivial_frame(
            pycolmap.Image(name=name, camera_id=1, image_id=k + 1,
                           points2D=[pycolmap.Point2D(p) for p in uv[:points_of[name]]]),
            pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), np.zeros(3)))
    for j in range(n):
        track = pycolmap.Track()
        for k, name in enumerate(sorted(points_of)):
            if j < points_of[name]:
                track.add_element(k + 1, j)
        rec.add_point3D(np.array([(uv[j, 0] - 320.0) / 500.0 * 4.0, 0.0, 4.0]), track,
                        np.zeros(3, np.uint8))
    if deregister:
        rec.deregister_frame(rec.find_image_with_name(deregister).frame_id)
    return rec


def _points(frames: list[store.FrameRecord], valid: bool = True, rec: Any = None
            ) -> lmod._MapPoints:
    pts = object.__new__(lmod._MapPoints)
    pts.reader = SimpleNamespace(depth=lambda fr: np.full((48, 64), 4.0, np.float32),
                                 valid=lambda fr: np.full((48, 64), valid))
    pts.frames = {f"{fr.name}.jpg": fr for fr in frames}
    pts.rec = rec
    pts._cache, pts._depth, pts._scale = {}, {}, {}
    return pts


def test_a_keyframe_the_model_holds_without_a_pose_has_no_model_points() -> None:
    # three keyframes see each point: f1 deregistered, f0 keeps its points
    pts = _points([frame(0), frame(1)], rec=_model(
        {"f000000.jpg": 30, "f000001.jpg": 30, "f000002.jpg": 30}, deregister="f000001.jpg"))
    assert pts._model_points("f000001.jpg") is None
    assert pts._model_points("f000009.jpg") is None
    xyz, has, uv = pts._model_points("f000000.jpg")  # type: ignore[misc]
    assert has.sum() == 30 and np.allclose(xyz[:, 2], 4.0)


def test_depth_of_a_keyframe_without_valid_depth_at_its_points_stays_unscaled(
        monkeypatch: pytest.MonkeyPatch) -> None:
    said: list[str] = []
    monkeypatch.setattr(lmod.log, "info", lambda msg, *a: said.append(msg % a))
    pts = _points([frame(0), frame(1)], valid=False,
                  rec=_model({"f000000.jpg": 25, "f000001.jpg": 25}))
    assert pts.depth_scale(frame(0)) == 1.0
    assert said and "too few keypoints with both an SfM point and depth" in said[0]


def test_matches_with_a_keyframe_the_map_lacks_give_no_points() -> None:
    pts = _points([frame(0)])
    m = lmod._Match("f000007.jpg", np.arange(3), np.arange(3), np.zeros((3, 2)), np.zeros((3, 2)))
    xyz, model, depth = pts.lookup_split(m)
    assert xyz.shape == (3, 3) and not model.any() and not depth.any()
    assert lmod.match_residual_deg(Pose.identity(), K, [m], pts.frames) == (0.0, 0)


def test_a_pose_needs_four_correspondences_that_are_not_degenerate() -> None:
    uv = np.random.default_rng(0).uniform(0, 600, (20, 2))
    assert lmod.solve_pose(uv[:3], np.ones((3, 3)), K, False) is None
    assert lmod.solve_pose(uv, np.tile([[0.0, 0.0, 5.0]], (20, 1)), K, False) is None


# --- locating one image -------------------------------------------------------------------------


def _scene(n: int, seed: int) -> tuple[Pose, np.ndarray, np.ndarray]:
    """A camera and ``n`` points it sees, with their pixels."""
    rng = np.random.default_rng(seed)
    T = look_at(np.array([0.5, -0.3, 1.4]), np.array([0.0, 4.0, 1.0]))
    cam = np.column_stack([rng.uniform(-1.5, 1.5, n), rng.uniform(-1.0, 1.0, n),
                           rng.uniform(3.0, 6.0, n)])
    uv = np.column_stack([K.fx * cam[:, 0] / cam[:, 2] + K.cx, K.fy * cam[:, 1] / cam[:, 2] + K.cy])
    return T, uv, T.apply(cam)


def _query() -> lmod._Query:
    return lmod._Query(Path("q.jpg"), 3, "locate_000003.jpg", (640, 480), None)


def test_a_model_pose_the_gate_rejects_falls_back_to_the_depth_points(
        monkeypatch: pytest.MonkeyPatch) -> None:
    T, uv, xyz = _scene(120, 1)
    asked: list[bool] = []

    def corr(m: Any, p: Any, depth: bool = True) -> tuple[np.ndarray, np.ndarray]:
        asked.append(depth)  # the model points alone first, then with the depth points
        return (uv, xyz) if depth else (uv[:60], xyz[:60])

    monkeypatch.setattr(lmod, "correspondences", corr)
    verdicts = iter([False, True])
    monkeypatch.setattr(lmod, "epipolar_gate",
                        lambda T, K, m, p, uv, xyz, n, name="": (T, 0.1, next(verdicts)))
    out = lmod._locate_one(_query(), K, False, [], None)  # type: ignore[arg-type]
    assert asked == [False, True] and out.located and out.inliers == 120
    np.testing.assert_allclose(out.T_map_cam.t, T.t, atol=1e-6)  # type: ignore[union-attr]
    assert out.key == "located_3"


def test_correspondences_that_agree_on_no_pose_leave_the_image_unlocated(
        monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(2)
    uv, xyz = rng.uniform(0, 600, (20, 2)), rng.uniform(-1, 1, (20, 3)) + [0.0, 0.0, 5.0]
    monkeypatch.setattr(lmod, "correspondences", lambda m, p, depth=True: (uv, xyz))
    out = lmod._locate_one(_query(), K, False, [], None)  # type: ignore[arg-type]
    assert not out.located
    assert out.reason.startswith("no consistent camera pose (") and "of 20 matches agree" \
        in out.reason


def test_a_pose_its_matches_contradict_leaves_the_image_unlocated(
        monkeypatch: pytest.MonkeyPatch) -> None:
    T, uv, xyz = _scene(40, 3)
    monkeypatch.setattr(lmod, "correspondences", lambda m, p, depth=True: (uv, xyz))
    monkeypatch.setattr(lmod, "epipolar_gate",
                        lambda T, K, m, p, uv, xyz, n, name="": (T, 0.73, False))
    out = lmod._locate_one(_query(), K, False, [], None)  # type: ignore[arg-type]
    assert not out.located
    assert out.reason == ("its pose contradicts its matches with the map's keyframes (median "
                          "epipolar distance 0.73° > 0.25°)")


def test_the_viewpoint_of_matches_without_usable_keyframe_depth_is_unknown() -> None:
    fr = frame(0)
    m_unknown = lmod._Match("f000009.jpg", np.arange(2), np.arange(2), np.zeros((2, 2)),
                            np.zeros((2, 2)))
    m_blind = lmod._Match("f000000.jpg", np.arange(2), np.arange(2), np.zeros((2, 2)),
                          np.full((2, 2), 100.0))
    points = SimpleNamespace(frames={"f000000.jpg": fr},
                             _depth_points=lambda f, uv: (
                                 np.zeros((len(uv), 3)), np.zeros(len(uv), bool)))
    assert lmod._viewpoint(Pose.identity(), [m_unknown, m_blind], points) == (False, np.inf)  # type: ignore[arg-type]


def test_a_reader_whose_map_json_vanished_is_stale(tmp_path: Path) -> None:
    root = tmp_path / "m"
    with store.MapTransaction(root) as tx:
        tx.commit({"update_count": 1})
    reader = store.MapReader(root)
    assert not lmod._stale(reader)
    (root / store.MAP_JSON).unlink()
    assert lmod._stale(reader)


def test_a_camera_located_with_the_map_s_lens_keeps_its_distortion() -> None:
    """``solve_pose``: the matches are where the lens's pinhole sees them, so a camera that
    shares the map's (held) intrinsics is that lens; one whose focal length is estimated is a
    pinhole."""
    from oh_my_slam.core.geometry import project

    lens = Intrinsics(500.0, 500.0, 320.0, 240.0, 640, 480, "colmap", -0.3)
    rng = np.random.default_rng(1)
    xyz = np.column_stack([rng.uniform(-2, 2, 60), rng.uniform(-1.5, 1.5, 60),
                           rng.uniform(3, 6, 60)])
    uv, _ = project(xyz, lens.K())
    held = lmod.solve_pose(uv, xyz, lens, False)
    assert held is not None and held[1].k == lens.k and held[1].fx == lens.fx
    free = lmod.solve_pose(uv, xyz, lens, True)
    assert free is not None and free[1].k == 0.0
    stream = ol.camera_stream(held[1])["stream_properties"]
    assert stream["intrinsics_custom"]["k"] == lens.k
