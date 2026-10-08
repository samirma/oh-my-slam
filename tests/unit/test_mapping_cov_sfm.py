"""Structure from motion (``mapping.sfm``) and the multi-view pose refinement
(``mapping.panorama``) at their edges: missing or mismatched COLMAP, failing runs, the COLMAP
database rules (cameras, pairs, components, LightGlue restores) and the reconstruction helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.mapping import panorama
from oh_my_slam.mapping import sfm as sfm_mod
from oh_my_slam.mapping.sfm import CameraPrior, Sfm, SfmError, SfmModel
from tests.synth.turning import head_pose, turning_rig

# --- COLMAP binary ------------------------------------------------------------------------------


def test_missing_colmap_names_the_install_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sfm_mod.shutil, "which", lambda name: None)
    with pytest.raises(SfmError, match="brew install colmap"):
        sfm_mod.colmap_bin()


@pytest.mark.parametrize("out", ["COLMAP 3.11.1 -- Structure-from-Motion", "something else"])
def test_colmap_cli_and_pycolmap_must_both_be_4_2(monkeypatch: pytest.MonkeyPatch,
                                                  out: str) -> None:
    monkeypatch.setattr(sfm_mod, "colmap_bin", lambda: "colmap")
    monkeypatch.setattr(sfm_mod.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=out))
    with pytest.raises(SfmError, match=r"must both be 4\.2\.x .* brew upgrade colmap"):
        sfm_mod.check_versions()


def _fake_colmap(tmp_path: Path, lines: list[str], code: int) -> str:
    script = tmp_path / "colmap"
    script.write_text("#!/bin/sh\n" + "".join(f"echo '{ln}'\n" for ln in lines) + f"exit {code}\n")
    script.chmod(0o755)
    return str(script)


def test_a_failing_colmap_run_reports_its_fatal_lines(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    lines = ["I20261006 reading", "F20261006 database.cc:12] Check failed: x", "I20261006 done"]
    monkeypatch.setattr(sfm_mod, "colmap_bin", lambda: _fake_colmap(tmp_path, lines, 3))
    with pytest.raises(SfmError) as err:
        sfm_mod._run(["feature_extractor"], tmp_path / "colmap.log")
    assert str(err.value) == ("colmap feature_extractor failed (3): "
                              "F20261006 database.cc:12] Check failed: x")


def test_a_failing_colmap_run_without_fatal_lines_reports_its_last_lines(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [f"I line {k}" for k in range(9)]
    monkeypatch.setattr(sfm_mod, "colmap_bin", lambda: _fake_colmap(tmp_path, lines, 1))
    with pytest.raises(SfmError) as err:
        sfm_mod._run(["matches_importer"], tmp_path / "colmap.log")
    assert str(err.value) == "colmap matches_importer failed (1): " + " | ".join(lines[-6:])


# --- pure helpers -------------------------------------------------------------------------------


def test_weak_links_ignore_pairs_of_keyframes_outside_the_capture_order() -> None:
    order = ["a", "b", "c"]
    verified = {frozenset(("a", "b")), frozenset(("b", "x"))}  # x: not a keyframe of the walk
    listed = verified | {frozenset(("b", "c")), frozenset(("c", "y")), frozenset(("c",))}
    cuts, todo = sfm_mod.weak_link_pairs(order, listed, verified, window=2, min_pairs=1)
    assert cuts == ["b"] and todo == {frozenset(("b", "c"))}


def test_epipolar_distance_of_two_views_at_one_centre_is_zero() -> None:
    K = Intrinsics(100.0, 100.0, 50.0, 50.0, 100, 100)
    uv = np.array([[10.0, 20.0], [70.0, 40.0]])
    np.testing.assert_array_equal(sfm_mod.epipolar_deg(Pose.identity(), K, Pose.identity(), K,
                                                       uv, uv + 5.0), [0.0, 0.0])


class _PoseModel:
    def __init__(self, poses: dict[str, Pose]) -> None:
        self.poses = poses

    @property
    def registered(self) -> list[str]:
        return sorted(self.poses)

    def pose(self, name: str) -> Pose:
        return self.poses[name]


def test_an_extension_sharing_fewer_than_two_frames_is_not_mapped_back() -> None:
    base = _PoseModel({"a": Pose.identity(), "b": Pose.identity()})
    out = _PoseModel({"a": Pose.identity(), "n": Pose.identity()})
    assert sfm_mod._back_onto(out, base) is None  # type: ignore[arg-type]


# --- SfmModel on a real reconstruction ----------------------------------------------------------


def _reconstruction(centres: list[float], points: bool = True) -> Any:
    """Keyframes f0, f1, ... at x = ``centres`` looking along +z (5 keypoints each), and one
    point 5 m ahead that every keyframe sees (keypoint 0) when ``points``."""
    import pycolmap

    rec = pycolmap.Reconstruction()
    rec.add_camera_with_trivial_rig(
        pycolmap.Camera.create_from_model_id(1, pycolmap.CameraModelId.SIMPLE_PINHOLE, 100.0,
                                             64, 48))
    for i, x in enumerate(centres):
        pts = [pycolmap.Point2D(np.array([10.0 + j, 20.0])) for j in range(5)]
        img = pycolmap.Image(name=f"f{i}.jpg", camera_id=1, image_id=i + 1, points2D=pts)
        rec.add_image_with_trivial_frame(
            img, pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), np.array([-x, 0.0, 0.0])))
    if points:
        track = pycolmap.Track()
        for i in range(len(centres)):
            track.add_element(i + 1, 0)
        rec.add_point3D(np.array([0.0, 0.0, 5.0]), track, np.zeros(3, np.uint8))
    return rec


def test_a_division_camera_keeps_its_lens_and_its_observations_are_the_pinholes() -> None:
    import pycolmap

    rec = _reconstruction([0.0, 0.5, 1.0])
    cam = pycolmap.Camera.create_from_model_id(1, pycolmap.CameraModelId.SIMPLE_DIVISION, 100.0,
                                               64, 48)
    cam.params = [100.0, 32.0, 24.0, -0.3]
    rec.cameras[1].model = cam.model
    rec.cameras[1].params = cam.params
    model = SfmModel(rec, "multiview")
    K = model.intrinsics("f0.jpg")
    assert K.k == pytest.approx(-0.3) and K.fx == 100.0
    assert sfm_mod.camera_intrinsics(_reconstruction([0.0]).cameras[1]).k == 0.0
    uv, _ = model.observations("f0.jpg", max_error=1e9)
    np.testing.assert_allclose(uv, K.pinhole_pixels(np.array([[10.0, 20.0]])))


def test_model_points_and_deregistering_an_unknown_keyframe() -> None:
    model = SfmModel(_reconstruction([0.0, 0.5, 1.0]), "sfm-global")
    np.testing.assert_allclose(model.points(), [[0.0, 0.0, 5.0]])
    model.deregister({"f1.jpg", "nope.jpg"})
    assert model.registered == ["f0.jpg", "f2.jpg"]


def test_baseline_ratio_needs_three_keyframes_and_their_points() -> None:
    assert SfmModel(_reconstruction([0.0, 0.5]), "m").baseline_ratio() == 0.0
    assert SfmModel(_reconstruction([0.0, 0.5, 1.0], points=False), "m").baseline_ratio() == 0.0
    # 0.5 m between neighbours, the point 5 m ahead
    assert SfmModel(_reconstruction([0.0, 0.5, 1.0]), "m").baseline_ratio() == pytest.approx(0.1)


def test_the_largest_reconstruction_must_continue_the_extended_model() -> None:
    s = object.__new__(Sfm)
    recs = {0: _reconstruction([0.0, 0.5, 1.0])}
    assert s._largest(recs, anchors={"other.jpg"}) is None
    assert s._largest(recs, anchors={"f0.jpg"}) is recs[0]


# --- the COLMAP database ------------------------------------------------------------------------


def _db(path: Path, names: list[str], size: tuple[int, int] = (64, 48),
        focal: float = 100.0) -> dict[str, int]:
    import pycolmap

    db = pycolmap.Database.open(str(path))
    try:
        cam = pycolmap.Camera.create_from_model_id(0, pycolmap.CameraModelId.SIMPLE_PINHOLE,
                                                   focal, *size)
        cid = db.write_camera(cam)
        return {n: db.write_image(pycolmap.Image(name=n, camera_id=cid)) for n in names}
    finally:
        db.close()


def _pair(path: Path, i: int, j: int, inliers: int, config: int = 2) -> None:
    import pycolmap

    db = pycolmap.Database.open(str(path))
    try:
        g = pycolmap.TwoViewGeometry()
        g.config = config
        g.inlier_matches = np.stack([np.arange(inliers)] * 2, 1).astype(np.uint32)
        db.write_two_view_geometry(i, j, g)
    finally:
        db.close()


def _keypoints(path: Path, ids: dict[str, int], n: int = 40,
               same: np.ndarray | None = None) -> None:
    """``n`` keypoints per image (``same``: these for every image)."""
    import pycolmap

    db = pycolmap.Database.open(str(path))
    try:
        for k, i in enumerate(ids.values()):
            kp = np.arange(2 * n, dtype=np.float32).reshape(n, 2) + 100 * k
            db.write_keypoints(i, kp if same is None else same.astype(np.float32))
    finally:
        db.close()


def test_extraction_skips_keyframes_whose_features_the_database_holds(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "db.db"
    ids = _db(db, ["a.jpg", "b.jpg"])
    _keypoints(db, ids)

    def no_run(args: list[str], log: Path) -> None:
        raise AssertionError("nothing to extract")

    monkeypatch.setattr(sfm_mod, "_run", no_run)
    s = Sfm(db, tmp_path, tmp_path / "work")
    assert s.extract(["a.jpg", "b.jpg"], CameraPrior(64, 48, existing_id=1)) == 1


def test_an_existing_camera_must_exist_and_match_the_size_and_focal(tmp_path: Path) -> None:
    db = tmp_path / "db.db"
    s = Sfm(db, tmp_path, tmp_path / "work")
    assert s.existing_camera(CameraPrior(64, 48, existing_id=1)) is None  # no database yet
    _db(db, ["a.jpg"])
    assert s.existing_camera(CameraPrior(64, 48, existing_id=1)) == 1
    assert s.existing_camera(CameraPrior(64, 48, existing_id=7)) is None  # no such camera
    assert s.existing_camera(CameraPrior(640, 480, existing_id=1)) is None  # another size
    assert s.existing_camera(CameraPrior(64, 48, same_focal_ids=(1,))) is None  # no focal prior
    assert s.existing_camera(CameraPrior(64, 48, focal=100.5, same_focal_ids=(1,))) == 1
    assert s.existing_camera(CameraPrior(64, 48, focal=120.0, same_focal_ids=(1,))) is None


def test_no_pairs_are_no_matching(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sfm_mod, "_run", lambda args, log: pytest.fail("ran COLMAP"))
    assert Sfm(tmp_path / "db.db", tmp_path, tmp_path / "w").match_pairs(set(), {}) == 0


def test_lightglue_failure_restores_a_pair_sift_had_not_matched(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "db.db"
    ids = _db(db, ["a.jpg", "b.jpg"])

    def fails(args: list[str], log: Path) -> None:
        raise SfmError("colmap matches_importer failed (1): no model")

    monkeypatch.setattr(sfm_mod, "_run", fails)
    s = Sfm(db, tmp_path, tmp_path / "work")
    res = s.rematch_lightglue({frozenset(("a.jpg", "b.jpg"))})
    assert res == {"pairs": 1, "verified_before": 0, "verified_after": 0}
    import pycolmap

    d = pycolmap.Database.open(str(db))
    try:  # nothing was there, nothing is written back
        a, b = ids["a.jpg"], ids["b.jpg"]
        assert not d.exists_matches(a, b) and not d.exists_two_view_geometry(a, b)
    finally:
        d.close()


def test_two_view_statistics_count_verified_pairs_among_the_names(tmp_path: Path) -> None:
    db = tmp_path / "db.db"
    ids = _db(db, ["a.jpg", "b.jpg", "c.jpg", "d.jpg"])
    a, b, c, d = (ids[n] for n in ("a.jpg", "b.jpg", "c.jpg", "d.jpg"))
    _pair(db, a, b, 30, config=5)  # panoramic
    _pair(db, b, c, 30, config=2)
    _pair(db, c, d, 5, config=2)  # not verified
    _pair(db, a, 99, 30)  # an image the database does not hold
    s = Sfm(db, tmp_path, tmp_path / "work")
    stats = s.two_view_stats({"a.jpg", "b.jpg", "c.jpg"})
    assert stats == {"verified_pairs": 2.0, "rotation_fraction": 0.5, "planar_fraction": 0.0}
    every = s.two_view_stats()  # (c, d) is not verified; (a, 99) counts without names
    assert every == {"verified_pairs": 3.0, "rotation_fraction": pytest.approx(1 / 3),
                     "planar_fraction": 0.0}
    assert s.two_view_stats({"d.jpg"}) == {"verified_pairs": 0.0, "rotation_fraction": 0.0,
                                           "planar_fraction": 0.0}
    assert s.match_graph() == {"a.jpg": {"b.jpg"}, "b.jpg": {"a.jpg", "c.jpg"},
                               "c.jpg": {"b.jpg"}, "d.jpg": set()}


def test_the_largest_component_of_the_match_graph(tmp_path: Path) -> None:
    db = tmp_path / "db.db"
    ids = _db(db, ["a.jpg", "b.jpg", "c.jpg", "d.jpg"])
    _pair(db, ids["a.jpg"], ids["b.jpg"], 30)
    s = Sfm(db, tmp_path, tmp_path / "work")
    assert s.largest_component({"a.jpg", "b.jpg", "c.jpg", "d.jpg"}) == {"a.jpg", "b.jpg"}
    alone = s.largest_component({"c.jpg", "d.jpg"})
    assert len(alone) == 1 and alone <= {"c.jpg", "d.jpg"}  # no pair: one keyframe


def test_posed_images_of_the_base_keep_their_pose_and_unknown_ones_are_skipped(
        tmp_path: Path) -> None:
    import pycolmap

    db = tmp_path / "db.db"
    ids = _db(db, ["f0.jpg", "f1.jpg", "f2.jpg", "f3.jpg"])
    assert ids["f0.jpg"] == 1
    # the keypoints of the reconstruction's images, and no matches
    _keypoints(db, ids, same=np.column_stack([10.0 + np.arange(5), np.full(5, 20.0)]))
    base = tmp_path / "base"
    base.mkdir()
    _reconstruction([0.0, 0.5, 1.0], points=False).write(str(base))
    s = Sfm(db, tmp_path, tmp_path / "work")
    moved = Pose(np.eye(3), np.array([7.0, 0.0, 0.0]))
    new = Pose(np.eye(3), np.array([2.0, 0.0, 0.0]))
    out = s.extend_with_poses(base, {"f0.jpg": moved, "f3.jpg": new, "zz.jpg": new},
                              tmp_path / "out", "test")
    assert out.method == "test" and out.registered == ["f0.jpg", "f1.jpg", "f2.jpg", "f3.jpg"]
    np.testing.assert_allclose(out.pose("f0.jpg").t, [0.0, 0.0, 0.0], atol=1e-12)  # kept
    np.testing.assert_allclose(out.pose("f3.jpg").t, [2.0, 0.0, 0.0], atol=1e-12)
    assert isinstance(out.rec, pycolmap.Reconstruction)


# --- panorama -----------------------------------------------------------------------------------


def test_verified_matches_keep_used_configurations_with_enough_inliers(tmp_path: Path) -> None:
    db = tmp_path / "db.db"
    ids = _db(db, ["a.jpg", "b.jpg", "c.jpg", "x.jpg"])
    _keypoints(db, ids)
    a, b, c, x = (ids[n] for n in ("a.jpg", "b.jpg", "c.jpg", "x.jpg"))
    _pair(db, b, a, 20, config=2)
    _pair(db, a, c, 30, config=1)  # degenerate: not a used configuration
    _pair(db, b, c, 10, config=2)  # too few inliers
    _pair(db, a, x, 30, config=2)  # x is not asked for
    (m,) = panorama.verified_matches(db, ["a.jpg", "b.jpg", "c.jpg"])
    assert (m.a, m.b, len(m.uv_a)) == ("a.jpg", "b.jpg", 20)
    np.testing.assert_array_equal(m.uv_a[1], [2.0 + 100 * 0, 3.0])  # a's keypoint 1


def test_a_lens_keypoints_are_matched_where_its_pinhole_sees_them(tmp_path: Path) -> None:
    """A camera with distortion (a multi-view map's refined lens, ``Sfm.set_distortion``): the
    matches are where its pinhole sees the keypoints (its keyframes' depth is on its grid)."""
    import pycolmap

    db = tmp_path / "db.db"
    ids = _db(db, ["a.jpg", "b.jpg"])
    _keypoints(db, ids)
    _pair(db, ids["b.jpg"], ids["a.jpg"], 20)
    other = _db(db, ["c.jpg"], size=(80, 60))  # another camera, not of the images named
    s = Sfm(db, tmp_path, tmp_path / "work")
    s.set_distortion({"a.jpg"}, 1.1, -0.2)
    d = pycolmap.Database.open(str(db))
    try:
        cam, kept = d.read_all_cameras()
        raw = np.asarray(d.read_keypoints(ids["b.jpg"]), np.float64)[:20, :2]
    finally:
        d.close()
    assert other and kept.model.name == "SIMPLE_PINHOLE" and kept.width == 80
    assert cam.model.name == "SIMPLE_DIVISION" and cam.has_prior_focal_length is False
    np.testing.assert_allclose(cam.params, [110.0, 32.0, 24.0, -0.2])
    assert panorama.distorted(cam)
    (m,) = panorama.verified_matches(db, ["a.jpg", "b.jpg"])
    d_ = (raw - [32.0, 24.0]) / 110.0
    pin = d_ / (1 - 0.2 * np.sum(d_ * d_, axis=1, keepdims=True)) * 110.0 + [32.0, 24.0]
    np.testing.assert_allclose(m.uv_b, pin)
    np.testing.assert_allclose(panorama.pinhole_pixels(cam, raw), pin)


def test_a_rebuild_fits_the_lens_again_from_a_pinhole(tmp_path: Path) -> None:
    """``Sfm.drop_distortion``: the lens camera becomes the pinhole of its focal length and
    principal point; a pinhole camera stays as it is."""
    import pycolmap

    db = tmp_path / "db.db"
    _db(db, ["a.jpg"])
    s = Sfm(db, tmp_path, tmp_path / "work")
    s.set_distortion({"a.jpg"}, 1.1, -0.2)
    s.drop_distortion(1)
    s.drop_distortion(1)  # a pinhole already
    d = pycolmap.Database.open(str(db))
    try:
        (cam,) = d.read_all_cameras()
    finally:
        d.close()
    assert cam.model.name == "SIMPLE_PINHOLE" and not panorama.distorted(cam)
    np.testing.assert_allclose(cam.params, [110.0, 32.0, 24.0])


def test_corner_shift_is_the_share_of_the_half_diagonal() -> None:
    assert panorama.corner_shift((640, 480), 400.0, 0.0) == 0.0
    # half-diagonal 400 px = 1 at f 400: k = -0.2 sees it at 1 / 0.8
    assert panorama.corner_shift((640, 480), 400.0, -0.2) == pytest.approx(0.25)


def test_a_multiview_model_takes_the_refined_camera(tmp_path: Path) -> None:
    db = tmp_path / "db.db"
    ids = _db(db, ["f0.jpg", "f1.jpg"])
    _keypoints(db, ids)
    poses = {"f0.jpg": Pose.identity(), "f1.jpg": Pose(np.eye(3), np.array([0.1, 0.0, 0.0]))}
    s = Sfm(db, tmp_path, tmp_path / "work")
    pin = s.triangulate_with_poses(poses, tmp_path / "pin", focal_scale=1.1)
    cam = pin.rec.cameras[1]
    assert cam.model.name == "SIMPLE_PINHOLE"
    np.testing.assert_allclose(cam.params, [110.0, 32.0, 24.0])
    lens = s.triangulate_with_poses(poses, tmp_path / "lens", focal_scale=1.1, distortion=-0.2)
    cam = lens.rec.cameras[1]
    assert cam.model.name == "SIMPLE_DIVISION"  # the database camera too (verified_matches)
    np.testing.assert_allclose(cam.params, [110.0, 32.0, 24.0, -0.2])


def test_refinement_without_matches_keeps_the_poses() -> None:
    T = head_pose(10.0)
    views = {"a": panorama.View(T, Intrinsics(435.0, 435.0, 320.0, 240.0, 640, 480))}
    fit = panorama.refine_poses([], views, {"a"})
    assert fit.poses == {"a": T} and fit.pairs == 0 and fit.matches == 0


def test_a_singular_step_is_retried_with_more_damping(monkeypatch: pytest.MonkeyPatch) -> None:
    truth = {f"f{k}": head_pose(20.0 * k) for k in range(4)}
    rig = turning_rig(truth, seed=1)
    rng = np.random.default_rng(0)
    from tests.synth.turning import perturb, rot_err_deg

    init = {n: T if n == "f0" else perturb(T, 1.0, 0.02, rng) for n, T in truth.items()}
    solve = np.linalg.solve
    calls: list[int] = []

    def flaky(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        calls.append(1)
        if len(calls) == 1:
            raise np.linalg.LinAlgError("Singular matrix")
        return solve(A, b)

    monkeypatch.setattr(np.linalg, "solve", flaky)
    fit = panorama.refine_poses(rig.pairs, rig.views(init), set(truth) - {"f0"})
    assert len(calls) > 2  # the failed step was retried
    assert max(rot_err_deg(fit.poses[n], truth[n]) for n in fit.poses) < 0.2


def test_centre_hints_without_fixed_keyframes_keep_each_centre() -> None:
    K = Intrinsics(435.0, 435.0, 320.0, 240.0, 640, 480)
    views = {"a": panorama.View(head_pose(0.0), K), "b": panorama.View(head_pose(30.0), K)}
    hints = panorama.centre_hints([], views, {"a", "b"})
    for n in ("a", "b"):
        np.testing.assert_array_equal(hints[n], views[n].pose.t)
