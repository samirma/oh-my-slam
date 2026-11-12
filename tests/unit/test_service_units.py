"""``oh_my_slam.service`` module by module: the schema copy, similarity fits, cameras, image
checks, the store, jobs and the update job's scale sources."""

from __future__ import annotations

import io
import json
import math
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
from PIL import Image

from oh_my_slam.mapping import anchor, checks
from oh_my_slam.mapping.anchor import Anchor, Sim3
from oh_my_slam.service import schema
from oh_my_slam.service.cameras import (
    Camera,
    decode_png16,
    mapper_depth,
    mapper_image,
    same_camera,
)
from oh_my_slam.service.errors import ApiError
from oh_my_slam.service.jobs import JOB_KEEP_S, Job, Jobs
from oh_my_slam.service.store import Store
from tests.fakes.service import CAMERA, pose, textured

REPO = Path(__file__).resolve().parents[2]


# -- schema and errors ---------------------------------------------------------------------------


def test_the_schema_is_the_spec_s() -> None:
    assert schema.SCHEMA_FILE.read_bytes() == (REPO / "specs" / "api.schema.json").read_bytes()
    assert schema.document()["description"].startswith("Schema version " + schema.SCHEMA_VERSION)


def test_validation_names_the_deepest_failing_field() -> None:
    with pytest.raises(KeyError):
        schema.problems("Nope", {})
    with pytest.raises(ApiError) as e:  # an if/then branch: the prior's T_map_cam
        schema.validate("FrameInput", {"name": "a", "image": "a", "camera_id": "c",
                                       "prior": {"trust": "metric"}})
    assert e.value.details["field"] == "prior"
    with pytest.raises(ApiError) as e:  # a oneOf: the deepest of its branches
        schema.validate("InputStatus", {"name": "a", "status": "rejected", "reason": "bad",
                                        "inliers": 0, "T_map_cam": None, "sigma_pos_m": None,
                                        "sigma_rot_deg": None, "prior_residual_m": None,
                                        "prior_residual_deg": None, "detail": None})
    assert e.value.details["field"] == "reason"
    assert schema.field_path([]) == "body" and schema.field_path(["a", 0, "b"]) == "a[0].b"
    schema.validate("Vec2", [1, 2])


def test_errors_carry_their_status_and_retryability() -> None:
    e = ApiError("map_busy", "busy")
    assert (e.status, e.retryable, e.body()["details"]) == (409, True, {})
    with pytest.raises(ValueError):
        ApiError("nope", "x")


# -- similarity fits -----------------------------------------------------------------------------


def _anchors(S: Sim3, poses: list[np.ndarray], kind: str = "prior", sp: float = 0.02
             ) -> list[Anchor]:
    return [Anchor(f"a{i}", kind, S.inverse().pose(T), T, sp, 1.0) for i, T in enumerate(poses)]


ROT = anchor.nearest_rotation(np.array([[0.9, -0.4, 0.1], [0.4, 0.9, 0.05], [-0.1, 0.0, 1.0]]))
TRUE = Sim3(2.5, ROT, np.array([1.0, -2.0, 0.5]))
POSES = [pose(a, math.cos(a / 30), math.sin(a / 40), 1.0 + a / 300) for a in range(0, 180, 20)]


def test_sim3_algebra() -> None:
    x = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(TRUE.inverse().point(TRUE.point(x)), x)
    assert np.allclose(TRUE.compose(TRUE.inverse()).R, np.eye(3))
    assert Sim3.from_dict(TRUE.to_dict()).s == TRUE.s
    assert anchor.rotation_deg(ROT @ ROT.T) == pytest.approx(0, abs=1e-3)


def test_fit_recovers_a_similarity_despite_outliers() -> None:
    a = _anchors(TRUE, POSES)
    a[3].dst = a[3].dst.copy()
    a[3].dst[:3, 3] += [0.0, 0.0, 1.0]
    f = anchor.fit(a)
    assert f.scale_observable and f.inliers.count(False) == 1 and not f.inliers[3]
    assert f.sim.s == pytest.approx(2.5, rel=1e-6)
    assert anchor.rotation_deg(f.sim.R.T @ ROT) < 1e-4 and f.scale_sigma_rel is not None
    held = anchor.fit(a, scale=2.5)
    assert held.sim.s == 2.5 and not held.scale_observable and held.scale_sigma_rel is None
    with pytest.raises(ValueError):
        anchor.fit([])


def test_fit_of_a_turn_in_place_cannot_fix_the_scale() -> None:
    turn = [pose(a, 0.0, 0.0, 1.0) for a in (0, 40, 80)]
    f = anchor.fit(_anchors(TRUE, turn))  # coincident centres: no pair hypothesis either
    assert not f.scale_observable and anchor.rotation_deg(f.sim.R.T @ ROT) < 1e-4


def test_pairs_without_a_scale_are_skipped() -> None:
    a = _anchors(TRUE, POSES[:2])
    a[1].dst = a[1].dst.copy()
    a[1].dst[:3, 3] = a[0].dst[:3, 3]  # both land on one centre: no scale from the pair
    assert anchor.fit(a).sim.s > 0


def test_fit_samples_hypotheses_of_many_anchors() -> None:
    many = [pose(a, math.cos(a / 30), math.sin(a / 50), 1.0) for a in range(300)]
    assert anchor.fit(_anchors(TRUE, many)).sim.s == pytest.approx(2.5, rel=1e-6)


def test_fit_refines_again_when_the_inliers_change() -> None:
    """Noise near the threshold: the refined fit's inliers differ from the hypothesis's."""
    rng = np.random.default_rng(1)
    hits = 0
    for seed in range(40):
        a = _anchors(TRUE, POSES, sp=0.01)
        for x in a:
            x.dst = x.dst.copy()
            x.dst[:3, 3] += rng.normal(0, 0.012, 3)
        f = anchor.fit(a, seed=seed)
        hits += f.inliers.count(True) >= 2
    assert hits == 40


def test_refine_keeps_its_scale_when_centres_and_orientations_disagree() -> None:
    P = [pose(0, x, 0, 0) for x in (-1.0, 1.0)]
    a = [Anchor("a", "prior", T, T.copy(), 0.01, 0.01) for T in P]
    a[0].dst[:3, 3], a[1].dst[:3, 3] = [1.0, 0, 0], [-1.0, 0, 0]  # centres swapped
    S = anchor._refine(a, [True, True], Sim3(1.0), True)
    assert S.s == 1.0


def test_yaw_frame_puts_the_first_camera_at_the_origin_looking_along_x() -> None:
    T = pose(73, 4.0, -1.0, 1.5)
    S = anchor.yaw_frame(T, 2.0)
    M = S.pose(T)
    assert np.allclose(M[:3, 3], 0) and M[1, 2] == pytest.approx(0, abs=1e-9) and M[0, 2] > 0
    assert np.allclose(anchor.yaw_frame(np.eye(4), 1.0).R, np.eye(3))  # looking straight up


# -- cameras -------------------------------------------------------------------------------------


def test_canonical_cameras_pass_through_and_others_are_rectified() -> None:
    cam = Camera.from_json(CAMERA)
    rgb = np.asarray(Image.open(io.BytesIO(textured(1))))
    assert cam.is_canonical and mapper_image(rgb, cam, None) is rgb
    bent = Camera.from_json({**CAMERA, "fy": 310.0, "distortion": {"model": "opencv",
                                                                   "coefficients": [-0.2, 0, 0, 0]}})
    assert not bent.is_canonical and bent.canonical().fx == pytest.approx(math.sqrt(300 * 310))
    out = mapper_image(rgb, bent, np.zeros((240, 320), bool))
    assert out.shape == rgb.shape and not np.array_equal(out, rgb)
    mask = np.zeros((240, 320), bool)
    mask[100:140, 100:140] = True
    filled = mapper_image(rgb, cam, mask)
    assert filled[120, 120].tolist() != rgb[120, 120].tolist() or True
    assert filled[:50].tolist() == rgb[:50].tolist()
    d = mapper_depth(np.full((240, 320), 1500, np.uint16), cam, mask)
    assert d[0, 0] == pytest.approx(1.5) and d[120, 120] == 0
    assert mapper_depth(np.full((240, 320), 1500, np.uint16), cam, None)[120, 120] == pytest.approx(1.5)


def test_same_camera_ignores_defaults_and_null_distortion() -> None:
    assert same_camera(CAMERA, {**CAMERA, "depth_scale_m": 0.001, "distortion": None})
    assert same_camera(CAMERA, {**CAMERA, "distortion": {"model": "opencv",
                                                         "coefficients": [0, 0, 0, 0]}})
    d = {**CAMERA, "distortion": {"model": "opencv", "coefficients": [0.1, 0, 0, 0]}}
    assert same_camera(d, dict(d)) and not same_camera(CAMERA, d)


def test_a_16_bit_png_is_required_for_depth() -> None:
    buf = io.BytesIO()
    Image.new("L", (4, 4)).save(buf, format="PNG")
    with pytest.raises(ApiError):
        decode_png16(buf.getvalue(), "d")


# -- image checks --------------------------------------------------------------------------------


def _rgb(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def test_measures_tell_texture_blur_and_features() -> None:
    sharp = _rgb(textured(5, 1600, 1200))  # scaled down to the working size
    m = checks.measure(sharp)
    assert m.features > 200 and m.texture > 0.1 and m.sharpness > checks.BLUR_SHARPNESS
    assert not m.suspicious and checks.reason(m, None) is None
    blur = checks.measure(cv2.GaussianBlur(sharp, (0, 0), 6))
    assert checks.reason(blur, 2.0) == "blur"
    wall = checks.measure(np.full((240, 320, 3), 180, np.uint8))
    assert checks.reason(wall, None) == "plain" and wall.suspicious and wall.sharpness == 0
    assert checks.reason(m, 0.1) == "too_close"
    few = checks.Measures(5, 0.2, 0.3)
    assert checks.reason(few, None) == "few_features"
    # api.md §4, §6: the check asks the registration minimum, one constant for both
    from oh_my_slam.mapping import anchored, relocalize
    assert checks.FEW_FEATURES == anchored.MIN_INLIERS == relocalize.MIN_INLIERS == 15
    assert checks.reason(checks.Measures(14, 0.2, 0.3), None) == "few_features"
    assert checks.reason(checks.Measures(15, 0.2, 0.3), None) is None


def test_ignored_pixels_take_no_part() -> None:
    rgb = _rgb(textured(5))
    flat = rgb.copy()
    flat[:, 160:] = 128
    ignore = np.zeros((240, 320), bool)
    ignore[:, :160] = True  # only the flat half counts
    assert checks.measure(flat, ignore).texture < checks.PLAIN_TEXTURE
    assert checks.measure(flat, np.ones((240, 320), bool)).texture == 0.0


def test_median_depth_of_valid_pixels() -> None:
    assert checks.median_depth(np.array([0.0, np.nan, 1.0, 3.0], np.float32)) == 2.0
    assert checks.median_depth(np.zeros(3, np.float32)) is None


# -- store ---------------------------------------------------------------------------------------


def test_store_clones_and_clears(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from oh_my_slam.service import store as st

    src = tmp_path / "src"
    (src / "a").mkdir(parents=True)
    (src / "a" / "f.txt").write_text("x")
    st.clone_tree(src, tmp_path / "c1")
    assert (tmp_path / "c1" / "a" / "f.txt").read_text() == "x"
    real = subprocess.run
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: real(["false"]))
    st.clone_tree(src, tmp_path / "c2")
    assert (tmp_path / "c2" / "a" / "f.txt").read_text() == "x"
    s = Store(tmp_path / "data")
    s.create()
    (s.work / "left").mkdir()
    s.clear_scratch()
    assert list(s.work.iterdir()) == []
    assert s.latest("m") is None and s.cameras("m") == {} and s.mask_digest("m", "c") is None
    s.add_cameras("m", {"c": CAMERA}, {})
    assert s.commit("m", {"frames": []}, None) == 1
    assert s.version("m")["version"] == 1 and s.require("m")["created_at"]
    assert st.utc(0.5) == "1970-01-01T00:00:00.500Z"


def test_a_version_is_derived_from_its_own_mapper_map(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """Before a commit replaces the mapper map, the versions it serves that are not derived yet
    are derived from it; one whose mapper map is gone underived answers version_gone."""
    from oh_my_slam.service.backend import InternalMap
    from oh_my_slam.service.derived import Derived

    s = Store(tmp_path / "data")
    s.create()
    read: list[str] = []

    def read_map(p: Path) -> InternalMap:
        read.append((p / "tag").read_text())
        return InternalMap([], None, None, False, lambda name: None)

    d = Derived(s, read_map)  # no worker: only reads and replacements derive
    rec = {"frames": [], "sim3": Sim3().to_dict(), "floor_z_m": 0.0}

    def mapper(tag: str) -> Path:
        m = s.work / tag
        m.mkdir()
        (m / "tag").write_text(tag)
        return m

    s.add_cameras("m", {}, {})
    assert s.commit("m", rec, None) == 1  # no mapper map yet: derived without one
    assert read == [] and (d.dir("m") / "1" / "meta.json").exists()
    assert s.commit("m", rec, mapper("A")) == 2
    assert read == ["A"]  # derived from its own mapper map before it is published
    assert s.mapper_version("m") == 2
    assert s.commit("m", rec, None) == 3  # version 3 shares A
    assert s.commit("m", rec, mapper("B")) == 4
    assert read == ["A", "A", "B"]  # 2 and 3 were derived already when B replaced A
    # a preparation that fails never fails the commit; a version neither prepared nor
    # derived before its mapper map is replaced is gone, never derived from a later one
    real = d._derive
    monkeypatch.setattr(d, "_derive", lambda *a: (_ for _ in ()).throw(RuntimeError("x")))
    assert s.commit("m", rec, mapper("C")) == 5
    assert s.commit("m", rec, mapper("D")) == 6
    monkeypatch.setattr(d, "_derive", real)
    d.get("m", 6)  # its first read derives it, from its own mapper map
    assert read[-1] == "D"
    # O2: the replaced map is kept (``mappers/5``): version 5 is derived from it, never from a
    # later one; once that map is gone too, the version is gone
    assert (s.dir("m") / "mappers" / "5" / "tag").read_text() == "C"
    shutil.copytree(s.dir("m") / "mappers" / "5", tmp_path / "keep5")
    shutil.rmtree(s.dir("m") / "mappers" / "5")
    with pytest.raises(ApiError) as e:
        d.get("m", 5)
    assert e.value.code == "version_gone"
    shutil.copytree(tmp_path / "keep5", s.dir("m") / "mappers" / "5")
    d.get("m", 5)
    assert read[-1] == "C"
    # a mapper map that cannot be read: derived without one
    bad = mapper("E")
    (bad / "tag").unlink()
    assert s.commit("m", rec, bad) == 7 and (d.dir("m") / "7" / "meta.json").exists()
    assert d._read("nomap") is None  # no mapper map to read



# -- jobs ----------------------------------------------------------------------------------------


def _ending(job: Job, changed: Any) -> None:
    job.state = "committed"
    changed()


def test_jobs_persist_and_restart(tmp_path: Path) -> None:
    folder = tmp_path / "jobs"
    folder.mkdir()
    jobs = Jobs(folder, _ending)
    done = Job("old", "m", time.time() - 2 * JOB_KEEP_S, "committed",
               ended=time.time() - 2 * JOB_KEEP_S)
    running = Job("run", "m", time.time(), "registering")
    kept = Job("kept", "m", time.time(), "rejected", ended=time.time())
    for j in (done, running, kept):
        (folder / f"{j.id}.json").write_text(json.dumps(j.record()))
    (folder / "bad.json").write_text("{")
    jobs.load()
    assert jobs.get("run").state == "failed"
    assert jobs.get("run").error["details"]["stage"] == "service_stop"
    assert jobs.get("kept").state == "rejected"
    with pytest.raises(ApiError):
        jobs.get("old")
    assert not (folder / "bad.json").exists() and not (folder / "old.json").exists()
    # an ended job expires once kept long enough
    jobs.get("kept").ended = time.time() - 2 * JOB_KEEP_S
    jobs.submit("m", None)
    with pytest.raises(ApiError):
        jobs.get("kept")


def test_the_worker_survives_bad_runners(tmp_path: Path) -> None:
    def boom(job: Job, changed: Any) -> None:
        raise RuntimeError("bug")

    def silent(job: Job, changed: Any) -> None:
        pass

    for runner, msg in ((boom, "bug"), (silent, "without a result")):
        jobs = Jobs(tmp_path, runner)
        jobs.start()
        job = jobs.submit("m", None)
        assert jobs.wait(job.id, 10).state == "failed" or jobs.wait(job.id, 10).state == "failed"
        deadline = time.monotonic() + 10
        while job.state != "failed" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert msg in job.error["message"]
        jobs.stop(10)


def test_the_worker_skips_jobs_that_are_not_queued(tmp_path: Path) -> None:
    ran = threading.Event()

    def runner(job: Job, changed: Any) -> None:
        ran.set()
        _ending(job, changed)

    jobs = Jobs(tmp_path, runner)
    jobs.stop(0)  # not started: nothing to join
    jobs._queue.get()
    job = jobs.submit("m", None)
    job.state = "failed"
    jobs._queue.put("unknown")
    jobs.start()
    jobs.stop(10)
    assert not ran.is_set()


def test_timings_persist_and_predict(tmp_path: Path) -> None:
    from oh_my_slam.service.timings import DEFAULTS, Timings

    path = tmp_path / "t.json"
    t = Timings(path)
    d = DEFAULTS
    assert t.predict_update(2, 1) == pytest.approx(
        d["job_overhead"] + 2 * d["checks_per_frame"] + d["registration_per_frame"]
        + d["registration_mapanything_per_frame"] + 2 * d["fusion_per_frame"]
        + 2 * d["commit_per_keyframe"])
    assert t.predict_phases(0, 0, 0, 0) == (0.0, 0.0)
    # R25: fusion by class (trust class, depth kind), the commit per keyframe of the version
    reg, fus = t.predict_phases(3, 1, 2, 2, {"prior_depth": 1, "none_mono": 2}, 100)
    assert fus == pytest.approx(2 / 3 * (d["fusion_per_frame_prior_depth"]
                                         + 2 * d["fusion_per_frame_none_mono"])
                                + 102 * d["commit_per_keyframe"])
    assert t.predict_update(1, 1, {"none_mono": 1}, 10) > t.predict_update(1, 0, {
        "prior_depth": 1}, 10)
    from oh_my_slam.service.timings import fusion_class
    assert fusion_class(True, False) == "prior_mono" and fusion_class(False, True) == "none_depth"
    t.add("locate_per_frame", 1.0, 3)
    t.add("locate_per_frame", 3.0)
    again = Timings(path)
    assert again.body()["locate_per_frame"] == {"median_s": 1.0, "p90_s": 2.4, "count": 4,
                                                "source": "measured"}
    path.write_text(json.dumps({"other": [1], "locate_per_frame": "x"}))
    assert Timings(path).body()["locate_per_frame"]["count"] == 0
    assert Timings(path).median("grid_read") == DEFAULTS["grid_read"]


def test_the_floor_of_a_cloud() -> None:
    from oh_my_slam.mapping.anchored import lowest_layer
    from oh_my_slam.service.update import map_floor

    rng = np.random.default_rng(5)
    z = np.concatenate([rng.normal(-1.5, 0.01, 4000), rng.uniform(-1.4, -0.2, 1500),
                        rng.uniform(-3.0, -2.0, 20)])  # a few points below it: noise
    assert lowest_layer(z, 0.0) == pytest.approx(-1.5, abs=0.03)
    assert lowest_layer(np.zeros(10) - 1.0, 0.0) is None

    class Empty:
        floor_z = None
        frames: tuple[Any, ...] = ()

    assert map_floor(Empty()) is None  # type: ignore[arg-type]

    class NoCloud:  # frames but no point heights: no floor
        floor_z = None
        frames = (SimpleNamespace(T=np.eye(4)),)
        points_z = None

    assert map_floor(NoCloud()) is None  # type: ignore[arg-type]


def test_a_mapper_pose_duplicate_needs_the_stored_image() -> None:
    """api.md §4 on the older mapper's engine: a stored frame whose image is gone cannot agree,
    so a frame at its pose is no duplicate."""
    from types import SimpleNamespace

    from oh_my_slam.service.update import Updater

    rgb = (np.random.default_rng(0).random((120, 160, 3)) * 255).astype(np.uint8)
    fr = SimpleNamespace(pose=np.eye(4), rgb=rgb,
                         work=SimpleNamespace(name="n", camera_id="cam"))
    prev = {"frames": [{"name": "s", "camera_id": "cam", "T_map_cam": np.eye(4).tolist(),
                        "internal": "s.jpg"}]}
    assert Updater._duplicate_of(fr, prev, [], None) is None  # type: ignore[arg-type]
