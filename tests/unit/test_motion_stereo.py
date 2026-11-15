"""Frame analysis with motion stereo (api.md §7 "Motion stereo", "Baseline scale", "Baseline
check", "Depth sigma"; oms/2.11), through the Starlette app with ``tests.fakes.service.FakeBackend``
(the fixtures of ``test_service_analyse``) on AiNex-like views (``tests.synth.stereo``): the
humanoid's head camera 0.33 m above the floor, pitched 25 degrees down, near obstacles 0.35-2 m
ahead, exact depth truth. The fake model's depth is the truth in its own metres (0.8 x), its
shape stretched and bent (``model_depth``). Every body is checked against api.schema.json; the
numbers are set against the evaluation's (``reports/motion_stereo_eval.md`` §4, §9)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oh_my_slam.core.types import Intrinsics
from oh_my_slam.mapping import motion
from oh_my_slam.mapping.anchored import Features, exp_so3, rot_deg
from oh_my_slam.reconstruction import stereo
from tests.synth import stereo as syn
from tests.unit.test_service_analyse import An, _knobs, an, decode, png

__all__ = ["_knobs", "an"]  # the fixtures

K = Intrinsics(syn.F, syn.F, syn.CX, syn.CY, syn.W, syn.H, "model")
BINS = ((0.2, 0.5), (0.5, 1.0), (1.0, 1.5))
SCHEMA = Path(__file__).resolve().parents[2] / "specs" / "api.schema.json"


@pytest.fixture(scope="module")
def views() -> dict[str, syn.Shot]:
    """The reference (the head at the origin) and the other views: a 10 cm side step, a 1 cm
    one, a 30 degree head pan about the neck 5.7 cm behind the optical centre; a level
    reference facing a wall 12 m away and a 5 cm step there; a flat grey image."""
    room = syn.ainex_room()
    far = syn.far_wall()
    out = {"ref": syn.render(room, syn.head(), 0), "step10": syn.render(room, syn.head(0.0, 0.10), 1),
           "step1": syn.render(room, syn.head(0.0, 0.01), 2), "pan30": syn.render(room, syn.panned(30), 3),
           "far": syn.render(far, syn.head(pitch_deg=0.0), 4),
           "far5": syn.render(far, syn.head(0.0, 0.05, pitch_deg=0.0), 5)}
    grey = syn.render(syn.Room(flat={syn.WALLS}).add((5, -50, -50), (6, 50, 50), (128, 128, 128),
                                                syn.WALLS), syn.head(), 6)
    out["grey"] = grey
    return out


def rel(v: dict[str, syn.Shot], other: str, ref: str = "ref") -> np.ndarray:
    return syn.relative(v[ref].T, v[other].T)


def frame(T: np.ndarray, part: str, sigma_pos_m: float = 0.002, sigma_rot_deg: float = 0.5
          ) -> dict[str, Any]:
    return {"image": part, "T_ref_frame": T.tolist(), "sigma_pos_m": sigma_pos_m,
            "sigma_rot_deg": sigma_rot_deg}


def analyse(an: An, v: dict[str, syn.Shot], frames: list[dict[str, Any]], height: float | None = None,
            outputs: tuple[str, ...] = ("depth", "depth_sigma", "range_scan"), ref: str = "ref",
            status: int = 200, **extra: Any) -> dict[str, Any]:
    """The analysis of view ``ref`` in the client frame of its pose, with ``frames`` (their
    ``image`` the names of ``v``'s views)."""
    an.backend.mono_depth[hashlib.sha256(v[ref].jpeg).hexdigest()] = syn.model_depth(v[ref].depth)
    more = extra.pop("_files", {})
    cam: dict[str, Any] = dict(syn.HUMANOID)
    if height is not None:
        cam["height_above_floor_m"] = height
    req = {"image": "ref", "camera": cam, "T_map_cam": v[ref].T.tolist(), "outputs": list(outputs),
           "voxel_m": 0, **extra}
    if frames:
        req["frames"] = frames
    files = {"ref": v[ref].jpeg, **{f["image"]: v[f["image"]].jpeg for f in frames
                                    if f["image"] in v}}
    files.update(more)
    return an.analyse(req, files, status)


def depth_m(doc: dict[str, Any]) -> np.ndarray:
    return decode(doc["depth"]).astype(np.float64) * 0.001


def errors(z: np.ndarray, shot: syn.Shot) -> list[float]:
    """The median |z / truth - 1| of the obstacle pixels (surface ids 2 and up) per BINS."""
    out = []
    for lo, hi in BINS:
        sel = (shot.ids >= 2) & (shot.depth >= lo) & (shot.depth < hi) & (z > 0)
        out.append(float(np.median(np.abs(z[sel] / shot.depth[sel] - 1.0))))
    return out


def beam(doc: dict[str, Any], bearing: float) -> dict[str, Any]:
    return next(b for b in doc["range_scan"] if b["bearing_deg"] == bearing)


# -- baseline scale ------------------------------------------------------------------------------


def test_a_side_step_with_its_exact_pose_gives_baseline_scale_and_metric_depth(
        an: An, views: dict[str, syn.Shot]) -> None:
    """A 10 cm side step known to 2 mm (σ_t/|t| 0.02): ``used``, ``scale_source`` baseline with
    hypot(σ_t/|t|, fit, spread), the depth within 1-2 % of the truth on obstacles up to 1.5 m
    (the evaluation: side steps 1.0 / 3.0 / 4.9 %; the model alone, even at the true scale,
    2-6 %), the box underfoot at its range in the scan."""
    T = rel(views, "step10")
    doc = analyse(an, views, [frame(T, "step10")])
    f = doc["motion"]["frames"][0]
    assert (f["index"], f["status"], doc["scale_source"]) == (0, "used", "baseline")
    assert f["points"] >= motion.BASELINE_MIN_POINTS * 10
    assert f["rotation_change_deg"] < 0.1 and f["sigma_rot_deg_post"] < 0.1
    refined = np.asarray(f["T_ref_frame_refined"])
    assert np.linalg.norm(refined[:3, 3]) == pytest.approx(0.10, abs=1e-6)  # |t| is the client's
    assert rot_deg(refined[:3, :3] @ T[:3, :3].T) < 0.1
    assert 0.02 <= doc["scale_sigma_rel"] < 0.08 and "baseline_check" not in doc
    err = errors(depth_m(doc), views["ref"])
    assert err[0] < 0.01 and err[1] < 0.015 and err[2] < 0.03, err
    near = beam(doc, 0.0)
    assert near["kind"] == "obstacle" and near["range_m"] == pytest.approx(0.35, rel=0.03)
    # the frame alone: the model's metres
    mono = analyse(an, views, [])
    assert mono["scale_source"] == "monocular" and "motion" not in mono
    assert min(errors(depth_m(mono), views["ref"])) > 0.1
    assert beam(mono, 0.0)["range_m"] < 0.32


def test_a_head_pan_s_rotation_is_refined_on_the_images(an: An, views: dict[str, syn.Shot]) -> None:
    """A 30 degree head pan (chord 2.9 cm, known to 1 mm) sent 1 degree off: the refined rotation
    lies within 0.1 degree of the truth, the sent translation's length is kept, and the pan
    still gives the scale (the evaluation: rotation after refinement 0.02-0.05 degrees on pans
    up to 30 degrees; obstacles 2.6 / 9.0 / 16.3 % on its 5.7 cm pans with an exact pose)."""
    T = rel(views, "pan30")
    sent = T.copy()
    sent[:3, :3] = exp_so3(np.radians([0.6, 0.8, 0.0])) @ T[:3, :3]
    doc = analyse(an, views, [frame(sent, "pan30", 0.001, 1.0)])
    f = doc["motion"]["frames"][0]
    refined = np.asarray(f["T_ref_frame_refined"])
    assert f["status"] == "used" and doc["scale_source"] == "baseline"
    assert rot_deg(refined[:3, :3] @ T[:3, :3].T) < 0.1
    assert f["rotation_change_deg"] == pytest.approx(1.0, abs=0.1)
    assert np.linalg.norm(refined[:3, 3]) == pytest.approx(np.linalg.norm(T[:3, 3]), abs=1e-6)
    err = errors(depth_m(doc), views["ref"])
    assert err[0] < 0.026 and err[1] < 0.09 and err[2] < 0.163, err


def test_a_rotation_10_degrees_off_is_inconsistent_unless_its_sigma_allows_it(
        an: An, views: dict[str, syn.Shot]) -> None:
    """The pan sent 10 degrees off: with sigma 1 degree the images' rotation lies beyond 3
    sigma, ``pose_inconsistent`` (its refined pose reported, no points used, the frame's own
    scale); with sigma 5 degrees it is refined to the truth and used."""
    T = rel(views, "pan30")
    sent = T.copy()
    sent[:3, :3] = exp_so3(np.radians([0.0, 10.0, 0.0])) @ T[:3, :3]
    doc = analyse(an, views, [frame(sent, "pan30", 0.001, 1.0)])
    f = doc["motion"]["frames"][0]
    assert (f["status"], f["points"], doc["scale_source"]) == ("pose_inconsistent", 0, "monocular")
    assert f["rotation_change_deg"] == pytest.approx(10.0, abs=0.2)
    assert rot_deg(np.asarray(f["T_ref_frame_refined"])[:3, :3] @ T[:3, :3].T) < 0.1
    assert depth_m(doc) == pytest.approx(depth_m(analyse(an, views, [])))  # as one frame
    loose = analyse(an, views, [frame(sent, "pan30", 0.001, 5.0)])["motion"]["frames"][0]
    assert loose["status"] == "used"
    assert rot_deg(np.asarray(loose["T_ref_frame_refined"])[:3, :3] @ T[:3, :3].T) < 0.1


def test_a_baseline_short_of_its_sigma_takes_the_camera_height_else_the_frame_s_own_scale(
        an: An, views: dict[str, syn.Shot]) -> None:
    """A 1 cm step known to 5 mm (σ_t/|t| 0.5): ``baseline_short``. With the camera's height the
    outputs take its scale with the motion-stereo shape (``camera_height``, ``baseline_check``
    with no ratio); without one, the frame as one frame (``monocular``)."""
    T = rel(views, "step1")
    doc = analyse(an, views, [frame(T, "step1", 0.005)], height=syn.HEAD_M)
    f = doc["motion"]["frames"][0]
    assert f["status"] == "baseline_short" and f["points"] >= motion.BASELINE_MIN_POINTS
    assert doc["scale_source"] == "camera_height"
    assert doc["baseline_check"] == {"baseline_scale_ratio": None, "sigma": None,
                                     "used": "camera_height"}
    assert doc["scale_check"]["model_height_m"] == pytest.approx(
        analyse(an, views, [], height=syn.HEAD_M)["scale_check"]["model_height_m"])  # the model's
    err = errors(depth_m(doc), views["ref"])
    assert err[0] < 0.05 and err[1] < 0.08, err
    alone = analyse(an, views, [frame(T, "step1", 0.005)])
    assert alone["scale_source"] == "monocular" and "baseline_check" not in alone
    assert alone["motion"]["frames"][0]["status"] == "baseline_short"
    assert np.array_equal(decode(alone["depth"]), decode(analyse(an, views, [])["depth"]))
    # a translation too short for any epipolar geometry: unrefined
    tiny = T.copy()
    tiny[:3, 3] = 1e-5
    f = analyse(an, views, [frame(tiny, "step1", 0.005)])["motion"]["frames"][0]
    assert f == {"index": 0, "status": "baseline_short", "T_ref_frame_refined": None,
                 "sigma_rot_deg_post": None, "rotation_change_deg": None, "points": 0}


def test_frames_without_texture_or_parallax_give_no_overlap_and_too_few_points(
        an: An, views: dict[str, syn.Shot]) -> None:
    """Two frames at once: a wall 12 m away seen 5 cm aside (its points' σ(ln z) about 0.6:
    ``too_few_points``) and a flat grey image (no features: ``no_overlap``, nothing refined);
    neither is an error, and the frame keeps its own scale."""
    doc = analyse(an, views, [frame(rel(views, "far5", "far"), "far5", 0.001),
                              frame(np.eye(4), "grey")], ref="far")
    a, b = doc["motion"]["frames"]
    assert (a["index"], a["status"]) == (0, "too_few_points") and a["points"] < 30
    assert a["T_ref_frame_refined"] is not None
    assert b == {"index": 1, "status": "no_overlap", "T_ref_frame_refined": None,
                 "sigma_rot_deg_post": None, "rotation_change_deg": None, "points": 0}
    assert doc["scale_source"] == "monocular"


# -- baseline check ------------------------------------------------------------------------------


def test_the_baseline_check_keeps_an_agreeing_baseline_and_falls_back_when_they_differ(
        an: An, views: dict[str, syn.Shot]) -> None:
    """With the camera's height: the exact 10 cm step agrees with it (ratio 1 within 3 sigma,
    ``used`` baseline); the step sent as 14 cm puts the baseline scale 1.4 x off, beyond 3
    sigma of the height's, and the outputs take the height's scale with the stereo shape."""
    T = rel(views, "step10")
    doc = analyse(an, views, [frame(T, "step10")], height=syn.HEAD_M)
    bc = doc["baseline_check"]
    assert doc["scale_source"] == "baseline" and bc["used"] == "baseline"
    assert bc["baseline_scale_ratio"] == pytest.approx(1.0, abs=0.03)
    assert bc["sigma"] == pytest.approx(math.hypot(doc["scale_sigma_rel"], 0.02 / syn.HEAD_M),
                                        abs=1e-4)
    long = T.copy()
    long[:3, 3] *= 1.4
    off = analyse(an, views, [frame(long, "step10")], height=syn.HEAD_M)
    bc = off["baseline_check"]
    assert off["motion"]["frames"][0]["status"] == "used"
    assert (off["scale_source"], bc["used"]) == ("camera_height", "camera_height")
    assert bc["baseline_scale_ratio"] == pytest.approx(1 / 1.4, abs=0.03)
    assert abs(math.log(bc["baseline_scale_ratio"])) > 3 * bc["sigma"]
    assert off["scale_sigma_rel"] == pytest.approx(0.02 / syn.HEAD_M, abs=1e-4)
    err = errors(depth_m(off), views["ref"])
    assert max(err) < 0.04, err
    # a height whose floor gives no plausible scale: no ratio, the baseline stays
    high = analyse(an, views, [frame(T, "step10")], height=5.0)
    assert high["baseline_check"] == {"baseline_scale_ratio": None, "sigma": None,
                                      "used": "baseline"}
    assert high["scale_source"] == "baseline"


# -- depth sigma ---------------------------------------------------------------------------------


def test_depth_sigma_with_motion_stereo_and_for_one_frame(an: An, views: dict[str, syn.Shot]) -> None:
    """``depth_sigma``: σ(ln z) x 1000 per pixel of ``depth``, 0 where it is 0. With stereo the
    scale's sigma combined with the correction's local one: never below the scale's, lower
    near the triangulated points than away from them; for one frame the scale's sigma with the
    model's own (hypot(0.10, 0.05))."""
    doc = analyse(an, views, [frame(rel(views, "step10"), "step10")])
    d, s = decode(doc["depth"]), decode(doc["depth_sigma"]).astype(np.float64)
    assert d.shape == s.shape == (syn.H, syn.W)
    assert not s[d == 0].any() and (s[d > 0] > 0).all()
    assert s[d > 0].min() >= round(doc["scale_sigma_rel"] * 1000) - 1
    assert np.percentile(s[d > 0], 95) > np.percentile(s[d > 0], 5) + 5
    one = analyse(an, views, [], outputs=("depth_sigma",))
    assert "depth" not in one
    s1 = decode(one["depth_sigma"])
    want = round(math.hypot(0.10, 0.05) * 1000)
    assert set(np.unique(s1).tolist()) <= {0, want} and (s1 == want).mean() > 0.5
    only = analyse(an, views, [], outputs=("depth",))
    assert "depth_sigma" not in only and np.array_equal((s1 > 0), decode(only["depth"]) > 0)


# -- other frames of the outputs, inputs ---------------------------------------------------------


def test_beside_a_depth_image_and_in_the_map_frame_the_scale_stays_theirs(
        an: An, views: dict[str, syn.Shot]) -> None:
    """A depth image is the depth: the frames are matched and reported, the depth output is the
    sensor's, the scale ``depth`` and no baseline check. In the map frame the outputs keep the
    map's scale, the depth the motion-stereo shape; the camera's mask leaves its pixels out of
    the other frames' matches too."""
    v = views
    T = rel(v, "step10")
    dep = png(np.rint(v["ref"].depth / 0.001).astype(np.uint16))
    doc = analyse(an, v, [frame(T, "step10")], height=syn.HEAD_M, depth="d", _files={"d": dep})
    assert doc["scale_source"] == "depth" and "baseline_check" not in doc
    assert doc["motion"]["frames"][0]["status"] == "used"
    assert np.abs(depth_m(doc) - v["ref"].depth).max() < 0.002
    mask = np.zeros((syn.H, syn.W), np.uint8)
    mask[:, :200] = 255
    an.backend.mono_depth[hashlib.sha256(v["ref"].jpeg).hexdigest()] = syn.model_depth(v["ref"].depth)
    req = {"image": "ref", "camera": {**syn.HUMANOID, "ignore_mask": "m"}, "map": "t",
           "T_map_cam": v["ref"].T.tolist(), "outputs": ["depth"],
           "frames": [frame(T, "step10")]}
    m = an.analyse(req, {"ref": v["ref"].jpeg, "step10": v["step10"].jpeg, "m": png(mask)})
    assert m["frame"] == "map" and m["scale_source"] not in ("baseline", "camera_height")
    assert m["motion"]["frames"][0]["status"] == "used" and "baseline_check" not in m
    assert not decode(m["depth"])[:, :200].any()


def test_frames_are_validated(an: An, views: dict[str, syn.Shot]) -> None:
    v = views
    T = rel(v, "step10")
    one = frame(T, "step10")

    def err(frames: list[dict[str, Any]], **files: bytes) -> dict[str, Any]:
        out: dict[str, Any] = analyse(an, v, frames, status=400, _files=files)["error"]
        assert out["code"] == "invalid_request"
        return out

    assert err([one] * (motion.MAX_MOTION_FRAMES + 1))["details"]["field"] == "frames"
    schema = json.loads(SCHEMA.read_text())["$defs"]["AnalyseRequest"]["properties"]["frames"]
    assert schema["maxItems"] == motion.MAX_MOTION_FRAMES
    assert err([{**one, "image": "nope"}])["details"]["field"] == "frames[0].image"
    small = png(np.zeros((240, 320, 3), np.uint8))
    assert err([one, {**one, "image": "small"}], small=small)["details"]["field"] == \
        "frames[1].image"
    skew = T.copy()
    skew[0, 1] += 0.3
    assert err([{**one, "T_ref_frame": skew.tolist()}])["details"]["field"] == \
        "frames[0].T_ref_frame"
    assert err([{**one, "sigma_pos_m": 0}])["details"]["field"] == "frames[0].sigma_pos_m"
    assert err([{k: x for k, x in one.items() if k != "sigma_rot_deg"}])["details"][
        "field"].startswith("frames[0]")


# -- the parts -----------------------------------------------------------------------------------


def test_the_fit_and_the_correction_on_their_own() -> None:
    """``reconstruction.stereo``: too few points fit nothing; points the depth does not hold
    correct nothing (the spread everywhere); a ln-linear model depth is fitted exactly."""
    z = np.full((48, 64), 2.0, np.float32)
    uv = np.array([[10.0, 10.0], [20.0, 30.0]])
    assert stereo.fit_scale(np.ones(5), np.ones(5), np.zeros(5)) is None
    assert stereo.stereo_depth(z, uv, np.ones(2), np.zeros(2)) is None
    out, local = stereo.sparse_dense(z, np.array([[-5.0, 3.0]]), np.ones(1), np.zeros(1), 0.04)
    assert np.array_equal(out, z) and np.allclose(local, 0.04)
    yy, xx = np.mgrid[0:48, 0:64]
    truth = (1.0 + xx / 64.0 + yy / 96.0).astype(np.float32)
    model = (0.7 * truth ** 0.9).astype(np.float32)
    pts = np.c_[xx.ravel()[::7], yy.ravel()[::7]].astype(np.float64)
    zt = truth.ravel()[::7].astype(np.float64)
    fit = stereo.fit_scale(stereo.sample(model, pts), zt, np.full(len(zt), 0.01))
    assert fit is not None and fit.b == pytest.approx(1 / 0.9 - 1, abs=0.01)
    assert np.allclose(fit.apply(model), truth, rtol=0.01)


def test_the_pair_s_edge_cases(monkeypatch: pytest.MonkeyPatch,
                               views: dict[str, syn.Shot]) -> None:
    """Matches no relative pose explains are ``no_overlap`` once refined; without an essential
    matrix only the sent pose starts the refinement; used pairs whose points the depth does
    not hold give no depth, and without a depth the frames are reported alone."""
    rng = np.random.default_rng(0)
    n = 40
    desc = rng.integers(0, 255, (n, 128)).astype(np.uint8)
    T = np.eye(4)
    T[0, 3] = 0.1
    o = motion.Other(np.zeros((syn.H, syn.W, 3), np.uint8), None, T, 0.002, 1.0)
    p = motion.pair(0, Features(rng.uniform([0, 0], [640, 480], (n, 2)).astype(np.float32), desc),
                    Features(rng.uniform([0, 0], [640, 480], (n, 2)).astype(np.float32),
                             desc.copy()), K, o)
    assert (p.status, p.T_refined) == ("no_overlap", None)
    v = views
    others = [motion.Other(v["step10"].rgb, None, rel(v, "step10"), 0.002, 0.5)]
    zero = np.zeros((syn.H, syn.W), np.float32)
    m = motion.motion_stereo(v["ref"].rgb, None, K, others, zero, K)
    assert m.frames[0].status == "used" and m.stereo is None and not m.qualified
    assert motion.motion_stereo(v["ref"].rgb, None, K, others, None, K).stereo is None
    monkeypatch.setattr(motion.cv2, "findEssentialMat", lambda *a, **k: (None, None))
    assert motion.image_start(np.ones((12, 3)), np.ones((12, 3)), 270.0, T[:3, 3]) is None
    m = motion.motion_stereo(v["ref"].rgb, None, K, others, syn.model_depth(v["ref"].depth), K)
    assert m.frames[0].status == "used" and m.qualified


# -- per-frame ignore_mask and the baseline check's ``used`` ------------------------------------


def test_a_frame_s_own_ignore_mask_hides_its_pixels_from_the_matches(
        an: An, views: dict[str, syn.Shot]) -> None:
    """A frame's ``ignore_mask`` (combined with the camera's) leaves it no features: with the
    whole image ignored the frame is ``no_overlap``, an empty mask changes nothing, and the
    other frame of the request is used as without any mask."""
    T = rel(views, "step10")
    full = png(np.full((syn.H, syn.W), 255, np.uint8))
    none = png(np.zeros((syn.H, syn.W), np.uint8))
    plain = analyse(an, views, [frame(T, "step10")])["motion"]["frames"]
    assert plain[0]["status"] == "used"
    doc = analyse(an, views, [{**frame(T, "step10"), "ignore_mask": "m0"},
                              {**frame(T, "step10"), "ignore_mask": "m1"}],
                  _files={"m0": none, "m1": full})
    a, b = doc["motion"]["frames"]
    assert a["status"] == "used" and a["points"] == plain[0]["points"]
    assert b["status"] == "no_overlap" and b["points"] == 0


def test_a_frame_s_ignore_mask_is_validated(an: An, views: dict[str, syn.Shot]) -> None:
    one = {**frame(rel(views, "step10"), "step10"), "ignore_mask": "m"}
    out = analyse(an, views, [one], status=400)["error"]
    assert out["code"] == "invalid_request" and out["details"]["field"] == "frames[0].ignore_mask"
    wrong = png(np.zeros((10, 12), np.uint8))
    out = analyse(an, views, [one], status=400, _files={"m": wrong})["error"]
    assert out["code"] == "invalid_request" and out["details"]["field"] == "frames[0].ignore_mask"


def test_without_a_pair_or_a_floor_the_baseline_check_says_monocular(
        an: An, views: dict[str, syn.Shot]) -> None:
    """A height is sent but the frame shows no floor to take it on and no pair is used: the
    scale is ``monocular`` and so is the check's ``used``."""
    doc = analyse(an, views, [frame(rel(views, "far5", "far"), "far5", 0.001)], ref="far",
                  height=syn.HEAD_M)
    assert doc["scale_source"] == "monocular"
    assert doc["baseline_check"] == {"baseline_scale_ratio": None, "sigma": None,
                                     "used": "monocular"}
