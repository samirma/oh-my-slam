"""oms/2.2 P13 (round 7): locate's height_sigma from the answer's reference keyframes, the
seeded component's sigma floor across engines and passes, the drift support rule and the
refused first estimate, drift-grown sigmas, and the older mapper's too-close exception."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oh_my_slam.mapping import relocalize as rl
from tests.synth.room import base_pose
from tests.unit.test_service_api import Client
from tests.unit.test_service_nav import nav  # noqa: F401
from tests.unit.test_service_prior_first import client  # noqa: F401

# -- locate's height_sigma (api.md §6, SI-5) -----------------------------------------------------


def _query() -> rl.Query:
    q = rl.Query(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8), np.eye(3), 640,
                 480)
    q.camera_z, q.camera_z_sigma, q.height_scale = 0.5, 0.01, True
    return q


def test_height_sigma_is_the_frame_s_else_the_references_else_the_default() -> None:
    """api.md §6 (P15), in order: 1. the frame's own height_sigma_m; 2. else the median
    height_sigma_m the answer's reference keyframes carried; 3. else the median of their
    sigma_pos_m / sqrt(3); 4. else the README default."""
    q = _query()
    q.camera_z_sigma_given = True  # 1. the frame's own, whatever the references carry
    assert rl.z_sigma(q, [0.1, 0.3], [0.3, 0.6]) == 0.01
    q.camera_z_sigma_given = False
    assert rl.z_sigma(q, [0.1, None, 0.3, 0.2], [3.0] * 4) == pytest.approx(0.2)  # 2. median
    r3 = math.sqrt(3.0)
    assert rl.z_sigma(q, [None, None], [0.3 * r3, 0.1 * r3, 0.2 * r3]) == pytest.approx(0.2)
    assert rl.z_sigma(q, [None], [0.0]) == 0.01  # 4. no sigma at all: the default
    assert rl.z_sigma(q, []) == 0.01
    assert rl.DEFAULT_HEIGHT_SIGMA_M == 0.02  # the README default
    T = base_pose(0, height=0.9)  # 0.4 m above the expected height
    assert rl.height_consistent(q, T, 0.01) is False
    assert rl.height_consistent(q, T, 0.01, 0.2) is True  # the references' sigma
    kf = rl.Keyframe("k", np.eye(4), np.eye(3), np.zeros((0, 2), np.float32),
                     np.zeros((0, 128), np.uint8), np.zeros((0, 3), np.float32),
                     height_sigma_m=0.3)
    idx = rl.Index([kf, rl.Keyframe("j", np.eye(4), np.eye(3), np.zeros((0, 2), np.float32),
                                    np.zeros((0, 128), np.uint8),
                                    np.zeros((0, 3), np.float32))])
    assert rl._ref_z_sigma(q, np.array([0, 0, 1]), idx) == 0.3
    assert rl._ref_z_sigma(q, np.array([1]), idx) == 0.01
    idx.keyframes[1].sigma_pos_m = 0.3 * math.sqrt(3.0)  # 3. its registered pose sigma
    assert rl._ref_z_sigma(q, np.array([1]), idx) == pytest.approx(0.3)
    # ranking takes each hypothesis's own height_sigma
    low, high = base_pose(0, height=0.5), base_pose(0, 1.0, height=0.9)
    assert rl.choose(q, [high, low], [100, 60], [(0.01, 0.5)] * 2) == 1
    assert rl.choose(q, [high, low], [100, 60], [(0.01, 0.5)] * 2, [0.3, 0.01]) == 0


def test_locate_reports_with_the_references_height_sigma(
        client: Client, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """The version's frames record the camera height sigma they carried (the index's
    keyframes take it), and a locate frame without its own sigma is tested with the one its
    answer's references give."""
    from tests.fakes.service import multipart
    from tests.unit.test_service_prior_first import CAM, frames_for, shot

    cam = {**CAM, "height_above_floor_m": 0.5, "height_sigma_m": 0.01}
    fr, files = frames_for(client, [base_pose(15.0 * k) for k in range(4)], tag="a",
                           depth=False, trust=None)
    job = client.update("m", fr, files, cameras={"cam": cam})
    assert job["state"] == "committed" and job["scale_source"] == "camera_height"
    seen: list[Any] = []
    real = rl.locate

    def spy(q: rl.Query, index: rl.Index) -> rl.Outcome:
        seen.append(([k.height_sigma_m for k in index.keyframes], q.camera_z_sigma_given))
        o = real(q, index)
        seen.append(o.height_sigma_m)
        return o

    monkeypatch.setattr(rl, "locate", spy)
    img, _, _ = shot(base_pose(20))
    data, ctype = multipart({"frames": [{"name": "q", "image": "q", "camera_id": "cam",
                                         "height_above_floor_m": 0.5}],
                             "record": False}, {"q": img})
    r = client.http.post("/api/maps/m/locate", content=data, headers={"content-type": ctype})
    assert r.status_code == 200, r.text
    assert client.check(r, "Locate")["frames"][0]["status"] == "located"
    sig, given = seen[0]
    assert sig and all(s == 0.01 for s in sig) and given is False
    assert seen[1] == 0.01


# -- drift (map.md §1, SI-1, SI-2, SI-7) ---------------------------------------------------------


def _drifting(name: str, T: np.ndarray, P: np.ndarray, index: int, inliers: int) -> Any:
    from types import SimpleNamespace

    from oh_my_slam.service.backend import InternalFrame

    return SimpleNamespace(
        work=SimpleNamespace(name=name, index=index, trust="drifting",
                             prior={"trust": "drifting", "T_map_cam": P.tolist(),
                                    "sigma_pos_m": 0.02, "sigma_rot_deg": 0.5}),
        internal=InternalFrame(name, name, T, 160.0, {"observations": inliers}),
        prior_residual=None)


def test_the_older_mapper_s_drift_grows_away_from_its_support() -> None:
    """Every engine (P13): a kept drifting frame's sigmas are the root sum of squares of its
    prior's and the drift estimate's own at its pose, which grows with the distance to the
    frames that supported it; a refused first estimate leaves the drift at the identity."""
    from oh_my_slam.mapping import anchored as an
    from oh_my_slam.mapping.anchor import Sim3
    from oh_my_slam.service.update import Updater, drift_grown

    P = base_pose(0)
    assert drift_grown(0.02, 0.5, P, None, None) == (0.02, 0.5)  # no estimate yet
    near = drift_grown(0.02, 0.5, P, (0.01, 0.2), [P[:3, 3].tolist()])
    assert near == pytest.approx((np.hypot(0.02, 0.01), np.hypot(0.5, 0.2)))
    far_c = (P[:3, 3] + [3.0, 4.0, 0.0]).tolist()
    far = drift_grown(0.02, 0.5, P, (0.01, 0.2), [far_c])
    assert far[0] == pytest.approx(math.hypot(0.02, 0.01, an.DRIFT_POS_REL * 5.0))
    assert far[1] == pytest.approx(math.hypot(0.5, 0.2, an.DRIFT_ROT_DEG_PER_M * 5.0))
    # the kept frame takes it, mapped through the estimate
    fr = _drifting("k", P, P, 0, 0)
    fr.pose = fr.sigma_pos_m = fr.sigma_rot_deg = None
    Updater._keep_at_prior(fr, Sim3(), (0.01, 0.2), [far_c])  # type: ignore[arg-type]
    assert (fr.sigma_pos_m, fr.sigma_rot_deg) == (round(far[0], 4), round(far[1], 3))
    # a first estimate: three supporting frames 0.5 m off their priors
    shift = np.eye(4)
    shift[0, 3] = 0.5
    poses = [base_pose(20 * k) for k in range(3)]
    frames = [_drifting(f"d{k}", T, shift @ T, k, 500) for k, T in enumerate(poses)]
    sim, sig, support = Updater._drifting_residuals(frames, Sim3(), None)  # type: ignore[arg-type]
    assert sim is not None and sig is not None and support is not None and len(support) == 3
    assert all(f.prior_residual[0] < 0.01 for f in frames)
    # refused (no frame can support it): the identity, priors as sent with their sigmas;
    # the frames are tested under it (0.5 m off: beyond the bound)
    weak = [_drifting(f"w{k}", T, shift @ T, k, 10) for k, T in enumerate(poses)]
    assert Updater._drifting_residuals(weak, Sim3(), None) == (  # type: ignore[arg-type]
        None, None, None)
    assert all(f.prior_residual[0] == pytest.approx(0.5, abs=1e-6) for f in weak)
    # the previous version's estimate, support and sigma stay without drifting frames
    prev = {"drift_sim": Sim3().to_dict(), "drift_sigma": [0.01, 0.2], "drift_support": support}
    got = Updater._drifting_residuals([], Sim3(), prev)  # type: ignore[arg-type]
    assert got[1] == (0.01, 0.2) and got[2] == support


# -- a component's seed (map.md §2 step 2; live check 3) -----------------------------------------


def test_a_seed_takes_its_image_pose_once_frames_registered_through_it(tmp_path: Path) -> None:
    """A seed is registered at its prior only while its images give it no pose: once frames
    registered (here the map's own keyframes are its neighbours), it is placed again and takes
    its image pose, with a prior residual (its prior 3 deg off), and the component links."""
    from oh_my_slam.mapping import anchored as an
    from tests.unit.test_anchored import frame, new_map, prior

    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0)), frame("a1", base_pose(30))])
    T = base_pose(15)
    P = T.copy()
    P[:3, :3] = an.exp_so3(np.array([0.0, 0.0, np.radians(3.0)])) @ T[:3, :3]
    f = frame("s", T, p=prior(P, "metric", 0.05, 5.0))
    pr = m._prepare(f)
    seed = m._seed(f)
    kf = m._keep(f, pr, seed, lambda _: None)
    kf.seed, kf.seed_sigma = kf.id, (0.05, 5.0)
    member = an.Keyframe("k9", "c", "cam", f.K, base_pose(16), 0.05, 5.0, "fused", None, "none",
                         seed=kf.id, seed_sigma=(0.05, 5.0))
    m.keyframes.append(member)
    pl = m._unseed(f, pr, seed, kf, [])
    assert pl.status == "registered" and pl.method != "prior" and pl.keyframe == kf.id
    assert pl.prior_residual is not None and pl.prior_residual[1] > 1.0
    assert an.rot_deg(kf.T[:3, :3].T @ T[:3, :3]) < 1.0  # the image pose, not the prior
    assert kf.seed is None and member.seed is None  # linked to the map
    # round 8: its image pose checked against the job's others and its scale evidence kept;
    # too close (with the inliers to stay registered), no scale evidence
    assert "s" in m._evidence_from
    m._evidence_from.clear()
    import dataclasses
    close = dataclasses.replace(pr, close=True)
    an_inliers, an.TOO_CLOSE_INLIERS = an.TOO_CLOSE_INLIERS, 0
    try:
        assert m._unseed(f, close, seed, kf, []).status == "registered"
    finally:
        an.TOO_CLOSE_INLIERS = an_inliers
    assert "s" not in m._evidence_from
    # with nothing else to register on, it stays the seed
    m2 = new_map(tmp_path / "alone")
    m2.update([frame("a0", base_pose(0))])
    g = frame("t", base_pose(150), p=prior(base_pose(150)))
    pr2 = m2._prepare(g)
    seed2 = m2._seed(g)
    kf2 = m2._keep(g, pr2, seed2, lambda _: None)
    assert m2._unseed(g, pr2, seed2, kf2, []) is seed2  # no candidate at all
    m3 = new_map(tmp_path / "apart")
    m3.update([frame("a0", base_pose(0))])
    h = frame("u", base_pose(90), p=prior(base_pose(90)))
    pr3 = m3._prepare(h)
    seed3 = m3._seed(h)
    kf3 = m3._keep(h, pr3, seed3, lambda _: None)
    assert "a0" in [k.name for k in m3.candidates_for(h, None, m3._predict(h))]
    assert m3._unseed(h, pr3, seed3, kf3, []) is seed3  # a candidate its images do not place on


# -- the too-close exception on the older mapper (api.md §4, SI-8) -------------------------------


def test_the_older_mapper_s_too_close_kept_frame_gives_its_depth(tmp_path: Path) -> None:
    """Every engine (P13): on the older mapper, a frame kept only because it is too close, with
    a metric prior and a depth image, is marked content and its depth image is committed with
    the version (``kept_depth.json``), read back for occupancy and heights."""
    from types import SimpleNamespace

    from oh_my_slam.service.derived import Derived
    from oh_my_slam.service.store import Store
    from oh_my_slam.service.update import KEPT_DEPTH, kept_depth

    z = np.full((24, 32), 0.15, np.float32)
    near = SimpleNamespace(unconfirmed="too_close", depth_m=z, detail=None,
                           work=SimpleNamespace(trust="metric", name="near"))
    far = SimpleNamespace(unconfirmed="too_close", depth_m=None, detail=None,
                          work=SimpleNamespace(trust="metric", name="far"))
    drift = SimpleNamespace(unconfirmed="too_close", depth_m=z, detail=None,
                            work=SimpleNamespace(trust="drifting", name="d"))
    got = kept_depth([near, far, drift])  # type: ignore[list-item]
    assert list(got) == ["near"]
    store = Store(tmp_path / "data")
    store.create()
    (store.dir("m") / "versions").mkdir(parents=True)
    store.version_file("m", 2, KEPT_DEPTH).write_text(__import__("json").dumps(got))
    d = Derived.__new__(Derived)
    d.store = store
    back = d._kept_depth("m", 2, "near")
    assert back is not None and np.allclose(back, 0.15)
    assert d._kept_depth("m", 2, "other") is None and d._kept_depth("m", 3, "near") is None
    assert d._kept_depth("m", 3, "x", {"x": "bm90IGEgcG5n"}) is None  # not a PNG


def test_an_older_mapper_kept_too_close_frame_feeds_the_grid(nav: Any) -> None:  # noqa: F811
    """End to end on the older mapper: the kept too-close frame's row is content, its depth is
    committed with the version, and the version's grid derives from it."""
    from tests.synth.room import png_depth
    from tests.unit.test_round6 import _metric
    from tests.unit.test_service_api import CAMERA, WORLD, frames_of, textured

    frames, files, _ = frames_of(nav, 3, prior=_metric)
    assert nav.update("m", frames, files, {"cam": CAMERA})["state"] == "committed"
    h, w = int(CAMERA["height"]), int(CAMERA["width"])
    files = {"near": textured(91), "neard": png_depth(np.full((h, w), 0.2, np.float32))}
    frames = [{"name": "near", "image": "near", "depth": "neard", "camera_id": "cam",
               "prior": _metric(0, WORLD[4])}]
    job = nav.update("m", frames, files)
    assert job["state"] == "committed", job["quality"]
    assert (job["inputs"][0]["status"], job["inputs"][0]["reason"]) == ("kept_at_prior",
                                                                        "too_close")
    rec = nav.service.store.version("m")
    row = next(r for r in rec["frames"] if r["name"] == "near")
    assert row["content"] is True
    assert nav.service.store.version_file("m", 2, "kept_depth.json").exists()
    from oh_my_slam.mapping import navgrid
    r = nav.http.get("/api/maps/m/grid")
    assert r.status_code == 200, r.text

    vol = navgrid.Volume.from_bytes(
        (nav.service.derived.dir("m") / "2" / "volume.npz").read_bytes())
    assert "near" in vol.frames  # its depth integrated at its prior
