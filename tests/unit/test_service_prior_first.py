"""The prior-first update path of ``server.sh`` (``service.prior_first``) end to end through the
Starlette app, on frames ray-cast from a textured room (``tests.synth.room``): a map started with
priors is anchored, keeps registered and provisional frames, reports quality and scale, and
commits only what passes the gate; a job whose frames are all rejected costs only the checks."""

from __future__ import annotations

import functools
import hashlib
import io
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from starlette.testclient import TestClient

from oh_my_slam.service.app import CAPABILITIES, Service, create_app
from oh_my_slam.service.jobs import Jobs
from oh_my_slam.service.store import Store
from oh_my_slam.service.timings import Timings
from oh_my_slam.service.update import Updater
from tests.fakes.service import FakeBackend, FakeInference, plain
from tests.synth.room import Room, base_pose, jpeg, png_depth, render
from tests.unit.test_service_api import Client

CAM = {"fx": 160.0, "fy": 160.0, "cx": 160.0, "cy": 120.0, "width": 320, "height": 240}
ROOM = Room()


@functools.lru_cache(maxsize=128)
def _shot(key: tuple[float, ...]) -> tuple[bytes, bytes, bytes]:
    from oh_my_slam.core.types import Intrinsics

    T = np.array(key).reshape(4, 4)
    rgb, z = render(ROOM, T, Intrinsics(160.0, 160.0, 160.0, 120.0, 320, 240))
    buf = io.BytesIO()
    np.save(buf, z)
    return jpeg(rgb), png_depth(z), buf.getvalue()


def shot(T: np.ndarray) -> tuple[bytes, bytes, np.ndarray]:
    img, dep, z = _shot(tuple(np.round(T, 9).ravel().tolist()))
    return img, dep, np.load(io.BytesIO(z))


@pytest.fixture(autouse=True)
def geometric(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests run the prior-first engine on every frame (``OH_MY_SLAM_NONE_ENGINE``); the
    MapAnything engine of camera-only frames has its own tests (``test_service_register_ma``)."""
    monkeypatch.setenv("OH_MY_SLAM_NONE_ENGINE", "geometric")


@pytest.fixture
def client(tmp_path: Path) -> Iterator[Client]:
    store = Store(tmp_path / "data")
    store.create()
    backend = FakeBackend()
    timings = Timings()
    updater = Updater(store, backend, timings)
    jobs = Jobs(store.jobs, updater, timings.predict_update)
    jobs.start()
    from oh_my_slam.service.derived import Derived

    derived = Derived(store, backend.read)
    derived.start()
    service = Service(store, jobs, FakeInference(), backend.require_inference, timings=timings,
                      version="test", dry_run=updater.dry_run, derived=derived)
    with TestClient(create_app(service)) as http:
        yield Client(http, backend, service)
    jobs.stop(timeout=10)
    derived.stop()


def prior(T: np.ndarray, trust: str = "metric", sp: float = 0.02, sr: float = 1.0
          ) -> dict[str, Any]:
    return {"trust": trust, "T_map_cam": np.asarray(T).tolist(), "sigma_pos_m": sp,
            "sigma_rot_deg": sr}


def frames_for(c: Client, poses: list[np.ndarray], *, tag: str = "f", depth: bool = True,
               trust: str | None = "metric", priors: list[np.ndarray] | None = None,
               **extra: Any) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    frames, files = [], {}
    for i, T in enumerate(poses):
        img, dep, z = shot(T)
        c.backend.mono_depth[hashlib.sha256(img).hexdigest()] = z
        files[f"{tag}{i}"] = img
        f: dict[str, Any] = {"name": f"{tag}{i}", "image": f"{tag}{i}", "camera_id": "cam",
                             **extra}
        if trust is not None:
            f["prior"] = prior((priors or poses)[i], trust)
        if depth:
            files[f"{tag}{i}d"] = dep
            f["depth"] = f"{tag}{i}d"
        frames.append(f)
    return frames, files


def test_health_lists_o3_jobs(client: Client) -> None:
    assert CAPABILITIES == ["O1", "O3-jobs", "O4", "O5", "O6", "O8", "O9", "O10"]
    assert client.get("/api/health", "Health")["capabilities"] == CAPABILITIES


def test_cold_start_then_a_sweep(client: Client) -> None:
    fr, files = frames_for(client, [base_pose(0)], tag="a")
    job = client.update("m", fr, files, cameras={"cam": CAM})
    assert job["state"] == "committed" and job["result_version"] == 1
    assert job["inputs"][0]["status"] == "registered"
    assert job["inputs"][0]["prior_residual_m"] == 0.0
    q = job["quality"]
    assert q["gauge_shift"] is None and q["scale_ratio_vs_prev"] is None
    assert q["prior_gravity_deg"] == pytest.approx(10.0, abs=0.1)  # the base camera's pitch
    assert job["scale_source"] == "depth" and [g for _, g in client.backend.infers] == [True]
    poses = [base_pose(15.0 + 30.0 * k) for k in range(6)]
    fr, files = frames_for(client, poses, tag="s")
    job = client.update("m", fr, files)
    assert job["state"] == "committed" and job["result_version"] == 2
    assert [i["status"] for i in job["inputs"]] == ["registered"] * 6
    for i, T in zip(job["inputs"], poses, strict=True):
        assert np.linalg.norm(np.array(i["T_map_cam"])[:3, 3] - T[:3, 3]) < 0.03
        assert i["inliers"] > 50 and i["sigma_pos_m"] >= 0.01
    q = job["quality"]
    # the new frames against the stored map: small for a consistent update
    assert q["gauge_shift"]["translation_m"] < 0.03 and q["gauge_shift"]["scale_ratio"] == 1.0
    assert q["previous_residuals"]["count"] == 6 and q["scale_ratio_vs_prev"] == 1.0
    assert q["prior_residuals"]["count"] == 7 and q["rejected_share"] == 0.0
    assert q["rebuilt"] is False and q["accepted"]
    assert set(job["timings_s"]) >= {"checks", "clone", "register", "fusion", "commit", "total"}
    frames = client.get("/api/maps/m/frames", "Frames")
    assert len(frames["frames"]) == 7 and {f["pose_state"] for f in frames["frames"]} == {"fused"}
    summary = client.get("/api/maps/m", "MapSummary")
    assert summary["floor_z_m"] is not None and abs(summary["floor_z_m"]) < 0.05
    v = client.service.store.version("m")
    assert v["engine"] == "anchored" and v["sim3"]["s"] == 1.0
    assert client.backend.calls == []  # the mapper never ran
    assert len(client.backend.infers) == 1  # depth images: no monocular depth asked


def test_monocular_depth_with_camera_height(client: Client) -> None:
    client.backend.mono_scale = 1.25
    cam = {**CAM, "height_above_floor_m": 0.5, "height_sigma_m": 0.01}
    fr, files = frames_for(client, [base_pose(0)], tag="a", depth=False)
    job = client.update("m", fr, files, cameras={"cam": cam})
    assert job["state"] == "committed" and job["scale_source"] == "camera_height"
    assert job["quality"]["prior_gravity_deg"] is not None
    assert [g for _, g in client.backend.infers] == [True]
    fr, files = frames_for(client, [base_pose(15.0 + 30.0 * k) for k in range(4)], tag="s",
                           depth=False, height_above_floor_m=0.5)
    job = client.update("m", fr, files)
    assert [i["status"] for i in job["inputs"]] == ["registered"] * 4
    assert [g for _, g in client.backend.infers][1:] == [False] * 4
    assert job["quality"]["scale_ratio_vs_prev"] is not None
    assert abs(job["quality"]["scale_ratio_vs_prev"] - 1.0) < 0.1
    assert client.service.store.version("m")["scale_source"] == "camera_height"


def test_a_job_whose_frames_are_all_rejected_is_cheap(client: Client) -> None:
    blank = np.full((240, 320, 3), 128, np.uint8)
    blank[:, :2] = 0  # a different image, as bare
    files = {"p0": plain(320, 240), "p1": jpeg(blank)}
    frames = [{"name": f"p{i}", "image": f"p{i}", "camera_id": "cam",
               "prior": prior(base_pose(10 * i))} for i in range(2)]
    t = time.perf_counter()
    job = client.update("m", frames, files, cameras={"cam": CAM})
    assert time.perf_counter() - t < 5.0
    assert job["state"] == "rejected" and job["quality"]["failed_checks"] == ["hard_limit"]
    assert [i["reason"] for i in job["inputs"]] == ["plain", "plain"]
    assert job["scale_source"] is None and client.backend.infers == []
    client.get("/api/maps/m", status=404)


def test_provisional_rejected_and_the_gate(client: Client) -> None:
    fr, files = frames_for(client, [base_pose(0), base_pose(35)], tag="a")
    client.update("m", fr, files, cameras={"cam": CAM})
    T = base_pose(20)
    off = T.copy()
    off[:3, 3] += [0.5, 0.0, 0.0]
    fr, files = frames_for(client, [base_pose(180), T], tag="b", priors=[base_pose(180), off])
    fr.append({"name": "a0", "image": "b0", "camera_id": "cam"})  # a duplicate name
    job = client.update("m", fr, files)
    assert [(i["status"], i["reason"]) for i in job["inputs"]] == [
        ("provisional", "no_overlap"), ("rejected", "prior_conflict"), ("rejected", "duplicate")]
    assert job["inputs"][0]["T_map_cam"] is not None
    assert job["inputs"][1]["T_map_cam"] is None and job["inputs"][1]["prior_residual_m"] > 0.3
    q = job["quality"]
    assert q["rejected_share"] == round(2 / 3, 4) and q["low_confidence_share"] == 1.0
    frames = client.get("/api/maps/m/frames", "Frames")["frames"]
    assert [f["pose_state"] for f in frames] == ["fused", "fused", "provisional"]
    summary = client.get("/api/maps/m", "MapSummary")
    assert summary["provisional_frames"] == 1
    # a gate that fails commits nothing
    fr, files = frames_for(client, [base_pose(10), T], tag="c", priors=[base_pose(10), off])
    job = client.update("m", fr, files, accept_if={"max_rejected_share": 0.2})
    assert job["state"] == "rejected" and job["quality"]["failed_checks"] == [
        "max_rejected_share"]
    assert job["result_version"] is None and client.service.store.latest("m") == 2
    # the derived reads skip the provisional frame as a locate reference
    grid = client.get("/api/maps/m/grid?cell_m=0.1", "Grid")
    assert grid["map_version"] == 2
    assert client.service.derived is not None
    dv = client.service.derived.get("m", 2)
    assert "b0" not in dv.meta["keyframes"] and "a0" in dv.meta["keyframes"]


def test_drifting_map_and_frames_without_prior(client: Client) -> None:
    truth = [base_pose(0, -1.0 + 0.2 * k, 0.0) for k in range(5)]
    odo = [T.copy() for T in truth]
    for P in odo:
        P[0, 3] = (P[0, 3] + 1.0) * 1.08 - 1.0
    fr, files = frames_for(client, truth[:1], tag="a", trust="drifting", priors=odo[:1])
    job = client.update("m", fr, files, cameras={"cam": CAM})
    assert job["state"] == "committed" and job["quality"]["prior_residuals"] is None
    fr, files = frames_for(client, truth[1:], tag="b", trust="drifting", priors=odo[1:])
    nf, nfiles = frames_for(client, [base_pose(0, -0.5, 0.0)], tag="n", trust=None)
    nf[0]["prior"] = {"trust": "none"}
    job = client.update("m", fr + nf, {**files, **nfiles})
    assert [i["status"] for i in job["inputs"]] == ["registered"] * 5
    assert job["inputs"][-1]["prior_residual_m"] is None
    last = np.array(job["inputs"][3]["T_map_cam"])
    assert np.linalg.norm(last[:3, 3] - truth[4][:3, 3]) < 0.03


def test_ignore_mask_and_new_map_without_priors(client: Client) -> None:
    mask = np.zeros((240, 320), np.uint8)
    mask[200:, :] = 255
    buf = io.BytesIO()
    Image.fromarray(mask).save(buf, "PNG")
    fr, files = frames_for(client, [base_pose(0), base_pose(20)], tag="a")
    fr[0]["ignore_mask"] = "m0"
    job = client.update("m", fr, {**files, "m0": buf.getvalue()}, cameras={"cam": CAM})
    assert [i["status"] for i in job["inputs"]] == ["registered", "registered"]
    # a new map without priors is anchored too: its first frame is the origin, level by its
    # gravity, and a turn in place registers on the images alone
    fr, files = frames_for(client, [base_pose(0), base_pose(20), base_pose(40)], tag="z",
                           trust=None, depth=False)
    job = client.update("other", fr, files, cameras={"cam": CAM})
    assert job["state"] == "committed" and not client.backend.calls
    assert [i["status"] for i in job["inputs"]] == ["registered"] * 3
    assert job["inputs"][0]["prior_residual_m"] is None
    v = client.service.store.version("other")
    assert v["engine"] == "anchored" and np.allclose(np.array(v["frames"][0]["T_map_cam"])[:3, 3], 0)


def test_inference_failure_fails_the_job(client: Client) -> None:
    client.backend.infer_error = True
    fr, files = frames_for(client, [base_pose(0)], tag="a", depth=False)
    job = client.update("m", fr, files, cameras={"cam": CAM})
    assert job["state"] == "failed" and job["error"]["code"] == "inference_unavailable"
    client.get("/api/maps/m", status=404)


def test_an_all_rejected_job_on_an_anchored_map(client: Client) -> None:
    fr, files = frames_for(client, [base_pose(0)], tag="a")
    client.update("m", fr, files, cameras={"cam": CAM})
    job = client.update("m", [{"name": "p", "image": "p", "camera_id": "cam",
                               "prior": prior(base_pose(5))}], {"p": plain(320, 240)})
    assert job["state"] == "rejected" and job["quality"]["failed_checks"] == ["hard_limit"]
    assert job["quality"]["scale_ratio_vs_prev"] is None and job["scale_source"] is None
    assert client.service.store.latest("m") == 1


def dry(c: Client, name: str, frames: list[dict[str, Any]], files: dict[str, bytes],
        **extra: Any) -> dict[str, Any]:
    from tests.fakes.service import multipart

    data, ctype = multipart({"frames": frames, **extra}, files)
    r = c.http.post(f"/api/maps/{name}/updates?dry_run=1", content=data,
                    headers={"content-type": ctype})
    assert r.status_code == 200, r.text
    out: dict[str, Any] = c.check(r, "DryRun")
    return out


def test_dry_run_of_a_new_map_writes_nothing(client: Client) -> None:
    mask = np.zeros((240, 320), np.uint8)
    mask[220:, :] = 255
    buf = io.BytesIO()
    Image.fromarray(mask).save(buf, "PNG")
    fr, files = frames_for(client, [base_pose(0)], tag="a")
    nf, nfiles = frames_for(client, [base_pose(15), base_pose(180)], tag="n", trust=None)
    files.update(nfiles)
    files["p"] = plain(320, 240)
    frames = fr + nf + [{"name": "p", "image": "p", "camera_id": "cam"}]
    out = dry(client, "m", frames, {**files, "mk": buf.getvalue()},
              cameras={"cam": {**CAM, "ignore_mask": "mk"}})
    rows = {r["name"]: r for r in out["frames"]}
    assert out["base_version"] is None and all(r["map_inliers"] is None for r in out["frames"])
    assert rows["a0"]["predicted_status"] == "registered" and rows["a0"]["features"] > 100
    assert rows["n0"]["predicted_status"] == "registered"
    assert rows["n0"]["neighbour_inliers"][0]["inliers"] >= 15
    assert (rows["n1"]["predicted_status"], rows["n1"]["predicted_reason"]) == (
        "rejected", "no_overlap")
    assert rows["p"]["plain"] and rows["p"]["predicted_reason"] == "plain"
    assert out["quality"]["rejected_share"] == 0.5 and out["quality"]["accepted"]
    assert out["predicted_s"]["registration"] > 0
    client.get("/api/maps/m", status=404)
    assert client.service.store.meta("m") is None  # not even the camera


def test_dry_run_against_an_anchored_map(client: Client) -> None:
    fr, files = frames_for(client, [base_pose(0), base_pose(35)], tag="a")
    client.update("m", fr, files, cameras={"cam": CAM})
    T = base_pose(20)
    off = T.copy()
    off[:3, 3] += [0.5, 0.0, 0.0]
    fr, files = frames_for(client, [base_pose(10), T, base_pose(180)], tag="b",
                           priors=[base_pose(10), off, base_pose(180)])
    fr.append({"name": "a0", "image": "b0", "camera_id": "cam"})
    out = dry(client, "m", fr, files, accept_if={"max_rejected_share": 0.1})
    rows = [(r["predicted_status"], r["predicted_reason"]) for r in out["frames"]]
    assert rows == [("registered", None), ("rejected", "prior_conflict"), ("registered", None),
                    ("rejected", "duplicate")]
    assert out["frames"][0]["map_inliers"] > 50 and out["base_version"] == 1
    q = out["quality"]
    assert not q["accepted"] and q["failed_checks"] == ["max_rejected_share"]
    assert q["prior_residuals"]["count"] == 2 and q["low_confidence_share"] == 0.5
    assert client.service.store.latest("m") == 1  # nothing committed


def _slip(T: np.ndarray, deg: float) -> np.ndarray:
    a = np.radians(deg)
    Rz = np.eye(4)
    Rz[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    return Rz @ T


def test_a_re_sweep_after_an_odometry_slip(client: Client) -> None:
    """Strong placements that agree on one correction of a drifting odometry re-anchor it in
    the dry run as in the update; the dry run predicts along the request."""
    poses = [base_pose(30 * k) for k in range(4)]
    fr, files = frames_for(client, poses, tag="a", trust="drifting")
    client.update("m", fr, files, cameras={"cam": CAM})
    truth = [base_pose(y) for y in (70, 80, 65, 50)]
    slips = [-30.0, 25.0, 25.0, 25.0]  # the first slipped otherwise: a conflict of its own
    fr, files = frames_for(client, truth, tag="s", trust="drifting",
                           priors=[_slip(T, d) for T, d in zip(truth, slips, strict=True)])
    out = dry(client, "m", fr, files)
    want = ["rejected"] + ["registered"] * 3
    assert [r["predicted_status"] for r in out["frames"]] == want
    assert out["frames"][0]["predicted_reason"] == "prior_conflict"
    job = client.update("m", fr, files)
    assert [i["status"] for i in job["inputs"]] == want


def test_dry_run_measures_suspicious_frames(client: Client) -> None:
    import cv2

    img = np.full((240, 320, 3), 128, np.uint8)
    cv2.rectangle(img, (40, 40), (280, 200), (30, 30, 30), 3)
    client.backend.mono = 0.2
    out = dry(client, "m", [{"name": "s", "image": "s", "camera_id": "cam"}],
              {"s": jpeg(img)}, cameras={"cam": CAM})
    assert out["frames"][0]["too_close"] and out["quality"]["failed_checks"] == ["hard_limit"]


def test_neighbour_links(monkeypatch: pytest.MonkeyPatch) -> None:
    import cv2

    from oh_my_slam.mapping.anchored import features
    from oh_my_slam.service import dry_run as dr

    img, _, _ = shot(base_pose(0))
    rgb = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    f = features(rgb)
    assert dr.link(f, f) > 100
    assert dr.link(f, features(np.full_like(rgb, 128))) == 0
    monkeypatch.setattr(cv2, "findFundamentalMat", lambda *a, **k: (None, None))
    assert dr.link(f, f) == 0


def test_grid_points_and_locate_read_an_anchored_map(client: Client) -> None:
    fr, files = frames_for(client, [base_pose(30.0 * k) for k in range(6)], tag="a")
    job = client.update("m", fr, files, cameras={"cam": CAM})
    assert job["state"] == "committed"
    deadline = time.monotonic() + 60
    while True:
        r = client.http.get("/api/maps/m/grid?cell_m=0.1")
        if r.status_code == 200 or time.monotonic() > deadline:
            break
        time.sleep(0.5)
    grid = client.check(r, "Grid")
    assert grid["map_version"] == 1 and grid["shape"][0] > 5
    import base64

    lay = grid["layers"]["state"]
    state = np.frombuffer(base64.b64decode(lay["data"]), np.uint8).reshape(grid["shape"])
    ox, oy = grid["origin_m"]
    for T in (base_pose(0), base_pose(90)):  # the cameras' cells are free
        c, rr = int((T[0, 3] - ox) / 0.1), int((T[1, 3] - oy) / 0.1)
        assert state[rr, c] == 1, (rr, c, state[max(rr - 1, 0):rr + 2, max(c - 1, 0):c + 2])
    assert abs(grid["floor_z_m"]) < 0.05, grid["floor_z_m"]
    pts = client.get("/api/maps/m/points?min=-3,-3,-1&max=3,3,3", "Points")
    assert pts["count"] > 1000 if "count" in pts else True
    from oh_my_slam.service.backend import read_anchored

    im = read_anchored(client.service.store.mapper_dir("m"))
    f0 = im.frames[0].name
    d = im.depth(f0)
    assert d is not None and d.shape == (240, 320) and im.image(f0) is not None
    c = im.cloud()
    assert c is not None and len(c.xyz) > 1000 and im.objects() == []
    img, _, _ = shot(base_pose(45))
    from tests.fakes.service import multipart

    data, ctype = multipart({"frames": [{"name": "q", "image": "q", "camera_id": "cam"}],
                             "record": False}, {"q": img})
    r = client.http.post("/api/maps/m/locate", content=data, headers={"content-type": ctype})
    assert r.status_code == 200, r.text


def test_a_keyframe_without_its_image_is_no_locate_reference(client: Client,
                                                            tmp_path: Path) -> None:
    fr, files = frames_for(client, [base_pose(0), base_pose(30)], tag="a")
    client.update("m", fr, files, cameras={"cam": CAM})
    store = client.service.store
    v = store.version("m")
    (store.mapper_dir("m") / "kf" / f"{v['frames'][1]['internal']}.jpg").unlink()
    assert client.service.derived is not None
    client.service.derived._derive("m", 1, v, tmp_path / "d")
    meta = json.loads((tmp_path / "d" / "meta.json").read_text())
    assert meta["keyframes"] == ["a0"]


def test_bare_walls_with_depth_and_a_drifting_prior_register(client: Client) -> None:
    """A bare wall (plain, no features) with a depth image and a prior is not rejected by the
    checks: its depth registers it on the map's (prior plus depth geometry)."""
    poses = [base_pose(26 * k) for k in range(3)]
    fr, files = frames_for(client, poses, tag="a", trust="drifting")
    client.update("m", fr, files, cameras={"cam": CAM})
    T = base_pose(78)
    fr, files = frames_for(client, [T], tag="w", trust="drifting")
    files["w0"] = plain(320, 240)
    job = client.update("m", fr, files)
    i = job["inputs"][0]
    assert (i["status"], i["reason"]) == ("registered", None)
    assert np.linalg.norm(np.array(i["T_map_cam"])[:3, 3] - T[:3, 3]) < 0.05
    v = client.service.store.version("m")
    assert v["frames"][-1]["method"] == "depth"
    # without a depth image a bare wall is still rejected by the checks
    nf, nfiles = frames_for(client, [base_pose(100)], tag="p", trust="drifting", depth=False)
    buf = io.BytesIO()
    Image.new("RGB", (320, 240), (90, 92, 95)).save(buf, format="JPEG")
    nfiles["p0"] = buf.getvalue()
    assert client.update("m", nf, nfiles)["inputs"][0]["reason"] == "plain"


def test_a_look_in_two_halves_with_a_dry_run_between(client: Client,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """The look sends its first half, dry-runs the second while the first fuses, then sends
    the second: the dry run reads the latest committed version at once (it never waits for the
    running job), and the second job registers against the first's result."""
    import threading

    from oh_my_slam.service import prior_first

    fr, files = frames_for(client, [base_pose(0)], tag="o")
    client.update("m", fr, files, cameras={"cam": CAM})
    fusing, release = threading.Event(), threading.Event()
    real = prior_first.fuse_objects

    def slow(*a: Any, **k: Any) -> Any:
        fusing.set()
        release.wait(20)
        return real(*a, **k)

    monkeypatch.setattr(prior_first, "fuse_objects", slow)
    first = [base_pose(26 * k) for k in range(1, 4)]
    second = [base_pose(26 * k) for k in range(4, 7)]
    f1, files1 = frames_for(client, first, tag="h1")
    j1 = client.post_update("m", {"frames": f1}, files1)
    assert fusing.wait(20)
    f2, files2 = frames_for(client, second, tag="h2")
    t = time.monotonic()
    out = dry(client, "m", f2, files2)
    assert time.monotonic() - t < 10 and not release.is_set()
    assert out["base_version"] == 1  # the first half is not committed yet
    j2 = client.post_update("m", {"frames": f2}, files2)
    assert client.get(f"/api/jobs/{j2['job_id']}", "Job")["state"] == "queued"
    release.set()
    d1, d2 = client.wait(j1["job_id"]), client.wait(j2["job_id"])
    assert (d1["state"], d1["result_version"]) == ("committed", 2)
    assert (d2["state"], d2["base_version"], d2["result_version"]) == ("committed", 2, 3)
    assert [i["status"] for i in d2["inputs"]] == ["registered"] * 3
    again = dry(client, "m", f2, files2)
    assert again["base_version"] == 3
    assert [r["predicted_reason"] for r in again["frames"]] == ["duplicate"] * 3


def test_keyframes_without_features_or_depth_are_skipped_with_trust_none(client: Client) -> None:
    """A map holding a keyframe stored without its features and one without its keypoint
    depths: an update and a dry run of frames with trust none skip them as references, never
    fail (HTTP 500)."""
    fr, files = frames_for(client, [base_pose(0), base_pose(30), base_pose(60)], tag="a")
    client.update("m", fr, files, cameras={"cam": CAM})
    store = client.service.store
    v = store.version("m")
    kf = store.mapper_dir("m") / "kf" / f"{v['frames'][1]['internal']}.npz"
    with np.load(kf) as z:
        kept = {k: z[k] for k in z.files if k != "kdepth"}
    np.savez(kf, **kept)  # a keyframe stored without its keypoint depths
    (store.mapper_dir("m") / "kf" / f"{v['frames'][2]['internal']}.npz").unlink()  # nor features
    nf, nfiles = frames_for(client, [base_pose(10), base_pose(20)], tag="n", trust=None)
    for f in nf:
        f["prior"] = {"trust": "none"}
    out = dry(client, "m", nf, nfiles)
    assert [r["predicted_status"] for r in out["frames"]] == ["registered"] * 2
    job = client.update("m", nf, nfiles)
    assert job["state"] == "committed"
    assert [i["status"] for i in job["inputs"]] == ["registered"] * 2


SKILL_GATE = {"max_residual_m": 0.10, "max_gauge_shift_m": 0.05, "max_rejected_share": 0.5}


def _bare(k: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (320, 240), (200 - 3 * k, 198, 190)).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


@pytest.mark.parametrize("trust", ["metric", "drifting"])
@pytest.mark.parametrize("slip", [0.0, 0.3])
def test_a_warping_update_with_priors_is_rejected_by_the_gate(client: Client, trust: str,
                                                             slip: float) -> None:
    """After a first textured frame the update's priors slip by 0.3 m: the one textured frame
    after the slip conflicts with its prior (rejected, a fifth of the update), and the bare
    walls are placed by their priors and depth alone, which cannot see a slide along the wall.
    Their residual to the map is the conflict's (``inherit_conflicts``): the gate rejects the
    update, in the dry run too. The same update with priors that agree is accepted."""
    poses = [base_pose(30 * k) for k in range(6)]
    fr, files = frames_for(client, poses, tag="a", trust=trust)
    client.update("m", fr, files, cameras={"cam": CAM})
    new = [base_pose(12 + 25 * k) for k in range(5)]
    shift = np.eye(4)
    shift[0, 3] = slip
    fr, files = frames_for(client, new, tag="b", trust=trust,
                           priors=[new[0]] + [shift @ T for T in new[1:]])
    for k in (1, 3, 4):
        files[f"b{k}"] = _bare(k)
    d = dry(client, "m", fr, files, accept_if=SKILL_GATE)
    job = client.update("m", fr, files, accept_if=SKILL_GATE)
    q = job["quality"]
    if slip == 0.0:
        assert job["state"] == "committed" and q["accepted"] and d["quality"]["accepted"]
        assert [i["status"] for i in job["inputs"]] == ["registered"] * 5
        assert q["previous_residuals"]["max_m"] < 0.02
        return
    assert job["state"] == "rejected" and "max_residual_m" in q["failed_checks"]
    assert "max_residual_m" in d["quality"]["failed_checks"]
    assert q["rejected_share"] == 0.2 and q["previous_residuals"]["max_m"] > 0.25
    assert [i["reason"] for i in job["inputs"]][2] == "prior_conflict"
    assert client.service.store.latest("m") == 1
