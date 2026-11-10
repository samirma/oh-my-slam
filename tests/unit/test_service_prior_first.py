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
