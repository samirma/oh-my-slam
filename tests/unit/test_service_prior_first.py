"""The prior-first update path of ``server.sh`` (``service.prior_first``) end to end through the
Starlette app, on frames ray-cast from a textured room (``tests.synth.room``): a map started with
priors is anchored, keeps registered and provisional frames, reports quality and scale, and
commits only what passes the gate; a job whose frames are all rejected costs only the checks."""

from __future__ import annotations

import functools
import hashlib
import io
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
    jobs = Jobs(store.jobs, Updater(store, backend, timings), timings.predict_update)
    jobs.start()
    service = Service(store, jobs, FakeInference(), backend.require_inference, timings=timings,
                      version="test")
    with TestClient(create_app(service)) as http:
        yield Client(http, backend, service)
    jobs.stop(timeout=10)


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
    assert CAPABILITIES == ["O1", "O3-jobs", "O4"]
    assert client.get("/api/health", "Health")["capabilities"] == CAPABILITIES


def test_cold_start_then_a_sweep(client: Client) -> None:
    fr, files = frames_for(client, [base_pose(0)], tag="a")
    job = client.update("m", fr, files, cameras={"cam": CAM})
    assert job["state"] == "committed" and job["result_version"] == 1
    assert job["inputs"][0]["status"] == "registered"
    assert job["inputs"][0]["prior_residual_m"] == 0.0
    q = job["quality"]
    assert q["gauge_shift"] is None and q["scale_ratio_vs_prev"] is None
    assert q["prior_gravity_deg"] is None  # a depth image: no gravity asked
    assert job["scale_source"] == "depth" and client.backend.infers == []
    poses = [base_pose(15.0 + 30.0 * k) for k in range(6)]
    fr, files = frames_for(client, poses, tag="s")
    job = client.update("m", fr, files)
    assert job["state"] == "committed" and job["result_version"] == 2
    assert [i["status"] for i in job["inputs"]] == ["registered"] * 6
    for i, T in zip(job["inputs"], poses, strict=True):
        assert np.linalg.norm(np.array(i["T_map_cam"])[:3, 3] - T[:3, 3]) < 0.03
        assert i["inliers"] > 50 and i["sigma_pos_m"] >= 0.01
    q = job["quality"]
    assert q["gauge_shift"] == {"translation_m": 0.0, "rotation_deg": 0.0, "scale_ratio": 1.0}
    assert q["previous_residuals"]["count"] == 1 and q["scale_ratio_vs_prev"] == 1.0
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
    fr, files = frames_for(client, [base_pose(0)], tag="a")
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
    assert [f["pose_state"] for f in frames] == ["fused", "provisional"]
    summary = client.get("/api/maps/m", "MapSummary")
    assert summary["provisional_frames"] == 1
    # a gate that fails commits nothing
    fr, files = frames_for(client, [base_pose(10), T], tag="c", priors=[base_pose(10), off])
    job = client.update("m", fr, files, accept_if={"max_rejected_share": 0.2})
    assert job["state"] == "rejected" and job["quality"]["failed_checks"] == [
        "max_rejected_share"]
    assert job["result_version"] is None and client.service.store.latest("m") == 2


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
    # a new map whose kept frames carry no prior goes to the mapper
    fr, files = frames_for(client, [base_pose(0)], tag="z", trust=None)
    client.backend.add(files["z0"], base_pose(0))
    job = client.update("other", fr, files, cameras={"cam": CAM})
    assert job["state"] == "committed" and client.backend.calls
    assert "engine" not in client.service.store.version("other")


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
