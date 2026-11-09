"""``server.sh``'s API (specs/api.md) end to end through the Starlette app, with the mapper replaced
by ``tests.fakes.service.FakeBackend``: every response body is checked against api.schema.json,
every response carries ``Server-Timing``; updates create and anchor maps (map.md §1) and report
per-input status (api.md §4)."""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from starlette.testclient import TestClient

from oh_my_slam.service import schema
from oh_my_slam.service.anchor import Sim3, rotation_deg
from oh_my_slam.service.app import Service, create_app
from oh_my_slam.service.jobs import Jobs
from oh_my_slam.service.store import Store
from oh_my_slam.service.timings import Timings
from oh_my_slam.service.update import Updater
from tests.fakes.service import (
    CAMERA,
    FakeBackend,
    FakeInference,
    multipart,
    plain,
    png16,
    pose,
    textured,
)

ERRORS = {400, 404, 405, 409, 410, 413, 500, 503}


class Client:
    def __init__(self, http: TestClient, backend: FakeBackend, service: Service) -> None:
        self.http = http
        self.backend = backend
        self.service = service

    def check(self, r: Any, name: str | None) -> Any:
        assert "server-timing" in r.headers and "total;dur=" in r.headers["server-timing"]
        if r.status_code == 204:
            assert r.content == b""
            return None
        doc = r.json()
        if r.status_code in ERRORS:
            assert schema.problems("ErrorBody", doc) == [], doc
        elif name:
            assert schema.problems(name, doc) == [], doc
        return doc

    def get(self, path: str, name: str | None = None, status: int = 200) -> Any:
        r = self.http.get(path)
        assert r.status_code == status, r.text
        return self.check(r, name)

    def post_update(self, name: str, request: dict[str, Any], files: dict[str, bytes],
                    status: int = 202, query: str = "") -> Any:
        data, ctype = multipart(request, files)
        r = self.http.post(f"/api/maps/{name}/updates{query}", content=data,
                           headers={"content-type": ctype})
        assert r.status_code == status, r.text
        doc = self.check(r, "Job")
        if status == 202:
            assert r.headers["location"] == f"/api/jobs/{doc['job_id']}"
            assert doc["state"] == "queued"
        return doc

    def wait(self, job_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + 30
        while True:
            doc = self.get(f"/api/jobs/{job_id}?wait_s=5", "Job")
            if doc["state"] not in ("queued", "registering") or time.monotonic() > deadline:
                return dict(doc)

    def update(self, name: str, frames: list[dict[str, Any]], files: dict[str, bytes],
               cameras: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
        req: dict[str, Any] = {"frames": frames, **extra}
        if cameras is not None:
            req["cameras"] = cameras
        return self.wait(self.post_update(name, req, files)["job_id"])


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


WORLD = [pose(0, 0, 0), pose(20, 0.3, 0.1), pose(40, 0.6, 0.3), pose(60, 0.8, 0.6),
         pose(80, 1.0, 1.0), pose(100, 1.1, 1.4)]


def frames_of(c: Client, n: int, start: int = 0, prior: Any = None, seed: int = 0
              ) -> tuple[list[dict[str, Any]], dict[str, bytes], list[np.ndarray]]:
    frames, files, truth = [], {}, []
    for i in range(start, start + n):
        img = c.backend.add(textured(seed * 100 + i), WORLD[i % len(WORLD)])
        files[f"img{i}"] = img
        f: dict[str, Any] = {"name": f"f{seed}_{i}", "image": f"img{i}", "camera_id": "cam"}
        if prior is not None:
            p = prior(i, WORLD[i % len(WORLD)])
            if p is not None:
                f["prior"] = p
        frames.append(f)
        truth.append(WORLD[i % len(WORLD)])
    return frames, files, truth


def centre_error(T: Any, ref: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(T)[:3, 3] - ref[:3, 3]))


# -- service endpoints ---------------------------------------------------------------------------


def test_health_reports_the_inference_server_and_one_lane(client: Client) -> None:
    doc = client.get("/api/health", "Health")
    assert doc["service"] == "ready" and doc["inference"]["state"] == "ready"
    assert doc["lanes"]["interactive"] == {"state": "idle", "queued": 0, "oldest_s": None}
    assert doc["version"] == "test"


def test_openapi_describes_every_endpoint_with_the_schema_defs(client: Client) -> None:
    doc = client.get("/api/openapi.json")
    assert doc["openapi"] == "3.1.0"
    assert set(doc["paths"]) == {"/api/health", "/api/openapi.json", "/api/maps/{name}/updates",
                                 "/api/jobs/{id}", "/api/maps/{name}", "/api/maps/{name}/frames"}
    assert set(doc["components"]["schemas"]) == set(schema.defs())
    text = str(doc)
    assert "#/$defs/" not in text
    post = doc["paths"]["/api/maps/{name}/updates"]["post"]
    assert post["responses"]["202"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/Job"}
    body = post["requestBody"]["content"]["multipart/form-data"]["schema"]
    assert body["properties"]["request"] == {"$ref": "#/components/schemas/UpdateRequest"}
    assert "409" in post["responses"] and "503" in post["responses"]
    assert doc["paths"]["/api/maps/{name}"]["delete"]["responses"]["204"] == {
        "description": "Delete the map and its versions"}


def test_unknown_routes_and_methods_are_typed_errors(client: Client) -> None:
    assert client.get("/api/nothing", status=404)["error"]["code"] == "invalid_request"
    r = client.http.put("/api/health")
    assert r.status_code == 405 and client.check(r, None)["error"]["details"]["field"] == "path"


def test_a_handler_bug_is_an_internal_error(client: Client, monkeypatch: pytest.MonkeyPatch
                                            ) -> None:
    monkeypatch.setattr(client.service.store, "require", lambda name: 1 / 0)
    doc = client.get("/api/maps/m", status=500)
    assert doc["error"]["code"] == "internal" and doc["error"]["details"]["stage"] == "get_map"


# -- the first update ----------------------------------------------------------------------------


def test_first_update_without_priors_follows_the_first_camera_rule(client: Client) -> None:
    frames, files, truth = frames_of(client, 4)
    job = client.update("m", frames, files, {"cam": CAMERA})
    assert job["state"] == "committed" and job["result_version"] == 1
    assert job["base_version"] is None and job["provisional_version"] is None
    assert job["scale_source"] == "monocular" and job["scale_sigma_rel"] == 0.1
    ins = job["inputs"]
    assert [i["status"] for i in ins] == ["registered"] * 4 and ins[0]["inliers"] == 400
    T0 = np.asarray(ins[0]["T_map_cam"])
    assert np.allclose(T0[:3, 3], 0, atol=1e-9)
    assert abs(T0[1, 2]) < 1e-9 and T0[0, 2] > 0  # looks along +x
    # relative geometry is the world's, at the mapper's scale (0.5 of it)
    d01 = centre_error(ins[1]["T_map_cam"], np.asarray(ins[0]["T_map_cam"]))
    assert d01 == pytest.approx(0.5 * centre_error(truth[1], truth[0]), rel=1e-6)
    q = job["quality"]
    assert q["accepted"] and q["failed_checks"] == [] and q["rejected_share"] == 0
    assert q["gauge_shift"] is None and q["scale_ratio_vs_prev"] is None
    assert q["prior_gravity_deg"] is None and q["rebuilt"] is False
    assert job["timings_s"]["total"] > 0 and "register" in job["timings_s"]
    s = client.get("/api/maps/m", "MapSummary")
    assert s["map_version"] == s["latest_version"] == 1 and s["frames"] == 4
    assert s["objects"] == 3 and s["cameras"] == {"cam": CAMERA} and s["running_job"] is None
    assert s["scale_source"] == "monocular" and s["floor_z_m"] == pytest.approx(-0.5)
    fr = client.get("/api/maps/m/frames", "Frames")
    assert [f["name"] for f in fr["frames"]] == [f["name"] for f in frames]
    assert all(f["pose_state"] == "fused" and f["added_version"] == 1 for f in fr["frames"])
    assert np.allclose(fr["frames"][1]["T_map_cam"], ins[1]["T_map_cam"], atol=1e-5)


def test_metric_priors_fix_the_frame_and_the_scale(client: Client) -> None:
    rng = np.random.default_rng(3)
    noise = {}

    def prior(i: int, T: np.ndarray) -> dict[str, Any]:
        P = T.copy()
        P[:3, 3] += noise.setdefault(i, rng.normal(0, 0.01, 3))
        return {"trust": "metric", "T_map_cam": P.tolist(), "sigma_pos_m": 0.02,
                "sigma_rot_deg": 1.0}

    frames, files, truth = frames_of(client, 6, prior=prior)
    job = client.update("m", frames, files, {"cam": CAMERA})
    assert job["state"] == "committed" and job["scale_source"] == "priors"
    assert 0 < job["scale_sigma_rel"] < 0.05
    for i, t in zip(job["inputs"], truth, strict=True):
        assert centre_error(i["T_map_cam"], t) < 0.03
        assert i["prior_residual_m"] < 0.04 and i["prior_residual_deg"] < 0.5
    q = job["quality"]
    assert q["prior_residuals"]["count"] == 6 and q["prior_residuals"]["max_m"] < 0.04
    assert q["prior_gravity_deg"] == pytest.approx(0, abs=0.5)


def test_a_conflicting_prior_is_rejected_and_the_frame_mapped_again_without_it(
        client: Client) -> None:
    def prior(i: int, T: np.ndarray) -> dict[str, Any]:
        P = T.copy()
        if i == 2:
            P[:3, 3] += [0.0, 0.0, 0.5]  # half a metre off
        return {"trust": "metric", "T_map_cam": P.tolist(), "sigma_pos_m": 0.02}

    frames, files, truth = frames_of(client, 6, prior=prior)
    job = client.update("m", frames, files, {"cam": CAMERA})
    assert job["state"] == "committed"
    bad = job["inputs"][2]
    assert bad["status"] == "rejected" and bad["reason"] == "prior_conflict"
    assert bad["T_map_cam"] is None and bad["prior_residual_m"] == pytest.approx(0.5, abs=0.02)
    assert len(client.backend.calls) == 2 and len(client.backend.calls[1]) == 5
    assert client.get("/api/maps/m", "MapSummary")["frames"] == 5
    assert job["quality"]["rejected_share"] == pytest.approx(1 / 6, abs=1e-3)


def test_drifting_priors_fix_the_frame_and_report_their_residuals(client: Client) -> None:
    frames, files, truth = frames_of(client, 5, prior=lambda i, T: {"trust": "drifting",
                                                                      "T_map_cam": T.tolist()})
    job = client.update("m", frames, files, {"cam": CAMERA})
    assert job["scale_source"] == "priors"
    for i, t in zip(job["inputs"], truth, strict=True):
        assert centre_error(i["T_map_cam"], t) < 1e-6 and i["prior_residual_m"] < 1e-6


def test_priors_that_cannot_fix_the_scale_take_it_from_the_camera_heights(
        client: Client) -> None:
    """A camera turning in place: one centre, so the priors fix the frame, not the scale."""
    turn = [pose(a, 0.0, 0.0, 1.2) for a in (0, 30, 60, 90)]
    frames, files = [], {}
    for i, T in enumerate(turn):
        files[f"t{i}"] = client.backend.add(textured(500 + i), T)
        frames.append({"name": f"t{i}", "image": f"t{i}", "camera_id": "cam",
                       "prior": {"trust": "metric", "T_map_cam": T.tolist()},
                       **({"height_above_floor_m": 1.2, "height_sigma_m": 0.01} if i else {})})
    cam = {**CAMERA, "height_above_floor_m": 1.2}
    job = client.update("m", frames, files, {"cam": cam})
    assert job["scale_source"] == "camera_height"
    assert np.allclose(job["inputs"][2]["T_map_cam"], turn[2], atol=1e-5)


def test_depth_images_give_the_scale(client: Client) -> None:
    frames, files, truth = frames_of(client, 3)
    for i, f in enumerate(frames):
        files[f"d{i}"] = png16(np.full((240, 320), 2000))  # 2 m at the default 1 mm
        f["depth"] = f"d{i}"
    job = client.update("m", frames, files, {"cam": CAMERA})
    assert job["scale_source"] == "depth"
    d01 = centre_error(job["inputs"][1]["T_map_cam"], np.asarray(job["inputs"][0]["T_map_cam"]))
    assert d01 == pytest.approx(centre_error(truth[1], truth[0]), rel=1e-6)


def test_a_distorted_camera_with_a_mask_is_rectified_for_the_mapper(client: Client) -> None:
    import io

    from PIL import Image

    mask = np.zeros((240, 320), np.uint8)
    mask[200:, :] = 255
    buf = io.BytesIO()
    Image.fromarray(mask).save(buf, format="PNG")
    cam = {**CAMERA, "cx": 158.0, "distortion": {"model": "opencv",
                                                 "coefficients": [-0.1, 0.01, 0, 0]},
           "ignore_mask": "mask"}
    img = textured(42)
    job = client.update("m", [{"name": "a", "image": "a", "camera_id": "cam"}],
                        {"a": img, "mask": buf.getvalue()}, {"cam": cam})
    # the mapper got a resampled image the fake does not know: not placed
    assert job["state"] == "rejected" and job["inputs"][0]["reason"] == "no_overlap"
    assert job["quality"]["failed_checks"] == ["hard_limit"]
    assert client.service.store.mask_digest("m", "cam")


# -- per-input status ----------------------------------------------------------------------------


def test_each_input_has_a_status_and_a_reason(client: Client) -> None:
    frames, files, truth = frames_of(client, 3)
    # a plain wall, a too-close frame (its depth PNG), one the mapper does not place, a blur
    import cv2

    files["wall"] = plain()
    frames.append({"name": "wall", "image": "wall", "camera_id": "cam"})
    near = client.backend.add(textured(77), WORLD[3])
    files["near"], files["nd"] = near, png16(np.full((240, 320), 150))
    frames.append({"name": "near", "image": "near", "camera_id": "cam", "depth": "nd"})
    lost = textured(78)
    files["lost"] = lost
    frames.append({"name": "lost", "image": "lost", "camera_id": "cam"})
    arr = cv2.imdecode(np.frombuffer(textured(79), np.uint8), cv2.IMREAD_COLOR)
    ok, enc = cv2.imencode(".jpg", cv2.GaussianBlur(arr, (0, 0), 4))
    files["blur"] = enc.tobytes()
    frames.append({"name": "blur", "image": "blur", "camera_id": "cam"})
    files["dup"] = files["img0"]
    frames.append({"name": "dup", "image": "dup", "camera_id": "cam"})
    job = client.update("m", frames, files, {"cam": CAMERA})
    reasons = {i["name"]: (i["status"], i["reason"]) for i in job["inputs"]}
    assert reasons["f0_0"] == ("registered", None)
    assert reasons["wall"] == ("rejected", "plain")
    assert reasons["near"] == ("rejected", "too_close")
    assert reasons["lost"] == ("rejected", "no_overlap")
    assert reasons["blur"] == ("rejected", "blur")
    assert reasons["dup"] == ("rejected", "duplicate")
    lost_in = next(i for i in job["inputs"] if i["name"] == "lost")
    assert lost_in["T_map_cam"] is None and lost_in["inliers"] == 0
    assert lost_in["prior_residual_m"] is None
    assert job["quality"]["rejected_share"] == pytest.approx(5 / 8, abs=1e-3)


def test_a_suspicious_frame_is_measured_with_the_monocular_depth(client: Client) -> None:
    """Few features: its median depth comes from the depth model, here too close."""
    import cv2

    img = np.full((240, 320, 3), 128, np.uint8)
    cv2.rectangle(img, (40, 40), (280, 200), (30, 30, 30), 3)
    ok, enc = cv2.imencode(".jpg", img)
    files = {"a": enc.tobytes()}
    client.backend.mono = 0.2
    job = client.update("m", [{"name": "a", "image": "a", "camera_id": "cam"}], files,
                        {"cam": CAMERA})
    assert job["inputs"][0]["reason"] == "too_close"
    client.backend.mono = None
    job = client.update("m", [{"name": "a", "image": "a", "camera_id": "cam"}], files)
    assert job["inputs"][0]["reason"] == "few_features"  # placed by no one either way


def test_a_registered_frame_whose_aligned_depth_is_too_close_is_rejected(client: Client) -> None:
    frames, files, truth = frames_of(client, 3)
    client.backend.depth_m[next(iter(client.backend.truth))] = 0.2  # img0, 0.2 m away
    job = client.update("m", frames, files, {"cam": CAMERA})
    assert job["inputs"][0]["reason"] == "too_close" and job["state"] == "committed"
    assert len(client.backend.calls) == 2


def test_nothing_registered_is_a_rejected_job_and_no_map(client: Client) -> None:
    frames, files, truth = frames_of(client, 2)
    client.backend.nothing = True
    job = client.update("m", frames, files, {"cam": CAMERA})
    assert job["state"] == "rejected" and job["result_version"] is None
    assert [i["reason"] for i in job["inputs"]] == ["no_overlap", "no_overlap"]
    assert job["quality"]["failed_checks"] == ["hard_limit"]
    assert client.get("/api/maps/m", status=404)["error"]["code"] == "map_not_found"
    files2 = {"w": plain()}
    job = client.update("m", [{"name": "w", "image": "w", "camera_id": "cam"}], files2)
    assert job["state"] == "rejected" and job["inputs"][0]["reason"] == "plain"
    assert client.backend.calls == [["0000.jpg", "0001.jpg"]]  # the wall never reached it


def test_low_confidence_frames_count_in_the_quality_report(client: Client) -> None:
    client.backend.stats = {"pose_matches": 30, "pose_residual_deg": 20.0}
    frames, files, truth = frames_of(client, 3)
    job = client.update("m", frames, files, {"cam": CAMERA})
    assert job["quality"]["low_confidence_share"] == 1.0
    assert job["inputs"][0]["sigma_pos_m"] > 0.1


# -- later updates -------------------------------------------------------------------------------


def test_later_updates_keep_the_frame_and_report_no_gauge_shift(client: Client) -> None:
    frames, files, _ = frames_of(client, 3)
    first = client.update("m", frames, files, {"cam": CAMERA})
    frames, files, _ = frames_of(client, 3, start=3)
    job = client.update("m", frames, files)  # the camera is known: omitted
    assert job["state"] == "committed" and job["result_version"] == 2 and job["base_version"] == 1
    g = job["quality"]["gauge_shift"]
    assert g["translation_m"] < 1e-6 and g["rotation_deg"] < 1e-3 and g["scale_ratio"] == 1.0
    assert job["quality"]["previous_residuals"]["count"] == 3
    assert job["quality"]["scale_ratio_vs_prev"] == 1.0
    fr = client.get("/api/maps/m/frames", "Frames")
    assert len(fr["frames"]) == 6 and fr["map_version"] == 2
    assert np.allclose(fr["frames"][0]["T_map_cam"], first["inputs"][0]["T_map_cam"], atol=1e-5)
    assert [f["added_version"] for f in fr["frames"]] == [1, 1, 1, 2, 2, 2]
    old = client.get("/api/maps/m/frames?version=1", "Frames")
    assert len(old["frames"]) == 3 and old["map_version"] == 1
    assert client.get("/api/maps/m?version=1", "MapSummary")["latest_version"] == 2


def test_a_rebuild_is_aligned_back_onto_the_previous_version(client: Client) -> None:
    frames, files, _ = frames_of(client, 4)
    first = client.update("m", frames, files, {"cam": CAMERA})
    a = math.radians(35)
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    client.backend.rebuild_with = Sim3(1.7, Rz, np.array([5.0, -3.0, 0.2]))
    frames, files, _ = frames_of(client, 2, start=4)
    job = client.update("m", frames, files)
    q = job["quality"]
    assert q["rebuilt"] is True
    assert q["gauge_shift"]["rotation_deg"] == pytest.approx(35, abs=1e-6)
    assert q["gauge_shift"]["scale_ratio"] == pytest.approx(0.5 / 1.7, rel=1e-4)
    assert q["scale_ratio_vs_prev"] == pytest.approx(1.0, abs=1e-4)
    assert q["gauge_shift"]["translation_m"] > 1
    assert q["previous_residuals"]["max_m"] < 1e-6
    fr = client.get("/api/maps/m/frames", "Frames")["frames"]
    for i in range(4):
        T, ref = np.asarray(fr[i]["T_map_cam"]), np.asarray(first["inputs"][i]["T_map_cam"])
        assert np.allclose(T, ref, atol=1e-6)
    # the new frames sit where the first map's scale puts them
    d = centre_error(fr[4]["T_map_cam"], np.asarray(fr[0]["T_map_cam"]))
    assert d == pytest.approx(0.5 * centre_error(WORLD[4], WORLD[0]), rel=1e-6)
    assert rotation_deg(np.eye(3)) == 0


def test_metric_priors_on_a_later_update_join_the_anchors(client: Client) -> None:
    frames, files, _ = frames_of(client, 3, prior=lambda i, T: {"trust": "metric",
                                                                  "T_map_cam": T.tolist()})
    client.update("m", frames, files, {"cam": CAMERA})
    frames, files, truth = frames_of(client, 3, start=3, prior=lambda i, T: {
        "trust": "metric", "T_map_cam": T.tolist()})
    job = client.update("m", frames, files)
    assert job["scale_source"] == "priors"
    assert job["quality"]["prior_residuals"]["count"] == 6
    assert all(centre_error(i["T_map_cam"], t) < 1e-6
               for i, t in zip(job["inputs"], truth, strict=True))


def test_a_later_update_on_a_monocular_map_keeps_its_scale_source(client: Client) -> None:
    frames, files, _ = frames_of(client, 3)
    client.update("m", frames, files, {"cam": CAMERA})
    frames, files, _ = frames_of(client, 1, start=3, prior=lambda i, T: {
        "trust": "metric", "T_map_cam": (Sim3(0.5).pose(T)).tolist()})
    job = client.update("m", frames, files)
    assert job["scale_source"] == "monocular"


def test_a_name_or_an_image_already_in_the_map_is_a_duplicate(client: Client) -> None:
    frames, files, _ = frames_of(client, 2)
    client.update("m", frames, files, {"cam": CAMERA})
    again = [{"name": "f0_0", "image": "x", "camera_id": "cam"},
             {"name": "new", "image": "img1", "camera_id": "cam"}]
    job = client.update("m", again, {"x": textured(999), "img1": files["img1"]})
    assert [i["reason"] for i in job["inputs"]] == ["duplicate", "duplicate"]
    assert job["state"] == "rejected"


def test_a_mapper_that_keeps_no_stored_frame_keeps_the_previous_transform(
        client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    frames, files, _ = frames_of(client, 2)
    client.update("m", frames, files, {"cam": CAMERA})
    real = client.backend.read

    def forget(mapper_dir: Path) -> Any:
        m = real(mapper_dir)
        m.frames = m.frames[2:]
        return m

    monkeypatch.setattr(client.backend, "read", forget)
    frames, files, _ = frames_of(client, 2, start=2)
    job = client.update("m", frames, files)
    assert job["quality"]["gauge_shift"] == {"translation_m": 0.0, "rotation_deg": 0.0,
                                             "scale_ratio": 1.0}
    assert len(client.get("/api/maps/m/frames", "Frames")["frames"]) == 4


# -- the gate ------------------------------------------------------------------------------------


def test_accept_if_rejects_without_committing(client: Client) -> None:
    frames, files, _ = frames_of(client, 3)
    client.update("m", frames, files, {"cam": CAMERA})
    client.backend.rebuild_with = Sim3(0.5, np.eye(3), np.array([0.0, 0.0, 0.0]))
    client.backend.noise = {next(iter(client.backend.truth)): np.array([0.0, 0.05, 0.0])}
    frames, files, _ = frames_of(client, 2, start=3)
    files["w"] = plain()
    frames.append({"name": "w", "image": "w", "camera_id": "cam"})
    job = client.update("m", frames, files, accept_if={
        "max_residual_m": 0.001, "max_gauge_shift_m": 0.0, "max_rejected_share": 0.1})
    assert job["state"] == "rejected" and job["result_version"] is None
    assert job["quality"]["accepted"] is False
    assert set(job["quality"]["failed_checks"]) == {"max_residual_m", "max_gauge_shift_m",
                                                    "max_rejected_share"}
    assert client.get("/api/maps/m", "MapSummary")["latest_version"] == 1
    job = client.update("m", frames[:1], files, accept_if={"max_residual_m": 10.0})
    assert job["state"] == "committed"


# -- request validation --------------------------------------------------------------------------


def one(**f: Any) -> dict[str, Any]:
    return {"frames": [{"name": "a", "image": "a", "camera_id": "cam", **f}],
            "cameras": {"cam": CAMERA}}


@pytest.mark.parametrize(("request_", "files", "field"), [
    ({"frames": []}, {}, "frames"),
    (one(extra=1), {"a": b""}, "frames[0]"),
    (one(prior={"trust": "metric"}), {}, "frames[0].prior"),
    (one(), {}, "frames[0].image"),
    ({**one(), "cameras": {}}, {"a": b"x"}, "frames[0].camera_id"),
    ({"frames": one()["frames"] * 2, "cameras": {"cam": CAMERA}}, {"a": textured(1)},
     "frames[1].name"),
    (one(), {"a": b"not an image"}, "frames[0].image"),
    (one(depth="d"), {"a": textured(1), "d": textured(2)}, "frames[0].depth"),
    (one(depth="d"), {"a": textured(1), "d": png16(np.zeros((10, 10)))}, "frames[0].depth"),
    (one(depth="d"), {"a": textured(1), "d": b"junk"}, "frames[0].depth"),
    (one(prior={"trust": "metric", "T_map_cam": (2 * np.eye(4)).tolist()}), {"a": textured(1)},
     "frames[0].prior.T_map_cam"),
    ({**one(), "cameras": {"cam": {**CAMERA, "ignore_mask": "m"}}}, {"a": textured(1)},
     "cameras.cam.ignore_mask"),
    ({**one(), "cameras": {"cam": {**CAMERA, "ignore_mask": "m"}}},
     {"a": textured(1), "m": textured(3)}, "cameras.cam.ignore_mask"),
    ({**one(), "cameras": {"cam": {**CAMERA, "ignore_mask": "m"}}},
     {"a": textured(1), "m": png16(np.zeros((10, 10)))}, "cameras.cam.ignore_mask"),
    ({**one(), "cameras": {"cam": {**CAMERA, "ignore_mask": "m"}}},
     {"a": textured(1), "m": b"junk"}, "cameras.cam.ignore_mask"),
])
def test_invalid_requests_name_the_failing_field(client: Client, request_: dict[str, Any],
                                                 files: dict[str, bytes], field: str) -> None:
    doc = client.post_update("m", request_, files, status=400)
    assert doc["error"]["code"] == "invalid_request" and not doc["error"]["retryable"]
    assert doc["error"]["details"]["field"] == field, doc


def test_an_image_of_another_size_or_format_is_refused(client: Client) -> None:
    import io

    from PIL import Image

    gif = io.BytesIO()
    Image.new("RGB", (320, 240)).save(gif, format="GIF")
    for data in (textured(1, 100, 80), gif.getvalue()):
        doc = client.post_update("m", one(), {"a": data}, status=400)
        assert doc["error"]["details"]["field"] == "frames[0].image"


def test_bodies_that_are_not_multipart_requests_are_refused(client: Client) -> None:
    r = client.http.post("/api/maps/m/updates", json={"frames": []})
    assert r.status_code == 400 and client.check(r, None)["error"]["details"]["field"] == "body"
    data, ctype = multipart({"frames": []}, {})
    for body, field in ((data.replace(b"request", b"other", 1), "request"),
                        (data.replace(b'{"frames": []}', b"{nope"), "request"),
                        (data.rsplit(b"--", 2)[0], "body"),
                        (data.replace(b"\r\n\r\n", b"\r\n", 1), "body"),
                        (data.replace(b'; name="request"', b"", 1), "body"),
                        (multipart({"frames": []}, {}, (("request", b"{}"),))[0], "request")):
        r = client.http.post("/api/maps/m/updates", content=body, headers={"content-type": ctype})
        assert r.status_code == 400, body
        assert client.check(r, None)["error"]["details"]["field"] == field
    r = client.http.post("/api/maps/bad name!/updates", content=data,
                         headers={"content-type": ctype})
    assert client.check(r, None)["error"]["details"]["field"] == "name"
    doc = client.post_update("m", one(), {"a": textured(1)}, status=400, query="?dry_run=1")
    assert doc["error"]["details"]["field"] == "dry_run"


def test_a_redefined_camera_is_a_conflict(client: Client) -> None:
    import io

    from PIL import Image

    m = io.BytesIO()
    Image.fromarray(np.zeros((240, 320), np.uint8)).save(m, format="PNG")
    cam = {**CAMERA, "ignore_mask": "m"}
    frames, files, _ = frames_of(client, 2)
    client.update("m", frames, {**files, "m": m.getvalue()}, {"cam": cam})
    # the same camera again (its mask under another part name) is accepted
    frames, files, _ = frames_of(client, 1, start=2)
    ok = client.update("m", frames, {**files, "m2": m.getvalue()},
                       {"cam": {**CAMERA, "ignore_mask": "m2", "depth_scale_m": 0.001}})
    assert ok["state"] == "committed"
    doc = client.post_update("m", {**one(), "cameras": {"cam": {**CAMERA, "fx": 301.0}}},
                             {"a": textured(1)}, status=409)
    assert doc["error"]["code"] == "camera_conflict" and doc["error"]["details"]["field"] == \
        "cameras.cam"
    other = io.BytesIO()
    Image.fromarray(np.full((240, 320), 255, np.uint8)).save(other, format="PNG")
    doc = client.post_update("m", {**one(), "cameras": {"cam": {**CAMERA, "ignore_mask": "x"}}},
                             {"a": textured(1), "x": other.getvalue()}, status=409)
    assert doc["error"]["code"] == "camera_conflict"


def test_uploads_over_the_cap_or_the_free_disk_are_refused(client: Client,
                                                           monkeypatch: pytest.MonkeyPatch
                                                           ) -> None:
    client.service.upload_cap = 1000
    data, ctype = multipart(one(), {"a": textured(1)})
    r = client.http.post("/api/maps/m/updates", content=data, headers={"content-type": ctype})
    assert r.status_code == 413 and client.check(r, None)["error"]["code"] == "payload_too_large"

    def chunks() -> Iterator[bytes]:
        yield data[:600]
        yield data[600:]

    r = client.http.post("/api/maps/m/updates", content=chunks(), headers={"content-type": ctype})
    assert r.status_code == 413
    client.service.upload_cap = 10 ** 9
    import shutil

    monkeypatch.setattr(shutil, "disk_usage", lambda p: type("U", (), {"free": 10})())
    r = client.http.post("/api/maps/m/updates", content=data, headers={"content-type": ctype})
    assert r.status_code == 413 and "free_bytes" in client.check(r, None)["error"]["details"]


def test_updates_need_the_inference_server(client: Client) -> None:
    client.backend.ready = False
    doc = client.post_update("m", one(), {"a": textured(1)}, status=503)
    assert doc["error"]["code"] == "inference_unavailable" and doc["error"]["retryable"]


def test_a_job_whose_inference_server_goes_down_fails(client: Client,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    frames, files, _ = frames_of(client, 1)
    client.service.jobs.stop(timeout=10)
    jid = client.post_update("m", {"frames": frames, "cameras": {"cam": CAMERA}}, files)["job_id"]
    client.backend.ready = False
    client.service.jobs.start()
    job = client.wait(jid)
    assert job["state"] == "failed" and job["error"]["code"] == "inference_unavailable"


# -- jobs and maps -------------------------------------------------------------------------------


def test_jobs_wait_and_unknown_ids(client: Client) -> None:
    assert client.get("/api/jobs/nope", status=404)["error"]["code"] == "job_not_found"
    for bad in ("x", "31", "-1"):
        doc = client.get(f"/api/jobs/nope?wait_s={bad}", status=400)
        assert doc["error"]["details"]["field"] == "wait_s"
    frames, files, _ = frames_of(client, 1)
    job = client.update("m", frames, files, {"cam": CAMERA})
    t0 = time.monotonic()
    again = client.get(f"/api/jobs/{job['job_id']}?wait_s=0.2", "Job")
    assert again["state"] == "committed" and time.monotonic() - t0 >= 0.2
    assert client.get(f"/api/jobs/{job['job_id']}", "Job") == again


def test_a_queued_job_shows_in_health_and_the_summary(client: Client,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    frames, files, _ = frames_of(client, 2)
    client.update("m", frames[:1], files, {"cam": CAMERA})
    gate = threading.Event()
    real = client.backend.register

    def slow(*a: Any) -> Any:
        gate.wait(10)
        return real(*a)

    monkeypatch.setattr(client.backend, "register", slow)
    j1 = client.post_update("m", {"frames": frames[1:]}, files)
    deadline = time.monotonic() + 10
    while client.service.jobs.running is None and time.monotonic() < deadline:
        time.sleep(0.01)
    j2 = client.post_update("m", {"frames": frames[1:]}, files)
    h = client.get("/api/health", "Health")
    assert h["lanes"]["interactive"]["state"] == "busy" and h["lanes"]["interactive"]["queued"] == 1
    assert h["lanes"]["interactive"]["oldest_s"] >= 0
    assert client.get("/api/maps/m", "MapSummary")["running_job"] == j1["job_id"]
    doc = client.http.delete("/api/maps/m")
    assert doc.status_code == 409 and client.check(doc, None)["error"]["code"] == "map_busy"
    gate.set()
    assert client.wait(j1["job_id"])["state"] == "committed"
    assert client.wait(j2["job_id"])["inputs"][0]["reason"] == "duplicate"


def test_health_counts_a_queued_job_before_the_worker_takes_it(client: Client) -> None:
    client.service.jobs.stop(timeout=10)
    frames, files, _ = frames_of(client, 1)
    client.post_update("m", {"frames": frames, "cameras": {"cam": CAMERA}}, files)
    h = client.get("/api/health", "Health")["lanes"]["interactive"]
    assert h["state"] == "idle" and h["queued"] == 1 and h["oldest_s"] >= 0


def test_versions_that_do_not_exist_or_are_gone(client: Client) -> None:
    frames, files, _ = frames_of(client, 1)
    client.update("m", frames, files, {"cam": CAMERA})
    for q in ("0", "x", "7"):
        doc = client.get(f"/api/maps/m?version={q}", status=400)
        assert doc["error"]["details"]["field"] == "version"
    (client.service.store.dir("m") / "versions" / "1.json").unlink()
    assert client.get("/api/maps/m/frames", status=410)["error"]["code"] == "version_gone"


def test_delete_removes_a_map(client: Client) -> None:
    frames, files, _ = frames_of(client, 1)
    client.update("m", frames, files, {"cam": CAMERA})
    r = client.http.delete("/api/maps/m")
    assert r.status_code == 204 and client.check(r, None) is None
    assert client.get("/api/maps/m", status=404)["error"]["code"] == "map_not_found"
    r = client.http.delete("/api/maps/m")
    assert r.status_code == 404 and client.check(r, None)["error"]["code"] == "map_not_found"


# -- scale sources at their edges ----------------------------------------------------------------


def test_depth_that_the_map_cannot_compare_falls_back_to_the_camera_height(client: Client) -> None:
    frames, files, truth = frames_of(client, 3)
    shas = list(client.backend.truth)
    del client.backend.depth_m[shas[0]]  # no aligned depth in the map
    files["d0"] = png16(np.full((240, 320), 2000))
    files["d1"] = png16(np.zeros((240, 320)))  # no valid depth pixel
    frames[0]["depth"], frames[1]["depth"] = "d0", "d1"
    frames[2]["height_above_floor_m"] = 1.0
    frames[1]["height_above_floor_m"] = 0.0  # a camera at the floor tells no scale
    job = client.update("m", frames, files, {"cam": CAMERA})
    assert job["scale_source"] == "camera_height"


def test_heights_without_a_floor_and_a_map_without_a_cloud(client: Client,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    client.backend.floor_z = None
    real = client.backend.read

    def no_cloud(mapper_dir: Path) -> Any:
        m = real(mapper_dir)
        m.bounds = None
        return m

    monkeypatch.setattr(client.backend, "read", no_cloud)
    frames, files, _ = frames_of(client, 2)
    frames[0]["height_above_floor_m"] = 1.0
    job = client.update("m", frames, files, {"cam": CAMERA})
    assert job["scale_source"] == "monocular"
    s = client.get("/api/maps/m", "MapSummary")
    assert s["floor_z_m"] is None and s["bounds_m"]["min"] == s["bounds_m"]["min"]


# -- oms/2.1: capabilities, expected durations, per-frame masks ----------------------------------


def test_health_lists_capabilities_and_default_then_measured_timings(client: Client) -> None:
    h = client.get("/api/health", "Health")
    assert h["capabilities"] == ["O1", "O4"]
    reg = h["timings"]["registration_per_frame"]
    assert reg["source"] == "default" and reg["count"] == 0 and reg["median_s"] > 0
    frames, files, _ = frames_of(client, 3)
    client.update("m", frames, files, {"cam": CAMERA})
    t = client.get("/api/health", "Health")["timings"]
    assert t["registration_per_frame"]["source"] == "measured"
    assert t["registration_per_frame"]["count"] == 3 and t["fusion_per_frame"]["count"] == 3
    assert t["grid_read"]["source"] == "default"


def test_eta_counts_the_queue_ahead_and_the_job_s_own_frames(
        client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    client.service.jobs.predict = lambda n: 10.0 * n
    gate = threading.Event()
    real = client.backend.register

    def slow(*a: Any) -> Any:
        gate.wait(10)
        return real(*a)

    monkeypatch.setattr(client.backend, "register", slow)
    frames, files, _ = frames_of(client, 3)
    j1 = client.post_update("m", {"frames": frames[:2], "cameras": {"cam": CAMERA}}, files)
    assert 0 < j1["eta_s"] <= 20.0
    deadline = time.monotonic() + 10
    while client.service.jobs.running is None and time.monotonic() < deadline:
        time.sleep(0.01)
    j2 = client.post_update("m", {"frames": frames[2:]}, files)
    assert 20.0 < j2["eta_s"] <= 30.0  # the running job's remainder plus its own 10 s
    running = client.get(f"/api/jobs/{j1['job_id']}", "Job")
    assert running["eta_s"] <= 20.0 and running["ended_at"] is None
    gate.set()
    done = client.wait(j2["job_id"])
    assert done["eta_s"] == 0.0 and done["ended_at"] and done["latest_version"] == 2


def test_a_frame_mask_is_combined_with_the_camera_s(client: Client) -> None:
    import io

    from PIL import Image

    def png(rows: slice) -> bytes:
        m = np.zeros((240, 320), np.uint8)
        m[rows, :] = 255
        buf = io.BytesIO()
        Image.fromarray(m).save(buf, format="PNG")
        return buf.getvalue()

    frames, files, _ = frames_of(client, 2)
    frames[0]["ignore_mask"] = "fm"
    files["fm"] = png(slice(0, 120))
    files["cm"] = png(slice(120, 240))
    job = client.update("m", frames, files, {"cam": {**CAMERA, "ignore_mask": "cm"}})
    # both halves ignored: nothing left to measure, a plain frame
    assert job["inputs"][0]["reason"] == "plain"
    frames, files, _ = frames_of(client, 1, start=2)
    frames[0]["ignore_mask"] = "fm"
    files["fm"] = png(slice(0, 60))
    job = client.update("n", frames, files, {"cam": CAMERA})
    assert job["inputs"][0]["reason"] == "no_overlap"  # inpainted: the mapper's input changed
    doc = client.post_update("m", {"frames": [{**frames[0], "ignore_mask": "x"}]},
                             {**files, "x": png16(np.zeros((10, 10)))}, status=400)
    assert doc["error"]["details"]["field"] == "frames[0].ignore_mask"


# -- the main path: a cold start with metric priors, then drifting or none --------------------


def test_metric_first_frames_then_drifting_and_none_land_in_the_priors_frame(
        client: Client) -> None:
    """Every robot starts with metric priors (the start pose on the floor is the origin, the
    camera pose from kinematics); its later frames come as drifting or none."""
    client.backend.gauge = Sim3(0.37, anchor_rot(25), np.array([3.0, -2.0, 0.4]))
    frames, files, truth = frames_of(client, 3, prior=lambda i, T: {
        "trust": "metric", "T_map_cam": T.tolist(), "sigma_pos_m": 0.01, "sigma_rot_deg": 0.5})
    first = client.update("m", frames, files, {"cam": CAMERA})
    assert first["scale_source"] == "priors"
    drift = Sim3(1.0, anchor_rot(-30), np.array([5.0, 5.0, 0.0]))  # odometry's own frame
    frames, files, truth = frames_of(client, 3, start=3, prior=lambda i, T: {
        "trust": "drifting", "T_map_cam": drift.pose(T).tolist()} if i < 5 else None)
    job = client.update("m", frames, files)
    assert job["state"] == "committed" and job["scale_source"] == "priors"
    for i, t in zip(job["inputs"], truth, strict=True):
        assert centre_error(i["T_map_cam"], t) < 1e-6
        assert rotation_deg(np.asarray(i["T_map_cam"])[:3, :3].T @ t[:3, :3]) < 0.1
    assert job["inputs"][0]["prior_residual_m"] < 1e-6 and job["inputs"][2]["prior_residual_m"] is None
    assert job["quality"]["gauge_shift"]["translation_m"] < 1e-6
    frames, files, truth = frames_of(client, 2, start=0, seed=9)
    job = client.update("m", frames, files)  # trust none: anchored on the stored frames
    for i, t in zip(job["inputs"], truth, strict=True):
        assert centre_error(i["T_map_cam"], t) < 1e-6


def anchor_rot(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
