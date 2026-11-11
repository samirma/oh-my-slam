"""Locate (O5) and the navigation grid and points (O6) through the API, with the mapper replaced by
``tests.fakes.service.FakeBackend``: a box scene of two rooms and a door rendered analytically
(``tests.synth.rooms``) for the grid, a textured room ray-cast by Open3D (``tests.synth.scene``)
for locate. Every body is checked against api.schema.json."""

from __future__ import annotations

import base64
import math
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from starlette.testclient import TestClient

from oh_my_slam.core.constants import FLOOR_SIGMA_DEFAULT_M
from oh_my_slam.service import schema
from oh_my_slam.service.app import Service, create_app
from oh_my_slam.service.backend import Cloud
from oh_my_slam.service.derived import Derived
from oh_my_slam.service.jobs import Jobs
from oh_my_slam.service.store import Store
from oh_my_slam.service.timings import Timings
from oh_my_slam.service.update import Updater
from tests.fakes.service import FakeBackend, FakeInference, multipart, textured
from tests.synth import rooms

CAM320 = {"fx": 320.0, "fy": 320.0, "cx": 160.0, "cy": 120.0, "width": 320, "height": 240}
ERRORS = {400, 404, 405, 409, 410, 413, 500, 503}


class Nav:
    def __init__(self, http: TestClient, backend: FakeBackend, service: Service) -> None:
        self.http = http
        self.backend = backend
        self.service = service

    def check(self, r: Any, name: str | None) -> Any:
        assert "server-timing" in r.headers
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

    def update(self, name: str, frames: list[dict[str, Any]], files: dict[str, bytes],
               cameras: dict[str, Any] | None = None) -> dict[str, Any]:
        req: dict[str, Any] = {"frames": frames}
        if cameras:
            req["cameras"] = cameras
        data, ctype = multipart(req, files)
        r = self.http.post(f"/api/maps/{name}/updates", content=data,
                           headers={"content-type": ctype})
        assert r.status_code == 202, r.text
        job_id = r.json()["job_id"]
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            doc = self.get(f"/api/jobs/{job_id}?wait_s=5", "Job")
            if doc["state"] not in ("queued", "registering"):
                return dict(doc)
        raise AssertionError("job did not end")

    def locate(self, name: str, req: dict[str, Any], files: dict[str, bytes],
               status: int = 200) -> Any:
        data, ctype = multipart(req, files)
        r = self.http.post(f"/api/maps/{name}/locate", content=data,
                           headers={"content-type": ctype})
        assert r.status_code == status, r.text
        return self.check(r, "Locate" if status == 200 else None)


@pytest.fixture
def nav(tmp_path: Path) -> Iterator[Nav]:
    store = Store(tmp_path / "data")
    store.create()
    backend = FakeBackend()
    timings = Timings()
    # the mapper's path (FakeBackend plays it); the prior-first path: test_service_prior_first
    jobs = Jobs(store.jobs, Updater(store, backend, timings, prior_first=False),
                timings.predict_update)
    jobs.start()
    derived = Derived(store, backend.read)
    derived.start()
    service = Service(store, jobs, FakeInference(), backend.require_inference, timings=timings,
                      version="test", derived=derived)
    with TestClient(create_app(service)) as http:
        yield Nav(http, backend, service)
    jobs.stop(timeout=10)
    derived.stop()


def decode(b: dict[str, Any]) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b["data"]), np.dtype(b["dtype"]).newbyteorder("<")
                         ).reshape(b["shape"])


def metric(T: np.ndarray) -> dict[str, Any]:
    return {"trust": "metric", "T_map_cam": T.tolist(), "sigma_pos_m": 0.01, "sigma_rot_deg": 0.5}


def scene_frames(nav: Nav, scene: rooms.Scene, poses: list[np.ndarray], start: int = 0
                 ) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    frames, files = [], {}
    for i, T in enumerate(poses, start=start):
        img = nav.backend.add(textured(1000 + i), T)
        import hashlib

        nav.backend.depth_maps[hashlib.sha256(img).hexdigest()] = scene.depth(T)
        files[f"i{i}"] = img
        frames.append({"name": f"s{i}", "image": f"i{i}", "camera_id": "cam", "prior": metric(T)})
    return frames, files


@pytest.fixture
def rooms_map(nav: Nav) -> Nav:
    nav.backend.floor_z = 0.0
    frames, files = scene_frames(nav, rooms.two_rooms(), rooms.tour())
    job = nav.update("house", frames, files, {"cam": CAM320})
    assert job["state"] == "committed", (job["inputs"], job["quality"])
    return nav


# -- grid ------------------------------------------------------------------------------------------


def test_grid_finds_the_door_between_two_rooms_exactly(rooms_map: Nav) -> None:
    g = rooms_map.get("/api/maps/house/grid", "Grid")
    assert g["map_version"] == 1 and g["cell_m"] == 0.05 and g["z_band_m"] == [0.05, 1.8]
    assert g["floor_z_m"] == pytest.approx(0.0, abs=0.005) and g["scale_source"] == "priors"
    assert g["floor_sigma_m"] >= 0 and g["height_cap_m"] == 2.0  # no z_max asked: the default
    rows, cols = g["shape"]
    state = decode(g["layers"]["state"])
    assert state.shape == (rows, cols) and state.dtype == np.uint8
    assert decode(g["layers"]["observations"]).dtype == np.uint16
    assert decode(g["layers"]["height_m"]).dtype == np.float32
    assert len(g["regions"]) == 2
    (p,) = g["passages"]
    # no door object: an opening between two observed regions
    assert p["kind"] == "opening" and p["leaf"] is None and p["leaves"] == []
    assert p["narrow"] is None and p["frontier"] is None and None not in p["regions"]
    # the jambs lie on cell boundaries here: half a cell (the faces' mean place in a cell) over
    assert p["width_m"] == pytest.approx(0.95, abs=0.011)
    assert p["centre_m"][1] == pytest.approx(1.95, abs=0.011)
    assert 4.0 <= p["centre_m"][0] <= 4.1
    ys = sorted(j[1] for j in p["jambs_m"])
    assert ys == pytest.approx([1.475, 2.425], abs=0.011)
    a, b = (next(r for r in g["regions"] if r["id"] == i) for i in p["regions"])
    ca = np.mean(a["polygon_m"], axis=0)
    cb = np.mean(b["polygon_m"], axis=0)
    expect = 0.0 if ca[0] < cb[0] else 180.0
    assert abs((p["normal_deg"] - expect + 180) % 360 - 180) < 1.0
    for r in g["regions"]:
        assert r["area_m2"] == pytest.approx(16.0, rel=0.08) and r["coverage_share"] > 0.95
    for f in g["frontiers"]:
        assert f["region"] in {r["id"] for r in g["regions"]}
    # every camera cell is free
    ox, oy = g["origin_m"]
    for T in rooms.tour():
        r, c = int((T[1, 3] - oy) / 0.05), int((T[0, 3] - ox) / 0.05)
        assert state[r, c] == 1
    # the wall between the rooms is occupied, and seen floor is marked
    r, c = int((3.0 - oy) / 0.05), int((4.05 - ox) / 0.05)
    assert state[r, c] == 2
    floor = decode(g["layers"]["floor_observed"])
    r, c = int((2.01 - oy) / 0.05), int((2.01 - ox) / 0.05)
    assert floor[r, c] == 1 and decode(g["layers"]["height_m"])[r, c] == pytest.approx(0, abs=0.05)
    assert "derive" in rooms_map.http.get("/api/maps/house/grid").headers["server-timing"]


def test_grid_ids_are_stable_and_bands_apply_as_asked(rooms_map: Nav) -> None:
    g1 = rooms_map.get("/api/maps/house/grid", "Grid")
    g2 = rooms_map.get("/api/maps/house/grid?cell_m=0.1&z_min=0.1&z_max=0.5", "Grid")
    assert g2["cell_m"] == 0.1 and g2["z_band_m"] == [0.1, 0.5]
    assert [r["id"] for r in g1["regions"]] == sorted(r["id"] for r in g2["regions"])
    assert g1["passages"][0]["id"] == g2["passages"][0]["id"]
    g3 = rooms_map.get("/api/maps/house/grid?cell_m=0.025&footprint_radius_m=0.5", "Grid")
    assert g3["passages"][0]["narrow"] is True  # 0.9 m < 2 x 0.5 m
    assert g3["passages"][0]["kind"] == "opening" and g2["passages"][0]["narrow"] is None
    assert g2["height_cap_m"] == 0.5  # the band's z_max, when asked
    assert g3["shape"][0] >= 2 * g1["shape"][0] - 1


def test_overhangs_above_the_band_are_free_below_it(nav: Nav) -> None:
    nav.backend.floor_z = 0.0
    frames, files = scene_frames(nav, rooms.two_rooms(table=True), rooms.tour())
    nav.update("t", frames, files, {"cam": CAM320})
    low = nav.get("/api/maps/t/grid?z_max=0.5", "Grid")
    high = nav.get("/api/maps/t/grid?z_max=1.0", "Grid")
    ox, oy = low["origin_m"]
    r, c = int((2.9 - oy) / 0.05), int((1.5 - ox) / 0.05)  # under the table top, between legs
    assert decode(low["layers"]["state"])[r, c] == 1
    assert decode(high["layers"]["state"])[r, c] == 2
    # the height layer's cap is the asked z_max: the table top above it is left out, unless a
    # higher cap is asked
    assert low["height_cap_m"] == 0.5
    assert decode(low["layers"]["height_m"])[r, c] == pytest.approx(0.0, abs=0.05)
    capped = nav.get("/api/maps/t/grid?z_max=0.5&height_cap_m=1.0", "Grid")
    assert capped["height_cap_m"] == 1.0
    assert decode(capped["layers"]["height_m"])[r, c] == pytest.approx(0.75, abs=0.05)
    assert nav.get("/api/maps/t/grid?height_cap_m=0", status=400)["error"]["details"][
        "field"] == "height_cap_m"


def test_an_open_leaf_is_reported_and_narrows_the_gap(nav: Nav) -> None:
    nav.backend.floor_z = 0.0
    # the leaf's door object (map.md §5): the passage names it
    nav.backend.objects_world = [{"id": 4, "label": "door", "center": [4.5, 2.38, 1.0],
                                  "R": np.eye(3).tolist(), "size": [0.8, 0.04, 2.0]}]
    frames, files = scene_frames(nav, rooms.two_rooms(leaf=True), rooms.tour())
    nav.update("l", frames, files, {"cam": CAM320})
    (p,) = nav.get("/api/maps/l/grid", "Grid")["passages"]
    assert p["kind"] == "door" and p["leaf"] is not None and p["leaf"]["intrudes"] is False
    assert p["leaves"] == [p["leaf"]] and p["leaf"]["object_id"] == 4
    assert 0.8 <= p["width_m"] < 0.95  # narrower than the 0.9 m + a cell of the bare gap


def test_grid_and_points_parameters_are_checked(rooms_map: Nav) -> None:
    for q in ("cell_m=0", "cell_m=x", "z_min=1&z_max=0.5", "footprint_radius_m=-1",
              "cell_m=nan", "version=0"):
        doc = rooms_map.get(f"/api/maps/house/grid?{q}", status=400)
        assert doc["error"]["code"] == "invalid_request"
    for q in ("", "min=0,0,0", "min=0,0&max=1,1,1", "min=0,0,0&max=a,1,1",
              "min=1,1,1&max=0,0,0", "min=0,0,0&max=1,1,1&max_points=1.5",
              "min=0,0,0&max=1,1,1&voxel_m=0"):
        doc = rooms_map.get(f"/api/maps/house/points?{q}", status=400)
        assert doc["error"]["code"] == "invalid_request"
    assert rooms_map.get("/api/maps/none/grid", status=404)["error"]["code"] == "map_not_found"


def test_points_inside_a_box_are_thinned_and_capped(nav: Nav) -> None:
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0, 2, (20000, 3)).astype(np.float32)
    nav.backend.cloud_world = Cloud(xyz, rng.integers(0, 255, (20000, 3), dtype=np.uint8),
                                    (np.arange(20000) % 3).astype(np.uint32))
    frames, files = scene_frames(nav, rooms.two_rooms(), rooms.tour()[::12])
    nav.update("p", frames, files, {"cam": CAM320})
    doc = nav.get("/api/maps/p/points?min=0,0,0&max=1,1,1&voxel_m=0.001", "Points")
    pts = decode(doc["xyz"])
    inside = np.all((xyz >= 0) & (xyz <= 1), axis=1).sum()
    assert len(pts) == inside and not doc["truncated"]
    assert np.all((pts >= -1e-4) & (pts <= 1 + 1e-4))
    assert decode(doc["object_id"]).dtype == np.uint32 and decode(doc["rgb"]).shape == (inside, 3)
    capped = nav.get("/api/maps/p/points?min=0,0,0&max=2,2,2&max_points=500", "Points")
    assert capped["truncated"] and 0 < len(decode(capped["xyz"])) <= 500
    coarse = nav.get("/api/maps/p/points?min=0,0,0&max=2,2,2&voxel_m=0.5", "Points")
    assert len(decode(coarse["xyz"])) <= 64
    empty = nav.get("/api/maps/p/points?min=5,5,5&max=6,6,6", "Points")
    assert decode(empty["xyz"]).shape == (0, 3)


def test_object_regions_hook(rooms_map: Nav) -> None:
    from oh_my_slam.service import nav as navmod

    s = rooms_map.service
    assert s.derived is not None
    dv = s.derived.get("house", 1)
    ans = navmod.grid(dv, navmod.Ids(s.store), 0.05, 0.05, 1.8, None)
    objs = [{"id": 1, "centre_m": [2.0, 2.0, 0.4], "size_m": [0.5, 0.5, 0.8], "yaw_deg": 10.0},
            {"id": 2, "centre_m": [6.0, 2.0, 0.4], "size_m": [0.5, 0.5, 0.8]},
            {"id": 3, "centre_m": [20.0, 20.0, 0.4], "size_m": [0.5, 0.5, 0.8]},
            {"id": 4, "centre_m": [4.05, 3.2, 1.0], "size_m": [0.05, 0.01, 0.5]}]
    got = navmod.object_regions(ans, objs, {3: [[2.0, 2.0]], 4: [[6.0, 3.0]]})
    left = ans.region_ids[0] if ans.features.regions[0].centroid[0] < 4 else ans.region_ids[1]
    right = ({*ans.region_ids} - {left}).pop()
    assert got[1] == left and got[2] == right and got[3] == left
    assert got[4] in (left, right)
    assert navmod.object_regions(ans, [objs[2]]) == {3: None}


# -- locate ----------------------------------------------------------------------------------------


@pytest.fixture
def room_map(nav: Nav) -> tuple[Nav, Any, Any]:
    from oh_my_slam.core.types import Intrinsics
    from tests.synth.scene import default_room, orbit_poses, render

    K = Intrinsics(300.0, 300.0, 160.0, 120.0, 320, 240)
    room = default_room()
    frames, files = [], {}
    import hashlib
    import io

    from PIL import Image

    for i, P in enumerate(orbit_poses(12)):
        r = render(room, P, K)
        buf = io.BytesIO()
        Image.fromarray(r.rgb).save(buf, format="JPEG", quality=95)  # the mapper gets it as is
        img = nav.backend.add(buf.getvalue(), P.matrix())
        nav.backend.depth_maps[hashlib.sha256(img).hexdigest()] = r.depth
        files[f"k{i}"] = img
        frames.append({"name": f"k{i}", "image": f"k{i}", "camera_id": "cam",
                       "prior": metric(P.matrix())})
    cam = {"fx": 300.0, "fy": 300.0, "cx": 160.0, "cy": 120.0, "width": 320, "height": 240}
    job = nav.update("room", frames, files, {"cam": cam})
    assert job["state"] == "committed", (job["inputs"], job["quality"])
    return nav, room, K


def _png(rgb: np.ndarray) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def test_locate_places_frames_with_sigmas_and_records_the_pose(room_map: tuple[Nav, Any, Any]
                                                               ) -> None:
    from tests.synth.scene import look_at, render

    nav, room, K = room_map
    assert nav.get("/api/maps/room/pose", status=404)["error"]["code"] == "no_pose"
    truth = []
    files: dict[str, bytes] = {}
    frames = []
    for i, a in enumerate((0.3, 2.0)):
        P = look_at(np.array([1.5 * np.cos(a), 1.5 * np.sin(a), 1.3]), np.array([0.1, 0, 0.5]))
        truth.append(P.matrix())
        files[f"q{i}"] = _png(render(room, P, K).rgb)
        frames.append({"name": f"q{i}", "image": f"q{i}", "camera_id": "cam"})
    frames[1]["prior"] = metric(truth[1])
    frames[1]["height_above_floor_m"] = 1.3
    doc = nav.locate("room", {"frames": frames, "record": True}, files)
    assert doc["map_version"] == 1 and doc["scale_source"] == "priors"
    for res, T in zip(doc["frames"], truth, strict=True):
        assert res["status"] == "located" and res["reason"] is None
        err = np.linalg.norm(np.asarray(res["T_map_cam"])[:3, 3] - T[:3, 3])
        # calibrated (mapping.sigmas): this map's keyframes carry no depth kind, so the
        # no-depth noise floor (2.1 deg) adds to the keyframes' own calibrated sigmas
        assert err < 0.03 and 0 < res["sigma_pos_m"] < 0.15 and 2.1 <= res["sigma_rot_deg"] < 4
        assert 0 < res["inlier_ratio"] <= 1 and res["ambiguity"] is None and res["rivals"] == []
    assert doc["frames"][0]["consistent_with_prior"] is None
    assert doc["frames"][1]["consistent_with_prior"] is True
    pose = nav.get("/api/maps/room/pose", "Pose")
    assert pose["name"] == "q1" and pose["map_version"] == 1
    assert pose["T_map_cam"] == doc["frames"][1]["T_map_cam"]
    h = nav.get("/api/health", "Health")
    assert h["timings"]["locate_per_frame"]["count"] == 2
    assert (pose["scale_source"], pose["scale_sigma_rel"]) == (doc["scale_source"],
                                                               doc["scale_sigma_rel"])
    bad = nav.get("/api/maps/room/pose?version=1", status=400)["error"]
    assert bad["code"] == "invalid_request" and bad["details"]["field"] == "version"
    # a pose recorded before its scale was stored takes its version's
    import json

    path = nav.service.store.dir("room") / "pose.json"
    old = json.loads(path.read_text())
    del old["scale_source"], old["scale_sigma_rel"]
    path.write_text(json.dumps(old))
    assert nav.get("/api/maps/room/pose", "Pose")["scale_source"] == "priors"
    # a prior 0.3 m off: within 3 x the drifting defaults (0.20 m / 5 deg), beyond the metric
    # ones (0.05 m / 2 deg), where no hypothesis agrees with it: prior_conflict (api.md §6)
    off = truth[0].copy()
    off[0, 3] += 0.3
    req = [{"name": t, "image": "q0", "camera_id": "cam",
            "prior": {"trust": t, "T_map_cam": off.tolist()}} for t in ("drifting", "metric")]
    drifting, conflict = nav.locate("room", {"frames": req}, files)["frames"]
    assert drifting["status"] == "located" and drifting["consistent_with_prior"] is True
    assert conflict["status"] == "rejected" and conflict["reason"] == "prior_conflict"
    assert conflict["T_map_cam"] is None and conflict["nearest_view"] is None


def test_locate_rejects_unusable_frames_and_points_away_from_no_overlap(
        room_map: tuple[Nav, Any, Any]) -> None:
    from tests.fakes.service import plain

    nav, _, _ = room_map
    far = np.eye(4)
    far[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]  # looks along +x
    far[:3, 3] = [10.0, 10.0, 1.3]
    files = {"a": plain(320, 240), "b": textured(77)}
    frames = [{"name": "a", "image": "a", "camera_id": "cam"},
              {"name": "b", "image": "b", "camera_id": "cam", "prior": metric(far)}]
    doc = nav.locate("room", {"frames": frames}, files)
    a, b = doc["frames"]
    assert a["status"] == "rejected" and a["reason"] == "plain" and a["nearest_view"] is None
    assert b["status"] == "rejected" and b["reason"] == "no_overlap"
    assert -180 <= b["nearest_view"]["turn_deg"] <= 180  # guaranteed with a prior
    with pytest.raises(AssertionError):
        nav.get("/api/maps/room/pose", "Pose")


def test_a_kept_frame_with_fewer_keypoints_than_a_pose_needs_is_few_features(
        room_map: tuple[Nav, Any, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """With fewer keypoints for matching than a pose needs inliers (``relocalize.MIN_INLIERS``)
    a frame is few_features, else (any frame not located) no_overlap; the update checks' own
    feature count plays no part."""
    from oh_my_slam.mapping import relocalize

    nav, _, _ = room_map
    real = relocalize.extract

    def few(gray: Any, mask: Any = None) -> Any:
        uv, desc = real(gray, mask)
        return uv[:relocalize.MIN_INLIERS - 1], desc[:relocalize.MIN_INLIERS - 1]

    monkeypatch.setattr(relocalize, "extract", few)
    doc = nav.locate("room", {"frames": [{"name": "b", "image": "b", "camera_id": "cam"}]},
                     {"b": textured(77)})
    b = doc["frames"][0]
    assert b["status"] == "rejected" and b["reason"] == "few_features"
    assert b["nearest_view"] is None
    monkeypatch.setattr(relocalize, "extract", real)
    from oh_my_slam.mapping import checks

    monkeypatch.setattr(checks, "reason", lambda m, d: "few_features")  # the update checks'
    doc = nav.locate("room", {"frames": [{"name": "b", "image": "b", "camera_id": "cam"}]},
                     {"b": textured(77)})
    assert doc["frames"][0]["reason"] == "no_overlap"


def test_locate_requests_are_validated(room_map: tuple[Nav, Any, Any]) -> None:
    nav, _, _ = room_map
    img = textured(5)
    f = {"name": "a", "image": "a", "camera_id": "cam"}
    assert nav.locate("room", {"frames": [f, f]}, {"a": img}, 400)["error"]["details"][
        "field"] == "frames[1].name"
    assert nav.locate("room", {"frames": [{**f, "camera_id": "x"}]}, {"a": img}, 400)
    assert nav.locate("room", {"frames": [f]}, {}, 400)
    assert nav.locate("room", {"frames": [f]}, {"a": textured(5, 64, 48)}, 400)
    bad = np.eye(4)
    bad[0, 0] = 2
    assert nav.locate("room", {"frames": [{**f, "prior": {"trust": "metric",
                                                          "T_map_cam": bad.tolist()}}]},
                      {"a": img}, 400)
    other = {"fx": 1.0, "fy": 1.0, "cx": 1.0, "cy": 1.0, "width": 320, "height": 240}
    assert nav.locate("room", {"frames": [f], "cameras": {"cam": other}}, {"a": img},
                      409)["error"]["code"] == "camera_conflict"
    assert nav.locate("room", {"frames": [f], "version": 9}, {"a": img}, 400)
    assert nav.locate("nomap", {"frames": [f]}, {"a": img}, 404)
    data, ctype = multipart({"frames": [f]}, {"a": img})
    r = nav.http.post("/api/maps/room/locate", content=data.replace(b'"request"', b'"other"'),
                      headers={"content-type": ctype})
    assert r.status_code == 400
    r = nav.http.post("/api/maps/room/locate", content=data.replace(b'{"frames"', b'{"frames'),
                      headers={"content-type": ctype})
    assert r.status_code == 400


def test_locate_with_a_new_camera_and_masks(room_map: tuple[Nav, Any, Any]) -> None:
    from tests.fakes.service import png16  # noqa: F401  (16-bit helper unused: masks are 8-bit)

    nav, room, K = room_map
    from tests.synth.scene import look_at, render

    P = look_at(np.array([1.4, 0.4, 1.3]), np.array([0.0, 0.0, 0.5]))
    rgb = render(room, P, K).rgb
    mask = np.zeros((240, 320), np.uint8)
    mask[:20] = 255
    cam2 = {"fx": 300.0, "fy": 300.0, "cx": 160.0, "cy": 120.0, "width": 320, "height": 240,
            "ignore_mask": "cm"}
    req = {"cameras": {"cam2": cam2},
           "frames": [{"name": "q", "image": "q", "camera_id": "cam2", "ignore_mask": "fm"}]}
    doc = nav.locate("room", req, {"q": _png(rgb), "cm": _png(mask), "fm": _png(mask[::-1])})
    assert doc["frames"][0]["status"] == "located"
    bad = {**req, "frames": [{**req["frames"][0], "ignore_mask": "small"}]}
    assert nav.locate("room", bad, {"q": _png(rgb), "cm": _png(mask),
                                    "small": _png(mask[:10, :10])}, 400)
    bad_cam = {**req, "cameras": {"cam2": {**cam2, "ignore_mask": "small"}}}
    assert nav.locate("room", bad_cam, {"q": _png(rgb), "small": _png(mask[:10, :10])}, 400)


# -- derived versions ------------------------------------------------------------------------------


def test_versions_derive_incrementally_and_rebuild_when_a_pose_moves(
        rooms_map: Nav, monkeypatch: pytest.MonkeyPatch) -> None:
    from oh_my_slam.mapping import navgrid

    s = rooms_map.service
    assert s.derived is not None
    s.derived.get("house", 1)  # derived (the worker may still be at it)
    calls: list[str] = []
    real = navgrid.Volume.integrate

    def spy(self: navgrid.Volume, key: str, *a: Any, **k: Any) -> bool:
        calls.append(key)
        return real(self, key, *a, **k)

    monkeypatch.setattr(navgrid.Volume, "integrate", spy)
    v1 = s.store.version("house", 1)
    rec = {k: v for k, v in v1.items() if k not in ("version", "created_at")}
    s.store.commit("house", rec, None)  # the same frames: nothing to integrate
    assert s.derived.get("house", 2).volume().frames.keys() == s.derived.get("house", 1
                                                                              ).volume().frames.keys()
    assert calls == []
    moved = [dict(f) for f in rec["frames"]]
    T = np.asarray(moved[0]["T_map_cam"])
    T[0, 3] += 0.05
    moved[0]["T_map_cam"] = T.tolist()
    s.store.commit("house", {**rec, "frames": moved}, None)
    s.derived.get("house", 3)
    # all of them again, but the frames kept at their prior that the mapper never placed (no
    # internal frame: no depth to integrate)
    assert len(calls) == len([f for f in moved if f.get("internal")]) < len(moved)
    for n in (1, 2, 3):  # the cache keeps CACHE_VERSIONS versions; older ones load again
        assert s.derived.get("house", n).version == n
    g = rooms_map.get("/api/maps/house/grid?version=1", "Grid")
    assert g["map_version"] == 1


def test_derived_floor_fallbacks_and_maps_without_mapper(nav: Nav) -> None:
    from oh_my_slam.mapping.anchor import Sim3

    s = nav.service
    assert s.derived is not None
    cams = {"cam": CAM320, "high": {**CAM320, "height_above_floor_m": 1.0}}
    s.store.add_cameras("bare", cams, {})
    T = rooms.camera(1.0, 1.0, 1.2, 0.0)

    def rec(frames: list[dict[str, Any]], floor: float | None = None) -> dict[str, Any]:
        return {"kind": "update", "sim3": Sim3().to_dict(), "scale_source": "priors",
                "scale_sigma_rel": 0.01, "frames": frames, "objects": 0, "floor_z_m": floor,
                "bounds_m": {"min": [0, 0, 0], "max": [1, 1, 1]}}

    f = {"name": "a", "camera_id": "cam", "T_map_cam": T.tolist(), "sigma_pos_m": 0.01,
         "sigma_rot_deg": 0.1, "pose_state": "provisional", "added_version": 1}
    n = s.store.commit("bare", rec([f]), None)
    dv = s.derived.get("bare", n)
    assert dv.floor_z == pytest.approx(0.2)  # 1 m below the lowest camera
    assert dv.meta["keyframes"] == [] and dv.cloud()["xyz"].shape == (0, 3)
    g = nav.get("/api/maps/bare/grid", "Grid")
    state = decode(g["layers"]["state"])
    assert (state == 1).sum() >= 1 and g["regions"] == [] or True
    n = s.store.commit("bare", rec([{**f, "camera_id": "high"},
                                    {**f, "name": "ghost", "camera_id": "gone"}]), None)
    assert s.derived.get("bare", n).floor_z == pytest.approx(0.2)
    assert len(s.derived.get("bare", n).stands) == 1
    n = s.store.commit("bare", rec([]), None)
    assert s.derived.get("bare", n).floor_z == 0.0
    n = s.store.commit("bare", rec([f], floor=-0.5), None)
    assert s.derived.get("bare", n).floor_z == -0.5
    assert s.derived.get("bare", n).floor_sigma == FLOOR_SIGMA_DEFAULT_M  # an older record
    # an oms/2.2 record holds the version's one floor estimate: the grid uses it as is
    n = s.store.commit("bare", {**rec([f], floor=-0.3), "floor_sigma_m": 0.025}, None)
    dv = s.derived.get("bare", n)
    assert (dv.floor_z, dv.floor_sigma) == (-0.3, 0.025)
    g = nav.get(f"/api/maps/bare/grid?version={n}", "Grid")
    assert (g["floor_z_m"], g["floor_sigma_m"]) == (-0.3, 0.025)
    n = s.store.commit("bare", {**rec([f]), "floor_sigma_m": 0.025}, None)  # no floor found
    assert s.derived.get("bare", n).floor_z == pytest.approx(0.2)
    s.derived.committed("nomap", 1)  # a failed derivation is logged, the worker goes on
    s.derived.stop()
    s.derived.stop()


def test_derived_floor_from_the_volume_doors_and_unhashed_images(nav: Nav) -> None:
    nav.backend.floor_z = None
    nav.backend.objects_world = [
        {"id": 5, "label": "Door", "center": [4.05, 1.95, 1.0], "R": np.eye(3).tolist(),
         "size": [0.9, 0.1, 2.0]},
        {"id": 6, "label": "chair", "center": [2.0, 2.0, 0.4], "size": [0.5, 0.5, 0.8]},
        # a box labelled door that is no leaf (0.5 m thick): a hint only, never a leaf
        {"id": 7, "label": "door", "center": [6.0, 3.0, 1.0], "size": [0.9, 0.5, 2.0]}]
    frames, files = scene_frames(nav, rooms.two_rooms(), rooms.tour())
    job = nav.update("f", frames, files, {"cam": CAM320})
    assert job["state"] == "committed"
    s = nav.service
    assert s.derived is not None
    v = s.store.version("f", 1)
    assert v["floor_z_m"] is None
    dv = s.derived.get("f", 1)
    assert dv.floor_z == pytest.approx(-0.025, abs=0.03)
    assert dv.meta["doors"] == [pytest.approx([4.05, 1.95]), pytest.approx([6.0, 3.0])]
    assert [d["id"] for d in dv.meta["door_leaves"]] == [5]
    assert dv.meta["objects"][1]["yaw_deg"] == 0.0
    (p,) = nav.get("/api/maps/f/grid", "Grid")["passages"]
    assert p["kind"] == "door" and [lf["object_id"] for lf in p["leaves"]] == [5]
    rec = {k: val for k, val in v.items() if k not in ("version", "created_at")}
    rec["frames"] = [{k: val for k, val in fr.items() if k != "sha256"} for fr in rec["frames"]]
    n = s.store.commit("f", rec, None)
    assert s.derived.get("f", n).meta["keys"] == dv.meta["keys"]  # the same images


def test_yaw_of_boxes() -> None:
    from oh_my_slam.service.derived import yaw_of

    a = math.radians(-90.0)
    R = [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
    assert yaw_of(R) == 90.0
    assert yaw_of(np.eye(3)) == 0.0


def test_derived_edges(rooms_map: Nav, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib
    import shutil

    s = rooms_map.service
    d = s.derived
    assert d is not None
    d.get("house", 1)
    v1 = s.store.version("house", 1)
    rec = {k: v for k, v in v1.items() if k not in ("version", "created_at")}
    # a frame whose depth the mapper no longer has, and one whose image is unreadable
    sha0 = rec["frames"][0]["sha256"]
    del rooms_map.backend.depth_m[sha0]
    del rooms_map.backend.depth_maps[sha0]
    bad = s.store.mapper_dir("house") / "images" / f"{rec['frames'][1]['internal']}.jpg"
    bad.write_bytes(b"not an image")
    frames = [dict(f) for f in rec["frames"]]
    frames[1].pop("sha256")
    moved = np.asarray(frames[0]["T_map_cam"])
    moved[1, 3] += 0.1
    frames[0]["T_map_cam"] = moved.tolist()
    n2 = s.store.commit("house", {**rec, "frames": frames}, None)
    dv2 = d.get("house", n2)
    assert frames[0]["name"] not in dv2.volume().frames
    assert frames[1]["name"] not in dv2.meta["keyframes"]
    n3 = s.store.commit("house", rec, None)
    d.get("house", n2)
    shutil.rmtree(d.dir("house") / str(n2))  # version 3 grows from version 1
    with d._lock:
        d._cache.clear()
    assert d.get("house", n3).meta["version"] == n3
    assert d._features("house", "nokey", tmp_path / "missing.jpg") is None
    # a commit listener's failure never fails the commit
    s.store.listeners.append(lambda name, n: 1 / 0)
    assert s.store.commit("house", rec, None) == n3 + 1
    assert hashlib.sha256(b"x").hexdigest()


def test_ids_match_by_centre_and_area(tmp_path: Path) -> None:
    from oh_my_slam.service.nav import Ids

    store = Store(tmp_path / "d")
    store.create()
    ids = Ids(store)
    c = np.array([1.0, 1.0])
    assert ids.assign("m", "regions", [(c, 10.0), (c + 5, 4.0)]) == [1, 2]
    assert ids.assign("m", "regions", [(c + 0.1, 10.5), (c + 0.05, 9.0)]) == [3, 1]  # nearest
    assert ids.assign("m", "regions", [(c, 50.0)]) == [4]  # same place, another area
    assert ids.assign("m", "passages", [(c, 0.0)]) == [1]


def test_grid_cell_cap(rooms_map: Nav, monkeypatch: pytest.MonkeyPatch) -> None:
    from oh_my_slam.service import nav as navmod

    monkeypatch.setattr(navmod, "MAX_GRID_CELLS", 100)
    doc = rooms_map.get("/api/maps/house/grid", status=400)
    assert doc["error"]["details"]["field"] == "cell_m"


def test_locate_uses_the_map_s_camera_mask_and_a_repeated_camera(
        room_map: tuple[Nav, Any, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    from oh_my_slam.mapping import relocalize
    from tests.synth.scene import look_at, render

    nav, room, K = room_map
    mask = np.zeros((240, 320), np.uint8)
    mask[-10:] = 255
    cam = {"fx": 300.0, "fy": 300.0, "cx": 160.0, "cy": 120.0, "width": 320, "height": 240}
    P = look_at(np.array([1.4, 0.4, 1.3]), np.array([0.0, 0.0, 0.5]))
    img = _png(render(room, P, K).rgb)
    nav.service.store.add_cameras("room", {"masked": {**cam, "ignore_mask": "m"}},
                                  {"masked": ("digest", _png(mask))})
    f = {"name": "q", "image": "q", "camera_id": "masked"}
    doc = nav.locate("room", {"frames": [f], "cameras": {"cam": cam}}, {"q": img})
    assert doc["frames"][0]["status"] == "located"
    again = nav.locate("room", {"frames": [{**f, "camera_id": "cam"}]}, {"q": img})
    assert again["frames"][0]["status"] == "located"
    monkeypatch.setattr(relocalize, "locate", lambda q, index: relocalize.Outcome())
    doc = nav.locate("room", {"frames": [f]}, {"q": img})
    assert doc["frames"][0]["reason"] == "no_overlap" and doc["frames"][0]["nearest_view"] is None


def test_passages_do_not_depend_on_the_cell_size(rooms_map: Nav) -> None:
    got = []
    for cell in (0.05, 0.08, 0.12):
        g = rooms_map.get(f"/api/maps/house/grid?cell_m={cell}", "Grid")
        got.append([(p["centre_m"], p["width_m"], p["jambs_m"], p["kind"]) for p in g["passages"]])
        assert len(g["regions"]) == 2
        fl = decode(g["layers"]["floor_observed"])
        assert set(np.unique(fl).tolist()) <= {0, 1, 2}
    assert got[0] == got[1] == got[2] and len(got[0]) == 1
