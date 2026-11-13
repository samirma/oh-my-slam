"""``POST /api/frames/analyse`` (O7, api.md §7) through the Starlette app, the models replaced by
``tests.fakes.service.FakeBackend``: the ray-cast table scene (``tests.synth.table``) with exact
depth and exact masks, analysed in the map frame of an anchored map of it, in a client's own
frame and in the gravity-aligned camera frame. Every body is checked against api.schema.json."""

from __future__ import annotations

import base64
import hashlib
import io
import math
import threading
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from PIL import Image
from starlette.testclient import TestClient

from oh_my_slam.core import rle
from oh_my_slam.mapping import analysis
from oh_my_slam.mapping.anchored_objects import Box, View
from oh_my_slam.segmentation.api import Detection, color_for_id
from oh_my_slam.service import analyse
from oh_my_slam.service.analyse import Analyser
from oh_my_slam.service.app import Service, create_app
from oh_my_slam.service.cameras import Camera
from oh_my_slam.service.derived import Derived
from oh_my_slam.service.jobs import Jobs
from oh_my_slam.service.store import Store
from oh_my_slam.service.timings import Timings
from oh_my_slam.service.update import Updater
from tests.fakes.service import FakeBackend, FakeInference, multipart
from tests.synth.table import K_TABLE, TOP
from tests.unit.test_anchored_objects import EYES
from tests.unit.test_service_api import Client
from tests.unit.test_service_objects import CAM, _shot, frames

ALL = ["objects", "masks", "points", "range_scan", "depth"]
BOTTOM = 400  # the wrist camera's ignore mask: the image rows from here down


class An(Client):
    def analyse(self, req: dict[str, Any], files: dict[str, bytes], status: int = 200) -> Any:
        data, ctype = multipart(req, files)
        r = self.http.post("/api/frames/analyse", content=data, headers={"content-type": ctype})
        assert r.status_code == status, r.text
        return self.check(r, "Analysis")


def png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def decode(b: dict[str, Any]) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b["data"]), np.dtype(b["dtype"]).newbyteorder("<")
                         ).reshape(b["shape"])


def metres(depth_png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(depth_png))).astype(np.float32) * 0.001


@pytest.fixture(scope="module")
def an(tmp_path_factory: pytest.TempPathFactory) -> Iterator[An]:
    """A service with the anchored map ``t`` of the table scene (four keyframes, the label
    ``Red Mug``, the cameras ``cam`` and ``wrist``, whose mask ignores the bottom rows)."""
    store = Store(tmp_path_factory.mktemp("analyse") / "data")
    store.create()
    backend = FakeBackend()
    timings = Timings()
    updater = Updater(store, backend, timings)
    jobs = Jobs(store.jobs, updater, timings.predict_update)
    jobs.start()
    derived = Derived(store, backend.read)
    derived.start()
    service = Service(store, jobs, FakeInference(), backend.require_inference, timings=timings,
                      version="test", dry_run=updater.dry_run, derived=derived,
                      analyser=Analyser(store, backend))
    with TestClient(create_app(service)) as http:
        a = An(http, backend, service)
        fr, files = frames(a, "full", EYES[:4], "a")
        mask = np.zeros((K_TABLE.height, K_TABLE.width), np.uint8)
        mask[BOTTOM:] = 255
        files["wm"] = png(mask)
        job = a.update("t", fr, files, labels=["Red Mug"],
                       cameras={"cam": CAM, "wrist": {**CAM, "ignore_mask": "wm"}})
        assert job["state"] == "committed", job
        yield a
    jobs.stop(timeout=10)
    derived.stop()


@pytest.fixture(autouse=True)
def _knobs(an: An) -> Iterator[None]:
    """Each test starts with the backend's analysis knobs as a new one's."""
    b = an.backend
    yield
    b.up, b.other_detections, b.other_depth, b.ready, b.mono_scale = (0.0, -1.0, 0.0), [], None, \
        True, 1.0
    b.mono_depth.clear()


@pytest.fixture(scope="module")
def shot(an: An) -> tuple[bytes, bytes, np.ndarray, list[Detection]]:
    """A view of the table no keyframe has: its JPEG, depth PNG, pose and exact detections."""
    img, dep, T, dets = _shot("full", EYES[4])
    an.backend.detections[hashlib.sha256(img).hexdigest()] = dets
    return img, dep, T, dets


def on_a_surface(p: np.ndarray, tol: float = 0.06) -> bool:
    """Whether a map point (x, y) lies on a wall of the room or on the table, within ``tol``
    (half of it at the table's edge): a beam's range is its nearest point's, anywhere across
    its width."""
    wall = min(abs(abs(p[0]) - 3.0), abs(abs(p[1]) - 2.5)) < tol
    table = -tol / 2 <= p[0] <= 1.2 + tol / 2 and -tol / 2 <= p[1] <= 0.8 + tol / 2
    return wall or table


def beam_points(doc: dict[str, Any], origin: np.ndarray, kind: str, yaw_deg: float = 0.0
                ) -> np.ndarray:
    out = [origin[:2] + b["range_m"] * np.array([math.cos(math.radians(b["bearing_deg"] + yaw_deg)),
                                                 math.sin(math.radians(b["bearing_deg"] + yaw_deg))])
           for b in doc["range_scan"] if b["kind"] == kind]
    return np.array(out).reshape(-1, 2)


# -- the map frame -------------------------------------------------------------------------------


def test_a_frame_in_the_map_frame_gets_the_map_s_ids_and_changes_nothing(
        an: An, shot: tuple[bytes, bytes, np.ndarray, list[Detection]]) -> None:
    img, dep, T, dets = shot
    before = (an.get("/api/maps/t", "MapSummary"), an.get("/api/maps/t/versions", "Versions"),
              len(an.service.jobs.queued()), sorted(p.name for p in an.service.store.work.iterdir()))
    known = {o["label"]: o for o in an.get("/api/maps/t/objects", "Objects")["objects"]}
    calls, infers = getattr(an.backend, "interactive_calls", 0), len(an.backend.infers)
    count = an.get("/api/health", "Health")["timings"]["analyse_per_frame"]["count"]
    doc = an.analyse({"image": "i", "depth": "d", "camera_id": "cam", "map": "t",
                      "T_map_cam": T.tolist(), "outputs": ALL,
                      "labels": [" Apple ", "cup", "apple"]}, {"i": img, "d": dep})
    summary = before[0]
    assert doc["frame"] == "map" and doc["schema"] == "oms/2.7"
    assert (doc["scale_source"], doc["scale_sigma_rel"]) == (summary["scale_source"],
                                                             summary["scale_sigma_rel"])
    # a depth image with a pose asks the detector alone, as an interactive request, with the
    # map's vocabulary and the client's labels, from the default threshold
    assert an.backend.interactive_calls == calls + 1 and len(an.backend.infers) == infers
    assert an.backend.detected[-1] == ("frame.jpg", (640, 480))
    assert an.backend.prompted[-1] == ["Red Mug", "Apple", "cup", "apple"]
    assert an.backend.asked[-1] == (0.5, True)
    # every detection is a map object: its persistent id, its place in the map
    got = {o["label"]: o for o in doc["objects"]}
    assert set(got) == {"table", "box", "basket", "Apple"}
    R, c = T[:3, :3], T[:3, 3]
    masks = {d.label: d.mask for d in dets}
    for label, o in got.items():
        ref = known[label.lower()]
        assert o["id"] == ref["id"] and o["color"] == ref["color"] == list(color_for_id(o["id"]))
        assert o["detected_as"] == [label] and o["score"] == 0.9
        assert np.linalg.norm(np.subtract(o["centre_m"], ref["centre_m"])) < 0.06
        assert o["centre_cam_m"] == pytest.approx((np.array(o["centre_m"]) - c) @ R, abs=2e-4)
        assert o["sigma_centre_m"] > 0 and set(o["extent_seen"]) == {"x", "y", "z"}
        m = rle.decode(o["mask"])
        assert m.shape == (480, 640) and m.any() and not (m & ~masks[label.lower()]).any()
        rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
        assert o["bbox_px"] == [cols[0], rows[0], cols[-1] + 1, rows[-1] + 1]
    assert got["box"]["mean_rgb"] == pytest.approx([200, 40, 40], abs=3)
    assert got["box"]["centre_m"] == pytest.approx([0.25, 0.225, TOP + 0.06], abs=0.03)
    assert "state" not in got["box"] and "detection_votes" not in got["box"]
    # the best object per client label, in request order (equal labels name the same object)
    assert doc["matches"] == [
        {"label": "Apple", "object_id": known["apple"]["id"], "score": 0.9},
        {"label": "cup", "object_id": None, "score": None},
        {"label": "apple", "object_id": known["apple"]["id"], "score": 0.9}]
    # the points are in the map frame, one per 2 cm voxel, the objects' carrying their ids
    xyz, ids, rgb = (decode(doc["points"][k]) for k in ("xyz", "object_id", "rgb"))
    assert xyz.dtype == np.float32 and ids.dtype == np.uint32 and rgb.dtype == np.uint8
    assert xyz.shape == (len(ids), 3) == rgb.shape and len(xyz) > 5000
    assert len(np.unique(np.floor(xyz / 0.02).astype(int), axis=0)) > 0.99 * len(xyz)
    assert set(ids.tolist()) <= {0, *(o["id"] for o in got.values())}
    assert xyz[:, 2].min() == pytest.approx(0.0, abs=0.01)  # the floor
    box = Box(np.array(known["box"]["centre_m"]), np.array(known["box"]["size_m"]),
              known["box"]["yaw_deg"])
    mine = xyz[ids == known["box"]["id"]]
    assert len(mine) > 50 and box.contains(mine, 0.03).all()
    assert rgb[ids == known["box"]["id"]].mean(0) == pytest.approx([200, 40, 40], abs=12)
    # the range scan: bearings in the map frame, ranges from the camera
    scan = doc["range_scan"]
    assert [b["bearing_deg"] for b in scan] == [float(k) for k in range(-179, 181)]
    hits = beam_points(doc, c, "obstacle")
    assert len(hits) > 40 and all(on_a_surface(p) for p in hits)
    assert any(abs(p[0] + 3.0) < 0.06 for p in hits) and any(0 < p[0] < 1.2 for p in hits)
    ends = beam_points(doc, c, "floor_end")
    assert len(ends) > 5 and (np.abs(ends[:, 0]) < 3.0).all() and (np.abs(ends[:, 1]) < 2.5).all()
    unseen = [b for b in scan if b["kind"] == "unseen"]
    assert len(unseen) > 200 and all(b["range_m"] is None for b in unseen)
    yaw = math.degrees(math.atan2(T[1, 2], T[0, 2]))  # nothing behind the camera is seen
    assert all(abs((b["bearing_deg"] - yaw + 180) % 360 - 180) < 60 for b in scan
               if b["kind"] != "unseen")
    # the depth, as the input depth
    out = decode(doc["depth"])
    assert out.dtype == np.uint16 and out.shape == (480, 640)
    assert np.array_equal(out, np.asarray(Image.open(io.BytesIO(dep))))
    # nothing changed, nothing is left behind
    assert before == (an.get("/api/maps/t", "MapSummary"),
                      an.get("/api/maps/t/versions", "Versions"), len(an.service.jobs.queued()),
                      sorted(p.name for p in an.service.store.work.iterdir()))
    t = an.get("/api/health", "Health")["timings"]["analyse_per_frame"]
    assert t["count"] == count + 1 and t["source"] == "measured"


def test_monocular_depth_is_brought_onto_the_map(
        an: An, shot: tuple[bytes, bytes, np.ndarray, list[Detection]],
        monkeypatch: pytest.MonkeyPatch) -> None:
    img, dep, T, _ = shot
    z = metres(dep)
    an.backend.mono_depth[hashlib.sha256(img).hexdigest()] = z * 0.8  # the model's metres
    req = {"image": "i", "camera_id": "cam", "map": "t", "T_map_cam": T.tolist(),
           "outputs": ["objects", "depth"]}
    known = {o["label"]: o for o in an.get("/api/maps/t/objects", "Objects")["objects"]}

    def check(doc: dict[str, Any]) -> None:
        assert doc["frame"] == "map" and doc["scale_source"] in ("priors", "depth")
        assert an.backend.infers[-1] == ("frame.jpg", False)  # depth, no gravity: it has a pose
        got = {o["label"]: o for o in doc["objects"]}
        assert {k: o["id"] for k, o in got.items()} == {k: o["id"] for k, o in known.items()}
        assert got["box"]["centre_m"] == pytest.approx([0.25, 0.225, TOP + 0.06], abs=0.03)
        assert np.abs(decode(doc["depth"]) * 0.001 - z).max() < 0.02

    # against the map's cloud as the pose sees it
    check(an.analyse(req, {"i": img}))
    # a pose that sees too little of the cloud: its height above the map's floor
    monkeypatch.setattr(analysis, "MAP_ALIGN_MIN_POINTS", 10 ** 9)
    check(an.analyse(req, {"i": img}))
    # neither the cloud nor a floor plane (a flat depth): the model's metres as they are
    again = png(np.asarray(Image.open(io.BytesIO(img))))  # other bytes: the fake's flat 2 m
    doc = an.analyse({**req, "outputs": ["depth", "range_scan"]}, {"i": again})
    assert doc["frame"] == "map" and "objects" not in doc
    assert (decode(doc["depth"]) == 2000).all()
    assert {b["kind"] for b in doc["range_scan"]} <= {"obstacle", "floor_end", "unseen"}


def test_the_map_gives_the_camera_and_its_mask_without_a_pose(
        an: An, shot: tuple[bytes, bytes, np.ndarray, list[Detection]]) -> None:
    img, dep, T, dets = shot
    an.backend.other_detections = dets  # the masked frame is encoded again
    an.backend.up = T[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    doc = an.analyse({"image": "i", "depth": "d", "camera_id": "wrist", "map": "t",
                      "outputs": ["objects", "masks", "depth"]}, {"i": img, "d": dep})
    # no pose: the gravity-aligned camera frame, at the depth image's scale, local ids
    assert (doc["frame"], doc["scale_source"], doc["scale_sigma_rel"]) == ("gravity_camera",
                                                                           "depth", 0.01)
    assert an.backend.infers[-1] == ("frame.jpg", True)  # the gravity the model sees
    assert an.backend.prompted[-1] == ["Red Mug"]  # the map's vocabulary
    assert [o["id"] for o in doc["objects"]] == list(range(1_000_000, 1_000_000 + len(doc["objects"])))
    assert {"box", "basket"} <= {o["label"] for o in doc["objects"]}
    # the map's mask of the camera: nothing from the ignored rows
    out = decode(doc["depth"])
    assert not out[BOTTOM:].any() and np.array_equal(out[:BOTTOM],
                                                     np.asarray(Image.open(io.BytesIO(dep)))[:BOTTOM])
    for o in doc["objects"]:
        assert not rle.decode(o["mask"])[BOTTOM:].any() and o["bbox_px"][3] <= BOTTOM


# -- frames without a map ------------------------------------------------------------------------


def test_a_pose_without_a_map_gives_the_client_s_frame(
        an: An, shot: tuple[bytes, bytes, np.ndarray, list[Detection]]) -> None:
    img, dep, T, _ = shot
    a = math.radians(30.0)
    S = np.array([[math.cos(a), -math.sin(a), 0.0, 10.0], [math.sin(a), math.cos(a), 0.0, -4.0],
                  [0.0, 0.0, 1.0, 5.0], [0.0, 0.0, 0.0, 1.0]])  # the client's frame of the scene
    P = S @ T
    req = {"image": "i", "depth": "d", "camera": CAM, "T_map_cam": P.tolist(),
           "outputs": ["points", "range_scan", "depth"], "voxel_m": 0, "bearing_step_deg": 2.5,
           "z_band_m": [0.05, 1.0]}
    # points, range scan and depth of a depth image with a pose need no model
    an.backend.ready = False
    infers, detected = len(an.backend.infers), len(an.backend.detected)
    doc = an.analyse(req, {"i": img, "d": dep})
    assert (len(an.backend.infers), len(an.backend.detected)) == (infers, detected)
    assert (doc["frame"], doc["scale_source"], doc["scale_sigma_rel"]) == ("client", "depth", 0.01)
    assert "objects" not in doc and "matches" not in doc
    xyz, ids = decode(doc["points"]["xyz"]), decode(doc["points"]["object_id"])
    assert len(xyz) > 250_000 and not ids.any()  # every pixel that is a point, no object
    world = (xyz - S[:3, 3]) @ S[:3, :3]  # back in the scene's frame
    assert world[:, 2].min() == pytest.approx(0.0, abs=0.01)
    assert (np.abs(world[:, 0]) < 3.02).all() and (np.abs(world[:, 1]) < 2.52).all()
    # the scan: its own floor plane (z = 5 there), bearings in the client's frame
    scan = doc["range_scan"]
    assert [b["bearing_deg"] for b in scan] == [round(2.5 * k, 6) for k in range(-71, 73)]
    hits = (beam_points(doc, P[:3, 3], "obstacle") - S[:2, 3]) @ S[:2, :2]
    assert len(hits) > 15 and all(on_a_surface(p, 0.12) for p in hits)  # beams 2.5 deg wide
    # objects need the detector
    with_objects = {**req, "outputs": ["objects"]}
    err = an.analyse(with_objects, {"i": img, "d": dep}, status=503)["error"]
    assert err["code"] == "inference_unavailable" and err["retryable"]
    an.backend.ready = True
    doc = an.analyse(with_objects, {"i": img, "d": dep})
    assert an.backend.prompted[-1] == []  # the default vocabulary alone
    got = {o["label"]: o for o in doc["objects"]}
    assert [o["id"] for o in doc["objects"]] == [1_000_000, 1_000_001, 1_000_002, 1_000_003]
    assert [o["color"] for o in doc["objects"]] == [list(color_for_id(k)) for k in (1, 2, 3, 4)]
    assert "mask" not in got["box"]
    centre = (np.array(got["box"]["centre_m"]) - S[:3, 3]) @ S[:3, :3]
    assert centre == pytest.approx([0.25, 0.225, TOP + 0.06], abs=0.03)
    assert got["box"]["centre_cam_m"] == pytest.approx(
        (np.array([0.25, 0.225, TOP + 0.06]) - T[:3, 3]) @ T[:3, :3], abs=0.03)
    # the table stands on the frame's own floor
    table = got["table"]
    assert table["centre_m"][2] - table["size_m"][2] / 2 == pytest.approx(5.0, abs=0.02)
    assert table["size_m"][2] == pytest.approx(TOP, abs=0.02)


def test_the_gravity_camera_frame_and_the_camera_height_scale(
        an: An, shot: tuple[bytes, bytes, np.ndarray, list[Detection]]) -> None:
    img, dep, T, _ = shot
    z, eye = metres(dep), T[:3, 3]
    an.backend.mono_depth[hashlib.sha256(img).hexdigest()] = z * 0.8
    an.backend.up = T[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    yaw = math.degrees(math.atan2(T[1, 2], T[0, 2]))
    a = math.radians(yaw)
    Rz = np.array([[math.cos(a), math.sin(a), 0.0], [-math.sin(a), math.cos(a), 0.0],
                   [0.0, 0.0, 1.0]])  # the scene's frame turned to the viewing direction
    truth = Rz @ (np.array([0.25, 0.225, TOP + 0.06]) - eye)
    req = {"image": "i", "outputs": ["objects", "range_scan", "depth"],
           "camera": {**CAM, "height_above_floor_m": float(eye[2])}}
    doc = an.analyse(req, {"i": img})
    assert an.backend.infers[-1] == ("frame.jpg", True)
    assert (doc["frame"], doc["scale_source"]) == ("gravity_camera", "camera_height")
    assert doc["scale_sigma_rel"] == pytest.approx(0.02 / eye[2], abs=1e-5)
    box = {o["label"]: o for o in doc["objects"]}["box"]
    assert box["id"] >= 1_000_000
    assert box["centre_m"] == pytest.approx(truth, abs=0.03)  # the camera at the origin, z up
    assert box["centre_cam_m"] == pytest.approx(
        (np.array([0.25, 0.225, TOP + 0.06]) - eye) @ T[:3, :3], abs=0.03)
    assert np.abs(decode(doc["depth"]) * 0.001 - z).max() < 0.03  # at the height's scale
    # bearings counter-clockwise from the viewing direction, ranges from the camera
    hits = beam_points(doc, eye, "obstacle", yaw)
    assert len(hits) > 40 and all(on_a_surface(p) for p in hits)
    assert all(abs(b["bearing_deg"]) < 60 for b in doc["range_scan"] if b["kind"] != "unseen")
    # the camera's own height sigma
    doc = an.analyse({**req, "camera": {**req["camera"], "height_sigma_m": 0.05}}, {"i": img})
    assert doc["scale_sigma_rel"] == pytest.approx(0.05 / eye[2], abs=1e-5)
    # without a height (one below the floor is none): the model's metres, the floor its own plane
    doc = an.analyse({**req, "camera": {**CAM, "height_above_floor_m": -1.0}}, {"i": img})
    assert doc["scale_source"] == "monocular"
    doc = an.analyse({**req, "camera": CAM}, {"i": img})
    assert (doc["scale_source"], doc["scale_sigma_rel"]) == ("monocular", 0.1)
    box = {o["label"]: o for o in doc["objects"]}["box"]
    assert box["centre_m"] == pytest.approx(0.8 * truth, abs=0.03)
    table = {o["label"]: o for o in doc["objects"]}["table"]
    assert table["centre_m"][2] - table["size_m"][2] / 2 == pytest.approx(-0.8 * eye[2], abs=0.02)
    assert np.abs(decode(doc["depth"]) * 0.001 - 0.8 * z).max() < 0.02


def test_a_depth_image_without_a_value_and_a_frame_that_sees_no_floor(an: An) -> None:
    cam = {"fx": 300.0, "fy": 300.0, "cx": 400.0, "cy": 300.0, "width": 800, "height": 600}
    img = png(np.full((600, 800, 3), 90, np.uint8))
    empty = png(np.zeros((600, 800), np.uint16))
    an.backend.other_depth = np.full((480, 640), 2.0, np.float32)  # a grid smaller than the image
    patch = np.zeros((480, 640), bool)
    patch[200:280, 280:360] = True
    an.backend.other_detections = [Detection("book", 0.9, "test", patch, (280.0, 200.0, 360.0, 280.0))]
    doc = an.analyse({"image": "i", "depth": "d", "camera": {**cam, "height_above_floor_m": 0.9},
                      "outputs": ALL, "voxel_m": 0.05}, {"i": img, "d": empty})
    # the depth image holds nothing: the monocular depth; no floor plane in it: no height scale
    assert an.backend.infers[-1] == ("frame.jpg", True)
    assert (doc["frame"], doc["scale_source"]) == ("gravity_camera", "monocular")
    out = decode(doc["depth"])
    assert out.shape == (600, 800) and (out == 2000).all()
    # a wall 2 m ahead: the camera's height puts the floor under it, so the wall is an obstacle
    kinds = {b["bearing_deg"]: b for b in doc["range_scan"]}
    assert kinds[0.0]["kind"] == "obstacle" and kinds[0.0]["range_m"] == pytest.approx(2.0, abs=0.01)
    assert kinds[180.0]["kind"] == "unseen"
    (book,) = doc["objects"]
    assert book["label"] == "book" and book["centre_cam_m"] == pytest.approx([0.0, 0.0, 2.0], abs=0.02)
    assert book["bbox_px"] == [350.0, 250.0, 450.0, 350.0]  # in the image's pixels
    assert rle.decode(book["mask"]).shape == (600, 800)
    ids = decode(doc["points"]["object_id"])
    assert set(ids.tolist()) == {0, 1_000_000}
    # without a height the scan takes the floor 1 m below the camera
    doc = an.analyse({"image": "i", "camera": cam, "outputs": ["range_scan"]}, {"i": img})
    kinds = {b["bearing_deg"]: b for b in doc["range_scan"]}
    assert kinds[0.0]["kind"] == "obstacle" and kinds[0.0]["range_m"] == pytest.approx(2.0, abs=0.01)


# -- labels, matches, masks ----------------------------------------------------------------------


def test_labels_prompt_the_detector_and_name_the_best_match(
        an: An, shot: tuple[bytes, bytes, np.ndarray, list[Detection]]) -> None:
    img, dep, T, dets = shot
    by = {d.label: d for d in dets}

    def named(label: str, to: str, score: float) -> Detection:
        d = by[label]
        return Detection(to, score, d.source, d.mask, d.box)

    sha = hashlib.sha256(img).hexdigest()
    an.backend.detections[sha] = [named("table", "table", 0.6), named("box", "cup", 0.8),
                                  named("basket", "cup", 0.8), named("apple", "computer mouse", 0.4)]
    try:
        req = {"image": "i", "depth": "d", "camera": CAM, "T_map_cam": T.tolist(),
               "outputs": ["points"], "labels": ["Cup", "mouse", "Red_Mug", "TABLE"]}
        doc = an.analyse(req, {"i": img, "d": dep})
        # labels imply objects and matches; the defaults are prompted with them
        assert an.backend.prompted[-1] == ["Cup", "mouse", "Red_Mug", "TABLE"]
        assert an.backend.asked[-1] == (0.5, True)
        got = [(o["id"], o["label"], o["score"]) for o in doc["objects"]]
        assert got == [(1_000_000, "Cup", 0.8), (1_000_001, "Cup", 0.8), (1_000_002, "TABLE", 0.6)]
        assert doc["matches"] == [  # two cups of one score: the lower id
            {"label": "Cup", "object_id": 1_000_000, "score": 0.8},
            {"label": "mouse", "object_id": None, "score": None},
            {"label": "Red_Mug", "object_id": None, "score": None},
            {"label": "TABLE", "object_id": 1_000_002, "score": 0.6}]
        assert set(decode(doc["points"]["object_id"]).tolist()) == {0, 1_000_000, 1_000_001,
                                                                    1_000_002}
        # a lower threshold: everything the detector scores from its floor, and the object
        # detected as the prompt "mouse" takes the client's wording
        doc = an.analyse({**req, "min_score": 0.3, "labels_only": True}, {"i": img, "d": dep})
        assert an.backend.asked[-1] == (0.05, False)  # the labels alone
        assert [(o["label"], o["score"]) for o in doc["objects"]] == [
            ("Cup", 0.8), ("Cup", 0.8), ("TABLE", 0.6), ("mouse", 0.4)]
        assert doc["matches"][1] == {"label": "mouse", "object_id": doc["objects"][3]["id"],
                                     "score": 0.4}
        # the threshold drops what scores below it, whatever the detector's floor returned
        doc = an.analyse({**req, "min_score": 0.45, "outputs": ["objects"]}, {"i": img, "d": dep})
        assert an.backend.asked[-1] == (0.05, True)
        assert [o["label"] for o in doc["objects"]] == ["Cup", "Cup", "TABLE"]
        # a higher one: the best cup is the best that is returned
        doc = an.analyse({**req, "min_score": 0.7, "labels": ["table"]}, {"i": img, "d": dep})
        assert [o["label"] for o in doc["objects"]] == ["cup", "cup"]
        assert doc["matches"] == [{"label": "table", "object_id": None, "score": None}]
        # the map's vocabulary joins the prompts, but not with labels_only
        both = {**req, "camera_id": "cam", "map": "t"}
        del both["camera"]
        an.analyse(both, {"i": img, "d": dep})
        assert an.backend.prompted[-1] == ["Red Mug", "Cup", "mouse", "Red_Mug", "TABLE"]
        an.analyse({**both, "labels_only": True}, {"i": img, "d": dep})
        assert an.backend.prompted[-1] == ["Cup", "mouse", "Red_Mug", "TABLE"]
        assert an.backend.asked[-1] == (0.5, False)
        # no labels: no matches
        doc = an.analyse({**req, "labels": [], "outputs": ["depth"]}, {"i": img, "d": dep})
        assert doc["matches"] == [] and len(doc["objects"]) == 3
    finally:
        an.backend.detections[sha] = dets


def test_client_form_and_matches() -> None:
    assert analyse.client_form("red mug", [" x ", "Red_Mug", "RED MUG"]) == "Red_Mug"
    assert analyse.client_form("computer mouse", ["Mouse"]) == "Mouse"  # its prompt's wording
    assert analyse.client_form("computer mouse", ["Computer Mouse", "mouse"]) == "Computer Mouse"
    assert analyse.client_form("chair", ["table"]) == "chair"
    objs = [{"id": 7, "label": "Cup", "score": 0.6}, {"id": 3, "label": "cup", "score": 0.9},
            {"id": 2, "label": "CUP", "score": 0.9}, {"id": 1, "label": "bowl", "score": 1.0}]
    assert analyse.matches(["cup", "Bowl", "plate"], objs) == [
        {"label": "cup", "object_id": 2, "score": 0.9},
        {"label": "Bowl", "object_id": 1, "score": 1.0},
        {"label": "plate", "object_id": None, "score": None}]


# -- cameras and ignored pixels ------------------------------------------------------------------


def test_ignored_pixels_give_nothing(an: An, shot: tuple[bytes, bytes, np.ndarray, list[Detection]]
                                     ) -> None:
    img, dep, T, dets = shot
    an.backend.other_detections = dets
    left = np.zeros((480, 640), np.uint8)
    left[:, :240] = 255  # most of the box (its columns 197-259)
    low = np.zeros((480, 640), np.uint8)
    low[BOTTOM:] = 255
    doc = an.analyse({"image": "i", "depth": "d", "T_map_cam": T.tolist(),
                      "camera": {**CAM, "ignore_mask": "cm"}, "ignore_mask": "fm",
                      "outputs": ["masks", "depth", "points"], "voxel_m": 0},
                     {"i": img, "d": dep, "cm": png(left), "fm": png(low)})
    ignored = (left > 0) | (low > 0)
    labels = {o["label"] for o in doc["objects"]}  # masks imply objects
    assert "box" not in labels and {"table", "basket"} <= labels  # mostly ignored: no detection
    for o in doc["objects"]:
        assert not (rle.decode(o["mask"]) & ignored).any()
    out = decode(doc["depth"])
    sent = np.asarray(Image.open(io.BytesIO(dep)))
    assert not out[ignored].any() and np.array_equal(out[~ignored], sent[~ignored])
    assert len(decode(doc["points"]["xyz"])) <= int((~ignored).sum())
    # the camera's mask alone
    doc = an.analyse({"image": "i", "depth": "d", "T_map_cam": T.tolist(),
                      "camera": {**CAM, "ignore_mask": "cm"}, "outputs": ["depth"]},
                     {"i": img, "d": dep, "cm": png(left)})
    out = decode(doc["depth"])
    assert not out[:, :240].any() and np.array_equal(out[:, 240:], sent[:, 240:])
    # a monocular depth has none there either
    an.backend.other_depth = metres(dep)
    doc = an.analyse({"image": "i", "T_map_cam": T.tolist(), "outputs": ["depth"],
                      "camera": {**CAM, "ignore_mask": "cm"}}, {"i": img, "cm": png(left)})
    out = decode(doc["depth"])
    assert doc["scale_source"] == "monocular" and not out[:, :240].any() and out[:, 240:].all()
    # masks that ignore nothing leave the frame as sent
    none = png(np.zeros((480, 640), np.uint8))
    doc = an.analyse({"image": "i", "depth": "d", "T_map_cam": T.tolist(), "ignore_mask": "fm",
                      "camera": {**CAM, "ignore_mask": "cm"}, "outputs": ["depth", "objects"]},
                     {"i": img, "d": dep, "cm": none, "fm": none})
    assert np.array_equal(decode(doc["depth"]), sent) and len(doc["objects"]) == 4


def test_a_camera_that_is_not_its_canonical_pinhole(
        an: An, shot: tuple[bytes, bytes, np.ndarray, list[Detection]],
        monkeypatch: pytest.MonkeyPatch) -> None:
    img, dep, T, _ = shot
    cam = {**CAM, "cx": 300.0}  # the canonical pinhole is the image shifted 20 px to the right
    patch = np.zeros((480, 640), bool)
    patch[260:330, 300:400] = True  # on the table top
    edge = np.zeros((480, 640), bool)
    edge[100:200, 0:18] = True  # where the image does not reach
    an.backend.other_detections = [Detection("book", 0.9, "test", patch, (300.0, 260.0, 400.0, 330.0)),
                                   Detection("vase", 0.8, "test", edge, (0.0, 100.0, 18.0, 200.0))]
    req = {"image": "i", "depth": "d", "camera": cam, "T_map_cam": T.tolist(),
           "outputs": ["masks", "depth", "points"], "voxel_m": 0}
    doc = an.analyse(req, {"i": img, "d": dep})
    (book,) = doc["objects"]  # the vase lies beyond the image: no detection
    m = rle.decode(book["mask"])
    want = np.zeros((480, 640), bool)
    want[260:330, 280:380] = True  # the patch in the image's own pixels
    assert (m & want).sum() / (m | want).sum() > 0.9 and not (m & ~want).any()
    assert book["bbox_px"][0] >= 280 and book["bbox_px"][2] <= 380
    # the image's last 20 columns lie beyond the canonical pinhole: nothing from them
    out, sent = decode(doc["depth"]), np.asarray(Image.open(io.BytesIO(dep)))
    assert np.array_equal(out[:, :620], sent[:, :620]) and not out[:, 620:].any()
    assert len(decode(doc["points"]["xyz"])) > 250_000
    # an object none of whose pixels are in the image as sent is not returned
    monkeypatch.setattr(Camera, "unrectify", lambda self, a: np.zeros_like(a))
    doc = an.analyse({**req, "outputs": ["objects", "points"], "labels": ["book"]},
                     {"i": img, "d": dep})
    assert doc["objects"] == [] and doc["matches"][0]["object_id"] is None
    assert not decode(doc["points"]["object_id"]).any()  # nor do the points name it


def test_unrectify_inverts_rectify() -> None:
    cam = Camera(300.0, 320.0, 170.0, 110.0, 320, 240, (-0.12, 0.03, 0.001, -0.002))
    assert not cam.is_canonical
    v, u = np.mgrid[0:240, 0:320]
    img = ((u // 16 + v // 16) % 7 * 30).astype(np.uint8)
    back = cam.unrectify(cam.rectify(img, nearest=True))
    inner = np.zeros(img.shape, bool)
    inner[20:-20, 20:-20] = True
    assert (back[inner] == img[inner]).mean() > 0.9  # but at the squares' edges
    beyond = cam.beyond()
    assert beyond is not None and 0 < beyond.mean() < 0.5
    plain = Camera(300.0, 300.0, 160.0, 120.0, 320, 240)
    assert plain.unrectify(img) is img and plain.beyond() is None


# -- requests ------------------------------------------------------------------------------------


def test_requests_are_validated(an: An, shot: tuple[bytes, bytes, np.ndarray, list[Detection]]
                                ) -> None:
    img, dep, T, _ = shot
    ok = {"image": "i", "camera": CAM, "outputs": ["depth"], "depth": "d",
          "T_map_cam": T.tolist()}
    files = {"i": img, "d": dep}

    def field(req: dict[str, Any], parts: dict[str, bytes] = files, status: int = 400) -> str:
        err = an.analyse(req, parts, status)["error"]
        assert err["code"] == ("map_not_found" if status == 404 else "invalid_request")
        return str(err["details"].get("field"))

    small = png(np.zeros((10, 10), np.uint8))
    assert field({**ok, "outputs": []}) == "outputs"
    assert field({k: v for k, v in ok.items() if k != "camera"}) == "body"
    assert field({**ok, "camera_id": "cam", "map": "t"}) == "body"  # a camera, or a map's
    assert field({**ok, "labels": ["a", "  "]}) == "labels"
    assert field({**ok, "labels": ["x" * 65]}) == "labels"
    assert field({**ok, "labels": [f"l{i}" for i in range(65)]}) == "labels"
    assert field({**ok, "labels": "cup"}) == "labels"
    assert field({**ok, "labels_only": True}) == "labels_only"
    assert field({**ok, "labels_only": True, "labels": []}) == "labels_only"
    assert field({**ok, "labels_only": "yes", "labels": ["cup"]}) == "labels_only"
    assert field({**ok, "image": "nothing"}) == "image"
    assert field(ok, {"i": small, "d": dep}) == "image"
    assert field(ok, {"i": b"not an image", "d": dep}) == "image"
    assert field(ok, {"i": img, "d": png(np.zeros((10, 10), np.uint16))}) == "depth"
    assert field(ok, {"i": img, "d": img}) == "depth"
    assert field({**ok, "ignore_mask": "m"}, {**files, "m": small}) == "ignore_mask"
    assert field({**ok, "ignore_mask": "m"}) == "ignore_mask"
    masked = {**ok, "camera": {**CAM, "ignore_mask": "m"}}
    assert field(masked) == "camera.ignore_mask"
    assert field(masked, {**files, "m": small}) == "camera.ignore_mask"
    bent = T.copy()
    bent[0, 0] += 0.2
    assert field({**ok, "T_map_cam": bent.tolist()}) == "T_map_cam"
    assert field({**ok, "z_band_m": [1.0, 1.0]}) == "z_band_m"
    assert field({**ok, "bearing_step_deg": 0.01}) == "bearing_step_deg"
    assert field({**ok, "bearing_step_deg": 0}) == "bearing_step_deg"
    assert field({**ok, "min_score": 1.5}) == "min_score"
    by_id = {k: v for k, v in ok.items() if k != "camera"}
    assert field({**by_id, "camera_id": "none", "map": "t"}) == "camera_id"
    assert field({**by_id, "camera_id": "cam", "map": "nowhere"}, status=404) == "None"
    assert field({**ok, "map": "no/such"}) == "name"
    # the body itself
    r = an.http.post("/api/frames/analyse", content=b"{}", headers={"content-type": "text/plain"})
    assert r.status_code == 400 and an.check(r, None)["error"]["details"]["field"] == "body"
    data, ctype = multipart({}, {})
    r = an.http.post("/api/frames/analyse", content=data.replace(b'name="request"', b'name="r"'),
                     headers={"content-type": ctype})
    assert r.status_code == 400 and an.check(r, None)["error"]["details"]["field"] == "request"
    r = an.http.post("/api/frames/analyse", content=data.replace(b"{}", b"{]"),
                     headers={"content-type": ctype})
    assert r.status_code == 400 and "not JSON" in an.check(r, None)["error"]["message"]
    r = an.http.get("/api/frames/analyse")
    assert r.status_code == 405 and an.check(r, None)["error"]["code"] == "invalid_request"


def test_a_service_without_an_analyser_refuses(an: An, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(an.service, "analyser", None)
    err = an.analyse({"image": "i", "camera": CAM, "outputs": ["depth"]}, {}, status=400)["error"]
    assert err["code"] == "invalid_request" and "not served" in err["message"]


def test_the_endpoint_is_documented(an: An) -> None:
    doc = an.get("/api/openapi.json")
    post = doc["paths"]["/api/frames/analyse"]["post"]
    body = post["requestBody"]["content"]["multipart/form-data"]["schema"]
    assert body["properties"]["request"] == {"$ref": "#/components/schemas/AnalyseRequest"}
    assert post["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/Analysis"}
    assert {"400", "404", "413", "500", "503"} <= set(post["responses"])
    assert "O7" in an.get("/api/health", "Health")["capabilities"]


# -- while an update runs (C12) ------------------------------------------------------------------


def test_an_analysis_answers_while_an_update_of_the_map_runs(
        an: An, shot: tuple[bytes, bytes, np.ndarray, list[Detection]],
        monkeypatch: pytest.MonkeyPatch) -> None:
    img, dep, T, _ = shot
    fr, files = frames(an, "full", [(1.0, -0.9, 1.5), (0.2, -0.9, 1.5)], "b")
    assert an.update("busy", fr, files, cameras={"cam": CAM})["state"] == "committed"
    entered, release = threading.Event(), threading.Event()
    real = an.backend.detect

    def detect(image: Any, grid: Any, labels: Any = (), *a: Any, **k: Any) -> list[Any]:
        if image.name != "frame.jpg":  # the job's fusion waits; the analysis does not
            entered.set()
            assert release.wait(60)
        return real(image, grid, labels, *a, **k)

    monkeypatch.setattr(an.backend, "detect", detect)
    fr, files = frames(an, "full", [(1.9, 0.0, 1.5)], "c")
    job = an.post_update("busy", {"frames": fr}, files)
    try:
        assert entered.wait(60)
        doc = an.analyse({"image": "i", "depth": "d", "camera_id": "cam", "map": "busy",
                          "T_map_cam": T.tolist(), "outputs": ["objects", "range_scan"]},
                         {"i": img, "d": dep})
        assert an.get(f"/api/jobs/{job['job_id']}", "Job")["state"] in ("registering", "fusing")
        assert doc["frame"] == "map" and len(doc["objects"]) == 4
        assert an.get("/api/maps/busy", "MapSummary")["latest_version"] == 1  # as it stood
    finally:
        release.set()
    assert an.wait(job["job_id"])["state"] == "committed"
    assert an.get("/api/maps/busy", "MapSummary")["latest_version"] == 2


# -- mapping.analysis ----------------------------------------------------------------------------


def test_range_scan_beams() -> None:
    def ring(bearing: float, r: float, z: float, n: int = 5) -> np.ndarray:
        b = math.radians(bearing)
        return np.tile([1.0 + r * math.cos(b), 2.0 + r * math.sin(b), z], (n, 1))

    pts = np.vstack([
        ring(0.0, 3.0, 0.5), ring(0.0, 2.0, 0.5, 2), ring(0.0, 4.0, 0.5),  # two flying pixels
        ring(90.0, 1.0, 0.0), ring(90.0, 2.5, 0.01), ring(90.0, 3.0, 0.0, 2),  # floor alone
        ring(-90.0, 1.0, 2.5), ring(-90.0, 2.0, 0.055),  # above the band; within the floor band
        ring(45.0, 2.0, 0.0), ring(45.0, 1.5, 1.8),  # the floor, then an obstacle nearer
        ring(179.9, 2.0, 0.3), ring(-179.9, 1.0, 0.3),  # either side of ±180
        ring(135.0, 1.0, 0.5, 2)])  # too few points
    scan = analysis.range_scan(pts, np.array([1.0, 2.0, 1.3]), 0.0, 0.06, (0.05, 1.8), 45.0)
    got = {b["bearing_deg"]: (b["kind"], b["range_m"]) for b in scan}
    assert list(got) == [-135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 180.0]
    assert got[0.0] == ("obstacle", 3.0) and got[90.0] == ("floor_end", 2.5)
    assert got[-90.0] == ("floor_end", 2.0) and got[45.0] == ("obstacle", 1.5)
    assert got[180.0] == ("obstacle", 1.0) and got[135.0] == ("unseen", None)
    assert got[-45.0] == ("unseen", None) and got[-135.0] == ("unseen", None)
    # a step that does not divide 360: the end beams share what is left at ±180
    scan = analysis.range_scan(pts, np.array([1.0, 2.0, 1.3]), 0.0, 0.06, (0.05, 1.8), 70.0)
    got = {b["bearing_deg"]: (b["kind"], b["range_m"]) for b in scan}
    assert list(got) == [-140.0, -70.0, 0.0, 70.0, 140.0]
    assert got[140.0] == ("obstacle", 2.0) and got[-140.0] == ("obstacle", 1.0)
    # a floor elsewhere moves the band; no point at all: every beam unseen
    scan = analysis.range_scan(pts, np.array([1.0, 2.0, 1.3]), 2.4, 0.06, (0.05, 1.8), 90.0)
    assert {b["bearing_deg"]: b["kind"] for b in scan} == {
        -90.0: "obstacle", 0.0: "unseen", 90.0: "unseen", 180.0: "unseen"}
    scan = analysis.range_scan(np.zeros((0, 3)), np.zeros(3), 0.0, 0.06, (0.05, 1.8), 360.0)
    assert scan == [{"bearing_deg": 0.0, "range_m": None, "kind": "unseen"}]


def _floor_and_wall() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The depth of a camera 1 m above a floor, pitched 30 degrees down, with a wall 3 m ahead,
    its pose, and the points it sees."""
    from tests.synth.rooms import Scene, camera

    scene = Scene().add((-5.0, -5.0, -0.1), (5.0, 5.0, 0.0)).add((3.0, -5.0, 0.0), (3.1, 5.0, 3.0))
    T = camera(0.0, 0.0, 1.0, 0.0, -30.0)
    K = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]])
    z = scene.depth(T, K, 640, 480)
    v, u = np.nonzero(z > 0)
    pts = np.stack([(u - 320.0) / 400.0 * z[v, u], (v - 240.0) / 400.0 * z[v, u], z[v, u]], 1)
    return z, T, pts @ T[:3, :3].T + T[:3, 3]


def test_a_monocular_depth_s_scale() -> None:
    z, T, cloud = _floor_and_wall()
    R = T[:3, :3]
    # onto a camera height: the height over the camera's height above its own floor
    assert analysis.height_factor(z * 0.8, K_TABLE, R, 1.0) == pytest.approx(1.25, abs=0.01)
    assert analysis.height_factor(z, K_TABLE, R, None) is None
    assert analysis.height_factor(z, K_TABLE, R, 0.04) is None
    assert analysis.height_factor(z, K_TABLE, R, 0.1) is None  # a tenth of its floor's: no scale
    assert analysis.height_factor(np.full_like(z, 2.0), K_TABLE, np.eye(3), 1.0) is None
    # the floor of the frame: the plane nearest the height, the lowest without one
    assert analysis.own_floor_z(z, K_TABLE, T, 1.0) == pytest.approx(0.0, abs=0.01)
    assert analysis.own_floor_z(z, K_TABLE, T, None) == pytest.approx(0.0, abs=0.01)
    assert analysis.own_floor_z(np.full_like(z, 2.0), K_TABLE, np.eye(4), None) is None
    # onto a map: the cloud as the pose sees it, what lies behind its nearest points hidden
    hidden = cloud + np.array([0.0, 0.0, -1.5])  # another floor below: seen through nothing
    both = np.vstack([cloud[::3], hidden[::3]])
    assert analysis.map_factor(z * 0.8, K_TABLE, T, both) == pytest.approx(1.25, abs=0.01)
    assert analysis.map_factor(z * 0.8, K_TABLE, T, cloud[:150]) is None  # too little in view
    assert analysis.map_factor(z * 0.3, K_TABLE, T, cloud[::3]) is None  # no such scale
    behind = cloud @ np.diag([-1.0, 1.0, 1.0])
    assert analysis.map_factor(z, K_TABLE, T, behind) is None
    assert np.allclose(analysis.gravity_pose(None)[:3, :3] @ [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])


def test_known_objects_and_local_colours() -> None:
    rec = {"id": 4, "label": "cup", "detected_as": ["cup", "mug"], "centre_m": [1.0, 2.0, 0.5],
           "size_m": [0.2, 0.1, 0.1], "yaw_deg": 30.0, "state": "confirmed"}
    ids = np.array([0, 4, 9, 4, 4, 0], np.uint32)
    xyz = np.arange(18, dtype=np.float32).reshape(6, 3)
    (o,) = analysis.known_objects([rec, {**rec, "id": 5, "state": "struck"}],
                                  {"xyz": xyz, "object_id": ids})
    assert o.id == 4 and set(o.votes) == {"cup", "mug"} and o.box is not None
    assert np.array_equal(o.points, xyz[[1, 3, 4]]) and o.box.yaw_deg == 30.0
    many = {"xyz": np.zeros((5000, 3), np.float32), "object_id": np.full(5000, 4, np.uint32)}
    assert len(analysis.known_objects([rec], many)[0].points) == analysis.KNOWN_POINTS
    assert analyse.color_of(4) == color_for_id(4)
    assert analyse.color_of(1_000_002) == color_for_id(3)


def test_objects_of_a_view_are_boxed_filtered_and_matched() -> None:
    z, T, _ = _floor_and_wall()
    rgb = np.full((480, 640, 3), 120, np.uint8)

    def det(label: str, score: float, r0: int, r1: int, c0: int, c1: int) -> Detection:
        m = np.zeros((480, 640), bool)
        m[r0:r1, c0:c1] = True
        return Detection(label, score, "test", m, (float(c0), float(r0), float(c1), float(r1)))

    dets = [det("poster", 0.9, 30, 110, 100, 220), det("door", 0.8, 30, 80, 300, 420),
            det("sticker", 0.7, 468, 476, 316, 324), det("clock", 0.3, 30, 110, 480, 600)]
    ignore = np.zeros((480, 640), bool)
    ignore[30:60] = True
    view = View("frame", T, 0.0, K_TABLE, z, rgb, dets, ignore, 0.01)
    seen = analysis.objects(view, None, 0.02, 0.5, [])
    # the door (0.4 m of a wall) is no leaf, the sticker (2 cm) a fragment, the clock below the
    # threshold
    assert [(s.id, s.label) for s in seen] == [(1_000_000, "poster")]
    poster = seen[0]
    assert poster.box.centre[0] == pytest.approx(3.0, abs=0.02) and poster.sigma_centre_m > 0
    assert not (poster.mask & ignore).any() and poster.mask[60:110, 100:220].all()
    assert len(poster.pixels) > 100 and poster.mean_rgb == [120, 120, 120]
    # a map object where the poster is gives it its id; the clock, kept from 0.3, stays local
    from oh_my_slam.mapping.anchored_objects import MapObject

    there = MapObject(41, {"poster": 1.0}, box=Box(poster.box.centre, poster.box.size,
                                                   poster.box.yaw_deg))
    seen = analysis.objects(View("frame", T, 0.0, K_TABLE, z, rgb, dets), 0.0, 0.02, 0.3, [there])
    assert [(s.id, s.label) for s in seen] == [(41, "poster"), (1_000_000, "door"),
                                               (1_000_001, "clock")]
    door = seen[1].box  # with a floor, the door stands on it: a leaf
    assert door.bottom == pytest.approx(0.0, abs=0.01) and door.size[2] > 0.6
    # its points carry the ids; thinning keeps one per voxel
    pts = analysis.frame_points(View("frame", T, 0.0, K_TABLE, z, rgb, dets), seen)
    assert set(pts.object_id.tolist()) == {0, 41, 1_000_000, 1_000_001}
    assert (pts.xyz[pts.object_id == 41][:, 0] > 2.9).all()
    masked = analysis.frame_points(view, [])
    assert len(masked.xyz) < len(pts.xyz) and not masked.object_id.any()
    thin = analysis.thinned(pts, 0.1)
    assert len(thin.xyz) < len(pts.xyz) / 10 and len(thin.rgb) == len(thin.object_id) == len(thin.xyz)
    assert analysis.thinned(pts, 0.0) is pts
    empty = analysis.FramePoints(np.zeros((0, 3)), np.zeros(0, np.uint32), np.zeros((0, 3), np.uint8))
    assert analysis.thinned(empty, 0.1) is empty
    assert analysis.objects(View("frame", T, 0.0, K_TABLE, z, rgb, []), None, 0.02, 0.5, []) == []
