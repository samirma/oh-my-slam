"""``GET /api/maps/{name}/objects`` (O8) through the Starlette app on the prior-first path: a
ray-cast table scene (``tests.synth.table``) with depth images, metric priors and exact
detections; the records validate against ``Objects``, ids persist, the item taken away is
struck and gone the version after, the answer is kept per version, regions and doors come from
the grid; a map of the older mapper lists its objects too."""

from __future__ import annotations

import functools
import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from starlette.testclient import TestClient

from oh_my_slam.service.app import Service, create_app
from oh_my_slam.service.derived import Derived
from oh_my_slam.service.jobs import Jobs
from oh_my_slam.service.store import Store
from oh_my_slam.service.timings import Timings
from oh_my_slam.service.update import Updater
from tests.fakes.service import CAMERA, FakeBackend, FakeInference, pose
from tests.synth.room import jpeg, png_depth
from tests.synth.table import K_TABLE, TOP, TableScene, default_items, look_at
from tests.unit.test_anchored_objects import EYES, TARGET, detections
from tests.unit.test_service_api import Client

CAM = {"fx": K_TABLE.fx, "fy": K_TABLE.fy, "cx": K_TABLE.cx, "cy": K_TABLE.cy,
       "width": K_TABLE.width, "height": K_TABLE.height}


@pytest.fixture
def client(tmp_path: Path) -> Iterator[Client]:
    store = Store(tmp_path / "data")
    store.create()
    backend = FakeBackend()
    timings = Timings()
    updater = Updater(store, backend, timings)
    jobs = Jobs(store.jobs, updater, timings.predict_update)
    jobs.start()
    derived = Derived(store, backend.read)
    derived.start()
    service = Service(store, jobs, FakeInference(), backend.require_inference, timings=timings,
                      version="test", dry_run=updater.dry_run, derived=derived)
    with TestClient(create_app(service)) as http:
        yield Client(http, backend, service)
    jobs.stop(timeout=10)
    derived.stop()


@functools.lru_cache(maxsize=32)
def _shot(scene_key: str, eye: tuple[float, float, float]) -> tuple[bytes, bytes, Any, Any]:
    scene = SCENES[scene_key]()
    T = look_at(eye, TARGET)
    rgb, z, idx = scene.render(T)
    return jpeg(rgb, 95), png_depth(z), T, detections(scene, idx)


def _without_apple() -> TableScene:
    items = default_items()
    items[3].boxes = []
    return TableScene(items=items)


SCENES = {"full": TableScene, "taken": _without_apple}


def frames(c: Client, scene: str, eyes: list[tuple[float, float, float]], tag: str
           ) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    out, files = [], {}
    for i, eye in enumerate(eyes):
        img, dep, T, dets = _shot(scene, eye)
        c.backend.detections[hashlib.sha256(img).hexdigest()] = dets
        files[f"{tag}{i}"], files[f"{tag}{i}d"] = img, dep
        out.append({"name": f"{tag}{i}", "image": f"{tag}{i}", "depth": f"{tag}{i}d",
                    "camera_id": "cam", "prior": {"trust": "metric", "T_map_cam": T.tolist(),
                                                  "sigma_pos_m": 0.01, "sigma_rot_deg": 0.5}})
    return out, files


def moved(eyes: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """The eyes 6 cm aside: the same view again is a pose duplicate (api.md §4)."""
    return [(x + 0.06, y, z) for x, y, z in eyes]


def labels(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {o["label"]: o for o in doc["objects"]}


def test_objects_of_an_anchored_map(client: Client) -> None:
    fr, files = frames(client, "full", EYES[:4], "a")
    job = client.update("t", fr, files, cameras={"cam": CAM})
    assert job["state"] == "committed", job
    assert "objects" in job["timings_s"]
    assert client.get("/api/maps/t", "MapSummary")["objects"] == 4
    doc = client.get("/api/maps/t/objects", "Objects")
    assert doc["map_version"] == 1 and doc["scale_source"] in ("depth", "priors")
    o = labels(doc)
    assert set(o) == {"table", "box", "basket", "apple"}
    hs = sorted(s["height_m"] for s in doc["surfaces"])
    assert hs[0] == pytest.approx(0.0, abs=0.01) and hs[1] == pytest.approx(TOP, abs=0.01)
    assert o["apple"]["inside_of"]["id"] == o["basket"]["id"]
    assert o["box"]["supported_by"]["kind"] == "surface"
    assert o["box"]["mean_rgb"] == pytest.approx([200, 40, 40], abs=6)
    assert all(x["region"] is not None for x in doc["objects"])
    assert "seen_from" not in doc["objects"][0]
    # one floor estimate per version (map.md §5): the record's, which the grid answers as is
    rec = client.service.store.version("t", 1)
    grid = client.get("/api/maps/t/grid", "Grid")
    assert rec["floor_z_m"] == pytest.approx(0.0, abs=0.01) and rec["floor_sigma_m"] > 0
    assert (grid["floor_z_m"], grid["floor_sigma_m"]) == (rec["floor_z_m"], rec["floor_sigma_m"])
    # kept per version: the derivation worker computed it, the read loads it
    cache = client.service.store.dir("t") / "derived" / "1" / "objects.json"
    assert cache.exists()
    assert client.get("/api/maps/t/objects?version=1", "Objects") == doc
    assert client.get("/api/health", "Health")["timings"]["objects_read"]["count"] >= 1
    # the cloud's points carry the objects' ids
    pts = client.get("/api/maps/t/points?min=0.6,0.25,0.7&max=1.0,0.65,1.0", "Points")
    import base64

    ids = np.frombuffer(base64.b64decode(pts["object_id"]["data"]), np.uint32)
    assert {o["basket"]["id"], o["apple"]["id"]} <= set(ids.tolist())

    # the apple taken away: struck in version 2, the others keep their ids
    fr, files = frames(client, "taken", moved(EYES[:3]), "b")
    assert client.update("t", fr, files)["state"] == "committed"
    o2 = labels(client.get("/api/maps/t/objects", "Objects"))
    assert o2["apple"]["state"] == "struck" and o2["apple"]["id"] == o["apple"]["id"]
    assert {k: v["id"] for k, v in o2.items()} == {k: v["id"] for k, v in o.items()}
    assert client.get("/api/maps/t", "MapSummary")["objects"] == 4
    # and gone the version after
    fr, files = frames(client, "taken", moved(EYES[3:4]), "c")
    assert client.update("t", fr, files)["state"] == "committed"
    assert "apple" not in labels(client.get("/api/maps/t/objects", "Objects"))
    assert client.get("/api/maps/t", "MapSummary")["objects"] == 3


def test_detection_failure_fails_the_job(client: Client) -> None:
    client.backend.detect_error = True
    fr, files = frames(client, "full", EYES[:1], "a")
    job = client.update("t", fr, files, cameras={"cam": CAM})
    assert job["state"] == "failed"


def test_objects_of_a_map_of_the_older_mapper(tmp_path: Path) -> None:
    store = Store(tmp_path / "data")
    store.create()
    backend = FakeBackend(gauge=__import__("oh_my_slam.mapping.anchor", fromlist=["Sim3"]).Sim3())
    rec = {"score": 0.8, "label_votes": {"door": 1.6, "cabinet": 0.4}, "observations": 4,
           "created_update": 1, "last_seen_update": 7}
    backend.objects_world = [
        {"id": 3, "label": "door", "center": [1.0, 0.0, 1.0], "size": [0.9, 0.05, 2.0], **rec},
        {"id": None, "label": "ghost", "center": [0.0, 0.0, 0.5], "size": [0.1, 0.1, 0.1]},
        # no score recorded: not listed (never made up)
        {"id": 4, "label": "old", "center": [2.0, 0.0, 0.5], "size": [0.1, 0.1, 0.1]},
        # no points in the cloud: no observed colour, not listed
        {"id": 5, "label": "lamp", "center": [2.0, 1.0, 0.5], "size": [0.1, 0.1, 0.1], **rec},
        # map.md §5: a door is a leaf (thin, vertical, tall); an opening is never an object
        {"id": 6, "label": "door", "center": [3.0, 0.0, 1.0], "size": [0.9, 0.8, 2.0], **rec},
        {"id": 7, "label": "doorway", "center": [4.0, 0.0, 1.0], "size": [0.9, 0.05, 2.0],
         **rec}]
    from oh_my_slam.service.backend import Cloud

    xyz = np.array([[0.9, 0.0, 1.0], [1.1, 0.0, 1.0], [1.0, 0.0, 0.5], [0.0, 0.0, 0.0],
                    [3.0, 0.0, 1.0], [4.0, 0.0, 1.0]], np.float32)
    backend.cloud_world = Cloud(xyz, np.array([[200, 40, 40]] * 3 + [[0, 0, 0]] * 3, np.uint8),
                                np.array([3, 3, 3, 0, 6, 7], np.uint32))
    timings = Timings()
    updater = Updater(store, backend, timings, prior_first=False)
    jobs = Jobs(store.jobs, updater, timings.predict_update)
    jobs.start()
    derived = Derived(store, backend.read)
    service = Service(store, jobs, FakeInference(), backend.require_inference, timings=timings,
                      version="test", derived=derived)
    from tests.fakes.service import textured

    with TestClient(create_app(service)) as http:
        c = Client(http, backend, service)
        imgs = {f"i{k}": backend.add(textured(k), pose(10.0 * k, 0.1 * k, 0.0)) for k in range(3)}
        job = c.update("m", [{"name": n, "image": n, "camera_id": "c"} for n in imgs], imgs,
                       cameras={"c": CAMERA})
        assert job["state"] == "committed", job
        doc = c.get("/api/maps/m/objects", "Objects")
        assert [o["label"] for o in doc["objects"]] == ["door"]
        o = doc["objects"][0]
        assert doc["surfaces"] == [] and o["state"] == "confirmed"
        assert o["score"] == 0.8 and o["detected_as"] == ["door", "cabinet"]
        assert o["mean_rgb"] == [200, 40, 40] and o["sigma_centre_m"] > 0
        # the mapper's update ids are the versions, at most the version read
        assert o["first_seen_version"] == 1 and o["last_seen_version"] == 1
        assert o["extent_seen"] == {"x": "partial", "y": "partial", "z": "partial"}
        c.get("/api/maps/nothing/objects", status=404)
    jobs.stop(timeout=10)


def test_an_empty_anchored_map_lists_nothing(client: Client) -> None:
    fr, files = frames(client, "full", EYES[:1], "a")
    client.backend.detections.clear()
    job = client.update("e", fr, files, cameras={"cam": CAM})
    assert job["state"] == "committed"
    doc = client.get("/api/maps/e/objects", "Objects")
    assert doc["objects"] == [] and len(doc["surfaces"]) >= 1


def test_detections_on_a_coarser_grid_than_the_depth(client: Client) -> None:
    from oh_my_slam.core.types import Intrinsics

    K = Intrinsics(500.0, 500.0, 400.0, 300.0, 800, 600)
    scene = TableScene()
    frames_, files = [], {}
    for i, eye in enumerate(EYES[:2]):
        T = look_at(eye, TARGET)
        rgb, z, idx = scene.render(T, K)
        img = jpeg(rgb, 95)
        client.backend.detections[hashlib.sha256(img).hexdigest()] = detections(scene, idx)
        files[f"k{i}"], files[f"k{i}d"] = img, png_depth(z)
        frames_.append({"name": f"k{i}", "image": f"k{i}", "depth": f"k{i}d", "camera_id": "big",
                        "prior": {"trust": "metric", "T_map_cam": T.tolist(),
                                  "sigma_pos_m": 0.01, "sigma_rot_deg": 0.5}})
    cam = {"fx": 500.0, "fy": 500.0, "cx": 400.0, "cy": 300.0, "width": 800, "height": 600}
    job = client.update("b", frames_, files, cameras={"big": cam})
    assert job["state"] == "committed", job
    assert {g for _, g in client.backend.detected} == {(640, 480)}
    assert {"table", "box", "basket"} <= set(labels(client.get("/api/maps/b/objects", "Objects")))


def test_views_from_above_are_looked_at_closer(client: Client) -> None:
    """Keyframes looking down at the table top are looked at closer (the backend's
    ``look_closer`` gets their raised areas); the other labels it gives a detection count as
    evidence for the object's label."""
    from oh_my_slam.segmentation.api import Detection

    def closer(image: Path, regions: list[Any], dets: list[Detection]) -> list[Detection]:
        return [Detection(d.label, d.score, d.source, d.mask, d.box,
                          {"globe": 0.05} if d.label == "box" else {}) for d in dets]

    client.backend.closer = closer
    above = [(0.62, 0.42, TOP + 0.9), (0.58, 0.38, TOP + 0.95)]
    fr, files = frames(client, "full", above, "u")
    job = client.update("u", fr, files, cameras={"cam": CAM})
    assert job["state"] == "committed", job
    assert client.backend.looked and all(boxes for _, boxes in client.backend.looked)
    box = labels(client.get("/api/maps/u/objects", "Objects"))["box"]
    assert box["detected_as"] == ["box"]


def test_birdseye_of_each_version(client: Client) -> None:
    """GET birdseye.png draws the version from above; GET birdseye gives the same metadata; its
    pixel_to_map takes each object's footprint centre onto an outline pixel of its colour."""
    import io
    import json

    from PIL import Image

    fr, files = frames(client, "full", EYES[:4], "a")
    assert client.update("t", fr, files, cameras={"cam": CAM})["state"] == "committed"
    meta = client.get("/api/maps/t/birdseye", "Birdseye")
    assert meta["map"] == "t" and meta["map_version"] == 1
    summary = client.get("/api/maps/t", "MapSummary")  # the version's scale (map.md §8)
    assert (meta["scale_source"], meta["scale_sigma_rel"]) == (summary["scale_source"],
                                                               summary["scale_sigma_rel"])
    r = client.http.get("/api/maps/t/birdseye.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert "server-timing" in r.headers
    img = Image.open(io.BytesIO(r.content))
    assert img.size == (meta["width"], meta["height"])
    text = img.text  # type: ignore[attr-defined]
    assert json.loads(text["pixel_to_map"]) == meta["pixel_to_map"]
    assert text["map_version"] == "1" and text["map"] == "t"
    def drawn(meta: dict[str, Any], png: bytes, o: dict[str, Any]) -> bool:
        """Whether the object's colour is drawn around its centre's pixel."""
        A = np.asarray(meta["pixel_to_map"])
        u, v, _ = np.linalg.inv(np.vstack([A, [0, 0, 1]])) @ [*o["centre_m"][:2], 1.0]
        rgb = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
        r_px = max(o["size_m"][:2]) / A[0, 0] / 2 + 3
        ys, xs = np.mgrid[0:rgb.shape[0], 0:rgb.shape[1]]
        near = np.hypot(xs - u, ys - v) <= r_px
        return bool((rgb[near] == o["color"]).all(axis=1).any())

    objects = client.get("/api/maps/t/objects", "Objects")["objects"]
    assert all(drawn(meta, r.content, o) for o in objects)
    # kept per version: a second read loads it; a past version answers its own
    assert client.get("/api/maps/t/birdseye?version=1", "Birdseye") == meta
    fr2, files2 = frames(client, "taken", moved(EYES[1:3]), "b")
    assert client.update("t", fr2, files2)["state"] == "committed"
    meta2 = client.get("/api/maps/t/birdseye", "Birdseye")
    assert meta2["map_version"] == 2
    png2 = client.http.get("/api/maps/t/birdseye.png").content
    o2 = labels(client.get("/api/maps/t/objects", "Objects"))
    assert o2["apple"]["state"] == "struck"
    assert not drawn(meta2, png2, o2["apple"])  # seen gone: not drawn
    assert drawn(meta2, png2, o2["table"])
    assert client.get("/api/maps/t/birdseye?version=1", "Birdseye") == meta
    client.get("/api/maps/t/birdseye?version=9", status=400)
    client.get("/api/maps/none/birdseye.png", status=404)
