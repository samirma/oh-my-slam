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
    fr, files = frames(client, "taken", EYES[:3], "b")
    assert client.update("t", fr, files)["state"] == "committed"
    o2 = labels(client.get("/api/maps/t/objects", "Objects"))
    assert o2["apple"]["state"] == "struck" and o2["apple"]["id"] == o["apple"]["id"]
    assert {k: v["id"] for k, v in o2.items()} == {k: v["id"] for k, v in o.items()}
    assert client.get("/api/maps/t", "MapSummary")["objects"] == 4
    # and gone the version after
    fr, files = frames(client, "taken", EYES[3:4], "c")
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
    backend = FakeBackend(gauge=__import__("oh_my_slam.service.anchor", fromlist=["Sim3"]).Sim3())
    backend.objects_world = [
        {"id": 3, "label": "door", "center": [1.0, 0.0, 1.0], "size": [0.9, 0.05, 2.0]},
        {"id": None, "label": "ghost", "center": [0.0, 0.0, 0.5], "size": [0.1, 0.1, 0.1]}]
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
        assert doc["surfaces"] == [] and doc["objects"][0]["state"] == "confirmed"
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
