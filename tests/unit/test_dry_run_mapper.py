"""The dry run and the height gate of a map the older mapper builds (``Updater(prior_first=
False)``): the dry run predicts statuses without a registration (README "Dry run"), and a
registered frame whose camera height disagrees with the one sent gives no content (map.md §2)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient

from oh_my_slam.service.app import Service, create_app
from oh_my_slam.service.jobs import Jobs
from oh_my_slam.service.store import Store
from oh_my_slam.service.timings import Timings
from oh_my_slam.service.update import Updater
from tests.fakes.service import CAMERA, FakeBackend, FakeInference, plain, png16, pose, textured
from tests.synth.room import jpeg as jpeg_of
from tests.unit.test_service_api import Client
from tests.unit.test_service_prior_first import dry


@pytest.fixture
def client(tmp_path: Path) -> Iterator[Client]:
    """The older mapper's path for every map, with dry runs."""
    store = Store(tmp_path / "data")
    store.create()
    backend = FakeBackend()
    timings = Timings()
    updater = Updater(store, backend, timings, prior_first=False)
    jobs = Jobs(store.jobs, updater, timings.predict_update)
    jobs.start()
    service = Service(store, jobs, FakeInference(), backend.require_inference, timings=timings,
                      version="test", dry_run=updater.dry_run)
    with TestClient(create_app(service)) as http:
        yield Client(http, backend, service)
    jobs.stop(timeout=10)


def _metric(T: np.ndarray) -> dict[str, Any]:
    return {"trust": "metric", "T_map_cam": T.tolist(), "sigma_pos_m": 0.02,
            "sigma_rot_deg": 1.0}


def _suspicious() -> bytes:
    img = np.full((240, 320, 3), 128, np.uint8)
    cv2.rectangle(img, (40, 40), (280, 200), (30, 30, 30), 3)
    return jpeg_of(img)


def test_a_dry_run_on_the_older_mapper(client: Client) -> None:
    T0, T1 = pose(0, 0.0, 0.0), pose(30, 0.2, 0.0)
    files = {"a": client.backend.add(textured(1), T0), "b": textured(1), "p": plain(),
             "n": textured(9), "c": client.backend.add(textured(2), T1),
             "cd": png16(np.full((240, 320), 200)), "q": textured(1), "s": _suspicious()}
    frames = [
        {"name": "a", "image": "a", "camera_id": "cam", "prior": _metric(T0)},
        # no prior, linked to a, 0.2 m away: rejected too_close
        {"name": "q", "image": "q", "camera_id": "cam", "depth": "cd"},
        {"name": "b", "image": "b", "camera_id": "cam", "prior": _metric(T0)},  # duplicate
        {"name": "p", "image": "p", "camera_id": "cam", "prior": _metric(T1)},  # plain: kept
        {"name": "n", "image": "n", "camera_id": "cam"},  # no prior, no link: no_overlap
        {"name": "c", "image": "c", "camera_id": "cam", "prior": _metric(T1),
         "depth": "cd"},  # 0.2 m away: kept too_close
        {"name": "s", "image": "s", "camera_id": "cam"},  # suspicious, measured: too_close
    ]
    client.backend.mono = 0.2
    out = dry(client, "m", frames, files, cameras={"cam": CAMERA})
    rows = {r["name"]: (r["predicted_status"], r["predicted_reason"]) for r in out["frames"]}
    assert rows == {"a": ("registered", None), "b": ("rejected", "duplicate"),
                    "p": ("kept_at_prior", "plain"), "n": ("rejected", "no_overlap"),
                    "c": ("kept_at_prior", "too_close"), "q": ("rejected", "too_close"),
                    "s": ("rejected", "too_close")}
    by = {r["name"]: r for r in out["frames"]}
    assert by["b"]["duplicate_of"] == "a" and by["c"]["scene_distance_m"] == 0.2
    assert all(r["map_inliers"] is None for r in out["frames"])
    assert out["quality"]["height_inconsistent"] == []


def test_the_height_gate_on_the_older_mapper(client: Client) -> None:
    turn = [pose(a, 0.0, 0.0, 1.2) for a in (0, 30, 60, 90)]
    frames, files = [], {}
    for i, T in enumerate(turn):
        files[f"t{i}"] = client.backend.add(textured(700 + i), T)
        frames.append({"name": f"t{i}", "image": f"t{i}", "camera_id": "cam",
                       "prior": _metric(T), "height_above_floor_m": 1.2,
                       "height_sigma_m": 0.01})
    frames[2]["height_above_floor_m"] = 0.5
    job = client.update("m", frames, files, {"cam": CAMERA})
    assert [i["status"] for i in job["inputs"]] == ["registered"] * 4
    assert job["quality"]["height_inconsistent"] == ["t2"]
    rows = {r["name"]: r for r in client.service.store.version("m")["frames"]}
    assert rows["t2"]["height_inconsistent"] and "height_inconsistent" not in rows["t1"]
