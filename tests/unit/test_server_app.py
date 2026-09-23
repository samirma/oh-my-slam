from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from oh_my_slam.client import protocol as p
from oh_my_slam.core import rle
from oh_my_slam.server.app import ServerState, create_app
from oh_my_slam.server.gpu_worker import GpuWorker
from oh_my_slam.server.models import Registry, build_registry


@pytest.fixture
def image(tmp_path: Path) -> Path:
    path = tmp_path / "img.jpg"
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)).save(path)
    return path


def make_state(loaded: bool = True, max_queue: int = 8) -> ServerState:
    reg = build_registry(stub=True)
    worker = GpuWorker(max_queue=max_queue)
    worker.start()
    state = ServerState(registry=reg, worker=worker)
    if loaded:
        worker.call(reg.load_all, "cpu")
        state.loading = False
    return state


def test_health_and_loading_state(image: Path, tmp_path: Path) -> None:
    state = make_state(loaded=False)
    client = TestClient(create_app(state))
    h = client.get(p.ROUTE_HEALTH).json()
    assert h["status"] == "loading"
    req = p.GeometryRequest(image_path=str(image), out_dir=str(tmp_path / "o"))
    assert client.post(p.ROUTE_GEOMETRY, json=req.model_dump()).status_code == 503
    state.worker.call(state.registry.load_all, "cpu")
    state.loading = False
    h = p.Health.model_validate(client.get(p.ROUTE_HEALTH).json())
    assert h.status == "degraded"  # SAM 3 stub refuses to load, like a missing HF access
    assert h.models["segment_sam3"].error
    assert h.queue_limit == 8
    state.stopping = True
    assert client.get(p.ROUTE_HEALTH).json()["status"] == "stopping"
    state.worker.stop()


def test_endpoints_with_stub_models(image: Path, tmp_path: Path) -> None:
    state = make_state()
    client = TestClient(create_app(state))
    out = tmp_path / "o"
    out.mkdir()
    g = client.post(
        p.ROUTE_GEOMETRY,
        json=p.GeometryRequest(image_path=str(image), out_dir=str(out), max_side=200).model_dump(),
    )
    assert g.status_code == 200, g.text
    geo = p.GeometryResponse.model_validate(g.json())
    assert (geo.width, geo.height, geo.orig_width, geo.orig_height) == (200, 150, 400, 300)
    assert np.load(geo.depth_path).shape == (150, 200)
    assert geo.timings.compute_s >= 0

    gr = client.post(p.ROUTE_GRAVITY, json=p.GravityRequest(image_path=str(image)).model_dump())
    assert p.GravityResponse.model_validate(gr.json()).up_cam == [0.0, -1.0, 0.0]

    s = client.post(
        p.ROUTE_SEGMENT,
        json=p.SegmentRequest(image_path=str(image), labels=["chair", "table"],
                              max_side=200).model_dump(),
    )
    seg = p.SegmentResponse.model_validate(s.json())
    assert seg.mode == "degraded" and len(seg.instances) == 2
    m = rle.decode(seg.instances[0].mask)
    assert m.shape == (150, 200) and m.any()

    mv = client.post(
        p.ROUTE_MULTIVIEW,
        json=p.MultiviewRequest(image_paths=[str(image)] * 3, out_dir=str(out)).model_dump(),
    )
    assert len(p.MultiviewResponse.model_validate(mv.json()).views) == 3
    state.worker.stop()


def test_error_mapping(tmp_path: Path) -> None:
    state = make_state()
    client = TestClient(create_app(state))
    r = client.post(
        p.ROUTE_GEOMETRY,
        json=p.GeometryRequest(image_path=str(tmp_path / "missing.jpg"),
                               out_dir=str(tmp_path)).model_dump(),
    )
    assert r.status_code == 400 and r.json()["error"] == "InputError"
    assert client.post(p.ROUTE_GEOMETRY, json={"bad": 1}).status_code == 422
    # a model that failed to load answers 500 with the load error
    state.registry.state["gravity"].loaded = False
    state.registry.state["gravity"].error = "boom"
    r = client.post(p.ROUTE_GRAVITY, json={"image_path": "x"})
    assert r.status_code == 500 and r.json()["detail"] == "boom"
    state.worker.stop()


def test_queue_full_returns_503(image: Path, tmp_path: Path) -> None:
    state = make_state(max_queue=1)
    gate = threading.Event()
    state.worker.submit(gate.wait, 5)  # occupy the worker
    deadline = time.monotonic() + 5
    while state.worker.depth != 1 or state.worker._queue.qsize() and time.monotonic() < deadline:
        time.sleep(0.01)
    state.worker.submit(lambda: None)  # fill the queue
    client = TestClient(create_app(state))
    r = client.post(p.ROUTE_GRAVITY, json=p.GravityRequest(image_path=str(image)).model_dump())
    assert r.status_code == 503 and r.json()["error"] == "busy"
    gate.set()
    state.worker.stop()


def test_registry_status_rules() -> None:
    reg = Registry()

    class A:
        key, name, required = "a", "A", True

        def load(self, device: str) -> None:
            raise RuntimeError("no weights")

        def warmup(self) -> None:
            pass

    reg.add(A())
    reg.load_all("cpu")
    assert reg.status() == "error" and reg.get("a") is None
