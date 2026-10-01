"""Real-model integration tests (``-m models``): the inference server must be running
(``./start_inference_server.sh``) and ``OH_MY_SLAM_TEST_REAL_SERVER=1`` set so the tests talk
to the real runtime directory."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from oh_my_slam.client import protocol as p
from oh_my_slam.client.client import InferenceClient
from oh_my_slam.core import rle

pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(os.environ.get("OH_MY_SLAM_TEST_REAL_SERVER") != "1",
                       reason="set OH_MY_SLAM_TEST_REAL_SERVER=1 with a running server"),
]

SAMPLE = Path(__file__).resolve().parents[2] / "examples" / "restaurant.jpg"  # spec §5


@pytest.fixture(scope="module")
def client() -> InferenceClient:
    c = InferenceClient()
    h = c.require_ready()
    assert h.status == "ready"
    return c


@pytest.fixture(scope="module")
def sample() -> Path:
    if not SAMPLE.exists():
        pytest.skip(f"sample image missing: {SAMPLE}")
    return SAMPLE


def test_smoke_health_reports_models(client: InferenceClient) -> None:
    h = client.health()
    assert h.models["geometry"].loaded and h.models["segment_yoloe"].loaded
    assert h.device in ("mps", "cpu")


def test_smoke_geometry(client: InferenceClient, sample: Path, tmp_path: Path) -> None:
    t = time.perf_counter()
    g = client.geometry(p.GeometryRequest(image_path=str(sample), out_dir=str(tmp_path)))
    assert time.perf_counter() - t < 30
    d = np.load(g.depth_path)
    assert d.shape == (g.height, g.width) and max(g.width, g.height) <= 1024
    assert np.isfinite(d).all() and (d > 0).mean() > 0.5
    assert g.descriptor is not None and abs(np.linalg.norm(g.descriptor) - 1) < 1e-3


def test_geometry_kept_forward_is_resolved_as_the_network_gives(
        client: InferenceClient, sample: Path, tmp_path: Path) -> None:
    """The mapper's focal re-run: a kept forward pass re-solved with another focal length gives
    what running the network again gives, bit for bit, without running it."""

    def geometry(name: str, **kw: object) -> tuple[p.GeometryResponse, np.ndarray, np.ndarray]:
        g = client.geometry(p.GeometryRequest(
            image_path=str(sample), out_dir=str(tmp_path / name), max_side=768, num_tokens=1400,
            want_descriptor=False, **kw))  # type: ignore[arg-type]
        return g, np.load(g.depth_path), np.load(g.mask_path)

    network = geometry("network", fov_x_deg=55.0)  # nothing kept: the network runs
    first = geometry("first", keep_forward=True)
    resolved = geometry("resolved", fov_x_deg=55.0)
    assert first[0].fov_x_deg != pytest.approx(55.0, abs=0.5)  # the focal changes the result
    assert resolved[0].intrinsics == network[0].intrinsics
    np.testing.assert_array_equal(resolved[1], network[1])
    np.testing.assert_array_equal(resolved[2], network[2])
    assert resolved[0].timings.compute_s < 0.5 * network[0].timings.compute_s


def test_smoke_gravity(client: InferenceClient, sample: Path) -> None:
    gr = client.gravity(p.GravityRequest(image_path=str(sample)))
    assert abs(np.linalg.norm(gr.up_cam) - 1) < 1e-3
    assert gr.up_cam[1] < -0.5  # roughly level photo: up points to -y (OpenCV)


def test_smoke_segment(client: InferenceClient, sample: Path) -> None:
    s = client.segment_image(p.SegmentRequest(image_path=str(sample), labels=["chair", "person"]))
    assert s.instances
    for inst in s.instances[:5]:
        m = rle.decode(inst.mask)
        assert m.shape == (s.height, s.width) and m.any()
        assert inst.label in ("chair", "person")


def test_multiview(client: InferenceClient, tmp_path: Path) -> None:
    frames = sorted(Path("/Users/U124317/robot_view/ainex-captures").glob("00[1-4]_*.jpg"))
    if len(frames) < 3:
        pytest.skip("ainex frames missing")
    mv = client.multiview(p.MultiviewRequest(image_paths=[str(f) for f in frames],
                                             out_dir=str(tmp_path)))
    assert len(mv.views) == len(frames)
    T0 = np.asarray(mv.views[0].pose)
    assert np.allclose(T0[3], [0, 0, 0, 1])
