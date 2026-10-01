"""The client sends each request's image at the size the server reads it (``client.images``): the
server's model adapters, run on what the client sends, must give exactly the response the original
image gives (their model calls replaced by deterministic functions of the pixels they read)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from oh_my_slam.client import protocol as p
from oh_my_slam.client.client import InferenceClient
from oh_my_slam.client.images import request_image, request_rgb
from oh_my_slam.core import rle
from oh_my_slam.core.images import load_rgb
from oh_my_slam.server.models.geometry_moge import MoGeGeometry
from oh_my_slam.server.models.gravity_geocalib import _INPUT_SIDE, GeoCalibGravity
from oh_my_slam.server.models.multiview_mapanything import _LOAD_SIDE
from oh_my_slam.server.models.seg_yoloe import YoloeSegmenter


def _photo(path: Path, w: int, h: int, orientation: int = 1, seed: int = 0) -> Path:
    """A JPEG with smooth structure and noise (LANCZOS and the JPEG decoder both matter)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack([x * 255 / w, y * 255 / h, (x + y) % 256], -1)
    rgb = np.clip(base + rng.normal(0, 20, (h, w, 3)), 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb)
    exif = img.getexif()
    if orientation != 1:
        exif[0x0112] = orientation
    img.save(path, quality=92, exif=exif)
    return path


@pytest.fixture
def photo(tmp_path: Path) -> Path:
    return _photo(tmp_path / "photo.jpg", 1600, 1200)


@pytest.mark.parametrize("orientation", [1, 6])
@pytest.mark.parametrize("side", [640, 768, 1024])
def test_server_reads_the_same_pixels(tmp_path: Path, side: int, orientation: int) -> None:
    src = _photo(tmp_path / "a.jpg", 1500, 1000, orientation)
    with request_image(src, side) as sent:
        assert sent.path != src and sent.path.suffix == ".bmp" and sent.downscaled
        np.testing.assert_array_equal(load_rgb(sent.path, side), load_rgb(src, side))
        w, h = Image.open(sent.path).size
        uw, uh = (1000, 1500) if orientation == 6 else (1500, 1000)
        assert sent.scale == (w / uw, h / uh) and max(w, h) == side
    assert not sent.path.exists()  # removed once the request is done
    np.testing.assert_array_equal(request_rgb(src, side), load_rgb(src, side))


def test_small_or_unreadable_images_are_sent_as_they_are(tmp_path: Path) -> None:
    small = _photo(tmp_path / "small.jpg", 640, 480)
    with request_image(small, 768) as sent:
        assert sent.path == small and not sent.downscaled
    missing = tmp_path / "missing.jpg"
    with request_image(missing, 768) as sent:
        assert sent.path == missing


class _Server:
    """The server's adapters, as ``InferenceClient._post`` (model calls stubbed on the pixels)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.seen: list[str] = []

        def infer(_: Any, rgb: np.ndarray, fov: float | None, tokens: int,
                  fp16: bool | None = None) -> dict[str, Any]:
            h, w = rgb.shape[:2]
            f = 0.8 if fov is None else 0.5 / np.tan(np.radians(fov) / 2)
            return {"depth": 1.0 + rgb.mean(-1) / 100.0, "mask": rgb[..., 0] > 20,
                    "intrinsics": np.array([[f, 0, 0.5], [0, f * w / h, 0.5], [0, 0, 1]])}

        def estimate(_: Any, rgb: np.ndarray, focal: float | None) -> dict[str, Any]:
            f = (focal or 300.0) * 1.03 + float(rgb.mean())
            return {"up_cam": [0.0, -1.0, 0.0], "roll_deg": 0.0, "pitch_deg": 0.0,
                    "roll_unc_deg": 1.0, "pitch_unc_deg": 1.0, "focal_px_small": f,
                    "focal_unc_px_small": f / 7, "vfov_deg": 50.0}

        def detect(_: Any, rgb: np.ndarray, labels: list[str], conf: float, iou: float,
                   imgsz: int) -> list[dict[str, Any]]:
            m = rgb[..., 1] > 128
            return [{"label": labels[0], "score": 0.9, "source": "yoloe",
                     "box_xyxy": [1.0, 2.0, float(rgb.shape[1]), float(rgb.shape[0])],
                     "mask_array": m}]

        monkeypatch.setattr(MoGeGeometry, "infer_array", infer)
        monkeypatch.setattr(GeoCalibGravity, "estimate_array", estimate)
        monkeypatch.setattr(YoloeSegmenter, "detect_array", detect)
        self.adapters = {p.ROUTE_GEOMETRY: MoGeGeometry(), p.ROUTE_GRAVITY: GeoCalibGravity(),
                         p.ROUTE_SEGMENT: YoloeSegmenter()}

    def post(self, route: str, req: Any, model: type) -> Any:
        self.seen.append(req.image_path)
        return model.model_validate(self.adapters[route].run(req))


def test_responses_are_those_of_the_original(photo: Path, tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    server = _Server(monkeypatch)
    client = InferenceClient(tmp_path / "none.sock")
    monkeypatch.setattr(client, "_post", server.post)
    for fov in (None, 63.0):
        req = p.GeometryRequest(image_path=str(photo), out_dir=str(tmp_path / "a"), max_side=768,
                                fov_x_deg=fov, want_descriptor=False)
        ref = server.post(p.ROUTE_GEOMETRY, req, p.GeometryResponse)
        ref_depth = np.load(ref.depth_path)
        got = client.geometry(req)
        assert server.seen[-1].endswith(".bmp")
        assert got == ref
        np.testing.assert_array_equal(np.load(got.depth_path), ref_depth)
    for focal in (None, 1234.5678):
        greq = p.GravityRequest(image_path=str(photo), focal_px=focal)
        assert client.gravity(greq) == server.post(p.ROUTE_GRAVITY, greq, p.GravityResponse)
    for side in (1024, 768):
        sreq = p.SegmentRequest(image_path=str(photo), labels=["chair"], max_side=side)
        got_s = client.segment_image(sreq)
        ref_s = server.post(p.ROUTE_SEGMENT, sreq, p.SegmentResponse)
        assert got_s == ref_s and rle.decode(got_s.instances[0].mask).shape == (
            768 * 3 // 4 if side == 768 else 768, side)


def test_multiview_intrinsics_reach_the_model_unchanged(photo: Path, tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """The multi-view adapter scales a view's intrinsics to the image it read (float32): what it
    computes from the downscaled file and the client's intrinsics is what it computed before."""

    def server_K(path: str, K: list[list[float]]) -> np.ndarray:  # the adapter's arithmetic
        rgb = load_rgb(Path(path), max_side=_LOAD_SIDE)
        w, h = Image.open(path).size
        Kn = np.asarray(K, dtype=np.float32).copy()
        Kn[0] *= rgb.shape[1] / w
        Kn[1] *= rgb.shape[0] / h
        return Kn

    K = [[1401.37, 0.0, 799.5], [0.0, 1399.91, 600.25], [0.0, 0.0, 1.0]]
    seen: list[tuple[list[str], list[np.ndarray | None]]] = []

    def post(route: str, req: Any, model: type) -> Any:  # while the files exist
        assert req.intrinsics is not None
        seen.append((req.image_paths, [None if k is None else server_K(i, k)
                                       for i, k in zip(req.image_paths, req.intrinsics, strict=True)]))
        return p.MultiviewResponse(views=[])

    client = InferenceClient(tmp_path / "none.sock")
    monkeypatch.setattr(client, "_post", post)
    client.multiview(p.MultiviewRequest(image_paths=[str(photo), str(photo)], out_dir=str(tmp_path),
                                        intrinsics=[K, None]))
    paths, Ks = seen[0]
    assert all(i.endswith(".bmp") for i in paths) and Ks[1] is None
    np.testing.assert_array_equal(Ks[0], server_K(str(photo), K))


def test_protocol_sides_are_the_server_sides() -> None:
    assert p.GRAVITY_SIDE == _INPUT_SIDE and p.MULTIVIEW_SIDE == _LOAD_SIDE
