"""The inference server's model adapters (spec §2.1) around their models, offline (spec §4): torch
and the model libraries are the NumPy doubles of ``tests.fakes.fake_torch``, so what is checked is
the adapters' own work — how each model is loaded and warmed up, what it is given (pixels, units,
priors, devices) and how its output becomes the protocol response (units, masks, files)."""

from __future__ import annotations

import math
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from oh_my_slam.client import protocol as p
from oh_my_slam.core import rle
from oh_my_slam.core.images import load_rgb
from oh_my_slam.server import models
from oh_my_slam.server.models import geometry_moge as gm
from oh_my_slam.server.models import gravity_geocalib as gg
from oh_my_slam.server.models import multiview_mapanything as mv
from oh_my_slam.server.models import seg_yoloe as sy
from tests.fakes.fake_torch import Device, Tensor, installed, libraries, make_torch


def _image(path: Path, w: int, h: int, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(path)
    return path


@pytest.fixture
def torch() -> Iterator[types.ModuleType]:
    fake = make_torch()
    with installed({"torch": fake}):
        yield fake


# ------------------------------------------------------------------------------------------------
# registry and device


@pytest.mark.parametrize(("available", "device"), [(True, "mps"), (False, "cpu")])
def test_mps_is_preferred_when_available(available: bool, device: str) -> None:
    with installed({"torch": make_torch(mps_available=available)}):
        assert models.select_device() == device


def test_the_registry_holds_one_adapter_per_server_route() -> None:
    reg = models.build_registry()
    assert list(reg.adapters) == ["geometry", "gravity", "segment_yoloe", "multiview"]
    assert [type(a) for a in reg.adapters.values()] == [
        gm.MoGeGeometry, gg.GeoCalibGravity, sy.YoloeSegmenter, mv.MapAnythingMultiview]
    # nothing is loaded until the GPU thread loads it: the server is not ready yet
    assert reg.status() == "error" and all(reg.get(k) is None for k in reg.adapters)


def test_loading_logs_each_model_and_every_failure() -> None:
    class Adapter:
        def __init__(self, key: str, fail: bool) -> None:
            self.key, self.name, self.fail = key, key.upper(), fail
            self.precision = None if fail else "fp16"

        def load(self, device: str) -> None:
            if self.fail:
                raise OSError("no weights")

        def warmup(self) -> None:
            pass

    reg = models.Registry()
    reg.add(Adapter("a", fail=True))
    reg.add(Adapter("b", fail=False))
    lines: list[str] = []
    reg.load_all("mps", log=lines.append)
    assert lines == ["loading A on mps", "failed to load A: OSError: no weights", "loading B on mps"]
    assert reg.state["a"].error == "OSError: no weights" and reg.get("b") is not None
    assert reg.status() == "error" and reg.precision == "fp16" and reg.device == "mps"


# ------------------------------------------------------------------------------------------------
# GeoCalib


class _GeoCalib:
    last: _GeoCalib

    def __init__(self, weights: str) -> None:
        self.weights, self.device, self.evaluated = weights, None, False
        self.calls: list[tuple[Tensor, Any]] = []
        self.result: dict[str, Any] = {}
        _GeoCalib.last = self

    def to(self, device: Device) -> _GeoCalib:
        self.device = device
        return self

    def eval(self) -> _GeoCalib:
        self.evaluated = True
        return self

    def calibrate(self, img: Tensor, priors: Any = None) -> dict[str, Any]:
        self.calls.append((img, priors))
        return self.result


def _calibration(f: list[float], vec: list[float], device: str = "mps", unc: bool = True
                 ) -> dict[str, Any]:
    grav = types.SimpleNamespace(vec3d=Tensor([vec], device), roll=Tensor([0.1], device),
                                 pitch=Tensor([-0.2], device))
    res: dict[str, Any] = {"gravity": grav,
                           "camera": types.SimpleNamespace(f=Tensor([f], device))}
    if unc:
        res |= {"roll_uncertainty": Tensor([0.01], device),
                "pitch_uncertainty": Tensor([0.02], device),
                "focal_uncertainty": Tensor([5.0], device)}
    return res


@pytest.fixture
def geocalib(torch: types.ModuleType) -> Iterator[gg.GeoCalibGravity]:
    with installed(libraries({"geocalib": {"GeoCalib": _GeoCalib}})):
        g = gg.GeoCalibGravity()
        g.load("mps")
        yield g


def test_geocalib_loads_the_pinhole_weights_on_the_device_and_warms_up(
        geocalib: gg.GeoCalibGravity) -> None:
    model = _GeoCalib.last
    assert geocalib.model is model and geocalib.device == "mps"
    assert model.weights == "pinhole" and model.device == "mps" and model.evaluated
    model.result = _calibration([300.0, 300.0], [0.0, -1.0, 0.0])
    geocalib.warmup()
    (img, priors), = model.calls
    assert img.shape == (3, 240, 320) and img.device == "mps" and priors is None


def test_geocalib_reads_the_image_at_640_and_answers_in_full_resolution_pixels(
        geocalib: gg.GeoCalibGravity, tmp_path: Path) -> None:
    path = _image(tmp_path / "g.png", 1280, 960)
    model = _GeoCalib.last
    model.result = _calibration([480.0, 500.0], [0.0, -2.0, 0.0])
    res = p.GravityResponse.model_validate(geocalib.run(p.GravityRequest(image_path=str(path),
                                                                         focal_px=1000.0)))
    (img, priors), = model.calls
    # the model reads the 640 px image as a float CHW tensor in [0, 1] on the device ...
    rgb = load_rgb(path, max_side=640)
    assert img.shape == (3, 480, 640) and img.device == "mps" and img.dtype == np.float32
    np.testing.assert_allclose(img.a, rgb.transpose(2, 0, 1) / 255.0, rtol=1e-6)
    # ... with the focal prior in its pixels (half the original's)
    assert float(priors["focal"]) == 500.0 and priors["focal"].device == "mps"
    # the answer: unit up vector, degrees, and focal lengths in pixels of the original image
    assert res.up_cam == [0.0, -1.0, 0.0]
    assert res.roll_deg == pytest.approx(math.degrees(0.1))
    assert res.pitch_deg == pytest.approx(math.degrees(-0.2))
    assert res.roll_unc_deg == pytest.approx(math.degrees(0.01))
    assert res.pitch_unc_deg == pytest.approx(math.degrees(0.02))
    assert res.focal_px == pytest.approx(1000.0) and res.focal_unc_px == pytest.approx(10.0)
    assert res.vfov_deg == pytest.approx(math.degrees(2 * math.atan(480 / (2 * 500.0))))


def test_geocalib_without_prior_single_focal_and_missing_uncertainties(
        geocalib: gg.GeoCalibGravity, tmp_path: Path) -> None:
    path = _image(tmp_path / "s.png", 320, 240)  # small: read as it is
    model = _GeoCalib.last
    model.result = _calibration([250.0], [0.0, 0.0, 3.0], unc=False)
    res = geocalib.run(p.GravityRequest(image_path=str(path)))
    assert model.calls[0][1] is None  # no prior
    assert res["focal_px"] == pytest.approx(250.0) and res["up_cam"] == [0.0, 0.0, 1.0]
    assert res["roll_unc_deg"] == res["pitch_unc_deg"] == res["focal_unc_px"] == 0.0
    assert res["vfov_deg"] == pytest.approx(math.degrees(2 * math.atan(240 / 500.0)))
    model.result = _calibration([0.0], [0.0, -1.0, 0.0])
    assert geocalib.run(p.GravityRequest(image_path=str(path)))["vfov_deg"] == 0.0  # no focal


# ------------------------------------------------------------------------------------------------
# MapAnything


class _MapAnything:
    last: _MapAnything

    def __init__(self) -> None:
        self.repo, self.device, self.evaluated = "", None, False
        self.calls: list[tuple[Any, dict[str, Any], bool]] = []
        self.preds: list[dict[str, Tensor]] = []

    @classmethod
    def from_pretrained(cls, repo: str) -> _MapAnything:
        cls.last = cls()
        cls.last.repo = repo
        return cls.last

    def to(self, device: Device) -> _MapAnything:
        self.device = device
        return self

    def eval(self) -> _MapAnything:
        self.evaluated = True
        return self

    def infer(self, processed: Any, **kwargs: Any) -> list[dict[str, Tensor]]:
        self.calls.append((processed, kwargs, _MapAnything.torch.state["inference_mode"]))
        return self.preds

    torch: types.ModuleType


def _preprocess(views: list[dict[str, Any]], resize_mode: str, size: int) -> dict[str, Any]:
    return {"views": views, "resize_mode": resize_mode, "size": size}


def _prediction(i: int, scale: bool = True) -> dict[str, Tensor]:
    h, w = 4, 6
    pose = np.eye(4)
    pose[0, 3] = float(i)
    depth = np.full((1, h, w, 1), 2.0 + i, np.float32)
    depth[0, 0, 0, 0] = np.nan  # not finite
    mask = np.ones((1, h, w, 1), bool)
    mask[0, 1, 1, 0] = False  # masked out
    pred = {"camera_poses": Tensor(pose[None], "mps"),
            "intrinsics": Tensor(np.array([[[90.0, 0, 3], [0, 90.0, 2], [0, 0, 1]]]), "mps"),
            "depth_z": Tensor(depth, "mps"), "mask": Tensor(mask, "mps"),
            "conf": Tensor(np.full((1, h, w), 1.5 + i, np.float32), "mps")}
    if scale:
        pred["metric_scaling_factor"] = Tensor([[1.25]], "mps")
    return pred


@pytest.fixture
def mapanything(torch: types.ModuleType) -> Iterator[mv.MapAnythingMultiview]:
    _MapAnything.torch = torch
    with installed(libraries({"mapanything.models": {"MapAnything": _MapAnything},
                              "mapanything.utils.image": {"preprocess_inputs": _preprocess}})):
        yield mv.MapAnythingMultiview()


def test_mapanything_loads_the_apache_checkpoint_on_the_device(
        mapanything: mv.MapAnythingMultiview) -> None:
    mapanything.load("mps")
    model = _MapAnything.last
    assert mapanything.model is model and mapanything.device == "mps"
    assert model.repo == "facebook/map-anything-apache" == mv.REPO_ID
    assert model.device == "mps" and model.evaluated
    mapanything.warmup()  # nothing to run without a real view
    assert model.calls == []


def test_mapanything_views_carry_the_known_intrinsics_and_poses(
        mapanything: mv.MapAnythingMultiview, torch: types.ModuleType, tmp_path: Path) -> None:
    a = _image(tmp_path / "a.png", 2048, 1536, seed=1)
    b = _image(tmp_path / "b.png", 640, 480, seed=2)
    mapanything.load("mps")
    model = _MapAnything.last
    model.preds = [_prediction(0), _prediction(1)]
    K = [[1500.0, 0.0, 1024.0], [0.0, 1490.0, 768.0], [0.0, 0.0, 1.0]]
    T = np.eye(4)
    T[:3, 3] = [0.5, -1.0, 2.0]
    req = p.MultiviewRequest(image_paths=[str(a), str(b)], out_dir=str(tmp_path / "out"),
                             intrinsics=[K, None], poses=[T.tolist(), None], poses_metric=False,
                             resolution=518)
    res = p.MultiviewResponse.model_validate(mapanything.run(req))
    (processed, kwargs, in_inference_mode), = model.calls
    assert processed["resize_mode"] == "longest_side" and processed["size"] == 518
    va, vb = processed["views"]
    # each view: the image read at a long side of 1024 (never enlarged) ...
    np.testing.assert_array_equal(va["img"].a, load_rgb(a, max_side=1024))
    np.testing.assert_array_equal(vb["img"].a, load_rgb(b, max_side=1024))
    # ... with its intrinsics scaled to the image read (float32) and its anchor pose
    Kn = np.asarray(K, np.float32)
    Kn[:2] *= 0.5
    assert va["intrinsics"].dtype == np.float32
    np.testing.assert_array_equal(va["intrinsics"].a, Kn)
    np.testing.assert_array_equal(va["camera_poses"].a, T.astype(np.float32))
    assert va["is_metric_scale"].a.tolist() == [False]
    assert set(vb) == {"img"}  # nothing known about the second view
    assert in_inference_mode and kwargs == {
        "memory_efficient_inference": False, "use_amp": True, "amp_dtype": "fp16",
        "apply_mask": True, "mask_edges": True, "apply_confidence_mask": False}
    # the answer: poses, intrinsics on the processed grid, depth zeroed where invalid, files
    assert res.metric_scale == 1.25 and len(res.views) == 2
    for i, view in enumerate(res.views):
        assert (view.width, view.height) == (6, 4)
        assert view.pose[0][3] == float(i) and view.intrinsics[0][0] == 90.0
        depth = np.load(view.depth_path or "")
        assert depth.dtype == np.float32 and view.depth_path.endswith(f"mv_depth_{i:04d}.npy")
        assert depth[0, 0] == 0.0 and depth[1, 1] == 0.0 and np.all(depth[2:] == 2.0 + i)
        conf = np.load(view.conf_path or "")
        assert conf.dtype == np.float16 and np.all(conf == 1.5 + i)
    assert torch.calls == [("empty_cache",)]  # the MPS cache is released after each request


def test_mapanything_on_the_cpu_without_depth_or_scale(mapanything: mv.MapAnythingMultiview,
                                                       torch: types.ModuleType,
                                                       tmp_path: Path) -> None:
    img = _image(tmp_path / "c.png", 320, 240)
    mapanything.load("cpu")
    model = _MapAnything.last
    model.preds = [_prediction(i, scale=False) for i in range(9)]
    req = p.MultiviewRequest(image_paths=[str(img)] * 9, out_dir=str(tmp_path / "o"),
                             want_depth=False)
    res = mapanything.run(req)
    _, kwargs, _ = model.calls[0]
    assert kwargs["memory_efficient_inference"] and not kwargs["use_amp"]  # 9 views, CPU
    assert res["metric_scale"] == 1.0 and len(res["views"]) == 9
    assert all("depth_path" not in v and "conf_path" not in v for v in res["views"])
    assert not list((tmp_path / "o").iterdir()) and torch.calls == []


# ------------------------------------------------------------------------------------------------
# YOLOE


class _YOLOE:
    last: _YOLOE

    def __init__(self, weights: str) -> None:
        self.weights, self.device = weights, None
        self.events: list[tuple[Any, ...]] = []
        self.result: Any = None
        _YOLOE.last = self

    def to(self, device: str) -> None:
        self.device = device

    def get_text_pe(self, names: list[str]) -> tuple[str, ...]:
        self.events.append(("text_pe", tuple(names)))
        return ("pe", *names)

    def set_classes(self, names: list[str], pe: Any) -> None:
        self.events.append(("set_classes", tuple(names), pe))

    def predict(self, img: np.ndarray, **kwargs: Any) -> list[Any]:
        self.events.append(("predict", img, kwargs))
        return [self.result]


def _result(masks: np.ndarray | None, boxes: list[list[float]], scores: list[float],
            classes: list[int]) -> Any:
    class Boxes:
        xyxy = Tensor(np.asarray(boxes, np.float32).reshape(-1, 4), "mps")
        conf = Tensor(np.asarray(scores, np.float32), "mps")
        cls = Tensor(np.asarray(classes, np.float32), "mps")

        def __len__(self) -> int:
            return len(scores)

    m = None if masks is None else types.SimpleNamespace(data=Tensor(masks, "mps"))
    return types.SimpleNamespace(masks=m, boxes=Boxes())


@pytest.fixture
def yoloe(torch: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
          ) -> Iterator[sy.YoloeSegmenter]:
    monkeypatch.setattr(sy, "weights_dir", lambda: tmp_path / "weights")
    for var in ("YOLO_CONFIG_DIR", "YOLO_VERBOSE"):  # restored (unset) after the test
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    with installed(libraries({"ultralytics": {"YOLOE": _YOLOE}})):
        s = sy.YoloeSegmenter()
        s.load("mps")
        yield s


def test_yoloe_loads_its_weights_from_the_weights_folder(yoloe: sy.YoloeSegmenter,
                                                         tmp_path: Path) -> None:
    import os

    model = _YOLOE.last
    wd = tmp_path / "weights"
    assert yoloe.model is model and model.weights == str(wd / "yoloe-26x-seg.pt")
    assert model.device == yoloe.device == "mps"
    # Ultralytics keeps its settings in the weights folder and stays quiet
    assert (wd / "ultralytics").is_dir() and os.environ["YOLO_CONFIG_DIR"] == str(wd / "ultralytics")
    assert os.environ["YOLO_VERBOSE"] == "False"


def test_yoloe_warmup_sets_a_vocabulary_and_runs_once(yoloe: sy.YoloeSegmenter) -> None:
    model = _YOLOE.last
    model.result = _result(None, [], [], [])
    yoloe.warmup()
    assert model.events[:2] == [("text_pe", ("chair", "table")),
                                ("set_classes", ("chair", "table"), ("pe", "chair", "table"))]
    _, img, kwargs = model.events[2]
    assert img.shape == (320, 320, 3) and img.dtype == np.uint8 and not img.any()
    assert kwargs == {"imgsz": 320, "conf": 0.5, "device": "mps", "verbose": False}


def test_yoloe_text_embeddings_are_cached_per_vocabulary(yoloe: sy.YoloeSegmenter) -> None:
    model = _YOLOE.last
    model.result = _result(None, [], [], [])
    rgb = np.zeros((8, 8, 3), np.uint8)

    def vocabulary_events(labels: list[str]) -> list[tuple[Any, ...]]:
        n = len(model.events)
        assert yoloe.detect_array(rgb, labels, 0.1, 0.5, 64) == []
        return [e for e in model.events[n:] if e[0] != "predict"]

    assert vocabulary_events(["cup"]) == [("text_pe", ("cup",)),
                                          ("set_classes", ("cup",), ("pe", "cup"))]
    assert vocabulary_events(["cup"]) == []  # the active vocabulary: nothing to do
    assert vocabulary_events(["bed"])[0] == ("text_pe", ("bed",))
    # back to a known vocabulary: its embeddings are reused, not computed again
    assert vocabulary_events(["cup"]) == [("set_classes", ("cup",), ("pe", "cup"))]
    for k in range(20):  # the cache is bounded: it starts afresh beyond 16 vocabularies
        vocabulary_events([f"thing{k}"])
    assert len(yoloe._pe_cache) <= 17 and ("cup",) not in yoloe._pe_cache


def test_yoloe_detections_on_the_image_grid(yoloe: sy.YoloeSegmenter) -> None:
    model = _YOLOE.last
    rgb = np.zeros((40, 60, 3), np.uint8)
    rgb[..., 0], rgb[..., 2] = 200, 10  # red image: the model is given BGR
    small = np.zeros((3, 20, 30), np.float32)
    small[0, 2:6, 3:9] = 0.9  # an instance at half the image's resolution
    small[1, :, :] = 0.3  # below the mask threshold everywhere: dropped
    small[2, 10:12, 20:30] = 0.7
    model.result = _result(small, [[1, 2, 3, 4], [0, 0, 1, 1], [5, 6, 7, 8]], [0.8, 0.6, 0.4],
                           [1, 0, 0])
    dets = yoloe.detect_array(rgb, ["cup", "bowl"], 0.05, 0.6, 1024)
    _, img, kwargs = model.events[-1]
    assert img.flags.c_contiguous and np.all(img[..., 0] == 10) and np.all(img[..., 2] == 200)
    assert kwargs == {"imgsz": 1024, "conf": 0.05, "iou": 0.6, "device": "mps",
                      "retina_masks": True, "verbose": False, "max_det": 300, "agnostic_nms": True}
    assert [(d["label"], d["score"], d["source"]) for d in dets] == [
        ("bowl", pytest.approx(0.8), "yoloe"), ("cup", pytest.approx(0.4), "yoloe")]
    assert dets[0]["box_xyxy"] == [1.0, 2.0, 3.0, 4.0]
    for d, m in zip(dets, (small[0], small[2]), strict=True):  # nearest-neighbour upscaling
        np.testing.assert_array_equal(d["mask_array"], np.kron(m > 0.5, np.ones((2, 2), bool)))
    # masks already on the image grid are taken as they are
    full = np.zeros((1, 40, 60), np.float32)
    full[0, 5:9, 7:13] = 1.0
    model.result = _result(full, [[7, 5, 13, 9]], [0.9], [0])
    (det,) = yoloe.detect_array(rgb, ["cup", "bowl"], 0.05, 0.6, 1024)
    np.testing.assert_array_equal(det["mask_array"], full[0] > 0.5)


def test_yoloe_nothing_detected(yoloe: sy.YoloeSegmenter) -> None:
    model = _YOLOE.last
    rgb = np.zeros((10, 10, 3), np.uint8)
    for result in (_result(None, [[0, 0, 1, 1]], [0.9], [0]),  # no masks
                   _result(np.zeros((0, 10, 10)), [], [], [])):  # no boxes
        model.result = result
        assert yoloe.detect_array(rgb, ["cup"], 0.1, 0.5, 64) == []


def test_yoloe_response_masks_are_rle_on_the_request_grid(yoloe: sy.YoloeSegmenter,
                                                          tmp_path: Path) -> None:
    path = _image(tmp_path / "y.png", 1600, 1200)
    model = _YOLOE.last
    masks = np.zeros((1, 600, 800), np.float32)
    masks[0, 100:300, 200:500] = 1.0
    model.result = _result(masks, [[200, 100, 500, 300]], [0.7], [0])
    res = p.SegmentResponse.model_validate(yoloe.run(p.SegmentRequest(
        image_path=str(path), labels=["sofa"], max_side=800, imgsz=640)))
    assert (res.width, res.height) == (800, 600)
    _, img, kwargs = model.events[-1]
    assert img.shape == (600, 800, 3) and kwargs["imgsz"] == 640
    (inst,) = res.instances
    assert inst.label == "sofa" and inst.box_xyxy == [200.0, 100.0, 500.0, 300.0]
    np.testing.assert_array_equal(rle.decode(inst.mask), masks[0] > 0.5)


# ------------------------------------------------------------------------------------------------
# MoGe-2


class _Encoder:
    def __init__(self, with_cls: bool) -> None:
        self.with_cls = with_cls

    def forward(self, image: Tensor, num_tokens: int) -> Any:
        if not self.with_cls:
            return image
        cls = image.a.mean(axis=(2, 3)) + np.arange(1, 4, dtype=np.float32)  # (1, 3)
        return image, Tensor(cls, image.device)


class _MoGe:
    """MoGe-2's calling structure: ``infer`` runs ``self.forward`` (encoder, points head through
    ``_remap_points``, mask and scale heads), then a post-processing that depends on ``fov_x``."""

    built: _MoGe

    def __init__(self, normal_head: bool = True, sdpa: bool = True, cls: bool = True) -> None:
        self.encoder = _Encoder(cls)
        self.device = "cpu"
        self.forwards = 0
        self.sdpa = False
        if normal_head:
            self.normal_head = object()
        if sdpa:
            self.enable_pytorch_native_sdpa = lambda: setattr(self, "sdpa", True)

    @classmethod
    def from_pretrained(cls, repo: str) -> _MoGe:
        assert repo == gm.REPO_ID
        return cls.built

    def to(self, device: Device) -> _MoGe:
        self.device = str(device)
        return self

    def eval(self) -> _MoGe:
        return self

    def _remap_points(self, points: Tensor) -> Tensor:
        return points * 2.0

    def forward(self, image: Tensor, num_tokens: int) -> dict[str, Tensor]:
        self.forwards += 1
        self.encoder.forward(image, num_tokens)
        img = image.a[0]  # (3, H, W)
        head = np.stack([img[0], img[1], 1.0 + img[2]], axis=-1)[None]  # NHWC
        return {"points": self._remap_points(Tensor(head, image.device)),
                "mask": Tensor((img[0] > 0.3).astype(np.float32)[None], image.device),
                "metric_scale": Tensor(np.array([1.5], np.float32), image.device)}

    def infer(self, image: Tensor, num_tokens: int, fov_x: float | None = None,
              use_fp16: bool = True) -> dict[str, Tensor]:
        out = self.forward(image[None], num_tokens=num_tokens)
        f = 0.8 if fov_x is None else 0.5 / math.tan(math.radians(fov_x) / 2)
        depth = out["points"].a[0, ..., 2] * out["metric_scale"].a[0] * f
        K = np.array([[f, 0, 0.5], [0, f, 0.5], [0, 0, 1]], np.float32)
        return {"depth": Tensor(depth, self.device),
                "mask": Tensor(out["mask"].a[0] > 0.5, self.device),
                "intrinsics": Tensor(K, self.device)}


def _moge(model: _MoGe, device: str = "cpu") -> gm.MoGeGeometry:
    _MoGe.built = model
    g = gm.MoGeGeometry()
    g.load(device)
    return g


@pytest.fixture
def moge_lib(torch: types.ModuleType) -> Iterator[None]:
    with installed(libraries({"moge.model.v2": {"MoGeModel": _MoGe}})):
        yield


def test_moge_drops_the_normal_head_and_uses_native_attention_when_present(
        moge_lib: None) -> None:
    g = _moge(_MoGe(), "mps")
    assert not hasattr(g.model, "normal_head") and g.model.sdpa and g.model.device == "mps"
    assert g.fp16 and g.precision == "fp16-autocast"  # gate G2: fp16 autocast on MPS
    g.warmup()
    assert g.model.forwards == 1 and g._cls is None  # the warm-up leaves no descriptor behind


def test_moge_builds_without_a_normal_head_or_attention_switch_load_as_they_are(
        moge_lib: None) -> None:
    model = _MoGe(normal_head=False, sdpa=False)
    g = _moge(model)
    assert g.model is model and not model.sdpa
    assert not g.fp16 and g.precision == "fp32"  # the CPU runs in fp32


def test_moge_descriptor_is_the_normalised_class_token(moge_lib: None,
                                                       rng: np.random.Generator) -> None:
    rgb = rng.integers(0, 256, (12, 16, 3), dtype=np.uint8)
    out = _moge(_MoGe()).infer_array(rgb, None, 100)
    cls = rgb.astype(np.float32).transpose(2, 0, 1).mean(axis=(1, 2)) / 255.0 + [1, 2, 3]
    np.testing.assert_allclose(out["descriptor"], cls / np.linalg.norm(cls), rtol=1e-5)
    assert out["mask"].dtype == np.float32 and out["depth"].shape == (12, 16)
    # an encoder that gives no class token: no descriptor
    assert "descriptor" not in _moge(_MoGe(cls=False)).infer_array(rgb, None, 100)


def test_moge_resolving_a_kept_output_without_a_descriptor(moge_lib: None,
                                                           rng: np.random.Generator) -> None:
    """A kept forward pass is re-solved for another focal without running the network again,
    and gives what running it gives — no descriptor when the network gave none."""
    rgb = rng.integers(0, 256, (12, 16, 3), dtype=np.uint8)
    reference = _moge(_MoGe(cls=False)).infer_array(rgb, 70.0, 100)
    g = _moge(_MoGe(cls=False))
    first = g.infer_array(rgb, None, 100, keep=True)
    assert len(g.kept) == 1 and g.model.forwards == 1 and "descriptor" not in first
    resolved = g.infer_array(rgb, 70.0, 100)
    assert g.model.forwards == 1 and len(g.kept) == 0  # served by the kept output
    assert resolved.keys() == reference.keys() == {"depth", "mask", "intrinsics"}
    for k in reference:
        np.testing.assert_array_equal(resolved[k], reference[k])


def test_moge_response_in_pixels_with_its_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch
                                                ) -> None:
    path = _image(tmp_path / "m.png", 400, 300)
    depth = np.full((300, 400), 2.0, np.float32)
    depth[0, 0], depth[0, 1], depth[0, 2] = np.inf, -1.0, 0.0
    K = np.array([[0.8, 0, 0.5], [0, 0.8 * 4 / 3, 0.5], [0, 0, 1]])

    def infer(self: Any, rgb: np.ndarray, fov: float | None, tokens: int,
              fp16: bool | None = None, keep: bool = False) -> dict[str, Any]:
        assert rgb.shape == (300, 400, 3) and tokens == 1800 and keep
        return {"depth": depth, "intrinsics": K, "descriptor": np.array([0.6, 0.8], np.float32)}

    monkeypatch.setattr(gm.MoGeGeometry, "infer_array", infer)
    res = p.GeometryResponse.model_validate(gm.MoGeGeometry().run(p.GeometryRequest(
        image_path=str(path), out_dir=str(tmp_path / "o"), num_tokens=1800, keep_forward=True)))
    assert (res.width, res.height, res.orig_width, res.orig_height) == (400, 300, 400, 300)
    assert res.intrinsics == p.PixelIntrinsics(fx=320.0, fy=320.0, cx=200.0, cy=150.0)
    assert res.fov_x_deg == pytest.approx(math.degrees(2 * math.atan(400 / 640)))
    assert res.descriptor == pytest.approx([0.6, 0.8])
    # without a validity mask from the model, valid = finite and positive depth
    saved, valid = np.load(res.depth_path), np.load(res.mask_path)
    assert valid[0, :3].tolist() == [0, 0, 0] and saved[0, :3].tolist() == [0.0, 0.0, 0.0]
    assert valid.sum() == 300 * 400 - 3 and np.all(saved[valid.astype(bool)] == 2.0)
