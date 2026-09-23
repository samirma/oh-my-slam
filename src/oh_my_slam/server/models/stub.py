"""Deterministic stand-in models (``OH_MY_SLAM_SERVER_STUB=1``) for lifecycle and protocol tests.

They exercise the real server, worker thread, file hand-off and RLE paths without weights.
``OH_MY_SLAM_STUB_DELAY`` (seconds) slows every request down for queueing tests;
``OH_MY_SLAM_STUB_FAIL_SAM3=1`` (default) makes the optional refiner fail so the server is
``degraded`` just like without SAM 3 access.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from oh_my_slam.client.protocol import (
    GeometryRequest,
    GravityRequest,
    MultiviewRequest,
    SegmentRequest,
)
from oh_my_slam.core import rle
from oh_my_slam.core.atomic import atomic_save_npy
from oh_my_slam.core.images import load_rgb, upright_size


def _delay() -> None:
    d = float(os.environ.get("OH_MY_SLAM_STUB_DELAY", "0") or 0)
    if d > 0:
        time.sleep(d)


class _Base:
    required = True
    precision = "fp32"

    def load(self, device: str) -> None:
        self.device = device

    def warmup(self) -> None:
        pass


class StubGeometry(_Base):
    key = "geometry"
    name = "stub geometry"

    def run(self, req: GeometryRequest) -> dict[str, Any]:
        _delay()
        rgb = load_rgb(Path(req.image_path), max_side=req.max_side)
        ow, oh = upright_size(Path(req.image_path))
        h, w = rgb.shape[:2]
        fov = req.fov_x_deg or 60.0
        fx = w / (2 * math.tan(math.radians(fov) / 2))
        v = np.arange(h, dtype=np.float32)[:, None].repeat(w, 1)
        depth = (2.0 + 0.001 * v).astype(np.float32)
        out = Path(req.out_dir)
        atomic_save_npy(out / "depth.npy", depth)
        atomic_save_npy(out / "mask.npy", np.ones((h, w), np.uint8))
        normals = np.zeros((h, w, 3), np.float16)
        normals[..., 2] = -1
        atomic_save_npy(out / "normals.npy", normals)
        desc = np.resize(rgb.reshape(-1, 3).mean(0), 16).astype(np.float64) + 1e-3
        desc /= np.linalg.norm(desc)
        return {
            "width": w, "height": h, "orig_width": ow, "orig_height": oh,
            "intrinsics": {"fx": fx, "fy": fx, "cx": w / 2, "cy": h / 2},
            "fov_x_deg": fov,
            "depth_path": str(out / "depth.npy"), "mask_path": str(out / "mask.npy"),
            "normals_path": str(out / "normals.npy") if req.want_normals else None,
            "descriptor": desc.tolist() if req.want_descriptor else None,
            "timings": {},
        }


class StubGravity(_Base):
    key = "gravity"
    name = "stub gravity"

    def run(self, req: GravityRequest) -> dict[str, Any]:
        _delay()
        ow, oh = upright_size(Path(req.image_path))
        f = req.focal_px or ow / (2 * math.tan(math.radians(30)))
        return {
            "up_cam": [0.0, -1.0, 0.0], "roll_deg": 0.0, "pitch_deg": 0.0,
            "roll_unc_deg": 0.5, "pitch_unc_deg": 0.5, "focal_px": f, "focal_unc_px": 1.0,
            "vfov_deg": math.degrees(2 * math.atan(oh / (2 * f))), "timings": {},
        }


class StubSegment(_Base):
    key = "segment_yoloe"
    name = "stub segmenter"

    def run(self, req: SegmentRequest, refiner: Any = None) -> dict[str, Any]:
        _delay()
        rgb = load_rgb(Path(req.image_path), max_side=req.max_side)
        h, w = rgb.shape[:2]
        instances = []
        for i, label in enumerate(req.labels[:2]):
            m = np.zeros((h, w), bool)
            x0 = w // 8 + i * (w // 2)
            m[h // 4 : 3 * h // 4, x0 : x0 + w // 4] = True
            instances.append({
                "label": label, "score": 0.9 - 0.2 * i, "source": "yoloe",
                "box_xyxy": [float(x0), h / 4, float(x0 + w // 4), 3 * h / 4],
                "mask": rle.encode(m),
            })
        return {"width": w, "height": h, "mode": "degraded", "instances": instances,
                "timings": {}}


class StubSam3(_Base):
    key = "segment_sam3"
    name = "stub SAM 3"
    required = False

    def load(self, device: str) -> None:
        if os.environ.get("OH_MY_SLAM_STUB_FAIL_SAM3", "1") == "1":
            raise PermissionError("stub: gated model access denied")

    def refine(self, rgb: np.ndarray, dets: list, max_concepts: int) -> list:
        return dets


class StubMultiview(_Base):
    key = "multiview"
    name = "stub multiview"

    def run(self, req: MultiviewRequest) -> dict[str, Any]:
        _delay()
        views = []
        for i, _p in enumerate(req.image_paths):
            T = np.eye(4)
            T[0, 3] = 0.1 * i
            if req.poses is not None and req.poses[i] is not None:
                T = np.asarray(req.poses[i], dtype=np.float64)
            views.append({"pose": T.tolist(), "intrinsics": [[500, 0, 259], [0, 500, 194],
                                                             [0, 0, 1]],
                          "width": 518, "height": 388})
        return {"views": views, "metric_scale": 1.0, "timings": {}}


def stub_adapters() -> list[Any]:
    return [StubGeometry(), StubGravity(), StubSegment(), StubSam3(), StubMultiview()]
