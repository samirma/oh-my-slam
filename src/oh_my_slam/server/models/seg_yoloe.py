"""YOLOE-26x-seg with text prompts: open-vocabulary instance proposals with raw scores.

Weights (``yoloe-26x-seg.pt``) and the MobileCLIP2-B text encoder (``mobileclip2_b.ts``) are
downloaded by Ultralytics into the weights directory, which is the server's working directory.
Text embeddings are cached per vocabulary.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from oh_my_slam.client.protocol import SegmentRequest
from oh_my_slam.core import rle
from oh_my_slam.core.images import load_rgb
from oh_my_slam.core.paths import weights_dir

WEIGHTS = "yoloe-26x-seg.pt"


class YoloeSegmenter:
    key = "segment_yoloe"
    name = "YOLOE-26x-seg"
    required = True

    def __init__(self) -> None:
        self.model: Any = None
        self.device = "cpu"
        self._pe_cache: dict[tuple[str, ...], Any] = {}
        self._active: tuple[str, ...] | None = None

    def load(self, device: str) -> None:
        wd = weights_dir()
        (wd / "ultralytics").mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(wd / "ultralytics"))
        os.environ.setdefault("YOLO_VERBOSE", "False")
        from ultralytics import YOLOE

        self.device = device
        self.model = YOLOE(str(wd / WEIGHTS))
        self.model.to(device)

    def warmup(self) -> None:
        self._set_vocabulary(("chair", "table"))
        img = np.zeros((320, 320, 3), np.uint8)
        self.model.predict(img, imgsz=320, conf=0.5, device=self.device, verbose=False)

    def _set_vocabulary(self, names: tuple[str, ...]) -> None:
        if self._active == names:
            return
        pe = self._pe_cache.get(names)
        if pe is None:
            pe = self.model.get_text_pe(list(names))
            if len(self._pe_cache) > 16:
                self._pe_cache.clear()
            self._pe_cache[names] = pe
        self.model.set_classes(list(names), pe)
        self._active = names

    def detect_array(self, rgb: np.ndarray, labels: list[str], conf: float, iou: float,
                     imgsz: int) -> list[dict[str, Any]]:
        names = tuple(labels)
        self._set_vocabulary(names)
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        results = self.model.predict(
            bgr,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=self.device,
            retina_masks=True,
            verbose=False,
            max_det=300,
            agnostic_nms=True,
        )
        r = results[0]
        out: list[dict[str, Any]] = []
        if r.masks is None or r.boxes is None or len(r.boxes) == 0:
            return out
        masks = r.masks.data.detach().cpu().numpy() > 0.5
        boxes = r.boxes.xyxy.detach().cpu().numpy()
        scores = r.boxes.conf.detach().cpu().numpy()
        classes = r.boxes.cls.detach().cpu().numpy().astype(int)
        h, w = rgb.shape[:2]
        for m, b, s, c in zip(masks, boxes, scores, classes, strict=True):
            if m.shape != (h, w):
                import cv2

                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            if not m.any():
                continue
            out.append(
                {
                    "label": names[c],
                    "score": float(s),
                    "source": "yoloe",
                    "box_xyxy": [float(v) for v in b],
                    "mask_array": m,
                }
            )
        return out

    def run(self, req: SegmentRequest) -> dict[str, Any]:
        rgb = load_rgb(Path(req.image_path), max_side=req.max_side)
        h, w = rgb.shape[:2]
        dets = self.detect_array(rgb, req.labels, req.conf, req.iou, req.imgsz)
        instances = []
        for d in dets:
            m = d.pop("mask_array")
            d["mask"] = rle.encode(m)
            instances.append(d)
        return {"width": w, "height": h, "instances": instances, "timings": {}}
