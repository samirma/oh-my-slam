"""SAM 3 mask refinement and scoring (transformers ``Sam3Model``), optional.

The checkpoint ``facebook/sam3`` is gated on Hugging Face. When access is missing, loading fails
and the server runs ``degraded`` (YOLOE alone), which is the documented fallback (gate G3 / U1).
For the ≤ ``max_concepts`` best YOLOE concepts, the image features are computed once and each
concept is decoded with its text prompt; SAM 3 instances replace the YOLOE proposals of that
concept (SAM 3 scores are used as the raw score, source ``sam3``).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

REPO_ID = "facebook/sam3"
_THRESHOLD = 0.3


class Sam3Refiner:
    key = "segment_sam3"
    name = "SAM 3"
    required = False

    def __init__(self) -> None:
        self.model: Any = None
        self.processor: Any = None
        self.device = "cpu"

    def load(self, device: str) -> None:
        import torch
        from transformers import Sam3Model, Sam3Processor

        self.device = device
        self.processor = Sam3Processor.from_pretrained(REPO_ID)
        self.model = Sam3Model.from_pretrained(REPO_ID).to(torch.device(device)).eval()

    def warmup(self) -> None:
        img = np.zeros((256, 256, 3), np.uint8)
        self.refine(img, [], 1)

    def refine(self, rgb: np.ndarray, dets: list[dict[str, Any]], max_concepts: int) -> list[dict]:
        import torch
        from PIL import Image

        if not dets:
            return dets
        by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for d in dets:
            by_label[d["label"]].append(d)
        ranked = sorted(by_label, key=lambda k: -max(d["score"] for d in by_label[k]))
        concepts = ranked[:max_concepts]
        h, w = rgb.shape[:2]
        image = Image.fromarray(rgb)
        with torch.inference_mode():
            img_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            vision = self.model.get_vision_features(pixel_values=img_inputs.pixel_values)
            out: list[dict[str, Any]] = [d for k in ranked[max_concepts:] for d in by_label[k]]
            for concept in concepts:
                text = self.processor(text=concept, return_tensors="pt").to(self.device)
                outputs = self.model(vision_embeds=vision, input_ids=text.input_ids)
                res = self.processor.post_process_instance_segmentation(
                    outputs, threshold=_THRESHOLD, mask_threshold=0.5, target_sizes=[(h, w)]
                )[0]
                masks = res["masks"].detach().cpu().numpy().astype(bool)
                if len(masks) == 0:
                    out.extend(by_label[concept])
                    continue
                for m, s, b in zip(masks, res["scores"].cpu().numpy(), res["boxes"].cpu().numpy(),
                                   strict=True):
                    if m.any():
                        out.append({"label": concept, "score": float(s), "source": "sam3",
                                    "box_xyxy": [float(v) for v in b], "mask_array": m})
        return out
